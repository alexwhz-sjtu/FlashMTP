"""Compare one v2swa draft block through training and inference construction."""

import sys
from pathlib import Path

import torch
from torch import nn

root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root))

from specforge.core.flashmtp import OnlineFlashMTPModel
from specforge.modeling.draft.flashmtp import FlashMTPDraftModel


checkpoint = root / "cache/models/Qwen3-8B/Flashmtp_v2.3_qwen3_8b_ep3"
torch.manual_seed(42)
draft = FlashMTPDraftModel.from_pretrained(checkpoint, dtype=torch.bfloat16).cuda()
embedding = nn.Embedding(draft.config.vocab_size, draft.config.hidden_size).cuda().to(torch.bfloat16)
wrapper = OnlineFlashMTPModel(
    draft_model=draft,
    target_lm_head=nn.Identity(),
    target_embed_tokens=embedding,
    mask_token_id=draft.mask_token_id,
    block_size=draft.block_size,
    attention_backend="flex_attention",
    num_anchors=1,
)
pivot = 10
input_ids = torch.randint(0, draft.config.vocab_size - 1, (1, 32), device="cuda")
anchors = torch.tensor([[pivot]], device="cuda")
keep = torch.ones(1, 1, dtype=torch.bool, device="cuda")
loss_mask = torch.ones_like(input_ids)
target_hidden = torch.randn(
    1, 1, draft.condition_slot_count, draft.config.hidden_size,
    device="cuda", dtype=torch.bfloat16,
)

noise_train = wrapper._create_noise_embed(input_ids, anchors, keep)
history, history_starts, history_lengths = wrapper._prepare_history_sources(input_ids, loss_mask)
recent = draft.initialize_inference_condition(token_embeddings=embedding(input_ids[:, :pivot]))
draft_ids = torch.full((1, draft.block_size), draft.mask_token_id, device="cuda", dtype=torch.long)
draft_ids[:, 0] = input_ids[:, pivot]
noise_infer = draft.build_inference_query_embeddings(
    embedding, draft_ids,
    pivot_token_ids=input_ids[:, pivot - 1:pivot],
    window_embeddings=draft.inference_window_embeddings(recent),
)
train_ctx, train_pos = draft.build_block_position_ids(
    anchor_positions=anchors,
    history_position_ids=anchors.new_empty(1, 1, 0),
    history_keep_mask=torch.empty(1, 1, 0, dtype=torch.bool, device="cuda"),
)
infer_ctx, infer_pos = draft.build_inference_context(recent, target_hidden, pivot)
print("input_embeddings_equal", bool(torch.equal(noise_train, noise_infer)))
print("context_positions_equal", bool(torch.equal(train_ctx, infer_ctx)))
print("draft_positions_equal", bool(torch.equal(train_pos, infer_pos)))

draft.eval()
with torch.inference_mode():
    draft.config._attn_implementation = "flex_attention"
    train_out = wrapper._forward_packed_context(
        input_ids=input_ids, anchor_positions=anchors, block_keep_mask=keep,
        target_hidden=target_hidden, history_hidden_states=history,
        history_start_positions=history_starts,
        history_source_lengths=history_lengths, noise_embedding=noise_train,
    ).float()
    draft.config._attn_implementation = "flash_attention_2"
    infer_out = draft(
        target_hidden=target_hidden,
        noise_embedding=noise_infer,
        position_ids=infer_pos,
        rotary_position_ids=torch.cat([infer_ctx, infer_pos], dim=-1),
        is_causal=False,
    ).float()
delta = (train_out - infer_out).abs()
print("train_infer_mean_abs", float(delta.mean()))
print("train_infer_max_abs", float(delta.max()))
print("train_infer_relative_l2", float(torch.linalg.vector_norm(delta) / torch.linalg.vector_norm(train_out)))
