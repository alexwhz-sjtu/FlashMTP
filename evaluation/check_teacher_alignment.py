"""Numerically compare teacher training and inference backbone inputs at one anchor."""

from __future__ import annotations

import argparse

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from specforge.core.flashmtp import OnlineFlashMTPModel
from specforge.modeling.draft.flashmtp import (
    FlashMTPDraftModel,
    gather_pivot_multilayer_inference,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", required=True)
    parser.add_argument("--draft", required=True)
    parser.add_argument("--anchor", type=int, default=160)
    args = parser.parse_args()

    device = torch.device("cuda:0")
    target = AutoModelForCausalLM.from_pretrained(
        args.target,
        dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    ).to(device).eval()
    draft = FlashMTPDraftModel.from_pretrained(
        args.draft,
        dtype=torch.bfloat16,
        attn_implementation="flex_attention",
    ).to(device).eval()
    tokenizer = AutoTokenizer.from_pretrained(args.target)

    text = "Alignment check for teacher speculative decoding. " * 80
    input_ids = tokenizer.encode(text, return_tensors="pt").to(device)
    need = args.anchor + draft.block_size
    if input_ids.shape[1] < need:
        raise ValueError(f"Need at least {need} tokens, got {input_ids.shape[1]}")
    input_ids = input_ids[:, :need]
    loss_mask = torch.ones_like(input_ids, dtype=torch.float32)

    with torch.inference_mode():
        target_output = target(input_ids, output_hidden_states=True, use_cache=False)
        online = OnlineFlashMTPModel(
            draft_model=draft,
            target_lm_head=target.lm_head,
            target_embed_tokens=target.model.embed_tokens,
            mask_token_id=draft.mask_token_id,
            block_size=draft.block_size,
            num_anchors=1,
        )
        anchors = torch.tensor([[args.anchor]], device=device)
        keep = torch.ones_like(anchors, dtype=torch.bool)
        batch = online.prepare_batch(
            input_ids,
            target_output.hidden_states,
            loss_mask,
            anchor_positions=anchors,
            block_keep_mask=keep,
        )
        train_hidden = online.forward_backbone(batch, seq_len=input_ids.shape[1])

        # Production inference uses dense non-causal FlashAttention. The
        # training comparison above needs FlexAttention for its BlockMask.
        draft.config._attn_implementation = "flash_attention_2"

        a = args.anchor
        group_start = max(0, a - draft.anchor_group_size + 1)
        token_group_ids = input_ids[:, group_start : a + 1]
        draft_input_ids = torch.full(
            (1, draft.block_size),
            draft.mask_token_id,
            dtype=torch.long,
            device=device,
        )
        draft_input_ids[:, 0] = input_ids[:, a]
        noise = draft.build_inference_query_embeddings(
            target.model.embed_tokens,
            draft_input_ids,
            token_group_ids=token_group_ids,
        )
        recent = draft.initialize_inference_condition(
            target_output.hidden_states,
            pivot_index=a - 1,
        )
        history, context_pos, draft_pos = draft.build_inference_context(
            recent,
            gather_pivot_multilayer_inference(
                target_output.hidden_states,
                draft.target_layer_ids,
                a,
                draft.config.num_target_layers,
            ),
            a,
            token_group_length=token_group_ids.shape[1],
        )

        def inference_hidden(chs_index: int) -> torch.Tensor:
            chs = gather_pivot_multilayer_inference(
                target_output.hidden_states,
                draft.target_layer_ids,
                chs_index,
                draft.config.num_target_layers,
            )
            block = draft(
                target_hidden=chs,
                history_hidden=history,
                noise_embedding=noise,
                position_ids=draft_pos,
                rotary_position_ids=torch.cat([context_pos, draft_pos], dim=-1),
                is_causal=False,
            )
            return draft._prediction_hidden(block).unsqueeze(1)

        aligned_hidden = inference_hidden(a)
        lagged_hidden = inference_hidden(a - 1)

    def report(name: str, candidate: torch.Tensor) -> None:
        delta = (train_hidden.float() - candidate.float()).abs()
        print(
            f"{name}: max_abs={delta.max().item():.8f} "
            f"mean_abs={delta.mean().item():.8f} "
            f"cos={torch.nn.functional.cosine_similarity(train_hidden.float().flatten(), candidate.float().flatten(), dim=0).item():.8f}"
        )

    print(f"anchor={args.anchor} seq_len={input_ids.shape[1]}")
    report("inference_chs_at_a", aligned_hidden)
    report("inference_chs_at_a_minus_1", lagged_hidden)


if __name__ == "__main__":
    main()
