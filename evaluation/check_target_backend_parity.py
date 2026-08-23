"""Compare SGLang training hidden captures with HF inference hidden states."""

from __future__ import annotations

import argparse
import json
import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from specforge.core.flashmtp import OnlineFlashMTPModel
from specforge.data.preprocessing import preprocess_conversations
from specforge.data.template import TEMPLATE_REGISTRY
from specforge.distributed import get_tp_group, init_distributed
from specforge.modeling.draft.flashmtp import FlashMTPDraftModel
from specforge.modeling.target.flashmtp_target_model import (
    SGLangFlashMTPTargetModel,
)
from specforge.modeling.target.target_utils import (
    SGLangTPEmbeddingAdapter,
    SGLangTPLMHeadAdapter,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", required=True)
    parser.add_argument("--draft")
    parser.add_argument("--tp-size", type=int, default=1)
    parser.add_argument("--training-jsonl-stdin", action="store_true")
    parser.add_argument("--input-ids-pt")
    args = parser.parse_args()

    init_distributed(tp_size=args.tp_size)
    device = torch.device("cuda", torch.cuda.current_device())
    tokenizer = AutoTokenizer.from_pretrained(args.target)
    if args.input_ids_pt:
        ids = torch.load(args.input_ids_pt, map_location=device, weights_only=True)
        loss_mask = torch.ones_like(ids)
    elif args.training_jsonl_stdin:
        record = json.loads(sys.stdin.readline())
        processed = preprocess_conversations(
            tokenizer,
            [record["conversations"]],
            TEMPLATE_REGISTRY.get("qwen"),
            max_length=2048,
        )
        ids = processed["input_ids"][0].to(device)
        loss_mask = processed["loss_mask"][0].to(device)
    else:
        ids = tokenizer.encode(
            "SGLang and Hugging Face hidden state parity. " * 30,
            return_tensors="pt",
        ).to(device)[:, :256]
        loss_mask = torch.ones_like(ids)
    mask = torch.ones_like(ids)
    capture = [0, 1, 5, 8, 12, 16, 18, 19, 23, 27, 30, 34, 35]

    sglang_target = SGLangFlashMTPTargetModel.from_pretrained(
        args.target,
        torch_dtype=torch.bfloat16,
        mem_fraction_static=0.25,
        context_length=2048,
        attention_backend="fa3",
        enable_torch_compile=False,
        max_running_requests=ids.shape[0],
        max_total_tokens=2048,
    )
    sglang_target.set_capture_layers(capture)
    sg = sglang_target.generate_flashmtp_data(ids, mask, mask, return_logits=True)
    sg_hidden = {layer: value.detach().clone() for layer, value in sg.hidden_states.items()}

    hf = AutoModelForCausalLM.from_pretrained(
        args.target, dtype=torch.bfloat16, attn_implementation="flash_attention_2"
    ).to(device).eval()
    with torch.inference_mode():
        hf_out = hf(ids, output_hidden_states=True, use_cache=False)
    sg_logits = sg.logits.float()
    hf_logits = hf_out.logits.float()
    print(f"sg_logits={tuple(sg_logits.shape)} hf_logits={tuple(hf_logits.shape)}")
    for shift in (-1, 0, 1):
        if shift < 0:
            sg_slice, hf_slice = sg_logits[:, 1:, :], hf_logits[:, :-1, :]
        elif shift > 0:
            sg_slice, hf_slice = sg_logits[:, :-1, :], hf_logits[:, 1:, :]
        else:
            sg_slice, hf_slice = sg_logits, hf_logits
        agreement = sg_slice.argmax(dim=-1).eq(hf_slice.argmax(dim=-1)).float().mean()
        cosine = torch.nn.functional.cosine_similarity(
            sg_slice.flatten(), hf_slice.flatten(), dim=0
        )
        print(
            f"logits shift={shift:+d} argmax_agreement={agreement.item():.8f} "
            f"cos={cosine.item():.8f}"
        )

    mask_token_id = hf.config.vocab_size
    sg_model = sglang_target.model_runner.model
    sg_embed = SGLangTPEmbeddingAdapter(
        sg_model.get_input_embeddings(), get_tp_group(), mask_token_id
    )
    sg_head = SGLangTPLMHeadAdapter(sg_model.lm_head, get_tp_group())
    component_ids = torch.tensor(
        [[0, 1, 2, 100, hf.config.vocab_size - 1, mask_token_id]], device=device
    )
    hf_mask = hf.model.embed_tokens.weight.float().mean(dim=0).to(torch.bfloat16)
    hf_embed = hf.model.embed_tokens(component_ids.clamp_max(hf.config.vocab_size - 1))
    hf_embed[:, -1, :] = hf_mask
    sg_embed_out = sg_embed(component_ids)
    embed_delta = (sg_embed_out.float() - hf_embed.float()).abs()
    embed_cos = torch.nn.functional.cosine_similarity(
        sg_embed_out.float().flatten(), hf_embed.float().flatten(), dim=0
    )
    print(
        f"tp_components rank={torch.distributed.get_rank()} "
        f"embed_max={embed_delta.max().item():.8f} "
        f"embed_mean={embed_delta.mean().item():.8f} embed_cos={embed_cos.item():.8f}"
    )
    torch.manual_seed(1234 + torch.distributed.get_rank())
    test_hidden = torch.randn(1, 4, hf.config.hidden_size, device=device, dtype=torch.bfloat16)
    sg_head_out = sg_head(test_hidden).float()
    hf_head_out = hf.lm_head(test_hidden).float()
    head_delta = (sg_head_out - hf_head_out).abs()
    head_cos = torch.nn.functional.cosine_similarity(
        sg_head_out.flatten(), hf_head_out.flatten(), dim=0
    )
    print(
        f"tp_components rank={torch.distributed.get_rank()} "
        f"head_max={head_delta.max().item():.8f} "
        f"head_mean={head_delta.mean().item():.8f} head_cos={head_cos.item():.8f} "
        f"head_argmax={sg_head_out.argmax(-1).eq(hf_head_out.argmax(-1)).float().mean().item():.8f}"
    )
    print(f"tokens={ids.shape[1]} capture={capture}")
    for layer in capture:
        # HF hidden_states[0] is embedding; layer L output is at L+1.
        reference = hf_out.hidden_states[layer + 1]
        candidate = sg_hidden[layer]
        delta = (candidate.float() - reference.float()).abs()
        cosine = torch.nn.functional.cosine_similarity(
            candidate.float().flatten(), reference.float().flatten(), dim=0
        )
        print(
            f"layer={layer:02d} max_abs={delta.max().item():.8f} "
            f"mean_abs={delta.mean().item():.8f} cos={cosine.item():.8f}"
        )
        layer_matches = []
        for hf_layer in range(max(0, layer - 1), min(hf.config.num_hidden_layers, layer + 2)):
            hf_candidate = hf_out.hidden_states[hf_layer + 1]
            match_cos = torch.nn.functional.cosine_similarity(
                candidate.float().flatten(), hf_candidate.float().flatten(), dim=0
            )
            layer_matches.append(f"hf{hf_layer:02d}={match_cos.item():.8f}")
        token_matches = []
        for shift in (-1, 0, 1):
            if shift < 0:
                sg_slice, hf_slice = candidate[:, 1:, :], reference[:, :-1, :]
            elif shift > 0:
                sg_slice, hf_slice = candidate[:, :-1, :], reference[:, 1:, :]
            else:
                sg_slice, hf_slice = candidate, reference
            match_cos = torch.nn.functional.cosine_similarity(
                sg_slice.float().flatten(), hf_slice.float().flatten(), dim=0
            )
            token_matches.append(f"shift{shift:+d}={match_cos.item():.8f}")
        print("  layer_match " + " ".join(layer_matches))
        print("  token_match " + " ".join(token_matches))

    if args.draft:
        draft = FlashMTPDraftModel.from_pretrained(
            args.draft, dtype=torch.bfloat16, attn_implementation="flex_attention"
        ).to(device).eval()
        online = OnlineFlashMTPModel(
            draft_model=draft,
            target_lm_head=hf.lm_head,
            target_embed_tokens=hf.model.embed_tokens,
            mask_token_id=draft.mask_token_id,
            block_size=draft.block_size,
            num_anchors=128,
        )
        torch.manual_seed(42)
        anchors, keep = online.sample_anchor_positions(ids.shape[1], loss_mask)
        hf_hidden = {layer: hf_out.hidden_states[layer + 1] for layer in capture}

        def predict(hidden_states):
            batch = online.prepare_batch(
                ids, hidden_states, loss_mask,
                anchor_positions=anchors, block_keep_mask=keep,
            )
            hidden = online.forward_backbone(batch, seq_len=ids.shape[1])
            latent = draft.markov_head.forward_teacher_forcing(
                hidden_states=hidden,
                prev_token_ids=batch.prev_token_ids,
                output_mode=draft.markov_output_mode,
                initial_prev_token_ids=batch.initial_prev_token_ids,
            )
            pred = draft.markov_head.project_logits(latent).argmax(dim=-1)
            correct = pred.eq(batch.labels)
            prefix = correct.cumprod(dim=-1).sum(dim=-1).float() + 1
            return hidden, pred, prefix[keep]

        with torch.inference_mode():
            sg_draft_hidden, sg_pred, sg_prefix = predict(sg_hidden)
            hf_draft_hidden, hf_pred, hf_prefix = predict(hf_hidden)
        draft_delta = (sg_draft_hidden.float() - hf_draft_hidden.float()).abs()
        draft_cos = torch.nn.functional.cosine_similarity(
            sg_draft_hidden.float().flatten(), hf_draft_hidden.float().flatten(), dim=0
        )
        print(
            f"draft_hidden max_abs={draft_delta.max().item():.8f} "
            f"mean_abs={draft_delta.mean().item():.8f} cos={draft_cos.item():.8f}"
        )
        print(
            f"prediction_agreement={sg_pred.eq(hf_pred).float().mean().item():.8f} "
            f"sglang_prefix={sg_prefix.mean().item():.8f} "
            f"hf_prefix={hf_prefix.mean().item():.8f}"
        )


if __name__ == "__main__":
    main()
