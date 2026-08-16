#!/usr/bin/env python3
"""Profile one speculative decode step: target verify + draft breakdown.

Per step GPU ms:
  1. draft parallel backbone
  2. target lm_head on draft hidden (additive only)
  3. markov serial head
  4. target verify forward
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from specforge.modeling.draft.flashmtp import (
    FlashMTPDraftModel,
    gather_pivot_multilayer_inference,
    sample,
)


@dataclass
class StepBreakdown:
    mode: str
    markov_rank: int
    batch_size: int
    draft_backbone_ms: float
    target_lm_head_ms: float
    serial_head_ms: float
    target_verify_ms: float
    draft_total_ms: float
    step_total_ms: float
    verify_fraction: float


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _elapsed_ms(start: torch.cuda.Event, end: torch.cuda.Event) -> float:
    return float(start.elapsed_time(end))


def _avg(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


@torch.inference_mode()
def profile_steps(
    *,
    draft_ckpt: str,
    target_path: str,
    input_ids: torch.Tensor,
    batch_size: int,
    num_warmup: int,
    num_timed: int,
) -> StepBreakdown:
    try:
        import flash_attn  # noqa: F401

        attn_impl = "flash_attention_2"
    except ImportError:
        attn_impl = "sdpa"

    device = torch.device("cuda")
    target = AutoModelForCausalLM.from_pretrained(
        target_path,
        attn_implementation=attn_impl,
        dtype=torch.bfloat16,
    ).to(device).eval()
    draft = FlashMTPDraftModel.from_pretrained(
        draft_ckpt,
        dtype=torch.bfloat16,
    ).to(device).eval()

    bsz = batch_size
    if input_ids.shape[0] == 1 and bsz > 1:
        input_ids = input_ids.expand(bsz, -1).contiguous()

    block_size = draft.block_size
    num_input_tokens = input_ids.shape[1]
    position_ids = (
        torch.arange(num_input_tokens + block_size + 1, device=device)
        .unsqueeze(0)
        .expand(bsz, -1)
    )
    past_key_values_target = DynamicCache()
    output = target(
        input_ids,
        position_ids=position_ids[:, :num_input_tokens],
        past_key_values=past_key_values_target,
        use_cache=True,
        logits_to_keep=1,
        output_hidden_states=True,
    )
    target_hidden = gather_pivot_multilayer_inference(
        output.hidden_states,
        draft.target_layer_ids,
        -1,
        draft.config.num_target_layers,
    )

    start = num_input_tokens
    block_output_ids = torch.full(
        (bsz, block_size),
        draft.mask_token_id,
        dtype=torch.long,
        device=device,
    )
    block_output_ids[:, 0] = sample(output.logits, 0.0).squeeze(-1)
    noise_embedding = draft.build_inference_query_embeddings(
        target.model.embed_tokens,
        block_output_ids,
    )
    target_hidden = draft.build_inference_current_chs(
        target.model.embed_tokens, target_hidden, input_ids[:, -1:]
    )
    if draft.local_position:
        draft_block_pos = (
            torch.arange(
                draft.draft_query_length, device=device, dtype=torch.long
            )
            .unsqueeze(0)
            .expand(bsz, -1)
        )
        ctx_pos_part = torch.zeros(
            bsz, draft.chs_len_per_block, dtype=torch.long, device=device
        )
    else:
        draft_block_pos = draft.build_draft_query_position_ids(
            torch.full((bsz, 1), start, dtype=torch.long, device=device)
        ).reshape(bsz, -1)
        ctx_pos_part = torch.full(
            (bsz, draft.chs_len_per_block), start - 1, dtype=torch.long, device=device
        )
    full_rotary = torch.cat([ctx_pos_part, draft_block_pos], dim=-1)
    first_prev_token_ids = block_output_ids[:, 0]
    lm_head = target.lm_head
    mode = draft.markov_output_mode

    ev = [torch.cuda.Event(enable_timing=True) for _ in range(8)]

    bb_ms: list[float] = []
    lm_ms: list[float] = []
    ser_ms: list[float] = []
    tv_ms: list[float] = []

    for step in range(num_warmup + num_timed):
        _sync(device)
        ev[0].record()
        block_hidden = draft(
            target_hidden=target_hidden,
            noise_embedding=noise_embedding,
            position_ids=draft_block_pos,
            rotary_position_ids=full_rotary,
            past_key_values=None,
            use_cache=False,
            is_causal=False,
        )
        draft_hidden = draft._prediction_hidden(block_hidden)
        ev[1].record()
        ev[1].synchronize()

        base_logits = None
        lm_t = 0.0
        if draft.markov_head is None or mode == "additive":
            _sync(device)
            ev[2].record()
            base_logits = lm_head(draft_hidden)
            ev[3].record()
            ev[3].synchronize()
            lm_t = _elapsed_ms(ev[2], ev[3])

        _sync(device)
        ev[4].record()
        if draft.markov_head is None:
            sampled = sample(base_logits, 0.0)
        else:
            sampled, _ = draft.markov_head.sample_block_tokens(
                hidden_states=draft_hidden,
                first_prev_token_ids=first_prev_token_ids,
                output_mode=mode,
                base_logits=base_logits,
                temperature=0.0,
            )
        ev[5].record()
        ev[5].synchronize()

        draft_tokens = torch.cat([block_output_ids[:, :1], sampled], dim=1)
        verify_position_ids = position_ids[:, start : start + draft_tokens.size(1)]

        _sync(device)
        ev[6].record()
        target(
            draft_tokens,
            position_ids=verify_position_ids,
            past_key_values=past_key_values_target,
            use_cache=True,
            output_hidden_states=True,
        )
        ev[7].record()
        ev[7].synchronize()

        if step >= num_warmup:
            bb_ms.append(_elapsed_ms(ev[0], ev[1]))
            if mode == "additive":
                lm_ms.append(lm_t)
            else:
                lm_ms.append(0.0)
            ser_ms.append(_elapsed_ms(ev[4], ev[5]))
            tv_ms.append(_elapsed_ms(ev[6], ev[7]))

    bb = _avg(bb_ms)
    lm = _avg(lm_ms)
    ser = _avg(ser_ms)
    tv = _avg(tv_ms)
    draft_total = bb + lm + ser
    step_total = draft_total + tv
    return StepBreakdown(
        mode=mode,
        markov_rank=int(draft.markov_rank),
        batch_size=bsz,
        draft_backbone_ms=bb,
        target_lm_head_ms=lm,
        serial_head_ms=ser,
        target_verify_ms=tv,
        draft_total_ms=draft_total,
        step_total_ms=step_total,
        verify_fraction=tv / step_total if step_total > 0 else 0.0,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target-model-path", default="/data/wanghanzhen/models/Qwen3-8B")
    parser.add_argument("--draft-ckpt", required=True)
    parser.add_argument("--batch-sizes", default="1,8,32")
    parser.add_argument("--prompt", default="Solve step by step: What is 17 * 23?")
    parser.add_argument("--warmup-steps", type=int, default=20)
    parser.add_argument("--timed-steps", type=int, default=100)
    parser.add_argument("--output-json", default=None)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.target_model_path, trust_remote_code=True)
    input_ids = tokenizer(args.prompt, return_tensors="pt").input_ids.to("cuda")

    results: list[StepBreakdown] = []
    for batch_size in [int(x) for x in args.batch_sizes.split(",")]:
        print(f"\n=== batch={batch_size} ===", flush=True)
        stats = profile_steps(
            draft_ckpt=args.draft_ckpt,
            target_path=args.target_model_path,
            input_ids=input_ids,
            batch_size=batch_size,
            num_warmup=args.warmup_steps,
            num_timed=args.timed_steps,
        )
        results.append(stats)
        print(f"mode={stats.mode} rank={stats.markov_rank}")
        print(f"  target_verify_ms:    {stats.target_verify_ms:.4f}")
        print(f"  draft_backbone_ms:   {stats.draft_backbone_ms:.4f}")
        print(f"  target_lm_head_ms:   {stats.target_lm_head_ms:.4f}")
        print(f"  serial_head_ms:      {stats.serial_head_ms:.4f}")
        print(f"  draft_total_ms:      {stats.draft_total_ms:.4f}")
        print(f"  step_total_ms:       {stats.step_total_ms:.4f}")
        print(f"  verify_fraction:     {stats.verify_fraction*100:.1f}%")

    out = [asdict(r) for r in results]
    print("\n=== JSON ===")
    print(json.dumps(out, indent=2))
    if args.output_json:
        Path(args.output_json).write_text(json.dumps(out, indent=2) + "\n")


if __name__ == "__main__":
    main()
