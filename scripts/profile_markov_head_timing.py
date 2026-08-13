#!/usr/bin/env python3
"""Profile Markov-head timing for additive vs direct (rank=256).

Per decode step (after target prefill), measure GPU ms for:
  1. Draft parallel backbone forward
  2. Target lm_head on draft hidden states (additive only; 0 for direct)
  3. Markov serial head sampling (sample_block_tokens)
  4. Total head path = (2) + (3)
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
class HeadTimingStats:
    mode: str
    draft_ckpt: str
    block_size: int
    markov_rank: int
    batch_size: int
    input_len: int
    num_warmup_steps: int
    num_timed_steps: int
    draft_backbone_avg_ms: float
    target_lm_head_avg_ms: float
    markov_serial_head_avg_ms: float
    total_head_path_avg_ms: float


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _elapsed_ms(start: torch.cuda.Event, end: torch.cuda.Event) -> float:
    return float(start.elapsed_time(end))


def _avg(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


@torch.inference_mode()
def profile_one_mode(
    *,
    draft_ckpt: str,
    target_path: str,
    input_ids: torch.Tensor,
    batch_size: int,
    num_warmup_steps: int,
    num_timed_steps: int,
) -> HeadTimingStats:
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
    if draft.markov_head is None:
        raise ValueError(f"Draft checkpoint has no Markov head: {draft_ckpt}")

    bsz = batch_size
    if input_ids.shape[0] == 1 and bsz > 1:
        input_ids = input_ids.expand(bsz, -1).contiguous()

    block_size = draft.block_size
    num_input_tokens = input_ids.shape[1]
    position_ids = (
        torch.arange(num_input_tokens + block_size, device=device)
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
        include_embedding_chs=draft.include_embedding_chs,
    )

    start = num_input_tokens
    block_output_ids = torch.full(
        (bsz, block_size),
        draft.mask_token_id,
        dtype=torch.long,
        device=device,
    )
    block_output_ids[:, 0] = sample(output.logits, 0.0).squeeze(-1)
    target_block_pos = position_ids[:, start : start + block_size]
    if draft.local_position:
        draft_block_pos = (
            torch.arange(1, block_size + 1, device=device, dtype=torch.long)
            .unsqueeze(0)
            .expand(bsz, -1)
        )
        ctx_pos_part = torch.zeros(
            bsz, draft.chs_len_per_block, dtype=torch.long, device=device
        )
    else:
        draft_block_pos = target_block_pos
        ctx_pos_part = torch.full(
            (bsz, draft.chs_len_per_block), start - 1, dtype=torch.long, device=device
        )
    full_rotary = torch.cat([ctx_pos_part, draft_block_pos], dim=-1)
    noise_embedding = target.model.embed_tokens(block_output_ids)
    first_prev_token_ids = block_output_ids[:, 0]
    lm_head = target.lm_head
    mode = draft.markov_output_mode

    ev0 = torch.cuda.Event(enable_timing=True)
    ev1 = torch.cuda.Event(enable_timing=True)
    ev2 = torch.cuda.Event(enable_timing=True)
    ev3 = torch.cuda.Event(enable_timing=True)
    ev4 = torch.cuda.Event(enable_timing=True)
    ev5 = torch.cuda.Event(enable_timing=True)

    backbone_ms: list[float] = []
    lm_head_ms: list[float] = []
    serial_ms: list[float] = []

    total_iters = num_warmup_steps + num_timed_steps
    for step in range(total_iters):
        _sync(device)
        ev0.record()
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
        ev1.record()
        ev1.synchronize()
        bb_ms = _elapsed_ms(ev0, ev1)

        base_logits = None
        lm_ms = 0.0
        if mode == "additive":
            _sync(device)
            ev2.record()
            base_logits = lm_head(draft_hidden)
            ev3.record()
            ev3.synchronize()
            lm_ms = _elapsed_ms(ev2, ev3)

        _sync(device)
        ev4.record()
        draft.markov_head.sample_block_tokens(
            hidden_states=draft_hidden,
            first_prev_token_ids=first_prev_token_ids,
            output_mode=mode,
            base_logits=base_logits,
            temperature=0.0,
        )
        ev5.record()
        ev5.synchronize()
        ser_ms = _elapsed_ms(ev4, ev5)

        if step >= num_warmup_steps:
            backbone_ms.append(bb_ms)
            lm_head_ms.append(lm_ms)
            serial_ms.append(ser_ms)

    lm_avg = _avg(lm_head_ms)
    serial_avg = _avg(serial_ms)
    return HeadTimingStats(
        mode=mode,
        draft_ckpt=draft_ckpt,
        block_size=int(block_size),
        markov_rank=int(draft.markov_rank),
        batch_size=bsz,
        input_len=int(num_input_tokens),
        num_warmup_steps=num_warmup_steps,
        num_timed_steps=num_timed_steps,
        draft_backbone_avg_ms=_avg(backbone_ms),
        target_lm_head_avg_ms=lm_avg,
        markov_serial_head_avg_ms=serial_avg,
        total_head_path_avg_ms=lm_avg + serial_avg,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target-model-path", default="/data/wanghanzhen/models/Qwen3-8B")
    parser.add_argument("--additive-draft", required=True)
    parser.add_argument("--direct-draft", required=True)
    parser.add_argument("--prompt", default="Solve step by step: What is 17 * 23?")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--warmup-steps", type=int, default=30)
    parser.add_argument("--timed-steps", type=int, default=200)
    parser.add_argument("--output-json", default=None)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.target_model_path, trust_remote_code=True)
    input_ids = tokenizer(args.prompt, return_tensors="pt").input_ids.to("cuda")

    results: list[HeadTimingStats] = []
    for label, ckpt in [
        ("additive", args.additive_draft),
        ("direct", args.direct_draft),
    ]:
        print(f"\nProfiling {label}: {ckpt}", flush=True)
        stats = profile_one_mode(
            draft_ckpt=ckpt,
            target_path=args.target_model_path,
            input_ids=input_ids,
            batch_size=args.batch_size,
            num_warmup_steps=args.warmup_steps,
            num_timed_steps=args.timed_steps,
        )
        results.append(stats)
        print(f"  mode={stats.mode} rank={stats.markov_rank} block={stats.block_size}")
        print(f"  draft_backbone_avg_ms:      {stats.draft_backbone_avg_ms:.4f}")
        print(f"  target_lm_head_avg_ms:      {stats.target_lm_head_avg_ms:.4f}")
        print(f"  markov_serial_head_avg_ms:  {stats.markov_serial_head_avg_ms:.4f}")
        print(f"  total_head_path_avg_ms:     {stats.total_head_path_avg_ms:.4f}")

    summary = [asdict(r) for r in results]
    print("\n=== summary ===")
    print(json.dumps(summary, indent=2))
    if args.output_json:
        Path(args.output_json).write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    main()
