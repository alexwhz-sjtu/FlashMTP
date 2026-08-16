#!/usr/bin/env python3
"""Profile compile_serial_head on/off: per-step breakdown + end-to-end decode."""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.benchmark import load_benchmark_dataset, select_max_samples
from specforge.modeling.draft.flashmtp import (
    FlashMTPDraftModel,
    gather_pivot_multilayer_inference,
    markov_output_uses_base_lm_head,
    sample,
)


@dataclass
class StepBreakdown:
    compile_serial_head: bool
    draft_backbone_ms: float
    target_lm_head_ms: float
    serial_head_ms: float
    target_verify_ms: float
    draft_total_ms: float
    step_total_ms: float
    serial_head_fraction: float
    verify_fraction: float


@dataclass
class E2EResult:
    compile_serial_head: bool
    sample_index: int
    input_len: int
    output_tokens: int
    decode_steps: int
    avg_accept_length: float
    decode_wall_s: float
    s_per_output_token: float


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _elapsed_ms(start: torch.cuda.Event, end: torch.cuda.Event) -> float:
    return float(start.elapsed_time(end))


def _avg(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def _theoretical_speedup(serial_frac: float, serial_speedup: float) -> float:
    if serial_speedup <= 0:
        return 1.0
    return 1.0 / ((1.0 - serial_frac) + serial_frac / serial_speedup)


@torch.inference_mode()
def profile_one_step(
    *,
    draft: FlashMTPDraftModel,
    target: AutoModelForCausalLM,
    input_ids: torch.Tensor,
    compile_serial_head: bool,
    num_warmup: int,
    num_timed: int,
) -> StepBreakdown:
    device = target.device
    bsz = 1
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
        if draft.markov_head is None or markov_output_uses_base_lm_head(mode):
            _sync(device)
            ev[2].record()
            base_logits = lm_head(draft_hidden)
            ev[3].record()
            ev[3].synchronize()
            lm_t = _elapsed_ms(ev[2], ev[3])

        _sync(device)
        ev[4].record()
        sampled, _ = draft.sample_draft_tokens(
            draft_hidden=draft_hidden,
            lm_head=lm_head,
            first_prev_token_ids=first_prev_token_ids,
            temperature=0.0,
            compile_serial_head=compile_serial_head,
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
            lm_ms.append(lm_t)
            ser_ms.append(_elapsed_ms(ev[4], ev[5]))
            tv_ms.append(_elapsed_ms(ev[6], ev[7]))

    bb = _avg(bb_ms)
    lm = _avg(lm_ms)
    ser = _avg(ser_ms)
    tv = _avg(tv_ms)
    draft_total = bb + lm + ser
    step_total = draft_total + tv
    return StepBreakdown(
        compile_serial_head=compile_serial_head,
        draft_backbone_ms=bb,
        target_lm_head_ms=lm,
        serial_head_ms=ser,
        target_verify_ms=tv,
        draft_total_ms=draft_total,
        step_total_ms=step_total,
        serial_head_fraction=ser / step_total if step_total > 0 else 0.0,
        verify_fraction=tv / step_total if step_total > 0 else 0.0,
    )


@torch.inference_mode()
def profile_e2e(
    *,
    draft: FlashMTPDraftModel,
    target: AutoModelForCausalLM,
    input_ids: torch.Tensor,
    compile_serial_head: bool,
    max_new_tokens: int,
    verify_block_size: int,
    sample_index: int,
    num_warmup: int,
) -> E2EResult:
    device = target.device
    stop_token_ids = []
    if hasattr(target.config, "eos_token_id") and target.config.eos_token_id is not None:
        stop_token_ids = [int(target.config.eos_token_id)]

    for _ in range(num_warmup):
        draft.spec_generate(
            target=target,
            input_ids=input_ids,
            max_new_tokens=max_new_tokens,
            stop_token_ids=stop_token_ids,
            temperature=0.0,
            verify_block_size=verify_block_size,
            compile_serial_head=compile_serial_head,
            decode_timing_after_first_token=True,
        )
        _sync(device)

    _sync(device)
    t0 = time.perf_counter()
    output_ids = draft.spec_generate(
        target=target,
        input_ids=input_ids,
        max_new_tokens=max_new_tokens,
        stop_token_ids=stop_token_ids,
        temperature=0.0,
        verify_block_size=verify_block_size,
        compile_serial_head=compile_serial_head,
        decode_timing_after_first_token=True,
    )
    _sync(device)
    wall = time.perf_counter() - t0

    stats = draft.get_last_decode_stats()
    num_input = input_ids.shape[1]
    num_output = output_ids.shape[1] - num_input
    accept = stats.get("accept_lengths", [])
    avg_accept = _avg([float(x) for x in accept]) if accept else 0.0
    return E2EResult(
        compile_serial_head=compile_serial_head,
        sample_index=sample_index,
        input_len=int(num_input),
        output_tokens=int(num_output),
        decode_steps=int(stats.get("steps", 0)),
        avg_accept_length=avg_accept,
        decode_wall_s=wall,
        s_per_output_token=wall / num_output if num_output > 0 else 0.0,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--target-model-path",
        default="/share/dai-sys/wanghanzhen/models/Qwen/Qwen3-8B",
    )
    parser.add_argument(
        "--draft-ckpt",
        default="/share/dai-sys/wanghanzhen/projects/MTP/FlashMTP_v2/cache/models/flashmtp_v2_mhrnn_direct_r512_ce0.1_tv1.0_wb_0.0_bgemma_21_qwen3_8b",
    )
    parser.add_argument("--dataset", default="gsm8k")
    parser.add_argument("--max-samples", type=int, default=3)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--verify-block", type=int, default=None)
    parser.add_argument("--warmup-steps", type=int, default=30)
    parser.add_argument("--timed-steps", type=int, default=100)
    parser.add_argument("--e2e-warmup", type=int, default=1)
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    out_dir = (
        Path(args.output_dir)
        if args.output_dir
        else PROJECT_ROOT / "profile"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        import flash_attn  # noqa: F401

        attn_impl = "flash_attention_2"
    except ImportError:
        attn_impl = "sdpa"

    device = torch.device("cuda:0")
    torch.cuda.set_device(device)

    target = AutoModelForCausalLM.from_pretrained(
        args.target_model_path,
        attn_implementation=attn_impl,
        dtype=torch.bfloat16,
        trust_remote_code=True,
    ).to(device).eval()
    draft = FlashMTPDraftModel.from_pretrained(
        args.draft_ckpt,
        dtype=torch.bfloat16,
        trust_remote_code=True,
    ).to(device).eval()
    verify_block_size = (
        draft.proposal_length + 1
        if args.verify_block is None
        else int(args.verify_block)
    )
    if not 1 <= verify_block_size <= draft.proposal_length + 1:
        raise ValueError(
            f"--verify-block must be in [1, {draft.proposal_length + 1}], "
            f"got {verify_block_size}"
        )

    tokenizer = AutoTokenizer.from_pretrained(
        args.target_model_path, trust_remote_code=True
    )
    dataset = select_max_samples(load_benchmark_dataset(args.dataset), args.max_samples)

    step_results: list[dict] = []
    e2e_results: list[dict] = []

    # Use first sample prompt for micro-benchmark (fixed KV state)
    first_turn = dataset[0]["turns"][0]
    messages = [{"role": "user", "content": first_turn}]
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    micro_input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)

    for compile_flag in (False, True):
        label = "compile_on" if compile_flag else "compile_off"
        print(f"\n=== step breakdown ({label}) ===", flush=True)
        bd = profile_one_step(
            draft=draft,
            target=target,
            input_ids=micro_input_ids,
            compile_serial_head=compile_flag,
            num_warmup=args.warmup_steps,
            num_timed=args.timed_steps,
        )
        step_results.append(asdict(bd))
        print(
            f"  backbone={bd.draft_backbone_ms:.3f} lm_head={bd.target_lm_head_ms:.3f} "
            f"serial={bd.serial_head_ms:.3f} verify={bd.target_verify_ms:.3f} "
            f"step={bd.step_total_ms:.3f} serial_frac={bd.serial_head_fraction*100:.1f}%",
            flush=True,
        )

    off = step_results[0]
    on = step_results[1]
    serial_speedup = off["serial_head_ms"] / on["serial_head_ms"] if on["serial_head_ms"] > 0 else 1.0
    step_speedup_meas = off["step_total_ms"] / on["step_total_ms"] if on["step_total_ms"] > 0 else 1.0
    step_speedup_theory = _theoretical_speedup(off["serial_head_fraction"], serial_speedup)

    for si, instance in enumerate(dataset):
        turn = instance["turns"][0]
        messages = [{"role": "user", "content": turn}]
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
        for compile_flag in (False, True):
            label = "compile_on" if compile_flag else "compile_off"
            print(f"\n=== e2e sample={si} ({label}) ===", flush=True)
            res = profile_e2e(
                draft=draft,
                target=target,
                input_ids=input_ids,
                compile_serial_head=compile_flag,
                max_new_tokens=args.max_new_tokens,
                verify_block_size=verify_block_size,
                sample_index=si,
                num_warmup=args.e2e_warmup,
            )
            e2e_results.append(asdict(res))
            print(
                f"  tokens={res.output_tokens} steps={res.decode_steps} "
                f"accept={res.avg_accept_length:.2f} s/tok={res.s_per_output_token:.5f}",
                flush=True,
            )

    e2e_by_sample: dict[int, dict[str, E2EResult]] = {}
    for row in e2e_results:
        si = row["sample_index"]
        key = "on" if row["compile_serial_head"] else "off"
        e2e_by_sample.setdefault(si, {})[key] = row

    e2e_speedups = []
    for si, pair in e2e_by_sample.items():
        if "on" in pair and "off" in pair:
            off_t = pair["off"]["s_per_output_token"]
            on_t = pair["on"]["s_per_output_token"]
            if on_t > 0:
                e2e_speedups.append(off_t / on_t)

    payload = {
        "config": {
            "target_model": args.target_model_path,
            "draft_ckpt": args.draft_ckpt,
            "dataset": args.dataset,
            "max_samples": args.max_samples,
            "max_new_tokens": args.max_new_tokens,
            "verify_block": verify_block_size,
            "block_size": int(draft.block_size),
            "markov_output_mode": draft.markov_output_mode,
            "markov_rank": int(draft.markov_rank),
            "warmup_steps": args.warmup_steps,
            "timed_steps": args.timed_steps,
        },
        "step_breakdown": step_results,
        "step_analysis": {
            "serial_head_speedup": serial_speedup,
            "step_speedup_measured": step_speedup_meas,
            "step_speedup_theoretical": step_speedup_theory,
            "serial_head_fraction_off": off["serial_head_fraction"],
            "verify_fraction_off": off["verify_fraction"],
            "backbone_fraction_off": off["draft_backbone_ms"] / off["step_total_ms"],
        },
        "e2e_results": e2e_results,
        "e2e_analysis": {
            "mean_e2e_speedup": _avg(e2e_speedups),
            "per_sample_speedup": {
                str(si): pair["off"]["s_per_output_token"] / pair["on"]["s_per_output_token"]
                for si, pair in e2e_by_sample.items()
                if "on" in pair and "off" in pair and pair["on"]["s_per_output_token"] > 0
            },
            "accept_length_match": all(
                abs(pair["off"]["avg_accept_length"] - pair["on"]["avg_accept_length"]) < 0.01
                for pair in e2e_by_sample.values()
                if "on" in pair and "off" in pair
            ),
        },
    }

    json_path = out_dir / "compile_serial_head_timing.json"
    json_path.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"\nWrote {json_path}", flush=True)


if __name__ == "__main__":
    main()
