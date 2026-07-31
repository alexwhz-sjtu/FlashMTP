#!/usr/bin/env python3
"""Profile eager versus compiled FlashMTP serial heads with real weights."""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from torch import nn

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from specforge.modeling.draft.flashmtp import FlashMTPDraftModel


class _UnusedLMHead(nn.Module):
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        raise AssertionError("direct serial-head mode must not call the base LM head")


@dataclass
class SerialHeadProfile:
    label: str
    checkpoint: str
    head_type: str
    output_mode: str
    temperature: float
    batch_size: int
    prediction_length: int
    hidden_size: int
    markov_rank: int
    vocab_size: int
    warmup_iterations: int
    timed_iterations: int
    repeats: int
    eager_mean_ms: float
    eager_min_ms: float
    compiled_mean_ms: float
    compiled_min_ms: float
    steady_state_speedup: float
    compile_first_call_seconds: float


def _sync() -> None:
    torch.cuda.synchronize()


def _measure_ms(fn, *, iterations: int, repeats: int) -> list[float]:
    measurements: list[float] = []
    for _ in range(repeats):
        _sync()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(iterations):
            fn()
        end.record()
        end.synchronize()
        measurements.append(float(start.elapsed_time(end)) / iterations)
    return measurements


@torch.inference_mode()
def profile_checkpoint(
    *,
    label: str,
    checkpoint: str,
    temperatures: list[float],
    batch_size: int,
    warmup_iterations: int,
    timed_iterations: int,
    repeats: int,
) -> list[SerialHeadProfile]:
    device = torch.device("cuda")
    model = FlashMTPDraftModel.from_pretrained(
        checkpoint,
        dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    ).to(device).eval()
    if model.markov_head is None:
        raise ValueError(f"Checkpoint has no serial head: {checkpoint}")
    if model.markov_output_mode != "direct":
        raise ValueError(
            "This profile expects direct mode so base LM-head cost is excluded, "
            f"got {model.markov_output_mode!r}."
        )

    prediction_length = int(model.block_size) - 1
    hidden = torch.randn(
        batch_size,
        prediction_length,
        model.config.hidden_size,
        dtype=torch.bfloat16,
        device=device,
    )
    first_previous_ids = torch.randint(
        0,
        model.config.vocab_size,
        (batch_size,),
        dtype=torch.long,
        device=device,
    )
    unused_lm_head = _UnusedLMHead().to(device)
    results: list[SerialHeadProfile] = []

    for temperature in temperatures:
        temperature = float(temperature)

        def eager_call():
            return model.sample_draft_tokens(
                draft_hidden=hidden,
                lm_head=unused_lm_head,
                first_prev_token_ids=first_previous_ids,
                temperature=temperature,
                compile_serial_head=False,
            )

        for _ in range(warmup_iterations):
            eager_call()
        eager_ms = _measure_ms(
            eager_call,
            iterations=timed_iterations,
            repeats=repeats,
        )

        _sync()
        compile_start = time.perf_counter()
        model.sample_draft_tokens(
            draft_hidden=hidden,
            lm_head=unused_lm_head,
            first_prev_token_ids=first_previous_ids,
            temperature=temperature,
            compile_serial_head=True,
        )
        _sync()
        compile_first_call_seconds = time.perf_counter() - compile_start

        def compiled_call():
            return model.sample_draft_tokens(
                draft_hidden=hidden,
                lm_head=unused_lm_head,
                first_prev_token_ids=first_previous_ids,
                temperature=temperature,
                compile_serial_head=True,
            )

        for _ in range(warmup_iterations):
            compiled_call()
        compiled_ms = _measure_ms(
            compiled_call,
            iterations=timed_iterations,
            repeats=repeats,
        )

        eager_mean = sum(eager_ms) / len(eager_ms)
        compiled_mean = sum(compiled_ms) / len(compiled_ms)
        results.append(
            SerialHeadProfile(
                label=label,
                checkpoint=str(Path(checkpoint).resolve()),
                head_type=model.markov_head_type,
                output_mode=model.markov_output_mode,
                temperature=temperature,
                batch_size=batch_size,
                prediction_length=prediction_length,
                hidden_size=int(model.config.hidden_size),
                markov_rank=int(model.markov_rank),
                vocab_size=int(model.config.vocab_size),
                warmup_iterations=warmup_iterations,
                timed_iterations=timed_iterations,
                repeats=repeats,
                eager_mean_ms=eager_mean,
                eager_min_ms=min(eager_ms),
                compiled_mean_ms=compiled_mean,
                compiled_min_ms=min(compiled_ms),
                steady_state_speedup=eager_mean / compiled_mean,
                compile_first_call_seconds=compile_first_call_seconds,
            )
        )

    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rnn-easy-draft", required=True)
    parser.add_argument("--rnn-draft", required=True)
    parser.add_argument("--temperatures", default="0,1")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--warmup-iterations", type=int, default=20)
    parser.add_argument("--timed-iterations", type=int, default=200)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--output-json", type=Path, default=None)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this profile.")
    torch.manual_seed(0)
    temperatures = [float(item) for item in args.temperatures.split(",")]

    profiles: list[SerialHeadProfile] = []
    for label, checkpoint in (
        ("rnn_easy", args.rnn_easy_draft),
        ("rnn", args.rnn_draft),
    ):
        print(f"Profiling {label}: {checkpoint}", flush=True)
        profiles.extend(
            profile_checkpoint(
                label=label,
                checkpoint=checkpoint,
                temperatures=temperatures,
                batch_size=args.batch_size,
                warmup_iterations=args.warmup_iterations,
                timed_iterations=args.timed_iterations,
                repeats=args.repeats,
            )
        )
        torch.cuda.empty_cache()

    output = [asdict(profile) for profile in profiles]
    print(json.dumps(output, indent=2))
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(output, indent=2) + "\n")


if __name__ == "__main__":
    main()
