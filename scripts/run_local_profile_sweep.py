#!/usr/bin/env python3
"""Sweep additive r256 vs direct r512 profile at multiple batch sizes.

Runs draft micro-profile + full spec-step breakdown per (model, batch) in
isolated subprocesses to avoid GPU OOM.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _run(cmd: list[str]) -> dict:
    proc = subprocess.run(
        cmd,
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        print(proc.stdout, file=sys.stderr)
        print(proc.stderr, file=sys.stderr)
        raise RuntimeError(f"Command failed ({proc.returncode}): {' '.join(cmd)}")
    return json.loads(proc.stdout.strip().split("===JSON_RESULT===")[-1])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--additive-ckpt",
        default=str(
            PROJECT_ROOT
            / "cache/models/flashmtp_h100_prefix_condition_fuse18_sample_80000_nlayers5_block_16_mhrnn_additive_r256_wb_0.0_maxlen4096_epochs6_Qwen3-8B/epoch_6_step_59496"
        ),
    )
    parser.add_argument(
        "--direct-ckpt",
        default=str(
            PROJECT_ROOT
            / "cache/models/flashmtp_h100_prefix_condition_fuse18_sample_80000_nlayers5_block_16_mhrnn_direct_r512_wb_0.2_bgemma_21_maxlen4096_epochs6_Qwen3-8B/epoch_6_step_59496"
        ),
    )
    parser.add_argument("--target-model-path", default="/data/wanghanzhen/models/Qwen3-8B")
    parser.add_argument("--batch-sizes", default="1,8,32,64")
    parser.add_argument("--warmup-steps", type=int, default=20)
    parser.add_argument("--timed-steps", type=int, default=100)
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    batch_sizes = [int(x) for x in args.batch_sizes.split(",")]
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.output_dir or PROJECT_ROOT / f"log/local_profile_sweep_{ts}")
    out_dir.mkdir(parents=True, exist_ok=True)

    worker = PROJECT_ROOT / "scripts/_profile_sweep_worker.py"
    results: list[dict] = []

    models = [
        ("additive_r256", args.additive_ckpt),
        ("direct_r512", args.direct_ckpt),
    ]

    for label, ckpt in models:
        for bs in batch_sizes:
            print(f"\n>>> {label} batch={bs}", flush=True)
            cmd = [
                sys.executable,
                str(worker),
                "--draft-ckpt",
                ckpt,
                "--target-model-path",
                args.target_model_path,
                "--batch-size",
                str(bs),
                "--warmup-steps",
                str(args.warmup_steps),
                "--timed-steps",
                str(args.timed_steps),
            ]
            row = _run(cmd)
            row["label"] = label
            results.append(row)
            out_path = out_dir / f"{label}_b{bs}.json"
            out_path.write_text(json.dumps(row, indent=2) + "\n")
            print(
                f"  draft={row['draft_total_ms']:.2f}ms "
                f"verify={row['target_verify_ms']:.2f}ms "
                f"step={row['step_total_ms']:.2f}ms",
                flush=True,
            )

    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps(results, indent=2) + "\n")
    print(f"\nWrote {summary_path}")


if __name__ == "__main__":
    main()
