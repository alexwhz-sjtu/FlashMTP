#!/usr/bin/env python3
"""模拟训练时 `FlashMTP` 的 per-block 掩码比例 r 的抽样分布（与 specforge.core.flashmtp 一致）。"""

from __future__ import annotations

import argparse
import math
from typing import Sequence

import numpy as np

SCHEDULES = ("uniform", "cosine", "mask_high")


def sample_mask_ratios(
    schedule: str,
    ratio_min: float,
    ratio_max: float,
    n: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """与 FlashMTP._sample_diffusion_mask_ratios 中 block_keep=True 时相同的 r 分布。"""
    u = rng.random(n)
    lo, hi = ratio_min, ratio_max
    if schedule == "uniform":
        r = lo + (hi - lo) * u
    elif schedule == "cosine":
        r = lo + (hi - lo) * (np.sin(u * (math.pi / 2.0)) ** 2)
    elif schedule == "mask_high":
        r = lo + (hi - lo) * np.sqrt(u)
    else:
        raise ValueError(f"Unknown schedule: {schedule!r}; expected one of {SCHEDULES}")
    return r.astype(np.float64)


def ascii_histogram(
    x: np.ndarray,
    lo: float,
    hi: float,
    n_bins: int = 20,
    width: int = 50,
    as_probability: bool = True,
) -> str:
    counts, edges = np.histogram(x, bins=n_bins, range=(lo, hi))
    n = int(counts.sum())
    if n == 0 or counts.size == 0:
        return "(empty)\n"
    if as_probability:
        vals = counts.astype(np.float64) / n
        m = float(vals.max())
    else:
        vals = counts.astype(np.float64)
        m = float(vals.max())
    if m <= 0:
        return "(empty)\n"
    lines = []
    for i, v in enumerate(vals):
        left, right = edges[i], edges[i + 1]
        bar = "#" * max(1, int(width * v / m)) if v > 0 else ""
        if as_probability:
            lines.append(
                f"  [{left:5.3f}, {right:5.3f})  P={v:.5f}  {100.0 * v:6.2f}%  {bar}"
            )
        else:
            lines.append(f"  [{left:5.3f}, {right:5.3f})  {int(v):6d}  {bar}")
    if as_probability:
        s = float(vals.sum())
        lines.append(
            f"  (sum of P over bins = {s:.5f}; =1 iff all samples in [{lo}, {hi}])"
        )
    return "\n".join(lines) + "\n"


def summarize(name: str, r: np.ndarray) -> str:
    qs = (5, 25, 50, 75, 95)
    pct = {q: float(np.percentile(r, q)) for q in qs}
    return (
        f"--- {name} ---\n"
        f"  n={len(r)}  mean={r.mean():.4f}  std={r.std():.4f}  "
        f"min={r.min():.4f}  max={r.max():.4f}\n"
        f"  p5={pct[5]:.4f}  p25={pct[25]:.4f}  p50={pct[50]:.4f}  "
        f"p75={pct[75]:.4f}  p95={pct[95]:.4f}\n"
    )


def main() -> None:
    p = argparse.ArgumentParser(
        description="Simulate mask ratio r ~ schedule (same as training FlashMTP discrete_diffusion)."
    )
    p.add_argument(
        "--schedule",
        type=str,
        default="all",
        help=f"One of {SCHEDULES}, or 'all' to print all (default: all).",
    )
    p.add_argument(
        "--diffusion-mask-ratio-min",
        type=float,
        default=0.1,
        dest="ratio_min",
    )
    p.add_argument(
        "--diffusion-mask-ratio-max",
        type=float,
        default=1.0,
        dest="ratio_max",
    )
    p.add_argument(
        "--samples",
        type=int,
        default=200_000,
        help="Number of i.i.d. r draws per schedule.",
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--bins",
        type=int,
        default=20,
        help="Histogram bins in [ratio_min, ratio_max].",
    )
    p.add_argument(
        "--no-hist",
        action="store_true",
        help="Only print summary stats, skip ASCII histogram.",
    )
    p.add_argument(
        "--raw-count",
        action="store_true",
        help="Histogram y-axis: raw counts (default: empirical probability per bin).",
    )
    args = p.parse_args()

    if not (0.0 <= args.ratio_min <= args.ratio_max <= 1.0):
        p.error("Need 0 <= ratio_min <= ratio_max <= 1.0")
    if args.samples < 1:
        p.error("samples must be >= 1")

    rng = np.random.default_rng(args.seed)
    if args.schedule == "all":
        schedules: Sequence[str] = SCHEDULES
    else:
        if args.schedule not in SCHEDULES:
            p.error(f"schedule must be one of {SCHEDULES} or 'all'")
        schedules = (args.schedule,)

    for sch in schedules:
        r = sample_mask_ratios(
            sch, args.ratio_min, args.ratio_max, args.samples, rng
        )
        print(summarize(f"{sch}  (r in [{args.ratio_min}, {args.ratio_max}])", r))
        if not args.no_hist:
            if args.raw_count:
                print("  Histogram (count per bin):")
            else:
                print("  Histogram (P per bin = count / n, plus %):")
            print(
                ascii_histogram(
                    r,
                    args.ratio_min,
                    args.ratio_max,
                    n_bins=args.bins,
                    as_probability=not args.raw_count,
                )
            )


if __name__ == "__main__":
    main()
