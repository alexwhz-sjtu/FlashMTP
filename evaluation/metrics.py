"""Aggregate and print benchmark timing / acceptance metrics."""

from __future__ import annotations

from itertools import chain

import numpy as np
from rich import print

from evaluation.generation import decode_wall_seconds, decode_weight


def summarize_responses(responses: list[dict], block_size: int, batch_size: int) -> None:
    baseline_key = 1
    spec_key = block_size

    w_base = sum(decode_weight(r[baseline_key]) for r in responses)
    w_spec = sum(decode_weight(r[spec_key]) for r in responses)
    d_base = sum(decode_wall_seconds(r[baseline_key]) for r in responses)
    d_spec = sum(decode_wall_seconds(r[spec_key]) for r in responses)

    t_base = d_base / max(w_base, 1)
    t_spec = d_spec / max(w_spec, 1)
    tp_base = w_base / max(d_base, 1e-9)
    tp_spec = w_spec / max(d_spec, 1e-9)

    t_base_mean = float(np.mean([r[baseline_key].time_per_output_token for r in responses]))
    t_spec_mean = float(np.mean([r[spec_key].time_per_output_token for r in responses]))

    print(
        f"Decoding speedup (token-weighted, batch_size={batch_size}): "
        f"{t_base / max(t_spec, 1e-30):.2f}  |  "
        f"throughput baseline={tp_base:.2f} flashmtp={tp_spec:.2f} "
        f"ratio={tp_spec / max(tp_base, 1e-30):.2f}"
    )
    print(
        f"  Global decode s/token baseline={t_base:.6f} flashmtp={t_spec:.6f} | "
        f"per-sample mean {t_base_mean:.6f} / {t_spec_mean:.6f} "
        f"(ratio {t_base_mean / max(t_spec_mean, 1e-30):.2f})"
    )

    acceptance_lengths = list(chain(*[r[spec_key].acceptance_lengths for r in responses]))
    if not acceptance_lengths:
        print("Acceptance length histogram: (empty)")
        return

    histogram = [
        acceptance_lengths.count(b) / len(acceptance_lengths) for b in range(block_size + 1)
    ]
    print(f"Acceptance length histogram: {[f'{x * 100:.1f}%' for x in histogram]}")
    avg_accept = sum(index * frac for index, frac in enumerate(histogram))
    print(f"Average acceptance length: {avg_accept:.2f}")
