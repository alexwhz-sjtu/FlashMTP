#!/usr/bin/env python3
"""Print summary from compile_serial_head_timing.json."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("json_path", nargs="?", default="compile_serial_head_timing.json")
    args = parser.parse_args()
    path = Path(args.json_path)
    data = json.loads(path.read_text())

    off = next(r for r in data["step_breakdown"] if not r["compile_serial_head"])
    on = next(r for r in data["step_breakdown"] if r["compile_serial_head"])
    sa = data["step_analysis"]
    ea = data["e2e_analysis"]

    print(f"=== Step breakdown ({path.name}) ===")
    print(f"{'component':<18} {'compile_off_ms':>14} {'compile_on_ms':>14} {'speedup':>8}")
    for key, label in [
        ("draft_backbone_ms", "draft_backbone"),
        ("target_lm_head_ms", "target_lm_head"),
        ("serial_head_ms", "markov_serial_head"),
        ("target_verify_ms", "target_verify"),
        ("step_total_ms", "step_total"),
    ]:
        a, b = off[key], on[key]
        sp = a / b if b > 0 else float("inf")
        print(f"{label:<18} {a:14.3f} {b:14.3f} {sp:8.3f}x")

    print()
    print(f"serial_head fraction (off): {sa['serial_head_fraction_off']*100:.1f}%")
    print(f"verify fraction (off):      {sa['verify_fraction_off']*100:.1f}%")
    print(f"serial_head speedup:        {sa['serial_head_speedup']:.3f}x")
    print(f"step speedup (measured):    {sa['step_speedup_measured']:.3f}x")
    print(f"step speedup (theoretical): {sa['step_speedup_theoretical']:.3f}x")
    print(f"e2e speedup (mean):         {ea['mean_e2e_speedup']:.3f}x")
    print(f"accept lengths match:       {ea['accept_length_match']}")


if __name__ == "__main__":
    main()
