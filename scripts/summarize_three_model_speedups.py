#!/usr/bin/env python3
"""Summarize logs produced by run_three_model_speedup_benchmarks.sh."""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path


PATTERNS = {
    "turns": re.compile(r"turns:\s*(\d+)"),
    "speedup": re.compile(r"token-weighted speedup:\s*([0-9.]+)x"),
    "throughput_ratio": re.compile(r"throughput ratio:\s*([0-9.]+)x"),
    "unweighted_speedup": re.compile(r"unweighted speedup:\s*([0-9.]+)x"),
    "decode_times": re.compile(
        r"decode s/token baseline=([0-9.eE+-]+)\s+flashmtp=([0-9.eE+-]+)"
    ),
    "acceptance": re.compile(r"average acceptance length:\s*([0-9.]+)"),
    "elapsed": re.compile(r"Total elapsed time:\s*([0-9.]+)s"),
}


def last_match(pattern: re.Pattern[str], text: str):
    matches = pattern.findall(text)
    return matches[-1] if matches else None


def parse_log(log_path: Path) -> dict[str, str]:
    text = log_path.read_text(encoding="utf-8", errors="replace")
    decode_times = last_match(PATTERNS["decode_times"], text)
    return {
        "turns": last_match(PATTERNS["turns"], text) or "",
        "token_weighted_speedup": last_match(PATTERNS["speedup"], text) or "",
        "throughput_ratio": last_match(PATTERNS["throughput_ratio"], text) or "",
        "unweighted_speedup": last_match(PATTERNS["unweighted_speedup"], text)
        or "",
        "baseline_s_per_token": decode_times[0] if decode_times else "",
        "flashmtp_s_per_token": decode_times[1] if decode_times else "",
        "average_acceptance_length": last_match(PATTERNS["acceptance"], text)
        or "",
        "elapsed_seconds": last_match(PATTERNS["elapsed"], text) or "",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_root", type=Path)
    args = parser.parse_args()

    run_root = args.run_root.resolve()
    manifest_path = run_root / "manifest.tsv"
    rows: list[dict[str, str]] = []
    with manifest_path.open(encoding="utf-8") as manifest_file:
        for item in csv.DictReader(manifest_file, delimiter="\t"):
            log_path = Path(item["log_path"])
            status_path = Path(item["status_path"])
            status = (
                status_path.read_text(encoding="utf-8", errors="replace").strip()
                if status_path.exists()
                else "missing"
            )
            row = dict(item)
            row["status"] = status.splitlines()[0] if status else "unknown"
            row.update(parse_log(log_path) if log_path.exists() else {})
            rows.append(row)

    fieldnames = [
        "model",
        "temperature",
        "dataset",
        "requested_samples",
        "gpu",
        "status",
        "turns",
        "token_weighted_speedup",
        "throughput_ratio",
        "unweighted_speedup",
        "baseline_s_per_token",
        "flashmtp_s_per_token",
        "average_acceptance_length",
        "elapsed_seconds",
        "draft_path",
        "log_path",
        "status_path",
    ]
    summary_path = run_root / "summary.csv"
    with summary_path.open("w", encoding="utf-8", newline="") as summary_file:
        writer = csv.DictWriter(summary_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    completed = sum(row["status"] == "completed" for row in rows)
    failed = sum(row["status"].startswith("failed") for row in rows)
    print(
        f"Wrote {summary_path}: total={len(rows)} completed={completed} "
        f"failed={failed} incomplete={len(rows) - completed - failed}"
    )


if __name__ == "__main__":
    main()
