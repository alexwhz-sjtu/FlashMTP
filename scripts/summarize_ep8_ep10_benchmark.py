#!/usr/bin/env python3
"""Summarize the ep8/ep10 long- and short-context benchmark run."""

from __future__ import annotations

import argparse
import csv
import json
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

DISPLAY_NAMES = {
    "longbench_v2_64000_32000_single_document_qa": "LongBench-v2 single-document QA",
    "longbench_v2_64000_32000_multi_document_qa": "LongBench-v2 multi-document QA",
    "longbench_v2_64000_32000_long_dialogue": "LongBench-v2 long-dialogue",
    "longbench_v2_64000_32000_structured_data": "LongBench-v2 structured-data",
    "longbench_v2_64000_32000_in_context_learning": "LongBench-v2 in-context learning",
    "longbench_v2_64000_32000_code_repo": "LongBench-v2 code-repo",
}


def last_match(pattern: re.Pattern[str], text: str):
    matches = pattern.findall(text)
    return matches[-1] if matches else None


def parse_log(log_path: Path) -> dict[str, str]:
    text = log_path.read_text(encoding="utf-8", errors="replace")
    overall_match = re.search(
        r"^=== Overall \(batch_size=\d+\) ===\s*(.*?)(?=^===|\Z)",
        text,
        flags=re.MULTILINE | re.DOTALL,
    )
    overall = overall_match.group(1) if overall_match else ""
    decode_times = last_match(PATTERNS["decode_times"], overall)
    return {
        "turns": last_match(PATTERNS["turns"], overall) or "",
        "average_acceptance_length": last_match(PATTERNS["acceptance"], overall) or "",
        "token_weighted_speedup": last_match(PATTERNS["speedup"], overall) or "",
        "throughput_ratio": last_match(PATTERNS["throughput_ratio"], overall) or "",
        "unweighted_speedup": last_match(PATTERNS["unweighted_speedup"], overall) or "",
        "baseline_s_per_token": decode_times[0] if decode_times else "",
        "flashmtp_s_per_token": decode_times[1] if decode_times else "",
        "elapsed_seconds": last_match(PATTERNS["elapsed"], text) or "",
    }


def read_status(status_path: Path) -> str:
    if not status_path.exists():
        return "missing"
    lines = status_path.read_text(encoding="utf-8", errors="replace").splitlines()
    return lines[0] if lines else "unknown"


def markdown_report(rows: list[dict[str, str]]) -> str:
    lines = [
        "# FlashMTP ep8 / ep10 benchmark",
        "",
        "Configuration: max-samples=50, max-new-tokens=512, batch-size=1, "
        "temperature=0, input length unrestricted.",
        "",
    ]
    for model in ("ep10", "ep8"):
        lines.extend(
            [
                f"## {model}",
                "",
                "| Dataset | Turns | Average acceptance length | Speedup | Status |",
                "|---|---:|---:|---:|---|",
            ]
        )
        for row in rows:
            if row["model"] != model:
                continue
            dataset = DISPLAY_NAMES.get(row["dataset"], row["dataset"])
            speedup = (
                f'{row["token_weighted_speedup"]}x'
                if row["token_weighted_speedup"]
                else "—"
            )
            acceptance = row["average_acceptance_length"] or "—"
            turns = row["turns"] or "—"
            lines.append(
                f'| {dataset} | {turns} | {acceptance} | {speedup} | {row["status"]} |'
            )
        lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_root", type=Path)
    args = parser.parse_args()

    run_root = args.run_root.resolve()
    rows: list[dict[str, str]] = []
    with (run_root / "manifest.tsv").open(encoding="utf-8") as manifest_file:
        for item in csv.DictReader(manifest_file, delimiter="\t"):
            log_path = Path(item["log_path"])
            status_path = Path(item["status_path"])
            row = dict(item)
            row["status"] = read_status(status_path)
            row.update(parse_log(log_path) if log_path.exists() else {})
            rows.append(row)

    fieldnames = [
        "model",
        "dataset",
        "gpu",
        "status",
        "turns",
        "average_acceptance_length",
        "token_weighted_speedup",
        "throughput_ratio",
        "unweighted_speedup",
        "baseline_s_per_token",
        "flashmtp_s_per_token",
        "elapsed_seconds",
        "draft_path",
        "log_path",
        "status_path",
    ]
    with (run_root / "summary.csv").open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    (run_root / "results.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (run_root / "report.md").write_text(markdown_report(rows), encoding="utf-8")

    completed = sum(row["status"] == "completed" for row in rows)
    failed = sum(row["status"] == "failed" for row in rows)
    print(
        f"total={len(rows)} completed={completed} failed={failed} "
        f"incomplete={len(rows) - completed - failed}"
    )


if __name__ == "__main__":
    main()
