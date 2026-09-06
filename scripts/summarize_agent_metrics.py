#!/usr/bin/env python3
"""Print per-turn context/acceptance metrics and aggregate target speedup."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


FIELDS = (
    "run_id",
    "harness",
    "task",
    "model",
    "decode_mode",
    "request_index",
    "turn_index",
    "context_tokens_at_turn_start",
    "output_tokens",
    "average_accept_length",
    "average_draft_matches",
    "accept_lengths",
    "decode_wall_time_s",
    "generation_wall_time_s",
    "generation_tokens_per_s",
    "decode_tokens_per_s",
    "tool_call_count",
    "finish_reason",
)


def load_rows(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON on {path}:{line_number}: {exc}") from exc
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("metrics_jsonl", type=Path)
    parser.add_argument("--csv", type=Path)
    parser.add_argument("--run-id", help="Only include rows with this run_id tag")
    parser.add_argument("--harness", help="Only include rows with this harness tag")
    parser.add_argument("--model", help="Only include requests whose model exactly matches")
    parser.add_argument(
        "--min-context-tokens-exclusive",
        type=int,
        help="Only include requests whose context_tokens_at_turn_start is greater than this value",
    )
    args = parser.parse_args()
    rows = load_rows(args.metrics_jsonl)
    filters = {
        "run_id": args.run_id,
        "harness": args.harness,
        "model": args.model,
    }
    rows = [
        row
        for row in rows
        if all(value is None or str(row.get(key)) == value for key, value in filters.items())
    ]
    if args.min_context_tokens_exclusive is not None:
        rows = [
            row
            for row in rows
            if int(row.get("context_tokens_at_turn_start", 0))
            > args.min_context_tokens_exclusive
        ]

    print(
        "request\tmodel\tturn\tcontext\toutput\tavg_accept\taccept_lengths"
        "\tgeneration_s\tdecode_s\tgeneration_tok/s\tdecode_tok/s\ttool_calls"
    )
    for request_index, row in enumerate(rows, 1):
        print(
            "\t".join(
                str(value)
                for value in (
                    request_index,
                    row.get("model", "-"),
                    row.get("turn_index", "-"),
                    row.get("context_tokens_at_turn_start", "-"),
                    row.get("output_tokens", "-"),
                    row.get("average_accept_length", "-"),
                    json.dumps(row.get("accept_lengths", []), separators=(",", ":")),
                    round(float(row.get("generation_wall_time_s", 0.0)), 4),
                    round(float(row.get("decode_wall_time_s", 0.0)), 4),
                    round(float(row.get("generation_tokens_per_s", 0.0)), 2),
                    round(float(row.get("decode_tokens_per_s", 0.0)), 2),
                    row.get("tool_call_count", 0),
                )
            )
        )

    grouped: dict[tuple[str, str, str], dict[str, float]] = defaultdict(
        lambda: {"wall": 0.0, "decode": 0.0, "tokens": 0.0, "requests": 0.0}
    )
    for row in rows:
        key = (str(row.get("task", "-")), str(row.get("harness", "-")), str(row["decode_mode"]))
        grouped[key]["wall"] += float(row.get("generation_wall_time_s", 0.0))
        grouped[key]["decode"] += float(row.get("decode_wall_time_s", 0.0))
        grouped[key]["tokens"] += float(row.get("output_tokens", 0.0))
        grouped[key]["requests"] += 1

    print("\naggregate (only compare target/flashmtp when request and token counts match closely):")
    for key, values in sorted(grouped.items()):
        task, harness, mode = key
        tps = values["tokens"] / max(values["wall"], 1e-9)
        print(
            f"{task}/{harness}/{mode}: requests={int(values['requests'])} "
            f"tokens={int(values['tokens'])} generation_wall={values['wall']:.3f}s "
            f"throughput={tps:.2f} tok/s"
        )
    task_harness_pairs = {(task, harness) for task, harness, _ in grouped}
    for task, harness in sorted(task_harness_pairs):
        target = grouped.get((task, harness, "target"))
        flash = grouped.get((task, harness, "flashmtp"))
        if target and flash and flash["wall"] > 0:
            print(f"{task}/{harness}: wall-time speedup target/flashmtp={target['wall'] / flash['wall']:.3f}x")

    if args.csv:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        with args.csv.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=FIELDS, extrasaction="ignore")
            writer.writeheader()
            for request_index, row in enumerate(rows, 1):
                exported = dict(row)
                exported["request_index"] = request_index
                exported["accept_lengths"] = json.dumps(row.get("accept_lengths", []))
                writer.writerow(exported)


if __name__ == "__main__":
    main()
