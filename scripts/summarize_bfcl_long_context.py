#!/usr/bin/env python3
"""Map BFCL model calls to sessions/turns and summarize long-context acceptance."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


FORCED_TOOL_REASON_PREFIX = (
    "<tool_reason>\n"
    "Based on the user's request and the available context, I will "
)


def visible_response_text(row: dict[str, Any]) -> str:
    text = str(row.get("response_text") or "")
    if (
        row.get("visible_tool_reason") == "forced"
        and text
        and not text.startswith("<tool_reason>")
    ):
        return FORCED_TOOL_REASON_PREFIX + text
    return text


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON at {path}:{line_number}: {exc}") from exc
    return rows


def quantile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


def flatten_bfcl_calls(result_rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    calls: list[dict[str, Any]] = []
    for session in result_rows:
        if "input_token_count" not in session:
            continue
        input_counts = session["input_token_count"]
        output_counts = session["output_token_count"]
        latencies = session["latency"]
        if not (len(input_counts) == len(output_counts) == len(latencies)):
            raise ValueError(f"turn count mismatch in {session['id']}")
        for user_turn_index, (turn_inputs, turn_outputs, turn_latencies) in enumerate(
            zip(input_counts, output_counts, latencies, strict=True)
        ):
            if not (len(turn_inputs) == len(turn_outputs) == len(turn_latencies)):
                raise ValueError(
                    f"step count mismatch in {session['id']} turn {user_turn_index}"
                )
            for step_index, (input_tokens, output_tokens, latency_s) in enumerate(
                zip(turn_inputs, turn_outputs, turn_latencies, strict=True)
            ):
                calls.append(
                    {
                        "session_id": session["id"],
                        "bfcl_user_turn_index": user_turn_index,
                        "bfcl_step_index": step_index,
                        "bfcl_input_tokens": input_tokens,
                        "bfcl_output_tokens": output_tokens,
                        "bfcl_latency_s": latency_s,
                    }
                )
    return calls


def bucket_name(context_tokens: int) -> str:
    if context_tokens <= 40_000:
        return "32k-40k"
    if context_tokens <= 64_000:
        return "40k-64k"
    return "64k+"


def acceptance_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    emitted = [
        int(length)
        for row in rows
        for length in row.get("accept_lengths", [])
    ]
    draft_matches = [
        int(length)
        for row in rows
        for length in row.get("draft_match_lengths", [])
    ]
    contexts = [int(row["context_tokens_at_turn_start"]) for row in rows]
    request_acceptance = [float(row["average_accept_length"]) for row in rows]
    request_matches = [float(row["average_draft_matches"]) for row in rows]
    response_texts = [visible_response_text(row) for row in rows]
    return {
        "qualifying_model_calls": len(rows),
        "distinct_sessions": len({row["session_id"] for row in rows}),
        "distinct_bfcl_user_turns": len(
            {(row["session_id"], row["bfcl_user_turn_index"]) for row in rows}
        ),
        "speculative_verification_steps": len(emitted),
        "context_tokens": {
            "min": min(contexts) if contexts else None,
            "p50": quantile(contexts, 0.5),
            "p90": quantile(contexts, 0.9),
            "max": max(contexts) if contexts else None,
        },
        "emitted_tokens_per_verification_step": {
            "micro_mean": statistics.fmean(emitted) if emitted else None,
            "macro_mean_per_model_call": (
                statistics.fmean(request_acceptance) if request_acceptance else None
            ),
            "p50": quantile(emitted, 0.5),
            "p90": quantile(emitted, 0.9),
            "histogram": dict(sorted(Counter(emitted).items())),
        },
        "accepted_draft_tokens_per_verification_step": {
            "micro_mean": statistics.fmean(draft_matches) if draft_matches else None,
            "macro_mean_per_model_call": (
                statistics.fmean(request_matches) if request_matches else None
            ),
            "p50": quantile(draft_matches, 0.5),
            "p90": quantile(draft_matches, 0.9),
            "histogram": dict(sorted(Counter(draft_matches).items())),
        },
        "output_tokens": sum(int(row["output_tokens"]) for row in rows),
        "finish_reason_counts": dict(
            sorted(Counter(str(row["finish_reason"]) for row in rows).items())
        ),
        "output_structure": {
            "responses_with_visible_tool_reason": sum(
                "</tool_reason>" in text for text in response_texts
            ),
            "responses_with_tool_call": sum(
                "<tool_call>" in text for text in response_texts
            ),
            "responses_with_think": sum(
                "<think>" in text or "</think>" in text for text in response_texts
            ),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        action="append",
        nargs=3,
        required=True,
        metavar=("METRICS_JSONL", "RUN_ID", "BFCL_RESULT_JSONL"),
        help="May be repeated; metrics are filtered to the exact run ID.",
    )
    parser.add_argument("--threshold", type=int, default=32_000)
    parser.add_argument("--output-prefix", type=Path, required=True)
    args = parser.parse_args()

    mapped: list[dict[str, Any]] = []
    source_checks: list[dict[str, Any]] = []
    for metric_name, run_id, result_name in args.source:
        metric_path = Path(metric_name)
        result_path = Path(result_name)
        metrics = [
            row for row in load_jsonl(metric_path) if str(row.get("run_id")) == run_id
        ]
        metrics.sort(key=lambda row: int(row["request_index_within_run"]))
        result_rows = load_jsonl(result_path)
        failed_result_ids = [
            str(row.get("id", "unknown"))
            for row in result_rows
            if "input_token_count" not in row
        ]
        bfcl_calls = flatten_bfcl_calls(result_rows)
        if len(metrics) < len(bfcl_calls):
            raise ValueError(
                f"request count mismatch for {run_id}: metrics={len(metrics)}, "
                f"BFCL calls={len(bfcl_calls)}"
            )
        # A failed BFCL session can still emit several successful request metrics
        # before the request that exceeds the context limit.  Its result row has
        # no per-call token counts, so align completed BFCL calls as an ordered
        # subsequence instead of assuming that unmatched metrics are trailing.
        metric_index = 0
        matched_metrics: list[dict[str, Any]] = []
        ignored_metric_indices: list[int] = []
        for bfcl_call in bfcl_calls:
            expected = (
                int(bfcl_call["bfcl_input_tokens"]),
                int(bfcl_call["bfcl_output_tokens"]),
            )
            while metric_index < len(metrics):
                metric = metrics[metric_index]
                observed = (
                    int(metric["context_tokens_at_turn_start"]),
                    int(metric["output_tokens"]),
                )
                if observed == expected:
                    matched_metrics.append(metric)
                    metric_index += 1
                    break
                ignored_metric_indices.append(metric_index)
                metric_index += 1
            else:
                raise ValueError(
                    f"could not align BFCL call for {run_id}: expected={expected}, "
                    f"matched={len(matched_metrics)}/{len(bfcl_calls)}"
                )
        ignored_metric_indices.extend(range(metric_index, len(metrics)))
        for metric, bfcl_call in zip(matched_metrics, bfcl_calls, strict=True):
            mapped.append({**metric, **bfcl_call})

        # Recover successful requests made inside sessions whose final BFCL row
        # is an error (typically the next request exceeded the context limit).
        # Such requests remain valid acceptance observations.  Ordered,
        # contiguous unmatched blocks correspond to ordered failed result rows.
        ignored_blocks: list[list[int]] = []
        for index in ignored_metric_indices:
            if not ignored_blocks or index != ignored_blocks[-1][-1] + 1:
                ignored_blocks.append([index])
            else:
                ignored_blocks[-1].append(index)
        recovered_failed_calls = 0
        if failed_result_ids and len(ignored_blocks) == len(failed_result_ids):
            for failed_id, block in zip(
                failed_result_ids, ignored_blocks, strict=True
            ):
                for failed_step_index, index in enumerate(block):
                    metric = metrics[index]
                    mapped.append(
                        {
                            **metric,
                            "session_id": failed_id,
                            "bfcl_user_turn_index": None,
                            "bfcl_step_index": failed_step_index,
                            "bfcl_input_tokens": metric[
                                "context_tokens_at_turn_start"
                            ],
                            "bfcl_output_tokens": metric["output_tokens"],
                            "bfcl_latency_s": metric.get("request_wall_time_s"),
                            "mapped_from_failed_session_metrics": True,
                        }
                    )
                    recovered_failed_calls += 1
            ignored_metric_indices = []
        source_checks.append(
            {
                "run_id": run_id,
                "metrics_path": str(metric_path.resolve()),
                "bfcl_result_path": str(result_path.resolve()),
                "model_calls": len(matched_metrics) + recovered_failed_calls,
                "recovered_failed_session_metric_calls": recovered_failed_calls,
                "ignored_unmapped_metric_calls": len(ignored_metric_indices),
                "ignored_unmapped_request_indices": [
                    int(metrics[index]["request_index_within_run"])
                    for index in ignored_metric_indices
                ],
                "ignored_failed_bfcl_result_ids": failed_result_ids,
                "context_token_mismatches": 0,
            }
        )

    qualifying = [
        row
        for row in mapped
        if int(row["context_tokens_at_turn_start"]) > args.threshold
    ]
    qualifying.sort(
        key=lambda row: (
            int(row["context_tokens_at_turn_start"]),
            row["session_id"],
            row["bfcl_user_turn_index"],
            row["bfcl_step_index"],
        )
    )

    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in qualifying:
        buckets[bucket_name(int(row["context_tokens_at_turn_start"]))].append(row)

    session_ids = {row["session_id"] for row in mapped}
    summary = {
        "filter": {
            "field": "context_tokens_at_turn_start",
            "operator": ">",
            "threshold_tokens": args.threshold,
            "note": "Calls at or below the threshold are excluded from acceptance statistics.",
        },
        "configuration": {
            "temperature_values": sorted({row.get("temperature") for row in mapped}),
            "decode_modes": sorted({str(row.get("decode_mode")) for row in mapped}),
            "context_limits": sorted({int(row["context_limit"]) for row in mapped}),
            "rope_scaling": sorted({str(row.get("rope_scaling")) for row in mapped}),
        },
        "coverage": {
            "candidate_sessions": len(session_ids),
            "all_model_calls": len(mapped),
            "excluded_model_calls_at_or_below_threshold": len(mapped) - len(qualifying),
        },
        "source_checks": source_checks,
        "overall": acceptance_summary(qualifying),
        "by_context_bucket": {
            name: acceptance_summary(buckets.get(name, []))
            for name in ("32k-40k", "40k-64k", "64k+")
        },
    }

    prefix = args.output_prefix
    prefix.parent.mkdir(parents=True, exist_ok=True)
    summary_path = prefix.with_name(prefix.name + "_summary.json")
    calls_path = prefix.with_name(prefix.name + "_calls.csv")
    turns_path = prefix.with_name(prefix.name + "_user_turns.csv")
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")

    call_fields = [
        "run_id",
        "session_id",
        "bfcl_user_turn_index",
        "bfcl_step_index",
        "request_index_within_run",
        "context_tokens_at_turn_start",
        "output_tokens",
        "response_text",
        "speculative_steps",
        "average_accept_length",
        "average_draft_matches",
        "accept_lengths",
        "draft_match_lengths",
        "finish_reason",
        "temperature",
        "request_wall_time_s",
        "generation_wall_time_s",
        "prefill_wall_time_s",
        "decode_wall_time_s",
    ]
    with calls_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=call_fields, extrasaction="ignore")
        writer.writeheader()
        for row in qualifying:
            export = dict(row)
            export["accept_lengths"] = json.dumps(row.get("accept_lengths", []))
            export["draft_match_lengths"] = json.dumps(
                row.get("draft_match_lengths", [])
            )
            export["response_text"] = visible_response_text(row)
            writer.writerow(export)

    grouped_turns: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in qualifying:
        grouped_turns[(row["session_id"], row["bfcl_user_turn_index"])].append(row)
    turn_fields = [
        "session_id",
        "bfcl_user_turn_index",
        "qualifying_model_calls",
        "min_context_tokens",
        "max_context_tokens",
        "speculative_verification_steps",
        "emitted_acceptance_micro_mean",
        "draft_matches_micro_mean",
        "output_tokens",
    ]
    with turns_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=turn_fields)
        writer.writeheader()
        for (session_id, user_turn), rows in sorted(grouped_turns.items()):
            sub = acceptance_summary(rows)
            writer.writerow(
                {
                    "session_id": session_id,
                    "bfcl_user_turn_index": user_turn,
                    "qualifying_model_calls": len(rows),
                    "min_context_tokens": min(
                        int(row["context_tokens_at_turn_start"]) for row in rows
                    ),
                    "max_context_tokens": max(
                        int(row["context_tokens_at_turn_start"]) for row in rows
                    ),
                    "speculative_verification_steps": sub[
                        "speculative_verification_steps"
                    ],
                    "emitted_acceptance_micro_mean": sub[
                        "emitted_tokens_per_verification_step"
                    ]["micro_mean"],
                    "draft_matches_micro_mean": sub[
                        "accepted_draft_tokens_per_verification_step"
                    ]["micro_mean"],
                    "output_tokens": sub["output_tokens"],
                }
            )

    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"wrote {summary_path}")
    print(f"wrote {calls_path}")
    print(f"wrote {turns_path}")


if __name__ == "__main__":
    main()
