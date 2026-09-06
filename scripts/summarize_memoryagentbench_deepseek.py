#!/usr/bin/env python3
"""Summarize DFlash and DSpark MemoryAgentBench acceptance runs."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


CATEGORIES = ("factconsolidation_64k", "eventqa_64k", "detectiveqa_free")
BACKENDS = ("dflash", "dspark")


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def summarize_pairs(responses: list[dict], metrics: list[dict]) -> dict[str, Any]:
    if len(responses) != len(metrics):
        raise ValueError(f"response/metric count mismatch: {len(responses)} != {len(metrics)}")
    raw = [int(x) for row in metrics for x in row["accept_lengths"]]
    matches = [int(x) for row in metrics for x in row["draft_match_lengths"]]
    if len(raw) != len(matches):
        raise ValueError("accept and draft-match step counts differ")
    by_context: dict[str, list[int]] = defaultdict(list)
    for response, metric in zip(responses, metrics, strict=True):
        by_context[str(response["context_index"])].extend(
            int(x) for x in metric["draft_match_lengths"]
        )
    return {
        "requests_completed": len(responses),
        "verification_steps": len(matches),
        "output_tokens": sum(int(row["output_tokens"]) for row in responses),
        "prompt_tokens_min": min((int(row["input_tokens"]) for row in responses), default=0),
        "prompt_tokens_max": max((int(row["input_tokens"]) for row in responses), default=0),
        "mean_anchor_inclusive_accept_length": sum(raw) / len(raw) if raw else 0.0,
        "mean_accepted_draft_tokens": sum(matches) / len(matches) if matches else 0.0,
        "proposal_acceptance_rate": sum(matches) / (len(matches) * 7) if matches else 0.0,
        "anchor_inclusive_histogram_counts": dict(sorted(Counter(raw).items())),
        "accepted_draft_histogram_counts": dict(sorted(Counter(matches).items())),
        "per_context_mean_accepted_draft_tokens": {
            key: sum(values) / len(values)
            for key, values in sorted(by_context.items(), key=lambda item: int(item[0]))
        },
        "requests_hitting_max_new_tokens": sum(
            row.get("finish_reason") == "length" for row in responses
        ),
        "request_wall_time_s": sum(float(row["request_wall_time"]) for row in responses),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary: dict[str, Any] = {
        "target_model": "/share/dai-sys/wanghanzhen/models/Qwen/Qwen3-4B",
        "temperature": 0.0,
        "max_new_tokens": 512,
        "block_size": 7,
        "rope_scaling": "yarn",
        "rope_factor": 4.0,
        "acceptance_semantics": {
            "anchor_inclusive": "tokens emitted per verification step (normally accepted draft tokens + 1 target token)",
            "accepted_draft_tokens": "number of the seven proposed draft tokens accepted per verification step",
        },
        "models": {},
    }
    for backend in BACKENDS:
        model_summary: dict[str, Any] = {"categories": {}}
        all_responses: list[dict] = []
        all_metrics: list[dict] = []
        for category in CATEGORIES:
            job = args.run_root / backend / category
            responses = load_jsonl(job / "responses.jsonl")
            metrics = load_jsonl(job / "metrics.jsonl")
            model_summary["categories"][category] = summarize_pairs(responses, metrics)
            all_responses.extend(responses)
            all_metrics.extend(metrics)
        model_summary["overall"] = summarize_pairs(all_responses, all_metrics)
        summary["models"][backend] = model_summary

    output = args.run_root / "combined_summary.json"
    output.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    lines = [
        "# MemoryAgentBench DeepSeek acceptance summary",
        "",
        "Temperature 0, max_new_tokens 512, seven draft proposals per verification step.",
        "",
        "| Model | Category | Requests | Output tokens | Verify steps | Raw mean | Accepted draft mean | Proposal acceptance |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for backend in BACKENDS:
        for category in (*CATEGORIES, "overall"):
            metrics = (
                summary["models"][backend]["overall"]
                if category == "overall"
                else summary["models"][backend]["categories"][category]
            )
            lines.append(
                f"| {backend} | {category} | {metrics['requests_completed']} | "
                f"{metrics['output_tokens']} | {metrics['verification_steps']} | "
                f"{metrics['mean_anchor_inclusive_accept_length']:.6f} | "
                f"{metrics['mean_accepted_draft_tokens']:.6f} | "
                f"{metrics['proposal_acceptance_rate']:.4%} |"
            )
    (args.run_root / "combined_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
