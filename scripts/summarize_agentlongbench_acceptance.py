#!/usr/bin/env python3
"""Summarize the three Qwen3-4B AgentLongBench acceptance runs."""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def percentile(values: list[int], fraction: float) -> int | None:
    if not values:
        return None
    ordered = sorted(values)
    return ordered[round((len(ordered) - 1) * fraction)]


def distribution(lengths: list[int]) -> dict[str, Any]:
    counts = Counter(lengths)
    total = len(lengths)
    return {
        "verification_steps": total,
        "mean_acceptance_length": sum(lengths) / total if total else None,
        "p50_acceptance_length": percentile(lengths, 0.50),
        "p90_acceptance_length": percentile(lengths, 0.90),
        "p95_acceptance_length": percentile(lengths, 0.95),
        "p99_acceptance_length": percentile(lengths, 0.99),
        "max_acceptance_length": max(lengths) if lengths else None,
        "acceptance_length_histogram_counts": {
            str(key): value for key, value in sorted(counts.items())
        },
        "acceptance_length_histogram_rates": {
            str(key): value / total for key, value in sorted(counts.items())
        },
    }


def parse_flashmtp(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    pattern = re.compile(
        r"\[Sample (\d+) \| Turn 0\] Acceptance lengths "
        r"\(position:length\):\s*(.*?)\n"
        r"\[Sample \1 \| Turn 0\] Average acceptance length:",
        re.DOTALL,
    )
    matches = pattern.findall(text)
    sample_ids = [int(sample_id) for sample_id, _ in matches]
    if sample_ids != list(range(50)):
        raise ValueError(f"FlashMTP log does not contain exactly samples 0..49: {sample_ids}")
    lengths = []
    for _, body in matches:
        lengths.extend(int(value) for value in re.findall(r"\b\d+:(\d+)\b", body))
    result = {"samples": len(matches), **distribution(lengths)}
    result["source"] = str(path)
    return result


def parse_dflash(path: Path) -> dict[str, Any]:
    rows = load_jsonl(path)
    lengths = [int(value) for row in rows for value in row["accept_lengths"]]
    draft_matches = [
        int(value) for row in rows for value in row.get("draft_match_lengths", [])
    ]
    result = {
        "samples": len(rows),
        "output_tokens": sum(int(row["output_tokens"]) for row in rows),
        "mean_draft_matches": (
            sum(draft_matches) / len(draft_matches) if draft_matches else None
        ),
        **distribution(lengths),
    }
    result["source"] = str(path)
    return result


def parse_dspark(path: Path) -> dict[str, Any]:
    rows = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(rows, list) or len(rows) != 1:
        raise ValueError(f"expected one DSpark summary row in {path}")
    row = rows[0]
    return {
        "samples": int(row["num_samples"]),
        "verification_steps": None,
        "mean_acceptance_length": float(row["acceptance_length"]),
        "p50_acceptance_length": None,
        "p90_acceptance_length": None,
        "p95_acceptance_length": None,
        "p99_acceptance_length": None,
        "max_acceptance_length": None,
        "acceptance_length_histogram_counts": None,
        "acceptance_length_histogram_rates": None,
        "draft_tokens_per_proposal": float(row["draft_tokens_per_proposal"]),
        "verify_rate": float(row["verify_rate"]),
        "accept_rates_by_position": row["accept_rates_by_position"],
        "source": str(path),
    }


def main() -> None:
    args = parse_args()
    manifest = load_jsonl(args.manifest)
    prompt_lengths = [int(row["prompt_tokens"]) for row in manifest]
    models = {
        "flashmtp_v2swa": parse_flashmtp(args.run_dir / "flashmtp_v2swa.log"),
        "dflash_qwen3_4b_deepseek": parse_dflash(
            args.run_dir / "dflash_qwen3_4b_metrics.jsonl"
        ),
        "dspark_qwen3_4b": parse_dspark(args.run_dir / "dspark_qwen3_4b.json"),
    }
    if any(model["samples"] != len(manifest) for model in models.values()):
        raise ValueError("not every model completed the fixed manifest")
    summary = {
        "dataset": "AgentLongBench",
        "selection": {
            "samples": len(manifest),
            "qwen_tokenizer": "/share/dai-sys/wanghanzhen/models/Qwen/Qwen3-4B",
            "requested_interval_inclusive": [48 * 1024, 96 * 1024],
            "actual_min_prompt_tokens": min(prompt_lengths),
            "actual_max_prompt_tokens": max(prompt_lengths),
            "actual_mean_prompt_tokens": sum(prompt_lengths) / len(prompt_lengths),
            "tool_message_count": sum(
                int(row["tool_message_count"]) for row in manifest
            ),
            "tool_result_chars": sum(
                int(row["tool_result_chars"]) for row in manifest
            ),
            "manifest": str(args.manifest),
        },
        "decode": {
            "temperature": 0.0,
            "max_new_tokens": 128,
            "batch_size": 1,
            "acceptance_semantics": (
                "emitted tokens per speculative verification step, normally "
                "accepted draft tokens plus one target correction/bonus token"
            ),
        },
        "models": models,
    }
    baseline = models["dflash_qwen3_4b_deepseek"]["mean_acceptance_length"]
    for model in models.values():
        mean = model["mean_acceptance_length"]
        model["mean_delta_vs_dflash"] = mean - baseline
        model["mean_ratio_vs_dflash"] = mean / baseline

    json_path = args.run_dir / "acceptance_summary.json"
    json_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    csv_path = args.run_dir / "acceptance_summary.csv"
    fields = [
        "model",
        "samples",
        "verification_steps",
        "mean_acceptance_length",
        "p50_acceptance_length",
        "p90_acceptance_length",
        "p95_acceptance_length",
        "max_acceptance_length",
        "mean_delta_vs_dflash",
        "mean_ratio_vs_dflash",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for name, model in models.items():
            writer.writerow({"model": name, **{key: model.get(key) for key in fields[1:]}})

    markdown_path = args.run_dir / "acceptance_summary.md"
    lines = [
        "# AgentLongBench 48K–96K acceptance summary",
        "",
        (
            f"Fixed samples: {len(manifest)}; actual Qwen prompt length: "
            f"{min(prompt_lengths):,}–{max(prompt_lengths):,} tokens "
            f"(mean {sum(prompt_lengths) / len(prompt_lengths):,.2f})."
        ),
        "",
        "| Model | Samples | Verify steps | Mean | P50 | P90 | P95 | Max | vs DFlash |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, model in models.items():
        value = lambda key: "—" if model.get(key) is None else str(model[key])
        lines.append(
            f"| {name} | {model['samples']} | {value('verification_steps')} | "
            f"{model['mean_acceptance_length']:.4f} | {value('p50_acceptance_length')} | "
            f"{value('p90_acceptance_length')} | {value('p95_acceptance_length')} | "
            f"{value('max_acceptance_length')} | {model['mean_ratio_vs_dflash']:.3f}x |"
        )
    lines.extend(
        [
            "",
            "Acceptance length is emitted tokens per verification step. DSpark's "
            "existing evaluator writes aggregate and per-position rates only, so its "
            "step count and percentiles are unavailable from this completed run.",
        ]
    )
    markdown_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
