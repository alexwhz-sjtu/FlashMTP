#!/usr/bin/env python3
"""Build a deterministic long-context AgentLongBench acceptance test set.

The output keeps the exact Qwen-rendered ``raw_prompt``.  Benchmark runners must
encode that field directly with ``add_special_tokens=False``; wrapping it in a
second chat template would change both the measured length and the model input.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

from transformers import AutoTokenizer


DEFAULT_AGENTLONG_ROOT = Path(
    "/share/dai-sys/wanghanzhen/datasets/AgentLongBench"
)
DEFAULT_TOKENIZER = Path("/share/dai-sys/wanghanzhen/models/Qwen/Qwen3-4B")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--agentlong-root", type=Path, default=DEFAULT_AGENTLONG_ROOT)
    parser.add_argument("--tokenizer", type=Path, default=DEFAULT_TOKENIZER)
    parser.add_argument(
        "--length-dir",
        action="append",
        default=None,
        help="AgentLongBench nominal length directory; repeatable (default: 64k).",
    )
    parser.add_argument("--min-tokens", type=int, default=48 * 1024)
    parser.add_argument("--max-tokens", type=int, default=96 * 1024)
    parser.add_argument("--max-samples", type=int, default=50)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary", type=Path)
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number} is not a JSON object")
            rows.append(row)
    return rows


def evenly_order(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Order one source's rows from across its token-length distribution."""
    rows = sorted(rows, key=lambda row: (row["prompt_tokens"], row["source_index"]))
    if len(rows) < 2:
        return rows
    order = []
    left, right = 0, len(rows) - 1
    while left <= right:
        middle = (left + right) // 2
        order.append(rows[middle])
        if middle != left:
            order.append(rows[left])
        if middle != right:
            order.append(rows[right])
        used = {left, middle, right}
        rows = [row for index, row in enumerate(rows) if index not in used]
        left, right = 0, len(rows) - 1
    # The iterative removal can only duplicate when the source has 2 rows.
    deduped = []
    seen = set()
    for row in order:
        key = (row["source_path"], row["source_index"])
        if key not in seen:
            seen.add(key)
            deduped.append(row)
    return deduped


def select_round_robin(
    candidates: list[dict[str, Any]], max_samples: int
) -> list[dict[str, Any]]:
    by_source: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in candidates:
        by_source[row["source_path"]].append(row)
    queues = {key: evenly_order(value) for key, value in sorted(by_source.items())}
    selected = []
    while len(selected) < max_samples and queues:
        empty = []
        for source, queue in queues.items():
            if queue:
                selected.append(queue.pop(0))
                if len(selected) == max_samples:
                    break
            if not queue:
                empty.append(source)
        for source in empty:
            queues.pop(source, None)
    return selected


def percentile(values: list[int], fraction: float) -> int | None:
    if not values:
        return None
    ordered = sorted(values)
    index = round((len(ordered) - 1) * fraction)
    return ordered[index]


def main() -> None:
    args = parse_args()
    if args.min_tokens < 1 or args.max_tokens < args.min_tokens:
        raise ValueError("invalid token interval")
    if args.max_samples < 1:
        raise ValueError("--max-samples must be positive")

    repo_root = args.agentlong_root.resolve()
    sys.path.insert(0, str(repo_root))
    from eval.common.mapping import infer_context_from_path, require_single_question_type
    from eval.common.question_logic import build_prompt

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    length_dirs = args.length_dir or ["64k"]
    data_files = []
    for length_dir in length_dirs:
        data_files.extend(
            sorted((repo_root / "benchmark").glob(f"*/{length_dir}/*/*.jsonl"))
        )
    if not data_files:
        raise FileNotFoundError(
            f"no AgentLongBench JSONL files found for {length_dirs} under {repo_root}"
        )

    candidates: list[dict[str, Any]] = []
    scanned = 0
    for path in sorted(set(data_files)):
        rows = load_jsonl(path)
        question_type = require_single_question_type(rows)
        _, knowledge_label, _, history_label = infer_context_from_path(path)
        relative_path = str(path.relative_to(repo_root))
        for source_index, sample in enumerate(rows):
            scanned += 1
            messages = build_prompt(
                question_type, sample, history_label, knowledge_label
            )
            raw_prompt = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            prompt_tokens = len(
                tokenizer.encode(raw_prompt, add_special_tokens=False)
            )
            if not args.min_tokens <= prompt_tokens <= args.max_tokens:
                continue
            tool_messages = [m for m in messages if m.get("role") == "tool"]
            candidates.append(
                {
                    "turns": ["raw_prompt is authoritative; do not chat-wrap this field"],
                    "raw_prompt": raw_prompt,
                    "prompt_tokens": prompt_tokens,
                    "source_path": relative_path,
                    "source_index": source_index,
                    "id": sample.get("id"),
                    "sample_id": sample.get("sample_id"),
                    "question_type": question_type,
                    "knowledge_type": knowledge_label,
                    "history_type": history_label,
                    "message_count": len(messages),
                    "tool_message_count": len(tool_messages),
                    "tool_result_chars": sum(
                        len(str(message.get("content") or ""))
                        for message in tool_messages
                    ),
                }
            )

    selected = select_round_robin(candidates, min(args.max_samples, len(candidates)))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for selection_index, row in enumerate(selected):
            output = dict(row)
            output["selection_index"] = selection_index
            handle.write(json.dumps(output, ensure_ascii=False) + "\n")

    lengths = [int(row["prompt_tokens"]) for row in selected]
    summary = {
        "tokenizer": str(args.tokenizer),
        "length_dirs": length_dirs,
        "min_tokens_inclusive": args.min_tokens,
        "max_tokens_inclusive": args.max_tokens,
        "scanned_samples": scanned,
        "eligible_samples": len(candidates),
        "selected_samples": len(selected),
        "selected_min_tokens": min(lengths) if lengths else None,
        "selected_max_tokens": max(lengths) if lengths else None,
        "selected_mean_tokens": sum(lengths) / len(lengths) if lengths else None,
        "selected_p50_tokens": percentile(lengths, 0.50),
        "selected_p90_tokens": percentile(lengths, 0.90),
        "selected_by_setting": dict(
            sorted(
                {
                    setting: sum(
                        f"{row['knowledge_type']}:{row['history_type']}" == setting
                        for row in selected
                    )
                    for setting in {
                        f"{row['knowledge_type']}:{row['history_type']}"
                        for row in selected
                    }
                }.items()
            )
        ),
    }
    summary_path = args.summary or args.output.with_suffix(".summary.json")
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
