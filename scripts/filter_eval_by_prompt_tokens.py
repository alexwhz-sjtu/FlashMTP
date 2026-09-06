#!/usr/bin/env python3
"""Filter evaluation records by the exact tokenized benchmark prompt length."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Iterable

from transformers import AutoTokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--format", choices=("lveval", "swe_bench"), required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--min-tokens", type=int, default=20_000)
    parser.add_argument("--max-tokens", type=int, default=40_000)
    parser.add_argument("--batch-size", type=int, default=8)
    return parser.parse_args()


def load_records(path: Path) -> list[dict]:
    if path.suffix == ".jsonl":
        with path.open("r", encoding="utf-8") as stream:
            return [json.loads(line) for line in stream if line.strip()]
    with path.open("r", encoding="utf-8") as stream:
        records = json.load(stream)
    if not isinstance(records, list):
        raise ValueError(f"Expected a JSON list in {path}")
    return records


def benchmark_prompt(record: dict, data_format: str) -> str:
    if data_format == "lveval":
        return f"{record['context']}\nQuestion: {record['input']}"
    return str(record["text"])


def batches(records: list[dict], size: int) -> Iterable[list[dict]]:
    for start in range(0, len(records), size):
        yield records[start : start + size]


def main() -> None:
    args = parse_args()
    if args.min_tokens > args.max_tokens:
        raise ValueError("--min-tokens must not exceed --max-tokens")

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    records = load_records(args.source)
    selected: list[dict] = []
    selected_lengths: list[int] = []

    for batch_index, batch in enumerate(batches(records, args.batch_size), start=1):
        prompts = [benchmark_prompt(record, args.format) for record in batch]
        rendered = [
            tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            for prompt in prompts
        ]
        encoded = tokenizer(
            rendered,
            add_special_tokens=False,
            padding=False,
            truncation=False,
            return_length=True,
        )
        lengths = encoded["length"]
        for record, length in zip(batch, lengths, strict=True):
            length = int(length)
            if args.min_tokens <= length <= args.max_tokens:
                selected.append(record)
                selected_lengths.append(length)
        if batch_index % 25 == 0:
            processed = min(batch_index * args.batch_size, len(records))
            print(f"processed={processed}/{len(records)} selected={len(selected)}", flush=True)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(f".{args.output.name}.tmp.{os.getpid()}")
    try:
        with temporary.open("w", encoding="utf-8") as stream:
            for record in selected:
                stream.write(json.dumps(record, ensure_ascii=False) + "\n")
        temporary.replace(args.output)
    finally:
        if temporary.exists():
            temporary.unlink()

    summary = {
        "source": str(args.source),
        "output": str(args.output),
        "source_records": len(records),
        "selected_records": len(selected),
        "min_selected_tokens": min(selected_lengths) if selected_lengths else None,
        "max_selected_tokens": max(selected_lengths) if selected_lengths else None,
    }
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
