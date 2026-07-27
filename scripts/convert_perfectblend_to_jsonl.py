#!/usr/bin/env python3
"""Convert local open-perfectblend parquet shards to FlashMTP training JSONL.

Input (ShareGPT-style, as in mlabonne/open-perfectblend):
  conversations: [{"from": "human"|"gpt"|..., "value": str}, ...]
  source: str

Output (compatible with train_flashmtp.py / regenerate_train_data.py):
  {"id": int, "conversations": [{"role": "user"|"assistant"|"system", "content": str}], "source": str}

Usage:
  python scripts/convert_perfectblend_to_jsonl.py \\
    --input-dir /data/wanghanzhen/datasets/open_perfectblend/data \\
    --output-file /data/wanghanzhen/datasets/open_perfectblend/open_perfectblend_train.jsonl
"""

from __future__ import annotations

import argparse
import glob
import json
import os
from typing import Any

import pyarrow.parquet as pq
from tqdm import tqdm

ROLE_MAPPING = {
    "human": "user",
    "gpt": "assistant",
    "chatgpt": "assistant",
    "bing": "assistant",
    "bard": "assistant",
    "system": "system",
}

VALID_ROLES = {"system", "user", "assistant"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-dir",
        type=str,
        default="/data/wanghanzhen/datasets/open_perfectblend/data",
        help="Directory containing train-*.parquet shards.",
    )
    parser.add_argument(
        "--output-file",
        type=str,
        default="/data/wanghanzhen/datasets/open_perfectblend/open_perfectblend_train.jsonl",
        help="Output JSONL path.",
    )
    parser.add_argument(
        "--stats-file",
        type=str,
        default=None,
        help="Optional JSON stats path (default: <output>.stats.json).",
    )
    return parser.parse_args()


def iter_parquet_rows(parquet_path: str):
    pf = pq.ParquetFile(parquet_path)
    for batch in pf.iter_batches():
        for row in batch.to_pylist():
            yield row


def normalize_sharegpt_conversations(raw: Any) -> list[dict[str, str]] | None:
    if not isinstance(raw, list) or not raw:
        return None

    convs: list[dict[str, str]] = []
    for message in raw:
        if not isinstance(message, dict):
            continue
        role_key = str(message.get("from", "")).strip().lower()
        if role_key not in ROLE_MAPPING:
            continue
        role = ROLE_MAPPING[role_key]
        content = message.get("value", "")
        if not isinstance(content, str):
            content = json.dumps(content, ensure_ascii=False)
        if role == "system" and not content.strip():
            continue
        if not content.strip():
            continue
        convs.append({"role": role, "content": content})

    if not convs:
        return None

    start = 0
    if convs[0]["role"] == "system":
        start = 1
    if start >= len(convs) or convs[start]["role"] != "user":
        return None
    if not any(x["role"] == "assistant" for x in convs):
        return None

    for role in convs:
        if role["role"] not in VALID_ROLES:
            return None
    return convs


def main() -> None:
    args = parse_args()
    parquet_files = sorted(glob.glob(os.path.join(args.input_dir, "*.parquet")))
    if not parquet_files:
        raise FileNotFoundError(f"No parquet files under {args.input_dir}")

    os.makedirs(os.path.dirname(os.path.abspath(args.output_file)), exist_ok=True)
    stats_file = args.stats_file or args.output_file.replace(".jsonl", ".stats.json")

    written = 0
    skipped = 0
    next_id = 0

    with open(args.output_file, "w", encoding="utf-8") as out_f:
        for parquet_path in parquet_files:
            for row in tqdm(
                iter_parquet_rows(parquet_path),
                desc=os.path.basename(parquet_path),
            ):
                convs = normalize_sharegpt_conversations(row.get("conversations"))
                if convs is None:
                    skipped += 1
                    continue
                source = row.get("source")
                if not isinstance(source, str):
                    source = str(source) if source is not None else ""
                record = {
                    "id": next_id,
                    "conversations": convs,
                    "source": source,
                }
                out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
                next_id += 1
                written += 1

    stats = {
        "input_dir": args.input_dir,
        "parquet_files": parquet_files,
        "output_file": args.output_file,
        "written": written,
        "skipped": skipped,
    }
    with open(stats_file, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    print(f"written: {written}")
    print(f"skipped: {skipped}")
    print(f"output: {args.output_file}")
    print(f"stats: {stats_file}")


if __name__ == "__main__":
    main()
