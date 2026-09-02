#!/usr/bin/env python3
"""Remove exact duplicate answers for the same complete prompt.

The first occurrence is retained. Prompt identity is based on the canonical
JSON representation of every conversation message before the final assistant
turn, rather than on source_id, so duplicate prompts from different sources
are handled as the same prompt.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any


ASSISTANT_ROLES = {
    "assistant",
    "assistant_analysis",
    "assistant_final",
    "assistant_commentary",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--meta", type=Path, default=None)
    return parser.parse_args()


def canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def prompt_and_answer(record: dict[str, Any]) -> tuple[Any, Any]:
    conversations = record.get("conversations")
    if not isinstance(conversations, list) or not conversations:
        raise ValueError("record has no non-empty conversations list")

    answer_index = len(conversations) - 1
    answer_message = conversations[answer_index]
    if not isinstance(answer_message, dict):
        raise ValueError("final conversation message is not an object")
    if answer_message.get("role") not in ASSISTANT_ROLES:
        raise ValueError("final conversation message is not an assistant turn")

    answer = answer_message.get("content")
    if not isinstance(answer, str):
        raise ValueError("final assistant content is not a string")
    return conversations[:answer_index], answer


def digest(value: Any) -> bytes:
    return hashlib.sha256(canonical_json(value)).digest()


def main() -> None:
    args = parse_args()
    input_file = args.input.resolve()
    output_file = args.output.resolve()
    if input_file == output_file:
        raise ValueError("--input and --output must be different files")
    if not input_file.is_file():
        raise FileNotFoundError(input_file)

    meta_file = (
        args.meta.resolve()
        if args.meta is not None
        else Path(f"{output_file}.meta.json")
    )
    output_file.parent.mkdir(parents=True, exist_ok=True)
    meta_file.parent.mkdir(parents=True, exist_ok=True)
    temporary_output = output_file.with_name(f".{output_file.name}.tmp.{os.getpid()}")

    seen: set[tuple[bytes, bytes]] = set()
    input_count = 0
    output_count = 0
    duplicate_count = 0

    try:
        with input_file.open("r", encoding="utf-8") as source, temporary_output.open(
            "w", encoding="utf-8", buffering=1
        ) as destination:
            for line_number, line in enumerate(source, 1):
                if not line.strip():
                    continue
                input_count += 1
                try:
                    record = json.loads(line)
                    prompt, answer = prompt_and_answer(record)
                except (json.JSONDecodeError, TypeError, ValueError) as exc:
                    raise ValueError(f"invalid input at line {line_number}: {exc}") from exc

                key = (digest(prompt), digest(answer))
                if key in seen:
                    duplicate_count += 1
                    continue
                seen.add(key)
                destination.write(json.dumps(record, ensure_ascii=False) + "\n")
                output_count += 1

        os.replace(temporary_output, output_file)
    finally:
        if temporary_output.exists():
            temporary_output.unlink()

    metadata = {
        "input": str(input_file),
        "output": str(output_file),
        "input_records": input_count,
        "output_records": output_count,
        "duplicates_removed": duplicate_count,
        "deduplication_key": "canonical full prompt + exact final assistant content",
        "kept": "first occurrence",
    }
    meta_file.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(metadata, ensure_ascii=False))


if __name__ == "__main__":
    main()
