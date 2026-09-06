#!/usr/bin/env python3
"""Send a fixed raw-prompt JSONL corpus to an OpenAI Completions endpoint."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import requests


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--server-url", required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--max-samples", type=int, default=50)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--timeout", type=float, default=1800.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    base_url = args.server_url.rstrip("/")
    rows = []
    with args.dataset.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    rows = rows[: args.max_samples]
    if not rows:
        raise ValueError("dataset contains no rows")

    session = requests.Session()
    health = session.get(f"{base_url}/health", timeout=30)
    health.raise_for_status()
    config = session.post(
        f"{base_url}/admin/config",
        json={
            "tags": {
                "run_id": args.run_id,
                "harness": "agentlongbench_raw_prompt",
                "task": "48k_96k_acceptance",
            }
        },
        timeout=30,
    )
    config.raise_for_status()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as output_handle:
        for index, row in enumerate(rows):
            started = time.perf_counter()
            response = session.post(
                f"{base_url}/v1/completions",
                json={
                    "model": args.model,
                    "prompt": row["raw_prompt"],
                    "max_tokens": args.max_tokens,
                    "temperature": 0,
                },
                timeout=args.timeout,
            )
            response.raise_for_status()
            body = response.json()
            actual_prompt_tokens = int(body["usage"]["prompt_tokens"])
            expected_prompt_tokens = int(row["prompt_tokens"])
            if actual_prompt_tokens != expected_prompt_tokens:
                raise ValueError(
                    f"row {index}: server counted {actual_prompt_tokens} prompt tokens, "
                    f"manifest counted {expected_prompt_tokens}"
                )
            result = {
                "selection_index": row.get("selection_index", index),
                "id": row.get("id"),
                "source_path": row.get("source_path"),
                "question_type": row.get("question_type"),
                "prompt_tokens": actual_prompt_tokens,
                "completion_tokens": int(body["usage"]["completion_tokens"]),
                "finish_reason": body["choices"][0].get("finish_reason"),
                "response_text": body["choices"][0].get("text", ""),
                "request_wall_time_s": time.perf_counter() - started,
            }
            output_handle.write(json.dumps(result, ensure_ascii=False) + "\n")
            output_handle.flush()
            print(
                f"[{index + 1}/{len(rows)}] prompt={actual_prompt_tokens} "
                f"output={result['completion_tokens']} "
                f"wall={result['request_wall_time_s']:.2f}s",
                flush=True,
            )


if __name__ == "__main__":
    main()
