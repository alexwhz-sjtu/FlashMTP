#!/usr/bin/env python3
"""Send the fixed MemoryAgentBench acceptance prompts to a local server."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import requests
from transformers import AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.memoryagentbench_acceptance import DATA_FILES, load_requests


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--category", choices=tuple(DATA_FILES), required=True)
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("/share/dai-sys/wanghanzhen/datasets/MemoryAgentBench/data"),
    )
    parser.add_argument("--server-url", required=True)
    parser.add_argument("--tokenizer-path", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output-jsonl", type=Path, required=True)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--request-limit", type=int)
    parser.add_argument("--timeout", type=float, default=3600.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    requests_to_send = load_requests(args.data_root, args.category)
    if args.request_limit is not None:
        requests_to_send = requests_to_send[: args.request_limit]

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)
    session = requests.Session()
    base_url = args.server_url.rstrip("/")
    health = session.get(f"{base_url}/health", timeout=60)
    health.raise_for_status()
    backend = health.json()["backend"]
    run_id = f"memoryagentbench-{backend}-{args.category}"
    config = session.post(
        f"{base_url}/admin/config",
        json={
            "tags": {
                "run_id": run_id,
                "harness": "memoryagentbench_acceptance",
                "category": args.category,
            }
        },
        timeout=60,
    )
    config.raise_for_status()

    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with args.output_jsonl.open("w", encoding="utf-8") as output_handle:
        for position, request in enumerate(requests_to_send, start=1):
            prompt = tokenizer.apply_chat_template(
                request["messages"],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            expected_input_tokens = len(
                tokenizer.encode(prompt, add_special_tokens=False)
            )
            started = time.perf_counter()
            response = session.post(
                f"{base_url}/v1/completions",
                json={
                    "model": args.model,
                    "prompt": prompt,
                    "max_tokens": args.max_new_tokens,
                    "temperature": args.temperature,
                },
                timeout=args.timeout,
            )
            response.raise_for_status()
            body = response.json()
            input_tokens = int(body["usage"]["prompt_tokens"])
            if input_tokens != expected_input_tokens:
                raise ValueError(
                    f"{request['request_id']}: server={input_tokens} "
                    f"client={expected_input_tokens} prompt tokens"
                )
            row = {key: value for key, value in request.items() if key != "messages"}
            row.update(
                {
                    "status": "completed",
                    "backend": backend,
                    "input_tokens": input_tokens,
                    "output_tokens": int(body["usage"]["completion_tokens"]),
                    "max_new_tokens": args.max_new_tokens,
                    "temperature": args.temperature,
                    "finish_reason": body["choices"][0].get("finish_reason"),
                    "generated_text": body["choices"][0].get("text", ""),
                    "request_wall_time": time.perf_counter() - started,
                }
            )
            output_handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            output_handle.flush()
            print(
                f"[{position}/{len(requests_to_send)}] {request['request_id']} "
                f"input={input_tokens} output={row['output_tokens']} "
                f"wall={row['request_wall_time']:.2f}s",
                flush=True,
            )


if __name__ == "__main__":
    main()
