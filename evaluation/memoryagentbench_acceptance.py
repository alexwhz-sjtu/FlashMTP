"""Measure FlashMTP acceptance lengths on selected MemoryAgentBench prompts.

This runner intentionally skips answer-quality scoring and target-only decoding.  It
builds one independent long-context chat prompt per selected question and records
the FlashMTP verification lengths returned by ``get_last_decode_stats()``.
"""

from __future__ import annotations

import argparse
import gc
import json
import random
import re
import sys
import time
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pyarrow.parquet as pq
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.model_loading import load_flashmtp_benchmark_models


SYSTEM_MESSAGE = (
    "You are a helpful assistant that can read a long context and answer questions "
    "based on it."
)

DATA_FILES = {
    "factconsolidation_64k": "FactConsolidation_64K_Qwen3-00000-of-00001.parquet",
    "eventqa_64k": "EventQA_64K_Qwen3-00000-of-00001.parquet",
    "detectiveqa_free": "DetectiveQA_le120K_Qwen3-00000-of-00001.parquet",
}

SAMPLE_COUNTS = {
    "factconsolidation_64k": 50,
    "eventqa_64k": 10,
    "detectiveqa_free": None,
}


def evenly_spaced_indices(length: int, count: int | None) -> list[int]:
    """Return deterministic, endpoint-inclusive, approximately even indices."""
    if count is None or count >= length:
        return list(range(length))
    if count <= 0:
        return []
    if count == 1:
        return [0]
    indices = [round(i * (length - 1) / (count - 1)) for i in range(count)]
    if len(set(indices)) != count:
        raise RuntimeError(f"Even sampling produced duplicate indices: {indices}")
    return indices


def detective_question_stem(question: str) -> str:
    """Remove the in-prompt worked example and all multiple-choice options."""
    marker = "Now Answer the Question:"
    if marker not in question:
        raise ValueError("DetectiveQA question is missing the expected marker")
    actual = question.split(marker, 1)[1]
    actual = re.split(r"\n\s*A\.\s", actual, maxsplit=1)[0]
    actual = re.sub(r"\s*Output:\s*$", "", actual).strip()
    if not actual:
        raise ValueError("DetectiveQA free-answer conversion produced an empty stem")
    return actual


def build_user_prompt(category: str, context: str, question: str) -> tuple[str, str]:
    if category == "factconsolidation_64k":
        prompt = (
            "The following context is a knowledge pool. Facts have serial numbers, "
            "and a larger serial number means a newer fact. Resolve conflicts using "
            "the newest applicable facts and answer the question from this knowledge "
            "pool.\n\n"
            f"[Knowledge Pool]\n{context}\n\n"
            f"Question: {question}\nAnswer:"
        )
        return prompt, question
    if category == "eventqa_64k":
        prompt = (
            "Read the following book context, then predict the event requested by the "
            "question.\n\n"
            f"[Book Context]\n{context}\n\n"
            f"{question}\n\nAnswer:"
        )
        return prompt, question
    if category == "detectiveqa_free":
        stem = detective_question_stem(question)
        prompt = (
            "Read the following detective story and answer the question using a "
            "free-form response. No answer choices are provided. Explain your answer "
            "based on evidence from the story.\n\n"
            f"[Story]\n{context}\n\n"
            f"Question: {stem}\nAnswer:"
        )
        return prompt, stem
    raise ValueError(f"Unknown category: {category}")


def load_requests(data_root: Path, category: str) -> list[dict]:
    path = data_root / DATA_FILES[category]
    rows = pq.read_table(path).to_pylist()
    per_context = SAMPLE_COUNTS[category]
    requests: list[dict] = []
    for context_index, row in enumerate(rows):
        questions = row["questions"]
        selected = evenly_spaced_indices(len(questions), per_context)
        source = row["metadata"]["source"]
        question_ids = row["metadata"].get("question_ids") or []
        qa_pair_ids = row["metadata"].get("qa_pair_ids") or []
        for question_index in selected:
            prompt, effective_question = build_user_prompt(
                category, row["context"], questions[question_index]
            )
            requests.append(
                {
                    "request_id": f"{category}:c{context_index}:q{question_index}",
                    "category": category,
                    "source": source,
                    "context_index": context_index,
                    "question_index": question_index,
                    "question_id": (
                        question_ids[question_index]
                        if question_index < len(question_ids)
                        else None
                    ),
                    "qa_pair_id": (
                        qa_pair_ids[question_index]
                        if question_index < len(qa_pair_ids)
                        else None
                    ),
                    "question": effective_question,
                    "messages": [
                        {"role": "system", "content": SYSTEM_MESSAGE},
                        {"role": "user", "content": prompt},
                    ],
                }
            )
    return requests


def read_completed_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    completed: set[str] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_number}") from exc
            if row.get("status") == "completed":
                completed.add(str(row["request_id"]))
    return completed


def make_model_args(args: argparse.Namespace) -> SimpleNamespace:
    return SimpleNamespace(
        model_name_or_path=args.model_name_or_path,
        draft_name_or_path=args.draft_name_or_path,
        block_size=args.block_size,
        verify_block=args.verify_block,
        mask_token_id=None,
        local_position=None,
        trust_remote_code=args.trust_remote_code,
        rope_scaling=args.rope_scaling,
        rope_factor=args.rope_factor,
        original_max_position_embeddings=args.original_max_position_embeddings,
    )


def summarize_rows(rows: list[dict], verify_block: int) -> dict:
    completed = [row for row in rows if row.get("status") == "completed"]
    raw = [length for row in completed for length in row["accept_lengths_raw"]]
    proposals = [length - 1 for length in raw]
    by_context: dict[str, list[int]] = defaultdict(list)
    for row in completed:
        by_context[str(row["context_index"])].extend(
            length - 1 for length in row["accept_lengths_raw"]
        )
    return {
        "requests_completed": len(completed),
        "verification_steps": len(raw),
        "output_tokens": sum(int(row["output_tokens"]) for row in completed),
        "prompt_tokens_min": min((row["input_tokens"] for row in completed), default=0),
        "prompt_tokens_max": max((row["input_tokens"] for row in completed), default=0),
        "mean_anchor_inclusive_accept_length": float(np.mean(raw)) if raw else 0.0,
        "mean_accepted_draft_tokens": float(np.mean(proposals)) if proposals else 0.0,
        "proposal_acceptance_rate": (
            float(sum(proposals) / (len(proposals) * (verify_block - 1)))
            if proposals and verify_block > 1
            else 0.0
        ),
        "anchor_inclusive_histogram_counts": {
            str(length): raw.count(length) for length in range(1, verify_block + 1)
        },
        "per_context_mean_accepted_draft_tokens": {
            context_id: float(np.mean(lengths)) if lengths else 0.0
            for context_id, lengths in sorted(by_context.items(), key=lambda item: int(item[0]))
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--category", choices=tuple(DATA_FILES), required=True)
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("/share/dai-sys/wanghanzhen/datasets/MemoryAgentBench/data"),
    )
    parser.add_argument("--model-name-or-path", required=True)
    parser.add_argument("--draft-name-or-path", required=True)
    parser.add_argument("--output-jsonl", type=Path, required=True)
    parser.add_argument("--summary-json", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--block-size", type=int, default=8)
    parser.add_argument("--verify-block", type=int, default=8)
    parser.add_argument("--rope-scaling", choices=("none", "yarn"), default="yarn")
    parser.add_argument("--rope-factor", type=float, default=4.0)
    parser.add_argument("--original-max-position-embeddings", type=int, default=40960)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--request-limit",
        type=int,
        default=None,
        help="Optional prefix limit for smoke tests; omitted for the full benchmark.",
    )
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--compile-serial-head", action="store_true")
    parser.add_argument(
        "--stochastic-verification-mode",
        choices=("match", "rejection"),
        default="match",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    requests = load_requests(args.data_root, args.category)
    if args.request_limit is not None:
        requests = requests[: args.request_limit]
    completed_ids = read_completed_ids(args.output_jsonl)
    pending = [row for row in requests if row["request_id"] not in completed_ids]
    print(
        f"category={args.category} requests={len(requests)} "
        f"completed={len(completed_ids)} pending={len(pending)}"
    )

    device = torch.device(args.device)
    torch.cuda.set_device(device)
    target, draft, tokenizer, draft_summary = load_flashmtp_benchmark_models(
        make_model_args(args), device
    )
    draft.set_config_block_size(args.block_size)
    if not 1 <= args.verify_block <= draft.max_verify_block_size:
        raise ValueError(
            f"verify_block={args.verify_block} exceeds maximum "
            f"{draft.max_verify_block_size}"
        )

    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    stop_token_ids = sorted(
        {
            int(token_id)
            for token_id in (
                tokenizer.eos_token_id,
                tokenizer.convert_tokens_to_ids("<|im_end|>"),
            )
            if token_id is not None and int(token_id) >= 0
        }
    )
    model_limit = int(target.config.max_position_embeddings)

    with args.output_jsonl.open("a", encoding="utf-8") as output_handle:
        for position, request in enumerate(pending, start=1):
            prompt = tokenizer.apply_chat_template(
                request["messages"],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            input_ids = tokenizer.encode(
                prompt, return_tensors="pt", add_special_tokens=False
            ).to(device)
            input_tokens = int(input_ids.shape[1])
            if input_tokens + args.max_new_tokens > model_limit:
                raise ValueError(
                    f"{request['request_id']} needs {input_tokens + args.max_new_tokens} "
                    f"tokens but configured model limit is {model_limit}"
                )

            started = time.perf_counter()
            output_ids = draft.spec_generate(
                target=target,
                input_ids=input_ids,
                max_new_tokens=args.max_new_tokens,
                stop_token_ids=stop_token_ids,
                temperature=args.temperature,
                verify_block_size=args.verify_block,
                stochastic_verification_mode=args.stochastic_verification_mode,
                compile_serial_head=args.compile_serial_head,
            )
            elapsed = time.perf_counter() - started
            stats = draft.get_last_decode_stats()
            generated_ids = output_ids[0, input_tokens:]
            generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
            raw_lengths = [int(value) for value in stats.get("accept_lengths", [])]
            proposal_lengths = [value - 1 for value in raw_lengths]

            result = {
                key: value for key, value in request.items() if key != "messages"
            }
            result.update(
                {
                    "status": "completed",
                    "input_tokens": input_tokens,
                    "output_tokens": int(generated_ids.numel()),
                    "max_new_tokens": args.max_new_tokens,
                    "temperature": args.temperature,
                    "block_size": args.block_size,
                    "verify_block": args.verify_block,
                    "verification_mode": args.stochastic_verification_mode,
                    "accept_lengths_raw": raw_lengths,
                    "accepted_draft_tokens": proposal_lengths,
                    "mean_anchor_inclusive_accept_length": (
                        float(np.mean(raw_lengths)) if raw_lengths else 0.0
                    ),
                    "mean_accepted_draft_tokens": (
                        float(np.mean(proposal_lengths)) if proposal_lengths else 0.0
                    ),
                    "proposal_acceptance_rate": (
                        float(
                            sum(proposal_lengths)
                            / (len(proposal_lengths) * (args.verify_block - 1))
                        )
                        if proposal_lengths and args.verify_block > 1
                        else 0.0
                    ),
                    "verification_steps": len(raw_lengths),
                    "decode_wall_time": float(stats.get("decode_wall_time", 0.0)),
                    "request_wall_time": elapsed,
                    "generated_text": generated_text,
                }
            )
            output_handle.write(json.dumps(result, ensure_ascii=False) + "\n")
            output_handle.flush()
            print(
                f"[{position}/{len(pending)}] {request['request_id']} "
                f"input={input_tokens} output={result['output_tokens']} "
                f"steps={len(raw_lengths)} mean_accept={result['mean_accepted_draft_tokens']:.4f} "
                f"wall={elapsed:.2f}s"
            )
            del input_ids, output_ids, generated_ids
            gc.collect()

    rows = []
    with args.output_jsonl.open("r", encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    summary = {
        "category": args.category,
        "model_name_or_path": args.model_name_or_path,
        "draft_name_or_path": args.draft_name_or_path,
        "temperature": args.temperature,
        "max_new_tokens": args.max_new_tokens,
        "block_size": args.block_size,
        "verify_block": args.verify_block,
        "rope_scaling": args.rope_scaling,
        "rope_factor": args.rope_factor,
        "model_context_limit": model_limit,
        "draft_config": draft_summary,
        "sampled_question_indices": {
            str(context_id): [
                row["question_index"]
                for row in requests
                if row["context_index"] == context_id
            ]
            for context_id in sorted({row["context_index"] for row in requests})
        },
        "metrics": summarize_rows(rows, args.verify_block),
    }
    args.summary_json.parent.mkdir(parents=True, exist_ok=True)
    args.summary_json.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary["metrics"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
