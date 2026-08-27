from __future__ import annotations

import argparse
import json
import os
import random
import re
import statistics
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import requests
import torch
from rich import print
from transformers import AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.utils import load_and_process_dataset

from sglang.srt.environ import envs
from sglang.srt.utils import get_device_sm, kill_process_tree
from sglang.test.test_utils import find_available_port

DATASET_PATH_FILE = Path(__file__).resolve().with_name("dataset_path.json")

INFINITEBENCH_PROMPTS = {
    "passkey": "There is an important info hidden inside a lot of irrelevant text. Find it and memorize it. I will quiz you about the important information.\n\n{context}\n\n{input}\n\nThe pass key is",
    "number_string": "There is an important info hidden inside a lot of irrelevant text. Find it. I will quiz you about the important information there.\n\n{context}\n\n{input}\n\nThe sequence of digits is",
    "kv_retrieval": "Extract the value corresponding to the specified key in the JSON object below.\n\n{context}\n\n{input}",
    "longbook_sum_eng": "Summarize the book below.\n\n{context}\n\nSummary:",
    "longbook_choice_eng": "Read the book and answer the question.\n\n{context}\n\nQuestion: {question}\nA. {OPTION_A}\nB. {OPTION_B}\nC. {OPTION_C}\nD. {OPTION_D}\n\nThe letter of the correct answer is",
    "longbook_qa_eng": "Read the book and answer the question. Be very concise in your answer.\n\n{context}\n\nQuestion: {question}\nAnswer:",
    "longbook_qa_chn": "阅读以下书籍然后回答问题。\n\n{context}\n\n问题：{question}\n答案：",
    "longdialogue_qa_eng": "Below is a dialogue script where one random occurrence of a character name is replaced with \"$$MASK$$\", and you should try to guess who that character is.\n\n{context}\n\nThe name that has been replaced with $$MASK$$ is likely",
    "math_find": "{prefix}\n\n{context}\n\n{input}",
    "math_calc": "Let us calculate the intermediate values of an expression.\n\nExpression: 1 + 3 + 4\nValues: [1, 4, 8]\n\nExpression: 8 - 3 + 2 - 4\nValues: [8, 5, 7, 3]\n\nExpression: {context}\nValues:",
    "code_run": "There is a function called {func} in the following Python code.\n\n{context}\n\nPlease compute the exact value of {func_call}. The value of {func_call} is",
    "code_debug": "Following is a Python code where exactly one of the functions/methods has a deliberate error that makes it crash.\n\n{context}\n\nOptions:\nA. {OPTION_A}\nB. {OPTION_B}\nC. {OPTION_C}\nD. {OPTION_D}\n\nThe correct option is:",
}


def format_longbench_v2_prompt(data: dict) -> str:
    if "context" not in data:
        raise ValueError("Missing 'context' field in LongBench_v2 item")
    if "question" not in data:
        raise ValueError("Missing 'question' field in LongBench_v2 item")
    return f"{data['context']}\n\nQuestion: {data['question']}"


def is_longbench_v2_dataset_path(dataset_path: Path) -> bool:
    """Match LongBench v2 shards: folders named ``longbench_v2`` or ``longbench_v2_*`` (context length in path)."""
    try:
        resolved = dataset_path.resolve()
    except OSError:
        resolved = dataset_path
    for part in resolved.parts:
        pl = part.lower()
        if pl == "longbench_v2" or pl.startswith("longbench_v2_"):
            return True
    return False


def load_longbench_v2_json_records(data: list, dataset_path: Path) -> list[dict]:
    if not isinstance(data, list):
        raise ValueError(f"{dataset_path} must contain a JSON list")
    return [{"turns": [format_longbench_v2_prompt(item)]} for item in data]


def is_multifieldqa_en_mixup_dataset_path(dataset_path: Path) -> bool:
    """LVEval mixup JSON: multifieldqa_en_mixup, lic_mixup, hotpotwikiqa_mixup (``context`` + ``input``)."""
    try:
        resolved = dataset_path.resolve()
    except OSError:
        resolved = dataset_path
    stem = resolved.stem.lower()
    markers = ("multifieldqa_en_mixup", "lic_mixup", "hotpotwikiqa_mixup")
    if any(m in stem for m in markers):
        return True
    for part in resolved.parts:
        pl = part.lower()
        if any(m in pl for m in markers):
            return True
    return False


def format_multifieldqa_en_mixup_prompt(data: dict) -> str:
    """LVEval: long ``context`` + ``input`` (question / instruction)."""
    if "context" not in data:
        raise ValueError("Missing 'context' field in multifieldqa_en_mixup item")
    if "input" not in data:
        raise ValueError("Missing 'input' field in multifieldqa_en_mixup item")
    ctx = data["context"]
    inp = data["input"]
    if not isinstance(ctx, str):
        ctx = str(ctx)
    if not isinstance(inp, str):
        inp = str(inp)
    return f"{ctx}\n\n{inp}"


def should_load_as_multifieldqa_en_mixup(
    data: list, dataset_path: Path, original_dataset_name: str
) -> bool:
    if not data or not isinstance(data[0], dict):
        return False
    row0 = data[0]
    if not isinstance(row0.get("input"), str) or not isinstance(row0.get("context"), str):
        return False
    alias = original_dataset_name.lower()
    aliases = (
        "multifieldqa_en_20k_40k",
        "multifieldqa_en_mixup",
        "multifieldqa_en_mixup_32k_ctx_20k_40k",
        "lic_mixup",
        "lic_mixup_32k_ctx_20k_40k",
        "hotpotwikiqa_mixup",
        "hotpotwikiqa_mixup_16k_ctx_20k_40k",
    )
    if alias in aliases:
        return True
    if Path(original_dataset_name).stem.lower() in aliases:
        return True
    return is_multifieldqa_en_mixup_dataset_path(dataset_path)


def load_multifieldqa_en_mixup_json_records(data: list, dataset_path: Path) -> list[dict]:
    if not isinstance(data, list):
        raise ValueError(f"{dataset_path} must contain a JSON list")
    instances: list[dict] = []
    for index, item in enumerate(data):
        if not isinstance(item, dict):
            raise ValueError(
                f"{dataset_path} index {index}: expected object, got {type(item)}"
            )
        instances.append({"turns": [format_multifieldqa_en_mixup_prompt(item)]})
    return instances


def is_swe_bench_style_json(data: list, dataset_path: Path, original_dataset_name: str) -> bool:
    """SWE-bench Parquet export: list of dicts with string ``text`` (and usually ``instance_id``)."""
    if not data or not isinstance(data[0], dict):
        return False
    row0 = data[0]
    if "text" not in row0 or not isinstance(row0.get("text"), str):
        return False
    stem = dataset_path.stem.lower()
    alias_stem = Path(str(original_dataset_name)).stem.lower()
    explicit_name = stem.startswith("swe_bench") or alias_stem.startswith("swe_bench")
    return explicit_name or "instance_id" in row0


def load_swe_bench_json_instances(data: list, dataset_path: Path) -> list[dict]:
    instances: list[dict] = []
    for index, item in enumerate(data):
        if not isinstance(item, dict):
            raise ValueError(f"{dataset_path} index {index}: expected object, got {type(item)}")
        if "text" not in item or item["text"] is None:
            raise ValueError(f"Missing 'text' in {dataset_path} at index {index}")
        text = item["text"]
        if not isinstance(text, str):
            text = str(text)
        instances.append({"turns": [text]})
    return instances


def infer_infinitebench_task(dataset_name: str, dataset_path: Path) -> str | None:
    candidates = [dataset_name, dataset_path.stem]
    for candidate in candidates:
        if candidate in INFINITEBENCH_PROMPTS:
            return candidate
    return None


def format_infinitebench_prompt(data: dict, task_name: str) -> str:
    template = INFINITEBENCH_PROMPTS[task_name]
    fields = {
        "context": data["context"],
        "input": data.get("input", ""),
        "question": data.get("input", ""),
    }

    options = data.get("options") or []
    for option_index, option_name in enumerate(["OPTION_A", "OPTION_B", "OPTION_C", "OPTION_D"]):
        if option_index < len(options):
            fields[option_name] = options[option_index]

    if task_name == "math_find":
        find_result = re.findall(r"The .+ of", data["input"])
        if not find_result:
            raise ValueError(f"Cannot infer math_find target from input: {data['input']}")
        fields["prefix"] = f"What is {find_result[0].lower()[:-3]} in the following list?"

    if task_name == "code_run":
        find_result = re.findall(r"func_[0-9]+\(-?[0-9]+\)", data["input"])
        if not find_result:
            raise ValueError(f"Cannot infer code_run function call from input: {data['input']}")
        fields["func_call"] = find_result[0]
        fields["func"] = fields["func_call"].split("(")[0]

    return template.format(**fields)


def resolve_dataset_path(dataset_name: str) -> str:
    if not DATASET_PATH_FILE.is_file():
        return dataset_name

    with DATASET_PATH_FILE.open("r", encoding="utf-8") as f:
        dataset_paths = json.load(f)

    if not isinstance(dataset_paths, dict):
        raise ValueError(f"{DATASET_PATH_FILE} must contain a JSON object")

    return dataset_paths.get(dataset_name, dataset_name)


def default_specbench_question_jsonl() -> Path:
    """Repo-root ``Spec-Bench/data/spec_bench/question.jsonl``."""
    return Path(__file__).resolve().parents[2] / "Spec-Bench/data/spec_bench/question.jsonl"


def specbench_dataset_meta(dataset_alias: str) -> tuple[bool, str | None]:
    """``specbench`` → all categories; ``specbench_math`` → filter ``math``."""
    s = dataset_alias.strip().lower()
    if s == "specbench":
        return True, None
    if s.startswith("specbench_"):
        cat = s[len("specbench_") :].strip("_")
        return True, (cat if cat else None)
    return False, None


def parse_specbench_category(dataset_alias: str) -> str | None:
    is_sb, cat = specbench_dataset_meta(dataset_alias)
    return cat if is_sb else None


def expand_specbench_turn_string(turn: str) -> list[str]:
    """Sub-questions in one JSON turn: ``|||`` or comma-separated when each segment ends with ``?``."""
    turn = turn.strip()
    if not turn:
        return []
    if "|||" in turn:
        return [p.strip() for p in turn.split("|||") if p.strip()]
    if ", " in turn:
        raw_parts = [p.strip() for p in turn.split(", ") if p.strip()]
        if len(raw_parts) >= 2 and all(p.endswith("?") for p in raw_parts):
            return raw_parts
    return [turn]


def flatten_specbench_turns(turns: list) -> list[str]:
    out: list[str] = []
    for t in turns:
        if not isinstance(t, str):
            t = str(t)
        out.extend(expand_specbench_turn_string(t))
    return out


def load_specbench_question_jsonl(dataset_path: Path, original_dataset_name: str) -> list[dict]:
    is_sb, category = specbench_dataset_meta(original_dataset_name)
    if not is_sb:
        raise ValueError("internal: load_specbench_question_jsonl requires a specbench dataset name")
    if not dataset_path.is_file():
        raise FileNotFoundError(f"Spec-Bench file not found: {dataset_path}")
    instances: list[dict] = []
    with dataset_path.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if category is not None and str(obj.get("category", "")).lower() != category.lower():
                continue
            turns = obj.get("turns")
            if not isinstance(turns, list) or not turns:
                raise ValueError(f"{dataset_path}:{line_number}: missing non-empty turns")
            flat = flatten_specbench_turns(turns)
            if not flat:
                continue
            instances.append(
                {
                    "turns": flat,
                    "specbench_chain_turns": True,
                    "category": str(obj.get("category", "")).strip() or None,
                }
            )
    if not instances:
        hint = f" for category={category!r}" if category else " (all categories)"
        raise ValueError(f"No Spec-Bench samples{hint} in {dataset_path}")
    return instances


def acceptance_length_tau(acceptance_lengths: list[int], block_size: int) -> float:
    if not acceptance_lengths:
        return float("nan")
    histogram = [
        acceptance_lengths.count(b) / len(acceptance_lengths) for b in range(block_size + 1)
    ]
    return sum(index * frac for index, frac in enumerate(histogram))


def print_specbench_category_acceptance_summary(
    responses: list[dict],
    categories: list[str | None],
    block_size: int,
) -> None:
    from collections import defaultdict

    by_category_steps: dict[str, list[int]] = defaultdict(list)
    by_category_turns: dict[str, list[float]] = defaultdict(list)
    for response, category in zip(responses, categories, strict=True):
        if not category:
            continue
        lengths = response[block_size].acceptance_lengths
        by_category_steps[category].extend(lengths)
        if lengths:
            by_category_turns[category].append(float(np.mean(lengths)))
    if not by_category_steps:
        return

    print("\nSpec-Bench acceptance length by category:")
    print(f"{'Category':<18} {'Turns':>6} {'Avg(step)':>10} {'Avg(turn)':>10}")
    print("-" * 48)
    for category in sorted(
        by_category_steps,
        key=lambda c: -acceptance_length_tau(by_category_steps[c], block_size),
    ):
        steps = by_category_steps[category]
        turns = by_category_turns[category]
        step_avg = acceptance_length_tau(steps, block_size)
        turn_avg = float(np.mean(turns)) if turns else float("nan")
        print(f"{category:<18} {len(turns):>6} {step_avg:>10.2f} {turn_avg:>10.2f}")


def load_benchmark_dataset(dataset_name: str):
    original_dataset_name = dataset_name
    is_specbench, _ = specbench_dataset_meta(original_dataset_name)
    if is_specbench:
        resolved = resolve_dataset_path(original_dataset_name)
        sb_file = Path(resolved)
        if not sb_file.is_file() or sb_file.suffix != ".jsonl":
            sb_file = default_specbench_question_jsonl()
        if not sb_file.is_file():
            raise FileNotFoundError(
                f"Spec-Bench question.jsonl not found (tried {resolved} and {sb_file})."
            )
        return load_specbench_question_jsonl(sb_file, original_dataset_name)

    dataset_name = resolve_dataset_path(original_dataset_name)
    dataset_path = Path(dataset_name)
    if dataset_path.is_file() and dataset_path.suffix == ".json":
        with dataset_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, list):
            raise ValueError(f"{dataset_path} must contain a JSON list")

        if (
            original_dataset_name.lower() == "longbench_v2"
            or is_longbench_v2_dataset_path(dataset_path)
        ):
            return load_longbench_v2_json_records(data, dataset_path)

        if should_load_as_multifieldqa_en_mixup(data, dataset_path, original_dataset_name):
            return load_multifieldqa_en_mixup_json_records(data, dataset_path)

        if is_swe_bench_style_json(data, dataset_path, original_dataset_name):
            return load_swe_bench_json_instances(data, dataset_path)

        raise ValueError(f"Unsupported JSON dataset: {dataset_path}")

    if dataset_path.is_file() and dataset_path.suffix == ".jsonl":
        task_name = infer_infinitebench_task(original_dataset_name, dataset_path)
        is_longbench_v2 = (
            original_dataset_name.lower() == "longbench_v2"
            or is_longbench_v2_dataset_path(dataset_path)
        )
        instances = []
        with dataset_path.open("r", encoding="utf-8") as f:
            # Some exports use a ``.jsonl`` name but store one pretty-printed JSON array.
            head = f.read(8192)
            f.seek(0)
            if head.lstrip("\ufeff").lstrip().startswith("["):
                data = json.load(f)
                if not isinstance(data, list):
                    raise ValueError(f"{dataset_path} must contain a JSON list")
                if (
                    original_dataset_name.lower() == "longbench_v2"
                    or is_longbench_v2_dataset_path(dataset_path)
                ):
                    return load_longbench_v2_json_records(data, dataset_path)
                if should_load_as_multifieldqa_en_mixup(data, dataset_path, original_dataset_name):
                    return load_multifieldqa_en_mixup_json_records(data, dataset_path)
                if is_swe_bench_style_json(data, dataset_path, original_dataset_name):
                    return load_swe_bench_json_instances(data, dataset_path)
                raise ValueError(
                    f"Unsupported JSON-array dataset with .jsonl extension: {dataset_path}"
                )

            for line_number, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue

                data = json.loads(line)
                if is_longbench_v2:
                    instances.append({"turns": [format_longbench_v2_prompt(data)]})
                    continue
                if "input" not in data:
                    raise ValueError(
                        f"Missing 'input' field in {dataset_path} at line {line_number}"
                    )
                if "context" not in data:
                    raise ValueError(
                        f"Missing 'context' field in {dataset_path} at line {line_number}"
                    )
                if task_name is not None:
                    prompt = format_infinitebench_prompt(data, task_name)
                else:
                    prompt = f"{data['context']}\nQuestion: {data['input']}"
                instances.append({"turns": [prompt]})
        return instances

    return load_and_process_dataset(dataset_name)


def select_max_samples(dataset, max_samples: int | None):
    if max_samples is None or len(dataset) <= max_samples:
        return dataset

    if hasattr(dataset, "shuffle") and hasattr(dataset, "select"):
        return dataset.shuffle(seed=0).select(range(max_samples))

    indices = list(range(len(dataset)))
    rng = random.Random(0)
    rng.shuffle(indices)
    return [dataset[i] for i in indices[:max_samples]]


def _is_blackwell() -> bool:
    if envs.IS_BLACKWELL.get():
        return True
    return get_device_sm() >= 100


def _flush_cache(base_url: str) -> None:
    response = requests.get(base_url + "/flush_cache", timeout=60)
    response.raise_for_status()


def _sampling_params(max_new_tokens: int, stop: list[str]) -> dict:
    params = {
        "temperature": 0.0,
        "top_p": 1.0,
        "top_k": 1,
        "max_new_tokens": int(max_new_tokens),
    }
    if stop:
        params["stop"] = stop
    return params


def _send_generate(
    base_url: str,
    prompt: str,
    *,
    max_new_tokens: int,
    stop: list[str],
    timeout_s: int,
) -> dict:
    started = time.perf_counter()
    response = requests.post(
        base_url + "/generate",
        json={
            "text": prompt,
            "sampling_params": _sampling_params(max_new_tokens, stop),
        },
        timeout=int(timeout_s),
    )
    response.raise_for_status()
    result = response.json()
    result["_client_latency_s"] = time.perf_counter() - started
    return result


def _send_generate_batch(
    base_url: str,
    prompts: list[str],
    *,
    max_new_tokens: int,
    stop: list[str],
    timeout_s: int,
) -> list[dict]:
    if not prompts:
        return []
    started = time.perf_counter()
    response = requests.post(
        base_url + "/generate",
        json={
            "text": prompts,
            "sampling_params": _sampling_params(max_new_tokens, stop),
        },
        timeout=int(timeout_s),
    )
    response.raise_for_status()
    outputs = response.json()
    if not isinstance(outputs, list):
        raise RuntimeError(
            "Expected a list from batched /generate, got "
            f"{type(outputs).__name__}."
        )
    elapsed = time.perf_counter() - started
    for output in outputs:
        output["_client_latency_s"] = elapsed
    return outputs


@dataclass(frozen=True)
class BenchMetrics:
    latency_s: float
    model_decode_time_s: float
    mean_request_latency_s: float
    p50_request_latency_s: float
    p99_request_latency_s: float
    output_tokens: int
    output_toks_per_s: float
    spec_accept_length: Optional[float]
    spec_verify_ct_sum: int
    responses: list[dict]


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round((len(ordered) - 1) * percentile)))
    return float(ordered[index])


def _write_response_records(response_json: Optional[str], records: list[dict]) -> None:
    if not response_json:
        return
    path = Path(response_json)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f".{path.name}.tmp")
    with tmp_path.open("w", encoding="utf-8") as file:
        json.dump(records, file, ensure_ascii=False, indent=2)
        file.write("\n")
    tmp_path.replace(path)


def _run_bench_requests(
    base_url: str,
    *,
    prompts: list[str],
    max_new_tokens: int,
    concurrency: int,
    batch_requests: bool,
    stop: list[str],
    timeout_s: int,
    expect_flashmtp: bool,
    on_response: Optional[Callable[[dict], None]] = None,
) -> BenchMetrics:
    # One full concurrency-sized batch is deliberately excluded for JIT/graph warmup.
    batch_size = max(int(concurrency), 1)
    warmup_count = min(batch_size, len(prompts))
    warmup_prompts = prompts[:warmup_count]
    if warmup_prompts:
        if batch_requests:
            warmup_outputs = _send_generate_batch(
                base_url,
                warmup_prompts,
                max_new_tokens=max_new_tokens,
                stop=stop,
                timeout_s=timeout_s,
            )
        else:
            with ThreadPoolExecutor(max_workers=batch_size) as pool:
                warmup_outputs = list(
                    pool.map(
                        lambda prompt: _send_generate(
                            base_url,
                            prompt,
                            max_new_tokens=max_new_tokens,
                            stop=stop,
                            timeout_s=timeout_s,
                        ),
                        warmup_prompts,
                    )
                )
        if on_response is not None:
            for index, output in enumerate(warmup_outputs):
                on_response(
                    {
                        "index": index,
                        "warmup": True,
                        "response": output.get("text", ""),
                        "meta_info": output.get("meta_info", {}) or {},
                        "client_latency_s": output.get("_client_latency_s"),
                    }
                )

    measured_prompts = prompts[warmup_count:]
    started = time.perf_counter()
    outputs_by_index: list[tuple[int, dict]] = []
    if batch_requests:
        for start_index in range(0, len(measured_prompts), batch_size):
            chunk = measured_prompts[start_index : start_index + batch_size]
            outputs = _send_generate_batch(
                base_url,
                chunk,
                max_new_tokens=max_new_tokens,
                stop=stop,
                timeout_s=timeout_s,
            )
            if len(outputs) != len(chunk):
                raise RuntimeError(
                    f"Batched output mismatch: {len(outputs)} outputs for {len(chunk)} prompts."
                )
            outputs_by_index.extend(
                (start_index + offset, output)
                for offset, output in enumerate(outputs)
            )
    else:
        with ThreadPoolExecutor(max_workers=batch_size) as pool:
            futures = {
                pool.submit(
                    _send_generate,
                    base_url,
                    prompt,
                    max_new_tokens=max_new_tokens,
                    stop=stop,
                    timeout_s=timeout_s,
                ): index
                for index, prompt in enumerate(measured_prompts)
            }
            for future in as_completed(futures):
                outputs_by_index.append((futures[future], future.result()))
    wall_latency = time.perf_counter() - started

    total_tokens = 0
    model_decode_time_sum = 0.0
    model_decode_time_count = 0
    spec_verify_ct_sum = 0
    # Globally weight acceptance by target verification positions. An
    # equal-weighted mean over requests over-represents short generations.
    accept_length_weighted_sum = 0.0
    request_latencies: list[float] = []
    records: list[dict] = []
    for measured_index, output in outputs_by_index:
        meta = output.get("meta_info", {}) or {}
        total_tokens += int(meta.get("completion_tokens", 0))
        if meta.get("model_decode_time") is not None:
            model_decode_time_sum += float(meta["model_decode_time"])
            model_decode_time_count += 1
        spec_verify_ct_sum += int(meta.get("spec_verify_ct", 0))
        if meta.get("spec_accept_length") is not None:
            accept_length_weighted_sum += (
                float(meta["spec_accept_length"])
                * int(meta.get("spec_verify_ct", 0))
            )
        request_latencies.append(float(output.get("_client_latency_s", 0.0)))
        record = {
            "index": warmup_count + measured_index,
            "warmup": False,
            "response": output.get("text", ""),
            "meta_info": meta,
            "client_latency_s": output.get("_client_latency_s"),
        }
        records.append(record)
        if on_response is not None:
            on_response(record)

    if expect_flashmtp and spec_verify_ct_sum <= 0:
        raise RuntimeError(
            "FlashMTP sanity check failed: no response reported spec_verify_ct > 0."
        )

    if model_decode_time_count != len(outputs_by_index):
        raise RuntimeError(
            "Server-side decode timing is incomplete: observed "
            f"{model_decode_time_count}/{len(outputs_by_index)} `model_decode_time` values."
        )

    return BenchMetrics(
        latency_s=float(wall_latency),
        model_decode_time_s=float(model_decode_time_sum),
        mean_request_latency_s=(
            float(statistics.mean(request_latencies)) if request_latencies else 0.0
        ),
        p50_request_latency_s=_percentile(request_latencies, 0.50),
        p99_request_latency_s=_percentile(request_latencies, 0.99),
        output_tokens=total_tokens,
        output_toks_per_s=total_tokens / max(model_decode_time_sum, 1e-6),
        spec_accept_length=(
            float(accept_length_weighted_sum / spec_verify_ct_sum)
            if spec_verify_ct_sum > 0
            else None
        ),
        spec_verify_ct_sum=spec_verify_ct_sum,
        responses=sorted(records, key=lambda record: record["index"]),
    )


def _wait_for_server(process: subprocess.Popen, base_url: str, timeout_s: float) -> None:
    deadline = time.monotonic() + timeout_s
    last_error = "server did not answer"
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(
                f"SGLang server exited with code {process.returncode}: {last_error}"
            )
        for endpoint in ("/health_generate", "/health"):
            try:
                response = requests.get(base_url + endpoint, timeout=2)
                if response.status_code == 200:
                    return
                last_error = f"{endpoint} returned {response.status_code}"
            except requests.RequestException as exc:
                last_error = str(exc)
        time.sleep(1)
    kill_process_tree(process.pid)
    raise TimeoutError(f"Timed out waiting for {base_url}: {last_error}")


def _launch_server(
    *,
    model: str,
    base_url: str,
    other_args: list[str],
    flashmtp: bool,
    timeout_s: float,
) -> subprocess.Popen:
    _, host, port = base_url.split(":")
    module = (
        "specforge.sglang_flashmtp.launch_server"
        if flashmtp
        else "sglang.launch_server"
    )
    command = [
        sys.executable,
        "-m",
        module,
        "--model-path",
        model,
        *other_args,
        "--host",
        host[2:],
        "--port",
        port,
    ]
    env = os.environ.copy()
    if flashmtp:
        env["SGLANG_FLASHMTP_ACTIVE"] = "1"
        overlap = "--disable-overlap-schedule" not in other_args
        env["SGLANG_ENABLE_SPEC_V2"] = "True" if overlap else "False"
        env["SGLANG_ENABLE_DFLASH_SPEC_V2"] = "True" if overlap else "False"
    print("[server] " + " ".join(command))
    process = subprocess.Popen(command, env=env, start_new_session=True)
    try:
        _wait_for_server(process, base_url, timeout_s)
    except Exception:
        if process.poll() is None:
            kill_process_tree(process.pid)
        raise
    return process


def _stop_server(process: subprocess.Popen) -> None:
    kill_process_tree(process.pid)
    try:
        process.wait(timeout=30)
    except subprocess.TimeoutExpired:
        process.kill()


def _server_gpu_memory_gib(tp_size: int) -> float:
    """Return aggregate device memory in use while the server is resident."""
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    device_ids = (
        [item.strip() for item in visible.split(",") if item.strip()]
        if visible
        else [str(index) for index in range(torch.cuda.device_count())]
    )
    selected = device_ids[: int(tp_size)]
    command = [
        "nvidia-smi",
        f"--id={','.join(selected)}",
        "--query-gpu=memory.used",
        "--format=csv,noheader,nounits",
    ]
    output = subprocess.run(
        command, check=True, capture_output=True, text=True
    ).stdout
    used_mib = sum(float(line.strip()) for line in output.splitlines() if line.strip())
    return used_mib / 1024.0


def _format_table(
    concurrencies: list[int],
    rows: list[tuple[str, dict[int, Optional[float]], str]],
) -> str:
    header = ["metric", *[str(value) for value in concurrencies]]
    lines = [
        "| " + " | ".join(header) + " |",
        "| " + " | ".join(["---"] * len(header)) + " |",
    ]
    for label, values, fmt in rows:
        cells = [label]
        for concurrency in concurrencies:
            value = values.get(concurrency)
            cells.append("N/A" if value is None else format(value, fmt))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def _parse_csv_ints(value: str) -> list[int]:
    result = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not result or any(item <= 0 for item in result):
        raise ValueError(f"Expected positive comma-separated integers, got {value!r}.")
    return result


def _build_yarn_rope_scaling(
    *,
    factor: float,
    original_max_position_embeddings: int,
    rope_theta: float,
) -> dict:
    if factor <= 1.0:
        raise ValueError(f"YaRN factor must be > 1, got {factor}.")
    return {
        "rope_type": "yarn",
        "factor": float(factor),
        "original_max_position_embeddings": int(original_max_position_embeddings),
        "rope_theta": float(rope_theta),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="HTTP throughput/latency benchmark for baseline SGLang and FlashMTP."
    )
    parser.add_argument("--output-md", default=None)
    parser.add_argument("--response-json", default="response.json")
    parser.add_argument("--dataset-name", "--dataset", dest="dataset_name", required=True)
    parser.add_argument(
        "--target-model",
        "--model-name-or-path",
        dest="target_model",
        default="Qwen/Qwen3-8B",
    )
    parser.add_argument(
        "--draft-model",
        "--draft-name-or-path",
        dest="draft_model",
        required=True,
    )
    parser.add_argument("--skip-baseline", action="store_true")
    parser.add_argument("--batch-requests", action="store_true")
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--timeout-s", type=int, default=3600)
    parser.add_argument("--server-launch-timeout-s", type=int, default=1800)
    parser.add_argument("--mem-fraction-static", type=float, default=0.75)
    parser.add_argument("--disable-radix-cache", action="store_true")
    parser.add_argument("--disable-overlap-schedule", action="store_true")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--max-running-requests", type=int, default=None)
    parser.add_argument("--context-length", type=int, default=None)
    parser.add_argument("--max-input-tokens", type=int, default=None)
    parser.add_argument(
        "--yarn",
        action="store_true",
        help="Enable YaRN via --json-model-override-args rope_scaling for the target model.",
    )
    parser.add_argument(
        "--yarn-factor",
        type=float,
        default=2.0,
        help="YaRN scaling factor. With Qwen3-8B native 40960, factor=2 yields 81920.",
    )
    parser.add_argument(
        "--yarn-original-max-pos",
        type=int,
        default=40960,
        help="Native max_position_embeddings before YaRN extension.",
    )
    parser.add_argument(
        "--yarn-rope-theta",
        type=float,
        default=1_000_000.0,
        help="rope_theta passed into the YaRN rope_scaling override.",
    )
    parser.add_argument("--tp-size", type=int, default=1)
    parser.add_argument("--concurrencies", default=None)
    parser.add_argument(
        "--questions-per-concurrency-base", type=int, default=10
    )
    parser.add_argument("--max-questions-per-config", type=int, default=1024)
    parser.add_argument("--attention-backends", default="flashinfer")
    parser.add_argument("--cuda-graph-batch-sizes", default="1,2,4,8")
    parser.add_argument("--block-size", type=int, default=None)
    parser.add_argument(
        "--port-base",
        type=int,
        default=20000,
        help="Starting port used to find an available server port.",
    )

    # Compatibility aliases from the former local Transformers benchmark.
    parser.add_argument("--batch-size", type=int, default=1, help=argparse.SUPPRESS)
    parser.add_argument("--temperature", type=float, default=0.0, help=argparse.SUPPRESS)
    parser.add_argument("--local-position", choices=("true", "false"), default=None, help=argparse.SUPPRESS)
    parser.add_argument("--sink-num", type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--use-draft-tree", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--draft-tree-trunc-thres", type=float, default=0.2, help=argparse.SUPPRESS)
    parser.add_argument("--draft-tree-expand-thres", type=float, default=0.5, help=argparse.SUPPRESS)
    parser.add_argument("--draft-tree-width", type=int, default=4, help=argparse.SUPPRESS)
    parser.add_argument("--draft-tree-entropy-ratio", type=float, default=0.4, help=argparse.SUPPRESS)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required.")
    if abs(args.temperature) > 1e-8:
        raise ValueError("FlashMTP SGLang supports only --temperature 0.")
    if args.local_position == "false":
        raise ValueError("Only local_position=true is supported.")
    if args.use_draft_tree:
        raise ValueError("Draft-tree decoding is outside the first FlashMTP SGLang scope.")
    if args.yarn and args.context_length is None:
        args.context_length = int(args.yarn_original_max_pos * args.yarn_factor)

    concurrencies = _parse_csv_ints(
        args.concurrencies if args.concurrencies is not None else str(args.batch_size)
    )
    max_concurrency = max(concurrencies)
    graph_batch_sizes = _parse_csv_ints(args.cuda_graph_batch_sizes)
    questions_by_concurrency = {
        concurrency: min(
            args.max_samples
            if args.max_samples is not None
            else args.questions_per_concurrency_base * concurrency,
            args.max_questions_per_config,
        )
        for concurrency in concurrencies
    }
    if any(count <= 0 for count in questions_by_concurrency.values()):
        raise ValueError("Every benchmark configuration needs at least one sample.")

    backends = [
        backend.strip()
        for backend in args.attention_backends.split(",")
        if backend.strip()
    ]
    device_sm = get_device_sm()
    backends = [
        backend
        for backend in backends
        if not (backend == "fa3" and device_sm != 90)
        and not (backend == "fa4" and device_sm < 100)
    ]
    if not backends:
        backends = ["flashinfer"]

    dataset = load_benchmark_dataset(args.dataset_name)
    if not dataset:
        raise RuntimeError(f"Dataset {args.dataset_name!r} is empty.")
    dataset = select_max_samples(dataset, args.max_samples)
    questions_by_concurrency = {
        concurrency: min(count, len(dataset))
        for concurrency, count in questions_by_concurrency.items()
    }
    required_prompts = max(questions_by_concurrency.values())
    tokenizer = AutoTokenizer.from_pretrained(args.target_model, trust_remote_code=True)
    max_input_tokens = (
        args.max_input_tokens
        if args.max_input_tokens is not None
        else args.context_length
    )
    prompts: list[str] = []
    skipped = 0
    for index in range(len(dataset)):
        item = dataset[index]
        turns = item.get("turns") or []
        if not turns:
            continue
        prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": str(turns[0])}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        if max_input_tokens is not None:
            token_count = len(tokenizer.encode(prompt, add_special_tokens=False))
            if token_count > max_input_tokens:
                skipped += 1
                continue
        prompts.append(prompt)
        if len(prompts) >= required_prompts:
            break
    if len(prompts) < required_prompts:
        raise RuntimeError(
            f"Built only {len(prompts)}/{required_prompts} prompts; "
            f"{skipped} exceeded max_input_tokens={max_input_tokens}."
        )

    baseline: dict[tuple[str, int], BenchMetrics] = {}
    flashmtp: dict[tuple[str, int], BenchMetrics] = {}
    server_memory_gib: dict[tuple[str, str], float] = {}
    response_records: list[dict] = []
    _write_response_records(args.response_json, response_records)

    def persist(runner: str, backend: str, concurrency: int, record: dict) -> None:
        response_records.append(
            {
                "dataset": args.dataset_name,
                "runner": runner,
                "backend": backend,
                "concurrency": concurrency,
                "question_index": record["index"],
                **{key: value for key, value in record.items() if key != "index"},
            }
        )
        _write_response_records(args.response_json, response_records)

    for backend in backends:
        common_args = [
            "--trust-remote-code",
            "--attention-backend",
            backend,
            "--tp-size",
            str(args.tp_size),
            "--dtype",
            args.dtype,
            "--mem-fraction-static",
            str(args.mem_fraction_static),
            "--max-running-requests",
            str(args.max_running_requests or max_concurrency),
            "--cuda-graph-bs",
            *[str(value) for value in graph_batch_sizes],
            "--cuda-graph-max-bs",
            str(max(graph_batch_sizes)),
        ]
        if args.disable_radix_cache:
            common_args.append("--disable-radix-cache")
        if args.yarn:
            rope_scaling = _build_yarn_rope_scaling(
                factor=args.yarn_factor,
                original_max_position_embeddings=args.yarn_original_max_pos,
                rope_theta=args.yarn_rope_theta,
            )
            common_args.extend(
                [
                    "--json-model-override-args",
                    json.dumps({"rope_scaling": rope_scaling}),
                ]
            )
        if args.context_length is not None:
            common_args.extend(["--context-length", str(args.context_length)])

        runners: list[tuple[str, bool]] = []
        if not args.skip_baseline:
            runners.append(("baseline", False))
        runners.append(("flashmtp", True))
        for runner_name, use_flashmtp in runners:
            port = find_available_port(args.port_base)
            base_url = f"http://127.0.0.1:{port}"
            server_args = list(common_args)
            if use_flashmtp:
                server_args.extend(
                    [
                        "--speculative-algorithm",
                        "FLASHMTP",
                        "--speculative-draft-model-path",
                        args.draft_model,
                    ]
                )
                if args.block_size is not None:
                    server_args.extend(
                        ["--speculative-num-draft-tokens", str(args.block_size)]
                    )
                if args.disable_overlap_schedule:
                    server_args.append("--disable-overlap-schedule")
                    # Qwen3.5 hybrid linear-attention models reject speculative
                    # decoding with the no-buffer radix-cache scheduler in spec-v1.
                    if "--disable-radix-cache" not in server_args:
                        server_args.append("--disable-radix-cache")
                else:
                    # Hybrid Qwen3.5 needs the extra-buffer Mamba scheduler to
                    # combine radix cache with speculative overlap scheduling.
                    server_args.extend(
                        ["--mamba-scheduler-strategy", "extra_buffer"]
                    )

            print(
                f"\n=== runner={runner_name} backend={backend} "
                f"tp={args.tp_size} overlap={use_flashmtp and not args.disable_overlap_schedule} ==="
            )
            memory_before_gib = _server_gpu_memory_gib(args.tp_size)
            process = _launch_server(
                model=args.target_model,
                base_url=base_url,
                other_args=server_args,
                flashmtp=use_flashmtp,
                timeout_s=float(args.server_launch_timeout_s),
            )
            try:
                server_memory_gib[(runner_name, backend)] = max(
                    0.0,
                    _server_gpu_memory_gib(args.tp_size) - memory_before_gib,
                )
                print(
                    f"[{runner_name}] incremental resident GPU memory="
                    f"{server_memory_gib[(runner_name, backend)]:.2f} GiB"
                )
                _send_generate(
                    base_url,
                    "Hello",
                    max_new_tokens=8,
                    stop=[],
                    timeout_s=min(args.timeout_s, 300),
                )
                for concurrency in concurrencies:
                    _flush_cache(base_url)
                    measured_count = questions_by_concurrency[concurrency]
                    request_prompts = [
                        prompts[index % len(prompts)]
                        for index in range(concurrency)
                    ] + prompts[:measured_count]
                    metrics = _run_bench_requests(
                        base_url,
                        prompts=request_prompts,
                        max_new_tokens=args.max_new_tokens,
                        concurrency=concurrency,
                        batch_requests=args.batch_requests,
                        stop=[],
                        timeout_s=args.timeout_s,
                        expect_flashmtp=use_flashmtp,
                        on_response=lambda record, r=runner_name, b=backend, c=concurrency: persist(
                            r, b, c, record
                        ),
                    )
                    target = flashmtp if use_flashmtp else baseline
                    target[(backend, concurrency)] = metrics
                    accept = (
                        "N/A"
                        if metrics.spec_accept_length is None
                        else f"{metrics.spec_accept_length:.3f}"
                    )
                    print(
                        f"[{runner_name}] concurrency={concurrency} n={measured_count} "
                        f"tok/s={metrics.output_toks_per_s:,.2f} "
                        f"decode={metrics.model_decode_time_s:.3f}s "
                        f"wall={metrics.latency_s:.3f}s "
                        f"mean/p50/p99={metrics.mean_request_latency_s:.3f}/"
                        f"{metrics.p50_request_latency_s:.3f}/"
                        f"{metrics.p99_request_latency_s:.3f}s "
                        f"accept={accept} verify_ct={metrics.spec_verify_ct_sum}"
                    )
            finally:
                _stop_server(process)

    markdown = [
        "# FlashMTP SGLang Benchmark",
        "",
        "## Settings",
        "",
        f"- dataset: `{args.dataset_name}`",
        f"- target: `{args.target_model}`",
        f"- draft: `{args.draft_model}`",
        f"- overlap: `{not args.disable_overlap_schedule}`",
        f"- TP: `{args.tp_size}`",
        f"- attention backends: `{', '.join(backends)}`",
        f"- context length: `{args.context_length}`",
        f"- yarn: `{args.yarn}`"
        + (
            f" (factor={args.yarn_factor}, original_max_pos={args.yarn_original_max_pos})"
            if args.yarn
            else ""
        ),
        f"- radix cache: `{not args.disable_radix_cache}`",
        "- timing scope: `first model decode schedule -> final decode/verify result processed`",
        "- excluded timing: `target prefill, HTTP/client, tokenization`",
        "- throughput weighting: `sum(output_tokens) / sum(model_decode_time)`",
        "- acceptance weighting: `sum(verify_ct * request_accept_length) / sum(verify_ct)`",
        "- first full batch excluded as warmup: `true`",
        "",
    ]
    for backend in backends:
        def values(source: dict, field: str) -> dict[int, Optional[float]]:
            return {
                concurrency: (
                    getattr(source[(backend, concurrency)], field)
                    if (backend, concurrency) in source
                    else None
                )
                for concurrency in concurrencies
            }

        baseline_tps = values(baseline, "output_toks_per_s")
        flashmtp_tps = values(flashmtp, "output_toks_per_s")
        speedup = {
            concurrency: (
                None
                if baseline_tps[concurrency] in (None, 0)
                else flashmtp_tps[concurrency] / baseline_tps[concurrency]
            )
            for concurrency in concurrencies
        }
        markdown.extend(
            [
                f"## Backend: `{backend}`",
                "",
                _format_table(
                    concurrencies,
                    [
                        ("baseline decode-only token-weighted output tok/s", baseline_tps, ",.2f"),
                        ("FlashMTP decode-only token-weighted output tok/s", flashmtp_tps, ",.2f"),
                        ("baseline measured output tokens", values(baseline, "output_tokens"), ",.0f"),
                        ("FlashMTP measured output tokens", values(flashmtp, "output_tokens"), ",.0f"),
                        ("baseline measured wall time (s)", values(baseline, "latency_s"), ".3f"),
                        ("FlashMTP measured wall time (s)", values(flashmtp, "latency_s"), ".3f"),
                        ("baseline measured model decode time (s)", values(baseline, "model_decode_time_s"), ".3f"),
                        ("FlashMTP measured model decode time (s)", values(flashmtp, "model_decode_time_s"), ".3f"),
                        ("speedup", speedup, ".3f"),
                        ("FlashMTP globally verification-weighted accept length", values(flashmtp, "spec_accept_length"), ".3f"),
                        ("baseline mean latency (s)", values(baseline, "mean_request_latency_s"), ".3f"),
                        ("FlashMTP mean latency (s)", values(flashmtp, "mean_request_latency_s"), ".3f"),
                        ("FlashMTP p99 latency (s)", values(flashmtp, "p99_request_latency_s"), ".3f"),
                        (
                            "baseline incremental GPU memory (GiB)",
                            {
                                concurrency: server_memory_gib.get(("baseline", backend))
                                for concurrency in concurrencies
                            },
                            ".2f",
                        ),
                        (
                            "FlashMTP incremental GPU memory (GiB)",
                            {
                                concurrency: server_memory_gib.get(("flashmtp", backend))
                                for concurrency in concurrencies
                            },
                            ".2f",
                        ),
                    ],
                ),
                "",
            ]
        )

    report = "\n".join(markdown) + "\n"
    print("\n" + report)
    if args.output_md:
        output_path = Path(args.output_md)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(report, encoding="utf-8")
        print(f"Wrote markdown report to {output_path}")
    if args.response_json:
        _write_response_records(args.response_json, response_records)
        print(f"Wrote responses to {args.response_json}")


if __name__ == "__main__":
    main()
