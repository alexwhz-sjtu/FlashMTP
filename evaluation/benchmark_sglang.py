from __future__ import annotations

import argparse
import json
import os
import re
import time
import statistics
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import requests
import torch
from transformers import AutoTokenizer

import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
VENV_BIN = PROJECT_ROOT / ".venv" / "bin"

from evaluation.utils import load_and_process_dataset

from sglang.srt.environ import envs
from sglang.srt.utils import get_device_sm, kill_process_tree
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    find_available_port,
    popen_launch_server,
)

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


def resolve_dataset_path(dataset_name: str) -> str:
    if not DATASET_PATH_FILE.is_file():
        return dataset_name

    with DATASET_PATH_FILE.open("r", encoding="utf-8") as f:
        dataset_paths = json.load(f)

    if not isinstance(dataset_paths, dict):
        raise ValueError(f"{DATASET_PATH_FILE} must contain a JSON object")

    return dataset_paths.get(dataset_name, dataset_name)


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


def load_benchmark_dataset(dataset_name: str):
    original_dataset_name = dataset_name
    dataset_name = resolve_dataset_path(dataset_name)
    dataset_path = Path(dataset_name)
    if dataset_path.is_file() and dataset_path.suffix == ".json":
        with dataset_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, list):
            raise ValueError(f"{dataset_path} must contain a JSON list")

        if original_dataset_name.lower() == "longbench_v2" or dataset_path.parent.name.lower() == "longbench_v2":
            return [{"turns": [format_longbench_v2_prompt(item)]} for item in data]

        raise ValueError(f"Unsupported JSON dataset: {dataset_path}")

    if dataset_path.is_file() and dataset_path.suffix == ".jsonl":
        task_name = infer_infinitebench_task(original_dataset_name, dataset_path)
        instances = []
        with dataset_path.open("r", encoding="utf-8") as f:
            for line_number, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue

                data = json.loads(line)
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


def _is_blackwell() -> bool:
    if envs.IS_BLACKWELL.get():
        return True
    return get_device_sm() >= 100


def _flush_cache(base_url: str) -> None:
    resp = requests.get(base_url + "/flush_cache", timeout=60)
    resp.raise_for_status()


def _send_generate(
    base_url: str,
    prompt: str,
    *,
    max_new_tokens: int,
    stop: list[str],
    timeout_s: int,
) -> dict:
    sampling_params: dict = {
        "temperature": 0.0,
        "top_p": 1.0,
        "top_k": 1,
        "max_new_tokens": int(max_new_tokens),
    }
    if stop:
        sampling_params["stop"] = stop
    resp = requests.post(
        base_url + "/generate",
        json={
            "text": prompt,
            "sampling_params": sampling_params,
        },
        timeout=int(timeout_s),
    )
    resp.raise_for_status()
    return resp.json()


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
    sampling_params: dict = {
        "temperature": 0.0,
        "top_p": 1.0,
        "top_k": 1,
        "max_new_tokens": int(max_new_tokens),
    }
    if stop:
        sampling_params["stop"] = stop
    resp = requests.post(
        base_url + "/generate",
        json={
            "text": prompts,
            "sampling_params": sampling_params,
        },
        timeout=int(timeout_s),
    )
    resp.raise_for_status()
    out = resp.json()
    if not isinstance(out, list):
        raise RuntimeError(
            "Expected a list response for batched /generate, but got "
            f"type={type(out).__name__}."
        )
    return out


@dataclass(frozen=True)
class BenchMetrics:
    latency_s: float
    output_tokens: int
    output_toks_per_s: float
    spec_accept_length: Optional[float]
    spec_verify_ct_sum: int
    responses: list[dict]


def _write_response_records(response_json: Optional[str], records: list[dict]) -> None:
    if not response_json:
        return

    path = Path(response_json)
    tmp_path = path.with_name(f".{path.name}.tmp")
    with tmp_path.open("w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)
        f.write("\n")
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
    # Drop the first batch from metrics to exclude one-time JIT/cuda-graph overhead
    bs = max(int(concurrency), 1)
    warmup_count = 0
    if len(prompts) > bs:
        warmup_prompts = prompts[:bs]
        warmup_count = bs
        if batch_requests:
            warmup_outs = _send_generate_batch(
                base_url,
                warmup_prompts,
                max_new_tokens=max_new_tokens,
                stop=stop,
                timeout_s=timeout_s,
            )
            if on_response is not None:
                for j, out in enumerate(warmup_outs):
                    on_response(
                        {
                            "index": j,
                            "warmup": True,
                            "response": out.get("text", ""),
                            "meta_info": out.get("meta_info", {}) or {},
                        }
                    )
        else:
            with ThreadPoolExecutor(max_workers=int(concurrency)) as pool:
                futures = {
                    pool.submit(
                        _send_generate,
                        base_url,
                        prompt,
                        max_new_tokens=max_new_tokens,
                        stop=stop,
                        timeout_s=timeout_s,
                    ): i
                    for i, prompt in enumerate(warmup_prompts)
                }
                for fut in as_completed(futures):
                    out = fut.result()
                    if on_response is not None:
                        on_response(
                            {
                                "index": futures[fut],
                                "warmup": True,
                                "response": out.get("text", ""),
                                "meta_info": out.get("meta_info", {}) or {},
                            }
                        )

        prompts = prompts[bs:]

    start = time.perf_counter()
    total_tokens = 0
    spec_verify_ct_sum = 0
    spec_accept_lengths: list[float] = []
    responses: list[dict] = []

    if batch_requests:
        bs = max(int(concurrency), 1)
        for start_idx in range(0, len(prompts), bs):
            chunk_prompts = prompts[start_idx : start_idx + bs]
            outs = _send_generate_batch(
                base_url,
                chunk_prompts,
                max_new_tokens=max_new_tokens,
                stop=stop,
                timeout_s=timeout_s,
            )
            if len(outs) != len(chunk_prompts):
                raise RuntimeError(
                    "Batched /generate output length mismatch: "
                    f"got {len(outs)} outputs for {len(chunk_prompts)} prompts."
                )

            for j, out in enumerate(outs):
                meta = out.get("meta_info", {}) or {}
                total_tokens += int(meta.get("completion_tokens", 0))
                spec_verify_ct_sum += int(meta.get("spec_verify_ct", 0))
                if "spec_accept_length" in meta:
                    try:
                        spec_accept_lengths.append(float(meta["spec_accept_length"]))
                    except (TypeError, ValueError):
                        pass
                record = {
                    "index": warmup_count + start_idx + j,
                    "warmup": False,
                    "response": out.get("text", ""),
                    "meta_info": meta,
                }
                responses.append(record)
                if on_response is not None:
                    on_response(record)
    else:
        with ThreadPoolExecutor(max_workers=int(concurrency)) as pool:
            futures = {
                pool.submit(
                    _send_generate,
                    base_url,
                    prompt,
                    max_new_tokens=max_new_tokens,
                    stop=stop,
                    timeout_s=timeout_s,
                ): i
                for i, prompt in enumerate(prompts)
            }
            for fut in as_completed(futures):
                out = fut.result()
                prompt_index = futures[fut]
                meta = out.get("meta_info", {}) or {}
                total_tokens += int(meta.get("completion_tokens", 0))
                spec_verify_ct_sum += int(meta.get("spec_verify_ct", 0))
                if "spec_accept_length" in meta:
                    try:
                        spec_accept_lengths.append(float(meta["spec_accept_length"]))
                    except (TypeError, ValueError):
                        pass
                record = {
                    "index": warmup_count + prompt_index,
                    "warmup": False,
                    "response": out.get("text", ""),
                    "meta_info": meta,
                }
                responses.append(record)
                if on_response is not None:
                    on_response(record)

    latency = time.perf_counter() - start
    toks_per_s = total_tokens / max(latency, 1e-6)

    if expect_flashmtp and spec_verify_ct_sum <= 0:
        raise RuntimeError(
            "FlashMTP sanity check failed: did not observe any `spec_verify_ct` in responses "
            "(server-side speculative decoding may not have been enabled)."
        )

    spec_accept_length = (
        float(statistics.mean(spec_accept_lengths)) if spec_accept_lengths else None
    )

    return BenchMetrics(
        latency_s=float(latency),
        output_tokens=int(total_tokens),
        output_toks_per_s=float(toks_per_s),
        spec_accept_length=spec_accept_length,
        spec_verify_ct_sum=int(spec_verify_ct_sum),
        responses=sorted(responses, key=lambda x: x["index"]),
    )


def _format_table(
    *,
    concurrencies: list[int],
    values: dict[int, Optional[float]],
    float_fmt: str,
) -> str:
    header = ["conc"] + [str(c) for c in concurrencies]
    lines = [
        "| " + " | ".join(header) + " |",
        "| " + " | ".join(["---"] * len(header)) + " |",
    ]
    row = ["value"]
    for c in concurrencies:
        v = values.get(c, None)
        row.append("N/A" if v is None else format(v, float_fmt))
    lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-md",
        type=str,
        default=None,
        help="Write a markdown report to this file (disabled by default).",
    )
    parser.add_argument(
        "--response-json",
        type=str,
        default="response.json",
        help="Write per-question generated responses to this JSON file.",
    )
    parser.add_argument("--dataset-name", type=str, default="gsm8k")
    parser.add_argument("--target-model", type=str, default="/data/wanghanzhen/models/Qwen/Qwen3-8B")
    parser.add_argument(
        "--draft-model",
        type=str,
        default="/data/wanghanzhen/Projects/MTP/NIPS26/FlashMTP_v5.1/cache/models/flashmtp_v5.1_fix_h100_sample_40000_think_off_nlayers5_block_16_maxlen4096_epochs6/epoch_6_step_29844",
    )
    parser.add_argument(
        "--skip-baseline",
        action="store_true",
        help="Skip running the baseline (target-only) sweep; only run FlashMTP and report N/A for baseline/speedup.",
    )
    parser.add_argument(
        "--batch-requests",
        action="store_true",
        help="Send prompts as server-side batched /generate requests (batch size = concurrency) instead of client-side concurrent requests.",
    )
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--timeout-s", type=int, default=3600)
    parser.add_argument("--mem-fraction-static", type=float, default=0.75)
    parser.add_argument("--disable-radix-cache", action="store_true")
    parser.add_argument("--dtype", type=str, default="bfloat16")
    parser.add_argument("--max-running-requests", type=int, default=64)
    parser.add_argument(
        "--context-length",
        type=int,
        default=None,
        help="Optional SGLang server context length override, e.g. 200000 for 160k-token prompts.",
    )
    parser.add_argument(
        "--max-input-tokens",
        type=int,
        default=None,
        help="Skip prompts whose tokenized input length exceeds this value. Defaults to --context-length when set.",
    )
    parser.add_argument(
        "--tp-size",
        type=int,
        default=1,
        help="Tensor parallel size (single value, no sweep).",
    )
    parser.add_argument("--nnodes", type=int, default=1, help="Number of SGLang server nodes.")
    parser.add_argument("--node-rank", type=int, default=0, help="Rank of this SGLang server node.")
    parser.add_argument(
        "--dist-init-addr",
        type=str,
        default=None,
        help="SGLang distributed init address, e.g. head_node_ip:20000.",
    )
    parser.add_argument(
        "--nccl-port",
        type=int,
        default=None,
        help="Optional NCCL init port passed to SGLang.",
    )
    parser.add_argument(
        "--concurrencies",
        type=str,
        default="1",
        help="Comma-separated list of client concurrency levels.",
    )
    parser.add_argument(
        "--questions-per-concurrency-base",
        type=int,
        default=10,
        help="num_questions = base * concurrency (default matches the sweep plan).",
    )
    parser.add_argument(
        "--max-questions-per-config",
        type=int,
        default=1024,
        help="Cap num_questions per (tp, concurrency) run (default: 1024).",
    )
    parser.add_argument(
        "--attention-backends",
        type=str,
        default="flashinfer",
        help="Comma-separated list. Will auto-skip fa3 unless SM90 (Hopper), and fa4 unless SM100+ (Blackwell).",
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this sweep.")
    if args.batch_requests and args.response_json:
        print(
            "Warning: --batch-requests returns one server-side batch at a time, "
            "so response JSON updates only after each batch completes. "
            "Omit --batch-requests for per-question completion writes."
        )

    server_env = os.environ.copy()
    if VENV_BIN.is_dir():
        server_env["PATH"] = f"{VENV_BIN}{os.pathsep}{server_env.get('PATH', '')}"

    concurrencies = [int(x) for x in args.concurrencies.split(",") if x.strip()]
    concurrencies = [c for c in concurrencies if c >= 1]
    if not concurrencies:
        raise RuntimeError("No concurrencies specified.")

    num_questions_by_conc = {
        c: min(int(args.questions_per_concurrency_base) * int(c), int(args.max_questions_per_config))
        for c in concurrencies
    }
    max_questions = max(num_questions_by_conc.values())
    max_concurrency = max(concurrencies)

    attention_backends = [s.strip() for s in args.attention_backends.split(",") if s.strip()]
    is_blackwell = _is_blackwell()
    device_sm = get_device_sm()
    if device_sm != 90:
        attention_backends = [b for b in attention_backends if b != "fa3"]
    if device_sm < 100:
        attention_backends = [b for b in attention_backends if b != "fa4"]
    attention_backends = attention_backends or ["flashinfer"]

    print(f"Loading dataset: {args.dataset_name}...")
    dataset = load_benchmark_dataset(args.dataset_name)
    required_questions = max_questions + max_concurrency
    
    if len(dataset) < required_questions:
         print(f"Warning: Dataset has {len(dataset)} items, but need up to {required_questions}. Reusing items.")

    tokenizer = AutoTokenizer.from_pretrained(args.target_model)
    max_input_tokens = args.max_input_tokens
    if max_input_tokens is None and args.context_length is not None:
        max_input_tokens = int(args.context_length)

    prompts: list[str] = []
    skipped_too_long = 0
    # Build prompts list
    for i in range(max(len(dataset), required_questions)):
        item = dataset[i % len(dataset)]
        user_content = item["turns"][0] # Extract the formatted turn
        
        # Apply chat template
        prompt_text = tokenizer.apply_chat_template(
            [{"role": "user", "content": user_content}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        if max_input_tokens is not None:
            prompt_len = len(tokenizer.encode(prompt_text, add_special_tokens=False))
            if prompt_len > max_input_tokens:
                skipped_too_long += 1
                continue
        prompts.append(prompt_text)
        if len(prompts) >= required_questions:
            break
    if len(prompts) < required_questions:
        raise RuntimeError(
            f"Only built {len(prompts)} prompts after filtering, but need {required_questions}. "
            f"Skipped {skipped_too_long} prompts longer than {max_input_tokens} tokens."
        )
    if skipped_too_long:
        print(
            f"Skipped {skipped_too_long} prompts longer than {max_input_tokens} tokens."
        )

    # Results indexed by (backend, concurrency) for baseline + dflash.
    # Removed TP dimension from keys since we aren't sweeping it.
    baseline_toks: dict[tuple[str, int], Optional[float]] = {}
    dflash_toks: dict[tuple[str, int], Optional[float]] = {}
    dflash_accept_len: dict[tuple[str, int], Optional[float]] = {}
    response_records: list[dict] = []
    _write_response_records(args.response_json, response_records)

    def persist_response(runner: str, backend: str, conc: int, record: dict) -> None:
        response_records.append(
            {
                "dataset": args.dataset_name,
                "runner": runner,
                "backend": backend,
                "concurrency": conc,
                "question_index": record["index"],
                "warmup": bool(record.get("warmup", False)),
                "response": record["response"],
                "meta_info": record["meta_info"],
            }
        )
        _write_response_records(args.response_json, response_records)
    
    tp = args.tp_size  # Fixed TP size

    for backend in attention_backends:
        port_base = find_available_port(20000)

        common_server_args: list[str] = [
            "--model-type",
            "llm",
            "--trust-remote-code",
            "--attention-backend",
            backend,
            "--tp-size",
            str(tp),
            "--nnodes",
            str(args.nnodes),
            "--node-rank",
            str(args.node_rank),
            "--dtype",
            str(args.dtype),
            "--mem-fraction-static",
            str(args.mem_fraction_static),
            "--max-running-requests",
            str(args.max_running_requests),
        ]
        if args.dist_init_addr is not None:
            common_server_args.extend(["--dist-init-addr", str(args.dist_init_addr)])
        if args.nccl_port is not None:
            common_server_args.extend(["--nccl-port", str(args.nccl_port)])
        common_server_args.extend(
            ["--cuda-graph-bs", *[str(i) for i in range(1, 33)], "--cuda-graph-max-bs", "32"]
        )
        if args.disable_radix_cache:
            common_server_args.append("--disable-radix-cache")
        if args.context_length is not None:
            common_server_args.extend(["--context-length", str(args.context_length)])

        if not args.skip_baseline:
            print(f"\n=== backend={backend} tp={tp} (baseline) ===")
            baseline_port = port_base
            baseline_url = f"http://127.0.0.1:{baseline_port}"
            baseline_proc = popen_launch_server(
                args.target_model,
                baseline_url,
                timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
                other_args=common_server_args,
                env=server_env,
            )
            try:
                # Warm up.
                _send_generate(
                    baseline_url,
                    "Hello",
                    max_new_tokens=8,
                    stop=[],
                    timeout_s=min(int(args.timeout_s), 300),
                )

                for conc in concurrencies:
                    n = num_questions_by_conc[conc]
                    _flush_cache(baseline_url)
                    print(
                        f"[warmup] run 1 warmup batch (size={conc}) after /flush_cache; excluded from metrics."
                    )
                    metrics = _run_bench_requests(
                        baseline_url,
                        prompts=prompts[: n + conc],
                        max_new_tokens=int(args.max_new_tokens),
                        concurrency=int(conc),
                        batch_requests=bool(args.batch_requests),
                        stop=[],
                        timeout_s=int(args.timeout_s),
                        expect_flashmtp=False,
                        on_response=lambda record, b=backend, c=conc: persist_response(
                            "baseline", b, c, record
                        ),
                    )
                    baseline_toks[(backend, conc)] = metrics.output_toks_per_s
                    print(
                        f"[baseline] conc={conc:>2} n={n:<4} "
                        f"toks/s={metrics.output_toks_per_s:,.2f} "
                        f"latency={metrics.latency_s:.1f}s "
                    )
            finally:
                kill_process_tree(baseline_proc.pid)
                try:
                    baseline_proc.wait(timeout=30)
                except Exception:
                    pass

        print(f"\n=== backend={backend} tp={tp} (FlashMTP v5.1 via SGLang DFLASH path) ===")
        dflash_port = find_available_port(port_base + 1)
        dflash_url = f"http://127.0.0.1:{dflash_port}"
        dflash_proc = popen_launch_server(
            args.target_model,
            dflash_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=[
                *common_server_args,
                "--speculative-algorithm",
                "DFLASH",
                "--speculative-draft-model-path",
                args.draft_model,
            ],
            env=server_env,
        )
        try:
            _send_generate(
                dflash_url,
                "Hello",
                max_new_tokens=8,
                stop=[],
                timeout_s=min(int(args.timeout_s), 300),
            )
            for conc in concurrencies:
                n = num_questions_by_conc[conc]
                _flush_cache(dflash_url)
                print(
                    f"[warmup] run 1 warmup batch (size={conc}) after /flush_cache; excluded from metrics."
                )

                metrics = _run_bench_requests(
                    dflash_url,
                    prompts=prompts[: n + conc],
                    max_new_tokens=int(args.max_new_tokens),
                    concurrency=int(conc),
                    batch_requests=bool(args.batch_requests),
                    stop=[],
                    timeout_s=int(args.timeout_s),
                    expect_flashmtp=True,
                    on_response=lambda record, b=backend, c=conc: persist_response(
                        "flashmtp", b, c, record
                    ),
                )
                dflash_toks[(backend, conc)] = metrics.output_toks_per_s
                dflash_accept_len[(backend, conc)] = metrics.spec_accept_length
                print(
                    f"[FlashMTP] conc={conc:>2} n={n:<4} "
                    f"toks/s={metrics.output_toks_per_s:,.2f} "
                    f"latency={metrics.latency_s:.1f}s "
                    f"accept_len={metrics.spec_accept_length:.3f} "
                    f"spec_verify_ct_sum={metrics.spec_verify_ct_sum}"
                )
        finally:
            kill_process_tree(dflash_proc.pid)
            try:
                dflash_proc.wait(timeout=30)
            except Exception:
                pass

    # Render markdown.
    md_lines: list[str] = []
    md_lines.append("# FlashMTP v5.1 SGLang Bench Report")
    md_lines.append("")
    md_lines.append("## Settings")
    md_lines.append(f"- dataset: `{args.dataset_name}`")
    md_lines.append(f"- target_model: `{args.target_model}`")
    md_lines.append(f"- draft_model: `{args.draft_model}`")
    md_lines.append(f"- max_new_tokens: `{args.max_new_tokens}`")
    md_lines.append(f"- attention_backends: `{', '.join(attention_backends)}`")
    md_lines.append(f"- tp_size: `{tp}`")
    md_lines.append(f"- nnodes: `{args.nnodes}`")
    md_lines.append(f"- node_rank: `{args.node_rank}`")
    md_lines.append(f"- dist_init_addr: `{args.dist_init_addr}`")
    md_lines.append(f"- concurrencies: `{', '.join(str(x) for x in concurrencies)}`")
    md_lines.append(f"- questions_per_concurrency: `base={args.questions_per_concurrency_base}`")
    md_lines.append(f"- device_sm: `{device_sm}`")
    md_lines.append(f"- is_blackwell: `{is_blackwell}`")
    md_lines.append(f"- skip_baseline: `{bool(args.skip_baseline)}`")
    md_lines.append("- drop_first_batch: `true`")
    md_lines.append("")

    for backend in attention_backends:
        md_lines.append(f"## Backend: `{backend}`")
        md_lines.append("")

        baseline_values = {
            c: baseline_toks.get((backend, c), None) for c in concurrencies
        }
        dflash_values = {
            c: dflash_toks.get((backend, c), None) for c in concurrencies
        }
        speedup_values: dict[int, Optional[float]] = {}
        for c in concurrencies:
            b = baseline_values.get(c, None)
            d = dflash_values.get(c, None)
            speedup_values[c] = None if (b is None or d is None or b <= 0) else (d / b)

        md_lines.append("### Baseline output tok/s")
        md_lines.append(
            _format_table(
                concurrencies=concurrencies,
                values=baseline_values,
                float_fmt=",.2f",
            )
        )
        md_lines.append("")
        
        md_lines.append("### FlashMTP output tok/s")
        md_lines.append(
            _format_table(
                concurrencies=concurrencies,
                values=dflash_values,
                float_fmt=",.2f",
            )
        )
        md_lines.append("")

        md_lines.append("### Speedup (FlashMTP / baseline)")
        md_lines.append(
            _format_table(
                concurrencies=concurrencies,
                values=speedup_values,
                float_fmt=".3f",
            )
        )
        md_lines.append("")

        md_lines.append("### FlashMTP acceptance length")
        md_lines.append(
            _format_table(
                concurrencies=concurrencies,
                values={
                    c: dflash_accept_len.get((backend, c), None)
                    for c in concurrencies
                },
                float_fmt=".3f",
            )
        )
        md_lines.append("")

    if args.output_md:
        with open(args.output_md, "w", encoding="utf-8") as f:
            f.write("\n".join(md_lines))
            f.write("\n")
        print(f"\nWrote markdown report to: {args.output_md}")
    else:
        print("\nMarkdown report disabled (pass --output-md to write one).")

    if args.response_json:
        _write_response_records(args.response_json, response_records)
        print(f"Wrote per-question responses to: {args.response_json}")


if __name__ == "__main__":
    main()
