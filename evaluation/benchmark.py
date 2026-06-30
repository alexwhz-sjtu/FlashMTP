import argparse
import json
import random
import re
import sys
import time
from collections import defaultdict
from itertools import chain
from pathlib import Path
from types import SimpleNamespace
from loguru import logger
import numpy as np
import torch
from rich import print
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from specforge.modeling.draft.flashmtp import FlashMTPDraftModel
from specforge.modeling.draft.flashmtp import sample

from evaluation import distributed as dist
from evaluation.utils import load_and_process_dataset

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


LONGBENCH_V2_ALIAS_LENGTH_BUCKETS: dict[str, tuple[int, int]] = {
    "longbench_v2_32k_64k": (32000, 64000),
}


def parse_longbench_v2_length_bucket(
    dataset_path: Path, dataset_alias: str = ""
) -> tuple[int, int] | None:
    """Parse ``longbench_v2_{max}_{min}`` shard names into ``(min_len, max_len)`` token bounds."""
    alias = dataset_alias.strip().lower()
    if alias in LONGBENCH_V2_ALIAS_LENGTH_BUCKETS:
        return LONGBENCH_V2_ALIAS_LENGTH_BUCKETS[alias]

    for name in (alias, *dataset_path.parts):
        pl = str(name).lower()
        m = re.fullmatch(r"longbench_v2_(\d+)_(\d+)(?:_.*)?", pl)
        if m:
            upper, lower = int(m.group(1)), int(m.group(2))
            return min(lower, upper), max(lower, upper)
    return None


def is_longbench_v2_dataset_path(dataset_path: Path) -> bool:
    """Match LongBench v2 shards: ``longbench_v2`` or ``longbench_v2_*`` (e.g. ``64000_32000`` = 32k–64k)."""
    try:
        resolved = dataset_path.resolve()
    except OSError:
        resolved = dataset_path
    for part in resolved.parts:
        pl = part.lower()
        if pl == "longbench_v2" or pl.startswith("longbench_v2_"):
            return True
    return False


def _filter_longbench_v2_by_length(
    data: list, min_len: int, max_len: int, dataset_path: Path
) -> list:
    filtered = [
        item
        for item in data
        if isinstance(item, dict)
        and (
            item.get("length") is None
            or min_len <= int(item["length"]) <= max_len
        )
    ]
    if not filtered and data:
        raise ValueError(
            f"No LongBench v2 samples in length range [{min_len}, {max_len}] "
            f"from {dataset_path} (had {len(data)} records)"
        )
    return filtered


def load_longbench_v2_json_records(
    data: list, dataset_path: Path, dataset_alias: str = ""
) -> list[dict]:
    if not isinstance(data, list):
        raise ValueError(f"{dataset_path} must contain a JSON list")
    bucket = parse_longbench_v2_length_bucket(dataset_path, dataset_alias)
    if bucket is not None:
        min_len, max_len = bucket
        data = _filter_longbench_v2_by_length(data, min_len, max_len, dataset_path)
    return [{"turns": [format_longbench_v2_prompt(item)]} for item in data]


def load_longbench_v2_shard_directory(
    shard_dir: Path, dataset_alias: str = ""
) -> list[dict]:
    json_files = sorted(p for p in shard_dir.glob("*.json") if p.is_file())
    if not json_files:
        raise FileNotFoundError(f"No JSON files under LongBench v2 shard dir: {shard_dir}")
    instances: list[dict] = []
    for json_file in json_files:
        with json_file.open("r", encoding="utf-8") as f:
            data = json.load(f)
        instances.extend(
            load_longbench_v2_json_records(data, json_file, dataset_alias)
        )
    if not instances:
        raise ValueError(f"No LongBench v2 samples loaded from shard dir: {shard_dir}")
    return instances


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
                    "category": str(obj.get("category", "unknown")),
                    "question_id": obj.get("question_id"),
                }
            )
    if not instances:
        hint = f" for category={category!r}" if category else " (all categories)"
        raise ValueError(f"No Spec-Bench samples{hint} in {dataset_path}")
    return instances


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
    if dataset_path.is_dir():
        if (
            original_dataset_name.lower() == "longbench_v2"
            or is_longbench_v2_dataset_path(dataset_path)
            or original_dataset_name.lower().startswith("longbench_v2_")
        ):
            return load_longbench_v2_shard_directory(dataset_path, original_dataset_name)
        raise ValueError(f"Unsupported dataset directory: {dataset_path}")

    if dataset_path.is_file() and dataset_path.suffix == ".json":
        with dataset_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, list):
            raise ValueError(f"{dataset_path} must contain a JSON list")

        if (
            original_dataset_name.lower() == "longbench_v2"
            or is_longbench_v2_dataset_path(dataset_path)
            or original_dataset_name.lower().startswith("longbench_v2_")
        ):
            return load_longbench_v2_json_records(
                data, dataset_path, original_dataset_name
            )

        if should_load_as_multifieldqa_en_mixup(data, dataset_path, original_dataset_name):
            return load_multifieldqa_en_mixup_json_records(data, dataset_path)

        if is_swe_bench_style_json(data, dataset_path, original_dataset_name):
            return load_swe_bench_json_instances(data, dataset_path)

        raise ValueError(f"Unsupported JSON dataset: {dataset_path}")

    if dataset_path.is_file() and dataset_path.suffix == ".jsonl":
        task_name = infer_infinitebench_task(original_dataset_name, dataset_path)
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
                    or original_dataset_name.lower().startswith("longbench_v2_")
                ):
                    return load_longbench_v2_json_records(
                        data, dataset_path, original_dataset_name
                    )
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


def cuda_time() -> float:
    torch.cuda.synchronize()
    return time.perf_counter()


def decode_weight(run) -> int:
    """Token count for amortized decode metrics (matches dflash benchmark)."""
    n = int(getattr(run, "num_tokens_for_decode_rate", run.num_output_tokens))
    return int(run.batch_size) * max(n, 1)


def decode_wall_seconds(run) -> float:
    return float(run.time_per_output_token) * decode_weight(run)


def compute_benchmark_stats(responses: list, block_size: int) -> dict:
    """Aggregate decode speedup and acceptance length over benchmark turn responses."""
    if not responses:
        return {
            "num_turns": 0,
            "token_weighted_speedup": 0.0,
            "throughput_ratio": 0.0,
            "baseline_s_per_token": 0.0,
            "flashmtp_s_per_token": 0.0,
            "unweighted_speedup": 0.0,
            "avg_accept_length": 0.0,
            "histogram": [0.0] * (block_size + 1),
        }

    w1 = sum(decode_weight(r[1]) for r in responses)
    wb = sum(decode_weight(r[block_size]) for r in responses)
    d1 = sum(decode_wall_seconds(r[1]) for r in responses)
    db = sum(decode_wall_seconds(r[block_size]) for r in responses)
    t1 = d1 / max(w1, 1)
    tb = db / max(wb, 1)
    throughput1 = w1 / max(d1, 1e-30)
    throughputb = wb / max(db, 1e-30)
    t1_unweighted = float(np.mean([r[1].time_per_output_token for r in responses]))
    tb_unweighted = float(np.mean([r[block_size].time_per_output_token for r in responses]))

    acceptance_lengths = list(chain(*[r[block_size].acceptance_lengths for r in responses]))
    if acceptance_lengths:
        histogram = [
            acceptance_lengths.count(b) / len(acceptance_lengths) for b in range(block_size + 1)
        ]
        avg_accept = float(sum(index * prob for index, prob in enumerate(histogram)))
    else:
        histogram = [0.0] * (block_size + 1)
        avg_accept = 0.0

    return {
        "num_turns": len(responses),
        "token_weighted_speedup": t1 / max(tb, 1e-30),
        "throughput_ratio": throughputb / max(throughput1, 1e-30),
        "baseline_s_per_token": t1,
        "flashmtp_s_per_token": tb,
        "unweighted_speedup": t1_unweighted / max(tb_unweighted, 1e-30),
        "avg_accept_length": avg_accept,
        "histogram": histogram,
    }


def group_responses_by_category(responses: list) -> dict[str, list]:
    groups: dict[str, list] = defaultdict(list)
    for response in responses:
        groups[str(response.get("category", "unknown"))].append(response)
    return dict(groups)


def print_benchmark_stats(stats: dict, block_size: int, title: str) -> None:
    print(f"\n=== {title} ===")
    print(f"  turns: {stats['num_turns']}")
    print(
        f"  token-weighted speedup: {stats['token_weighted_speedup']:.2f}x | "
        f"throughput ratio: {stats['throughput_ratio']:.2f}x | "
        f"unweighted speedup: {stats['unweighted_speedup']:.2f}x"
    )
    print(
        f"  decode s/token baseline={stats['baseline_s_per_token']:.6f} "
        f"flashmtp={stats['flashmtp_s_per_token']:.6f}"
    )
    histogram = stats["histogram"]
    hist_text = ", ".join(f"{x * 100:.1f}%" for x in histogram)
    print(f"  acceptance length histogram: [{hist_text}]")
    print(f"  average acceptance length: {stats['avg_accept_length']:.2f}")


@torch.inference_mode()
def run_benchmark_warmup(
    target: AutoModelForCausalLM,
    draft_model: FlashMTPDraftModel,
    tokenizer: AutoTokenizer,
    block_size: int,
    device: torch.device,
    batch_size: int,
    max_new_tokens: int,
    temperature: float,
    stop_token_ids: list[int],
) -> None:
    """One short baseline + speculative pass to warm CUDA kernels before timing."""
    warmup_prompt = [{"role": "user", "content": "Warmup."}]
    input_text = tokenizer.apply_chat_template(
        warmup_prompt, tokenize=False, add_generation_prompt=True, enable_thinking=False
    )
    input_ids = tokenizer.encode(input_text, return_tensors="pt").to(device)
    if batch_size > 1:
        input_ids = input_ids.expand(batch_size, -1).contiguous()
    warmup_new_tokens = min(16, max_new_tokens)
    target_generate(
        target=target,
        input_ids=input_ids,
        max_new_tokens=warmup_new_tokens,
        stop_token_ids=stop_token_ids,
        temperature=temperature,
        decode_timing_after_first_token=False,
    )
    flashmtp_generate(
        model=draft_model,
        target=target,
        input_ids=input_ids,
        max_new_tokens=warmup_new_tokens,
        block_size=block_size,
        stop_token_ids=stop_token_ids,
        temperature=temperature,
        decode_timing_after_first_token=False,
    )
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    cuda_time()


@torch.inference_mode()
def target_generate(
    target: AutoModelForCausalLM,
    input_ids: torch.Tensor,
    max_new_tokens: int,
    stop_token_ids: list[int],
    temperature: float = 0.0,
    decode_timing_after_first_token: bool = False,
) -> SimpleNamespace:
    batch_size_dim = input_ids.shape[0]
    num_input_tokens = input_ids.shape[1]
    max_length = num_input_tokens + max_new_tokens
    output_ids = torch.empty(
        (batch_size_dim, max_length), dtype=torch.long, device=input_ids.device
    )
    output_ids[:, :num_input_tokens] = input_ids
    position_ids = torch.arange(max_length, device=input_ids.device).unsqueeze(0).expand(
        batch_size_dim, -1
    )
    past_key_values_target = DynamicCache()
    stop_tensor = (
        torch.tensor(stop_token_ids, device=input_ids.device, dtype=torch.long)
        if stop_token_ids
        else None
    )

    prefill_start = cuda_time()
    output = target(
        input_ids,
        position_ids=position_ids[:, :num_input_tokens],
        past_key_values=past_key_values_target,
        use_cache=True,
        logits_to_keep=1,
        output_hidden_states=False,
    )
    next_token = sample(output.logits, temperature)
    time_to_first_token = (cuda_time() - prefill_start) / batch_size_dim

    decode_start: float | None = None if decode_timing_after_first_token else cuda_time()
    start = input_ids.shape[1]
    while start < max_length:
        output_ids[:, start : start + 1] = next_token
        start += 1

        if decode_timing_after_first_token and decode_start is None:
            decode_start = cuda_time()

        if stop_tensor is not None and torch.all(
            torch.isin(next_token.squeeze(-1), stop_tensor)
        ):
            break
        if start >= max_length:
            break

        token_position_ids = position_ids[:, start - 1 : start]
        output = target(
            next_token,
            position_ids=token_position_ids,
            past_key_values=past_key_values_target,
            use_cache=True,
            logits_to_keep=1,
            output_hidden_states=False,
        )
        next_token = sample(output.logits, temperature)

    output_ids = output_ids[:, :start]

    num_output_tokens = output_ids.shape[1] - num_input_tokens
    if decode_start is None:
        decode_start = cuda_time()
    total_decode_time = cuda_time() - decode_start
    rate_tokens = max(num_output_tokens - (1 if decode_timing_after_first_token else 0), 1)
    time_per_output_token = total_decode_time / (batch_size_dim * rate_tokens)
    throughput_tokens_per_sec = (batch_size_dim * rate_tokens) / max(total_decode_time, 1e-9)

    return SimpleNamespace(
        output_ids=output_ids,
        num_input_tokens=num_input_tokens,
        num_output_tokens=num_output_tokens,
        num_tokens_for_decode_rate=rate_tokens,
        batch_size=batch_size_dim,
        time_to_first_token=time_to_first_token,
        time_per_output_token=time_per_output_token,
        throughput_tokens_per_sec=float(throughput_tokens_per_sec),
        decode_wall_time=total_decode_time,
        acceptance_lengths=[1] * num_output_tokens,
    )


@torch.inference_mode()
def flashmtp_generate(
    model: FlashMTPDraftModel,
    target: AutoModelForCausalLM,
    input_ids: torch.Tensor,
    max_new_tokens: int,
    block_size: int,
    stop_token_ids: list[int],
    temperature: float = 0.0,
    decode_timing_after_first_token: bool = False,
    *,
    use_draft_tree: bool = False,
    draft_tree_trunc_thres: float = 0.2,
    draft_tree_expand_thres: float = 0.5,
    draft_tree_width: int = 4,
    draft_tree_entropy_ratio: float = 0.4,
) -> SimpleNamespace:
    original_block_size = model.block_size
    model.block_size = block_size
    try:
        if use_draft_tree:
            output_ids = model.spec_generate_with_draft_tree(
                target=target,
                input_ids=input_ids,
                max_new_tokens=max_new_tokens,
                stop_token_ids=stop_token_ids,
                temperature=temperature,
                trunc_thres=draft_tree_trunc_thres,
                expand_thres=draft_tree_expand_thres,
                tree_width=draft_tree_width,
                entropy_ratio=draft_tree_entropy_ratio,
                decode_timing_after_first_token=decode_timing_after_first_token,
            )
        else:
            output_ids = model.spec_generate(
                target=target,
                input_ids=input_ids,
                max_new_tokens=max_new_tokens,
                stop_token_ids=stop_token_ids,
                temperature=temperature,
                decode_timing_after_first_token=decode_timing_after_first_token,
            )
    finally:
        model.block_size = original_block_size

    stats = model.get_last_decode_stats()
    bsz = int(input_ids.shape[0])
    num_input_tokens = input_ids.shape[1]
    num_output_tokens = output_ids.shape[1] - num_input_tokens
    decode_wall_time = float(stats.get("decode_wall_time", 0.0))
    rate_tokens = max(num_output_tokens - (1 if decode_timing_after_first_token else 0), 1)
    time_per_output_token = decode_wall_time / (bsz * max(rate_tokens, 1))
    throughput_tokens_per_sec = (bsz * rate_tokens) / max(decode_wall_time, 1e-9)

    return SimpleNamespace(
        output_ids=output_ids,
        num_input_tokens=num_input_tokens,
        num_output_tokens=num_output_tokens,
        num_tokens_for_decode_rate=rate_tokens,
        batch_size=bsz,
        time_to_first_token=0.0,
        time_per_output_token=time_per_output_token,
        throughput_tokens_per_sec=float(throughput_tokens_per_sec),
        decode_wall_time=decode_wall_time,
        acceptance_lengths=stats.get("accept_lengths", []),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name-or-path", type=str, default='/data/wanghanzhen/models/Qwen/Qwen3-8B')
    parser.add_argument("--draft-name-or-path", type=str, default='/data/wanghanzhen/Projects/MTP/NIPS26/FlashMTP_v5.1/cache/models/flashmtp_v5.1_fix_h100_sample_40000_think_off_nlayers5_block_16_maxlen4096_epochs6/epoch_6_step_29844')
    parser.add_argument("--block-size", type=int, default=None)
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--max-samples", type=int, default=10)
    parser.add_argument("--max-new-tokens", type=int, default=4096)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Replicate each prompt along the batch dimension (expand) for throughput runs. "
        "Use temperature=0 so FlashMTP speculative steps stay aligned across the batch.",
    )
    parser.add_argument(
        "--sink-num",
        type=int,
        default=None,
        help="Optional legacy override; ignored if checkpoint has no sink_num.",
    )
    parser.add_argument(
        "--local-position",
        type=str,
        default=None,
        choices=("true", "false"),
        help="Override draft local_position: true = CHS rope ids 0, draft block ids 1..block_size; "
        "false = global positions. Default: use checkpoint flashmtp_config.",
    )
    parser.add_argument(
        "--use-draft-tree",
        action="store_true",
        help="Use FlashMTPDraftModel.spec_generate_with_draft_tree (block_size from model config).",
    )
    parser.add_argument(
        "--draft-tree-trunc-thres",
        type=float,
        default=0.2,
        help="Truncate spine when two consecutive top1 probs are below this.",
    )
    parser.add_argument(
        "--draft-tree-expand-thres",
        type=float,
        default=0.5,
        help="Top1 prob threshold for the first expandable depth.",
    )
    parser.add_argument(
        "--draft-tree-width",
        type=int,
        default=4,
        help="Fixed branch width w (includes top1) after expansion starts.",
    )
    parser.add_argument(
        "--draft-tree-entropy-ratio",
        type=float,
        default=0.4,
        help="Require top-w normalized entropy / log(w) > this to start expansion.",
    )
    args = parser.parse_args()

    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    dist.init()
    torch.cuda.set_device(dist.local_rank())
    device = torch.device(f"cuda:{dist.local_rank()}")

    def has_flash_attn():
        try:
            import flash_attn
            return True
        except ImportError:
            logger.warning("flash_attn is not installed. Falling back to torch.sdpa. The speedup will be lower.")
            return False

    installed_flash_attn = has_flash_attn()
    print(f"flash attention installed: {installed_flash_attn}")

    target = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        attn_implementation="flash_attention_2" if installed_flash_attn else "sdpa",
        dtype=torch.bfloat16,
    ).to(device).eval()

    draft_model = FlashMTPDraftModel.from_pretrained(
        args.draft_name_or_path,
        attn_implementation="flash_attention_2" if installed_flash_attn else "sdpa",
        dtype=torch.bfloat16,
    ).to(device).eval()

    fcfg = getattr(draft_model.config, "flashmtp_config", None) or {}
    if args.sink_num is not None and fcfg.get("sink_num") is not None:
        eff_sink = args.sink_num
        if draft_model.config.flashmtp_config is None:
            draft_model.config.flashmtp_config = {}
        draft_model.config.flashmtp_config["sink_num"] = eff_sink
        if hasattr(draft_model, "sink_num"):
            draft_model.sink_num = int(eff_sink)
        logger.info(f"Overriding sink_num={eff_sink} (legacy)")
    if args.local_position is not None:
        lp = args.local_position == "true"
        draft_model.local_position = lp
        if draft_model.config.flashmtp_config is None:
            draft_model.config.flashmtp_config = {}
        draft_model.config.flashmtp_config["local_position"] = lp
        logger.info(f"Overriding local_position={lp} (from --local-position {args.local_position})")
    if fcfg.get("sink_num") is not None and hasattr(draft_model, "sink_num"):
        logger.info(
            f"FlashMTP draft (legacy): sink_num={draft_model.sink_num}, "
            f"block_size={draft_model.block_size}"
        )
    logger.info(
        f"FlashMTP draft: pivot_fuse_mode={getattr(draft_model, 'pivot_fuse_mode', fcfg.get('pivot_fuse_mode'))}, "
        f"num_middle_layers_n={fcfg.get('num_middle_layers_n', 'n/a')}, "
        f"target_layer_ids={getattr(draft_model, 'target_layer_ids', None)}, "
        f"train_lm_head={fcfg.get('train_lm_head', getattr(draft_model, 'train_lm_head', False))}, "
        f"local_position={getattr(draft_model, 'local_position', fcfg.get('local_position', False))}, "
        f"block_size={draft_model.block_size}"
    )
    block_size = args.block_size if args.block_size is not None else draft_model.block_size
    if args.use_draft_tree:
        logger.info(
            "Draft tree decode: trunc_thres={} expand_thres={} width={} entropy_ratio={} "
            "(block_size={} from model)",
            args.draft_tree_trunc_thres,
            args.draft_tree_expand_thres,
            args.draft_tree_width,
            args.draft_tree_entropy_ratio,
            draft_model.block_size,
        )

    if args.batch_size < 1:
        raise ValueError("--batch-size must be >= 1")
    if args.batch_size > 1 and args.temperature > 1e-5:
        logger.warning(
            "batch_size>1 with temperature>0 can desynchronize FlashMTP across batch rows; "
            "prefer --temperature 0 for batched benchmarking."
        )

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    stop_token_ids = [token_id for token_id in [tokenizer.eos_token_id] if token_id is not None]
    dataset = load_benchmark_dataset(args.dataset)
    dataset = select_max_samples(dataset, args.max_samples)

    if dist.is_main():
        print("Running CUDA warmup (baseline + FlashMTP, short decode)...")
    run_benchmark_warmup(
        target=target,
        draft_model=draft_model,
        tokenizer=tokenizer,
        block_size=block_size,
        device=device,
        batch_size=args.batch_size,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        stop_token_ids=stop_token_ids,
    )

    benchmark_start = cuda_time()
    responses = []
    indices = range(dist.rank(), len(dataset), dist.size())
    for idx in tqdm(indices, disable=not dist.is_main()):
        instance = dataset[idx]
        messages = []
        chain_turns = bool(instance.get("specbench_chain_turns"))
        decode_after_first = chain_turns
        sample_category = instance.get("category")
        prev_assistant = ""
        for turn_index, turn_q in enumerate(instance["turns"]):
            if chain_turns:
                user_content = turn_q if turn_index == 0 else f"{prev_assistant}\n\n{turn_q}"
            else:
                user_content = turn_q
            messages.append({"role": "user", "content": user_content})
            input_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=True)
            input_ids = tokenizer.encode(input_text, return_tensors="pt").to(target.device)
            if args.batch_size > 1:
                input_ids = input_ids.expand(args.batch_size, -1).contiguous()
            category_suffix = f" | category={sample_category}" if sample_category else ""
            print(
                f"\n[Sample {idx} | Turn {turn_index}{category_suffix}] Input length: "
                f"{input_ids.shape[1]} tokens ({len(user_content)} chars), "
                f"batch_size={input_ids.shape[0]}"
            )

            response = {}
            response[1] = target_generate(
                target=target,
                input_ids=input_ids,
                max_new_tokens=args.max_new_tokens,
                stop_token_ids=stop_token_ids,
                temperature=args.temperature,
                decode_timing_after_first_token=decode_after_first,
            )
            response[block_size] = flashmtp_generate(
                model=draft_model,
                target=target,
                input_ids=input_ids,
                max_new_tokens=args.max_new_tokens,
                block_size=block_size,
                stop_token_ids=stop_token_ids,
                temperature=args.temperature,
                decode_timing_after_first_token=decode_after_first,
                use_draft_tree=args.use_draft_tree,
                draft_tree_trunc_thres=args.draft_tree_trunc_thres,
                draft_tree_expand_thres=args.draft_tree_expand_thres,
                draft_tree_width=args.draft_tree_width,
                draft_tree_entropy_ratio=args.draft_tree_entropy_ratio,
            )
            
            spec_response = response[block_size]
            generated_ids = spec_response.output_ids[0, spec_response.num_input_tokens:]
            output_text = tokenizer.decode(generated_ids, skip_special_tokens=True)

            acceptance_lengths_text = ", ".join(
                [f"{position}:{length}" for position, length in enumerate(spec_response.acceptance_lengths)]
            )
            avg_acceptance_length = np.mean(spec_response.acceptance_lengths)
            print(f"\n[Sample {idx} | Turn {turn_index}] Response:\n{output_text}")
            print(
                f"[Sample {idx} | Turn {turn_index}] Decode timing "
                f"(cuda wall after prefill, amortized s/token, batch={args.batch_size}): "
                f"baseline={response[1].time_per_output_token:.6f}, "
                f"flashmtp={spec_response.time_per_output_token:.6f} | "
                f"tok/s (batch total): baseline={response[1].throughput_tokens_per_sec:.2f}, "
                f"flashmtp={spec_response.throughput_tokens_per_sec:.2f}"
            )
            print(
                f"[Sample {idx} | Turn {turn_index}] Acceptance lengths (position:length): "
                f"{acceptance_lengths_text}"
            )
            print(f"[Sample {idx} | Turn {turn_index}] Average acceptance length: {avg_acceptance_length:.2f}")

            messages.append({"role": "assistant", "content": output_text})
            if chain_turns:
                prev_assistant = output_text
            if sample_category is not None:
                response["category"] = sample_category
            if instance.get("question_id") is not None:
                response["question_id"] = instance["question_id"]
            responses.append(response)

    if dist.size() > 1:
        responses = dist.gather(responses, dst=0)
        if not dist.is_main():
            return
        responses = list(chain(*responses))

    overall_stats = compute_benchmark_stats(responses, block_size)
    print_benchmark_stats(
        overall_stats,
        block_size,
        title=f"Overall (batch_size={args.batch_size})",
    )

    category_groups = group_responses_by_category(responses)
    if len(category_groups) > 1 or (
        len(category_groups) == 1 and "unknown" not in category_groups
    ):
        print("\n=== Per-category breakdown ===")
        for category in sorted(category_groups):
            cat_stats = compute_benchmark_stats(category_groups[category], block_size)
            print_benchmark_stats(cat_stats, block_size, title=f"category={category}")

    total_elapsed_time = cuda_time() - benchmark_start
    print(f"Total elapsed time: {total_elapsed_time:.2f}s")

if __name__ == "__main__":
    main()
