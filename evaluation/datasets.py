"""Benchmark dataset loading: normalize all sources to ``{"turns": [...], ...}`` instances."""

from __future__ import annotations

import json
import random
import re
from pathlib import Path
from typing import Callable

from evaluation.utils import load_and_process_dataset

DATASET_PATH_FILE = Path(__file__).resolve().with_name("dataset_path.json")

# ---------------------------------------------------------------------------
# Shared instance helpers
# ---------------------------------------------------------------------------

Instance = dict  # keys: turns (list[str]); optional specbench_chain_turns (bool)


def single_turn(prompt: str, **extra) -> Instance:
    return {"turns": [prompt], **extra}


def multi_turn(turns: list[str], **extra) -> Instance:
    return {"turns": turns, **extra}


def prompts_to_instances(prompts: list[str], **extra) -> list[Instance]:
    return [single_turn(p, **extra) for p in prompts]


def rows_to_instances(
    rows: list[dict],
    formatter: Callable[[dict], str],
    *,
    path: Path | None = None,
) -> list[Instance]:
    out: list[Instance] = []
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            label = f"{path} index {index}" if path else f"index {index}"
            raise ValueError(f"{label}: expected object, got {type(row)}")
        out.append(single_turn(formatter(row)))
    return out


def select_max_samples(dataset, max_samples: int | None):
    if max_samples is None or len(dataset) <= max_samples:
        return dataset
    if hasattr(dataset, "shuffle") and hasattr(dataset, "select"):
        return dataset.shuffle(seed=0).select(range(max_samples))
    indices = list(range(len(dataset)))
    rng = random.Random(0)
    rng.shuffle(indices)
    return [dataset[i] for i in indices[:max_samples]]


def resolve_dataset_path(dataset_name: str) -> str:
    if not DATASET_PATH_FILE.is_file():
        return dataset_name
    with DATASET_PATH_FILE.open("r", encoding="utf-8") as f:
        dataset_paths = json.load(f)
    if not isinstance(dataset_paths, dict):
        raise ValueError(f"{DATASET_PATH_FILE} must contain a JSON object")
    return dataset_paths.get(dataset_name, dataset_name)


def _read_json_list(path: Path) -> list:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"{path} must contain a JSON list")
    return data


def _path_has_marker(path: Path, markers: tuple[str, ...]) -> bool:
    try:
        resolved = path.resolve()
    except OSError:
        resolved = path
    stem = resolved.stem.lower()
    if any(m in stem for m in markers):
        return True
    return any(any(m in part.lower() for m in markers) for part in resolved.parts)


# ---------------------------------------------------------------------------
# LongBench v2
# ---------------------------------------------------------------------------

def _is_longbench_v2(path: Path) -> bool:
    try:
        resolved = path.resolve()
    except OSError:
        resolved = path
    for part in resolved.parts:
        pl = part.lower()
        if pl == "longbench_v2" or pl.startswith("longbench_v2_"):
            return True
    return False


def _format_longbench_v2(row: dict) -> str:
    if "context" not in row:
        raise ValueError("Missing 'context' in LongBench_v2 item")
    if "question" not in row:
        raise ValueError("Missing 'question' in LongBench_v2 item")
    return f"{row['context']}\n\nQuestion: {row['question']}"


def _load_longbench_v2(rows: list, path: Path) -> list[Instance]:
    return rows_to_instances(rows, _format_longbench_v2, path=path)


# ---------------------------------------------------------------------------
# LVEval mixup (context + input)
# ---------------------------------------------------------------------------

_MIXUP_MARKERS = (
    "multifieldqa_en_mixup",
    "lic_mixup",
    "hotpotwikiqa_mixup",
)
_MIXUP_ALIASES = frozenset(
    {
        "multifieldqa_en_20k_40k",
        "multifieldqa_en_mixup",
        "multifieldqa_en_mixup_32k_ctx_20k_40k",
        "lic_mixup",
        "lic_mixup_32k_ctx_20k_40k",
        "hotpotwikiqa_mixup",
        "hotpotwikiqa_mixup_16k_ctx_20k_40k",
    }
)


def _format_context_input(row: dict) -> str:
    if "context" not in row:
        raise ValueError("Missing 'context' field")
    if "input" not in row:
        raise ValueError("Missing 'input' field")
    ctx, inp = row["context"], row["input"]
    return f"{ctx if isinstance(ctx, str) else str(ctx)}\n\n{inp if isinstance(inp, str) else str(inp)}"


def _is_mixup(rows: list, path: Path, alias: str) -> bool:
    if not rows or not isinstance(rows[0], dict):
        return False
    row0 = rows[0]
    if not isinstance(row0.get("input"), str) or not isinstance(row0.get("context"), str):
        return False
    key = alias.lower()
    if key in _MIXUP_ALIASES or Path(alias).stem.lower() in _MIXUP_ALIASES:
        return True
    return _path_has_marker(path, _MIXUP_MARKERS)


# ---------------------------------------------------------------------------
# SWE-bench JSON export
# ---------------------------------------------------------------------------

def _is_swe_bench(rows: list, path: Path, alias: str) -> bool:
    if not rows or not isinstance(rows[0], dict):
        return False
    row0 = rows[0]
    if "text" not in row0 or not isinstance(row0.get("text"), str):
        return False
    stem = path.stem.lower()
    alias_stem = Path(str(alias)).stem.lower()
    if stem.startswith("swe_bench") or alias_stem.startswith("swe_bench"):
        return True
    return "instance_id" in row0


def _format_swe_bench(row: dict) -> str:
    if "text" not in row or row["text"] is None:
        raise ValueError("Missing 'text' in SWE-bench item")
    text = row["text"]
    return text if isinstance(text, str) else str(text)


# ---------------------------------------------------------------------------
# InfiniteBench (jsonl, context + input + task template)
# ---------------------------------------------------------------------------

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


def _infer_infinitebench_task(alias: str, path: Path) -> str | None:
    for candidate in (alias, path.stem):
        if candidate in INFINITEBENCH_PROMPTS:
            return candidate
    return None


def _format_infinitebench(row: dict, task_name: str) -> str:
    template = INFINITEBENCH_PROMPTS[task_name]
    fields = {
        "context": row["context"],
        "input": row.get("input", ""),
        "question": row.get("input", ""),
    }
    options = row.get("options") or []
    for i, name in enumerate(["OPTION_A", "OPTION_B", "OPTION_C", "OPTION_D"]):
        if i < len(options):
            fields[name] = options[i]
    if task_name == "math_find":
        find_result = re.findall(r"The .+ of", row["input"])
        if not find_result:
            raise ValueError(f"Cannot infer math_find target from input: {row['input']}")
        fields["prefix"] = f"What is {find_result[0].lower()[:-3]} in the following list?"
    if task_name == "code_run":
        find_result = re.findall(r"func_[0-9]+\(-?[0-9]+\)", row["input"])
        if not find_result:
            raise ValueError(f"Cannot infer code_run function call from input: {row['input']}")
        fields["func_call"] = find_result[0]
        fields["func"] = fields["func_call"].split("(")[0]
    return template.format(**fields)


def _format_context_question(row: dict) -> str:
    return f"{row['context']}\nQuestion: {row['input']}"


# ---------------------------------------------------------------------------
# Spec-Bench
# ---------------------------------------------------------------------------

def _default_specbench_path() -> Path:
    return Path(__file__).resolve().parents[2] / "Spec-Bench/data/spec_bench/question.jsonl"


def _specbench_meta(alias: str) -> tuple[bool, str | None]:
    s = alias.strip().lower()
    if s == "specbench":
        return True, None
    if s.startswith("specbench_"):
        cat = s[len("specbench_") :].strip("_")
        return True, cat if cat else None
    return False, None


def _expand_specbench_turn(turn: str) -> list[str]:
    turn = turn.strip()
    if not turn:
        return []
    if "|||" in turn:
        return [p.strip() for p in turn.split("|||") if p.strip()]
    if ", " in turn:
        parts = [p.strip() for p in turn.split(", ") if p.strip()]
        if len(parts) >= 2 and all(p.endswith("?") for p in parts):
            return parts
    return [turn]


def _flatten_specbench_turns(turns: list) -> list[str]:
    out: list[str] = []
    for t in turns:
        out.extend(_expand_specbench_turn(str(t) if not isinstance(t, str) else t))
    return out


def _load_specbench(path: Path, alias: str) -> list[Instance]:
    is_sb, category = _specbench_meta(alias)
    if not is_sb:
        raise ValueError("internal: _load_specbench requires a specbench alias")
    if not path.is_file():
        raise FileNotFoundError(f"Spec-Bench file not found: {path}")
    instances: list[Instance] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if category is not None and str(obj.get("category", "")).lower() != category.lower():
                continue
            turns = obj.get("turns")
            if not isinstance(turns, list) or not turns:
                raise ValueError(f"{path}:{line_no}: missing non-empty turns")
            flat = _flatten_specbench_turns(turns)
            if flat:
                instances.append(multi_turn(flat, specbench_chain_turns=True))
    if not instances:
        hint = f" for category={category!r}" if category else " (all categories)"
        raise ValueError(f"No Spec-Bench samples{hint} in {path}")
    return instances


# ---------------------------------------------------------------------------
# JSON / JSONL dispatch
# ---------------------------------------------------------------------------

def _load_json_array(rows: list, path: Path, alias: str) -> list[Instance]:
    if alias.lower() == "longbench_v2" or _is_longbench_v2(path):
        return _load_longbench_v2(rows, path)
    if _is_mixup(rows, path, alias):
        return rows_to_instances(rows, _format_context_input, path=path)
    if _is_swe_bench(rows, path, alias):
        return rows_to_instances(rows, _format_swe_bench, path=path)
    raise ValueError(f"Unsupported JSON dataset: {path}")


def _load_jsonl_lines(path: Path, alias: str) -> list[Instance]:
    task = _infer_infinitebench_task(alias, path)
    formatter: Callable[[dict], str]
    if task is not None:
        formatter = lambda row: _format_infinitebench(row, task)
    else:
        formatter = _format_context_question

    instances: list[Instance] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if "input" not in row:
                raise ValueError(f"Missing 'input' in {path} at line {line_no}")
            if "context" not in row:
                raise ValueError(f"Missing 'context' in {path} at line {line_no}")
            instances.append(single_turn(formatter(row)))
    return instances


def _load_jsonl_file(path: Path, alias: str) -> list[Instance]:
    with path.open("r", encoding="utf-8") as f:
        head = f.read(8192)
    # Some exports use .jsonl extension but store one JSON array.
    if head.lstrip("\ufeff").lstrip().startswith("["):
        return _load_json_array(_read_json_list(path), path, alias)
    return _load_jsonl_lines(path, alias)


def load_benchmark_dataset(dataset_name: str):
    """Load dataset by alias or path; returns list[Instance] or HF Dataset."""
    alias = dataset_name
    is_specbench, _ = _specbench_meta(alias)
    if is_specbench:
        resolved = resolve_dataset_path(alias)
        path = Path(resolved)
        if not path.is_file() or path.suffix != ".jsonl":
            path = _default_specbench_path()
        if not path.is_file():
            raise FileNotFoundError(
                f"Spec-Bench question.jsonl not found (tried {resolved} and {path})."
            )
        return _load_specbench(path, alias)

    resolved = resolve_dataset_path(alias)
    path = Path(resolved)

    if path.is_file() and path.suffix == ".json":
        return _load_json_array(_read_json_list(path), path, alias)

    if path.is_file() and path.suffix == ".jsonl":
        return _load_jsonl_file(path, alias)

    return load_and_process_dataset(resolved)
