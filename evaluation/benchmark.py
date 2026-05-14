import argparse
import json
import random
import re
import sys
import time
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

@torch.inference_mode()
def target_generate(
    target: AutoModelForCausalLM,
    input_ids: torch.Tensor,
    max_new_tokens: int,
    stop_token_ids: list[int],
    temperature: float = 0.0,
) -> SimpleNamespace:
    num_input_tokens = input_ids.shape[1]
    max_length = num_input_tokens + max_new_tokens
    output_ids = torch.empty((1, max_length), dtype=torch.long, device=input_ids.device)
    output_ids[:, :num_input_tokens] = input_ids
    position_ids = torch.arange(max_length, device=input_ids.device).unsqueeze(0)
    past_key_values_target = DynamicCache()

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
    time_to_first_token = cuda_time() - prefill_start

    decode_start = cuda_time()
    start = input_ids.shape[1]
    while start < max_length:
        output_ids[:, start : start + 1] = next_token
        start += 1

        if stop_token_ids is not None and next_token.item() in stop_token_ids:
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
    total_decode_time = cuda_time() - decode_start
    time_per_output_token = total_decode_time / max(num_output_tokens, 1)

    return SimpleNamespace(
        output_ids=output_ids,
        num_input_tokens=num_input_tokens,
        num_output_tokens=num_output_tokens,
        time_to_first_token=time_to_first_token,
        time_per_output_token=time_per_output_token,
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
) -> SimpleNamespace:
    original_block_size = model.block_size
    model.block_size = block_size
    try:
        output_ids = model.spec_generate(
            target=target,
            input_ids=input_ids,
            max_new_tokens=max_new_tokens,
            stop_token_ids=stop_token_ids,
            temperature=temperature,
        )
    finally:
        model.block_size = original_block_size

    stats = model.get_last_decode_stats()
    num_input_tokens = input_ids.shape[1]
    num_output_tokens = output_ids.shape[1] - num_input_tokens
    measured_decode_time = stats.get(
        "total_time",
        stats.get("target_total_time", 0.0) + stats.get("draft_total_time", 0.0),
    )
    time_per_output_token = measured_decode_time / max(num_output_tokens, 1)

    return SimpleNamespace(
        output_ids=output_ids,
        num_input_tokens=num_input_tokens,
        num_output_tokens=num_output_tokens,
        time_to_first_token=0.0,
        time_per_output_token=time_per_output_token,
        acceptance_lengths=stats.get("accept_lengths", []),
        target_prefill_time=0.0,
        target_decode_time=measured_decode_time,
        target_total_time=measured_decode_time,
        draft_total_time=stats.get("draft_total_time", 0.0),
        accepted_tokens=stats.get("accepted_tokens", 0),
        total_time=measured_decode_time,
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
        "--sink-num",
        type=int,
        default=None,
        help="Override draft flashmtp_config sink_num for this run only (default: from checkpoint).",
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
    eff_sink = fcfg.get("sink_num")
    if args.sink_num is not None:
        eff_sink = args.sink_num
        if draft_model.config.flashmtp_config is None:
            draft_model.config.flashmtp_config = {}
        draft_model.config.flashmtp_config["sink_num"] = eff_sink
        draft_model.sink_num = int(eff_sink)
    if eff_sink is None:
        raise ValueError(
            "Checkpoint config.flashmtp_config must contain 'sink_num'. "
            "Pass --sink-num to set it for this run."
        )
    logger.info(
        f"FlashMTP draft: sink_num={draft_model.sink_num}, "
        f"chs_len={draft_model.sink_num + 1}, block_size={draft_model.block_size}"
    )
    block_size = args.block_size if args.block_size is not None else draft_model.block_size

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    dataset = load_benchmark_dataset(args.dataset)
    dataset = select_max_samples(dataset, args.max_samples)

    benchmark_start = cuda_time()
    responses = []
    indices = range(dist.rank(), len(dataset), dist.size())
    for idx in tqdm(indices, disable=not dist.is_main()):
        instance = dataset[idx]
        messages = []
        for turn_index, user_content in enumerate(instance["turns"]):
            messages.append({"role": "user", "content": user_content})
            input_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
            input_ids = tokenizer.encode(input_text, return_tensors="pt").to(target.device)
            print(
                f"\n[Sample {idx} | Turn {turn_index}] Input length: "
                f"{input_ids.shape[1]} tokens ({len(user_content)} chars)"
            )

            response = {}
            response[1] = target_generate(
                target=target,
                input_ids=input_ids,
                max_new_tokens=args.max_new_tokens,
                stop_token_ids=[tokenizer.eos_token_id],
                temperature=args.temperature,
            )
            response[block_size] = flashmtp_generate(
                model=draft_model,
                target=target,
                input_ids=input_ids,
                max_new_tokens=args.max_new_tokens,
                block_size=block_size,
                stop_token_ids=[tokenizer.eos_token_id],
                temperature=args.temperature,
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
                f"[Sample {idx} | Turn {turn_index}] Decode timing: "
                f"baseline={response[1].time_per_output_token:.6f}s/token, "
                f"flashmtp_total={spec_response.time_per_output_token:.6f}s/token"
            )
            print(
                f"[Sample {idx} | Turn {turn_index}] Token stats: "
                f"accepted={spec_response.accepted_tokens}"
            )
            print(
                f"[Sample {idx} | Turn {turn_index}] Acceptance lengths (position:length): "
                f"{acceptance_lengths_text}"
            )
            print(f"[Sample {idx} | Turn {turn_index}] Average acceptance length: {avg_acceptance_length:.2f}")

            messages.append({"role": "assistant", "content": output_text})
            responses.append(response)

    if dist.size() > 1:
        responses = dist.gather(responses, dst=0)
        if not dist.is_main():
            return
        responses = list(chain(*responses))

    t1 = np.mean([r[1].time_per_output_token for r in responses])
    tb = np.mean([r[block_size].time_per_output_token for r in responses])
    print(f"Decoding speedup: {t1 / tb:.2f}")

    acceptance_lengths = list(chain(*[r[block_size].acceptance_lengths for r in responses]))
    histogram = [acceptance_lengths.count(b) / len(acceptance_lengths) for b in range(block_size + 1)]
    print(f"Acceptance length histogram: {[f'{x * 100:.1f}%' for x in histogram]}")
    
    tau = 0
    for index, num in enumerate(histogram): 
        tau += index * num 
    print(f"Average Acceptance length: {tau:.2f}")

    print(f"Average Acceptance length: {tau:.2f}")

    total_elapsed_time = cuda_time() - benchmark_start
    print(f"Total elapsed time: {total_elapsed_time:.2f}s")

if __name__ == "__main__":
    main()
