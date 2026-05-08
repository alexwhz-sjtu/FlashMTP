import argparse
import random
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
        generate_start = cuda_time()
        output_ids = model.spec_generate(
            target=target,
            input_ids=input_ids,
            max_new_tokens=max_new_tokens,
            stop_token_ids=stop_token_ids,
            temperature=temperature,
        )
        total_time = cuda_time() - generate_start
    finally:
        model.block_size = original_block_size

    stats = model.get_last_decode_stats()
    num_input_tokens = input_ids.shape[1]
    num_output_tokens = output_ids.shape[1] - num_input_tokens
    measured_decode_time = stats.get("target_total_time", 0.0) + stats.get(
        "draft_total_time", 0.0
    )
    time_per_output_token = measured_decode_time / max(num_output_tokens, 1)

    return SimpleNamespace(
        output_ids=output_ids,
        num_input_tokens=num_input_tokens,
        num_output_tokens=num_output_tokens,
        time_to_first_token=0.0,
        time_per_output_token=time_per_output_token,
        acceptance_lengths=stats.get("accept_lengths", []),
        target_total_time=stats.get("target_total_time", 0.0),
        draft_total_time=stats.get("draft_total_time", 0.0),
        total_time=total_time,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name-or-path", type=str, default='/data/wanghanzhen/models/Qwen/Qwen3-8B')
    parser.add_argument("--draft-name-or-path", type=str, default='/data/wanghanzhen/Projects/MTP/NIPS26/FlashMTP_v5.1/cache/models/flashmtp_v5.1_h100_sample_40000_think_off_nlayers5_block__maxlen4096_epochs6/epoch_6_step_29844')
    parser.add_argument("--block-size", type=int, default=None)
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--max-samples", type=int, default=10)
    parser.add_argument("--max-new-tokens", type=int, default=4096)
    parser.add_argument("--temperature", type=float, default=0.0)
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

    target = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        # attn_implementation="flash_attention_2" if installed_flash_attn else "sdpa",
        dtype=torch.bfloat16,
    ).to(device).eval()

    draft_model = FlashMTPDraftModel.from_pretrained(
        args.draft_name_or_path,
        # attn_implementation="flash_attention_2" if installed_flash_attn else "sdpa",
        dtype=torch.bfloat16,
    ).to(device).eval()

    block_size = args.block_size if args.block_size is not None else draft_model.block_size

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    dataset = load_and_process_dataset(args.dataset)

    if args.max_samples is not None and len(dataset) > args.max_samples:
        dataset = dataset.shuffle(seed=0).select(range(args.max_samples))

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

    tau = np.mean([np.mean(r[block_size].acceptance_lengths) for r in responses])
    print(f"Average Acceptance length: {tau:.2f}")

    acceptance_lengths = list(chain(*[r[block_size].acceptance_lengths for r in responses]))
    histogram = [acceptance_lengths.count(b) / len(acceptance_lengths) for b in range(block_size + 1)]
    print(f"Acceptance length histogram: {[f'{x * 100:.1f}%' for x in histogram]}")

    total_elapsed_time = cuda_time() - benchmark_start
    print(f"Total elapsed time: {total_elapsed_time:.2f}s")

if __name__ == "__main__":
    main()
