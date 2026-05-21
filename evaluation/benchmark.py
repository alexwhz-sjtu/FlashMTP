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
from transformers import AutoModelForCausalLM, AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from specforge.modeling.draft.flashmtp import FlashMTPDraftModel, sample

from evaluation import distributed as dist
from evaluation.utils import load_and_process_dataset

def cuda_time() -> float:
    torch.cuda.synchronize()
    return time.perf_counter()

def resolve_mask_token_id(draft_model: FlashMTPDraftModel, tokenizer: AutoTokenizer) -> int:
    mask_token_id = draft_model.mask_token_id
    if mask_token_id is None:
        mask_token_id = tokenizer.mask_token_id
    if mask_token_id is None:
        raise ValueError(
            "mask_token_id is None. Please use a draft checkpoint whose config contains "
            "flashmtp_config['mask_token_id'], or pass/load a tokenizer with mask_token_id."
        )
    return int(mask_token_id)

@torch.inference_mode()
def target_generate(
    target: AutoModelForCausalLM,
    input_ids: torch.Tensor,
    max_new_tokens: int,
    stop_token_ids: list[int],
    temperature: float = 0.0,
) -> SimpleNamespace:
    target.eval()
    num_input_tokens = input_ids.shape[1]

    prefill_start = cuda_time()
    output = target(
        input_ids,
        use_cache=True,
        logits_to_keep=1,
    )
    next_token = sample(output.logits, temperature)
    time_to_first_token = cuda_time() - prefill_start

    decode_start = cuda_time()
    generated = [next_token]
    past_key_values = output.past_key_values
    while len(generated) < max_new_tokens:
        if stop_token_ids is not None and int(generated[-1].item()) in stop_token_ids:
            break
        output = target(
            generated[-1],
            past_key_values=past_key_values,
            use_cache=True,
            logits_to_keep=1,
        )
        past_key_values = output.past_key_values
        generated.append(sample(output.logits, temperature))

    generated_ids = torch.cat(generated, dim=1)
    output_ids = torch.cat([input_ids, generated_ids], dim=1)

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
        target_total_time=time_to_first_token + total_decode_time,
        draft_total_time=0.0,
        steps=num_output_tokens,
    )


@torch.inference_mode()
def flashmtp_generate(
    model: FlashMTPDraftModel,
    target: AutoModelForCausalLM,
    input_ids: torch.Tensor,
    max_new_tokens: int,
    stop_token_ids: list[int],
    temperature: float = 0.0,
) -> SimpleNamespace:
    start_time = cuda_time()
    output_ids = model.spec_generate(
        target=target,
        input_ids=input_ids,
        max_new_tokens=max_new_tokens,
        stop_token_ids=stop_token_ids,
        temperature=temperature,
    )
    total_time = cuda_time() - start_time
    stats = model.get_last_decode_stats()
    num_input_tokens = input_ids.shape[1]
    num_output_tokens = output_ids.shape[1] - num_input_tokens
    timed_total = stats.get("target_total_time", 0.0) + stats.get("draft_total_time", 0.0)
    time_per_output_token = (timed_total or total_time) / max(num_output_tokens, 1)

    return SimpleNamespace(
        output_ids=output_ids,
        num_input_tokens=num_input_tokens,
        num_output_tokens=num_output_tokens,
        time_to_first_token=0.0,
        time_per_output_token=time_per_output_token,
        acceptance_lengths=stats.get("accept_lengths", []),
        target_total_time=stats.get("target_total_time", 0.0),
        draft_total_time=stats.get("draft_total_time", 0.0),
        steps=stats.get("steps", 0),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name-or-path", "--target-model-path", type=str, default='/data/wanghanzhen/models/Qwen/Qwen3-8B')
    parser.add_argument("--draft-name-or-path", "--draft-model-path", type=str, default='/data/wanghanzhen/Projects/MTP/NIPS26/FlashMTP_exp/cache/models/flashmtp_exp_h100_sample_40000_think_off_nlayers5_block_16_maxlen4096_epochs6/epoch_6_step_29844')
    parser.add_argument("--block-size", type=int, default=None)
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--max-samples", type=int, default=10)
    parser.add_argument("--max-new-tokens", type=int, default=4096)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--think", action="store_true")
    parser.add_argument("--trust-remote-code", action="store_true")
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
    print(f"Using draft model: {args.draft_name_or_path}")

    def has_flash_attn():
        try:
            import flash_attn
            return True
        except ImportError:
            logger.warning("flash_attn is not installed. Falling back to torch.sdpa. The speedup will be lower.")
            return False

    has_flash_attn()

    target = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        # attn_implementation="flash_attention_2" if installed_flash_attn else "sdpa",
        dtype=torch.bfloat16,
        trust_remote_code=args.trust_remote_code,
    ).to(device).eval()

    draft_model = FlashMTPDraftModel.from_pretrained(
        args.draft_name_or_path,
        # attn_implementation="flash_attention_2" if installed_flash_attn else "sdpa",
        dtype=torch.bfloat16,
        trust_remote_code=args.trust_remote_code,
    ).to(device).eval()

    block_size = args.block_size if args.block_size is not None else draft_model.block_size
    draft_model.block_size = block_size
    draft_model.config.block_size = block_size

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=args.trust_remote_code)
    mask_token_id = resolve_mask_token_id(draft_model, tokenizer)
    draft_model.mask_token_id = mask_token_id
    draft_model.config.flashmtp_config["mask_token_id"] = mask_token_id
    stop_token_ids = [token_id for token_id in [tokenizer.eos_token_id] if token_id is not None]
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
            input_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=args.think)
            input_ids = tokenizer.encode(input_text, return_tensors="pt").to(target.device)

            response = {}
            response[1] = target_generate(
                target=target,
                input_ids=input_ids,
                max_new_tokens=args.max_new_tokens,
                stop_token_ids=stop_token_ids,
                temperature=args.temperature,
            )
            response[block_size] = flashmtp_generate(
                model=draft_model,
                target=target,
                input_ids=input_ids,
                max_new_tokens=args.max_new_tokens,
                stop_token_ids=stop_token_ids,
                temperature=args.temperature,
            )

            spec_response = response[block_size]
            generated_ids = spec_response.output_ids[0, spec_response.num_input_tokens:]
            output_text = tokenizer.decode(generated_ids, skip_special_tokens=True)

            acceptance_lengths_text = ", ".join(
                [f"{position}:{length}" for position, length in enumerate(spec_response.acceptance_lengths)]
            )
            avg_acceptance_length = (
                float(np.mean(spec_response.acceptance_lengths))
                if spec_response.acceptance_lengths
                else 0.0
            )
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

    t1 = sum(
        r[1].time_per_output_token * r[1].num_output_tokens for r in responses
    ) / sum(r[1].num_output_tokens for r in responses)
    tb = sum(
        r[block_size].time_per_output_token * r[block_size].num_output_tokens
        for r in responses
    ) / sum(r[block_size].num_output_tokens for r in responses)
    print(f"Decoding speedup: {t1 / tb:.2f}")

    mean_acceptance_values = [
        np.mean(r[block_size].acceptance_lengths)
        for r in responses
        if r[block_size].acceptance_lengths
    ]
    tau = float(np.mean(mean_acceptance_values)) if mean_acceptance_values else 0.0
    print(f"Average Acceptance length: {tau:.2f}")

    acceptance_lengths = list(chain(*[r[block_size].acceptance_lengths for r in responses]))
    histogram = [
        acceptance_lengths.count(b) / len(acceptance_lengths)
        for b in range(block_size + 1)
    ] if acceptance_lengths else [0.0 for _ in range(block_size + 1)]
    print(f"Acceptance length histogram: {[f'{x * 100:.1f}%' for x in histogram]}")

    total_elapsed_time = cuda_time() - benchmark_start
    print(f"Total elapsed time: {total_elapsed_time:.2f}s")

if __name__ == "__main__":
    main()
