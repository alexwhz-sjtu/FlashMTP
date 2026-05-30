"""FlashMTP throughput benchmark CLI."""

from __future__ import annotations

import argparse
import random
import sys
from itertools import chain
from pathlib import Path

import numpy as np
import torch
from loguru import logger
from rich import print
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluation import distributed as dist
from evaluation.datasets import load_benchmark_dataset, select_max_samples
from evaluation.generation import cuda_time, flashmtp_generate, run_benchmark_warmup, target_generate
from evaluation.metrics import summarize_responses
from evaluation.models import configure_draft_model, load_draft_model, load_target_model, load_tokenizer

# Backward-compatible re-exports for profile scripts.
__all__ = ["load_benchmark_dataset", "select_max_samples"]


def set_seed(seed: int = 0) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def build_user_content(
    turn_q: str,
    turn_index: int,
    chain_turns: bool,
    prev_assistant: str,
) -> str:
    if not chain_turns:
        return turn_q
    return turn_q if turn_index == 0 else f"{prev_assistant}\n\n{turn_q}"


def run_turn_benchmark(
    *,
    turn_index: int,
    idx: int,
    target,
    draft_model,
    tokenizer,
    input_ids,
    block_size: int,
    max_new_tokens: int,
    stop_token_ids: list[int],
    temperature: float,
    decode_after_first: bool,
) -> tuple[dict, str]:
    response = {
        1: target_generate(
            target=target,
            input_ids=input_ids,
            max_new_tokens=max_new_tokens,
            stop_token_ids=stop_token_ids,
            temperature=temperature,
            decode_timing_after_first_token=decode_after_first,
        ),
        block_size: flashmtp_generate(
            model=draft_model,
            target=target,
            input_ids=input_ids,
            max_new_tokens=max_new_tokens,
            block_size=block_size,
            stop_token_ids=stop_token_ids,
            temperature=temperature,
            decode_timing_after_first_token=decode_after_first,
        ),
    }
    spec = response[block_size]
    generated_ids = spec.output_ids[0, spec.num_input_tokens :]
    output_text = tokenizer.decode(generated_ids, skip_special_tokens=True)

    accept_pairs = ", ".join(
        f"{pos}:{length}" for pos, length in enumerate(spec.acceptance_lengths)
    )
    avg_accept = np.mean(spec.acceptance_lengths) if spec.acceptance_lengths else 0.0
    print(f"\n[Sample {idx} | Turn {turn_index}] Response:\n{output_text}")
    print(
        f"[Sample {idx} | Turn {turn_index}] Decode s/token "
        f"baseline={response[1].time_per_output_token:.6f} "
        f"flashmtp={spec.time_per_output_token:.6f} | "
        f"tok/s baseline={response[1].throughput_tokens_per_sec:.2f} "
        f"flashmtp={spec.throughput_tokens_per_sec:.2f}"
    )
    print(f"[Sample {idx} | Turn {turn_index}] Acceptance: {accept_pairs}")
    print(f"[Sample {idx} | Turn {turn_index}] Avg acceptance: {avg_accept:.2f}")
    return response, output_text


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="FlashMTP decode throughput benchmark")
    parser.add_argument(
        "--model-name-or-path",
        type=str,
        default="/data/wanghanzhen/models/Qwen/Qwen3-8B",
    )
    parser.add_argument(
        "--draft-name-or-path",
        type=str,
        default=(
            "/data/wanghanzhen/Projects/MTP/NIPS26/FlashMTP_v5.1/cache/models/"
            "flashmtp_v5.1_fix_h100_sample_40000_think_off_nlayers5_block_16_maxlen4096_epochs6/"
            "epoch_6_step_29844"
        ),
    )
    parser.add_argument("--block-size", type=int, default=None)
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--max-samples", type=int, default=10)
    parser.add_argument("--max-new-tokens", type=int, default=4096)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Replicate prompt on batch dim. Use temperature=0 for aligned speculative steps.",
    )
    parser.add_argument("--sink-num", type=int, default=None, help="Legacy override.")
    parser.add_argument(
        "--local-position",
        type=str,
        default=None,
        choices=("true", "false"),
        help="Override draft local_position (default: checkpoint config).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(0)

    dist.init()
    torch.cuda.set_device(dist.local_rank())
    device = torch.device(f"cuda:{dist.local_rank()}")

    target = load_target_model(args.model_name_or_path, device)
    draft_model = load_draft_model(args.draft_name_or_path, device)
    configure_draft_model(
        draft_model, sink_num=args.sink_num, local_position=args.local_position
    )

    block_size = args.block_size if args.block_size is not None else draft_model.block_size

    if args.batch_size < 1:
        raise ValueError("--batch-size must be >= 1")
    if args.batch_size > 1 and args.temperature > 1e-5:
        logger.warning("batch_size>1 with temperature>0 may desync FlashMTP across rows.")

    tokenizer = load_tokenizer(args.model_name_or_path)
    stop_token_ids = [tid for tid in [tokenizer.eos_token_id] if tid is not None]
    dataset = select_max_samples(load_benchmark_dataset(args.dataset), args.max_samples)

    if dist.is_main():
        print("Running CUDA warmup...")
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
    responses: list[dict] = []
    for idx in tqdm(range(dist.rank(), len(dataset), dist.size()), disable=not dist.is_main()):
        instance = dataset[idx]
        messages: list[dict] = []
        chain_turns = bool(instance.get("specbench_chain_turns"))
        decode_after_first = chain_turns
        prev_assistant = ""

        for turn_index, turn_q in enumerate(instance["turns"]):
            user_content = build_user_content(turn_q, turn_index, chain_turns, prev_assistant)
            messages.append({"role": "user", "content": user_content})
            input_text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
            )
            input_ids = tokenizer.encode(input_text, return_tensors="pt").to(target.device)
            if args.batch_size > 1:
                input_ids = input_ids.expand(args.batch_size, -1).contiguous()
            print(
                f"\n[Sample {idx} | Turn {turn_index}] "
                f"input_tokens={input_ids.shape[1]} batch={input_ids.shape[0]}"
            )

            response, output_text = run_turn_benchmark(
                turn_index=turn_index,
                idx=idx,
                target=target,
                draft_model=draft_model,
                tokenizer=tokenizer,
                input_ids=input_ids,
                block_size=block_size,
                max_new_tokens=args.max_new_tokens,
                stop_token_ids=stop_token_ids,
                temperature=args.temperature,
                decode_after_first=decode_after_first,
            )
            messages.append({"role": "assistant", "content": output_text})
            if chain_turns:
                prev_assistant = output_text
            responses.append(response)

    if dist.size() > 1:
        gathered = dist.gather(responses, dst=0)
        if not dist.is_main():
            return
        responses = list(chain(*gathered))

    summarize_responses(responses, block_size, args.batch_size)
    print(f"Total elapsed time: {cuda_time() - benchmark_start:.2f}s")


if __name__ == "__main__":
    main()
