"""FlashMTP draft profiling: per-iteration draft top-k and verify accept length -> JSONL.

Does not run baseline target-only decoding or speedup / acceptance statistics.

Each speculative **block** appends one summary object with ``block_start``, ``accept_length``,
``verify_match_pairs``, and target softmax **top-2** at the last accepted verify step
(``target_top2_at_last_accept`` + ``abs_pos_last_accept``) and at the first mismatch
(``target_top2_at_first_reject`` + ``abs_pos_first_reject``, null if the whole compared
prefix matched).
"""
import argparse
import json
import random
import sys
import time
from itertools import chain
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from loguru import logger
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from specforge.modeling.draft.flashmtp import FlashMTPDraftModel, sample
from specforge.modeling.draft.flashmtp_chunk_utils import normalize_decode_chunk_sizes

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
def flashmtp_generate_profiled(
    model: FlashMTPDraftModel,
    target: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    input_ids: torch.Tensor,
    max_new_tokens: int,
    stop_token_ids: list[int],
    temperature: float,
    profile_top_k: int,
    profile_records: list | None,
    silent: bool,
) -> SimpleNamespace:
    print_fn = (lambda *args, **kwargs: None) if silent else print
    start_time = cuda_time()
    output_ids = model.spec_generate_with_profile(
        target=target,
        tokenizer=tokenizer,
        input_ids=input_ids,
        max_new_tokens=max_new_tokens,
        stop_token_ids=stop_token_ids,
        temperature=temperature,
        top_k=profile_top_k,
        print_fn=print_fn,
        profile_records=profile_records,
    )
    total_time = cuda_time() - start_time
    stats = model.get_last_decode_stats()
    num_input_tokens = input_ids.shape[1]
    num_output_tokens = output_ids.shape[1] - num_input_tokens

    return SimpleNamespace(
        output_ids=output_ids,
        num_input_tokens=num_input_tokens,
        num_output_tokens=num_output_tokens,
        wall_seconds=total_time,
        steps=stats.get("steps", 0),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name-or-path", "--target-model-path", type=str, default="/data/wanghanzhen/models/Qwen/Qwen3-8B")
    parser.add_argument(
        "--draft-name-or-path",
        "--draft-model-path",
        type=str,
        default="/data/wanghanzhen/Projects/MTP/NIPS26/FlashMTP_exp/cache/models/flashmtp_exp_h100_sample_40000_think_off_nlayers5_block_16_maxlen4096_epochs6/epoch_6_step_29844",
    )
    parser.add_argument("--block-size", type=int, default=None)
    parser.add_argument(
        "--decode-chunk-sizes",
        type=str,
        default=None,
        help='Optional e.g. "4,4,4,4" (sum must equal block size).',
    )
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--max-samples", type=int, default=10)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument(
        "--profile-top-k",
        type=int,
        default=4,
        help="Top-k draft candidates per slot in JSONL (default 4).",
    )
    parser.add_argument(
        "--output-jsonl",
        type=str,
        default=None,
        help="JSONL path (default: log/spec_profile_<dataset>_n<max_samples>.jsonl under project root).",
    )
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

    try:
        import flash_attn  # noqa: F401
    except ImportError:
        logger.warning("flash_attn is not installed. Falling back to torch.sdpa.")

    target = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        dtype=torch.bfloat16,
        trust_remote_code=args.trust_remote_code,
    ).to(device).eval()

    draft_model = FlashMTPDraftModel.from_pretrained(
        args.draft_name_or_path,
        dtype=torch.bfloat16,
        trust_remote_code=args.trust_remote_code,
    ).to(device).eval()

    block_size = args.block_size if args.block_size is not None else draft_model.block_size
    draft_model.block_size = block_size
    draft_model.config.block_size = block_size

    if args.decode_chunk_sizes is not None:
        sizes = normalize_decode_chunk_sizes(args.decode_chunk_sizes, block_size)
        draft_model.decode_chunk_sizes = sizes
        if (
            not hasattr(draft_model.config, "flashmtp_config")
            or draft_model.config.flashmtp_config is None
        ):
            draft_model.config.flashmtp_config = {}
        draft_model.config.flashmtp_config["decode_chunk_sizes"] = sizes

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=args.trust_remote_code)
    mask_token_id = resolve_mask_token_id(draft_model, tokenizer)
    draft_model.mask_token_id = mask_token_id
    draft_model.config.flashmtp_config["mask_token_id"] = mask_token_id
    stop_token_ids = [token_id for token_id in [tokenizer.eos_token_id] if token_id is not None]
    dataset = load_and_process_dataset(args.dataset)

    if args.max_samples is not None and len(dataset) > args.max_samples:
        dataset = dataset.shuffle(seed=0).select(range(args.max_samples))

    log_dir = PROJECT_ROOT / "log"
    log_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = (
        Path(args.output_jsonl)
        if args.output_jsonl
        else log_dir / f"spec_profile_{args.dataset}_n{args.max_samples}.jsonl"
    )
    if not jsonl_path.is_absolute():
        jsonl_path = PROJECT_ROOT / jsonl_path

    all_rows: list[dict] = []
    indices = range(dist.rank(), len(dataset), dist.size())
    for idx in tqdm(indices, disable=not dist.is_main()):
        instance = dataset[idx]
        messages = []
        for turn_index, user_content in enumerate(instance["turns"]):
            messages.append({"role": "user", "content": user_content})
            input_text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True, enable_thinking=args.think
            )
            input_ids = tokenizer.encode(input_text, return_tensors="pt").to(target.device)

            profile_records: list[dict] = []
            spec_out = flashmtp_generate_profiled(
                model=draft_model,
                target=target,
                tokenizer=tokenizer,
                input_ids=input_ids,
                max_new_tokens=args.max_new_tokens,
                stop_token_ids=stop_token_ids,
                temperature=args.temperature,
                profile_top_k=args.profile_top_k,
                profile_records=profile_records,
                silent=True,
            )
            all_rows.extend(profile_records)

            if dist.is_main():
                text = tokenizer.decode(spec_out.output_ids[0, spec_out.num_input_tokens :], skip_special_tokens=True)
                logger.info("[sample {} turn {}] output_len={} preview={!r}", idx, turn_index, len(text), text[:200])

            messages.append(
                {
                    "role": "assistant",
                    "content": tokenizer.decode(
                        spec_out.output_ids[0, spec_out.num_input_tokens :], skip_special_tokens=True
                    ),
                }
            )

    if dist.size() > 1 and dist.is_initialized():
        gathered = dist.gather(all_rows, dst=0)
        if not dist.is_main():
            return
        all_rows = list(chain.from_iterable(gathered))

    if dist.is_main() or not dist.is_initialized():
        with open(jsonl_path, "w", encoding="utf-8") as f:
            for row in all_rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        logger.info("Wrote {} JSONL lines to {}", len(all_rows), jsonl_path)


if __name__ == "__main__":
    main()
