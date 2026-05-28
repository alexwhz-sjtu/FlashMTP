"""FlashMTP draft profiling (v1.1).

Modes (``--profile-mode``):

- ``jsonl`` (default): per-step draft top-k + accept length -> JSONL (via
  ``FlashMTPDraftModel.spec_generate_with_profile``).
- ``profile_time``: CUDA-event GPU timing — target verify avg ms and draft forward
  avg ms per batch size (same prompt replicated on batch dim).
- ``profile_token``: structured JSON — per-sample draft top-k, sampled tokens,
  accept length, and target verify (no terminal dump from ``spec_generate_with_profile``).
"""
from __future__ import annotations

import argparse
import gc
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
from torch import distributed as torch_dist
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parents[1]
_PROFILE_UTILS = Path(__file__).resolve().parent
for p in (PROJECT_ROOT, _PROFILE_UTILS):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from specforge.modeling.draft.flashmtp import FlashMTPDraftModel

from evaluation import distributed as dist
from evaluation.benchmark import load_benchmark_dataset, select_max_samples

from flashmtp_cuda_profile import profile_flashmtp_generate, stats_to_jsonable
from flashmtp_profile_format import compact_profile_token_lines


def cuda_time() -> float:
    torch.cuda.synchronize()
    return time.perf_counter()


def resolve_mask_token_id(draft_model: FlashMTPDraftModel, tokenizer: AutoTokenizer) -> int:
    mask_token_id = draft_model.mask_token_id
    if mask_token_id is None:
        mask_token_id = tokenizer.mask_token_id
    if mask_token_id is None:
        raise ValueError(
            "mask_token_id is None. Use a draft checkpoint with "
            "flashmtp_config['mask_token_id'], or a tokenizer with mask_token_id."
        )
    return int(mask_token_id)


def load_models_and_tokenizer(args: argparse.Namespace, device: torch.device):
    try:
        import flash_attn  # noqa: F401

        attn_impl = "flash_attention_2"
    except ImportError:
        logger.warning("flash_attn not installed; using sdpa.")
        attn_impl = "sdpa"

    target = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        attn_implementation=attn_impl,
        dtype=torch.bfloat16,
        trust_remote_code=args.trust_remote_code,
    ).to(device).eval()

    draft_model = FlashMTPDraftModel.from_pretrained(
        args.draft_name_or_path,
        attn_implementation=attn_impl,
        dtype=torch.bfloat16,
        trust_remote_code=args.trust_remote_code,
    ).to(device).eval()

    if args.local_position is not None:
        lp = args.local_position == "true"
        draft_model.local_position = lp
        if draft_model.config.flashmtp_config is None:
            draft_model.config.flashmtp_config = {}
        draft_model.config.flashmtp_config["local_position"] = lp
        logger.info("Overriding local_position={} (CLI)", lp)

    block_size = args.block_size if args.block_size is not None else draft_model.block_size
    draft_model.block_size = block_size
    draft_model.config.block_size = block_size

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path, trust_remote_code=args.trust_remote_code
    )
    mask_token_id = resolve_mask_token_id(draft_model, tokenizer)
    draft_model.mask_token_id = mask_token_id
    if draft_model.config.flashmtp_config is None:
        draft_model.config.flashmtp_config = {}
    draft_model.config.flashmtp_config["mask_token_id"] = mask_token_id

    return target, draft_model, tokenizer, mask_token_id


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
) -> SimpleNamespace:
    start_time = cuda_time()
    output_ids = model.spec_generate_with_profile(
        target=target,
        tokenizer=tokenizer,
        input_ids=input_ids,
        max_new_tokens=max_new_tokens,
        stop_token_ids=stop_token_ids,
        temperature=temperature,
        top_k=profile_top_k,
        print_fn=lambda *args, **kwargs: None,
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
        decode_stats=stats,
    )


def parse_batch_sizes(s: str) -> list[int]:
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def run_profile_time(
    args: argparse.Namespace,
    target: AutoModelForCausalLM,
    draft_model: FlashMTPDraftModel,
    tokenizer: AutoTokenizer,
    mask_token_id: int,
    device: torch.device,
) -> list[dict]:
    dataset = select_max_samples(load_benchmark_dataset(args.dataset), args.max_samples)

    stop_token_ids = [tid for tid in [tokenizer.eos_token_id] if tid is not None]
    batch_sizes = parse_batch_sizes(args.batch_sizes)
    sample_indices = list(range(min(args.profile_time_samples, len(dataset))))
    rows: list[dict] = []

    for si in sample_indices:
        instance = dataset[si]
        turn = instance["turns"][0]
        messages = [{"role": "user", "content": turn}]
        input_text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=args.think,
        )
        input_ids = tokenizer.encode(input_text, return_tensors="pt").to(device)

        for bsz in batch_sizes:
            key = f"{args.dataset}|sample{si}|bs{bsz}"
            logger.info("=== profile_time {} ===", key)
            for _ in range(args.warmup_runs):
                profile_flashmtp_generate(
                    model=draft_model,
                    target=target,
                    input_ids=input_ids.clone(),
                    mask_token_id=mask_token_id,
                    max_new_tokens=args.max_new_tokens,
                    stop_token_ids=stop_token_ids,
                    temperature=args.temperature,
                    batch_size=bsz,
                )
                torch.cuda.synchronize()
                gc.collect()
                torch.cuda.empty_cache()

            _, stats = profile_flashmtp_generate(
                model=draft_model,
                target=target,
                input_ids=input_ids.clone(),
                mask_token_id=mask_token_id,
                max_new_tokens=args.max_new_tokens,
                stop_token_ids=stop_token_ids,
                temperature=args.temperature,
                batch_size=bsz,
            )
            st = stats_to_jsonable(stats)
            decode_sum = float(st["target_verify_total_ms"]) + float(
                st["draft_forward_total_ms"]
            )
            st["decode_sum_ms"] = decode_sum
            if decode_sum > 1e-9:
                st["decode_target_time_share"] = float(st["target_verify_total_ms"]) / decode_sum
                st["decode_draft_time_share"] = float(st["draft_forward_total_ms"]) / decode_sum
            else:
                st["decode_target_time_share"] = None
                st["decode_draft_time_share"] = None
            rows.append(
                {
                    "dataset": args.dataset,
                    "sample_index": si,
                    "batch_size": bsz,
                    "max_new_tokens": args.max_new_tokens,
                    "block_size": draft_model.block_size,
                    "stats": st,
                }
            )
            logger.info(
                "  target_verify_avg_ms={:.3f} draft_forward_avg_ms={:.3f} steps={}",
                st["target_verify_avg_ms"],
                st["draft_forward_avg_ms"],
                st["num_target_verifies"],
            )

    return rows


def run_profile_token_or_jsonl(
    args: argparse.Namespace,
    target: AutoModelForCausalLM,
    draft_model: FlashMTPDraftModel,
    tokenizer: AutoTokenizer,
    *,
    write_jsonl: bool,
    write_token_json: bool,
) -> tuple[list[dict], list[dict]]:
    stop_token_ids = [tid for tid in [tokenizer.eos_token_id] if tid is not None]
    dataset = select_max_samples(load_benchmark_dataset(args.dataset), args.max_samples)

    all_rows: list[dict] = []
    token_samples: list[dict] = []
    indices = range(dist.rank(), len(dataset), dist.size())

    for idx in tqdm(indices, disable=not dist.is_main()):
        instance = dataset[idx]
        messages = []
        for turn_index, user_content in enumerate(instance["turns"]):
            messages.append({"role": "user", "content": user_content})
            input_text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=args.think,
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
            )
            if write_jsonl:
                all_rows.extend(profile_records)

            if write_token_json:
                token_samples.append(
                    {
                        "sample_index": idx,
                        "turn_index": turn_index,
                        "input_len": spec_out.num_input_tokens,
                        "output_len": spec_out.num_output_tokens,
                        "steps": spec_out.steps,
                        "lines": compact_profile_token_lines(profile_records),
                    }
                )

            if dist.is_main():
                text = tokenizer.decode(
                    spec_out.output_ids[0, spec_out.num_input_tokens :],
                    skip_special_tokens=True,
                )
                logger.info(
                    "[sample {} turn {}] output_len={} preview={!r}",
                    idx,
                    turn_index,
                    len(text),
                    text[:200],
                )

            messages.append(
                {
                    "role": "assistant",
                    "content": tokenizer.decode(
                        spec_out.output_ids[0, spec_out.num_input_tokens :],
                        skip_special_tokens=True,
                    ),
                }
            )

    return all_rows, token_samples


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--profile-mode",
        type=str,
        default="jsonl",
        choices=("jsonl", "profile_time", "profile_token"),
        help="jsonl: JSONL dump; profile_time: CUDA timing by batch size; "
        "profile_token: structured token profile JSON.",
    )
    parser.add_argument(
        "--model-name-or-path",
        "--target-model-path",
        type=str,
        default="/data/wanghanzhen/models/Qwen/Qwen3-8B",
    )
    parser.add_argument(
        "--draft-name-or-path",
        "--draft-model-path",
        type=str,
        required=True,
    )
    parser.add_argument("--block-size", type=int, default=None)
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--max-samples", type=int, default=10)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument(
        "--profile-top-k",
        type=int,
        default=4,
        help="Top-k draft candidates per slot (jsonl / profile_token).",
    )
    parser.add_argument(
        "--output-jsonl",
        type=str,
        default=None,
        help="JSONL path for jsonl mode.",
    )
    parser.add_argument(
        "--output-token-json",
        type=str,
        default=None,
        help="JSON path for profile_token mode.",
    )
    parser.add_argument(
        "--output-token-log",
        type=str,
        default=None,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--output-time-json",
        type=str,
        default=None,
        help="JSON path for profile_time mode.",
    )
    parser.add_argument(
        "--batch-sizes",
        type=str,
        default="1,16,32,64,128",
        help="Comma-separated batch sizes for profile_time.",
    )
    parser.add_argument(
        "--warmup-runs",
        type=int,
        default=1,
        help="Warmup runs per (sample, batch_size) cell in profile_time.",
    )
    parser.add_argument(
        "--profile-time-samples",
        type=int,
        default=2,
        help="Number of dataset samples for profile_time (uses first turn only).",
    )
    parser.add_argument(
        "--local-position",
        type=str,
        default=None,
        choices=("true", "false"),
        help="Override draft local_position; default: use checkpoint.",
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

    use_dist = args.profile_mode != "profile_time"
    if use_dist:
        dist.init()
        torch.cuda.set_device(dist.local_rank())
        device = torch.device(f"cuda:{dist.local_rank()}")
    else:
        device = torch.device("cuda:0")
        torch.cuda.set_device(device)

    target, draft_model, tokenizer, mask_token_id = load_models_and_tokenizer(args, device)
    log_dir = _PROFILE_UTILS / "log"
    log_dir.mkdir(parents=True, exist_ok=True)

    if args.profile_mode == "profile_time":
        if args.temperature >= 1e-5:
            raise ValueError("profile_time requires temperature≈0 for batched decode.")
        rows = run_profile_time(
            args, target, draft_model, tokenizer, mask_token_id, device
        )
        out_path = (
            Path(args.output_time_json)
            if args.output_time_json
            else log_dir / f"spec_profile_time_{args.dataset}.json"
        )
        if not out_path.is_absolute():
            out_path = _PROFILE_UTILS / out_path
        payload = {
            "profile_mode": "profile_time",
            "dataset": args.dataset,
            "target_model": args.model_name_or_path,
            "draft_model": args.draft_name_or_path,
            "max_new_tokens": args.max_new_tokens,
            "batch_sizes": parse_batch_sizes(args.batch_sizes),
            "warmup_runs": args.warmup_runs,
            "rows": rows,
        }
        out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        logger.info("Wrote profile_time results to {}", out_path)
        return

    write_jsonl = args.profile_mode == "jsonl"
    write_token_json = args.profile_mode == "profile_token"
    all_rows, token_samples = run_profile_token_or_jsonl(
        args,
        target,
        draft_model,
        tokenizer,
        write_jsonl=write_jsonl,
        write_token_json=write_token_json,
    )

    if use_dist and dist.size() > 1 and torch_dist.is_initialized():
        gathered_rows = dist.gather(all_rows, dst=0)
        gathered_samples = (
            dist.gather(token_samples, dst=0) if write_token_json else None
        )
        if not dist.is_main():
            return
        all_rows = list(chain.from_iterable(gathered_rows))
        if gathered_samples is not None:
            token_samples = list(chain.from_iterable(gathered_samples))

    if not use_dist or dist.is_main() or not torch_dist.is_initialized():
        if write_jsonl:
            jsonl_path = (
                Path(args.output_jsonl)
                if args.output_jsonl
                else log_dir / f"spec_profile_{args.dataset}_n{args.max_samples}.jsonl"
            )
            if not jsonl_path.is_absolute():
                jsonl_path = _PROFILE_UTILS / jsonl_path
            with open(jsonl_path, "w", encoding="utf-8") as f:
                for row in all_rows:
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
            logger.info("Wrote {} JSONL lines to {}", len(all_rows), jsonl_path)

        if write_token_json:
            token_json_arg = args.output_token_json or args.output_token_log
            token_path = (
                Path(token_json_arg)
                if token_json_arg
                else log_dir
                / f"spec_profile_token_{args.dataset}_n{args.max_samples}.json"
            )
            if not token_path.is_absolute():
                token_path = _PROFILE_UTILS / token_path
            payload = {
                "profile_mode": "profile_token",
                "dataset": args.dataset,
                "target_model": args.model_name_or_path,
                "draft_model": args.draft_name_or_path,
                "max_samples": args.max_samples,
                "max_new_tokens": args.max_new_tokens,
                "profile_top_k": args.profile_top_k,
                "temperature": args.temperature,
                "samples": token_samples,
            }
            token_path.write_text(
                json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
            )
            logger.info(
                "Wrote token profile JSON ({} samples) to {}",
                len(token_samples),
                token_path,
            )


if __name__ == "__main__":
    main()
