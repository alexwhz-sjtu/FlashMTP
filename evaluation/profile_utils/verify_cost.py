import argparse
import json
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


DEFAULT_MODEL_PATH = "/data/wanghanzhen/models/Qwen/Qwen3-8B"


def cuda_time(device: torch.device) -> float:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    return time.perf_counter()


def parse_lengths(lengths: str | None, min_len: int, max_len: int, step: int) -> list[int]:
    if lengths:
        return [int(item.strip()) for item in lengths.split(",") if item.strip()]
    return list(range(min_len, max_len + 1, step))


def random_tokens(
    batch_size: int,
    seq_len: int,
    vocab_size: int,
    device: torch.device,
    generator: torch.Generator,
) -> torch.Tensor:
    return torch.randint(
        low=0,
        high=vocab_size,
        size=(batch_size, seq_len),
        device=device,
        generator=generator,
        dtype=torch.long,
    )


@torch.inference_mode()
def prefill_cache(
    model: AutoModelForCausalLM,
    input_ids: torch.Tensor,
) -> object:
    output = model(
        input_ids=input_ids,
        use_cache=True,
        logits_to_keep=1,
    )
    return output.past_key_values


@torch.inference_mode()
def run_verify_once(
    model: AutoModelForCausalLM,
    prefix_ids: torch.Tensor,
    verify_ids: torch.Tensor,
    device: torch.device,
) -> float:
    past_key_values = prefill_cache(model, prefix_ids)
    start = cuda_time(device)
    _ = model(
        input_ids=verify_ids,
        past_key_values=past_key_values,
        use_cache=True,
        logits_to_keep=verify_ids.shape[1],
    )
    return cuda_time(device) - start


def mean(values: list[float]) -> float:
    return sum(values) / max(len(values), 1)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Profile the target-model cost of verifying a fixed number of candidate "
            "tokens at different prefix lengths."
        )
    )
    parser.add_argument("--model-name-or-path", type=str, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--min-len", type=int, default=1024)
    parser.add_argument("--max-len", type=int, default=32 * 1024)
    parser.add_argument("--step", type=int, default=1024)
    parser.add_argument(
        "--lengths",
        type=str,
        default=None,
        help="Comma-separated prefix lengths. Overrides --min-len/--max-len/--step.",
    )
    parser.add_argument("--verify-tokens", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeat", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--attn-implementation", type=str, default="flash_attention_2")
    parser.add_argument("--output-jsonl", type=Path, default=None)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("This profiling script expects a CUDA device.")

    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    generator = torch.Generator(device=device)
    generator.manual_seed(args.seed)

    model_kwargs = {"torch_dtype": torch.bfloat16}
    if args.attn_implementation:
        model_kwargs["attn_implementation"] = args.attn_implementation

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        **model_kwargs,
    ).to(device).eval()

    vocab_size = getattr(model.config, "vocab_size", len(tokenizer))
    lengths = parse_lengths(args.lengths, args.min_len, args.max_len, args.step)
    output_file = None
    if args.output_jsonl is not None:
        args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
        output_file = args.output_jsonl.open("w", encoding="utf-8")

    print(f"model: {args.model_name_or_path}")
    print(f"verify_tokens: {args.verify_tokens}, batch_size: {args.batch_size}")
    print(
        f"{'prefix_len':>10} {'avg_ms':>12} {'min_ms':>12} "
        f"{'tok/s':>12} {'samples':>8}"
    )

    try:
        for prefix_len in lengths:
            prefix_ids = random_tokens(
                args.batch_size,
                prefix_len,
                vocab_size,
                device,
                generator,
            )
            verify_ids = random_tokens(
                args.batch_size,
                args.verify_tokens,
                vocab_size,
                device,
                generator,
            )

            for _ in range(args.warmup):
                run_verify_once(model, prefix_ids, verify_ids, device)

            timings = [
                run_verify_once(model, prefix_ids, verify_ids, device)
                for _ in range(args.repeat)
            ]
            avg_time = mean(timings)
            min_time = min(timings)
            verified_tokens = args.batch_size * args.verify_tokens
            row = {
                "prefix_len": prefix_len,
                "verify_tokens": args.verify_tokens,
                "batch_size": args.batch_size,
                "avg_seconds": avg_time,
                "min_seconds": min_time,
                "tokens_per_second": verified_tokens / avg_time,
                "samples": args.repeat,
            }

            print(
                f"{prefix_len:10d} {avg_time * 1000:12.3f} "
                f"{min_time * 1000:12.3f} {row['tokens_per_second']:12.2f} "
                f"{args.repeat:8d}"
            )

            if output_file is not None:
                output_file.write(json.dumps(row, ensure_ascii=False) + "\n")
                output_file.flush()
    finally:
        if output_file is not None:
            output_file.close()


if __name__ == "__main__":
    main()
