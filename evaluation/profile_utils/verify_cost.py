import argparse
import json
import math
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


def parse_attn_implementations(attn_implementations: str) -> list[str | None]:
    implementations: list[str | None] = []
    for item in attn_implementations.split(","):
        implementation = item.strip()
        if not implementation:
            continue
        implementations.append(None if implementation.lower() in {"none", "default"} else implementation)
    return implementations


def sparse_plus_extra_lengths(
    base_lengths: list[int],
    sparse_ratio: float,
    extra_input_len: int,
) -> list[tuple[int, str, int]]:
    rows: list[tuple[int, str, int]] = []
    seen: set[tuple[str, int, int]] = set()
    for base_len in base_lengths:
        dense_len = base_len + extra_input_len
        sparse_len = math.ceil(base_len * sparse_ratio) + extra_input_len
        for case, input_len in (("dense_plus_extra", dense_len), ("sparse_plus_extra", sparse_len)):
            key = (case, base_len, input_len)
            if key not in seen:
                rows.append((base_len, case, input_len))
                seen.add(key)
    return rows


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
    decoder: torch.nn.Module,
    input_ids: torch.Tensor,
) -> object:
    output = decoder(
        input_ids=input_ids,
        use_cache=True,
    )
    return output.past_key_values


@torch.inference_mode()
def run_prefill_once(
    decoder: torch.nn.Module,
    input_ids: torch.Tensor,
    device: torch.device,
) -> float:
    start = cuda_time(device)
    _ = prefill_cache(decoder, input_ids)
    return cuda_time(device) - start


@torch.inference_mode()
def run_decode_once(
    decoder: torch.nn.Module,
    decode_ids: torch.Tensor,
    past_key_values: object,
    device: torch.device,
) -> float:
    start = cuda_time(device)
    _ = decoder(
        input_ids=decode_ids,
        past_key_values=past_key_values,
        use_cache=True,
    )
    return cuda_time(device) - start


@torch.inference_mode()
def run_verify_once(
    decoder: torch.nn.Module,
    prefix_ids: torch.Tensor,
    verify_ids: torch.Tensor,
    device: torch.device,
) -> float:
    past_key_values = prefill_cache(decoder, prefix_ids)
    start = cuda_time(device)
    _ = decoder(
        input_ids=verify_ids,
        past_key_values=past_key_values,
        use_cache=True,
    )
    return cuda_time(device) - start


def mean(values: list[float]) -> float:
    return sum(values) / max(len(values), 1)


def get_decoder(model: AutoModelForCausalLM) -> torch.nn.Module:
    if hasattr(model, "get_decoder"):
        return model.get_decoder()
    if hasattr(model, "model"):
        return model.model
    raise AttributeError(
        f"{type(model).__name__} does not expose a decoder via get_decoder() or .model"
    )


def load_model_and_decoder(
    model_name_or_path: str,
    attn_implementation: str | None,
    device: torch.device,
) -> tuple[AutoTokenizer, AutoModelForCausalLM, torch.nn.Module]:
    model_kwargs = {"torch_dtype": torch.bfloat16}
    if attn_implementation:
        model_kwargs["attn_implementation"] = attn_implementation

    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        **model_kwargs,
    ).to(device).eval()
    decoder = get_decoder(model)
    return tokenizer, model, decoder


def write_jsonl(output_file, row: dict) -> None:
    if output_file is not None:
        output_file.write(json.dumps(row, ensure_ascii=False) + "\n")
        output_file.flush()


def profile_verify(args: argparse.Namespace, device: torch.device) -> None:
    generator = torch.Generator(device=device)
    generator.manual_seed(args.seed)

    tokenizer, model, decoder = load_model_and_decoder(
        args.model_name_or_path,
        args.attn_implementation,
        device,
    )

    vocab_size = getattr(model.config, "vocab_size", len(tokenizer))
    lengths = parse_lengths(args.lengths, args.min_len, args.max_len, args.step)
    output_file = None
    if args.output_jsonl is not None:
        args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
        output_file = args.output_jsonl.open("w", encoding="utf-8")

    print(f"model: {args.model_name_or_path}")
    print("profile: decoder-only verify forward, excluding lm_head")
    print(f"attn_implementation: {args.attn_implementation or 'default'}")
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
                run_verify_once(decoder, prefix_ids, verify_ids, device)

            timings = [
                run_verify_once(decoder, prefix_ids, verify_ids, device)
                for _ in range(args.repeat)
            ]
            avg_time = mean(timings)
            min_time = min(timings)
            verified_tokens = args.batch_size * args.verify_tokens
            row = {
                "profile": "verify",
                "attn_implementation": args.attn_implementation or "default",
                "prefix_len": prefix_len,
                "verify_tokens": args.verify_tokens,
                "batch_size": args.batch_size,
                "avg_seconds": avg_time,
                "min_seconds": min_time,
                "tokens_per_second": verified_tokens / avg_time,
                "samples": args.repeat,
                "include_lm_head": False,
            }

            print(
                f"{prefix_len:10d} {avg_time * 1000:12.3f} "
                f"{min_time * 1000:12.3f} {row['tokens_per_second']:12.2f} "
                f"{args.repeat:8d}"
            )

            write_jsonl(output_file, row)
    finally:
        if output_file is not None:
            output_file.close()


def profile_prefill(args: argparse.Namespace, device: torch.device) -> None:
    output_file = None
    if args.output_jsonl is not None:
        args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
        output_file = args.output_jsonl.open("w", encoding="utf-8")

    base_lengths = parse_lengths(args.lengths, args.min_len, args.max_len, args.step)
    length_cases = sparse_plus_extra_lengths(
        base_lengths,
        args.sparse_ratio,
        args.extra_input_len,
    )
    attn_implementations = parse_attn_implementations(args.attn_implementations)

    print(f"model: {args.model_name_or_path}")
    print("profile: decoder-only prefill, excluding lm_head")
    print(f"batch_size: {args.batch_size}")
    print(
        f"{'attn':>18} {'case':>18} {'base_len':>10} {'input_len':>10} "
        f"{'avg_ms':>12} {'min_ms':>12} {'tok/s':>12} {'samples':>8}"
    )

    try:
        for attn_implementation in attn_implementations:
            generator = torch.Generator(device=device)
            generator.manual_seed(args.seed)
            tokenizer, model, decoder = load_model_and_decoder(
                args.model_name_or_path,
                attn_implementation,
                device,
            )
            vocab_size = getattr(model.config, "vocab_size", len(tokenizer))
            attn_label = attn_implementation or "default"

            for base_len, case, input_len in length_cases:
                input_ids = random_tokens(
                    args.batch_size,
                    input_len,
                    vocab_size,
                    device,
                    generator,
                )

                for _ in range(args.warmup):
                    run_prefill_once(decoder, input_ids, device)

                timings = [
                    run_prefill_once(decoder, input_ids, device)
                    for _ in range(args.repeat)
                ]
                avg_time = mean(timings)
                min_time = min(timings)
                prefilled_tokens = args.batch_size * input_len
                row = {
                    "profile": "prefill",
                    "attn_implementation": attn_label,
                    "case": case,
                    "base_len": base_len,
                    "input_len": input_len,
                    "sparse_ratio": args.sparse_ratio if case == "sparse_plus_extra" else None,
                    "extra_input_len": args.extra_input_len,
                    "batch_size": args.batch_size,
                    "avg_seconds": avg_time,
                    "min_seconds": min_time,
                    "tokens_per_second": prefilled_tokens / avg_time,
                    "samples": args.repeat,
                    "include_lm_head": False,
                }

                print(
                    f"{attn_label:>18} {case:>18} {base_len:10d} {input_len:10d} "
                    f"{avg_time * 1000:12.3f} {min_time * 1000:12.3f} "
                    f"{row['tokens_per_second']:12.2f} {args.repeat:8d}"
                )
                write_jsonl(output_file, row)

            del decoder
            del model
            del tokenizer
            torch.cuda.empty_cache()
    finally:
        if output_file is not None:
            output_file.close()


def profile_decode(args: argparse.Namespace, device: torch.device) -> None:
    output_file = None
    if args.output_jsonl is not None:
        args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
        output_file = args.output_jsonl.open("w", encoding="utf-8")

    base_lengths = parse_lengths(args.lengths, args.min_len, args.max_len, args.step)
    length_cases = sparse_plus_extra_lengths(
        base_lengths,
        args.sparse_ratio,
        args.extra_input_len,
    )
    attn_implementations = parse_attn_implementations(args.attn_implementations)

    print(f"model: {args.model_name_or_path}")
    print("profile: decoder-only decode with existing kv cache, excluding lm_head")
    print(f"decode_tokens: {args.decode_tokens}, batch_size: {args.batch_size}")
    print(
        f"{'attn':>18} {'case':>18} {'base_len':>10} {'kv_len':>10} "
        f"{'decode':>8} {'avg_ms':>12} {'min_ms':>12} {'tok/s':>12} "
        f"{'samples':>8}"
    )

    try:
        for attn_implementation in attn_implementations:
            generator = torch.Generator(device=device)
            generator.manual_seed(args.seed)
            tokenizer, model, decoder = load_model_and_decoder(
                args.model_name_or_path,
                attn_implementation,
                device,
            )
            vocab_size = getattr(model.config, "vocab_size", len(tokenizer))
            attn_label = attn_implementation or "default"

            for base_len, case, kv_cache_len in length_cases:
                cache_input_ids = random_tokens(
                    args.batch_size,
                    kv_cache_len,
                    vocab_size,
                    device,
                    generator,
                )
                decode_ids = random_tokens(
                    args.batch_size,
                    args.decode_tokens,
                    vocab_size,
                    device,
                    generator,
                )

                for _ in range(args.warmup):
                    past_key_values = prefill_cache(decoder, cache_input_ids)
                    run_decode_once(decoder, decode_ids, past_key_values, device)
                    del past_key_values

                timings = []
                for _ in range(args.repeat):
                    past_key_values = prefill_cache(decoder, cache_input_ids)
                    timings.append(
                        run_decode_once(decoder, decode_ids, past_key_values, device)
                    )
                    del past_key_values

                avg_time = mean(timings)
                min_time = min(timings)
                decoded_tokens = args.batch_size * args.decode_tokens
                row = {
                    "profile": "decode",
                    "attn_implementation": attn_label,
                    "case": case,
                    "base_len": base_len,
                    "kv_cache_len": kv_cache_len,
                    "decode_tokens": args.decode_tokens,
                    "sparse_ratio": args.sparse_ratio if case == "sparse_plus_extra" else None,
                    "extra_input_len": args.extra_input_len,
                    "batch_size": args.batch_size,
                    "avg_seconds": avg_time,
                    "min_seconds": min_time,
                    "tokens_per_second": decoded_tokens / avg_time,
                    "samples": args.repeat,
                    "include_lm_head": False,
                    "prefill_excluded": True,
                }

                print(
                    f"{attn_label:>18} {case:>18} {base_len:10d} "
                    f"{kv_cache_len:10d} {args.decode_tokens:8d} "
                    f"{avg_time * 1000:12.3f} {min_time * 1000:12.3f} "
                    f"{row['tokens_per_second']:12.2f} {args.repeat:8d}"
                )
                write_jsonl(output_file, row)

            del decoder
            del model
            del tokenizer
            torch.cuda.empty_cache()
    finally:
        if output_file is not None:
            output_file.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Profile target-model decoder cost for verify or prefill forwards."
        )
    )
    parser.add_argument("--model-name-or-path", type=str, default=DEFAULT_MODEL_PATH)
    parser.add_argument(
        "--profile",
        choices=["verify", "prefill", "decode"],
        default="verify",
    )
    parser.add_argument("--min-len", type=int, default=2)
    parser.add_argument("--max-len", type=int, default=4)
    parser.add_argument("--step", type=int, default=1)
    parser.add_argument(
        "--lengths",
        type=str,
        default=None,
        help="Comma-separated base lengths. Overrides --min-len/--max-len/--step.",
    )
    parser.add_argument("--verify-tokens", type=int, default=2048 + 80)
    parser.add_argument("--decode-tokens", type=int, default=1)
    parser.add_argument("--sparse-ratio", type=float, default=0.3)
    parser.add_argument("--extra-input-len", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeat", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--attn-implementation", type=str, default="eager")
    parser.add_argument(
        "--attn-implementations",
        type=str,
        default="flash_attention_2,eager",
        help=(
            "Comma-separated attention implementations used by --profile prefill. "
            "Use flash_attention_2,eager to compare FlashAttention on/off."
        ),
    )
    parser.add_argument("--output-jsonl", type=Path, default=None)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("This profiling script expects a CUDA device.")

    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    if args.profile == "verify":
        profile_verify(args, device)
    elif args.profile == "prefill":
        profile_prefill(args, device)
    else:
        profile_decode(args, device)


if __name__ == "__main__":
    main()
