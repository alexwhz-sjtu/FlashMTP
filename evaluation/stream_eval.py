#!/usr/bin/env python3
"""
Streaming evaluation demo for FlashMTP speculative decoding.

Supports three modes:
  - baseline:   original autoregressive decode
  - flashmtp:   FlashMTP speculative decode
  - both:       run baseline then flashmtp and compare side-by-side

Usage examples:
  # FlashMTP on a benchmark dataset
  cd /inspire/hdd/project/inference-chip/xujiaming-253308120313/whz/FlashMTP_v1.3
  source .venv/bin/activate
  python evaluation/stream_eval.py \
      --model-name-or-path /inspire/hdd/project/inference-chip/xujiaming-253308120313/whz/models/Qwen/Qwen3-8B \
      --draft-name-or-path /inspire/hdd/project/inference-chip/xujiaming-253308120313/whz/FlashMTP_v1.3/cache/models/flashmtp_fuse16_sample_2.3M_think_off_block_16_maxlen40960_ep8_ms_2_klw1.0_top128_ceg7_dkg14_dposall_ceposall/epoch_8_step_298432 \
      --prompt '''Sarah went to buy books from the store and spent $300 on the books. If each book was $15 and she gave an equal number of books to her 4 kids, how many books did each child get?
Please reason step by step, and put your final answer within \boxed{}.''' \
      --mode flashmtp

  # Side-by-side comparison on a single prompt
  python evaluation/stream_eval.py \
      --model-name-or-path /path/to/Qwen3-8B \
      --draft-name-or-path /path/to/flashmtp-draft \
      --mode both \
      --prompt "Explain quantum computing in simple terms."
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Callable

import numpy as np
import torch
from loguru import logger
from rich import box
from rich.align import Align
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from specforge.modeling.draft.flashmtp import FlashMTPDraftModel

from evaluation import benchmark as bench
from evaluation import distributed as dist

console = Console()


def rich_print(*args, **kwargs):
    """Thread-safe rich print wrapper."""
    console.print(*args, **kwargs)


class StreamingDecoder:
    """Incremental decoder that streams newly generated text to stdout.

    The tokenizer is invoked on the full generated token sequence each time so
    multi-token characters are handled correctly; only the suffix that has not
    yet been printed is written out.
    """

    def __init__(self, tokenizer, num_input_tokens: int):
        self.tokenizer = tokenizer
        self.num_input_tokens = num_input_tokens
        self.printed_text = ""

    def __call__(self, output_ids: torch.Tensor) -> None:
        """Callback compatible with bench.target_generate / flashmtp_generate."""
        generated_ids = output_ids[0, self.num_input_tokens :].tolist()
        full_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)

        # Find longest common prefix with already-printed text.
        common_len = 0
        max_common = min(len(full_text), len(self.printed_text))
        while common_len < max_common and full_text[common_len] == self.printed_text[common_len]:
            common_len += 1

        new_text = full_text[common_len:]
        if new_text:
            sys.stdout.write(new_text)
            sys.stdout.flush()
        self.printed_text = full_text


def load_models(args, device: torch.device):
    """Load target model and optionally FlashMTP draft model."""

    def has_flash_attn() -> bool:
        try:
            import flash_attn  # noqa: F401
            return True
        except ImportError:
            logger.warning("flash_attn not installed; falling back to sdpa.")
            return False

    use_flash = has_flash_attn()

    rich_print(f"[bold cyan]Loading target model:[/] {args.model_name_or_path}")
    target = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        attn_implementation="flash_attention_2" if use_flash else "sdpa",
        dtype=torch.bfloat16,
    ).to(device)
    target.eval()

    draft_model = None
    if args.mode in ("flashmtp", "both"):
        if not args.draft_name_or_path:
            raise ValueError("--draft-name-or-path is required for flashmtp/both mode.")
        rich_print(f"[bold cyan]Loading FlashMTP draft:[/] {args.draft_name_or_path}")
        draft_model = FlashMTPDraftModel.from_pretrained(
            args.draft_name_or_path,
            attn_implementation="flash_attention_2" if use_flash else "sdpa",
            dtype=torch.bfloat16,
        ).to(device)
        draft_model.eval()

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    return target, draft_model, tokenizer


def resolve_block_size(draft_model, args) -> int:
    """Return effective block_size for FlashMTP."""
    if args.block_size is not None:
        return args.block_size
    return getattr(draft_model, "block_size", 16)


def build_prompt_input(tokenizer, user_text: str, device: torch.device, batch_size: int):
    """Build chat-formatted input_ids from a single user turn."""
    messages = [{"role": "user", "content": user_text}]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    input_ids = tokenizer.encode(text, return_tensors="pt").to(device)
    if batch_size > 1:
        input_ids = input_ids.expand(batch_size, -1).contiguous()
    return input_ids


def run_baseline(
    target,
    tokenizer,
    input_ids: torch.Tensor,
    max_new_tokens: int,
    temperature: float,
    stop_token_ids: list[int],
    stream_callback: Callable[[torch.Tensor], None] | None = None,
) -> SimpleNamespace:
    """Run original autoregressive decode and return result namespace."""
    return bench.target_generate(
        target=target,
        input_ids=input_ids,
        max_new_tokens=max_new_tokens,
        stop_token_ids=stop_token_ids,
        temperature=temperature,
        decode_timing_after_first_token=False,
        stream_callback=stream_callback,
    )


def run_flashmtp(
    draft_model,
    target,
    input_ids: torch.Tensor,
    max_new_tokens: int,
    block_size: int,
    temperature: float,
    stop_token_ids: list[int],
    stream_callback: Callable[[torch.Tensor], None] | None = None,
) -> SimpleNamespace:
    """Run FlashMTP speculative decode and return result namespace."""
    return bench.flashmtp_generate(
        model=draft_model,
        target=target,
        input_ids=input_ids,
        max_new_tokens=max_new_tokens,
        block_size=block_size,
        stop_token_ids=stop_token_ids,
        temperature=temperature,
        decode_timing_after_first_token=False,
        stream_callback=stream_callback,
    )


def decode_response(result: SimpleNamespace, tokenizer) -> str:
    """Decode generated tokens (first batch row) skipping special tokens."""
    generated_ids = result.output_ids[0, result.num_input_tokens :]
    return tokenizer.decode(generated_ids, skip_special_tokens=True).strip()


def make_stats_table(
    mode: str,
    baseline_result: SimpleNamespace | None,
    flashmtp_result: SimpleNamespace | None,
    block_size: int | None,
) -> Table:
    """Build a rich table comparing baseline and FlashMTP metrics."""
    table = Table(
        title="Generation Metrics",
        box=box.ROUNDED,
        show_header=True,
        header_style="bold magenta",
    )
    table.add_column("Metric", justify="left", style="cyan", no_wrap=True)
    table.add_column("Baseline", justify="right", style="green")
    if flashmtp_result is not None:
        table.add_column("FlashMTP", justify="right", style="bright_yellow")
        if baseline_result is not None:
            table.add_column("Speedup", justify="right", style="bold red")

    def row(metric: str, base_val, flash_val=None, speedup=None):
        cols = [metric, base_val]
        if flashmtp_result is not None:
            cols.append(flash_val)
            if baseline_result is not None:
                cols.append(speedup)
        table.add_row(*cols)

    base_tok = baseline_result.num_output_tokens if baseline_result else 0
    flash_tok = flashmtp_result.num_output_tokens if flashmtp_result else 0

    row(
        "Output tokens",
        str(base_tok),
        str(flash_tok),
    )
    row(
        "Decode wall time (s)",
        f"{baseline_result.decode_wall_time:.3f}" if baseline_result else "N/A",
        f"{flashmtp_result.decode_wall_time:.3f}" if flashmtp_result else "N/A",
    )
    row(
        "Throughput (tok/s)",
        f"{baseline_result.throughput_tokens_per_sec:.2f}" if baseline_result else "N/A",
        f"{flashmtp_result.throughput_tokens_per_sec:.2f}" if flashmtp_result else "N/A",
        (
            f"{baseline_result.throughput_tokens_per_sec / max(flashmtp_result.throughput_tokens_per_sec, 1e-9):.2f}×"
            if baseline_result and flashmtp_result
            else "N/A"
        ),
    )
    row(
        "Time per output token (ms)",
        f"{baseline_result.time_per_output_token * 1000:.2f}" if baseline_result else "N/A",
        f"{flashmtp_result.time_per_output_token * 1000:.2f}" if flashmtp_result else "N/A",
    )

    if flashmtp_result is not None:
        accept_lengths = flashmtp_result.acceptance_lengths
        avg_accept = float(np.mean(accept_lengths)) if accept_lengths else 0.0
        total_steps = len(accept_lengths)
        row(
            "Avg. accepted length",
            "—",
            f"{avg_accept:.2f} / {block_size}",
        )
        row(
            "Total verify steps",
            "—",
            str(total_steps),
        )

    return table


def run_one_sample(
    idx: int,
    turn_index: int,
    user_text: str,
    target,
    draft_model,
    tokenizer,
    args,
    block_size: int | None,
    stop_token_ids: list[int],
) -> tuple[SimpleNamespace | None, SimpleNamespace | None]:
    """Run generation for one question and stream results as they are produced."""
    input_ids = build_prompt_input(tokenizer, user_text, target.device, args.batch_size)
    input_len = input_ids.shape[1]

    rich_print()
    rich_print("=" * console.width)
    rich_print(
        f"[bold blue]Sample {idx} | Turn {turn_index}[/]  "
        f"Input length: {input_len} tokens | Batch size: {args.batch_size}"
    )
    rich_print(Panel(
        user_text,
        title="[bold]Question[/]",
        border_style="blue",
        padding=(1, 2),
    ))

    baseline_result = None
    flashmtp_result = None

    if args.mode in ("baseline", "both"):
        rich_print(Panel.fit(
            "[bold green]Baseline Answer[/]",
            border_style="green",
            padding=(0, 2),
        ))
        sys.stdout.write("\n")
        sys.stdout.flush()
        decoder = StreamingDecoder(tokenizer, input_len)
        baseline_result = run_baseline(
            target=target,
            tokenizer=tokenizer,
            input_ids=input_ids,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            stop_token_ids=stop_token_ids,
            stream_callback=decoder,
        )
        sys.stdout.write("\n")
        sys.stdout.flush()

    if args.mode in ("flashmtp", "both"):
        rich_print(Panel.fit(
            "[bold bright_yellow]FlashMTP Answer[/]",
            border_style="bright_yellow",
            padding=(0, 2),
        ))
        sys.stdout.write("\n")
        sys.stdout.flush()
        decoder = StreamingDecoder(tokenizer, input_len)
        flashmtp_result = run_flashmtp(
            draft_model=draft_model,
            target=target,
            input_ids=input_ids,
            max_new_tokens=args.max_new_tokens,
            block_size=block_size,
            temperature=args.temperature,
            stop_token_ids=stop_token_ids,
            stream_callback=decoder,
        )
        sys.stdout.write("\n")
        sys.stdout.flush()

    rich_print(make_stats_table(args.mode, baseline_result, flashmtp_result, block_size))
    return baseline_result, flashmtp_result


def make_summary_line(
    baseline_results: list[SimpleNamespace],
    flashmtp_results: list[SimpleNamespace],
    mode: str,
    block_size: int | None,
) -> Text:
    """Return a single-line summary for the selected mode(s)."""

    def avg(values):
        return float(np.mean(values)) if values else 0.0

    parts: list[str] = []

    if mode in ("baseline", "both") and baseline_results:
        tps = avg([r.throughput_tokens_per_sec for r in baseline_results])
        ms = avg([r.time_per_output_token * 1000 for r in baseline_results])
        parts.append(f"Baseline: {tps:.2f} tok/s, {ms:.2f} ms/token")

    if mode in ("flashmtp", "both") and flashmtp_results:
        tps = avg([r.throughput_tokens_per_sec for r in flashmtp_results])
        ms = avg([r.time_per_output_token * 1000 for r in flashmtp_results])
        all_accept = list(np.concatenate([
            r.acceptance_lengths for r in flashmtp_results if r.acceptance_lengths
        ])) if flashmtp_results else []
        avg_accept = float(np.mean(all_accept)) if all_accept else 0.0
        accept_str = f"{avg_accept:.2f}"
        if block_size:
            accept_str += f"/{block_size}"
        flash_part = f"FlashMTP: {tps:.2f} tok/s, {ms:.2f} ms/token, avg_accept={accept_str}"
        if mode == "both" and baseline_results:
            base_tps = avg([r.throughput_tokens_per_sec for r in baseline_results])
            speedup = base_tps / max(tps, 1e-9)
            flash_part += f", speedup={speedup:.2f}x"
        parts.append(flash_part)

    return Text(" | ".join(parts), style="bold cyan")


def main() -> None:
    parser = argparse.ArgumentParser(description="Streaming FlashMTP evaluation demo")
    parser.add_argument("--model-name-or-path", type=str, required=True)
    parser.add_argument("--draft-name-or-path", type=str, default=None)
    parser.add_argument("--mode", type=str, default="both", choices=["baseline", "flashmtp", "both"])
    parser.add_argument("--dataset", type=str, default=None)
    parser.add_argument("--prompt", type=str, default=None)
    parser.add_argument("--max-samples", type=int, default=10)
    parser.add_argument("--max-new-tokens", type=int, default=4096)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--block-size", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    if args.prompt is None and args.dataset is None:
        parser.error("Either --prompt or --dataset must be provided.")
    if args.mode in ("flashmtp", "both") and not args.draft_name_or_path:
        parser.error("--draft-name-or-path is required for flashmtp/both mode.")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    dist.init()
    torch.cuda.set_device(dist.local_rank())
    device = torch.device(f"cuda:{dist.local_rank()}")

    target, draft_model, tokenizer = load_models(args, device)
    block_size = resolve_block_size(draft_model, args) if draft_model else None

    stop_token_ids = [tid for tid in [tokenizer.eos_token_id] if tid is not None]

    if args.prompt:
        dataset = [{"turns": [args.prompt]}]
    else:
        dataset = bench.load_benchmark_dataset(args.dataset)
        dataset = bench.select_max_samples(dataset, args.max_samples)

    if dist.is_main():
        rich_print(Panel.fit(
            Align.center(
                Text("FlashMTP Streaming Evaluation", style="bold cyan", justify="center")
            ),
            border_style="cyan",
            padding=(1, 4),
        ))
        rich_print(f"[bold]Mode:[/] {args.mode} | [bold]Dataset:[/] {args.dataset or 'single prompt'}")
        rich_print(f"[bold]Max new tokens:[/] {args.max_new_tokens} | [bold]Temperature:[/] {args.temperature}")
        if block_size:
            rich_print(f"[bold]FlashMTP block size:[/] {block_size}")

    # Warmup to avoid timing noise on first sample.
    if dist.is_main():
        rich_print("[dim]Running CUDA warmup...[/]")
    warmup_prompt = [{"role": "user", "content": "Warmup."}]
    warmup_text = tokenizer.apply_chat_template(
        warmup_prompt, tokenize=False, add_generation_prompt=True, enable_thinking=False
    )
    warmup_ids = tokenizer.encode(warmup_text, return_tensors="pt").to(device)
    if args.batch_size > 1:
        warmup_ids = warmup_ids.expand(args.batch_size, -1).contiguous()
    warmup_new_tokens = min(16, args.max_new_tokens)

    if args.mode in ("baseline", "both"):
        bench.target_generate(
            target=target,
            input_ids=warmup_ids,
            max_new_tokens=warmup_new_tokens,
            stop_token_ids=stop_token_ids,
            temperature=args.temperature,
            decode_timing_after_first_token=False,
        )
    if args.mode in ("flashmtp", "both"):
        bench.flashmtp_generate(
            model=draft_model,
            target=target,
            input_ids=warmup_ids,
            max_new_tokens=warmup_new_tokens,
            block_size=block_size,
            stop_token_ids=stop_token_ids,
            temperature=args.temperature,
            decode_timing_after_first_token=False,
        )
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    bench.cuda_time()

    baseline_results: list[SimpleNamespace] = []
    flashmtp_results: list[SimpleNamespace] = []

    indices = range(dist.rank(), len(dataset), dist.size())
    iterator = tqdm(indices, disable=not dist.is_main())
    for idx in iterator:
        instance = dataset[idx]
        for turn_index, turn_q in enumerate(instance["turns"]):
            base_res, flash_res = run_one_sample(
                idx=idx,
                turn_index=turn_index,
                user_text=turn_q,
                target=target,
                draft_model=draft_model,
                tokenizer=tokenizer,
                args=args,
                block_size=block_size,
                stop_token_ids=stop_token_ids,
            )
            if base_res is not None:
                baseline_results.append(base_res)
            if flash_res is not None:
                flashmtp_results.append(flash_res)

    if dist.size() > 1:
        baseline_results = dist.gather(baseline_results, dst=0)
        flashmtp_results = dist.gather(flashmtp_results, dst=0)
        if not dist.is_main():
            return
        baseline_results = [r for sub in baseline_results for r in sub]
        flashmtp_results = [r for sub in flashmtp_results for r in sub]

    if dist.is_main():
        rich_print()
        rich_print("=" * console.width)
        rich_print(f"[bold]Summary:[/] {make_summary_line(baseline_results, flashmtp_results, args.mode, block_size)}")
        rich_print("[bold green]Done.[/]")


if __name__ == "__main__":
    main()
