#!/usr/bin/env python3
"""
Streaming evaluation demo for FlashMTP / DFlash speculative decoding.

Supports five modes:
  - baseline:   original autoregressive decode
  - flashmtp:   FlashMTP speculative decode
  - dflash:     DFlash speculative decode
  - eagle:      Eagle3 speculative decode
  - both:       run baseline then flashmtp and compare side-by-side

Usage examples:
  # FlashMTP on a single prompt
  cd /share/dai-sys/wanghanzhen/projects/MTP/FlashMTP_v1.3
  source .venv/bin/activate
  python evaluation/stream_eval.py \
      --model-name-or-path /share/dai-sys/wanghanzhen/models/Qwen/Qwen3-8B \
      --draft-name-or-path /share/dai-sys/wanghanzhen/projects/MTP/FlashMTP_v1.3/cache/model/flashmtp_v1.3_2.3M \
      --mode flashmtp \
      --dataset /share/dai-sys/wanghanzhen/datasets/longbench_v2/Multi-Document_QA_sample_28.jsonl
      --prompt '''A regular hexagon can be divided into six equilateral triangles. If the perimeter of one of the triangles is 21 inches, what is the perimeter, in inches, of the regular hexagon?
Please reason step by step, and put your final answer within \boxed{}.'''  \
      --mode flashmtp

  # DFlash streaming decode (draft path has a default)
  python evaluation/stream_eval.py \
      --model-name-or-path /share/dai-sys/wanghanzhen/models/Qwen/Qwen3-8B \
      --mode dflash \
      --dataset /share/dai-sys/wanghanzhen/datasets/longbench_v2/Multi-Document_QA_sample_28.jsonl
      --prompt '''A regular hexagon can be divided into six equilateral triangles. If the perimeter of one of the triangles is 21 inches, what is the perimeter, in inches, of the regular hexagon?
Please reason step by step, and put your final answer within \boxed{}.'''

  # Eagle3 streaming decode (draft path defaults to AngelSlim Qwen3-8B_eagle3)
  uv run python evaluation/stream_eval.py \
      --model-name-or-path /share/dai-sys/wanghanzhen/models/Qwen/Qwen3-8B \
      --mode eagle \
      --prompt '''A regular hexagon can be divided into six equilateral triangles. If the perimeter of one of the triangles is 21 inches, what is the perimeter, in inches, of the regular hexagon?
Please reason step by step, and put your final answer within \boxed{}.'''

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
import re
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
from transformers import AutoModelForCausalLM, AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from specforge.modeling.auto import AutoEagle3DraftModel
from specforge.modeling.draft.dflash import DFlashDraftModel
from specforge.modeling.draft.flashmtp import FlashMTPDraftModel

from evaluation import benchmark as bench
from evaluation import distributed as dist

console = Console()

DEFAULT_DFLASH_DRAFT_PATH = "/share/dai-sys/wanghanzhen/models/z-lab/Qwen3-8B-DFlash-b16"
DEFAULT_EAGLE_DRAFT_PATH = "/share/dai-sys/wanghanzhen/models/AngelSlim/Qwen3-8B_eagle3"


def rich_print(*args, **kwargs):
    """Thread-safe rich print wrapper."""
    console.print(*args, **kwargs)


def clear_terminal() -> None:
    """Clear terminal screen before showing the question."""
    console.clear()


def prompt_preview(text: str, max_chars: int = 256) -> tuple[str | None, str]:
    """Return a complete-sentence context preview and an unabridged question."""
    separator = "\n\nQuestion:"
    context, found, question = text.rpartition(separator)
    if not found:
        if len(text) <= max_chars:
            return None, text
        context, question = text, ""

    if len(context) <= max_chars:
        context_preview = context
    else:
        sentences = [
            match.group(0).strip()
            for match in re.finditer(r".+?(?:[.!?。！？](?=\s|$)|$)", context, re.DOTALL)
            if match.group(0).strip()
        ]
        if len(sentences) >= 2:
            context_preview = (
                f"{sentences[0]}\n\n"
                "────────────  ⋯  ────────────\n\n"
                f"{sentences[-1]}"
            )
        else:
            paragraphs = [part.strip() for part in context.splitlines() if part.strip()]
            context_preview = (
                f"{paragraphs[0]}\n\n"
                "────────────  ⋯  ────────────\n\n"
                f"{paragraphs[-1]}"
                if len(paragraphs) >= 2
                else context
            )

    return context_preview, question.lstrip()


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
        """Callback compatible with bench.target_generate / flashmtp_generate / dflash_generate."""
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
    """Load target model and optionally FlashMTP / DFlash draft model."""

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
    elif args.mode == "dflash":
        draft_path = args.draft_name_or_path or DEFAULT_DFLASH_DRAFT_PATH
        rich_print(f"[bold cyan]Loading DFlash draft:[/] {draft_path}")
        draft_model = DFlashDraftModel.from_pretrained(
            draft_path,
            attn_implementation="flash_attention_2" if use_flash else "sdpa",
            dtype=torch.bfloat16,
        ).to(device)
        draft_model.eval()
    elif args.mode == "eagle":
        draft_path = args.draft_name_or_path or DEFAULT_EAGLE_DRAFT_PATH
        rich_print(f"[bold cyan]Loading Eagle draft:[/] {draft_path}")
        draft_model = AutoEagle3DraftModel.from_pretrained(
            draft_path,
            attention_backend="sdpa",
            torch_dtype=torch.bfloat16,
        ).to(device)
        draft_model.eval()

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    return target, draft_model, tokenizer


def resolve_block_size(draft_model, args) -> int:
    """Return effective block_size for speculative decoding."""
    if args.block_size is not None:
        return args.block_size
    if args.mode == "eagle":
        return bench.resolve_eagle_block_size(draft_model, args.block_size)
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


def run_dflash(
    draft_model,
    target,
    input_ids: torch.Tensor,
    mask_token_id: int,
    max_new_tokens: int,
    block_size: int,
    temperature: float,
    stop_token_ids: list[int],
    batch_size: int,
    stream_callback: Callable[[torch.Tensor], None] | None = None,
) -> SimpleNamespace:
    """Run DFlash speculative decode and return result namespace."""
    return bench.dflash_generate(
        model=draft_model,
        target=target,
        input_ids=input_ids,
        mask_token_id=mask_token_id,
        max_new_tokens=max_new_tokens,
        block_size=block_size,
        stop_token_ids=stop_token_ids,
        temperature=temperature,
        batch_size=batch_size,
        stream_callback=stream_callback,
    )

def run_eagle(
    draft_model,
    target,
    input_ids: torch.Tensor,
    max_new_tokens: int,
    block_size: int,
    temperature: float,
    stop_token_ids: list[int],
    stream_callback: Callable[[torch.Tensor], None] | None = None,
) -> SimpleNamespace:
    """Run Eagle3 speculative decode and return result namespace."""
    return bench.eagle_generate(
        model=draft_model,
        target=target,
        input_ids=input_ids,
        max_new_tokens=max_new_tokens,
        block_size=block_size,
        stop_token_ids=stop_token_ids,
        temperature=temperature,
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


def print_mode_stats(label: str, result: SimpleNamespace, block_size: int | None = None) -> None:
    """Print throughput and latency stats for one generation run."""
    parts = [
        f"[bold]{label}[/]",
        f"tokens={result.num_output_tokens}",
        f"throughput={result.throughput_tokens_per_sec:.2f} tok/s",
        f"latency={result.time_per_output_token * 1000:.2f} ms/token",
    ]
    accept_lengths = getattr(result, "acceptance_lengths", None)
    if accept_lengths and block_size:
        avg_accept = float(np.mean(accept_lengths))
        parts.append(f"avg_accept={avg_accept:.2f}/{block_size}")
        parts.append(f"verify_steps={len(accept_lengths)}")
    rich_print("  " + " | ".join(parts))


def print_sample_stats(
    baseline_result: SimpleNamespace | None,
    flashmtp_result: SimpleNamespace | None,
    dflash_result: SimpleNamespace | None,
    eagle_result: SimpleNamespace | None,
    block_size: int | None,
) -> None:
    """Print per-sample metrics after generation finishes."""
    rich_print()
    rich_print("[bold cyan]Metrics[/]")
    if baseline_result is not None:
        print_mode_stats("Baseline", baseline_result)
    if flashmtp_result is not None:
        print_mode_stats("FlashMTP", flashmtp_result, block_size)
    if dflash_result is not None:
        print_mode_stats("DFlash", dflash_result, block_size)
    if eagle_result is not None:
        print_mode_stats("Eagle", eagle_result, block_size)
    if flashmtp_result is not None and baseline_result is not None:
        speedup = (
            baseline_result.throughput_tokens_per_sec
            / max(flashmtp_result.throughput_tokens_per_sec, 1e-9)
        )
        rich_print(f"  [bold]Speedup (Baseline / FlashMTP):[/] {speedup:.2f}x")


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
    mask_token_id: int | None = None,
) -> tuple[
    SimpleNamespace | None,
    SimpleNamespace | None,
    SimpleNamespace | None,
    SimpleNamespace | None,
]:
    """Run generation for one question and stream results as they are produced."""
    input_ids = build_prompt_input(tokenizer, user_text, target.device, args.batch_size)
    input_len = input_ids.shape[1]

    if dist.is_main():
        clear_terminal()

    rich_print()
    rich_print("=" * console.width)
    rich_print(
        f"[bold blue]Sample {idx} | Turn {turn_index}[/]  "
        f"Input length: {input_len} tokens | Batch size: {args.batch_size}"
    )
    context_preview, question = prompt_preview(user_text)
    if context_preview is not None:
        rich_print(Panel(
            context_preview,
            title="[bold cyan]Context Preview[/]",
            border_style="cyan",
            padding=(1, 2),
        ))
    if question:
        rich_print(Panel(
            question,
            title="[bold blue]Question[/]",
            border_style="blue",
            padding=(1, 2),
        ))

    baseline_result = None
    flashmtp_result = None
    dflash_result = None
    eagle_result = None

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

    if args.mode == "dflash":
        rich_print(Panel.fit(
            "[bold magenta]DFlash Answer[/]",
            border_style="magenta",
            padding=(0, 2),
        ))
        sys.stdout.write("\n")
        sys.stdout.flush()
        decoder = StreamingDecoder(tokenizer, input_len)
        dflash_result = run_dflash(
            draft_model=draft_model,
            target=target,
            input_ids=input_ids,
            mask_token_id=mask_token_id,
            max_new_tokens=args.max_new_tokens,
            block_size=block_size,
            temperature=args.temperature,
            stop_token_ids=stop_token_ids,
            batch_size=args.batch_size,
            stream_callback=decoder,
        )
        sys.stdout.write("\n")
        sys.stdout.flush()

    if args.mode == "eagle":
        rich_print(Panel.fit(
            "[bold bright_blue]Eagle Answer[/]",
            border_style="bright_blue",
            padding=(0, 2),
        ))
        sys.stdout.write("\n")
        sys.stdout.flush()
        decoder = StreamingDecoder(tokenizer, input_len)
        eagle_result = run_eagle(
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

    if dist.is_main():
        print_sample_stats(
            baseline_result,
            flashmtp_result,
            dflash_result,
            eagle_result,
            block_size,
        )

    return baseline_result, flashmtp_result, dflash_result, eagle_result


def make_summary_line(
    baseline_results: list[SimpleNamespace],
    flashmtp_results: list[SimpleNamespace],
    dflash_results: list[SimpleNamespace],
    eagle_results: list[SimpleNamespace],
    mode: str,
    block_size: int | None,
) -> Text:
    """Return a prominent summary renderable for the selected mode(s)."""

    def avg(values):
        return float(np.mean(values)) if values else 0.0

    def spec_summary(label: str, results: list[SimpleNamespace]) -> str:
        tps = avg([r.throughput_tokens_per_sec for r in results])
        ms = avg([r.time_per_output_token * 1000 for r in results])
        all_accept = list(np.concatenate([
            r.acceptance_lengths for r in results if r.acceptance_lengths
        ])) if results else []
        avg_accept = float(np.mean(all_accept)) if all_accept else 0.0
        accept_str = f"{avg_accept:.2f}"
        if block_size:
            accept_str += f"/{block_size}"
        return f"{label}: {tps:.2f} tok/s, {ms:.2f} ms/token, avg_accept={accept_str}"

    parts: list[str] = []

    if mode in ("baseline", "both") and baseline_results:
        tps = avg([r.throughput_tokens_per_sec for r in baseline_results])
        ms = avg([r.time_per_output_token * 1000 for r in baseline_results])
        parts.append(f"Baseline: {tps:.2f} tok/s, {ms:.2f} ms/token")

    if mode in ("flashmtp", "both") and flashmtp_results:
        flash_part = spec_summary("FlashMTP", flashmtp_results)
        if mode == "both" and baseline_results:
            base_tps = avg([r.throughput_tokens_per_sec for r in baseline_results])
            flash_tps = avg([r.throughput_tokens_per_sec for r in flashmtp_results])
            flash_part += f", speedup={base_tps / max(flash_tps, 1e-9):.2f}x"
        parts.append(flash_part)

    if mode == "dflash" and dflash_results:
        parts.append(spec_summary("DFlash", dflash_results))

    if mode == "eagle" and eagle_results:
        parts.append(spec_summary("Eagle", eagle_results))

    summary = Text(" | ".join(parts), style="bold bright_cyan")
    return Panel.fit(
        Align.center(summary),
        border_style="bright_cyan",
        padding=(1, 4),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Streaming speculative decoding evaluation demo")
    parser.add_argument("--model-name-or-path", type=str, required=True)
    parser.add_argument(
        "--draft-name-or-path",
        type=str,
        default=None,
        help=(
            "Draft checkpoint path. "
            f"For dflash mode, defaults to {DEFAULT_DFLASH_DRAFT_PATH}; "
            f"for eagle mode, defaults to {DEFAULT_EAGLE_DRAFT_PATH}"
        ),
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="both",
        choices=["baseline", "flashmtp", "dflash", "eagle", "both"],
    )
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
    if args.mode == "eagle" and args.batch_size != 1:
        parser.error("eagle mode currently supports --batch-size 1 only.")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    dist.init()
    torch.cuda.set_device(dist.local_rank())
    device = torch.device(f"cuda:{dist.local_rank()}")

    target, draft_model, tokenizer = load_models(args, device)
    block_size = resolve_block_size(draft_model, args) if draft_model else None
    mask_token_id = (
        bench.resolve_mask_token_id(draft_model, tokenizer)
        if args.mode == "dflash"
        else None
    )

    stop_token_ids = [tid for tid in [tokenizer.eos_token_id] if tid is not None]

    if args.prompt:
        dataset = [{"turns": [args.prompt]}]
    else:
        dataset = bench.load_benchmark_dataset(args.dataset)
        dataset = bench.select_max_samples(dataset, args.max_samples)

    if dist.is_main():
        rich_print(Panel.fit(
            Align.center(
                Text("Speculative Decoding Streaming Evaluation", style="bold cyan", justify="center")
            ),
            border_style="cyan",
            padding=(1, 4),
        ))
        rich_print(f"[bold]Mode:[/] {args.mode} | [bold]Dataset:[/] {args.dataset or 'single prompt'}")
        rich_print(f"[bold]Max new tokens:[/] {args.max_new_tokens} | [bold]Temperature:[/] {args.temperature}")
        if block_size:
            if args.mode == "dflash":
                label = "DFlash"
            elif args.mode == "eagle":
                label = "Eagle"
            else:
                label = "FlashMTP"
            rich_print(f"[bold]{label} block size:[/] {block_size}")

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
    if args.mode == "dflash":
        bench.dflash_generate(
            model=draft_model,
            target=target,
            input_ids=warmup_ids,
            mask_token_id=mask_token_id,
            max_new_tokens=warmup_new_tokens,
            block_size=block_size,
            stop_token_ids=stop_token_ids,
            temperature=args.temperature,
            batch_size=args.batch_size,
        )
    if args.mode == "eagle":
        bench.eagle_generate(
            model=draft_model,
            target=target,
            input_ids=warmup_ids,
            max_new_tokens=warmup_new_tokens,
            block_size=block_size,
            stop_token_ids=stop_token_ids,
            temperature=args.temperature,
        )
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    bench.cuda_time()

    baseline_results: list[SimpleNamespace] = []
    flashmtp_results: list[SimpleNamespace] = []
    dflash_results: list[SimpleNamespace] = []
    eagle_results: list[SimpleNamespace] = []

    indices = range(dist.rank(), len(dataset), dist.size())
    for idx in indices:
        instance = dataset[idx]
        for turn_index, turn_q in enumerate(instance["turns"]):
            base_res, flash_res, dflash_res, eagle_res = run_one_sample(
                idx=idx,
                turn_index=turn_index,
                user_text=turn_q,
                target=target,
                draft_model=draft_model,
                tokenizer=tokenizer,
                args=args,
                block_size=block_size,
                stop_token_ids=stop_token_ids,
                mask_token_id=mask_token_id,
            )
            if base_res is not None:
                baseline_results.append(base_res)
            if flash_res is not None:
                flashmtp_results.append(flash_res)
            if dflash_res is not None:
                dflash_results.append(dflash_res)
            if eagle_res is not None:
                eagle_results.append(eagle_res)

    if dist.size() > 1:
        baseline_results = dist.gather(baseline_results, dst=0)
        flashmtp_results = dist.gather(flashmtp_results, dst=0)
        dflash_results = dist.gather(dflash_results, dst=0)
        eagle_results = dist.gather(eagle_results, dst=0)
        if not dist.is_main():
            return
        baseline_results = [r for sub in baseline_results for r in sub]
        flashmtp_results = [r for sub in flashmtp_results for r in sub]
        dflash_results = [r for sub in dflash_results for r in sub]
        eagle_results = [r for sub in eagle_results for r in sub]

    # if dist.is_main():
    #     rich_print()
    #     rich_print("=" * console.width)
    #     rich_print("[bold]Overall Summary[/]")
    #     rich_print(
    #         make_summary_line(
    #             baseline_results,
    #             flashmtp_results,
    #             dflash_results,
    #             eagle_results,
    #             args.mode,
    #             block_size,
    #         )
    #     )


if __name__ == "__main__":
    main()
