#!/usr/bin/env python3
"""Visualize FlashMTP spec-decode attention masks per prediction group.

**Decode chunks (``--decode-chunk-sizes``):** matches inference
``FlashMTPDraftModel.spec_generate`` — for each ``(group_start, group_end)`` from
``build_decode_chunk_prediction_groups``, builds
``create_flashmtp_full_chunk_symmetric_mask_with_chs`` (Q length ``block_size``,
KV length ``block_size + 1`` with a leading **CHS** column; no prepended pivot row
in the draft axis).

**No decode chunks (empty ``--decode-chunk-sizes``):** legacy slot-group layout via
``create_flashmtp_single_block_mask`` (short Q × long KV).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch

from specforge.modeling.draft.flashmtp import (
    build_flashmtp_prediction_groups,
    create_flashmtp_full_chunk_symmetric_mask_with_chs,
    create_flashmtp_single_block_mask,
)
from specforge.modeling.draft.flashmtp_chunk_utils import (
    build_decode_chunk_prediction_groups,
    normalize_decode_chunk_sizes,
)


def _kv_labels(kv_len: int, num_context: int = 1) -> list[str]:
    labels = [f"P{i}" for i in range(num_context)]
    labels += [f"D{j}" for j in range(kv_len - num_context)]
    return labels


def _block_slot_labels(block_size: int) -> list[str]:
    """One label per draft slot ``0 .. block_size-1`` (no separate P0 column)."""
    return [f"S{i}" for i in range(block_size)]


def _inference_kv_labels(block_size: int) -> list[str]:
    """KV axis: CHS + draft slots ``S0..``."""
    return ["CHS"] + _block_slot_labels(block_size)


def _q_labels(group_start: int, group_end: int) -> list[str]:
    return [f"Q s{group_start + i}" for i in range(group_end - group_start)]


def mask_to_bool2d(mask: torch.Tensor) -> torch.Tensor:
    """Visible positions (not masked with -inf)."""
    m = mask[0, 0]
    return m > (torch.finfo(m.dtype).min / 2)


def print_ascii(visible: torch.Tensor, q_labels: list[str], kv_labels: list[str]) -> None:
    v = visible.cpu().bool().numpy()
    col_w = max(3, max(len(c) for c in kv_labels))
    header = " " * 8 + "".join(f"{c:>{col_w}}" for c in kv_labels)
    print(header)
    for i, row in enumerate(v):
        name = q_labels[i][:10].ljust(8)
        chars = "".join(f"{'1' if x else '.':>{col_w}}" for x in row)
        print(f"{name}{chars}")


def save_text_report(
    groups_data: list[tuple[tuple[int, int], torch.Tensor, list[str], list[str]]],
    out_path: Path,
    title_prefix: str,
) -> None:
    lines: list[str] = [title_prefix, ""]
    for (gs, ge), vis, q_lab, kv_lab in groups_data:
        lines.append(f"=== group [{gs}, {ge})  KV_len={vis.shape[1]}  Q_len={vis.shape[0]} ===")
        v = vis.cpu().bool().numpy()
        col_w = max(3, max(len(c) for c in kv_lab))
        lines.append(" " * 10 + "".join(f"{c:>{col_w}}" for c in kv_lab))
        for i, row in enumerate(v):
            name = q_lab[i][:12].ljust(10)
            lines.append(name + "".join(f"{'1' if x else '.':>{col_w}}" for x in row))
        lines.append("")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf-8")


def save_figure(
    groups_data: list[tuple[tuple[int, int], torch.Tensor, list[str], list[str]]],
    out_path: Path,
    title_prefix: str,
    text_only: bool = False,
) -> None:
    if text_only:
        save_text_report(groups_data, out_path, title_prefix)
        return
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except ModuleNotFoundError:
        txt_path = out_path.with_suffix(".txt")
        save_text_report(groups_data, txt_path, title_prefix)
        print(
            f"matplotlib not installed; wrote text report to {txt_path.resolve()}",
            file=sys.stderr,
        )
        return

    n = len(groups_data)
    fig, axes = plt.subplots(n, 1, figsize=(max(10, 0.35 * len(groups_data[0][3])), 2.8 * n))
    if n == 1:
        axes = [axes]
    for ax, ((gs, ge), vis, q_lab, kv_lab) in zip(axes, groups_data):
        arr = vis.cpu().float().numpy()
        ax.imshow(
            arr,
            cmap="Blues",
            vmin=0,
            vmax=1,
            interpolation="nearest",
            aspect="auto",
            extent=(-0.5, len(kv_lab) - 0.5, len(q_lab) - 0.5, -0.5),
        )
        ax.set_xticks(range(len(kv_lab)))
        ax.set_yticks(range(len(q_lab)))
        ax.set_xticklabels(kv_lab, rotation=75, ha="right", fontsize=8)
        ax.set_yticklabels(q_lab, fontsize=8)
        ax.set_xlabel("KV index (CHS = target fused context; then draft P0/D/s)")
        ax.set_ylabel("Query index (draft positions along Q axis)")
        ax.set_title(
            f"{title_prefix} | group [{gs}, {ge})  Q×KV = {len(q_lab)}×{len(kv_lab)}"
        )
        ax.grid(which="minor", color="lightgray", linewidth=0.35)
        ax.set_xticks(np.arange(-0.5, len(kv_lab), 1), minor=True)
        ax.set_yticks(np.arange(-0.5, len(q_lab), 1), minor=True)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Visualize spec-decode single-block masks for each decode chunk group."
    )
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument(
        "--decode-chunk-sizes",
        type=str,
        default="4,4,4,4",
        help='Comma-separated chunk sizes summing to block_size, e.g. "4,4,4,4". '
        'Empty string = legacy slot-group mask (single_block), not decode-chunk inference.',
    )
    parser.add_argument(
        "--attention-backend",
        type=str,
        default="sdpa",
        choices=("sdpa", "eager", "flex_attention"),
        help="Dense tensor mask is built for sdpa/eager; flex_attention uses BlockMask (not plotted here).",
    )
    parser.add_argument("--output", type=Path, default=Path("spec_decode_masks.png"))
    parser.add_argument("--ascii", action="store_true", help="Print ASCII for each group to stdout.")
    parser.add_argument(
        "--text-only",
        action="store_true",
        help="Write a .txt grid report to --output (no matplotlib).",
    )
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    block_size = args.block_size
    device = torch.device(args.device)
    dtype = torch.float32

    decode_chunks: list[int] | None = None
    if args.decode_chunk_sizes.strip():
        decode_chunks = normalize_decode_chunk_sizes(args.decode_chunk_sizes, block_size)
        groups = build_decode_chunk_prediction_groups(decode_chunks, block_size)
        title_prefix = f"decode_chunk_sizes={decode_chunks}"
    else:
        groups = build_flashmtp_prediction_groups(block_size)
        title_prefix = "decode_chunk_sizes=None (slot_group)"

    if args.attention_backend == "flex_attention":
        print(
            "Warning: flex_attention returns BlockMask, not a dense tensor; "
            "using sdpa for visualization.",
            file=sys.stderr,
        )
        attn_backend = "sdpa"
    else:
        attn_backend = args.attention_backend

    groups_data: list[tuple[tuple[int, int], torch.Tensor, list[str], list[str]]] = []

    print(f"block_size={block_size}, {title_prefix}")
    print(f"prediction_groups={groups}\n")

    for group_start, group_end in groups:
        chunk_len = group_end - group_start
        if decode_chunks is not None:
            mask = create_flashmtp_full_chunk_symmetric_mask_with_chs(
                group_start=group_start,
                group_end=group_end,
                block_size=block_size,
                decode_chunk_sizes=decode_chunks,
                batch_size=1,
                device=device,
                attention_backend=attn_backend,
                dtype=dtype,
            )
            q_lab = _block_slot_labels(block_size)
            kv_lab = _inference_kv_labels(block_size)
        else:
            mask = create_flashmtp_single_block_mask(
                batch_size=1,
                block_size=block_size,
                device=device,
                attention_backend=attn_backend,
                dtype=dtype,
                num_context_tokens=1,
                decode_chunk_sizes=None,
                noise_seq_len=group_end,
                q_noise_seq_len=chunk_len,
                q_slot_offset=group_start,
            )
            kv_lab = _kv_labels(mask.shape[-1], num_context=1)
            q_lab = _q_labels(group_start, group_end)
        if mask is None:
            raise RuntimeError("Expected dense mask tensor (sdpa/eager), got None.")
        visible = mask_to_bool2d(mask).to(dtype)
        groups_data.append(((group_start, group_end), visible, q_lab, kv_lab))
        print(
            f"--- group [{group_start}, {group_end})  "
            f"decode_chunk_inference={decode_chunks is not None}  "
            f"mask_shape Q×KV={mask.shape[-2]}×{mask.shape[-1]} ---"
        )
        if args.ascii:
            print_ascii(visible, q_lab, kv_lab)
            print()

    save_figure(groups_data, args.output, title_prefix, text_only=args.text_only)
    if args.text_only:
        print(f"Saved text report: {args.output.resolve()}")
    elif args.output.exists():
        print(f"Saved figure: {args.output.resolve()}")
    else:
        print(f"Saved text report (matplotlib missing): {args.output.with_suffix('.txt').resolve()}")


if __name__ == "__main__":
    main()
