#!/usr/bin/env python3
"""Visualize FlashMTP clean/mask stream training attention masks."""

import argparse
from pathlib import Path
from typing import Optional, Sequence

from specforge.modeling.draft.flashmtp import (
    build_flashmtp_slot_to_chunk,
    resolve_flashmtp_chunk_sizes,
)


def build_labels(block_size: int, num_blocks: int):
    q_labels = []
    kv_labels = [f"H{b}" for b in range(num_blocks)]
    for block in range(num_blocks):
        q_labels.extend(f"C{block}:{slot}" for slot in range(block_size))
        q_labels.extend(f"M{block}:{slot}" for slot in range(block_size))
        kv_labels.extend(f"C{block}:{slot}" for slot in range(block_size))
        kv_labels.extend(f"M{block}:{slot}" for slot in range(block_size))
    return q_labels, kv_labels


def is_visible(
    q_idx: int,
    kv_idx: int,
    block_size: int,
    num_blocks: int,
    slot_to_chunk: list[int],
) -> bool:
    stream_block_size = 2 * block_size
    total_chs_len = num_blocks

    q_block = q_idx // stream_block_size
    q_stream_offset = q_idx % stream_block_size
    q_is_mask = q_stream_offset >= block_size
    q_slot = q_stream_offset % block_size
    q_group = slot_to_chunk[q_slot]

    if kv_idx < total_chs_len:
        return kv_idx == q_block

    draft_kv_idx = kv_idx - total_chs_len
    kv_block = draft_kv_idx // stream_block_size
    kv_stream_offset = draft_kv_idx % stream_block_size
    kv_is_mask = kv_stream_offset >= block_size
    kv_slot = kv_stream_offset % block_size
    kv_group = slot_to_chunk[kv_slot]

    if kv_block != q_block:
        return False

    if q_is_mask:
        return ((not kv_is_mask) and kv_group < q_group) or (
            kv_is_mask and kv_group == q_group
        )

    return (not kv_is_mask) and kv_group <= q_group


def build_mask(
    block_size: int,
    num_blocks: int,
    chunk_sizes: Optional[Sequence[int]] = None,
):
    resolved = resolve_flashmtp_chunk_sizes(block_size, chunk_sizes)
    slot_to_chunk = build_flashmtp_slot_to_chunk(resolved).tolist()
    q_len = num_blocks * 2 * block_size
    kv_len = num_blocks + num_blocks * 2 * block_size
    return [
        [is_visible(q, kv, block_size, num_blocks, slot_to_chunk) for kv in range(kv_len)]
        for q in range(q_len)
    ]


def print_ascii(mask, q_labels, kv_labels, max_label_width: int = 6):
    label_width = max(max(len(label) for label in q_labels), max_label_width)
    header = " " * (label_width + 1)
    header += "".join(label[-2:].rjust(3) for label in kv_labels)
    print(header)
    print(" " * (label_width + 1) + "".join("---" for _ in kv_labels))

    for q_label, row in zip(q_labels, mask):
        cells = []
        for kv_label, visible in zip(kv_labels, row):
            if not visible:
                cells.append("  .")
            elif kv_label.startswith("H"):
                cells.append("  H")
            elif kv_label.startswith("C"):
                cells.append("  c")
            else:
                cells.append("  m")
        print(q_label.rjust(label_width) + " " + "".join(cells))

    print("\nLegend: H=own CHS pivot, c=visible clean token, m=visible mask token, .=hidden")
    print("Rows are queries; columns are keys/values.")


def save_png(mask, q_labels, kv_labels, output_path: Path):
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError as exc:
        raise SystemExit(
            "matplotlib and numpy are required for --output. "
            "Run without --output for ASCII visualization."
        ) from exc

    arr = np.array(mask, dtype=float)
    fig_width = max(8, len(kv_labels) * 0.32)
    fig_height = max(5, len(q_labels) * 0.28)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    ax.imshow(
        arr,
        cmap="Greys",
        interpolation="nearest",
        aspect="auto",
        extent=(-0.5, len(kv_labels) - 0.5, len(q_labels) - 0.5, -0.5),
    )
    ax.set_xticks(range(len(kv_labels)))
    ax.set_yticks(range(len(q_labels)))
    ax.set_xticklabels(kv_labels, rotation=90, fontsize=7)
    ax.set_yticklabels(q_labels, fontsize=7)
    ax.tick_params(axis="both", which="major", length=0, pad=2)
    ax.set_xticks(np.arange(-0.5, len(kv_labels), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(q_labels), 1), minor=True)
    ax.set_xlabel("KV: CHS, clean stream, mask stream")
    ax.set_ylabel("Q: clean stream, mask stream")
    ax.set_title("FlashMTP training attention mask")
    ax.grid(which="minor", color="lightgray", linewidth=0.4)
    ax.grid(which="major", visible=False)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def parse_chunk_sizes(value: Optional[str], block_size: int) -> Optional[list[int]]:
    if value is None:
        return None
    return [int(x.strip()) for x in value.split(",") if x.strip()]


def main():
    parser = argparse.ArgumentParser(
        description="Visualize FlashMTP clean/mask stream training attention mask."
    )
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--num-blocks", type=int, default=1)
    parser.add_argument(
        "--chunk-sizes",
        type=str,
        default=None,
        help="Comma-separated chunk sizes summing to block-size, e.g. 4,4,4,4",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional png path. ASCII is always printed.",
    )
    args = parser.parse_args()

    if args.block_size < 2:
        raise SystemExit("--block-size must be >= 2")
    if args.num_blocks < 1:
        raise SystemExit("--num-blocks must be >= 1")

    chunk_sizes = parse_chunk_sizes(args.chunk_sizes, args.block_size)
    resolved = resolve_flashmtp_chunk_sizes(args.block_size, chunk_sizes)
    print(f"chunk_sizes: {list(resolved)}")

    q_labels, kv_labels = build_labels(args.block_size, args.num_blocks)
    mask = build_mask(args.block_size, args.num_blocks, chunk_sizes)
    print_ascii(mask, q_labels, kv_labels)

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        save_png(mask, q_labels, kv_labels, args.output)
        print(f"\nSaved: {args.output}")


if __name__ == "__main__":
    main()
