#!/usr/bin/env python3
"""Visualize FlashMTP clean/mask stream training attention masks."""

import argparse
from pathlib import Path
from typing import Optional

from specforge.modeling.draft.flashmtp_chunk_utils import normalize_decode_chunk_sizes


def slot_group(slot: int) -> int:
    """Mirror FlashMTP groups: anchor, [1], [2,3], then chunks of 4."""
    if slot <= 1:
        return slot
    if slot < 4:
        return 2
    return 3 + (slot - 4) // 4


def build_labels(block_size: int, num_blocks: int):
    """Row/column names for Q (draft) and KV (CHS + draft).

    Mask stream slot 0 uses the **same token embedding as clean anchor** (training
    `OnlineFlashMTPModel._create_noise_embed`); label ``M{b}:0=C`` reads as
    ``M`` stream slot ``0`` equals ``C`` (clean) anchor.
    """
    q_labels = []
    kv_labels = [f"H{b}" for b in range(num_blocks)]
    for block in range(num_blocks):
        q_labels.extend(f"C{block}:{slot}" for slot in range(block_size))
        q_labels.extend(
            f"M{block}:0=C" if slot == 0 else f"M{block}:{slot}"
            for slot in range(block_size)
        )
        kv_labels.extend(f"C{block}:{slot}" for slot in range(block_size))
        kv_labels.extend(
            f"M{block}:0=C" if slot == 0 else f"M{block}:{slot}"
            for slot in range(block_size)
        )
    return q_labels, kv_labels


def _decode_chunk_tables(
    decode_chunk_sizes: list[int], block_size: int
) -> tuple[list[int], list[int], list[int]]:
    acc = 0
    _chunk_lo: list[int] = []
    _chunk_hi: list[int] = []
    _chunk_of_slot = [0] * block_size
    for sz in decode_chunk_sizes:
        lo, hi = acc, acc + int(sz)
        _chunk_lo.append(lo)
        _chunk_hi.append(hi)
        for s in range(lo, hi):
            _chunk_of_slot[s] = len(_chunk_lo) - 1
        acc = hi
    return _chunk_of_slot, _chunk_lo, _chunk_hi


def is_visible(
    q_idx: int,
    kv_idx: int,
    block_size: int,
    num_blocks: int,
    decode_chunk_tables: Optional[tuple[list[int], list[int], list[int]]] = None,
) -> bool:
    """Return whether training attention allows q_idx -> kv_idx.

    Must match ``specforge.core.flashmtp.create_flashmtp_block_mask`` for
    ``chs_len_per_block == 1`` and all blocks valid (``block_keep_mask=True``).
    """
    stream_block_size = 2 * block_size
    total_chs_len = num_blocks

    q_block = q_idx // stream_block_size
    q_stream_offset = q_idx % stream_block_size
    q_is_mask = q_stream_offset >= block_size
    q_slot = q_stream_offset % block_size

    if kv_idx < total_chs_len:
        return kv_idx == q_block

    draft_kv_idx = kv_idx - total_chs_len
    kv_block = draft_kv_idx // stream_block_size
    kv_stream_offset = draft_kv_idx % stream_block_size
    kv_is_mask = kv_stream_offset >= block_size
    kv_slot = kv_stream_offset % block_size

    if kv_block != q_block:
        return False

    if decode_chunk_tables is None:
        q_group = slot_group(q_slot)
        kv_group = slot_group(kv_slot)
        if q_is_mask:
            return ((not kv_is_mask) and kv_group < q_group) or (
                kv_is_mask and kv_group == q_group
            )
        return (not kv_is_mask) and kv_group <= q_group

    _chunk_of_slot, _chunk_lo, _chunk_hi = decode_chunk_tables

    qs, kvs = int(q_slot), int(kv_slot)
    cq = _chunk_of_slot[qs]
    ckv = _chunk_of_slot[kvs]
    lo_q, hi_q = _chunk_lo[cq], _chunk_hi[cq]

    def _in_range(s: int, lo: int, hi: int) -> bool:
        return lo <= s < hi

    if ckv > cq:
        return False
    if ckv < cq:
        return not kv_is_mask
    if cq == 0:
        if q_is_mask:
            return (
                (kv_is_mask and _in_range(kvs, lo_q, hi_q) and (kvs > 0))
                or ((not kv_is_mask) and kvs == 0)
            )
        return (not kv_is_mask) and _in_range(kvs, lo_q, hi_q)
    if q_is_mask:
        return kv_is_mask and _in_range(kvs, lo_q, hi_q) and (kvs > 0)
    return (not kv_is_mask) and _in_range(kvs, lo_q, hi_q)


def build_mask(
    block_size: int,
    num_blocks: int,
    decode_chunk_sizes: Optional[list[int]] = None,
):
    q_len = num_blocks * 2 * block_size
    kv_len = num_blocks + num_blocks * 2 * block_size
    tables = None
    if decode_chunk_sizes is not None:
        tables = _decode_chunk_tables(decode_chunk_sizes, block_size)
    return [
        [is_visible(q, kv, block_size, num_blocks, tables) for kv in range(kv_len)]
        for q in range(q_len)
    ]


def print_ascii(mask, q_labels, kv_labels, max_label_width: int = 8):
    label_width = max(max(len(label) for label in q_labels), max_label_width)
    header = " " * (label_width + 1)
    col_w = 7
    for lab in kv_labels:
        piece = lab if len(lab) <= col_w else (lab[: col_w - 1] + "…")
        header += piece.rjust(col_w)
    print(header)
    print(" " * (label_width + 1) + "".join("-" * col_w for _ in kv_labels))

    for q_label, row in zip(q_labels, mask):
        cells = []
        for kv_label, visible in zip(kv_labels, row):
            if not visible:
                cells.append("." * col_w)
            elif kv_label.startswith("H"):
                cells.append("H".center(col_w))
            elif kv_label.startswith("C"):
                cells.append("c".center(col_w))
            else:
                cells.append("m".center(col_w))
        print(q_label.rjust(label_width) + " " + "".join(cells))

    print(
        "\nLegend: H=CHS pivot, c=clean-stream KV visible, m=mask-stream KV visible, "
        "·=hidden. Rows=Q, cols=KV. Label M{b}:0=C = mask stream slot 0 uses same "
        "token as C{b}:0 (anchor)."
    )
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


def main():
    parser = argparse.ArgumentParser(
        description="Visualize FlashMTP clean/mask stream training attention mask."
    )
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--num-blocks", type=int, default=1)
    parser.add_argument(
        "--output",
        type=Path,
        default='./mask.png',
        help="Optional png path. ASCII is always printed.",
    )
    parser.add_argument(
        "--decode-chunk-sizes",
        type=str,
        default='4,4,4,4',
        help="e.g. 4,4,4,4 with --block-size 16 (same as training decode_chunk_sizes).",
    )
    args = parser.parse_args()

    if args.block_size < 2:
        raise SystemExit("--block-size must be >= 2")
    if args.num_blocks < 1:
        raise SystemExit("--num-blocks must be >= 1")

    decode_chunks = None
    if args.decode_chunk_sizes:
        decode_chunks = normalize_decode_chunk_sizes(
            args.decode_chunk_sizes, args.block_size
        )

    q_labels, kv_labels = build_labels(args.block_size, args.num_blocks)
    mask = build_mask(args.block_size, args.num_blocks, decode_chunks)
    print_ascii(mask, q_labels, kv_labels)

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        save_png(mask, q_labels, kv_labels, args.output)
        print(f"\nSaved: {args.output}")


if __name__ == "__main__":
    main()
