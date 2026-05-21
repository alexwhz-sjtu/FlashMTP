# coding=utf-8
"""Custom decode chunk layout for FlashMTP (training mask + batched inference).

The first chunk length **includes** slot 0 (anchor): only slots ``1 .. c0-1`` are
supervised in that chunk. Remaining chunks cover consecutive speculative slots.
``sum(sizes)`` must equal ``block_size``.
"""

from __future__ import annotations

from typing import Any, Optional

import torch


def parse_decode_chunk_sizes_str(s: str | None) -> Optional[list[int]]:
    """Parse ``"4,4,4,4"`` into positive ints; empty / None -> None."""
    if s is None or (isinstance(s, str) and not s.strip()):
        return None
    parts = [p.strip() for p in s.split(",") if p.strip()]
    if not parts:
        return None
    out: list[int] = []
    for p in parts:
        v = int(p)
        if v < 1:
            raise ValueError(f"decode chunk size must be >= 1, got {v!r} in {s!r}")
        out.append(v)
    return out


def normalize_decode_chunk_sizes(raw: Any, block_size: int) -> Optional[list[int]]:
    """Normalize config value to ``list[int]`` or None. Validates sum == block_size."""
    if raw is None:
        return None
    if isinstance(raw, str):
        sizes = parse_decode_chunk_sizes_str(raw)
        if sizes is None:
            return None
    elif isinstance(raw, (list, tuple)):
        sizes = [int(x) for x in raw]
        for z in sizes:
            if z < 1:
                raise ValueError(f"decode_chunk_sizes entries must be >= 1, got {sizes!r}")
    else:
        raise TypeError(f"decode_chunk_sizes must be list/tuple/str/None, got {type(raw)}")

    total = sum(sizes)
    if total != block_size:
        raise ValueError(
            f"decode_chunk_sizes must sum to block_size={block_size}, got {sizes!r} (sum={total})"
        )
    return sizes


def build_decode_chunk_prediction_groups(
    decode_chunk_sizes: list[int], block_size: int
) -> list[tuple[int, int]]:
    """Return half-open ``[lo, hi)`` slot ranges for successive draft forward passes.

    Slot 0 is anchor (no supervision). Chunk 0 spans ``[0, c0)``; if ``c0 > 1``,
    the first pass predicts slots ``[0, c0)``. Later passes predict ``[b_k, b_{k+1})``.
    """
    sizes = normalize_decode_chunk_sizes(decode_chunk_sizes, block_size)
    assert sizes is not None
    boundaries = [0]
    for s in sizes:
        boundaries.append(boundaries[-1] + int(s))
    groups: list[tuple[int, int]] = []
    if sizes[0] > 1:
        groups.append((0, boundaries[1]))
    for k in range(1, len(sizes)):
        lo, hi = boundaries[k], boundaries[k + 1]
        groups.append((lo, hi))
    return groups


def slot_to_chunk_group_tensor(
    decode_chunk_sizes: list[int], block_size: int, device: torch.device
) -> torch.Tensor:
    """``slot_chunk[s]`` = decode chunk index for slot ``s`` in ``0 .. block_size-1``."""
    sizes = normalize_decode_chunk_sizes(decode_chunk_sizes, block_size)
    assert sizes is not None
    t = torch.empty(block_size, dtype=torch.long, device=device)
    acc = 0
    for i, sz in enumerate(sizes):
        t[acc : acc + sz] = i
        acc += sz
    return t
