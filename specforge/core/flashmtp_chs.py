# coding=utf-8
"""CHS gather indices and role-based RoPE position ids for FlashMTP (sink + single pivot)."""

import torch


def build_chs_sequence_indices(
    anchor_positions: torch.Tensor,
    seq_len: int,
    sink_num: int,
) -> torch.Tensor:
    """Sequence indices to gather target hidden: [0..sink_num-1] + [anchor-1] per block.

    Args:
        anchor_positions: (B, N) anchor index per draft block.
        seq_len: target sequence length.
        sink_num: number of attention-sink tokens from the sequence start.

    Returns:
        (B, N * (sink_num + 1)) long tensor of indices in [0, seq_len-1].
    """
    bsz, n_blocks = anchor_positions.shape
    device = anchor_positions.device
    smax = max(seq_len - 1, 0)
    sink_idx = torch.arange(sink_num, device=device, dtype=torch.long).clamp(max=smax)
    sink_idx = sink_idx.view(1, 1, -1).expand(bsz, n_blocks, -1)
    pivot_idx = (anchor_positions - 1).clamp(min=0, max=smax).unsqueeze(-1)
    return torch.cat([sink_idx, pivot_idx], dim=-1).view(bsz, n_blocks * (sink_num + 1))


def build_chs_rope_position_ids(
    bsz: int,
    n_blocks: int,
    sink_num: int,
    device: torch.device,
    dtype: torch.dtype = torch.long,
) -> torch.Tensor:
    """RoPE ids for CHS prefix: per block [0..sink_num-1, sink_num] (sink roles + pivot)."""
    block_ids = torch.arange(sink_num + 1, device=device, dtype=dtype)
    return block_ids.view(1, 1, -1).expand(bsz, n_blocks, -1).reshape(bsz, n_blocks * (sink_num + 1))


def build_draft_rope_position_ids(
    bsz: int,
    n_blocks: int,
    block_size: int,
    sink_num: int,
    device: torch.device,
    dtype: torch.dtype = torch.long,
) -> torch.Tensor:
    """RoPE ids for draft block tokens: per block sink_num+1 .. sink_num+block_size."""
    offs = torch.arange(block_size, device=device, dtype=dtype) + (sink_num + 1)
    return offs.view(1, 1, -1).expand(bsz, n_blocks, -1).reshape(bsz, n_blocks * block_size)
