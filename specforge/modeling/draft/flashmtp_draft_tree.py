"""Draft candidate tree construction and causal tree-verify utilities.

Used by ``FlashMTPDraftModel.spec_generate_with_draft_tree``.

Causal semantics (one target forward over all tree nodes):

- **Edges**: every node at depth ``d>0`` has ``parent_index`` = flat index of spine
  top1 at depth ``d-1``; logical child at ``d+1`` is spine top1 at ``d+1``.
- **RoPE**: ``position_ids[i] = start + depth[i]`` (siblings at the same depth share
  the same position).
- **Attention**: ancestor mask — row ``i`` sees prefix KV + nodes on the path
  anchor → … → spine[d-1] → node ``i``; no cross-sibling attention.
- **Accept**: ``logits[accepted_node_at_{d-1}]`` predicts the token at depth ``d``;
  accept any same-depth candidate (top1 or branch) that matches.
"""

from __future__ import annotations

import math
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Optional

import torch
from torch import nn
from transformers import DynamicCache


def _sample(logits: torch.Tensor, temperature: float = 0.0) -> torch.Tensor:
    if temperature < 1e-5:
        return torch.argmax(logits, dim=-1)
    bsz, seq_len, vocab_size = logits.shape
    logits = logits.view(-1, vocab_size) / temperature
    probs = torch.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1).view(bsz, seq_len)


@dataclass
class DraftTreeSpec:
    """Flattened draft token tree for one speculative step (``block_size`` nodes max)."""

    token_ids: list[int]
    depth: list[int]
    slot_in_depth: list[int]
    parent_index: list[int]
    spine_top1_index: list[int]
    is_chain: bool
    trunc_depth: Optional[int] = None
    expand_start_depth: Optional[int] = None
    width_at_depth: Optional[list[int]] = None


# ---------------------------------------------------------------------------
# Candidate tree construction
# ---------------------------------------------------------------------------


def draft_slot_top1_and_entropy(
    logits: torch.Tensor, tree_width: int
) -> tuple[int, float, list[int], float]:
    """Per-slot top1 id/prob, top-``tree_width`` ids, normalized entropy ratio."""
    probs = torch.softmax(logits.float(), dim=-1)
    top1_prob, top1_id = probs.max(dim=-1)
    w = max(int(tree_width), 1)
    k = min(w, probs.shape[-1])
    top_probs, top_ids = torch.topk(probs, k=k, dim=-1)
    top_ids_list = top_ids.tolist()
    if w <= 1:
        return int(top1_id.item()), float(top1_prob.item()), top_ids_list, 0.0
    p_norm = top_probs / top_probs.sum().clamp_min(1e-12)
    entropy = -(p_norm * p_norm.log().clamp_min(-1e-12)).sum()
    h_max = math.log(w)
    ratio = float(entropy.item() / h_max) if h_max > 1e-9 else 0.0
    return int(top1_id.item()), float(top1_prob.item()), top_ids_list, ratio


def expand_gate(
    top1_prob: float, entropy_ratio: float, expand_thres: float, x: float
) -> bool:
    return top1_prob < expand_thres and entropy_ratio > x


def find_trunc_depth(
    top1_probs: list[float], trunc_thres: float, max_depth: int
) -> Optional[int]:
    """Return cut depth ``d`` (keep anchor..d); ``None`` if no truncation."""
    for d in range(1, max_depth):
        if top1_probs[d] < trunc_thres and top1_probs[d + 1] < trunc_thres:
            return d
    return None


def build_draft_tree_from_logits(
    draft_logits: torch.Tensor,
    anchor_id: int,
    block_size: int,
    tree_width: int,
    trunc_thres: float,
    expand_thres: float,
    entropy_ratio: float,
) -> DraftTreeSpec:
    """Build draft tree from draft logits ``(block_size - 1, vocab)``.

    Phases:
      A. Truncate spine when two consecutive top1 probs < ``trunc_thres``.
      B. Find first expand depth ``d0`` with ``expand_gate``; chain-expand width ``w``.
      C. Flatten nodes (BFS by depth): each node's ``parent_index`` = spine top1 at d-1.
    """
    b = int(block_size)
    w = max(int(tree_width), 1)
    max_draft_depth = b - 1
    slot_stats: list[tuple[int, float, list[int], float]] = []
    for s in range(max_draft_depth):
        slot_stats.append(draft_slot_top1_and_entropy(draft_logits[s], w))

    top1_ids = [anchor_id] + [s[0] for s in slot_stats]
    top1_probs = [1.0] + [s[1] for s in slot_stats]

    trunc_depth = find_trunc_depth(top1_probs, trunc_thres, max_draft_depth)
    if trunc_depth is None:
        spine_depth = max_draft_depth
    else:
        spine_depth = trunc_depth

    if trunc_depth is None and spine_depth == max_draft_depth:
        token_ids = top1_ids[:b]
        depth = list(range(b))
        parent_index = [-1] + list(range(0, b - 1))
        slot_in_depth = [0] * b
        spine_top1_index = list(range(b))
        return DraftTreeSpec(
            token_ids=token_ids,
            depth=depth,
            slot_in_depth=slot_in_depth,
            parent_index=parent_index,
            spine_top1_index=spine_top1_index,
            is_chain=True,
            trunc_depth=None,
            expand_start_depth=None,
            width_at_depth=[1] * b,
        )

    width_at_depth = [1] * (spine_depth + 1)
    expand_start: Optional[int] = None
    total_nodes = spine_depth + 1
    d0: Optional[int] = None
    for d in range(1, spine_depth + 1):
        _, prob, _, ent_ratio = slot_stats[d - 1]
        if expand_gate(prob, ent_ratio, expand_thres, entropy_ratio):
            d0 = d
            break
    if d0 is not None:
        expand_start = d0
        for d in range(d0, spine_depth + 1):
            if total_nodes + (w - 1) > b:
                break
            width_at_depth[d] = w
            total_nodes += w - 1

    token_ids: list[int] = []
    depth: list[int] = []
    slot_in_depth: list[int] = []
    parent_index: list[int] = []
    spine_top1_index: list[int] = [-1] * (spine_depth + 1)

    token_ids.append(anchor_id)
    depth.append(0)
    slot_in_depth.append(0)
    parent_index.append(-1)
    spine_top1_index[0] = 0

    for d in range(1, spine_depth + 1):
        top1_tid, _, topw_ids, _ = slot_stats[d - 1]
        parent_idx = spine_top1_index[d - 1]
        if width_at_depth[d] == 1:
            idx = len(token_ids)
            token_ids.append(top1_tid)
            depth.append(d)
            slot_in_depth.append(0)
            parent_index.append(parent_idx)
            spine_top1_index[d] = idx
        else:
            ordered: list[int] = []
            seen: set[int] = set()
            for tid in [top1_tid] + topw_ids:
                if tid not in seen:
                    ordered.append(tid)
                    seen.add(tid)
                if len(ordered) >= w:
                    break
            spine_top1_index[d] = len(token_ids)
            for slot_i, tid in enumerate(ordered):
                token_ids.append(tid)
                depth.append(d)
                slot_in_depth.append(slot_i)
                parent_index.append(parent_idx)

    is_chain = all(width_at_depth[d] == 1 for d in range(1, spine_depth + 1))
    return DraftTreeSpec(
        token_ids=token_ids,
        depth=depth,
        slot_in_depth=slot_in_depth,
        parent_index=parent_index,
        spine_top1_index=spine_top1_index,
        is_chain=is_chain,
        trunc_depth=trunc_depth,
        expand_start_depth=expand_start,
        width_at_depth=width_at_depth,
    )


def format_tree_spec(spec: DraftTreeSpec) -> str:
    """Human-readable dump for inspection."""
    lines = [
        f"nodes={len(spec.token_ids)} is_chain={spec.is_chain} "
        f"trunc_depth={spec.trunc_depth} expand_start={spec.expand_start_depth}",
    ]
    if spec.width_at_depth is not None:
        lines.append(f"width_at_depth={spec.width_at_depth}")
    max_d = max(spec.depth) if spec.depth else 0
    for d in range(max_d + 1):
        idxs = [i for i, dd in enumerate(spec.depth) if dd == d]
        parts = []
        for i in sorted(idxs, key=lambda j: spec.slot_in_depth[j]):
            star = " *" if i == spec.spine_top1_index[d] else ""
            parts.append(
                f"[{i}] slot{spec.slot_in_depth[i]} id={spec.token_ids[i]} "
                f"parent={spec.parent_index[i]}{star}"
            )
        lines.append(f"  depth {d}: " + " | ".join(parts))
    spine = spine_token_sequence(spec)
    lines.append(f"  spine tokens: {spine}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Tree verify: mask, position_ids, one-shot target forward
# ---------------------------------------------------------------------------


def build_tree_ancestor_attention_mask(
    parent_index: list[int],
    device: torch.device,
    dtype: torch.dtype,
    *,
    kv_length: Optional[int] = None,
    kv_offset: int = 0,
) -> torch.Tensor:
    """Additive mask ``(1, 1, N, kv_length)``: 0 = attend, ``finfo.min`` = mask out.

    Visibility rule:
      - Query row ``i`` may attend KV column ``kv_offset + j`` iff ``j`` is on the
        ancestor chain of node ``i`` (walk ``parent_index`` until -1).
      - Columns ``[0, kv_offset)`` are all open (committed prefix KV in cache).

    ``position_ids`` for the same forward should be ``start + depth[i]`` (siblings at
    a depth share the same absolute position). Use with ``attn_implementation='eager'``
    and ``attention_mask={'full_attention': mask}``; FlashAttention2 does not support
    this mask shape.
    """
    n = len(parent_index)
    kv_len = int(kv_length) if kv_length is not None else n
    mask = torch.full((1, 1, n, kv_len), torch.finfo(dtype).min, device=device, dtype=dtype)
    if kv_offset > 0:
        mask[:, :, :, :kv_offset] = 0
    tree_start = kv_offset
    for i in range(n):
        j = i
        while j >= 0:
            mask[0, 0, i, tree_start + j] = 0
            j = parent_index[j]
    return mask


def build_tree_verify_position_ids(
    spec: DraftTreeSpec, start: int, device: torch.device
) -> torch.LongTensor:
    """``position_ids[i] = start + depth[i]`` (siblings share depth / position)."""
    return torch.tensor(
        [start + d for d in spec.depth], device=device, dtype=torch.long
    ).unsqueeze(0)


def attention_mask_bool(spec: DraftTreeSpec, past_len: int) -> torch.Tensor:
    """Boolean visibility ``(N, past_len + N)`` for debugging (True = can attend)."""
    n = len(spec.token_ids)
    kv_len = past_len + n
    mask = torch.zeros(n, kv_len, dtype=torch.bool)
    if past_len > 0:
        mask[:, :past_len] = True
    float_mask = build_tree_ancestor_attention_mask(
        spec.parent_index,
        torch.device("cpu"),
        torch.float32,
        kv_length=kv_len,
        kv_offset=past_len,
    )
    mask[:, past_len:] = float_mask[0, 0].eq(0)
    return mask


def format_attention_mask(spec: DraftTreeSpec, past_len: int = 0) -> str:
    """ASCII grid: rows/cols = tree node indices (after prefix columns)."""
    n = len(spec.token_ids)
    bool_m = attention_mask_bool(spec, past_len)
    tree_part = bool_m[:, past_len:]
    header = "     " + "".join(f"{j:3d}" for j in range(n))
    lines = [header, f"prefix cols [0,{past_len}) all True for every row"]
    for i in range(n):
        row = "".join("." if tree_part[i, j] else "x" for j in range(n))
        lines.append(f"{i:3d}  {row}  depth={spec.depth[i]} id={spec.token_ids[i]}")
    return "\n".join(lines)


@contextmanager
def eager_attention_context(target: nn.Module):
    """FlashAttention2 cannot take custom tree masks; use eager for tree verify."""
    configs: list = [target.config]
    if hasattr(target, "model") and getattr(target.model, "config", None) is not None:
        configs.append(target.model.config)
    old = [getattr(c, "_attn_implementation", None) for c in configs]
    try:
        for c in configs:
            c._attn_implementation = "eager"
        yield
    finally:
        for c, prev in zip(configs, old):
            if prev is not None:
                c._attn_implementation = prev


def target_forward_tree_verify(
    target: nn.Module,
    spec: DraftTreeSpec,
    start: int,
    past_key_values: DynamicCache,
    device: torch.device,
    dtype: torch.dtype,
):
    """One target forward over all tree nodes with ancestor mask + shared depth positions."""
    n = len(spec.token_ids)
    past_len = int(past_key_values.get_seq_length())
    tree_ids = torch.tensor(spec.token_ids, device=device, dtype=torch.long).unsqueeze(0)
    tree_pos = build_tree_verify_position_ids(spec, start, device)
    mask = build_tree_ancestor_attention_mask(
        spec.parent_index,
        device,
        dtype,
        kv_length=past_len + n,
        kv_offset=past_len,
    )
    with eager_attention_context(target):
        return target(
            tree_ids,
            position_ids=tree_pos,
            attention_mask={"full_attention": mask},
            past_key_values=past_key_values,
            use_cache=True,
            output_hidden_states=True,
        )


# ---------------------------------------------------------------------------
# Acceptance helpers
# ---------------------------------------------------------------------------


def spine_token_sequence(spec: DraftTreeSpec) -> list[int]:
    """Top1 token at each depth (index 0 = anchor)."""
    return [spec.token_ids[spec.spine_top1_index[d]] for d in range(len(spec.spine_top1_index))]


def nodes_by_depth_sorted(spec: DraftTreeSpec) -> list[list[int]]:
    """Flat node indices grouped by depth (spine top1 slot 0 first within each depth)."""
    max_depth = max(spec.depth)
    grouped: list[list[int]] = [[] for _ in range(max_depth + 1)]
    for i in range(len(spec.token_ids)):
        grouped[spec.depth[i]].append(i)
    for d in range(max_depth + 1):
        grouped[d].sort(key=lambda i: spec.slot_in_depth[i])
    return grouped


def _logits_predict_token(logits: torch.Tensor, node_idx: int, temperature: float) -> int:
    row = logits[node_idx]
    if temperature < 1e-5:
        return int(row.argmax().item())
    return int(_sample(row.view(1, 1, -1), temperature).item())


def greedy_accept_tree_causal(
    spec: DraftTreeSpec,
    logits: torch.Tensor,
    temperature: float = 0.0,
) -> tuple[list[int], Optional[int], int, int]:
    """Causal greedy accept after ``target_forward_tree_verify``.

    At depth ``d``, use ``logits[accepted_idx_{d-1}]`` (the accepted node on the path
    through spine-top1 parents) and match against any candidate at depth ``d``.

    Returns ``(accepted_tokens_incl_anchor, correction_or_none, accepted_draft_count,
    last_accepted_flat_index)``.
    """
    by_depth = nodes_by_depth_sorted(spec)
    max_depth = max(spec.depth)
    accepted_tokens = [spec.token_ids[spec.spine_top1_index[0]]]
    accepted_node = spec.spine_top1_index[0]
    accepted_draft = 0
    correction: Optional[int] = None

    for d in range(1, max_depth + 1):
        pred = _logits_predict_token(logits, accepted_node, temperature)
        matched_idx: Optional[int] = None
        for node_idx in by_depth[d]:
            if spec.token_ids[node_idx] == pred:
                matched_idx = node_idx
                break
        if matched_idx is None:
            correction = pred
            break
        accepted_tokens.append(spec.token_ids[matched_idx])
        accepted_node = matched_idx
        accepted_draft = d

    return accepted_tokens, correction, accepted_draft, accepted_node


def greedy_accept_spine_with_branches(
    spec: DraftTreeSpec,
    posterior: torch.Tensor,
    temperature: float = 0.0,
) -> tuple[list[int], Optional[int], int]:
    """Legacy: spine-only forward + per-depth match (no tree mask)."""
    spine_ids = spine_token_sequence(spec)
    by_depth = nodes_by_depth_sorted(spec)
    accepted = [spine_ids[0]]
    accepted_draft = 0
    correction: Optional[int] = None

    for d in range(1, len(spine_ids)):
        if temperature < 1e-5:
            pred = int(posterior[0, d - 1].argmax().item())
        else:
            pred = int(_sample(posterior[:, d - 1 : d], temperature).item())
        matched_tid: Optional[int] = None
        for node_idx in by_depth[d]:
            if spec.token_ids[node_idx] == pred:
                matched_tid = spec.token_ids[node_idx]
                break
        if matched_tid is None:
            correction = pred
            break
        accepted.append(matched_tid)
        accepted_draft = d

    return accepted, correction, accepted_draft


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Inspect draft tree + verify mask")
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--tree-width", type=int, default=4)
    parser.add_argument("--trunc-thres", type=float, default=0.2)
    parser.add_argument("--expand-thres", type=float, default=0.5)
    parser.add_argument("--entropy-ratio", type=float, default=0.4)
    parser.add_argument("--past-len", type=int, default=100)
    args = parser.parse_args()

    torch.manual_seed(0)
    logits = torch.randn(args.block_size - 1, 128)
    spec = build_draft_tree_from_logits(
        logits,
        anchor_id=7,
        block_size=args.block_size,
        tree_width=args.tree_width,
        trunc_thres=args.trunc_thres,
        expand_thres=args.expand_thres,
        entropy_ratio=args.entropy_ratio,
    )
    print(format_tree_spec(spec))
    print()
    print(format_attention_mask(spec, past_len=args.past_len))
