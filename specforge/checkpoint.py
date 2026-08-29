"""Helpers for saving and loading distributed training state.

The custom BF16 optimizer owns local FP32 copies of FSDP parameter shards.  Its
state therefore cannot be represented by rank 0's optimizer state alone: every
rank must persist and restore its own shard.
"""

import os
from typing import Any, Optional

import torch
import torch.distributed as dist

_DISTRIBUTED_STATE_KEY = "distributed_training_state"
_DISTRIBUTED_STATE_FORMAT = "rank_local_optimizer_v1"


def ranked_training_state_path(checkpoint_dir: str, rank: int) -> str:
    """Return the training-state filename owned by one global rank."""
    return os.path.join(checkpoint_dir, f"training_state_rank_{rank:05d}.pt")


def distributed_training_state_exists(checkpoint_dir: str) -> bool:
    """Return whether the checkpoint's committed/common state exists."""
    return os.path.isfile(os.path.join(checkpoint_dir, "training_state.pt"))


def _rank_and_world_size(
    rank: Optional[int] = None, world_size: Optional[int] = None
) -> tuple[int, int]:
    if rank is None:
        rank = dist.get_rank() if dist.is_initialized() else 0
    if world_size is None:
        world_size = dist.get_world_size() if dist.is_initialized() else 1
    return rank, world_size


def _atomic_torch_save(state: dict[str, Any], path: str) -> None:
    tmp_path = f"{path}.tmp.{os.getpid()}"
    try:
        torch.save(state, tmp_path)
        os.replace(tmp_path, path)
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def save_distributed_training_state(
    checkpoint_dir: str,
    state: dict[str, Any],
    *,
    rank: Optional[int] = None,
    world_size: Optional[int] = None,
    process_group: Optional[dist.ProcessGroup] = None,
) -> None:
    """Save rank-local optimizer state and a rank-0 compatibility file.

    In distributed runs, all ranks must call this function.  Each rank first
    writes its local optimizer shard.  Only after every shard is durable does
    rank 0 publish ``training_state.pt``.
    """
    if process_group is not None:
        rank = dist.get_rank(process_group) if rank is None else rank
        world_size = (
            dist.get_world_size(process_group) if world_size is None else world_size
        )
    rank, world_size = _rank_and_world_size(rank, world_size)
    os.makedirs(checkpoint_dir, exist_ok=True)

    state_to_save = dict(state)
    state_to_save[_DISTRIBUTED_STATE_KEY] = {
        "format": _DISTRIBUTED_STATE_FORMAT,
        "rank": rank,
        "world_size": world_size,
    }

    if world_size > 1:
        _atomic_torch_save(
            state_to_save, ranked_training_state_path(checkpoint_dir, rank)
        )

    if dist.is_initialized():
        dist.barrier(group=process_group)

    if rank == 0:
        _atomic_torch_save(
            state_to_save, os.path.join(checkpoint_dir, "training_state.pt")
        )

    if dist.is_initialized():
        dist.barrier(group=process_group)


def load_distributed_training_state(
    checkpoint_dir: str,
    *,
    map_location: Any = "cpu",
    rank: Optional[int] = None,
    world_size: Optional[int] = None,
    process_group: Optional[dist.ProcessGroup] = None,
) -> Optional[dict[str, Any]]:
    """Load this rank's optimizer shard, with legacy checkpoint fallback."""
    if process_group is not None:
        rank = dist.get_rank(process_group) if rank is None else rank
        world_size = (
            dist.get_world_size(process_group) if world_size is None else world_size
        )
    rank, world_size = _rank_and_world_size(rank, world_size)
    ranked_path = ranked_training_state_path(checkpoint_dir, rank)
    common_path = os.path.join(checkpoint_dir, "training_state.pt")
    if not os.path.isfile(common_path):
        return None
    state_path = ranked_path if os.path.isfile(ranked_path) else common_path

    state = torch.load(state_path, map_location=map_location, weights_only=False)
    metadata = state.get(_DISTRIBUTED_STATE_KEY)
    if metadata is None:
        # Checkpoints created before rank-local optimizer saving are readable for
        # backward compatibility. Their optimizer state may only be complete on
        # rank 0, so BF16Optimizer's compatibility checks still apply.
        return state

    if metadata.get("format") != _DISTRIBUTED_STATE_FORMAT:
        raise RuntimeError(
            f"Unsupported distributed training-state format in {state_path}: "
            f"{metadata.get('format')!r}."
        )
    saved_world_size = metadata.get("world_size")
    saved_rank = metadata.get("rank")
    if saved_world_size != world_size:
        raise RuntimeError(
            "Cannot restore rank-local optimizer state with a different world "
            f"size: checkpoint={saved_world_size}, current={world_size}."
        )
    if saved_rank != rank:
        expected_path = ranked_training_state_path(checkpoint_dir, rank)
        raise RuntimeError(
            f"Optimizer state for global rank {rank} is missing. Expected "
            f"{expected_path}, but found state for rank {saved_rank}."
        )
    return state
