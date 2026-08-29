from datetime import timedelta
from dataclasses import dataclass
from typing import Any, Optional

import torch
import torch.distributed as dist
from yunchang.globals import PROCESS_GROUP, set_seq_parallel_pg

from specforge.utils import print_with_rank

_DEVICE_MESH = None
_TP_DEVICE_MESH = None
_TP_GROUP = None
_DP_DEVICE_MESH = None
_DP_GROUP = None
_DRAFT_DP_GROUP = None
_DRAFT_SP_GROUP = None
_SP_ULYSSES_GROUP = None
_SP_RING_GROUP = None
_DISAGG_TOPOLOGY = None
_BRIDGE_GROUP = None


@dataclass(frozen=True)
class DisaggregatedTopology:
    """Role and process-group metadata for node-local target/draft pipelines."""

    rank: int
    local_rank: int
    node_rank: int
    nnodes: int
    nproc_per_node: int
    target_ranks_per_node: int
    draft_ranks_per_node: int
    target_tp_size: int
    role: str
    target_replica_local_rank: Optional[int]
    target_tp_rank: Optional[int]
    target_tp_leader_global_rank: Optional[int]
    draft_local_rank: Optional[int]
    target_tp_group: Optional[dist.ProcessGroup]
    bridge_group: Optional[dist.ProcessGroup]
    draft_group: Optional[dist.ProcessGroup]

    @property
    def target_replicas_per_node(self) -> int:
        return self.target_ranks_per_node // self.target_tp_size

    @property
    def is_target(self) -> bool:
        return self.role == "target"

    @property
    def is_draft(self) -> bool:
        return self.role == "draft"

    @property
    def is_target_leader(self) -> bool:
        return self.is_target and self.target_tp_rank == 0

    @property
    def draft_global_ranks(self) -> list[int]:
        return [
            node * self.nproc_per_node + self.target_ranks_per_node + local
            for node in range(self.nnodes)
            for local in range(self.draft_ranks_per_node)
        ]

    @property
    def node_target_leader_ranks(self) -> list[int]:
        base = self.node_rank * self.nproc_per_node
        return [
            base + replica * self.target_tp_size
            for replica in range(self.target_replicas_per_node)
        ]

    @property
    def node_draft_ranks(self) -> list[int]:
        base = self.node_rank * self.nproc_per_node + self.target_ranks_per_node
        return [base + local for local in range(self.draft_ranks_per_node)]


def get_tp_group():
    global _TP_GROUP
    return _TP_GROUP


def get_dp_group():
    global _DP_GROUP
    return _DP_GROUP


def get_draft_dp_group():
    global _DRAFT_DP_GROUP
    return _DRAFT_DP_GROUP


def get_draft_sp_group():
    global _DRAFT_SP_GROUP
    return _DRAFT_SP_GROUP


def get_device_mesh():
    global _DEVICE_MESH
    return _DEVICE_MESH


def get_tp_device_mesh():
    global _TP_DEVICE_MESH
    return _TP_DEVICE_MESH


def get_dp_device_mesh():
    global _DP_DEVICE_MESH
    return _DP_DEVICE_MESH


def get_sp_ulysses_group():
    global _SP_ULYSSES_GROUP
    return _SP_ULYSSES_GROUP


def get_sp_ring_group():
    global _SP_RING_GROUP
    return _SP_RING_GROUP


def get_disaggregated_topology() -> Optional[DisaggregatedTopology]:
    return _DISAGG_TOPOLOGY


def get_bridge_group():
    return _BRIDGE_GROUP


def init_disaggregated(
    *,
    timeout: int,
    target_ranks_per_node: int,
    draft_ranks_per_node: int,
    target_tp_size: int,
) -> DisaggregatedTopology:
    """Initialize node-local target pipelines and a global draft training group.

    Every rank calls every ``new_group`` in the same order. Target TP and bridge
    groups contain ranks from exactly one node; only the draft group spans nodes.
    """
    import os

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    dist.init_process_group(
        backend="nccl",
        timeout=timedelta(minutes=timeout),
        device_id=torch.device("cuda", local_rank),
    )
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    # torchrun always exports LOCAL_RANK; using it avoids modulo assumptions when
    # CUDA_VISIBLE_DEVICES differs between nodes.
    nproc_per_node = target_ranks_per_node + draft_ranks_per_node
    if world_size % nproc_per_node != 0:
        raise ValueError(
            f"WORLD_SIZE={world_size} must be divisible by per-node ranks "
            f"{nproc_per_node}."
        )
    if target_ranks_per_node <= 0 or draft_ranks_per_node <= 0:
        raise ValueError("target and draft ranks per node must both be positive.")
    if target_ranks_per_node % target_tp_size != 0:
        raise ValueError(
            f"target_ranks_per_node={target_ranks_per_node} must be divisible by "
            f"target_tp_size={target_tp_size}."
        )
    nnodes = world_size // nproc_per_node
    node_rank = rank // nproc_per_node
    if local_rank >= nproc_per_node:
        raise ValueError(
            f"LOCAL_RANK={local_rank} is outside configured per-node world "
            f"size {nproc_per_node}."
        )

    target_replicas = target_ranks_per_node // target_tp_size
    current_tp_group = None
    current_bridge_group = None
    current_draft_group = None

    # Node-local TP groups. No rank list can cross a node boundary.
    for node in range(nnodes):
        node_base = node * nproc_per_node
        for replica in range(target_replicas):
            ranks = list(
                range(
                    node_base + replica * target_tp_size,
                    node_base + (replica + 1) * target_tp_size,
                )
            )
            group = dist.new_group(ranks=ranks, backend="nccl")
            if rank in ranks:
                current_tp_group = group

    # One local bridge communicator per node: target TP leaders and local draft.
    for node in range(nnodes):
        node_base = node * nproc_per_node
        leaders = [
            node_base + replica * target_tp_size
            for replica in range(target_replicas)
        ]
        drafts = [
            node_base + target_ranks_per_node + local
            for local in range(draft_ranks_per_node)
        ]
        ranks = leaders + drafts
        group = dist.new_group(ranks=ranks, backend="nccl")
        if rank in ranks:
            current_bridge_group = group

    draft_ranks = [
        node * nproc_per_node + target_ranks_per_node + local
        for node in range(nnodes)
        for local in range(draft_ranks_per_node)
    ]
    draft_group = dist.new_group(ranks=draft_ranks, backend="nccl")
    if rank in draft_ranks:
        current_draft_group = draft_group

    role = "target" if local_rank < target_ranks_per_node else "draft"
    target_replica = local_rank // target_tp_size if role == "target" else None
    target_tp_rank = local_rank % target_tp_size if role == "target" else None
    leader_rank = (
        node_rank * nproc_per_node + target_replica * target_tp_size
        if target_replica is not None
        else None
    )
    draft_local_rank = (
        local_rank - target_ranks_per_node if role == "draft" else None
    )
    topology = DisaggregatedTopology(
        rank=rank,
        local_rank=local_rank,
        node_rank=node_rank,
        nnodes=nnodes,
        nproc_per_node=nproc_per_node,
        target_ranks_per_node=target_ranks_per_node,
        draft_ranks_per_node=draft_ranks_per_node,
        target_tp_size=target_tp_size,
        role=role,
        target_replica_local_rank=target_replica,
        target_tp_rank=target_tp_rank,
        target_tp_leader_global_rank=leader_rank,
        draft_local_rank=draft_local_rank,
        target_tp_group=current_tp_group,
        bridge_group=current_bridge_group,
        draft_group=current_draft_group,
    )

    global _TP_GROUP, _DP_GROUP, _DRAFT_DP_GROUP, _DRAFT_SP_GROUP
    global _SP_ULYSSES_GROUP, _SP_RING_GROUP, _DISAGG_TOPOLOGY, _BRIDGE_GROUP
    _TP_GROUP = current_tp_group
    _DP_GROUP = current_draft_group
    _DRAFT_DP_GROUP = current_draft_group
    _DRAFT_SP_GROUP = None
    _SP_ULYSSES_GROUP = None
    _SP_RING_GROUP = None
    _DISAGG_TOPOLOGY = topology
    _BRIDGE_GROUP = current_bridge_group
    print_with_rank(
        f"disaggregate role={role}, node={node_rank}, local_rank={local_rank}, "
        f"target_tp_rank={target_tp_rank}, draft_local_rank={draft_local_rank}"
    )
    # Initialize every new NCCL communicator before compute/P2P starts.
    dist.barrier()
    return topology


def init_distributed(
    timeout: int = 10, tp_size: int = 1, sp_ulysses_size: int = 1, sp_ring_size: int = 1
):
    """Initialize distributed training.

    Args:
        timeout(int): Timeout for collective communication in minutes
        tp_size(int): The degree of tensor parallelism
    """
    dist.init_process_group(backend="nccl", timeout=timedelta(minutes=timeout))
    local_rank = dist.get_rank() % torch.cuda.device_count()
    torch.cuda.set_device(local_rank)
    print_with_rank(f"bind to device {local_rank}")

    world_size = dist.get_world_size()
    dp_size = world_size // tp_size
    assert (
        world_size == tp_size * dp_size
    ), f"world size must be divisible by tp size, now {world_size=}, {(tp_size * dp_size)=} "

    device_mesh = dist.device_mesh.init_device_mesh(
        "cuda", (dp_size, tp_size), mesh_dim_names=("dp", "tp")
    )

    assert (
        world_size % (sp_ulysses_size * sp_ring_size) == 0
    ), f"World size ({world_size}) cannot be evenly divided by total SP size ({sp_ulysses_size*sp_ring_size})"

    draft_dp_size = world_size // (sp_ulysses_size * sp_ring_size)
    draft_device_mesh = dist.device_mesh.init_device_mesh(
        "cuda",
        (draft_dp_size, sp_ulysses_size * sp_ring_size),
        mesh_dim_names=("draft_dp", "sp"),
    )
    set_seq_parallel_pg(sp_ulysses_size, sp_ring_size, dist.get_rank(), world_size)

    print_with_rank(f"device mesh: {device_mesh}")
    tp_group = device_mesh.get_group("tp")
    dp_group = device_mesh.get_group("dp")

    sp_ulysses_group = PROCESS_GROUP.ULYSSES_PG
    sp_ring_group = PROCESS_GROUP.RING_PG
    # we need to create a 1D submesh
    tp_device_mesh = dist.DeviceMesh.from_group(tp_group, device_type="cuda")

    global _TP_GROUP, _DP_GROUP, _DEVICE_MESH, _TP_DEVICE_MESH, _DP_DEVICE_MESH, _SP_RING_GROUP, _SP_ULYSSES_GROUP, _DRAFT_DP_GROUP, _DRAFT_SP_GROUP
    _DEVICE_MESH = device_mesh
    _TP_GROUP = tp_group
    _TP_DEVICE_MESH = tp_device_mesh
    _SP_ULYSSES_GROUP = sp_ulysses_group
    _SP_RING_GROUP = sp_ring_group
    _DP_GROUP = dp_group
    _DRAFT_DP_GROUP = draft_device_mesh.get_group("draft_dp")
    _DRAFT_SP_GROUP = draft_device_mesh.get_group("sp")
    _DP_DEVICE_MESH = dist.DeviceMesh.from_group(dp_group, device_type="cuda")


def destroy_distributed():
    global _TP_GROUP, _DP_GROUP, _SP_ULYSSES_GROUP, _SP_RING_GROUP, _DRAFT_DP_GROUP
    seen = set()
    for group in (
        _TP_GROUP,
        _DP_GROUP,
        _SP_ULYSSES_GROUP,
        _SP_RING_GROUP,
        _DRAFT_DP_GROUP,
        _DRAFT_SP_GROUP,
        _BRIDGE_GROUP,
    ):
        if group is None or group in seen:
            continue
        seen.add(group)
        dist.destroy_process_group(group)
    dist.destroy_process_group()


def shard_tensor(
    tensor: torch.Tensor, process_group: dist.ProcessGroup = None, dim: int = -1
) -> torch.Tensor:
    rank = dist.get_rank(process_group)
    size = dist.get_world_size(process_group)
    return tensor.chunk(size, dim=dim)[rank].contiguous()


def gather_tensor(
    tensor: torch.Tensor, process_group: dist.ProcessGroup = None, dim: int = -1
) -> torch.Tensor:
    size = dist.get_world_size(process_group)
    obj_list = [torch.empty_like(tensor) for _ in range(size)]
    dist.all_gather(obj_list, tensor, group=process_group)
    gather_tensor = torch.cat(obj_list, dim=dim)
    return gather_tensor


def all_gather_tensor(
    local_tensor: torch.Tensor,
    group: Optional[dist.ProcessGroup] = None,
    async_op: bool = False,
):
    sp_world_size = dist.get_world_size(group=group)
    output_shape = list(local_tensor.shape)
    output_shape[0] = output_shape[0] * sp_world_size
    output = torch.empty(
        output_shape, dtype=local_tensor.dtype, device=local_tensor.device
    )
    dist.all_gather_into_tensor(output, local_tensor, group=group, async_op=async_op)
    return output


# Adapted from https://github.com/volcengine/verl/blob/a0e8e4472b8b472409defb0c8fcc5162301450af/verl/utils/ulysses.py#L194
class Gather(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: Any,
        group: dist.ProcessGroup,
        local_tensor: torch.Tensor,
        gather_dim: int,
        grad_scaler: bool = True,
        async_op=False,
    ) -> torch.Tensor:
        ctx.group = group
        ctx.gather_dim = gather_dim
        ctx.grad_scaler = grad_scaler
        ctx.async_op = async_op

        sp_world_size = dist.get_world_size(group=group)
        ctx.sp_world_size = sp_world_size

        sp_rank = dist.get_rank(group=group)
        ctx.sp_rank = sp_rank

        local_shape = list(local_tensor.size())
        split_size = local_shape[0]
        part_size = local_shape[gather_dim]  # store original size
        ctx.part_size = part_size

        output = all_gather_tensor(local_tensor, group, async_op)
        return torch.cat(output.split(split_size, dim=0), dim=gather_dim)

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> Any:
        if ctx.grad_scaler:
            grad_output = grad_output * ctx.sp_world_size
        return (
            None,
            grad_output.split(ctx.part_size, dim=ctx.gather_dim)[
                ctx.sp_rank
            ].contiguous(),
            None,
            None,
            None,
            None,
        )


def gather_outputs_and_unpad(
    x: torch.Tensor,
    gather_dim: int,
    grad_scaler: bool = True,
    group: Optional[dist.ProcessGroup] = None,
):
    """
    Gather a tensor across a process group and optionally unpad its padded elements.

    Args:
        x (Tensor): Input tensor to gather.
        gather_dim (int): Dimension along which to gather across ranks.
        grad_scaler (bool): Whether to apply gradient scaling during gather. Defaults to True.
        group (ProcessGroup, optional): Process group for gathering. If None, uses
            `get_ulysses_sequence_parallel_group()`. If still None, returns `x` unchanged.

    Returns:
        Tensor: The gathered tensor, with padding removed if requested.
    """
    if not group:
        group = get_draft_sp_group()
    if torch.distributed.get_world_size(group) == 1:
        return x
    x = Gather.apply(group, x, gather_dim, grad_scaler)
    return x


def is_tp_rank_0():
    """Return True if current process is rank 0 in its TP group."""
    tp_group = get_tp_group()
    if tp_group is None:
        return True
    return dist.get_rank(group=tp_group) == 0


def get_tp_data_shard(tensor: torch.Tensor, dim: int = 0) -> torch.Tensor:
    """Return this TP rank's slice along ``dim`` (for per-rank draft micro-batches)."""
    tp_group = get_tp_group()
    if tp_group is None or dist.get_world_size(tp_group) == 1:
        return tensor
    return shard_tensor(tensor, process_group=tp_group, dim=dim)
