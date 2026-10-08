"""Compact target-to-draft packets and deterministic node-local routing."""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Iterable, Optional

import torch
import torch.distributed as dist


@dataclass(frozen=True)
class RouteFragment:
    producer: int
    draft: int
    producer_start: int
    producer_end: int
    draft_start: int
    draft_end: int

    @property
    def size(self) -> int:
        return self.producer_end - self.producer_start


def build_node_routes(
    *, producers: int, drafts: int, node_batch_size: int
) -> list[RouteFragment]:
    """Intersect producer and draft ranges in one canonical node batch."""
    if min(producers, drafts, node_batch_size) <= 0:
        raise ValueError("producers, drafts and node_batch_size must be positive")
    if node_batch_size % producers:
        raise ValueError("node_batch_size must be divisible by target producers")
    if node_batch_size % drafts:
        raise ValueError("node_batch_size must be divisible by draft ranks")
    producer_batch = node_batch_size // producers
    draft_batch = node_batch_size // drafts
    routes: list[RouteFragment] = []
    for producer in range(producers):
        p0, p1 = producer * producer_batch, (producer + 1) * producer_batch
        for draft in range(drafts):
            d0, d1 = draft * draft_batch, (draft + 1) * draft_batch
            start, end = max(p0, d0), min(p1, d1)
            if start < end:
                routes.append(
                    RouteFragment(
                        producer,
                        draft,
                        start - p0,
                        end - p0,
                        start - d0,
                        end - d0,
                    )
                )
    if sum(route.size for route in routes) != node_batch_size:
        raise RuntimeError("route construction did not cover the node batch")
    return routes


@dataclass(frozen=True)
class DraftPacketSpec:
    batch_size: int
    max_length: int
    num_anchors: int
    num_target_layers: int
    hidden_size: int
    prediction_length: int
    num_history_layers: int = 0
    include_target_prediction_hidden: bool = False

    @property
    def schema_id(self) -> int:
        return (
            2
            + int(self.num_history_layers > 0)
            + 2 * int(self.include_target_prediction_hidden)
        )


@dataclass
class DraftBatchPacket:
    """Fixed-shape GPU payload. Full-vocabulary logits are never transported."""

    input_ids: torch.Tensor
    loss_mask: torch.Tensor
    anchor_positions: torch.Tensor
    block_keep_mask: torch.Tensor
    target_hidden: torch.Tensor
    target_history_hidden: Optional[torch.Tensor]
    target_prediction_hidden: Optional[torch.Tensor]

    @classmethod
    def empty(
        cls, spec: DraftPacketSpec, *, device: torch.device | str
    ) -> "DraftBatchPacket":
        b, length, n = spec.batch_size, spec.max_length, spec.num_anchors
        h, c, k = spec.hidden_size, spec.num_target_layers, spec.prediction_length
        return cls(
            input_ids=torch.empty((b, length), dtype=torch.long, device=device),
            loss_mask=torch.empty((b, length), dtype=torch.float32, device=device),
            anchor_positions=torch.empty((b, n), dtype=torch.long, device=device),
            block_keep_mask=torch.empty((b, n), dtype=torch.bool, device=device),
            target_hidden=torch.empty(
                (b, n, c, h), dtype=torch.bfloat16, device=device
            ),
            target_history_hidden=(
                torch.empty(
                    (b, length, spec.num_history_layers, h),
                    dtype=torch.bfloat16,
                    device=device,
                )
                if spec.num_history_layers
                else None
            ),
            target_prediction_hidden=(
                torch.empty((b, n, k, h), dtype=torch.bfloat16, device=device)
                if spec.include_target_prediction_hidden
                else None
            ),
        )

    def tensors(self) -> Iterable[torch.Tensor]:
        for field in fields(self):
            value = getattr(self, field.name)
            if value is not None:
                yield value

    def batch_slice(self, start: int, end: int) -> list[torch.Tensor]:
        return [tensor[start:end] for tensor in self.tensors()]


@dataclass(frozen=True)
class FamilyPacketSpec:
    """Fixed full-context schema for DFlash-family training."""

    batch_size: int
    max_length: int
    context_width: int
    hidden_size: int
    include_target_last_hidden: bool = False

    @property
    def schema_id(self) -> int:
        return 10 + int(self.include_target_last_hidden)


@dataclass
class FamilyBatchPacket:
    """DFlash-family features; never contains full-vocabulary logits."""

    input_ids: torch.Tensor
    loss_mask: torch.Tensor
    target_context_hidden: torch.Tensor
    target_last_hidden: Optional[torch.Tensor]

    @classmethod
    def empty(
        cls, spec: FamilyPacketSpec, *, device: torch.device | str
    ) -> "FamilyBatchPacket":
        b, length = spec.batch_size, spec.max_length
        return cls(
            input_ids=torch.empty((b, length), dtype=torch.long, device=device),
            loss_mask=torch.empty((b, length), dtype=torch.float32, device=device),
            target_context_hidden=torch.empty(
                (b, length, spec.context_width),
                dtype=torch.bfloat16,
                device=device,
            ),
            target_last_hidden=(
                torch.empty(
                    (b, length, spec.hidden_size),
                    dtype=torch.bfloat16,
                    device=device,
                )
                if spec.include_target_last_hidden
                else None
            ),
        )

    def tensors(self) -> Iterable[torch.Tensor]:
        for field in fields(self):
            value = getattr(self, field.name)
            if value is not None:
                yield value

    def batch_slice(self, start: int, end: int) -> list[torch.Tensor]:
        return [tensor[start:end] for tensor in self.tensors()]


def copy_packet_slice(
    source: DraftBatchPacket,
    destination: DraftBatchPacket,
    *,
    source_start: int,
    source_end: int,
    destination_start: int,
    destination_end: int,
) -> None:
    source_tensors = list(source.tensors())
    destination_tensors = list(destination.tensors())
    if len(source_tensors) != len(destination_tensors):
        raise ValueError("source and destination packet schemas differ")
    if source_end - source_start != destination_end - destination_start:
        raise ValueError("source and destination packet slices differ in size")
    for source_tensor, destination_tensor in zip(
        source_tensors, destination_tensors, strict=True
    ):
        destination_tensor[destination_start:destination_end].copy_(
            source_tensor[source_start:source_end]
        )


class NodePacketTransport:
    """NCCL P2P transport over target leaders and draft ranks on one node."""

    def __init__(self, *, topology, routes: list[RouteFragment], profile=False):
        self.topology = topology
        self.routes = routes
        self.profile = bool(profile)
        self.group = topology.bridge_group
        if self.group is None:
            raise ValueError("rank is not a member of its node bridge group")
        self.stream = torch.cuda.Stream(priority=-1)

    def _producer_rank(self, producer: int) -> int:
        return self.topology.node_target_leader_ranks[producer]

    def _draft_rank(self, draft: int) -> int:
        return self.topology.node_draft_ranks[draft]

    def send(
        self, packet: DraftBatchPacket, *, batch_id: int, phase_id: int, schema_id: int
    ):
        producer = self.topology.target_replica_local_rank
        if not self.topology.is_target_leader or producer is None:
            raise RuntimeError("only target TP leaders send draft packets")
        ops = []
        keepalive = list(packet.tensors())
        ready = torch.cuda.Event()
        ready.record(torch.cuda.current_stream())
        with torch.cuda.stream(self.stream):
            self.stream.wait_event(ready)
            start = torch.cuda.Event(enable_timing=True) if self.profile else None
            if start is not None:
                start.record(self.stream)
            for route in self.routes:
                if route.producer != producer:
                    continue
                peer = self._draft_rank(route.draft)
                meta = torch.tensor(
                    [batch_id, phase_id, schema_id], dtype=torch.long, device="cuda"
                )
                keepalive.append(meta)
                ops.append(dist.P2POp(dist.isend, meta, peer, group=self.group))
                for tensor in packet.batch_slice(
                    route.producer_start, route.producer_end
                ):
                    value = tensor if tensor.is_contiguous() else tensor.contiguous()
                    if value is not tensor:
                        keepalive.append(value)
                    ops.append(dist.P2POp(dist.isend, value, peer, group=self.group))
            works = dist.batch_isend_irecv(ops) if ops else []
            done = torch.cuda.Event(enable_timing=self.profile)
            done.record(self.stream)
        return works, keepalive, start, done

    def receive(
        self,
        packet: DraftBatchPacket,
        *,
        batch_id: int,
        phase_id: int,
        schema_id: int,
    ):
        draft = self.topology.draft_local_rank
        if not self.topology.is_draft or draft is None:
            raise RuntimeError("only draft ranks receive draft packets")
        ops, metas = [], []
        with torch.cuda.stream(self.stream):
            start = torch.cuda.Event(enable_timing=True) if self.profile else None
            if start is not None:
                start.record(self.stream)
            for route in self.routes:
                if route.draft != draft:
                    continue
                peer = self._producer_rank(route.producer)
                meta = torch.empty((3,), dtype=torch.long, device="cuda")
                metas.append(meta)
                ops.append(dist.P2POp(dist.irecv, meta, peer, group=self.group))
                for tensor in packet.batch_slice(route.draft_start, route.draft_end):
                    ops.append(dist.P2POp(dist.irecv, tensor, peer, group=self.group))
            works = dist.batch_isend_irecv(ops) if ops else []
            done = torch.cuda.Event(enable_timing=self.profile)
            done.record(self.stream)
        return works, metas, (batch_id, phase_id, schema_id), start, done

    @staticmethod
    def wait_receive(handle) -> float:
        works, metas, expected, start, done = handle
        for work in works:
            work.wait()
        done.synchronize()
        for meta in metas:
            actual = tuple(int(value) for value in meta.tolist())
            if actual != expected:
                raise RuntimeError(
                    f"packet metadata mismatch: expected {expected}, got {actual}"
                )
        return 0.0 if start is None else start.elapsed_time(done)

    @staticmethod
    def wait_send(handle) -> float:
        works, _keepalive, start, done = handle
        for work in works:
            work.wait()
        done.synchronize()
        return 0.0 if start is None else start.elapsed_time(done)
