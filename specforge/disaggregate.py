"""Compact node-local target-to-draft packets and deterministic routing."""

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
    *, producers: int,
    drafts: int,
    node_batch_size: int,
) -> list[RouteFragment]:
    """Intersect producer and draft ranges in a canonical node-local batch."""
    if producers <= 0 or drafts <= 0 or node_batch_size <= 0:
        raise ValueError("producers, drafts, and node_batch_size must be positive.")
    if node_batch_size % producers:
        raise ValueError("node_batch_size must be divisible by target producers.")
    if node_batch_size % drafts:
        raise ValueError("node_batch_size must be divisible by draft ranks.")
    producer_batch = node_batch_size // producers
    draft_batch = node_batch_size // drafts
    routes: list[RouteFragment] = []
    for producer in range(producers):
        p0, p1 = producer * producer_batch, (producer + 1) * producer_batch
        for draft in range(drafts):
            d0, d1 = draft * draft_batch, (draft + 1) * draft_batch
            start, end = max(p0, d0), min(p1, d1)
            if start >= end:
                continue
            routes.append(
                RouteFragment(
                    producer=producer,
                    draft=draft,
                    producer_start=start - p0,
                    producer_end=end - p0,
                    draft_start=start - d0,
                    draft_end=end - d0,
                )
            )
    if sum(route.size for route in routes) != node_batch_size:
        raise RuntimeError("route construction did not cover the node batch exactly.")
    return routes


@dataclass(frozen=True)
class DraftPacketSpec:
    batch_size: int
    max_length: int
    num_anchors: int
    num_target_layers: int
    hidden_size: int
    prediction_length: int
    include_target_prediction_hidden: bool


@dataclass
class DraftBatchPacket:
    """GPU tensors consumed without any live target model or KV state."""

    input_ids: torch.Tensor
    loss_mask: torch.Tensor
    anchor_positions: torch.Tensor
    block_keep_mask: torch.Tensor
    target_hidden: torch.Tensor
    target_prediction_hidden: Optional[torch.Tensor]
    rollout_ids: Optional[torch.Tensor]
    rollout_validity: Optional[torch.Tensor]

    @classmethod
    def empty(
        cls,
        spec: DraftPacketSpec,
        *,
        device: torch.device | str,
        temp_rollout: bool,
    ) -> "DraftBatchPacket":
        b, l, n = spec.batch_size, spec.max_length, spec.num_anchors
        h, c, k = spec.hidden_size, spec.num_target_layers, spec.prediction_length
        return cls(
            input_ids=torch.empty((b, l), dtype=torch.long, device=device),
            loss_mask=torch.empty((b, l), dtype=torch.float32, device=device),
            anchor_positions=torch.empty((b, n), dtype=torch.long, device=device),
            block_keep_mask=torch.empty((b, n), dtype=torch.bool, device=device),
            target_hidden=torch.empty((b, n, c, h), dtype=torch.bfloat16, device=device),
            target_prediction_hidden=(
                torch.empty((b, n, k, h), dtype=torch.bfloat16, device=device)
                if spec.include_target_prediction_hidden
                else None
            ),
            rollout_ids=(
                torch.empty((b, n, k), dtype=torch.long, device=device)
                if temp_rollout
                else None
            ),
            rollout_validity=(
                torch.empty((b, n, k), dtype=torch.bool, device=device)
                if temp_rollout
                else None
            ),
        )

    def tensors(self) -> Iterable[torch.Tensor]:
        for item in fields(self):
            value = getattr(self, item.name)
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
    src_tensors = list(source.tensors())
    dst_tensors = list(destination.tensors())
    if len(src_tensors) != len(dst_tensors):
        raise ValueError("source and destination packet schemas differ.")
    for src, dst in zip(src_tensors, dst_tensors):
        dst[destination_start:destination_end].copy_(src[source_start:source_end])


class NodePacketTransport:
    """NCCL P2P transport whose route table never crosses node boundaries."""

    def __init__(
        self,
        *,
        topology,
        routes: list[RouteFragment],
        profile: bool = False,
    ):
        self.topology = topology
        self.routes = routes
        self.profile = bool(profile)
        self.group = topology.bridge_group
        if self.group is None:
            raise ValueError("rank is not a member of its node bridge group.")
        self.stream = torch.cuda.Stream(priority=-1)

    def _producer_global_rank(self, producer: int) -> int:
        return self.topology.node_target_leader_ranks[producer]

    def _draft_global_rank(self, draft: int) -> int:
        return self.topology.node_draft_ranks[draft]

    def send(self, packet: DraftBatchPacket, batch_id: int):
        producer = self.topology.target_replica_local_rank
        if not self.topology.is_target_leader or producer is None:
            raise RuntimeError("only target TP leaders send draft packets.")
        ops = []
        keepalive: list[torch.Tensor] = list(packet.tensors())
        ready = torch.cuda.Event()
        ready.record(torch.cuda.current_stream())
        with torch.cuda.stream(self.stream):
            self.stream.wait_event(ready)
            comm_start = None
            if self.profile:
                comm_start = torch.cuda.Event(enable_timing=True)
                comm_start.record(self.stream)
            for route in self.routes:
                if route.producer != producer:
                    continue
                peer = self._draft_global_rank(route.draft)
                meta = torch.tensor([batch_id], dtype=torch.long, device="cuda")
                keepalive.append(meta)
                ops.append(dist.P2POp(dist.isend, meta, peer, group=self.group))
                for tensor in packet.batch_slice(
                    route.producer_start, route.producer_end
                ):
                    if not tensor.is_contiguous():
                        tensor = tensor.contiguous()
                        keepalive.append(tensor)
                    ops.append(dist.P2POp(dist.isend, tensor, peer, group=self.group))
            works = dist.batch_isend_irecv(ops) if ops else []
            done = torch.cuda.Event(enable_timing=self.profile)
            done.record(self.stream)
        return works, keepalive, comm_start, done

    def receive(self, packet: DraftBatchPacket, expected_batch_id: int):
        draft = self.topology.draft_local_rank
        if not self.topology.is_draft or draft is None:
            raise RuntimeError("only draft ranks receive draft packets.")
        ops = []
        metas: list[torch.Tensor] = []
        with torch.cuda.stream(self.stream):
            comm_start = None
            if self.profile:
                comm_start = torch.cuda.Event(enable_timing=True)
                comm_start.record(self.stream)
            for route in self.routes:
                if route.draft != draft:
                    continue
                peer = self._producer_global_rank(route.producer)
                meta = torch.empty((1,), dtype=torch.long, device="cuda")
                metas.append(meta)
                ops.append(dist.P2POp(dist.irecv, meta, peer, group=self.group))
                for tensor in packet.batch_slice(route.draft_start, route.draft_end):
                    ops.append(dist.P2POp(dist.irecv, tensor, peer, group=self.group))
            works = dist.batch_isend_irecv(ops) if ops else []
            done = torch.cuda.Event(enable_timing=self.profile)
            done.record(self.stream)
        return works, metas, expected_batch_id, comm_start, done

    @staticmethod
    def wait_receive(handle) -> float:
        works, metas, expected_batch_id, comm_start, done = handle
        for work in works:
            work.wait()
        done.synchronize()
        for meta in metas:
            actual = int(meta.item())
            if actual != expected_batch_id:
                raise RuntimeError(
                    f"packet batch id mismatch: expected {expected_batch_id}, got {actual}."
                )
        return 0.0 if comm_start is None else comm_start.elapsed_time(done)

    @staticmethod
    def wait_send(handle) -> float:
        works, _keepalive, comm_start, done = handle
        for work in works:
            work.wait()
        done.synchronize()
        return 0.0 if comm_start is None else comm_start.elapsed_time(done)
