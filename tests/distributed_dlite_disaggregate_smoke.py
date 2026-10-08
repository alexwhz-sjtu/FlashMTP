"""NCCL transport smoke test; launch with torchrun (2 or 8 GPUs)."""

import argparse

import torch
import torch.distributed as dist

from specforge.disaggregate import (
    DraftBatchPacket,
    DraftPacketSpec,
    NodePacketTransport,
    build_node_routes,
)
from specforge.distributed import destroy_distributed, init_disaggregated


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target-ranks", type=int, default=6)
    parser.add_argument("--draft-ranks", type=int, default=2)
    parser.add_argument("--target-tp", type=int, default=2)
    parser.add_argument("--target-ep", type=int, default=2)
    parser.add_argument("--node-batch", type=int, default=6)
    args = parser.parse_args()
    topology = init_disaggregated(
        timeout=5,
        target_ranks_per_node=args.target_ranks,
        draft_ranks_per_node=args.draft_ranks,
        target_tp_size=args.target_tp,
        target_ep_size=args.target_ep,
    )
    routes = build_node_routes(
        producers=topology.target_replicas_per_node,
        drafts=topology.draft_ranks_per_node,
        node_batch_size=args.node_batch,
    )
    if topology.is_target:
        value = torch.tensor([topology.target_tp_rank], device="cuda")
        dist.all_reduce(value, group=topology.target_tp_group)
        assert int(value.item()) == sum(range(topology.target_tp_size))
    if topology.is_target and not topology.is_target_leader:
        dist.barrier()
        destroy_distributed()
        return
    local_batch = (
        args.node_batch // topology.target_replicas_per_node
        if topology.is_target
        else args.node_batch // topology.draft_ranks_per_node
    )
    spec = DraftPacketSpec(local_batch, 4, 2, 2, 4, 2)
    packet = DraftBatchPacket.empty(spec, device="cuda")
    transport = NodePacketTransport(topology=topology, routes=routes)
    if topology.is_target_leader:
        for index, tensor in enumerate(packet.tensors()):
            tensor.fill_(100 * topology.target_replica_local_rank + index + 1)
        transport.wait_send(
            transport.send(packet, batch_id=17, phase_id=3, schema_id=spec.schema_id)
        )
    else:
        transport.wait_receive(
            transport.receive(packet, batch_id=17, phase_id=3, schema_id=spec.schema_id)
        )
        reduced = torch.tensor([float(topology.draft_local_rank + 1)], device="cuda")
        dist.all_reduce(reduced, group=topology.draft_group)
        assert int(reduced.item()) == sum(range(1, topology.draft_ranks_per_node + 1))
    dist.barrier()
    destroy_distributed()


if __name__ == "__main__":
    main()
