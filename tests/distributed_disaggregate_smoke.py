"""Two-GPU NCCL smoke test for the node-local packet transport."""

import argparse

import torch

from specforge.disaggregate import (
    DraftBatchPacket,
    DraftPacketSpec,
    NodePacketTransport,
    build_node_routes,
)
from specforge.distributed import destroy_distributed, init_disaggregated


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target-ranks", type=int, default=1)
    parser.add_argument("--draft-ranks", type=int, default=1)
    parser.add_argument("--target-tp", type=int, default=1)
    parser.add_argument("--node-batch", type=int, default=2)
    args = parser.parse_args()
    topology = init_disaggregated(
        timeout=2,
        target_ranks_per_node=args.target_ranks,
        draft_ranks_per_node=args.draft_ranks,
        target_tp_size=args.target_tp,
    )
    routes = build_node_routes(
        producers=topology.target_replicas_per_node,
        drafts=topology.draft_ranks_per_node,
        node_batch_size=args.node_batch,
    )
    if not (topology.is_target_leader or topology.is_draft):
        torch.distributed.barrier()
        destroy_distributed()
        return
    local_batch = (
        args.node_batch // topology.target_replicas_per_node
        if topology.is_target
        else args.node_batch // topology.draft_ranks_per_node
    )
    spec = DraftPacketSpec(
        batch_size=local_batch,
        max_length=4,
        num_anchors=2,
        num_target_layers=2,
        hidden_size=4,
        prediction_length=2,
        include_target_prediction_hidden=True,
    )
    transport = NodePacketTransport(topology=topology, routes=routes)
    if topology.is_target_leader:
        packet = DraftBatchPacket.empty(spec, device="cuda", temp_rollout=True)
        for index, tensor in enumerate(packet.tensors()):
            tensor.fill_(100 * topology.target_replica_local_rank + index + 1)
        transport.wait_send(transport.send(packet, batch_id=17))
    else:
        packet = DraftBatchPacket.empty(spec, device="cuda", temp_rollout=True)
        transport.wait_receive(transport.receive(packet, expected_batch_id=17))
        relevant = [route for route in routes if route.draft == topology.draft_local_rank]
        for index, tensor in enumerate(packet.tensors()):
            for route in relevant:
                expected = torch.full_like(
                    tensor[route.draft_start : route.draft_end],
                    100 * route.producer + index + 1,
                )
                torch.testing.assert_close(
                    tensor[route.draft_start : route.draft_end], expected
                )
    torch.distributed.barrier()
    destroy_distributed()


if __name__ == "__main__":
    main()
