import unittest

import torch

from specforge.disaggregate import (
    DraftBatchPacket,
    DraftPacketSpec,
    build_node_routes,
    copy_packet_slice,
)
from specforge.core.flashmtp import create_flashmtp_dense_mask


class DisaggregateRoutingTest(unittest.TestCase):
    def test_dense_debug_mask_keeps_blocks_isolated(self) -> None:
        keep = torch.tensor([[True, False]])
        mask = create_flashmtp_dense_mask(
            keep, chs_len_per_block=1, block_size=2, dtype=torch.float32
        )
        self.assertEqual(tuple(mask.shape), (1, 1, 4, 6))
        allowed = mask[0, 0] == 0
        # Valid block 0 sees CHS_0 and both draft positions from block 0.
        self.assertEqual(allowed[0].nonzero().flatten().tolist(), [0, 2, 3])
        # Invalid block 1 retains only CHS_1 as a finite softmax fallback.
        self.assertEqual(allowed[2].nonzero().flatten().tolist(), [1])

    def test_six_target_producers_to_two_drafts(self) -> None:
        routes = build_node_routes(producers=6, drafts=2, node_batch_size=12)
        self.assertEqual(len(routes), 6)
        self.assertTrue(all(route.size == 2 for route in routes))
        self.assertEqual([route.draft for route in routes], [0, 0, 0, 1, 1, 1])

    def test_tp_two_three_producers_can_split_middle_shard(self) -> None:
        routes = build_node_routes(producers=3, drafts=2, node_batch_size=12)
        middle = [route for route in routes if route.producer == 1]
        self.assertEqual([(route.draft, route.size) for route in middle], [(0, 2), (1, 2)])
        self.assertEqual(sum(route.size for route in routes), 12)

    def test_packet_schema_carries_hidden_not_logits(self) -> None:
        spec = DraftPacketSpec(
            batch_size=2,
            max_length=8,
            num_anchors=3,
            num_target_layers=2,
            hidden_size=4,
            prediction_length=2,
            include_target_prediction_hidden=True,
        )
        source = DraftBatchPacket.empty(spec, device="cpu", temp_rollout=True)
        destination = DraftBatchPacket.empty(spec, device="cpu", temp_rollout=True)
        source.target_prediction_hidden.fill_(7)
        copy_packet_slice(
            source,
            destination,
            source_start=0,
            source_end=2,
            destination_start=0,
            destination_end=2,
        )
        self.assertEqual(tuple(destination.target_prediction_hidden.shape), (2, 3, 2, 4))
        self.assertTrue(torch.equal(source.target_prediction_hidden, destination.target_prediction_hidden))
        self.assertFalse(hasattr(destination, "target_logits"))


if __name__ == "__main__":
    unittest.main()
