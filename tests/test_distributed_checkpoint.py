import os
import tempfile
import unittest

import torch

from specforge.checkpoint import (
    load_distributed_training_state,
    ranked_training_state_path,
    save_distributed_training_state,
)


class DistributedCheckpointTest(unittest.TestCase):
    def test_each_rank_loads_its_own_optimizer_state(self) -> None:
        with tempfile.TemporaryDirectory() as checkpoint_dir:
            for rank in (1, 0):
                save_distributed_training_state(
                    checkpoint_dir,
                    {"optimizer_state_dict": {"rank_value": rank}},
                    rank=rank,
                    world_size=2,
                )

            rank0 = load_distributed_training_state(
                checkpoint_dir, rank=0, world_size=2
            )
            rank1 = load_distributed_training_state(
                checkpoint_dir, rank=1, world_size=2
            )

            self.assertEqual(rank0["optimizer_state_dict"]["rank_value"], 0)
            self.assertEqual(rank1["optimizer_state_dict"]["rank_value"], 1)
            self.assertTrue(
                os.path.isfile(ranked_training_state_path(checkpoint_dir, 0))
            )
            self.assertTrue(
                os.path.isfile(ranked_training_state_path(checkpoint_dir, 1))
            )
            self.assertTrue(
                os.path.isfile(os.path.join(checkpoint_dir, "training_state.pt"))
            )

    def test_missing_rank_shard_does_not_fall_back_to_rank_zero(self) -> None:
        with tempfile.TemporaryDirectory() as checkpoint_dir:
            save_distributed_training_state(
                checkpoint_dir,
                {"optimizer_state_dict": {}},
                rank=0,
                world_size=2,
            )

            with self.assertRaisesRegex(RuntimeError, "global rank 1 is missing"):
                load_distributed_training_state(checkpoint_dir, rank=1, world_size=2)

    def test_world_size_mismatch_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as checkpoint_dir:
            save_distributed_training_state(
                checkpoint_dir,
                {"optimizer_state_dict": {}},
                rank=0,
                world_size=2,
            )

            with self.assertRaisesRegex(RuntimeError, "different world size"):
                load_distributed_training_state(checkpoint_dir, rank=0, world_size=4)

    def test_legacy_single_file_checkpoint_still_loads(self) -> None:
        with tempfile.TemporaryDirectory() as checkpoint_dir:
            legacy_state = {"epoch": 3, "optimizer_state_dict": {}}
            torch.save(legacy_state, os.path.join(checkpoint_dir, "training_state.pt"))

            loaded = load_distributed_training_state(
                checkpoint_dir, rank=7, world_size=8
            )

            self.assertEqual(loaded, legacy_state)


if __name__ == "__main__":
    unittest.main()
