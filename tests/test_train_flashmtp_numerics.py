import unittest
from unittest import mock
from types import SimpleNamespace

import torch

from scripts.train_flashmtp import (
    nonfinite_tensor_names,
    record_skipped_update,
    validate_numeric_training_args,
    validate_training_batch,
)


class FlashMTPTrainingNumericsTest(unittest.TestCase):
    def _args(self, **overrides):
        values = {
            "learning_rate": 1e-3,
            "max_grad_norm": 1.0,
            "final_ce_weight": 1.0,
            "tv_loss_weight": 0.0,
            "base_lm_ce_weight": 0.0,
            "target_hidden_noise_ratio": 0.1,
            "warmup_ratio": 0.04,
            "num_epochs": 2,
            "batch_size": 1,
            "block_size": 4,
            "num_anchors": 8,
            "accumulation_steps": 1,
            "ce_chunk_size": 32,
            "log_interval": 10,
            "eval_interval": 50,
            "save_interval": 100,
        }
        values.update(overrides)
        return SimpleNamespace(**values)

    def test_valid_batch_and_hyperparameters(self) -> None:
        validate_numeric_training_args(self._args())
        errors = validate_training_batch(
            torch.tensor([[0, 3, 7]]),
            torch.ones(1, 3, dtype=torch.long),
            torch.tensor([[0, 1, 1]]),
            vocab_size=8,
        )
        self.assertEqual(errors, [])

    def test_out_of_range_label_and_nonbinary_mask_are_reported(self) -> None:
        errors = validate_training_batch(
            torch.tensor([[0, 8]]),
            torch.tensor([[1, 2]]),
            torch.ones(1, 2, dtype=torch.long),
            vocab_size=8,
        )
        self.assertTrue(any("out of range" in error for error in errors))
        self.assertTrue(any("other than 0/1" in error for error in errors))

    def test_nonfinite_tensors_are_named(self) -> None:
        names = nonfinite_tensor_names(
            {
                "finite": torch.ones(2),
                "nan_tensor": torch.tensor([float("nan")]),
                "missing": None,
            }
        )
        self.assertEqual(names, ["nan_tensor"])

    def test_skipped_update_warning_does_not_crash(self) -> None:
        tracker = mock.Mock()
        with self.assertLogs("scripts.train_flashmtp", level="WARNING") as logs:
            record_skipped_update(
                tracker,
                global_step=1,
                skipped_update_count=1,
                reason="nonfinite_gradients: params=[weight]",
            )

        self.assertIn("skipped unsafe training update", logs.output[0])
        tracker.log.assert_called_once_with(
            {
                "train/skipped_unsafe_update": 1,
                "train/skipped_unsafe_updates_total": 1,
            },
            step=1,
        )

    def test_invalid_numeric_args_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "learning-rate"):
            validate_numeric_training_args(self._args(learning_rate=float("nan")))
        with self.assertRaisesRegex(ValueError, "loss weight"):
            validate_numeric_training_args(
                self._args(
                    final_ce_weight=0.0,
                    tv_loss_weight=0.0,
                    base_lm_ce_weight=0.0,
                )
            )


if __name__ == "__main__":
    unittest.main()
