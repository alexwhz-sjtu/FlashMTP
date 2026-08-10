import unittest
from types import SimpleNamespace

from scripts.train_flashmtp import (
    serial_prefix_loss_for_epoch,
    validate_serial_loss_schedule,
)


class SerialLossScheduleTest(unittest.TestCase):
    def _args(self, **overrides):
        values = {
            "num_epochs": 6,
            "serial_full_loss_epochs": 4,
            "serial_prefix_loss_epochs": 2,
            "serial_loss_correct_prefix_only": False,
        }
        values.update(overrides)
        return SimpleNamespace(**values)

    def test_switches_after_full_loss_stage(self) -> None:
        args = self._args()
        validate_serial_loss_schedule(args)
        self.assertFalse(serial_prefix_loss_for_epoch(args, 0))
        self.assertFalse(serial_prefix_loss_for_epoch(args, 3))
        self.assertTrue(serial_prefix_loss_for_epoch(args, 4))
        self.assertTrue(serial_prefix_loss_for_epoch(args, 5))

    def test_stage_counts_must_match_total_epochs(self) -> None:
        with self.assertRaisesRegex(ValueError, "must sum to --num-epochs"):
            validate_serial_loss_schedule(
                self._args(serial_full_loss_epochs=3, serial_prefix_loss_epochs=2)
            )

    def test_static_prefix_mode_remains_supported(self) -> None:
        args = self._args(
            serial_full_loss_epochs=None,
            serial_prefix_loss_epochs=None,
            serial_loss_correct_prefix_only=True,
        )
        validate_serial_loss_schedule(args)
        self.assertTrue(serial_prefix_loss_for_epoch(args, 0))

    def test_zero_length_stage_is_supported(self) -> None:
        prefix_from_start = self._args(
            serial_full_loss_epochs=0,
            serial_prefix_loss_epochs=6,
        )
        validate_serial_loss_schedule(prefix_from_start)
        self.assertTrue(serial_prefix_loss_for_epoch(prefix_from_start, 0))

        full_loss_only = self._args(
            serial_full_loss_epochs=6,
            serial_prefix_loss_epochs=0,
        )
        validate_serial_loss_schedule(full_loss_only)
        self.assertFalse(serial_prefix_loss_for_epoch(full_loss_only, 5))


if __name__ == "__main__":
    unittest.main()
