import unittest
from unittest import mock

try:
    import torch
except (ImportError, OSError) as exc:
    raise unittest.SkipTest(f"PyTorch runtime is unavailable: {exc}") from exc

from scripts.dlite.dlite_training import normalize_accumulated_gradients


class _RecordingOptimizer:
    def __init__(self):
        self.scale = None

    def scale_model_gradients(self, scale):
        self.scale = float(scale)


class GradientNormalizationTest(unittest.TestCase):
    def test_uses_global_window_denominator(self):
        optimizer = _RecordingOptimizer()

        def add_remote_denominator(denominator, *, op, group):
            self.assertEqual(op, torch.distributed.ReduceOp.SUM)
            self.assertEqual(group, "dp")
            denominator.add_(7.0)

        with (
            mock.patch("torch.distributed.is_available", return_value=True),
            mock.patch("torch.distributed.is_initialized", return_value=True),
            mock.patch("torch.distributed.get_world_size", return_value=2),
            mock.patch(
                "torch.distributed.all_reduce",
                side_effect=add_remote_denominator,
            ),
        ):
            global_denominator = normalize_accumulated_gradients(
                optimizer,
                torch.tensor(3.0),
                accumulation_steps=4,
                group="dp",
            )

        self.assertEqual(global_denominator, 10.0)
        self.assertAlmostEqual(optimizer.scale, 0.8)

    def test_rejects_invalid_global_denominator(self):
        optimizer = _RecordingOptimizer()
        for denominator in (0.0, -1.0, float("nan"), float("inf")):
            with self.subTest(denominator=denominator):
                with self.assertRaisesRegex(ValueError, "finite and positive"):
                    normalize_accumulated_gradients(
                        optimizer,
                        torch.tensor(denominator),
                        accumulation_steps=2,
                    )


if __name__ == "__main__":
    unittest.main()
