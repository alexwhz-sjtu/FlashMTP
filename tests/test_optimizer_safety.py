import unittest
from unittest import mock

import torch
from torch import nn

from specforge.optimizer import BF16Optimizer


class BF16OptimizerSafetyTest(unittest.TestCase):
    def _optimizer(self, model: nn.Module, max_grad_norm: float = 1.0) -> BF16Optimizer:
        return BF16Optimizer(
            model,
            lr=1e-3,
            max_grad_norm=max_grad_norm,
            total_steps=10,
            warmup_ratio=0.1,
        )

    def test_nonfinite_gradient_skips_update_and_scheduler(self) -> None:
        model = nn.Linear(2, 1, bias=False)
        optimizer = self._optimizer(model)
        before = model.weight.detach().clone()
        scheduler_epoch = optimizer.scheduler.last_epoch
        model.weight.grad = torch.full_like(model.weight, float("nan"))

        result = optimizer.step()

        self.assertFalse(result.updated)
        self.assertEqual(result.reason, "nonfinite_or_missing_gradients")
        self.assertTrue(torch.equal(model.weight, before))
        self.assertEqual(optimizer.scheduler.last_epoch, scheduler_epoch)
        self.assertIsNone(model.weight.grad)

    def test_finite_gradient_is_globally_clipped_and_updated(self) -> None:
        model = nn.Linear(2, 1, bias=False)
        optimizer = self._optimizer(model, max_grad_norm=0.25)
        model.weight.grad = torch.tensor([[3.0, 4.0]])
        captured_norms: list[float] = []
        original_step = optimizer.optimizer.step

        def capture_step(*args, **kwargs):
            grads = [
                param.grad.reshape(-1)
                for param in optimizer.fp32_params
                if param.grad is not None
            ]
            captured_norms.append(float(torch.cat(grads).norm().item()))
            return original_step(*args, **kwargs)

        optimizer.optimizer.step = capture_step
        result = optimizer.step()

        self.assertTrue(result.updated)
        self.assertAlmostEqual(result.grad_norm, 5.0, places=5)
        self.assertLessEqual(captured_norms[0], 0.250001)

    def test_invalid_clip_norm_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "finite and positive"):
            self._optimizer(nn.Linear(2, 1), max_grad_norm=0.0)

    def test_partial_accumulation_gradient_can_be_rescaled(self) -> None:
        model = nn.Linear(2, 1, bias=False)
        optimizer = self._optimizer(model, max_grad_norm=10.0)
        model.weight.grad = torch.tensor([[0.5, 0.0]])

        optimizer.scale_model_gradients(4.0)
        result = optimizer.step()

        self.assertTrue(result.updated)
        self.assertAlmostEqual(result.grad_norm, 2.0, places=5)

    def test_global_norm_all_reduces_one_scalar_only(self) -> None:
        model = nn.Linear(2, 1, bias=False)
        optimizer = self._optimizer(model)
        model.weight.grad = torch.tensor([[3.0, 4.0]])
        reduced_tensors: list[torch.Tensor] = []

        def capture_all_reduce(tensor, op):
            reduced_tensors.append(tensor)

        with (
            mock.patch("specforge.optimizer.dist.is_initialized", return_value=True),
            mock.patch(
                "specforge.optimizer.dist.all_reduce", side_effect=capture_all_reduce
            ),
        ):
            result = optimizer.step()

        self.assertTrue(result.updated)
        self.assertEqual(len(reduced_tensors), 1)
        self.assertEqual(reduced_tensors[0].ndim, 0)
        self.assertEqual(reduced_tensors[0].numel(), 1)

    def test_nonfinite_loaded_adam_moments_are_discarded(self) -> None:
        source_model = nn.Linear(2, 1, bias=False)
        source = self._optimizer(source_model)
        source_model.weight.grad = torch.ones_like(source_model.weight)
        self.assertTrue(source.step().updated)
        state = source.state_dict()
        first_state = next(iter(state["optimizer_state_dict"]["state"].values()))
        first_state["exp_avg"].fill_(float("nan"))

        target = self._optimizer(nn.Linear(2, 1, bias=False))
        loaded = target.load_state_dict(state)

        self.assertFalse(loaded)
        self.assertEqual(len(target.optimizer.state), 0)


if __name__ == "__main__":
    unittest.main()
