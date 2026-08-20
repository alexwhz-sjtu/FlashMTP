import unittest
from unittest import mock

import torch
from torch import nn

from specforge.lr_scheduler import CosineAnnealingWarmupLR
from specforge.optimizer import BF16Optimizer


class ModelWithMarkovHead(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.backbone = nn.Linear(2, 2)
        self.markov_head = nn.Linear(2, 1)


class BF16OptimizerSafetyTest(unittest.TestCase):
    def _optimizer(self, model: nn.Module, max_grad_norm: float = 1.0) -> BF16Optimizer:
        return BF16Optimizer(
            model,
            lr=1e-3,
            max_grad_norm=max_grad_norm,
            total_steps=10,
            warmup_ratio=0.1,
        )

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

        self.assertAlmostEqual(result, 5.0, places=5)
        self.assertLessEqual(captured_norms[0], 0.250001)

    def test_invalid_clip_norm_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "positive"):
            self._optimizer(nn.Linear(2, 1), max_grad_norm=0.0)

    def test_partial_accumulation_gradient_can_be_rescaled(self) -> None:
        model = nn.Linear(2, 1, bias=False)
        optimizer = self._optimizer(model, max_grad_norm=10.0)
        model.weight.grad = torch.tensor([[0.5, 0.0]])

        optimizer.scale_model_gradients(4.0)
        result = optimizer.step()

        self.assertAlmostEqual(result, 2.0, places=5)

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
            optimizer.step()

        self.assertEqual(len(reduced_tensors), 1)
        self.assertEqual(reduced_tensors[0].ndim, 0)
        self.assertEqual(reduced_tensors[0].numel(), 1)

    def test_markov_head_uses_separate_lr_group(self) -> None:
        optimizer = BF16Optimizer(
            ModelWithMarkovHead(),
            lr=1e-3,
            markov_lr_multiplier=0.5,
            total_steps=10,
            warmup_ratio=0.2,
        )

        learning_rates = optimizer.get_learning_rates()
        self.assertEqual(set(learning_rates), {"backbone", "markov_head"})
        self.assertAlmostEqual(
            learning_rates["markov_head"] / learning_rates["backbone"], 0.5
        )

        for _ in range(5):
            optimizer.scheduler.step()
            learning_rates = optimizer.get_learning_rates()
            self.assertAlmostEqual(
                learning_rates["markov_head"] / learning_rates["backbone"], 0.5
            )

    def test_legacy_single_group_checkpoint_restores_moments_and_splits_lr(
        self,
    ) -> None:
        source_model = ModelWithMarkovHead()
        legacy_params = [
            param.detach().clone().float().requires_grad_(True)
            for param in source_model.parameters()
        ]
        legacy_optimizer = torch.optim.AdamW(legacy_params, lr=1e-3)
        legacy_scheduler = CosineAnnealingWarmupLR(
            legacy_optimizer, total_steps=10, warmup_steps=2
        )
        for _ in range(8):
            for param in legacy_params:
                param.grad = torch.ones_like(param)
            legacy_optimizer.step()
            legacy_optimizer.zero_grad(set_to_none=True)
            legacy_scheduler.step()
        legacy_lr = legacy_optimizer.param_groups[0]["lr"]

        optimizer = BF16Optimizer(
            ModelWithMarkovHead(),
            lr=2e-3,
            markov_lr_multiplier=0.5,
            total_steps=10,
            warmup_ratio=0.2,
        )
        loaded = optimizer.load_state_dict(
            {
                "optimizer_state_dict": legacy_optimizer.state_dict(),
                "scheduler_state_dict": legacy_scheduler.state_dict(),
            }
        )

        self.assertTrue(loaded)
        self.assertEqual(len(optimizer.optimizer.state), len(optimizer.fp32_params))
        learning_rates = optimizer.get_learning_rates()
        self.assertAlmostEqual(learning_rates["backbone"], legacy_lr)
        self.assertAlmostEqual(learning_rates["markov_head"], legacy_lr * 0.5)
        self.assertAlmostEqual(
            optimizer.scheduler.base_lrs[1] / optimizer.scheduler.base_lrs[0], 0.5
        )
        optimizer.optimizer.step()
        optimizer.scheduler.step()
        learning_rates = optimizer.get_learning_rates()
        self.assertAlmostEqual(
            learning_rates["markov_head"] / learning_rates["backbone"], 0.5
        )

    def test_optimizer_state_saves_group_parameter_names(self) -> None:
        optimizer = BF16Optimizer(
            ModelWithMarkovHead(),
            lr=1e-3,
            markov_lr_multiplier=0.5,
            total_steps=10,
        )

        state = optimizer.state_dict()

        self.assertEqual(
            state["optimizer_param_names"],
            [
                ["backbone.weight", "backbone.bias"],
                ["markov_head.weight", "markov_head.bias"],
            ],
        )


if __name__ == "__main__":
    unittest.main()
