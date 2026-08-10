import logging
import math
from dataclasses import dataclass

import torch
import torch.distributed as dist

from specforge.lr_scheduler import CosineAnnealingWarmupLR
from specforge.utils import print_on_rank0

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class OptimizerStepResult:
    """Result of one guarded optimizer update."""

    updated: bool
    grad_norm: float
    reason: str | None = None


class BF16Optimizer:
    def __init__(
        self,
        model,
        lr,
        weight_decay=0.0,
        max_grad_norm=0.5,
        total_steps=800_000,
        warmup_ratio=0.015,
    ):
        # TODO: For now, we only support cosine annealing warmup lr scheduler and AdamW optimizer
        # TODO: We should make these parameters configurable
        #   These magic numbers: weight_decay=0.0, max_grad_norm=0.5, total_steps=800k, warmup_steps=12k are copied from
        #   https://github.com/SafeAILab/EAGLE/blob/main/eagle/traineagle3/ds_config.json
        self.model = model
        named_model_params = [
            (name, param)
            for name, param in model.named_parameters()
            if param.requires_grad
        ]
        self.model_param_names = [name for name, _ in named_model_params]
        self.model_params = [param for _, param in named_model_params]
        self.max_grad_norm = float(max_grad_norm)
        if not math.isfinite(self.max_grad_norm) or self.max_grad_norm <= 0:
            raise ValueError(
                f"max_grad_norm must be finite and positive, got {max_grad_norm}."
            )
        self.fp32_params = [
            p.detach().clone().to(torch.float32) for p in self.model_params
        ]
        for mp in self.fp32_params:
            mp.requires_grad = True
        self.optimizer = torch.optim.AdamW(
            self.fp32_params, lr=lr, weight_decay=weight_decay
        )
        self.scheduler = CosineAnnealingWarmupLR(
            self.optimizer,
            total_steps=total_steps,
            warmup_steps=int(warmup_ratio * total_steps),
        )

    def zero_grad(self) -> None:
        """Clear both FSDP model gradients and FP32 optimizer gradients."""
        self.optimizer.zero_grad(set_to_none=True)
        for param in self.model_params:
            param.grad = None
        for param in self.fp32_params:
            param.grad = None

    def scale_model_gradients(self, factor: float) -> None:
        """Scale accumulated FSDP gradients before they are copied to FP32."""
        if not math.isfinite(factor) or factor <= 0:
            raise ValueError(
                f"gradient scale must be finite and positive, got {factor}."
            )
        with torch.no_grad():
            for param in self.model_params:
                if param.grad is not None:
                    param.grad.mul_(factor)

    def _global_grad_norm(self) -> tuple[torch.Tensor, bool]:
        """Return the global L2 norm using one scalar all-reduce.

        Gradients remain sharded and local. Only their local squared L2 norm is
        summed across ranks; NaN/Inf gradients naturally make that scalar
        non-finite on every rank after the reduction.
        """
        if not self.fp32_params:
            return torch.tensor(0.0), False

        device = self.fp32_params[0].device
        gradients = [param.grad for param in self.fp32_params if param.grad is not None]
        if gradients:
            local_norms = torch._foreach_norm(gradients, 2)
            local_squared_norm = (
                torch.stack([norm.double() for norm in local_norms]).square().sum()
            )
        else:
            # Propagate the unexpected missing-gradient condition through the
            # same scalar reduction so every rank makes the same skip decision.
            local_squared_norm = torch.full(
                (), float("nan"), device=device, dtype=torch.float64
            )

        if dist.is_available() and dist.is_initialized():
            # Communication payload: exactly one 0-D scalar, never gradients.
            dist.all_reduce(local_squared_norm, op=dist.ReduceOp.SUM)
        global_norm = local_squared_norm.sqrt()
        return global_norm, bool(torch.isfinite(global_norm).item())

    def _optimizer_state_is_finite(self) -> bool:
        """Check loaded Adam moments consistently across distributed ranks."""
        if not self.fp32_params:
            return False
        device = self.fp32_params[0].device
        local_bad = torch.zeros((), device=device, dtype=torch.int32)
        for param_state in self.optimizer.state.values():
            for value in param_state.values():
                if isinstance(value, torch.Tensor) and not torch.isfinite(value).all():
                    local_bad.fill_(1)
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(local_bad, op=dist.ReduceOp.MAX)
        return local_bad.item() == 0

    def _nonfinite_gradient_reason(self, grad_norm: torch.Tensor) -> str:
        """Describe a rejected update before gradients are cleared.

        This path only runs after the scalar global norm is non-finite, so the
        additional per-gradient scans do not affect normal-step performance.
        """
        missing_names: list[str] = []
        nonfinite_names: list[str] = []
        nonfinite_elements = 0
        gradient_elements = 0
        for name, param in zip(self.model_param_names, self.model_params, strict=True):
            grad = param.grad
            if grad is None:
                missing_names.append(name)
                continue
            gradient_elements += grad.numel()
            finite_mask = torch.isfinite(grad)
            if not bool(finite_mask.all().item()):
                nonfinite_names.append(name)
                nonfinite_elements += int((~finite_mask).sum().item())

        def summarize(names: list[str], limit: int = 8) -> str:
            shown = names[:limit]
            suffix = f", ... (+{len(names) - limit})" if len(names) > limit else ""
            return ", ".join(shown) + suffix

        if nonfinite_names:
            return (
                "nonfinite_gradients: "
                f"params=[{summarize(nonfinite_names)}], "
                f"nonfinite_elements={nonfinite_elements}/{gradient_elements}, "
                f"missing_grad_params={len(missing_names)}"
            )
        if gradient_elements == 0:
            return (
                "missing_gradients: no trainable parameter received a gradient; "
                f"missing_params=[{summarize(missing_names)}]"
            )
        return (
            "nonfinite_global_grad_norm_with_finite_elements: "
            f"global_norm={float(grad_norm.item())}, "
            f"gradient_elements={gradient_elements}, "
            f"missing_grad_params={len(missing_names)}"
        )

    def step(self) -> OptimizerStepResult:
        with torch.no_grad():
            for p, mp in zip(self.model_params, self.fp32_params):
                mp.grad = (
                    p.grad.detach().to(torch.float32) if p.grad is not None else None
                )

        grad_norm, finite = self._global_grad_norm()
        if not finite:
            reason = self._nonfinite_gradient_reason(grad_norm)
            rank = dist.get_rank() if dist.is_initialized() else 0
            logger.warning(
                "rank %s: skipped optimizer update: %s",
                rank,
                reason,
            )
            self.zero_grad()
            return OptimizerStepResult(
                updated=False,
                grad_norm=float("nan"),
                reason=reason,
            )

        grad_norm_value = float(grad_norm.item())
        clip_coefficient = min(
            self.max_grad_norm / (grad_norm_value + 1e-6),
            1.0,
        )
        if clip_coefficient < 1.0:
            with torch.no_grad():
                for param in self.fp32_params:
                    if param.grad is not None:
                        param.grad.mul_(clip_coefficient)

        self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)
        self.scheduler.step()
        with torch.no_grad():
            for p, mp in zip(self.model_params, self.fp32_params):
                p.data.copy_(mp.data.to(p.dtype))
                p.grad = None
        return OptimizerStepResult(updated=True, grad_norm=grad_norm_value)

    def _param_state_compatible(self, param, param_state) -> bool:
        for key, value in param_state.items():
            if key == "step" or not isinstance(value, torch.Tensor):
                continue
            if value.numel() != param.numel():
                return False
        return True

    def _optimizer_state_compatible(self, optimizer_state_dict) -> bool:
        """Return True when every saved Adam moment matches current fp32 param shapes."""
        ckpt_params = optimizer_state_dict["param_groups"][0]["params"]
        if len(ckpt_params) != len(self.fp32_params):
            print_on_rank0(
                "Optimizer checkpoint has "
                f"{len(ckpt_params)} params, expected {len(self.fp32_params)}."
            )
            return False

        for idx, (param, param_id) in enumerate(
            zip(self.fp32_params, ckpt_params, strict=True)
        ):
            param_state = optimizer_state_dict["state"].get(param_id)
            if param_state is None:
                continue
            if not self._param_state_compatible(param, param_state):
                print_on_rank0(
                    "Optimizer state shape mismatch at param index "
                    f"{idx}: checkpoint moments do not match model "
                    f"{tuple(param.shape)} ({param.numel()} elements)."
                )
                return False
        return True

    def _load_optimizer_state_partial(
        self, optimizer_state_dict
    ) -> tuple[bool, int, int, int]:
        """Load compatible Adam moments; skip missing or shape-mismatched params."""
        ckpt_opt = optimizer_state_dict
        ckpt_params = ckpt_opt["param_groups"][0]["params"]
        current_pids = self.optimizer.param_groups[0]["params"]

        if len(ckpt_params) != len(self.fp32_params):
            print_on_rank0(
                "Optimizer checkpoint param count "
                f"({len(ckpt_params)}) differs from model "
                f"({len(self.fp32_params)}); loading compatible entries only."
            )

        loaded = 0
        skipped = 0
        missing = 0
        for idx, (param, ckpt_pid, cur_pid) in enumerate(
            zip(self.fp32_params, ckpt_params, current_pids)
        ):
            param_state = ckpt_opt["state"].get(ckpt_pid)
            if param_state is None:
                missing += 1
                continue
            if not self._param_state_compatible(param, param_state):
                print_on_rank0(
                    "Skipping optimizer state for param index "
                    f"{idx}: incompatible moment shapes."
                )
                skipped += 1
                continue

            state = self.optimizer.state.setdefault(cur_pid, {})
            for key, value in param_state.items():
                if key == "step" or not isinstance(value, torch.Tensor):
                    state[key] = (
                        value.clone() if isinstance(value, torch.Tensor) else value
                    )
                    continue
                reshaped_value = (
                    value.reshape(param.shape).clone()
                    if value.shape != param.shape
                    else value.clone()
                )
                state[key] = reshaped_value.to(
                    device=param.device,
                    dtype=param.dtype,
                )
            loaded += 1

        # ``zip`` stops at the shorter list. Parameters not represented by the
        # checkpoint must be counted as missing instead of raising from
        # ``strict=True`` while attempting a best-effort legacy restore.
        missing += max(0, len(self.fp32_params) - len(ckpt_params))

        ckpt_pg = ckpt_opt["param_groups"][0]
        for key in ("lr", "betas", "eps", "weight_decay"):
            if key in ckpt_pg:
                self.optimizer.param_groups[0][key] = ckpt_pg[key]

        return loaded > 0, loaded, skipped, missing

    def load_state_dict(self, state_dict, load_optimizer: bool = True) -> bool:
        loaded_optimizer = False
        if load_optimizer:
            ckpt_opt = state_dict["optimizer_state_dict"]
            ckpt_state_count = len(ckpt_opt["state"])
            if ckpt_state_count < len(self.fp32_params):
                print_on_rank0(
                    "Optimizer checkpoint only has Adam moments for "
                    f"{ckpt_state_count}/{len(self.fp32_params)} parameters."
                )

            if self._optimizer_state_compatible(ckpt_opt):
                self.optimizer.load_state_dict(ckpt_opt)
                print_on_rank0("Successfully loaded optimizer state_dict.")
                loaded_optimizer = True
            else:
                any_loaded, loaded, skipped, missing = (
                    self._load_optimizer_state_partial(ckpt_opt)
                )
                if any_loaded:
                    print_on_rank0(
                        "Partially restored optimizer state: "
                        f"{loaded} loaded, {skipped} skipped (incompatible), "
                        f"{missing} missing from checkpoint."
                    )
                    loaded_optimizer = True
                else:
                    print_on_rank0(
                        "Could not restore optimizer state; Adam moments will be "
                        "reinitialized while scheduler state is still restored."
                    )
            if loaded_optimizer and not self._optimizer_state_is_finite():
                self.optimizer.state.clear()
                loaded_optimizer = False
                logger.warning(
                    "Loaded optimizer state contains NaN or Inf; discarded all "
                    "Adam moments to protect model weights."
                )
        self.scheduler.load_state_dict(state_dict["scheduler_state_dict"])
        print_on_rank0("Successfully loaded scheduler state_dict.")
        return loaded_optimizer

    def state_dict(self):
        return {
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
        }

    def get_learning_rate(self):
        return self.optimizer.param_groups[0]["lr"]
