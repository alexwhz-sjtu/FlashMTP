import copy

import torch
import torch.distributed as dist

from specforge.lr_scheduler import CosineAnnealingWarmupLR
from specforge.utils import print_on_rank0


class BF16Optimizer:
    def __init__(
        self,
        model,
        lr,
        weight_decay=0.0,
        max_grad_norm=0.5,
        total_steps=800_000,
        warmup_ratio=0.015,
        markov_lr_multiplier=1.0,
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
        self.param_names = [name for name, _ in named_model_params]
        self.model_params = [param for _, param in named_model_params]
        self.markov_lr_multiplier = float(markov_lr_multiplier)
        if self.markov_lr_multiplier <= 0:
            raise ValueError(
                "markov_lr_multiplier must be positive, got "
                f"{markov_lr_multiplier}."
            )
        self.max_grad_norm = float(max_grad_norm)
        if self.max_grad_norm <= 0:
            raise ValueError(f"max_grad_norm must be positive, got {max_grad_norm}.")
        self.fp32_params = [
            p.detach().clone().to(torch.float32) for p in self.model_params
        ]
        for mp in self.fp32_params:
            mp.requires_grad = True

        backbone_indices = [
            idx
            for idx, name in enumerate(self.param_names)
            if not self._is_markov_param(name)
        ]
        markov_indices = [
            idx
            for idx, name in enumerate(self.param_names)
            if self._is_markov_param(name)
        ]
        self._group_param_indices = []
        self._group_names = []
        optimizer_groups = []
        if backbone_indices:
            self._append_optimizer_group(
                optimizer_groups,
                group_name="backbone",
                param_indices=backbone_indices,
                lr=lr,
            )
        if markov_indices:
            self._append_optimizer_group(
                optimizer_groups,
                group_name="markov_head",
                param_indices=markov_indices,
                lr=lr * self.markov_lr_multiplier,
            )
        self.optimizer = torch.optim.AdamW(
            optimizer_groups, weight_decay=weight_decay
        )
        self.scheduler = CosineAnnealingWarmupLR(
            self.optimizer,
            total_steps=total_steps,
            warmup_steps=int(warmup_ratio * total_steps),
        )

    @staticmethod
    def _is_markov_param(name: str) -> bool:
        return name == "markov_head" or name.startswith("markov_head.")

    def _append_optimizer_group(
        self, optimizer_groups, group_name: str, param_indices, lr: float
    ) -> None:
        indices = list(param_indices)
        self._group_param_indices.append(indices)
        self._group_names.append(group_name)
        optimizer_groups.append(
            {
                "params": [self.fp32_params[idx] for idx in indices],
                "lr": lr,
                "group_name": group_name,
            }
        )

    def _optimizer_param_names(self) -> list[list[str]]:
        return [
            [self.param_names[idx] for idx in indices]
            for indices in self._group_param_indices
        ]

    def zero_grad(self) -> None:
        """Clear both FSDP model gradients and FP32 optimizer gradients."""
        self.optimizer.zero_grad(set_to_none=True)
        for param in self.model_params:
            param.grad = None
        for param in self.fp32_params:
            param.grad = None

    def scale_model_gradients(self, factor: float) -> None:
        """Scale accumulated FSDP gradients before they are copied to FP32."""
        if factor <= 0:
            raise ValueError(f"gradient scale must be positive, got {factor}.")
        with torch.no_grad():
            for param in self.model_params:
                if param.grad is not None:
                    param.grad.mul_(factor)

    def _global_grad_norm(self) -> torch.Tensor:
        """Return the global L2 norm using one scalar all-reduce.

        Gradients remain sharded and local. Only their local squared L2 norm is
        summed across ranks.
        """
        if not self.fp32_params:
            return torch.tensor(0.0)

        device = self.fp32_params[0].device
        gradients = [param.grad for param in self.fp32_params if param.grad is not None]
        if gradients:
            local_norms = torch._foreach_norm(gradients, 2)
            local_squared_norm = (
                torch.stack([norm.double() for norm in local_norms]).square().sum()
            )
        else:
            local_squared_norm = torch.zeros((), device=device, dtype=torch.float64)

        if dist.is_available() and dist.is_initialized():
            # Communication payload: exactly one 0-D scalar, never gradients.
            dist.all_reduce(local_squared_norm, op=dist.ReduceOp.SUM)
        return local_squared_norm.sqrt()

    def step(self) -> float:
        with torch.no_grad():
            for p, mp in zip(self.model_params, self.fp32_params):
                mp.grad = (
                    p.grad.detach().to(torch.float32) if p.grad is not None else None
                )

        grad_norm = self._global_grad_norm()
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
        return grad_norm_value

    def _param_state_compatible(self, param, param_state) -> bool:
        for key, value in param_state.items():
            if key == "step" or not isinstance(value, torch.Tensor):
                continue
            if value.numel() != param.numel():
                return False
        return True

    def _optimizer_state_compatible(
        self, optimizer_state_dict, saved_param_names=None
    ) -> bool:
        """Return True when every saved Adam moment matches current fp32 param shapes."""
        ckpt_groups = optimizer_state_dict["param_groups"]
        if len(ckpt_groups) != len(self.optimizer.param_groups):
            print_on_rank0(
                "Optimizer checkpoint has "
                f"{len(ckpt_groups)} parameter group(s), expected "
                f"{len(self.optimizer.param_groups)}."
            )
            return False
        if (
            saved_param_names is not None
            and saved_param_names != self._optimizer_param_names()
        ):
            print_on_rank0(
                "Optimizer checkpoint parameter names/order differ from the "
                "current model. Loading compatible entries by name."
            )
            return False

        for group_idx, (indices, ckpt_group) in enumerate(
            zip(self._group_param_indices, ckpt_groups, strict=True)
        ):
            ckpt_params = ckpt_group["params"]
            if len(ckpt_params) != len(indices):
                print_on_rank0(
                    f"Optimizer group {group_idx} has {len(ckpt_params)} params, "
                    f"expected {len(indices)}."
                )
                return False
            for param_idx, param_id in zip(indices, ckpt_params, strict=True):
                param = self.fp32_params[param_idx]
                param_state = optimizer_state_dict["state"].get(param_id)
                if param_state is None:
                    continue
                if not self._param_state_compatible(param, param_state):
                    print_on_rank0(
                        "Optimizer state shape mismatch for "
                        f"{self.param_names[param_idx]}: checkpoint moments do "
                        f"not match model {tuple(param.shape)} "
                        f"({param.numel()} elements)."
                    )
                    return False
        return True

    def _saved_params_by_name(
        self, optimizer_state_dict, saved_param_names=None
    ) -> dict[str, int]:
        """Map saved optimizer parameter ids to model names when possible."""
        ckpt_groups = optimizer_state_dict["param_groups"]
        if saved_param_names is not None and len(saved_param_names) == len(ckpt_groups):
            mapping = {}
            valid = True
            for names, group in zip(saved_param_names, ckpt_groups, strict=True):
                if len(names) != len(group["params"]):
                    valid = False
                    break
                mapping.update(zip(names, group["params"], strict=True))
            if valid:
                return mapping

        # Checkpoints made before parameter groups were introduced have one
        # flat group in model.named_parameters() order.
        if len(ckpt_groups) == 1:
            return dict(zip(self.param_names, ckpt_groups[0]["params"]))

        # Best effort for checkpoints that have groups but predate saved names.
        mapping = {}
        for names, group in zip(
            self._optimizer_param_names(), ckpt_groups
        ):
            mapping.update(zip(names, group["params"]))
        return mapping

    def _copy_group_hyperparameters(self, optimizer_state_dict) -> None:
        ckpt_groups = optimizer_state_dict["param_groups"]
        saved_by_name = {
            group.get("group_name"): group
            for group in ckpt_groups
            if group.get("group_name") is not None
        }
        legacy_group = ckpt_groups[0] if len(ckpt_groups) == 1 else None
        for group_idx, current_group in enumerate(self.optimizer.param_groups):
            source_group = saved_by_name.get(current_group.get("group_name"))
            if source_group is None:
                source_group = legacy_group or ckpt_groups[
                    min(group_idx, len(ckpt_groups) - 1)
                ]
            for key in ("betas", "eps", "weight_decay"):
                if key in source_group:
                    current_group[key] = source_group[key]

            if "lr" in source_group:
                current_group["lr"] = source_group["lr"]

        self._enforce_markov_lr_ratio()

    def _enforce_markov_lr_ratio(self) -> None:
        groups = {
            group.get("group_name"): group for group in self.optimizer.param_groups
        }
        if "backbone" in groups and "markov_head" in groups:
            groups["markov_head"]["lr"] = (
                groups["backbone"]["lr"] * self.markov_lr_multiplier
            )

    def _restore_group_names(self) -> None:
        for group, group_name in zip(
            self.optimizer.param_groups, self._group_names, strict=True
        ):
            group["group_name"] = group_name

    def _scheduler_state_for_current_groups(self, scheduler_state_dict):
        """Resize/rebase saved LR lists and preserve the configured head ratio."""
        migrated = copy.deepcopy(scheduler_state_dict)
        group_count = len(self.optimizer.param_groups)
        group_names = [
            group.get("group_name") for group in self.optimizer.param_groups
        ]

        def migrate(value):
            if isinstance(value, dict):
                for key, item in value.items():
                    if key in ("base_lrs", "_last_lr") and isinstance(item, list):
                        if not item:
                            continue
                        if len(item) == group_count:
                            new_lrs = list(item)
                        elif len(item) == 1:
                            new_lrs = [item[0]] * group_count
                        else:
                            new_lrs = list(item[:group_count])
                            new_lrs.extend([item[0]] * (group_count - len(new_lrs)))
                        if "backbone" in group_names and "markov_head" in group_names:
                            backbone_idx = group_names.index("backbone")
                            markov_idx = group_names.index("markov_head")
                            new_lrs[markov_idx] = (
                                new_lrs[backbone_idx] * self.markov_lr_multiplier
                            )
                        value[key] = new_lrs
                    else:
                        migrate(item)
            return value

        return migrate(migrated)

    def _load_optimizer_state_partial(
        self, optimizer_state_dict, saved_param_names=None
    ) -> tuple[bool, int, int, int]:
        """Load compatible Adam moments; skip missing or shape-mismatched params."""
        ckpt_opt = optimizer_state_dict
        saved_params = self._saved_params_by_name(ckpt_opt, saved_param_names)

        if len(saved_params) != len(self.fp32_params):
            print_on_rank0(
                "Optimizer checkpoint param count "
                f"({len(saved_params)}) differs from model "
                f"({len(self.fp32_params)}); loading compatible entries only."
            )

        loaded = 0
        skipped = 0
        missing = 0
        for idx, (name, param) in enumerate(zip(self.param_names, self.fp32_params)):
            ckpt_pid = saved_params.get(name)
            if ckpt_pid is None:
                missing += 1
                continue
            param_state = ckpt_opt["state"].get(ckpt_pid)
            if param_state is None:
                missing += 1
                continue
            if not self._param_state_compatible(param, param_state):
                print_on_rank0(
                    "Skipping optimizer state for param index "
                    f"{name} (index {idx}): incompatible moment shapes."
                )
                skipped += 1
                continue

            state = self.optimizer.state.setdefault(param, {})
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

        self._copy_group_hyperparameters(ckpt_opt)

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

            saved_param_names = state_dict.get("optimizer_param_names")
            if self._optimizer_state_compatible(ckpt_opt, saved_param_names):
                self.optimizer.load_state_dict(ckpt_opt)
                self._restore_group_names()
                self._enforce_markov_lr_ratio()
                print_on_rank0("Successfully loaded optimizer state_dict.")
                loaded_optimizer = True
            else:
                any_loaded, loaded, skipped, missing = (
                    self._load_optimizer_state_partial(
                        ckpt_opt, saved_param_names
                    )
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
        scheduler_state = self._scheduler_state_for_current_groups(
            state_dict["scheduler_state_dict"]
        )
        self.scheduler.load_state_dict(scheduler_state)
        # Loading a scheduler does not itself update optimizer.param_groups.
        # This is needed for --no-resume-optimizer and harmless for a full load.
        if not loaded_optimizer:
            last_lrs = scheduler_state.get("_last_lr")
            if last_lrs is not None:
                for group, group_lr in zip(
                    self.optimizer.param_groups, last_lrs, strict=True
                ):
                    group["lr"] = group_lr
        print_on_rank0("Successfully loaded scheduler state_dict.")
        return loaded_optimizer

    def state_dict(self):
        return {
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "optimizer_param_names": self._optimizer_param_names(),
            "markov_lr_multiplier": self.markov_lr_multiplier,
        }

    def get_learning_rate(self):
        learning_rates = self.get_learning_rates()
        return learning_rates.get("backbone", next(iter(learning_rates.values())))

    def get_learning_rates(self) -> dict[str, float]:
        return {
            group.get("group_name", f"group_{idx}"): group["lr"]
            for idx, group in enumerate(self.optimizer.param_groups)
        }
