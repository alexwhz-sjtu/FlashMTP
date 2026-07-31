import torch

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
    ):
        # TODO: For now, we only support cosine annealing warmup lr scheduler and AdamW optimizer
        # TODO: We should make these parameters configurable
        #   These magic numbers: weight_decay=0.0, max_grad_norm=0.5, total_steps=800k, warmup_steps=12k are copied from
        #   https://github.com/SafeAILab/EAGLE/blob/main/eagle/traineagle3/ds_config.json
        self.model = model
        self.model_params = [p for p in model.parameters() if p.requires_grad]
        self.max_grad_norm = max_grad_norm
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

    def step(self):
        with torch.no_grad():
            for p, mp in zip(self.model_params, self.fp32_params):
                mp.grad = (
                    p.grad.detach().to(torch.float32) if p.grad is not None else None
                )
        torch.nn.utils.clip_grad_norm_(self.fp32_params, self.max_grad_norm)
        self.optimizer.step()
        self.optimizer.zero_grad()
        self.scheduler.step()
        with torch.no_grad():
            for p, mp in zip(self.model_params, self.fp32_params):
                p.data.copy_(mp.data.to(p.dtype))
                p.grad = None

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

    def _load_optimizer_state_partial(self, optimizer_state_dict) -> tuple[bool, int, int, int]:
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
            zip(self.fp32_params, ckpt_params, current_pids, strict=True)
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
                    state[key] = value
                    continue
                state[key] = (
                    value.reshape(param.shape).clone()
                    if value.shape != param.shape
                    else value.clone()
                )
            loaded += 1

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
