import math
from warnings import warn

from torch.optim.lr_scheduler import CosineAnnealingLR as _CosineAnnealingLR
from torch.optim.lr_scheduler import LRScheduler as _LRScheduler


class _enable_get_lr_call:
    def __init__(self, o):
        self.o = o

    def __enter__(self):
        self.o._get_lr_called_within_step = True
        return self

    def __exit__(self, type, value, traceback):
        self.o._get_lr_called_within_step = False


class TwoStageScheduler(_LRScheduler):
    def __init__(self, optimizer, after_scheduler: _LRScheduler, last_epoch=-1):
        self.after_scheduler = after_scheduler
        self.finished = False
        super().__init__(optimizer, last_epoch)

    def state_dict(self):
        state_dict = {
            key: value for key, value in self.__dict__.items() if key not in "optimizer"
        }
        if isinstance(state_dict["after_scheduler"], _LRScheduler):
            state_dict["after_scheduler_type"] = type(
                state_dict["after_scheduler"]
            ).__name__
            state_dict["after_scheduler_dict"] = state_dict[
                "after_scheduler"
            ].state_dict()
            del state_dict["after_scheduler"]
        else:
            raise NotImplementedError()
        return state_dict

    def load_state_dict(self, state_dict):
        if "after_scheduler_dict" not in state_dict:
            warn(
                "after_scheduler_dict is not found, skip loading after_scheduler. This may cause unexpected behavior."
            )
        else:
            self.after_scheduler.load_state_dict(state_dict["after_scheduler_dict"])
        state_dict = {
            key: value
            for key, value in state_dict.items()
            if key not in ("after_scheduler_type", "after_scheduler_dict")
        }
        super().load_state_dict(state_dict)


class DelayerScheduler(TwoStageScheduler):
    """Starts with a flat lr schedule until it reaches N epochs then applies
    the specific scheduler (For example: ReduceLROnPlateau)

    Args:
        optimizer (:class:`torch.optim.Optimizer`): Wrapped optimizer.
        delay_epochs (int): Number of epochs to keep the initial lr until starting applying the scheduler.
        after_scheduler (:class:`torch.optim.lr_scheduler`): After target_epoch, use this scheduler.
        last_epoch (int, optional): The index of last epoch, defaults to -1. When last_epoch=-1,
            the schedule is started from the beginning or When last_epoch=-1, sets initial lr as lr.
    """

    def __init__(self, optimizer, delay_epochs, after_scheduler, last_epoch=-1):
        if delay_epochs < 0:
            raise ValueError(f"delay_epochs must >= 0, got {delay_epochs}")
        self.delay_epochs = delay_epochs
        super().__init__(optimizer, after_scheduler, last_epoch)

    def get_lr(self):
        if self.last_epoch >= self.delay_epochs:
            if not self.finished:
                self.after_scheduler.base_lrs = self.base_lrs
                self.finished = True
            with _enable_get_lr_call(self.after_scheduler):
                return self.after_scheduler.get_lr()

        return self.base_lrs

    def step(self, epoch=None):
        if self.finished:
            if epoch is None:
                self.after_scheduler.step(None)
                self._last_lr = self.after_scheduler.get_last_lr()
            else:
                self.after_scheduler.step(epoch - self.delay_epochs)
                self._last_lr = self.after_scheduler.get_last_lr()
        else:
            return super(DelayerScheduler, self).step(epoch)


class WarmupScheduler(TwoStageScheduler):
    """Starts with a linear warmup lr schedule until it reaches N epochs then applies
    the specific scheduler (For example: ReduceLROnPlateau).

    Args:
        optimizer (:class:`torch.optim.Optimizer`): Wrapped optimizer.
        warmup_epochs (int): Number of epochs to linearly warmup lr until starting applying the scheduler.
        after_scheduler (:class:`torch.optim.lr_scheduler`): After target_epoch, use this scheduler.
        last_epoch (int, optional): The index of last epoch, defaults to -1. When last_epoch=-1,
            the schedule is started from the beginning or When last_epoch=-1, sets initial lr as lr.
    """

    def __init__(self, optimizer, warmup_epochs, after_scheduler, last_epoch=-1):
        self.warmup_epochs = int(warmup_epochs)
        super().__init__(optimizer, after_scheduler, last_epoch)

    def get_lr(self):
        if self.last_epoch >= self.warmup_epochs:
            if not self.finished:
                self.after_scheduler.base_lrs = self.base_lrs
                self.finished = True
            return self.after_scheduler.get_lr()

        return [(self.last_epoch + 1) / self.warmup_epochs * lr for lr in self.base_lrs]

    def step(self, epoch=None):
        if self.finished:
            if epoch is None:
                self.after_scheduler.step(None)
                self._last_lr = self.after_scheduler.get_last_lr()
            else:
                self.after_scheduler.step(epoch - self.warmup_epochs)
                self._last_lr = self.after_scheduler.get_last_lr()
        else:
            return super().step(epoch)


class WarmupDelayerScheduler(TwoStageScheduler):
    """Starts with a linear warmup lr schedule until it reaches N epochs and a flat lr schedule
    until it reaches M epochs then applies the specific scheduler (For example: ReduceLROnPlateau).

    Args:
        optimizer (:class:`torch.optim.Optimizer`): Wrapped optimizer.
        warmup_epochs (int): Number of epochs to linearly warmup lr until starting applying the scheduler.
        delay_epochs (int): Number of epochs to keep the initial lr until starting applying the scheduler.
        after_scheduler (:class:`torch.optim.lr_scheduler`): After target_epoch, use this scheduler.
        last_epoch (int, optional): The index of last epoch, defaults to -1. When last_epoch=-1,
            the schedule is started from the beginning or When last_epoch=-1, sets initial lr as lr.
    """

    def __init__(
        self, optimizer, warmup_epochs, delay_epochs, after_scheduler, last_epoch=-1
    ):
        if delay_epochs < 0:
            raise ValueError(f"delay_epochs must >= 0, got {delay_epochs}")
        if warmup_epochs < 0:
            raise ValueError(f"warmup_epochs must >= 0, got {warmup_epochs}")
        self.warmup_epochs = warmup_epochs
        self.delay_epochs = delay_epochs
        super().__init__(optimizer, after_scheduler, last_epoch)

    def get_lr(self):
        if self.last_epoch >= self.warmup_epochs + self.delay_epochs:
            if not self.finished:
                self.after_scheduler.base_lrs = self.base_lrs
                # reset lr to base_lr
                for group, base_lr in zip(self.optimizer.param_groups, self.base_lrs):
                    group["lr"] = base_lr
                self.finished = True
            with _enable_get_lr_call(self.after_scheduler):
                return self.after_scheduler.get_lr()
        elif self.last_epoch >= self.warmup_epochs:
            return self.base_lrs

        return [(self.last_epoch + 1) / self.warmup_epochs * lr for lr in self.base_lrs]

    def step(self, epoch=None):
        if self.finished:
            if epoch is None:
                self.after_scheduler.step(None)
                self._last_lr = self.after_scheduler.get_last_lr()
            else:
                self.after_scheduler.step(epoch - self.warmup_epochs)
                self._last_lr = self.after_scheduler.get_last_lr()
        else:
            return super().step(epoch)


class CosineAnnealingLR(_CosineAnnealingLR):
    r"""Set the learning rate of each parameter group using a cosine annealing
    schedule, where :math:`\eta_{max}` is set to the initial lr and
    :math:`T_{cur}` is the number of epochs since the last restart in SGDR:

    .. math::
        \begin{aligned}
            \eta_t & = \eta_{min} + \frac{1}{2}(\eta_{max} - \eta_{min})\left(1
            + \cos\left(\frac{T_{cur}}{T_{max}}\pi\right)\right),
            & T_{cur} \neq (2k+1)T_{max}; \\
            \eta_{t+1} & = \eta_{t} + \frac{1}{2}(\eta_{max} - \eta_{min})
            \left(1 - \cos\left(\frac{1}{T_{max}}\pi\right)\right),
            & T_{cur} = (2k+1)T_{max}.
        \end{aligned}

    When last_epoch=-1, sets initial lr as lr. Notice that because the schedule
    is defined recursively, the learning rate can be simultaneously modified
    outside this scheduler by other operators. If the learning rate is set
    solely by this scheduler, the learning rate at each step becomes:

    .. math::
        \eta_t = \eta_{min} + \frac{1}{2}(\eta_{max} - \eta_{min})\left(1 +
        \cos\left(\frac{T_{cur}}{T_{max}}\pi\right)\right)

    It has been proposed in
    `SGDR: Stochastic Gradient Descent with Warm Restarts`_. Note that this only
    implements the cosine annealing part of SGDR, and not the restarts.

    .. _SGDR\: Stochastic Gradient Descent with Warm Restarts:
        https://arxiv.org/abs/1608.03983

    Args:
        optimizer (:class:`torch.optim.Optimizer`): Wrapped optimizer.
        total_steps (int): Number of total training steps.
        eta_min (int, optional): Minimum learning rate, defaults to 0.
        last_epoch (int, optional): The index of last epoch, defaults to -1. When last_epoch=-1,
            the schedule is started from the beginning or When last_epoch=-1, sets initial lr as lr.
    """

    def __init__(
        self,
        optimizer,
        total_steps: int,
        eta_min: int = 0,
        last_epoch: int = -1,
        **kwargs,
    ):
        super().__init__(optimizer, total_steps, eta_min=eta_min, last_epoch=last_epoch)


class CosineAnnealingWarmupLR(WarmupScheduler):
    """Cosine annealing learning rate scheduler with learning rate warmup. A linear warmup schedule will be applied.

    Args:
        optimizer (:class:`torch.optim.Optimizer`): Wrapped optimizer.
        total_steps (int): Number of total training steps.
        warmup_steps (int, optional): Number of warmup steps, defaults to 0.
        eta_min (int, optional): Minimum learning rate, defaults to 0.
        last_epoch (int, optional): The index of last epoch, defaults to -1. When last_epoch=-1,
            the schedule is started from the beginning or When last_epoch=-1, sets initial lr as lr.
    """

    def __init__(
        self,
        optimizer,
        total_steps: int,
        warmup_steps: int = 0,
        eta_min: float = 0.0,
        last_epoch: int = -1,
    ):
        base_scheduler = _CosineAnnealingLR(
            optimizer,
            total_steps - warmup_steps,
            eta_min=eta_min,
            last_epoch=last_epoch,
        )
        super().__init__(optimizer, warmup_steps, base_scheduler, last_epoch=last_epoch)


class ThreeStageDistillScheduler(_LRScheduler):
    """Three-stage learning rate scheduler for DFlash distillation training with optional warmup.

    Stage 0 (Warmup, optional): Linear warmup from 0 to initial_lr
    Stage 1 (Distill): Full cosine-annealing half-wave (phase 0 -> pi),
        from initial_lr to transition_lr_ratio * initial_lr
    Stage 2 (Transition): Constant lr at the junction of the two cosine halves
    Stage 3 (CE): Full cosine-annealing half-wave (phase 0 -> pi),
        from transition_lr_ratio * initial_lr down to eta_min

    Supports both absolute steps and relative ratios for flexible configuration.
    If stage steps are not provided, uses automatic allocation based on ratios.

    Args:
        optimizer (:class:`torch.optim.Optimizer`): Wrapped optimizer.
        total_steps (int): Number of total training steps.
        warmup_steps (int, optional): Number of warmup steps. If None, uses warmup_ratio.
        warmup_ratio (float): Ratio of total_steps for warmup (default: 0.04).
        distill_steps (int, optional): Number of steps for stage 1. If None, uses distill_ratio.
        transition_steps (int, optional): Number of steps for stage 2. If None, uses transition_ratio.
        distill_ratio (float): Ratio of remaining steps (after warmup) for stage 1 (default: 0.3).
        transition_ratio (float): Ratio of remaining steps for stage 2 (default: 0.2).
        distill_end_lr_ratio (float): Deprecated compatibility argument. The
            distill stage now ends at transition_lr_ratio.
        transition_lr_ratio (float): Constant lr ratio for stage 2 (default: 0.2).
        ce_start_lr_ratio (float): Deprecated compatibility argument. The CE
            stage now starts at transition_lr_ratio.
        eta_min_ratio (float): Minimum lr ratio for cosine annealing (default: 0.0).
        last_epoch (int, optional): The index of last epoch, defaults to -1.
    """

    def __init__(
        self,
        optimizer,
        total_steps: int,
        warmup_steps: int = None,
        warmup_ratio: float = 0.04,
        distill_steps: int = None,
        transition_steps: int = None,
        distill_ratio: float = 0.3,
        transition_ratio: float = 0.2,
        distill_end_lr_ratio: float = 0.9,
        transition_lr_ratio: float = 0.2,
        ce_start_lr_ratio: float = 0.9,
        eta_min_ratio: float = 0.0,
        last_epoch: int = -1,
    ):
        # Calculate warmup steps
        if warmup_steps is None:
            warmup_steps = int(total_steps * warmup_ratio)
        self.warmup_steps = max(0, warmup_steps)

        # Remaining steps after warmup
        remaining_steps = total_steps - self.warmup_steps

        # Auto-calculate steps from ratios if not explicitly provided
        if distill_steps is None:
            distill_steps = int(remaining_steps * distill_ratio)
        if transition_steps is None:
            transition_steps = int(remaining_steps * transition_ratio)

        if distill_steps < 0:
            raise ValueError(f"distill_steps must >= 0, got {distill_steps}")
        if transition_steps < 0:
            raise ValueError(f"transition_steps must >= 0, got {transition_steps}")

        self.distill_steps = distill_steps
        self.transition_steps = transition_steps
        self.ce_steps = remaining_steps - distill_steps - transition_steps

        # Ensure ce_steps is non-negative
        if self.ce_steps < 0:
            # Auto-adjust: prioritize distill, then transition, rest for ce
            self.distill_steps = int(remaining_steps * 0.3)
            self.transition_steps = int(remaining_steps * 0.2)
            self.ce_steps = remaining_steps - self.distill_steps - self.transition_steps
            import warnings
            warnings.warn(
                f"Steps exceed remaining_steps, auto-adjusted to: "
                f"distill={self.distill_steps}, transition={self.transition_steps}, ce={self.ce_steps}"
            )

        self.distill_end_lr_ratio = distill_end_lr_ratio
        self.transition_lr_ratio = transition_lr_ratio
        self.ce_start_lr_ratio = ce_start_lr_ratio
        self.eta_min_ratio = eta_min_ratio
        self.total_steps = total_steps

        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        current_step = self.last_epoch

        # Stage 0: Warmup - linear increase from 0 to base_lr
        if current_step < self.warmup_steps:
            if self.warmup_steps == 0:
                return list(self.base_lrs)
            progress = current_step / self.warmup_steps
            return [base_lr * progress for base_lr in self.base_lrs]

        # Adjust step for post-warmup stages
        step_after_warmup = current_step - self.warmup_steps

        # Stage 1: full cosine-annealing half-wave (phase 0 -> pi).  Its slope
        # is zero at both ends, so it joins warmup/transition smoothly.
        if step_after_warmup < self.distill_steps:
            if self.distill_steps == 0:
                ratio = self.transition_lr_ratio
            else:
                progress = step_after_warmup / self.distill_steps
                cosine_factor = 0.5 * (1.0 + math.cos(math.pi * progress))
                ratio = self.transition_lr_ratio + (1.0 - self.transition_lr_ratio) * cosine_factor
            return [base_lr * ratio for base_lr in self.base_lrs]

        # Stage 2: Transition - constant at transition_lr_ratio
        elif step_after_warmup < self.distill_steps + self.transition_steps:
            return [base_lr * self.transition_lr_ratio for base_lr in self.base_lrs]

        # Stage 3: full cosine-annealing half-wave (phase 0 -> pi).  Its slope
        # is zero at both ends, so it joins transition/eta_min smoothly.
        else:
            ce_step = step_after_warmup - self.distill_steps - self.transition_steps
            if self.ce_steps > 0:
                progress = ce_step / self.ce_steps
                cosine_factor = 0.5 * (1.0 + math.cos(math.pi * progress))
                ratio = self.eta_min_ratio + (self.transition_lr_ratio - self.eta_min_ratio) * cosine_factor
            else:
                ratio = self.eta_min_ratio
            return [base_lr * ratio for base_lr in self.base_lrs]

    def step(self, epoch=None):
        if epoch is None:
            self.last_epoch += 1
        else:
            self.last_epoch = epoch
        self._last_lr = self.get_lr()
        for param_group, lr in zip(self.optimizer.param_groups, self._last_lr):
            param_group["lr"] = lr
