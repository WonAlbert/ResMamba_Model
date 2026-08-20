from __future__ import annotations

import math
from typing import Any

import torch


def scale_lr_for_grad_accum(base_lr: float, gradient_accumulation_steps: int, *, enabled: bool) -> float:
    """按线性缩放规则：peak_lr = base_lr × gradient_accumulation_steps。"""
    if not enabled or gradient_accumulation_steps <= 1:
        return float(base_lr)
    return float(base_lr) * gradient_accumulation_steps


class LinearWarmupLR:
    """优化器步级别的线性 warmup，warmup 结束后保持 peak_lr。"""

    def __init__(self, optimizer: torch.optim.Optimizer, *, peak_lr: float, warmup_steps: int):
        self.optimizer = optimizer
        self.peak_lr = float(peak_lr)
        self.warmup_steps = max(0, int(warmup_steps))
        self.completed_steps = 0
        self._set_lr(self._lr_for_optimizer_step(0))

    def _set_lr(self, lr: float) -> None:
        for group in self.optimizer.param_groups:
            group["lr"] = lr * float(group.get("lr_scale", 1.0))

    def _lr_for_optimizer_step(self, step: int) -> float:
        if self.warmup_steps <= 0 or step >= self.warmup_steps:
            return self.peak_lr
        return self.peak_lr * (step + 1) / self.warmup_steps

    def step(self) -> float:
        lr_used = float(self.optimizer.param_groups[0]["lr"])
        self.completed_steps += 1
        self._set_lr(self._lr_for_optimizer_step(self.completed_steps))
        return lr_used

    def state_dict(self) -> dict[str, float | int]:
        return {
            "peak_lr": self.peak_lr,
            "warmup_steps": self.warmup_steps,
            "completed_steps": self.completed_steps,
        }

    def load_state_dict(self, state: dict[str, float | int]) -> None:
        self.completed_steps = int(state.get("completed_steps", 0))
        self._set_lr(self._lr_for_optimizer_step(self.completed_steps))

    def reset_progress(self) -> None:
        self.completed_steps = 0
        self._set_lr(self._lr_for_optimizer_step(0))


class WarmupCosineLR:
    """线性 warmup + cosine 衰减至 min_lr。"""

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        *,
        peak_lr: float,
        warmup_steps: int,
        total_steps: int,
        min_lr_ratio: float = 0.1,
    ) -> None:
        self.optimizer = optimizer
        self.peak_lr = float(peak_lr)
        self.warmup_steps = max(0, int(warmup_steps))
        self.total_steps = max(int(total_steps), self.warmup_steps + 1)
        self.min_lr = self.peak_lr * float(min_lr_ratio)
        self.completed_steps = 0
        self._set_lr(self._lr_at_step(0))

    def _set_lr(self, lr: float) -> None:
        for group in self.optimizer.param_groups:
            group["lr"] = lr * float(group.get("lr_scale", 1.0))

    def _lr_at_step(self, step: int) -> float:
        if self.warmup_steps > 0 and step < self.warmup_steps:
            return self.peak_lr * (step + 1) / self.warmup_steps
        if step >= self.total_steps:
            return self.min_lr
        progress = (step - self.warmup_steps) / max(1, self.total_steps - self.warmup_steps)
        progress = min(max(progress, 0.0), 1.0)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return self.min_lr + (self.peak_lr - self.min_lr) * cosine

    def step(self) -> float:
        lr_used = float(self.optimizer.param_groups[0]["lr"])
        self.completed_steps += 1
        if self.completed_steps < self.total_steps:
            self._set_lr(self._lr_at_step(self.completed_steps))
        else:
            self._set_lr(self.min_lr)
        return lr_used

    def state_dict(self) -> dict[str, float | int]:
        return {
            "peak_lr": self.peak_lr,
            "warmup_steps": self.warmup_steps,
            "total_steps": self.total_steps,
            "min_lr": self.min_lr,
            "completed_steps": self.completed_steps,
        }

    def load_state_dict(self, state: dict[str, float | int]) -> None:
        self.completed_steps = int(state.get("completed_steps", 0))
        step = min(self.completed_steps, max(0, self.total_steps - 1))
        self._set_lr(self._lr_at_step(step))

    def reconfigure(
        self,
        *,
        peak_lr: float | None = None,
        warmup_steps: int | None = None,
        total_steps: int | None = None,
        min_lr_ratio: float | None = None,
    ) -> None:
        if peak_lr is not None:
            self.peak_lr = float(peak_lr)
        if warmup_steps is not None:
            self.warmup_steps = max(0, int(warmup_steps))
        if total_steps is not None:
            self.total_steps = max(int(total_steps), self.warmup_steps + 1)
        if min_lr_ratio is not None:
            self.min_lr = self.peak_lr * float(min_lr_ratio)
        self.reset_progress()

    def reset_progress(self) -> None:
        self.completed_steps = 0
        self._set_lr(self._lr_at_step(0))


def use_cosine_lr_decay(name: str | None) -> bool:
    key = (name or "").strip().lower()
    return key in ("warmup_cosine", "cosine")


def resolve_lr_schedule(args: Any, train_cfg: dict | None = None) -> str:
    cfg = train_cfg or {}
    explicit = getattr(args, "lr_schedule", None) if args is not None else None
    name = explicit if explicit not in (None, "") else cfg.get("lr_schedule")
    if not name:
        return "warmup_cosine"
    return str(name)


def should_reset_lr_schedule_on_resume(
    *,
    reset_lr_schedule: bool | None,
    checkpoint_completed_steps: int,
    total_optimizer_steps: int,
    checkpoint_peak_lr: float,
    peak_lr: float,
) -> bool:
    if reset_lr_schedule is False:
        return False
    if reset_lr_schedule is True:
        return True
    if checkpoint_completed_steps >= total_optimizer_steps:
        return True
    if abs(float(checkpoint_peak_lr) - float(peak_lr)) > 1e-12:
        return True
    return False


def build_lr_scheduler(
    optimizer: torch.optim.Optimizer,
    *,
    peak_lr: float,
    warmup_steps: int,
    total_optimizer_steps: int,
    lr_min_ratio: float = 0.1,
    use_cosine_decay: bool = True,
):
    if use_cosine_decay:
        return WarmupCosineLR(
            optimizer,
            peak_lr=peak_lr,
            warmup_steps=warmup_steps,
            total_steps=total_optimizer_steps,
            min_lr_ratio=lr_min_ratio,
        )
    return LinearWarmupLR(optimizer, peak_lr=peak_lr, warmup_steps=warmup_steps)


class LightningCompatLRScheduler(torch.optim.lr_scheduler.LRScheduler):
    """把 ``WarmupCosineLR`` / ``LinearWarmupLR`` 接到 Lightning ``configure_optimizers``。"""

    def __init__(self, wrapped: LinearWarmupLR | WarmupCosineLR) -> None:
        self.wrapped = wrapped
        super().__init__(wrapped.optimizer, last_epoch=-1)

    def get_lr(self) -> list[float]:
        return [float(group["lr"]) for group in self.optimizer.param_groups]

    def step(self, epoch: int | None = None) -> None:  # noqa: ARG002
        if self._step_count == 0:
            self._step_count = 1
            self.last_epoch = 0
            self._last_lr = self.get_lr()
            return
        self.wrapped.step()
        self._step_count += 1
        self.last_epoch = int(getattr(self.wrapped, "completed_steps", self.last_epoch + 1))
        self._last_lr = self.get_lr()

    def state_dict(self) -> dict[str, Any]:
        return {
            "wrapped": self.wrapped.state_dict(),
            "_step_count": self._step_count,
            "last_epoch": self.last_epoch,
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        wrapped_state = state_dict.get("wrapped")
        if isinstance(wrapped_state, dict):
            self.wrapped.load_state_dict(wrapped_state)
            self._step_count = int(state_dict.get("_step_count", 1))
            self.last_epoch = int(state_dict.get("last_epoch", getattr(self.wrapped, "completed_steps", 0)))
        else:
            self.wrapped.load_state_dict(state_dict)
        self._last_lr = self.get_lr()


def as_torch_lr_scheduler(wrapped: LinearWarmupLR | WarmupCosineLR) -> LightningCompatLRScheduler:
    return LightningCompatLRScheduler(wrapped)


def apply_param_group_lr_scales(param_groups: list[dict[str, Any]], peak_lr: float) -> list[dict[str, Any]]:
    """让 ``WarmupCosineLR`` 按 ``lr_scale`` 保留 heads / tokenizer 等分组学习率。"""
    peak = float(peak_lr)
    for group in param_groups:
        base = float(group.get("lr", peak))
        group["lr_scale"] = (base / peak) if peak else 1.0
    return param_groups
