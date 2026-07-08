from __future__ import annotations

import math

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
            group["lr"] = lr

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
            group["lr"] = lr

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
