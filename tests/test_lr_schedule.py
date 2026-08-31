from __future__ import annotations

import argparse

import torch

from resmamba_signal_model.training.lr_schedule import (
    LightningCompatLRScheduler,
    WarmupCosineLR,
    as_torch_lr_scheduler,
    build_lr_scheduler,
    optimizer_steps_per_epoch,
    resolve_lr_schedule,
    scale_lr_for_grad_accum,
    should_reset_lr_schedule_on_resume,
    total_optimizer_steps_from_cfg,
    use_cosine_lr_decay,
)


def test_use_cosine_lr_decay() -> None:
    assert use_cosine_lr_decay("warmup_cosine")
    assert use_cosine_lr_decay("cosine")
    assert not use_cosine_lr_decay("warmup_constant")


def test_resolve_lr_schedule_defaults_to_cosine() -> None:
    args = argparse.Namespace(lr_schedule=None)
    assert resolve_lr_schedule(args, {}) == "warmup_cosine"
    assert resolve_lr_schedule(args, {"lr_schedule": "warmup_constant"}) == "warmup_constant"


def test_build_lr_scheduler_warmup_cosine() -> None:
    model = torch.nn.Linear(4, 2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.2e-3)
    scheduler = build_lr_scheduler(
        optimizer,
        peak_lr=1.2e-3,
        warmup_steps=10,
        total_optimizer_steps=100,
        lr_min_ratio=0.1,
        use_cosine_decay=True,
    )
    assert isinstance(scheduler, WarmupCosineLR)
    for _ in range(10):
        scheduler.step()
    warmup_lr = optimizer.param_groups[0]["lr"]
    assert abs(warmup_lr - 1.2e-3) < 1e-9
    for _ in range(40):
        scheduler.step()
    mid_lr = optimizer.param_groups[0]["lr"]
    assert mid_lr < warmup_lr
    for _ in range(50):
        scheduler.step()
    final_lr = optimizer.param_groups[0]["lr"]
    assert abs(final_lr - 1.2e-4) < 1e-8


def test_as_torch_lr_scheduler_wraps_warmup_cosine() -> None:
    model = torch.nn.Linear(4, 2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.0e-3)
    wrapped = build_lr_scheduler(
        optimizer,
        peak_lr=1.0e-3,
        warmup_steps=2,
        total_optimizer_steps=10,
        use_cosine_decay=True,
    )
    scheduler = as_torch_lr_scheduler(wrapped)
    assert isinstance(scheduler, torch.optim.lr_scheduler.LRScheduler)
    assert isinstance(scheduler, LightningCompatLRScheduler)
    assert isinstance(scheduler.wrapped, WarmupCosineLR)
    first = optimizer.param_groups[0]["lr"]
    scheduler.step()
    assert optimizer.param_groups[0]["lr"] >= first


def test_scale_lr_for_grad_accum() -> None:
    assert scale_lr_for_grad_accum(3e-4, 4, enabled=True) == 1.2e-3
    assert scale_lr_for_grad_accum(3e-4, 4, enabled=False) == 3e-4


def test_total_optimizer_steps_accounts_for_grad_accum() -> None:
    cfg = {"steps_per_epoch": 600, "epochs": 80, "gradient_accumulation_steps": 4}
    assert optimizer_steps_per_epoch(cfg) == 150
    assert total_optimizer_steps_from_cfg(cfg) == 12000
    cfg2 = {"steps_per_epoch": 400, "epochs": 40, "gradient_accumulation_steps": 2}
    assert optimizer_steps_per_epoch(cfg2) == 200
    assert total_optimizer_steps_from_cfg(cfg2) == 8000


def test_warmup_cosine_reconfigure_resets_progress() -> None:
    model = torch.nn.Linear(4, 2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.0e-3)
    scheduler = WarmupCosineLR(
        optimizer,
        peak_lr=1.0e-3,
        warmup_steps=10,
        total_steps=100,
        min_lr_ratio=0.1,
    )
    for _ in range(80):
        scheduler.step()
    assert scheduler.completed_steps == 80
    scheduler.reconfigure(peak_lr=1.0e-4, warmup_steps=5, total_steps=50, min_lr_ratio=0.1)
    assert scheduler.completed_steps == 0
    assert optimizer.param_groups[0]["lr"] < 1.0e-4


def test_should_reset_lr_schedule_on_resume() -> None:
    assert should_reset_lr_schedule_on_resume(
        reset_lr_schedule=None,
        checkpoint_completed_steps=5169,
        total_optimizer_steps=1815,
        checkpoint_peak_lr=4e-4,
        peak_lr=1e-4,
    )
    assert not should_reset_lr_schedule_on_resume(
        reset_lr_schedule=False,
        checkpoint_completed_steps=5169,
        total_optimizer_steps=1815,
        checkpoint_peak_lr=4e-4,
        peak_lr=1e-4,
    )
