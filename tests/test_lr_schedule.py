from __future__ import annotations

import argparse
from types import SimpleNamespace

import torch

from resmamba_signal_model.training.lr_schedule import WarmupCosineLR, scale_lr_for_grad_accum

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from train_pipeline import (  # noqa: E402
    build_lr_scheduler,
    resolve_lr_schedule,
    should_reset_lr_schedule_on_resume,
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


def test_build_lr_scheduler_stage2_cosine() -> None:
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


def test_scale_lr_for_grad_accum() -> None:
    assert scale_lr_for_grad_accum(3e-4, 4, enabled=True) == 1.2e-3
    assert scale_lr_for_grad_accum(3e-4, 4, enabled=False) == 3e-4


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
