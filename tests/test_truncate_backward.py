from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch

from resmamba_signal_model.models.model import SignalFoundationModel, SignalModelConfig
from resmamba_signal_model.training.freeze import apply_stage_freeze


def _ci_cfg(**kwargs) -> SignalModelConfig:
    base = dict(
        d_model=32,
        mamba_d_state=8,
        mamba_headdim=16,
        require_mamba_kernel=False,
        allow_fallback_mamba=True,
        attn_num_heads=4,
        patch_size=8,
        stem_channels=8,
        freq_bands=4,
        dropout=0.0,
        p_trunc=0.0,
        num_datasets=4,
        num_mod_classes=5,
        num_emitters=16,
        num_prototypes=8,
        build_task_heads=True,
        build_task_interface=False,
        train_encoder=True,
        train_decoder=True,
        train_heads=True,
    )
    base.update(kwargs)
    return SignalModelConfig(**base)


def make_batch() -> dict[str, torch.Tensor]:
    length = 64
    sample_mask = torch.ones(4, length, dtype=torch.bool)
    sample_mask[1, 40:] = False
    sample_mask[2, 24:] = False
    return {
        "iq": torch.randn(4, 2, length),
        "sample_mask": sample_mask,
        "length": torch.tensor([64, 40, 24, 64]),
        "dataset_id": torch.tensor([0, 1, 2, 2]),
        "mod_label_id": torch.tensor([3, 3, 4, 4]),
        "emitter_id": torch.tensor([9, 9, 10, 11]),
        "canonical_mod_label_id": torch.tensor([3, 3, 4, 4]),
    }


def test_truncate_backward_blocks_encoder_grads() -> None:
    model = SignalFoundationModel(_ci_cfg())
    apply_stage_freeze(model, "stage2", task="tx_modulation", train_cfg={"truncate_backward": True, "skip_recon": True})
    model.train()
    out = model(make_batch(), mode="task", task="tx_modulation")
    logits = out.get("task_logits")
    if logits is None:
        logits = out.get("tx_modulation_logits")
    logits.sum().backward()
    for param in model.encoder.parameters():
        assert param.grad is None
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.tx_modulation_head.parameters())
    assert not out["h_enc"].requires_grad


def test_truncate_backward_trains_head_not_encoder_pool() -> None:
    model = SignalFoundationModel(_ci_cfg())
    apply_stage_freeze(model, "stage2", task="ld_model", train_cfg={"truncate_backward": True, "skip_recon": True})
    model.train()
    out = model(make_batch(), mode="task", task="ld_model")
    out["task_logits"].sum().backward()
    for param in model.encoder_pool.parameters():
        assert param.grad is None
    assert any(p.grad is not None for p in model.ld_model_head.parameters())


def test_truncate_backward_ld_model_head_not_encoder() -> None:
    model = SignalFoundationModel(_ci_cfg())
    apply_stage_freeze(model, "stage2", task="ld_model", train_cfg={"truncate_backward": True, "skip_recon": True})
    model.train()
    out = model(make_batch(), mode="task", task="ld_model")
    out["task_logits"].sum().backward()
    for param in model.encoder.parameters():
        assert param.grad is None
    assert any(p.grad is not None and float(p.grad.abs().sum()) > 0 for p in model.ld_model_head.parameters())
    assert "task_pooled" in out
