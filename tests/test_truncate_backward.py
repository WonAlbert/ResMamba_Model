from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch

from resmamba_signal_model.models.model import SignalFoundationModel, SignalModelConfig


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
    }


def test_truncate_backward_blocks_encoder_grads() -> None:
    model = SignalFoundationModel(_ci_cfg())
    model.truncate_backward = True
    model.skip_recon = True
    model.train()
    for param in model.encoder.parameters():
        param.requires_grad = True
    for param in model.encoder_pool.parameters():
        param.requires_grad = True
    out = model(make_batch(), mode="task", task="modulation")
    out["modulation_logits"].sum().backward()
    for param in model.encoder.parameters():
        assert param.grad is None
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.modulation_head.parameters())
    assert any(p.grad is not None for p in model.task_interface.parameters())
    # encoder 权重无梯度；encoder_pool 在 detach(h_enc) 上重算，可读出可训
    assert not out["h_enc"].requires_grad
    assert any(p.grad is not None and float(p.grad.abs().sum()) > 0 for p in model.encoder_pool.parameters())


def test_truncate_backward_trains_view_adapters() -> None:
    """stage2 截断反传时，UTI specialist view adapters 仍必须收到梯度。"""
    model = SignalFoundationModel(_ci_cfg())
    model.truncate_backward = True
    model.skip_recon = True
    model.train()
    for param in model.parameters():
        param.requires_grad = False
    for param in model.task_interface.parameters():
        param.requires_grad = True
    for param in model.modulation_head.parameters():
        param.requires_grad = True
    for param in model.encoder_pool.parameters():
        param.requires_grad = True
    out = model(make_batch(), mode="task", task="modulation")
    out["modulation_logits"].sum().backward()
    adapter_grads = [
        param.grad
        for param in model.task_interface.view_adapters.parameters()
        if param.requires_grad
    ]
    assert adapter_grads
    assert any(grad is not None and float(grad.abs().sum()) > 0.0 for grad in adapter_grads)
    assert out["z_semantic"].requires_grad


def test_truncate_backward_trains_emitter_head_not_encoder() -> None:
    model = SignalFoundationModel(_ci_cfg())
    model.truncate_backward = True
    model.skip_recon = True
    model.train()
    for param in model.parameters():
        param.requires_grad = False
    for param in model.emitter_head.parameters():
        param.requires_grad = True
    for param in model.task_interface.parameters():
        param.requires_grad = True
    out = model(make_batch(), mode="task", task="emitter")
    out["emitter_logits"].sum().backward()
    for param in model.encoder.parameters():
        assert param.grad is None
    assert any(
        p.grad is not None and float(p.grad.abs().sum()) > 0
        for p in model.emitter_head.parameters()
    )
    assert "emitter_fingerprint" not in out
    assert "task_pooled" in out
