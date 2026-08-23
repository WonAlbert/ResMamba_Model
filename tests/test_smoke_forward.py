from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch

from resmamba_signal_model import SignalFoundationModel, SignalModelConfig
from resmamba_signal_model.data.rfdata import normalize_iq
from resmamba_signal_model.training.losses import foundation_pretrain_losses


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


def test_joint_power_and_complex_absmax_normalization() -> None:
    iq = torch.randn(2, 256) * 4.0 + 2.0
    joint = normalize_iq(iq, "joint_power")
    ab = normalize_iq(iq, "complex_absmax")
    assert joint.shape == iq.shape
    assert ab.shape == iq.shape
    assert joint.abs().max() <= 5.0


def test_foundation_forward_and_losses() -> None:
    cfg = _ci_cfg()
    model = SignalFoundationModel(cfg)
    model.eval()
    batch = make_batch()
    mae = model(batch, mode="pretrain")
    assert mae["z"].shape == (4, cfg.d_model)
    assert mae["mae_pred"].shape[0] == 4
    losses = foundation_pretrain_losses(mae, batch)
    assert losses
    assert all(torch.isfinite(v) for v in losses.values())
    mod = model(batch, mode="task", task="tx_modulation")
    assert "tx_modulation_logits" in mod or "task_logits" in mod
    clu = model(batch, mode="task", task="ld_clustering")
    assert "cluster_logits" in clu
    pred = model(batch, mode="task", task="prediction")
    assert pred["mae_pred"].shape[2:] == (2, cfg.patch_size)


def test_pretrain_forward_accepts_collate_modality_id_list() -> None:
    """真实 collate / combine_then_pack 把 modality_id 收成字符串列表，不能当 Tensor 用。"""
    model = SignalFoundationModel(_ci_cfg()).eval()
    batch = make_batch()
    batch["modality_id"] = ["rf", "rf", "rf", "rf"]
    out = model(batch, mode="pretrain")
    assert out["z"].shape == (4, model.cfg.d_model)

    packed = {
        "iq": [torch.randn(2, 64), torch.randn(2, 48), torch.randn(2, 80)],
        "length": torch.tensor([64, 48, 80]),
        "dataset_id": torch.tensor([0, 1, 2]),
        "modality_id": ["rf", "rf", "rf"],
    }
    packed_out = model(packed, mode="pretrain")
    assert packed_out["z"].shape[0] == 3
