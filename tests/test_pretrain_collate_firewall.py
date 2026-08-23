from __future__ import annotations

from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from resmamba_signal_model.models.model import SignalFoundationModel, SignalModelConfig
from resmamba_signal_model.training.data_module import pretrain_collate_firewall


def test_pretrain_collate_strips_ids() -> None:
    batch = {
        "iq": torch.randn(2, 2, 32),
        "length": torch.tensor([32, 32]),
        "dataset_id": torch.tensor([1, 2]),
        "mod_label_id": torch.tensor([3, 4]),
        "emitter_id": torch.tensor([5, 6]),
        "h5_path": ["dataset/h5/radchar_train.h5", "dataset/h5/radar_mod15_train.h5"],
        "receiver_id": ["rx1", "rx2"],
    }
    out = pretrain_collate_firewall(batch)
    assert "dataset_id" not in out
    assert "mod_label_id" not in out
    assert "emitter_id" not in out
    assert "h5_path" not in out
    assert "receiver_id" not in out
    assert out["moe_route_stem"] == ["radchar", "radar_mod15"]
    assert "length" in out


def test_pretrain_forward_invariant_to_fake_labels() -> None:
    cfg = SignalModelConfig(
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
        build_task_interface=True,
    )
    model = SignalFoundationModel(cfg)
    model.eval()
    iq = torch.randn(1, 2, 32)
    mask = torch.ones(1, 32, dtype=torch.bool)
    batch_a = pretrain_collate_firewall({"iq": iq, "sample_mask": mask, "length": torch.tensor([32])})
    batch_b = pretrain_collate_firewall(
        {
            "iq": iq,
            "sample_mask": mask,
            "length": torch.tensor([32]),
            "dataset_id": torch.tensor([99]),
            "mod_label_id": torch.tensor([7]),
        }
    )
    with torch.no_grad():
        torch.manual_seed(42)
        out_a = model(batch_a, mode="pretrain")
        torch.manual_seed(42)
        out_b = model(batch_b, mode="pretrain")
    assert torch.allclose(out_a["z_enc"], out_b["z_enc"], atol=1e-5, rtol=1e-4)
