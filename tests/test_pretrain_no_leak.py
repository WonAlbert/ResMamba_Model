from __future__ import annotations

import torch

from resmamba_signal_model.data.contracts import CAPTURE_METADATA_KEYS
from resmamba_signal_model.data.rfdata import variable_length_collate
from resmamba_signal_model.models.model import SignalFoundationModel, SignalModelConfig
from resmamba_signal_model.training.data_module import (
    SyntheticIQDataset,
    _collate_with_source,
    is_pretrain_blocked_key,
    pretrain_collate_firewall,
)


def _cfg(**overrides) -> SignalModelConfig:
    payload = dict(
        d_model=32,
        encoder_mamba_layers=1,
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
        num_emitters=8,
        num_prototypes=4,
        build_task_heads=False,
        build_task_interface=True,
    )
    payload.update(overrides)
    return SignalModelConfig(**payload)


def test_pretrain_collate_has_no_id_fields() -> None:
    samples = [SyntheticIQDataset(n=8, lengths=(32,), seed=0)[i] for i in range(4)]
    assert "mod_label_id" in samples[0]
    assert "dataset_id" in samples[0]
    batch = _collate_with_source("pretrain", stage="pretrain")(samples)
    blocked = (
        "mod_label_id",
        "emitter_id",
        "global_emitter_id",
        "global_label_id",
        "dataset_id",
        "task_type_id",
        "canonical_mod_label_id",
        *CAPTURE_METADATA_KEYS,
        "h5_path",
    )
    for key in blocked:
        assert key not in batch, key
        assert not is_pretrain_blocked_key("iq")
    assert "iq" in batch
    assert "length" in batch


def test_firewall_strips_forged_global_keys() -> None:
    samples = [SyntheticIQDataset(n=4, lengths=(16,), seed=1)[0]]
    raw = variable_length_collate(samples)
    raw["global_session_code"] = ["sess-0"]
    cleaned = pretrain_collate_firewall(raw)
    assert "global_session_code" not in cleaned
    assert "global_emitter_id" not in cleaned
    assert "dataset_id" not in cleaned


def test_same_iq_forged_labels_ids_yield_identical_pretrain_outputs() -> None:
    torch.manual_seed(0)
    model = SignalFoundationModel(_cfg()).eval()
    iq = torch.randn(2, 2, 64)
    mask = torch.ones(2, 64, dtype=torch.bool)
    base = {"iq": iq, "sample_mask": mask}
    forged = {
        **base,
        "mod_label_id": torch.tensor([1, 2]),
        "emitter_id": torch.tensor([3, 4]),
        "global_emitter_id": torch.tensor([5, 6]),
        "global_label_id": torch.tensor([7, 8]),
        "dataset_id": torch.tensor([1, 2]),
        "task_type_id": torch.tensor([1, 1]),
        "receiver_id": ["rx0", "rx1"],
        "session_id": ["s0", "s1"],
        "channel_id": ["c0", "c1"],
        "capture_id": ["cap0", "cap1"],
    }
    torch.manual_seed(42)
    out0 = model(base, mode="pretrain")
    torch.manual_seed(42)
    out1 = model(forged, mode="pretrain")
    for key in ("z", "recon_norm"):
        assert torch.allclose(out0[key], out1[key], atol=1.0e-5, rtol=1.0e-5), key
    if out0.get("uti_pooled") is not None and out1.get("uti_pooled") is not None:
        assert torch.allclose(out0["uti_pooled"], out1["uti_pooled"], atol=1.0e-5, rtol=1.0e-5)
    cleaned = pretrain_collate_firewall(forged)
    for key in (
        "mod_label_id",
        "emitter_id",
        "global_emitter_id",
        "global_label_id",
        "dataset_id",
        "task_type_id",
        *CAPTURE_METADATA_KEYS,
    ):
        assert key not in cleaned
