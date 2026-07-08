from __future__ import annotations

import torch

from resmamba_signal_model import ResMambaSignalConfig, ResMambaSignalModel
from resmamba_signal_model.data.packing import build_cu_seqlens, build_seq_idx
from resmamba_signal_model.data.rfdata import variable_length_collate
from resmamba_signal_model.training.losses import pretrain_smoke_losses


def _make_mixed_batch() -> dict:
    items = [
        {
            "iq": torch.randn(2, 128),
            "length": 128,
            "dataset_id": 0,
            "task_type_id": 0,
            "mod_label_id": 1,
            "emitter_id": 2,
        },
        {
            "iq": torch.randn(2, 256),
            "length": 256,
            "dataset_id": 1,
            "task_type_id": 1,
            "mod_label_id": 2,
            "emitter_id": 3,
        },
    ]
    return variable_length_collate(items)


def test_packing_utils() -> None:
    lengths = [20, 260]
    cu = build_cu_seqlens(lengths, device=torch.device("cpu"))
    seq_idx = build_seq_idx(lengths, device=torch.device("cpu"))
    assert cu.tolist() == [0, 20, 280]
    assert seq_idx.shape == (1, 280)
    assert seq_idx[0, :20].unique().tolist() == [0]
    assert seq_idx[0, 20:].unique().tolist() == [1]


def test_sequence_packing_mae_smoke() -> None:
    if not torch.cuda.is_available():
        return
    cfg = ResMambaSignalConfig(
        d_model=32,
        encoder_layers=1,
        decoder_layers=1,
        space_layers=1,
        max_tokens=64,
        patch_size=8,
        mamba_d_state=8,
        mamba_headdim=16,
        num_datasets=4,
        num_mod_classes=5,
        num_emitters=16,
        num_prototypes=8,
        sequence_packing=True,
    )
    model = ResMambaSignalModel(cfg).cuda()
    batch = _make_mixed_batch()
    batch["iq"] = [tensor.cuda() for tensor in batch["iq"]]
    for key in ("dataset_id", "task_type_id", "mod_label_id", "emitter_id", "length"):
        batch[key] = batch[key].cuda()
    mae = model(batch, mode="mae")
    assert mae["seq_idx"].shape[1] == mae["hidden"].shape[1]
    assert mae["modulation_repr"].shape == (2, cfg.d_model)
    assert mae["mae_pred"].shape[0] == 2
    losses = pretrain_smoke_losses(mae, batch)
    assert all(torch.isfinite(v) for v in losses.values())
