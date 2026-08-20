from __future__ import annotations

import torch

from resmamba_signal_model.data.packing import build_cu_seqlens, build_seq_idx, pack_valid_tokens, scatter_packed_tokens
from resmamba_signal_model.data.rfdata import variable_length_collate
from resmamba_signal_model.models.model import SignalFoundationModel, SignalModelConfig
from resmamba_signal_model.training.losses import foundation_pretrain_losses


def test_flip_packed_tokens_matches_flip_segments() -> None:
    lengths = [5, 1, 8]
    x = torch.randn(1, sum(lengths), 4)
    cu = build_cu_seqlens(lengths, device=x.device)
    from resmamba_signal_model.data.packing import flip_packed_tokens, flip_segments

    assert torch.equal(flip_packed_tokens(x, cu), flip_segments(x, cu))


def test_packing_utils() -> None:
    lengths = [20, 260]
    cu = build_cu_seqlens(lengths, device=torch.device("cpu"))
    seq_idx = build_seq_idx(lengths, device=torch.device("cpu"))
    assert cu.tolist() == [0, 20, 280]
    assert seq_idx.shape == (1, 280)
    assert seq_idx[0, :20].unique().tolist() == [0]
    assert seq_idx[0, 20:].unique().tolist() == [1]


def test_pack_scatter_roundtrip() -> None:
    tokens = torch.randn(2, 5, 3)
    valid = torch.tensor([[1, 1, 1, 0, 0], [1, 0, 1, 1, 0]], dtype=torch.bool)
    packed, cu, seq = pack_valid_tokens(tokens, valid)
    back = scatter_packed_tokens(packed, valid)
    assert torch.allclose(back[valid], tokens[valid])
    assert torch.equal(back[~valid], torch.zeros_like(back[~valid]))
    assert cu[-1].item() == int(valid.sum())


def test_sequence_packing_mae_smoke() -> None:
    items = [
        {"iq": torch.randn(2, 128), "length": 128, "dataset_id": 0, "task_type_id": 0, "mod_label_id": 1, "emitter_id": 2},
        {"iq": torch.randn(2, 256), "length": 256, "dataset_id": 1, "task_type_id": 1, "mod_label_id": 2, "emitter_id": 3},
    ]
    batch = variable_length_collate(items)
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
        sequence_packing=True,
        num_datasets=4,
    )
    model = SignalFoundationModel(cfg)
    model.eval()
    out = model(batch, mode="pretrain")
    assert out["z"].shape == (2, 32)
    assert out["mae_pred"].shape[0] == 2
    losses = foundation_pretrain_losses(out, batch)
    assert all(torch.isfinite(v) for v in losses.values())
