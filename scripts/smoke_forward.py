#!/usr/bin/env python
from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch

from resmamba_signal_model import SignalFoundationModel, SignalModelConfig
from resmamba_signal_model.training.losses import foundation_pretrain_losses, weighted_pretrain_loss


def tiny_cfg() -> SignalModelConfig:
    return SignalModelConfig(
        d_model=32,
        encoder_mamba_layers=4,
        encoder_transformer_layers=1,
        decoder_mamba_layers=2,
        mamba_d_state=8,
        mamba_headdim=16,
        require_mamba_kernel=False,
        allow_fallback_mamba=True,
        attn_num_heads=4,
        attn_window=64,
        patch_size=8,
        stem_channels=8,
        freq_bands=4,
        dropout=0.0,
        l_min=16,
        p_trunc=0.0,
        sequence_packing=True,
        num_datasets=8,
    )


def main() -> None:
    torch.manual_seed(7)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SignalFoundationModel(tiny_cfg()).to(device)
    model.eval()
    iq = [torch.randn(2, 128, device=device), torch.randn(2, 4096, device=device)]
    batch = {
        "iq": iq,
        "length": torch.tensor([128, 4096], device=device),
        "dataset_id": torch.tensor([0, 1], device=device),
    }
    with torch.no_grad():
        out = model(batch, mode="pretrain")
        out_task = model(batch, mode="downstream", task="ld_intrapulse")
    losses = foundation_pretrain_losses(out, batch)
    total, parts = weighted_pretrain_loss(
        out,
        batch,
        {"mae": 1.0, "physical": 0.2, "readout": 0.1, "structure": 0.2},
    )
    print("z", tuple(out["z"].shape), "recon", tuple(out["mae_pred"].shape), "tokens", tuple(out["tokens"].shape))
    print("losses", {k: round(float(v.detach()), 6) for k, v in parts.items()})
    print("total", float(total.detach()))
    assert out["z"].shape[0] == 2
    assert torch.isfinite(out["z"]).all()
    assert "moe_load_balance" in out
    assert torch.isfinite(out["moe_load_balance"])
    assert "mse" in parts
    aux = out_task.get("moe_gate_weights")
    assert aux
    tok_weights = aux[0][0]
    active = tok_weights.sum(dim=-1) > 0
    assert bool(active.any())
    assert (tok_weights[active].argmax(dim=-1) == 0).all()
    print("smoke_forward ok")


if __name__ == "__main__":
    main()
