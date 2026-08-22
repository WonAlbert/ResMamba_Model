from __future__ import annotations

import torch

from resmamba_signal_model.models.model import SignalFoundationModel, SignalModelConfig
from resmamba_signal_model.training.losses import foundation_pretrain_losses


def _tiny(**kwargs) -> SignalFoundationModel:
    payload = dict(
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
        mask_ratio=0.5,
        build_task_interface=True,
        build_task_heads=False,
    )
    payload.update(kwargs)
    return SignalFoundationModel(SignalModelConfig(**payload))


def test_mae_random_contiguous_mixed_masks() -> None:
    model = _tiny()
    patch_mask = torch.ones(2, 16, dtype=torch.bool)
    mae, suffix, span, strategy = model._mask_for_mode(patch_mask, "mae", mae_strategy="random")
    assert strategy == "random"
    assert not bool(span.any())
    assert not bool(suffix.any())
    assert bool(mae.any())
    assert int(mae[0].sum()) <= 15

    mae, suffix, span, strategy = model._mask_for_mode(patch_mask, "mae", mae_strategy="contiguous")
    assert strategy == "contiguous"
    assert torch.equal(mae, span)
    assert bool(span.any())
    assert int((patch_mask & ~span)[0].sum()) >= 1

    mae, suffix, span, strategy = model._mask_for_mode(patch_mask, "mae", mae_strategy="mixed")
    assert strategy == "mixed"
    union = mae | span
    assert bool(union.any())
    assert int((patch_mask & ~union)[0].sum()) >= 1


def test_first_class_mask_mode_names() -> None:
    model = _tiny()
    patch_mask = torch.ones(1, 12, dtype=torch.bool)
    _, _, _, strategy = model._mask_for_mode(patch_mask, "contiguous")
    assert strategy == "contiguous"
    _, _, span, strategy = model._mask_for_mode(patch_mask, "random")
    assert strategy == "random"
    assert not bool(span.any())


def test_pretrain_samples_mask_strategy_and_logs_it() -> None:
    torch.manual_seed(0)
    model = _tiny()
    model.train()
    batch = {
        "iq": torch.randn(2, 2, 64),
        "sample_mask": torch.ones(2, 64, dtype=torch.bool),
    }
    seen: set[str] = set()
    for _ in range(36):
        out = model(batch, mode="pretrain", mask_mode="mae")
        seen.add(str(out["mask_strategy"]))
        if seen == {"random", "contiguous", "mixed"}:
            break
    assert seen == {"random", "contiguous", "mixed"}


def test_contiguous_mae_loss_is_nonzero_impute_only_with_span() -> None:
    torch.manual_seed(1)
    model = _tiny()
    model.eval()
    batch = {
        "iq": torch.randn(2, 2, 64),
        "sample_mask": torch.ones(2, 64, dtype=torch.bool),
    }
    random_out = model(batch, mode="pretrain", mask_mode="random")
    contig_out = model(batch, mode="pretrain", mask_mode="contiguous")
    mixed_out = model(batch, mode="pretrain", mask_mode="mixed")
    rand_l = foundation_pretrain_losses(random_out, batch, include={"mae", "impute"})
    contig_l = foundation_pretrain_losses(contig_out, batch, include={"mae", "impute"})
    mixed_l = foundation_pretrain_losses(mixed_out, batch, include={"mae", "impute"})
    assert float(rand_l["mae"].detach()) > 0.0
    assert float(rand_l["impute"].detach()) == 0.0
    assert float(contig_l["mae"].detach()) > 0.0
    assert float(contig_l["impute"].detach()) > 0.0
    assert float(mixed_l["impute"].detach()) > 0.0
    assert random_out["mask_strategy"] == "random"
    assert contig_out["mask_strategy"] == "contiguous"
    assert mixed_out["mask_strategy"] == "mixed"


def test_suffix_and_span_modes_are_not_sampled() -> None:
    model = _tiny()
    model.train()
    batch = {
        "iq": torch.randn(2, 2, 64),
        "sample_mask": torch.ones(2, 64, dtype=torch.bool),
    }
    out = model(batch, mode="pretrain", mask_mode="suffix")
    assert out["mask_strategy"] == "suffix"
    assert bool(out["suffix_mask"].any())
    out = model(batch, mode="pretrain", mask_mode="span")
    assert out["mask_strategy"] == "span"
    assert bool(out["span_mask"].any())
