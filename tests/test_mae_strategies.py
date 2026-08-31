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


def test_mae_suffix_strategy_via_mae_probs() -> None:
    model = _tiny(mae_mask_probs={"random": 0.0, "contiguous": 0.0, "mixed": 0.0, "suffix": 1.0})
    patch_mask = torch.ones(2, 16, dtype=torch.bool)
    mae, suffix, span, strategy = model._mask_for_mode(patch_mask, "mae", mae_strategy="suffix")
    assert strategy == "suffix"
    assert not bool(mae.any())
    assert not bool(span.any())
    assert bool(suffix.any())
    assert int((patch_mask & ~suffix)[0].sum()) >= 1


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
    rand_l = foundation_pretrain_losses(random_out, batch, include={"mse", "impute"})
    contig_l = foundation_pretrain_losses(contig_out, batch, include={"mse", "impute"})
    mixed_l = foundation_pretrain_losses(mixed_out, batch, include={"mse", "impute"})
    assert float(rand_l["mse"].detach()) > 0.0
    assert float(rand_l["impute"].detach()) == 0.0
    assert float(contig_l["mse"].detach()) > 0.0
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


def test_adaptive_mae_mask_hardens_long_sequences() -> None:
    from resmamba_signal_model.models.model import adaptive_mae_mask_ratios

    num_valid = torch.tensor([8, 32, 64, 128], dtype=torch.long)
    ratios = adaptive_mae_mask_ratios(
        num_valid,
        base=0.5,
        ref_patches=8,
        min_ratio=0.45,
        max_ratio=0.75,
        log_scale=0.10,
    )
    assert abs(float(ratios[0]) - 0.5) < 1e-5
    assert float(ratios[1]) > float(ratios[0])
    assert float(ratios[2]) >= float(ratios[1])
    assert abs(float(ratios[2]) - 0.75) < 1e-5  # log2(8)*0.1+0.5=0.8 → clip
    assert abs(float(ratios[3]) - 0.75) < 1e-5

    model = _tiny(adaptive_mae_mask=True, mask_ratio=0.5, mae_mask_ref_patches=8, mae_mask_log_scale=0.10)
    short = torch.ones(1, 8, dtype=torch.bool)
    long = torch.ones(1, 64, dtype=torch.bool)
    torch.manual_seed(0)
    m_short = model.make_mae_mask(short)
    torch.manual_seed(0)
    m_long = model.make_mae_mask(long)
    # 长序列 mask 比例更高（允许 ±1 patch 的 round 误差）
    assert float(m_long.float().mean()) > float(m_short.float().mean()) + 0.05
