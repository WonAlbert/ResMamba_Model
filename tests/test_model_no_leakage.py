from __future__ import annotations

import torch

from resmamba_signal_model.models.model import SignalFoundationModel, SignalModelConfig
from resmamba_signal_model.models.task_interface import DEFAULT_TASKS
from resmamba_signal_model.training.freeze import apply_stage_freeze


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
        build_task_heads=True,
        task_names=DEFAULT_TASKS,
    )
    payload.update(overrides)
    return SignalModelConfig(**payload)


def _rewrite_targets(
    iq: torch.Tensor,
    target_mask: torch.Tensor,
    patch_size: int,
) -> torch.Tensor:
    samples = target_mask.repeat_interleave(patch_size, dim=1)[:, : iq.shape[-1]]
    rewritten = iq.clone()
    replacement = torch.randn_like(iq) * 100.0 + 37.0
    return torch.where(samples.unsqueeze(1), replacement, rewritten)


def _assert_predictions_equal(first: dict, second: dict) -> None:
    for key in (
        "tokens",
        "z",
        "z_general",
        "h_general",
        "recon_norm",
        "mae_pred",
        "pred_patches",
        "context_physics",
    ):
        assert torch.allclose(first[key], second[key], atol=1.0e-6, rtol=1.0e-6), key
    assert torch.equal(first["target_mask"], second["target_mask"])
    assert torch.allclose(first["revin_stats"].mean, second["revin_stats"].mean)
    assert torch.allclose(first["revin_stats"].std, second["revin_stats"].std)


def test_suffix_hidden_truth_rewrite_cannot_change_predictions() -> None:
    torch.manual_seed(7)
    model = SignalFoundationModel(_cfg()).eval()
    iq = torch.randn(2, 2, 64)
    sample_mask = torch.ones(2, 64, dtype=torch.bool)

    first = model(iq, sample_mask, mode="task", task="prediction")
    assert not first["mae_mask"].any()
    assert first["suffix_mask"].any()
    assert not first["span_mask"].any()
    assert torch.equal(first["target_mask"], first["suffix_mask"])

    rewritten = _rewrite_targets(iq, first["target_mask"], model.cfg.patch_size)
    second = model(rewritten, sample_mask, mode="task", task="prediction")
    _assert_predictions_equal(first, second)
    assert not torch.allclose(first["patch_targets"], second["patch_targets"])

    observed = first["observed_sample_mask"].to(dtype=iq.dtype)
    expected_mean = (iq * observed.unsqueeze(1)).sum(dim=-1) / observed.sum(dim=-1, keepdim=True)
    assert torch.allclose(first["revin_stats"].mean, expected_mean, atol=1.0e-6)


def test_span_hidden_truth_rewrite_cannot_change_predictions() -> None:
    torch.manual_seed(11)
    model = SignalFoundationModel(_cfg()).eval()
    iq = torch.randn(2, 2, 64)
    sample_mask = torch.ones(2, 64, dtype=torch.bool)

    torch.manual_seed(123)
    first = model(iq, sample_mask, mode="task", task="prediction")
    assert first["suffix_mask"].any()
    rewritten = _rewrite_targets(iq, first["target_mask"], model.cfg.patch_size)
    torch.manual_seed(123)
    second = model(rewritten, sample_mask, mode="task", task="prediction")
    _assert_predictions_equal(first, second)


def test_mae_hidden_truth_rewrite_cannot_change_predictions() -> None:
    torch.manual_seed(19)
    model = SignalFoundationModel(_cfg()).eval()
    iq = torch.randn(2, 2, 64)
    sample_mask = torch.ones(2, 64, dtype=torch.bool)

    torch.manual_seed(321)
    first = model(iq, sample_mask, mode="pretrain")
    rewritten = _rewrite_targets(iq, first["target_mask"], model.cfg.patch_size)
    torch.manual_seed(321)
    second = model(rewritten, sample_mask, mode="pretrain")
    for key in ("tokens", "z", "recon_norm", "mae_pred", "context_physics"):
        assert torch.allclose(first[key], second[key], atol=1.0e-6, rtol=1.0e-6), key


def test_discriminative_task_can_bypass_query_reconstruction() -> None:
    model = SignalFoundationModel(_cfg()).eval()
    model.skip_recon = True
    iq = torch.randn(2, 2, 64)
    mask = torch.ones(2, 64, dtype=torch.bool)

    discriminative = model(iq, mask, mode="task", task="tx_modulation")
    assert discriminative.get("query_h") is None
    assert torch.count_nonzero(discriminative["recon_norm"]) == 0
    assert "task_logits" in discriminative or "tx_modulation_logits" in discriminative

    generative = model(iq, mask, mode="task", task="prediction")
    assert generative["query_h"] is not None
    assert torch.count_nonzero(generative["recon_norm"]) > 0



def test_prediction_unified_generation_uses_decoder_not_head() -> None:
    torch.manual_seed(3)
    model = SignalFoundationModel(_cfg()).eval()
    iq = torch.randn(2, 2, 64)
    mask = torch.ones(2, 64, dtype=torch.bool)
    before = model(iq, mask, mode="task", task="prediction")["pred_patches"].clone()
    with torch.no_grad():
        for param in model.prediction_head.parameters():
            param.add_(1.5)
    after_head = model(iq, mask, mode="task", task="prediction")["pred_patches"]
    assert torch.allclose(before, after_head, atol=1.0e-5)
    with torch.no_grad():
        model.decoder.query_decoder.output.weight.add_(0.25)
    after_decoder = model(iq, mask, mode="task", task="prediction")["pred_patches"]
    assert not torch.allclose(before, after_decoder, atol=1.0e-5)


def test_stage2_unified_prediction_skips_head_training() -> None:
    torch.manual_seed(5)
    model = SignalFoundationModel(_cfg())
    apply_stage_freeze(
        model,
        "stage2",
        task="prediction",
        train_cfg={"truncate_backward": True, "skip_recon": True},
    )
    assert not any(param.requires_grad for param in model.prediction_head.parameters())


def test_stage2_legacy_prediction_trains_head() -> None:
    torch.manual_seed(6)
    model = SignalFoundationModel(_cfg(use_legacy_generation_heads=True, force_unified_generation=True))
    apply_stage_freeze(
        model,
        "stage2",
        task="prediction",
        train_cfg={"truncate_backward": True, "skip_recon": True},
    )
    assert any(param.requires_grad for param in model.prediction_head.parameters())
    assert not any(param.requires_grad for param in model.encoder.parameters())
    assert not any(param.requires_grad for param in model.decoder.parameters())


def test_legacy_prediction_head_affects_outputs_and_skips_decoder_recon() -> None:
    torch.manual_seed(7)
    model = SignalFoundationModel(
        _cfg(use_legacy_generation_heads=True, force_unified_generation=True, build_task_heads=True)
    )
    model.skip_recon = True
    model.eval()
    iq = torch.randn(2, 2, 64)
    mask = torch.ones(2, 64, dtype=torch.bool)
    before = model(iq, mask, mode="task", task="prediction")["pred_patches"].clone()
    assert model(iq, mask, mode="task", task="prediction").get("query_h") is None
    with torch.no_grad():
        for param in model.prediction_head.parameters():
            param.add_(1.5)
    after = model(iq, mask, mode="task", task="prediction")["pred_patches"]
    assert not torch.allclose(before, after, atol=1.0e-5)


def test_pretrain_ignores_dataset_id_in_decoder_condition() -> None:
    """预训练改变 dataset_id 不得改变骨干表征/重建（条件不注入域）。"""
    torch.manual_seed(0)
    model = SignalFoundationModel(_cfg(build_task_interface=True, build_task_heads=True)).eval()
    iq = torch.randn(2, 2, 64)
    mask = torch.ones(2, 64, dtype=torch.bool)
    batch0 = {"iq": iq, "sample_mask": mask, "dataset_id": torch.zeros(2, dtype=torch.long)}
    batch1 = {"iq": iq, "sample_mask": mask, "dataset_id": torch.ones(2, dtype=torch.long)}
    torch.manual_seed(42)
    out0 = model(batch0, mode="pretrain")
    torch.manual_seed(42)
    out1 = model(batch1, mode="pretrain")
    assert torch.allclose(out0["z"], out1["z"], atol=1.0e-5, rtol=1.0e-5)
    assert torch.allclose(out0["recon_norm"], out1["recon_norm"], atol=1.0e-5, rtol=1.0e-5)
