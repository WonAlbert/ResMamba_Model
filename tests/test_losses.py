import pytest
import torch
import torch.nn.functional as F

from resmamba_signal_model.training.logging_utils import should_log_loss_part
from resmamba_signal_model.training.losses import (
    PREDICTION_MAE_LOSS_SCALE,
    _clamp_loss,
    downstream_task_loss,
    foundation_pretrain_losses,
    mae_reconstruction_loss,
    resolve_recon_mask,
    safe_cross_entropy,
    sinkhorn_balanced_assignment,
    structure_preserving_loss,
    unsupervised_clustering_loss,
    vicreg_loss,
    weighted_pretrain_loss,
)


def test_safe_cross_entropy_filters_invalid_labels() -> None:
    logits = torch.tensor([[10.0, 0.0, -5.0], [0.0, 8.0, 1.0], [1.0, 2.0, 3.0]])
    labels = torch.tensor([0, 5, -1])
    loss = safe_cross_entropy(logits, labels)
    assert torch.isfinite(loss)
    assert loss.item() > 0


def test_safe_cross_entropy_clamps_large_logits() -> None:
    logits = torch.tensor([[1.0e4, -1.0e4]])
    labels = torch.tensor([0])
    loss = safe_cross_entropy(logits, labels)
    assert torch.isfinite(loss)


def test_mae_is_scale_invariant_in_norm_space() -> None:
    torch.manual_seed(0)
    mask = torch.ones(2, 4, dtype=torch.bool)
    small = torch.randn(2, 4, 2, 8)
    large = small * 250.0
    losses_small = foundation_pretrain_losses(
        {
            "recon_norm": small,
            "patch_targets_norm": torch.zeros_like(small),
            "mae_pred": large,
            "patch_targets": large * 0,
            "mae_mask": mask,
            "global_phys_pred": torch.zeros(2, 5),
            "global_phys_target": torch.zeros(2, 5),
        },
        include={"mae"},
    )
    losses_large_wrong = foundation_pretrain_losses(
        {
            "mae_pred": large,
            "patch_targets": torch.zeros_like(large),
            "mae_mask": mask,
            "global_phys_pred": torch.zeros(2, 5),
            "global_phys_target": torch.zeros(2, 5),
        },
        include={"mae"},
    )
    assert float(losses_small["mae"]) < 2.0
    assert float(losses_large_wrong["mae"]) > float(losses_small["mae"])


def test_prediction_mae_loss_scaled_to_ce_magnitude() -> None:
    pred = torch.zeros(2, 4, 2)
    target = torch.ones(2, 4, 2)
    mask = torch.ones(2, 4, dtype=torch.bool)
    raw = mae_reconstruction_loss(pred, target, mask)
    expected = F.smooth_l1_loss(pred[mask], target[mask])
    assert abs(float(raw) - float(expected)) < 1e-6
    loss, parts = downstream_task_loss(
        {"mae_pred": pred, "patch_targets": target, "mae_mask": mask},
        {},
        "prediction",
    )
    assert "mae" in parts and "mae_scaled" in parts
    assert abs(float(parts["mae"]) - float(raw)) < 1e-6
    assert abs(float(loss) - float(raw) * PREDICTION_MAE_LOSS_SCALE) < 1e-5
    assert PREDICTION_MAE_LOSS_SCALE >= 8.0
    loss2, parts2 = downstream_task_loss(
        {"pred_patches": pred, "patch_targets_norm": target, "mae_mask": mask, "z": torch.zeros(2, 4)},
        {},
        "prediction",
    )
    assert abs(float(parts2["mae"]) - float(raw)) < 1e-6
    assert abs(float(loss2) - float(raw) * PREDICTION_MAE_LOSS_SCALE) < 1e-5


def test_prediction_uses_suffix_mask_when_mae_mask_empty() -> None:
    pred = torch.zeros(2, 4, 2, 4)
    target = torch.ones_like(pred)
    mae_mask = torch.zeros(2, 4, dtype=torch.bool)
    suffix = torch.zeros(2, 4, dtype=torch.bool)
    suffix[:, 2:] = True
    target_mask = suffix.clone()
    outputs = {
        "pred_patches": pred,
        "patch_targets_norm": target,
        "mae_mask": mae_mask,
        "suffix_mask": suffix,
        "span_mask": torch.zeros_like(mae_mask),
        "target_mask": target_mask,
        "z": torch.zeros(2, 4),
    }
    mask = resolve_recon_mask(outputs, "prediction")
    assert mask is not None and bool(mask[:, 2:].all()) and not bool(mask[:, :2].any())
    loss, parts = downstream_task_loss(outputs, {}, "prediction")
    assert float(parts["mae"]) > 0.0
    assert torch.isfinite(loss)


def test_imputation_uses_span_mask_not_mae_mask() -> None:
    pred = torch.zeros(2, 4, 2, 4)
    target = torch.ones_like(pred)
    mae_mask = torch.zeros(2, 4, dtype=torch.bool)
    span = torch.zeros(2, 4, dtype=torch.bool)
    span[:, 1:3] = True
    outputs = {
        "pred_patches": pred,
        "patch_targets": target,
        "mae_mask": mae_mask,
        "span_mask": span,
        "target_mask": span,
        "z": torch.zeros(2, 4),
    }
    loss, parts = downstream_task_loss(outputs, {}, "imputation")
    assert float(parts["mae"]) > 0.0
    mae_only, _ = downstream_task_loss(
        {**outputs, "span_mask": torch.zeros_like(span), "target_mask": torch.zeros_like(span)},
        {},
        "imputation",
    )
    assert float(mae_only) == 0.0


def test_pretrain_mae_still_uses_mae_mask() -> None:
    pred = torch.zeros(2, 4, 2, 4)
    target = torch.ones_like(pred)
    mae_mask = torch.zeros(2, 4, dtype=torch.bool)
    mae_mask[:, :2] = True
    span = torch.zeros(2, 4, dtype=torch.bool)
    span[:, 2:] = True
    losses = foundation_pretrain_losses(
        {
            "recon_norm": pred,
            "patch_targets_norm": target,
            "mae_pred": pred,
            "patch_targets": target,
            "mae_mask": mae_mask,
            "span_mask": span,
            "target_mask": mae_mask | span,
            "global_phys_pred": torch.zeros(2, 5),
            "global_phys_target": torch.zeros(2, 5),
        },
        include={"mae", "impute"},
    )
    assert float(losses["mae"]) > 0.0
    assert float(losses["impute"]) > 0.0


def test_structure_phase_only_for_complex_pair() -> None:
    torch.manual_seed(0)
    pred = torch.randn(2, 3, 2, 8)
    target = torch.randn(2, 3, 2, 8)
    mask = torch.ones(2, 3, dtype=torch.bool)
    _, with_phase = structure_preserving_loss(pred, target, mask, complex_pair=True)
    _, no_phase = structure_preserving_loss(pred, target, mask, complex_pair=False)
    assert float(with_phase["structure_phase"]) > 0.0
    assert float(no_phase["structure_phase"]) == 0.0
    assert float(no_phase["structure_time"]) > 0.0
    assert float(no_phase["structure_spectrum"]) > 0.0


def test_uti_query_without_teacher_is_zero_not_recon_fallback() -> None:
    from resmamba_signal_model.training.losses import reconstruction_monitor_loss

    pred = torch.zeros(2, 4, 2, 4)
    target = torch.ones_like(pred)
    mask = torch.ones(2, 4, dtype=torch.bool)
    losses = foundation_pretrain_losses(
        {
            "recon_norm": pred,
            "patch_targets_norm": target,
            "mae_pred": pred,
            "patch_targets": target,
            "mae_mask": mask,
            "span_mask": mask,
            "target_mask": mask,
            "uti_query": torch.randn(2, 4, 8),
            "global_phys_pred": torch.zeros(2, 5),
            "global_phys_target": torch.zeros(2, 5),
        },
        include={"mae", "impute", "structure", "uti_query"},
    )
    assert float(losses["uti_query"]) == 0.0
    assert float(losses["structure_time"]) > 0.0
    recon = reconstruction_monitor_loss(losses)
    assert float(recon) == pytest.approx(
        float(losses["mae"] + 0.2 * losses["impute"] + 0.2 * losses["structure"])
    )


def test_clustering_train_loss_ignores_global_label_id() -> None:
    torch.manual_seed(0)
    z = F.normalize(torch.randn(8, 16), dim=-1)
    proto = F.normalize(torch.randn(6, 16), dim=-1)
    logits = z @ proto.t() / 0.1
    z2 = F.normalize(z + 0.02 * torch.randn_like(z), dim=-1)
    logits2 = z2 @ proto.t() / 0.1
    packed = {
        "z": z,
        "cluster_embedding": z,
        "cluster_logits": logits,
        "cluster_embedding_view2": z2,
        "cluster_logits_view2": logits2,
        "cluster_prototypes": proto,
    }
    loss_a, parts_a = downstream_task_loss(
        packed,
        {"global_label_id": torch.arange(8)},
        "clustering",
        recon_weight=0.0,
        phys_weight=0.0,
        domain_weight=0.0,
    )
    loss_b, parts_b = downstream_task_loss(
        packed,
        {"global_label_id": torch.zeros(8, dtype=torch.long)},
        "clustering",
        recon_weight=0.0,
        phys_weight=0.0,
        domain_weight=0.0,
    )
    assert "cluster_unsupervised" in parts_a
    assert "cluster_align" not in parts_a
    assert "cluster_consistency" in parts_a and "cluster_utilization" in parts_a
    assert torch.allclose(loss_a, loss_b)
    assert torch.allclose(parts_a["cluster_unsupervised"], parts_b["cluster_unsupervised"])


def test_supervised_clustering_opt_in_still_uses_labels() -> None:
    torch.manual_seed(1)
    z = F.normalize(torch.randn(8, 8), dim=-1)
    proto = F.normalize(torch.randn(4, 8), dim=-1)
    logits = z @ proto.t() / 0.1
    packed = {"z": z, "cluster_embedding": z, "cluster_logits": logits}
    loss_a, parts_a = downstream_task_loss(
        packed,
        {"global_label_id": torch.tensor([0, 0, 0, 0, 1, 1, 1, 1])},
        "clustering",
        recon_weight=0.0,
        phys_weight=0.0,
        domain_weight=0.0,
        supervised_clustering=True,
    )
    loss_b, _ = downstream_task_loss(
        packed,
        {"global_label_id": torch.arange(8)},
        "clustering",
        recon_weight=0.0,
        phys_weight=0.0,
        domain_weight=0.0,
        supervised_clustering=True,
    )
    assert "cluster_align" in parts_a
    assert not torch.allclose(loss_a, loss_b)



def test_mae_per_sample_clamp_limits_outlier() -> None:
    pred = torch.zeros(2, 4, 2, 4)
    target = torch.zeros_like(pred)
    target[0] = 1.0e6
    mask = torch.ones(2, 4, dtype=torch.bool)
    loss = mae_reconstruction_loss(pred, target, mask)
    assert float(loss) <= 5.0 + 1.0e-5
    assert torch.isfinite(loss)


def test_structure_spectrum_bounded_on_huge_magnitude() -> None:
    pred = torch.zeros(2, 3, 2, 8)
    target = torch.zeros_like(pred)
    target[..., 0, :] = 1.0e4
    mask = torch.ones(2, 3, dtype=torch.bool)
    total, parts = structure_preserving_loss(pred, target, mask, complex_pair=True)
    assert float(parts["structure_spectrum"]) <= 10.0
    assert float(parts["structure_time"]) <= 10.0
    assert float(total) <= 10.0
    assert torch.isfinite(total)

def test_structure_total_excludes_phase_term() -> None:
    torch.manual_seed(4)
    pred = torch.randn(2, 3, 2, 8)
    target = torch.randn(2, 3, 2, 8)
    mask = torch.ones(2, 3, dtype=torch.bool)
    total, parts = structure_preserving_loss(pred, target, mask, complex_pair=True)
    approx = float(parts["structure_time"] + parts["structure_spectrum"])
    assert abs(float(total) - approx) <= 1.0e-5
    assert float(parts["structure_phase"]) > 0.1
    assert float(total) < approx + float(parts["structure_phase"]) - 1.0e-5



def test_clamp_loss_replaces_nonfinite() -> None:
    assert float(_clamp_loss(torch.tensor(float("nan")))) == 0.0
    assert float(_clamp_loss(torch.tensor(float("inf")))) <= 10.0
    assert torch.isfinite(_clamp_loss(torch.tensor(float("-inf"))))


def test_sinkhorn_large_logits_are_finite_and_normalized() -> None:
    logits = torch.full((8, 16), 200.0)
    assign = sinkhorn_balanced_assignment(logits)
    assert torch.isfinite(assign).all()
    assert torch.allclose(assign.sum(dim=-1), torch.ones(8), atol=1.0e-3)


def test_unsupervised_clustering_zero_vectors_and_huge_logits_finite() -> None:
    z = torch.zeros(8, 16)
    logits = torch.full((8, 6), 1.0e4)
    loss, parts = unsupervised_clustering_loss(z, logits)
    assert torch.isfinite(loss)
    assert all(torch.isfinite(value).all() for value in parts.values())


def test_unsupervised_clustering_balance_mix_softens_sinkhorn() -> None:
    torch.manual_seed(0)
    z = F.normalize(torch.randn(32, 8), dim=-1)
    logits = torch.randn(32, 16) * 3.0
    _, hard = unsupervised_clustering_loss(z, logits, balance_mix=1.0, utilization_weight=0.0)
    _, soft = unsupervised_clustering_loss(z, logits, balance_mix=0.0, utilization_weight=0.0)
    assert torch.isfinite(hard["cluster_consistency"])
    assert torch.isfinite(soft["cluster_consistency"])
    # 纯 softmax 目标不应强制接近均匀占用
    usage_soft = F.softmax(logits, dim=-1).mean(dim=0)
    assert float(usage_soft.max() / usage_soft.clamp_min(1e-8).min()) > 1.5


def test_should_log_loss_part_skips_nonpositive_weights() -> None:
    weights = {"mae": 1.0, "domain": 0.0, "latent": -0.1, "vicreg": 0.25}
    assert should_log_loss_part("mae", weights)
    assert should_log_loss_part("loss/mae", weights)
    assert should_log_loss_part("modulation/total", weights)
    assert not should_log_loss_part("domain", weights)
    assert not should_log_loss_part("val/domain", weights)
    assert not should_log_loss_part("latent", weights)
    assert should_log_loss_part("vicreg", weights)
    # 未配置权重的诊断子项仍记录
    assert should_log_loss_part("structure_time", weights)
    assert not should_log_loss_part("modulation/_tokens", weights)


def test_weighted_pretrain_omits_zero_weight_parts() -> None:
    pred = torch.zeros(2, 4, 2, 8)
    target = torch.randn_like(pred)
    mask = torch.ones(2, 4, dtype=torch.bool)
    outputs = {
        "mae_pred": pred,
        "recon_norm": pred,
        "patch_targets": target,
        "patch_targets_norm": target,
        "mae_mask": mask,
        "target_mask": mask,
        "recon_mask": mask,
        "global_phys_pred": torch.zeros(2, 4),
        "global_phys_target": torch.zeros(2, 4),
        "domain_logits": torch.randn(2, 3),
        "z_enc": torch.randn(2, 8),
        "dataset_id": torch.tensor([0, 1]),
    }
    batch = {"dataset_id": outputs["dataset_id"]}
    _, parts = weighted_pretrain_loss(
        outputs,
        batch,
        {"mae": 1.0, "domain": 0.0, "latent": 0.0, "vicreg": 0.25, "readout": 0.1},
    )
    assert "mae" in parts and "vicreg" in parts and "readout" in parts
    assert "domain" not in parts
    assert "latent" not in parts


def test_vicreg_not_hard_clamped_and_has_grad() -> None:
    torch.manual_seed(0)
    # 近常数表征：旧实现会被 clamp 到 10 且梯度为 0
    base = torch.ones(32, 640)
    noise = 0.01 * torch.randn(32, 640)
    z = (base + noise).detach().requires_grad_(True)
    loss = vicreg_loss(z)
    assert torch.isfinite(loss)
    assert float(loss.detach()) > 10.0
    loss.backward()
    assert z.grad is not None and float(z.grad.norm()) > 0.0
    # 宽表征协方差经 /d^2 后尺度可控
    z2 = torch.randn(16, 640, requires_grad=True)
    loss2 = vicreg_loss(z2)
    assert torch.isfinite(loss2)
    assert float(loss2.detach()) < 100.0
    loss2.backward()
    assert z2.grad is not None and float(z2.grad.norm()) > 0.0
