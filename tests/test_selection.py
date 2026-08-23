from resmamba_signal_model.training.selection import (
    compute_selection_score,
    multitask_metric_geomean,
    resolve_checkpoint_monitor_name,
    resolve_selection_metric_name,
    specialist_relative_geomean,
)
import pytest


def test_emitter_default_selection_metric() -> None:
    assert resolve_selection_metric_name({}, stage="downstream", task="emitter") == "per_dataset_macro_acc"
    assert resolve_selection_metric_name({}, stage="downstream", task="modulation") == "f1"


def test_per_dataset_macro_acc() -> None:
    metrics = {
        "loss": 2.5,
        "acc": 0.16,
        "acc/adsb2": 0.20,
        "acc/wifi150": 0.18,
        "acc/radar_emitters": 0.40,
    }
    score = compute_selection_score(metrics, "per_dataset_macro_acc")
    assert abs(score - (0.20 + 0.18 + 0.40) / 3) < 1e-6


def test_val_loss_selection_is_raw_loss() -> None:
    metrics = {"loss": 2.0, "acc": 0.1}
    assert compute_selection_score(metrics, "val_loss") == 2.0


def test_joint_default_selection_is_geomean() -> None:
    assert resolve_selection_metric_name({}, stage="downstream", task="all") == "multitask_geomean"
    assert resolve_selection_metric_name({}, stage="stage2", task="all") == "multitask_geomean"
    assert resolve_selection_metric_name({}, stage="joint", task="all") == "specialist_geomean"
    assert resolve_selection_metric_name({}, stage="continual", task="all") == "multitask_geomean"
    assert resolve_selection_metric_name({"selection_metric": "specialist_geomean"}, stage="downstream", task="all") == "specialist_geomean"


def test_specialist_relative_geomean_clips_and_warns() -> None:
    score, details = specialist_relative_geomean(
        {
            "modulation": {"f1": 0.99},
            "emitter": {"acc": 0.50},
        },
        {"modulation": 1.0, "emitter": 1.0},
        drop_warn_threshold=0.05,
    )
    assert details["warnings"] == ["emitter"]
    assert 0.5 < score < 0.99


def test_multitask_geomean_is_unweighted() -> None:
    score, details = multitask_metric_geomean(
        {
            "modulation": {"f1": 0.81},
            "emitter": {"per_dataset_macro_acc": 0.25, "acc": 0.4},
        }
    )
    assert details["score"] == score
    assert 0.4 < score < 0.81


def test_clustering_selects_within_domain_nmi() -> None:
    assert resolve_selection_metric_name({}, stage="stage2", task="clustering") == "nmi_within_domain"
    assert resolve_checkpoint_monitor_name({}, stage="stage3", task="clustering") == "val/nmi_within_domain"
    assert resolve_checkpoint_monitor_name(
        {"selection_metric": "mean_nmi"},
        stage="stage3",
        task="clustering",
    ) == "val/nmi_within_domain"
    fallback = compute_selection_score({"nmi": 0.4}, "nmi_within_domain")
    assert fallback == pytest.approx(0.4)
    primary = compute_selection_score({"nmi": 0.4, "nmi_within_domain": 0.2}, "mean_nmi")
    assert primary == pytest.approx(0.2)


def test_stage2_default_checkpoint_monitor_name() -> None:
    assert resolve_checkpoint_monitor_name({}, stage="stage2", task="all") == "val/multitask_geomean"
    assert resolve_checkpoint_monitor_name({}, stage="joint", task="all") == "val/specialist_geomean"
    assert resolve_checkpoint_monitor_name(
        {"checkpoint_monitor": "val/f1_modulation"},
        stage="stage2",
        task="all",
    ) == "val/f1_modulation"
