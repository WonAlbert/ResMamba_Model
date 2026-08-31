from __future__ import annotations

import warnings

import numpy as np
import pytest
import torch

from resmamba_signal_model.training.logging_utils import format_val_epoch_metrics
from resmamba_signal_model.training.metrics import (
    classification_epoch_scores,
    clustering_epoch_scores,
    clustering_overseg_scores,
    macro_f1,
    macro_miss_rate,
    masked_patch_mae,
    masked_patch_mse,
    mean_clusters_per_class,
    nmi_score,
    reconstruction_epoch_scores,
)


def test_classification_epoch_scores_overall_and_per_dataset() -> None:
    preds = torch.tensor([0, 0, 1, 1, 1, 0])
    labels = torch.tensor([0, 0, 1, 0, 1, 1])
    dataset_ids = torch.tensor([2, 2, 2, 3, 3, 3])
    names = {2: "rml2016_10a", 3: "rml2018_1a"}
    report = classification_epoch_scores(preds, labels, dataset_ids, dataset_names=names)
    assert report["acc"] == 4 / 6
    assert report["n"] == 6
    assert set(report["datasets"]) == {"rml2016_10a", "rml2018_1a"}
    assert report["datasets"]["rml2016_10a"]["acc"] == 1.0
    assert report["datasets"]["rml2018_1a"]["acc"] == 1 / 3
    assert report["mean_acc"] == (1.0 + 1 / 3) / 2
    assert 0.0 <= report["f1"] <= 1.0
    assert 0.0 <= report["mean_f1"] <= 1.0
    assert report["miss_rate"] == pytest.approx(macro_miss_rate(preds, labels))


def test_macro_miss_rate_matches_one_minus_recall() -> None:
    preds = torch.tensor([0, 0, 1, 1])
    labels = torch.tensor([0, 1, 1, 1])
    from sklearn.metrics import recall_score

    recall = recall_score(labels.numpy(), preds.numpy(), average="macro", zero_division=0)
    assert macro_miss_rate(preds, labels) == pytest.approx(1.0 - recall)


def test_clustering_epoch_scores_per_dataset() -> None:
    preds = np.array([0, 0, 1, 1, 1, 0])
    labels = np.array([7, 7, 8, 8, 8, 9])
    dataset_ids = np.array([0, 0, 1, 1, 1, 1])
    report = clustering_epoch_scores(preds, labels, dataset_ids, dataset_names={0: "adsb2", 1: "wifi150"})
    assert "nmi" in report
    assert "acc" in report
    assert 0.0 <= report["acc"] <= 1.0
    assert set(report["datasets"]) == {"adsb2", "wifi150"}
    assert report["mean_nmi"] == (report["datasets"]["adsb2"]["nmi"] + report["datasets"]["wifi150"]["nmi"]) / 2
    assert report["mean_nmi_emitter"] == report["mean_nmi"]
    assert "mean_nmi_modulation" not in report
    assert "ari" in report
    assert -1.0 <= report["ari"] <= 1.0
    assert report["n_active_clusters"] >= 1.0
    assert report["mean_clusters_per_class"] >= 1.0
    assert 0.0 <= report["completeness"] <= 1.0
    assert 0.0 <= report["nmi_merged"] <= 1.0


def test_clustering_family_nmi_split() -> None:
    preds = np.array([0, 0, 1, 1, 2, 2, 3, 3])
    labels = np.array([1, 1, 2, 2, 7, 7, 8, 8])
    dataset_ids = np.array([0, 0, 1, 1, 2, 2, 3, 3])
    report = clustering_epoch_scores(
        preds,
        labels,
        dataset_ids,
        dataset_names={0: "rml2016_10a", 1: "rml2016_04c", 2: "wisig", 3: "adsb2"},
    )
    assert "mean_nmi_modulation" in report
    assert "mean_nmi_emitter" in report
    mod_mean = (
        report["datasets"]["rml2016_10a"]["nmi"] + report["datasets"]["rml2016_04c"]["nmi"]
    ) / 2
    sei_mean = (report["datasets"]["wisig"]["nmi"] + report["datasets"]["adsb2"]["nmi"]) / 2
    assert report["mean_nmi_modulation"] == pytest.approx(mod_mean)
    assert report["mean_nmi_emitter"] == pytest.approx(sei_mean)


def test_oversegmentation_raises_mean_clusters_per_class() -> None:
    # 一类被拆成多簇
    labels = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    overseg = np.array([0, 1, 2, 3, 4, 5, 6, 7])
    matched = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    assert mean_clusters_per_class(overseg, labels) == 4.0
    assert mean_clusters_per_class(matched, labels) == 1.0
    bad = clustering_overseg_scores(overseg, labels)
    good = clustering_overseg_scores(matched, labels)
    assert bad["completeness"] < good["completeness"]
    assert bad["nmi_merged"] >= 0.0


def test_high_cardinality_labels_do_not_warn() -> None:
    labels = np.arange(8)
    preds = np.arange(8)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert 0.0 <= macro_f1(preds, labels) <= 1.0
        assert 0.0 <= nmi_score(preds, labels) <= 1.0


def test_masked_patch_mse_and_mae() -> None:
    pred = torch.tensor([[[[0.0, 1.0], [2.0, 3.0]], [[1.0, 1.0], [1.0, 1.0]]]])
    target = torch.tensor([[[[0.0, 0.0], [0.0, 0.0]], [[2.0, 0.0], [0.0, 2.0]]]])
    mask = torch.tensor([[True, False]])
    mse = float(masked_patch_mse(pred, target, mask))
    mae = float(masked_patch_mae(pred, target, mask))
    assert mse == pytest.approx(3.5)
    assert mae == pytest.approx(1.5)
    assert mse >= mae


def test_reconstruction_epoch_scores_mse_mae() -> None:
    info = reconstruction_epoch_scores(0.2, 0.1, 4, {"a": {"mse": 0.3, "mae": 0.15, "n": 2.0}})
    assert info["mse"] == pytest.approx(0.2)
    assert info["mae"] == pytest.approx(0.1)
    assert info["mean_mse"] == pytest.approx(0.3)
    assert info["mean_mae"] == pytest.approx(0.15)


def test_format_val_epoch_metrics_task_lines() -> None:
    text = format_val_epoch_metrics(
        {"val/monitor": 0.3},
        epoch=1,
        task_report={
            "tx_modulation": {
                "kind": "classification",
                "acc": 0.8,
                "f1": 0.75,
                "miss_rate": 0.2,
                "n": 10,
                "datasets": {
                    "rml2016_10a": {"acc": 0.9, "f1": 0.88, "miss_rate": 0.1, "n": 5},
                    "rml2016_10b": {"acc": 0.7, "f1": 0.62, "miss_rate": 0.3, "n": 5},
                },
            },
            "prediction": {
                "kind": "reconstruction",
                "mse": 0.05,
                "mae": 0.03,
                "n": 8,
                "datasets": {"adsb2": {"mse": 0.04, "mae": 0.02, "n": 8}},
            },
            "ld_clustering": {
                "kind": "clustering",
                "acc": 0.6,
                "nmi": 0.55,
                "n": 12,
                "datasets": {"radchar": {"acc": 0.65, "nmi": 0.5, "n": 12}},
            },
        },
    )
    assert "acc=0.8" in text
    assert "miss_rate=0.2" in text
    assert "rml2016_10a  acc=0.9" in text
    assert "rml2016_10b" in text
    assert "prediction  mse=0.05  mae=0.03" in text
    assert "adsb2  mse=0.04  mae=0.02" in text
    assert "ld_clustering  acc=0.6  nmi=0.55" in text
    assert "radchar  acc=0.65  nmi=0.5" in text
