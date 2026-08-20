from __future__ import annotations

import warnings

import numpy as np
import torch

from resmamba_signal_model.training.metrics import (
    classification_epoch_scores,
    clustering_epoch_scores,
    clustering_overseg_scores,
    macro_f1,
    mean_clusters_per_class,
    nmi_score,
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


def test_clustering_epoch_scores_per_dataset() -> None:
    preds = np.array([0, 0, 1, 1, 1, 0])
    labels = np.array([7, 7, 8, 8, 8, 9])
    dataset_ids = np.array([0, 0, 1, 1, 1, 1])
    report = clustering_epoch_scores(preds, labels, dataset_ids, dataset_names={0: "adsb2", 1: "wifi150"})
    assert "nmi" in report
    assert set(report["datasets"]) == {"adsb2", "wifi150"}
    assert report["mean_nmi"] == (report["datasets"]["adsb2"]["nmi"] + report["datasets"]["wifi150"]["nmi"]) / 2
    assert "ari" in report
    assert -1.0 <= report["ari"] <= 1.0
    assert report["n_active_clusters"] >= 1.0
    assert report["mean_clusters_per_class"] >= 1.0
    assert 0.0 <= report["completeness"] <= 1.0
    assert 0.0 <= report["nmi_merged"] <= 1.0


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
