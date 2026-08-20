from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import adjusted_rand_score, f1_score, normalized_mutual_info_score


def _to_numpy(x: torch.Tensor | np.ndarray) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def _valid_mask(labels: torch.Tensor | np.ndarray, mask: torch.Tensor | np.ndarray | None = None) -> np.ndarray:
    labels_np = _to_numpy(labels)
    valid = labels_np >= 0
    if mask is not None:
        valid = valid & _to_numpy(mask).astype(bool)
    return valid


def accuracy(preds: torch.Tensor | np.ndarray, labels: torch.Tensor | np.ndarray, mask: torch.Tensor | np.ndarray | None = None) -> float:
    preds_np = _to_numpy(preds)
    labels_np = _to_numpy(labels)
    valid = _valid_mask(labels_np, mask)
    if valid.sum() == 0:
        return 0.0
    return float((preds_np[valid] == labels_np[valid]).mean())


def _sklearn_metric(fn, *args, **kwargs):
    """个体/聚类标签基数高时 sklearn 会误报回归任务，指标本身仍有效。"""
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="The number of unique classes is greater than 50%",
            category=UserWarning,
        )
        return fn(*args, **kwargs)


def macro_f1(preds: torch.Tensor | np.ndarray, labels: torch.Tensor | np.ndarray, mask: torch.Tensor | np.ndarray | None = None) -> float:
    preds_np = _to_numpy(preds)
    labels_np = _to_numpy(labels)
    valid = _valid_mask(labels_np, mask)
    if valid.sum() == 0:
        return 0.0
    return float(_sklearn_metric(f1_score, labels_np[valid], preds_np[valid], average="macro", zero_division=0))


def nmi_score(pred_clusters: torch.Tensor | np.ndarray, true_labels: torch.Tensor | np.ndarray, mask: torch.Tensor | np.ndarray | None = None) -> float:
    pred_np = _to_numpy(pred_clusters)
    true_np = _to_numpy(true_labels)
    valid = _valid_mask(true_np, mask)
    if valid.sum() < 2:
        return 0.0
    return float(_sklearn_metric(normalized_mutual_info_score, true_np[valid], pred_np[valid]))


def ari_score(pred_clusters: torch.Tensor | np.ndarray, true_labels: torch.Tensor | np.ndarray, mask: torch.Tensor | np.ndarray | None = None) -> float:
    pred_np = _to_numpy(pred_clusters)
    true_np = _to_numpy(true_labels)
    valid = _valid_mask(true_np, mask)
    if valid.sum() < 2:
        return 0.0
    return float(_sklearn_metric(adjusted_rand_score, true_np[valid], pred_np[valid]))


def dataset_display_name(dataset_id: int, names: dict[int, str] | None = None) -> str:
    if names and int(dataset_id) in names:
        return names[int(dataset_id)]
    return f"dataset_{int(dataset_id)}"


def nmi_within_domain(
    pred_clusters: torch.Tensor | np.ndarray,
    true_labels: torch.Tensor | np.ndarray,
    dataset_id: torch.Tensor | np.ndarray,
) -> float:
    pred_np = _to_numpy(pred_clusters)
    true_np = _to_numpy(true_labels)
    domain_np = _to_numpy(dataset_id)
    scores: list[float] = []
    for domain in np.unique(domain_np):
        mask = domain_np == domain
        if mask.sum() < 2:
            continue
        scores.append(nmi_score(pred_np[mask], true_np[mask]))
    if not scores:
        return 0.0
    return float(np.mean(scores))


def classification_epoch_scores(
    preds: torch.Tensor | np.ndarray,
    labels: torch.Tensor | np.ndarray,
    dataset_ids: torch.Tensor | np.ndarray | None = None,
    *,
    dataset_names: dict[int, str] | None = None,
) -> dict:
    """整体 acc / macro-F1，以及各 dataset_id 的 acc / F1 与数据集宏平均。"""
    preds_np = _to_numpy(preds)
    labels_np = _to_numpy(labels)
    overall_acc = accuracy(preds_np, labels_np)
    overall_f1 = macro_f1(preds_np, labels_np)
    n_valid = int(_valid_mask(labels_np).sum())
    datasets: dict[str, dict[str, float]] = {}
    if dataset_ids is not None:
        domain_np = _to_numpy(dataset_ids)
        for domain in np.unique(domain_np):
            did = int(domain)
            if did < 0:
                continue
            mask = domain_np == domain
            n = int(_valid_mask(labels_np[mask]).sum())
            if n == 0:
                continue
            name = dataset_display_name(did, dataset_names)
            datasets[name] = {
                "acc": accuracy(preds_np[mask], labels_np[mask]),
                "f1": macro_f1(preds_np[mask], labels_np[mask]),
                "n": float(n),
            }
    mean_acc = float(np.mean([row["acc"] for row in datasets.values()])) if datasets else overall_acc
    mean_f1 = float(np.mean([row["f1"] for row in datasets.values()])) if datasets else overall_f1
    return {
        "kind": "classification",
        "acc": overall_acc,
        "f1": overall_f1,
        "mean_acc": mean_acc,
        "mean_f1": mean_f1,
        "n": float(n_valid),
        "datasets": datasets,
    }


def clustering_epoch_scores(
    preds: torch.Tensor | np.ndarray,
    labels: torch.Tensor | np.ndarray,
    dataset_ids: torch.Tensor | np.ndarray | None = None,
    *,
    dataset_names: dict[int, str] | None = None,
) -> dict:
    preds_np = _to_numpy(preds)
    labels_np = _to_numpy(labels)
    overall = nmi_score(preds_np, labels_np)
    overall_ari = ari_score(preds_np, labels_np)
    n_valid = int(_valid_mask(labels_np).sum())
    datasets: dict[str, dict[str, float]] = {}
    if dataset_ids is not None:
        domain_np = _to_numpy(dataset_ids)
        for domain in np.unique(domain_np):
            did = int(domain)
            if did < 0:
                continue
            mask = domain_np == domain
            n = int(_valid_mask(labels_np[mask]).sum())
            if n < 2:
                continue
            name = dataset_display_name(did, dataset_names)
            datasets[name] = {
                "nmi": nmi_score(preds_np[mask], labels_np[mask]),
                "ari": ari_score(preds_np[mask], labels_np[mask]),
                "n": float(n),
            }
    mean_nmi = float(np.mean([row["nmi"] for row in datasets.values()])) if datasets else overall
    mean_ari = float(np.mean([row["ari"] for row in datasets.values()])) if datasets else overall_ari
    return {
        "kind": "clustering",
        "nmi": overall,
        "ari": overall_ari,
        "mean_nmi": mean_nmi,
        "mean_ari": mean_ari,
        "n": float(n_valid),
        "datasets": datasets,
    }


def reconstruction_epoch_scores(
    overall_ssim: float,
    overall_mse: float,
    n: int,
    per_dataset: dict[str, dict[str, float]] | None = None,
) -> dict:
    datasets = dict(per_dataset or {})
    mean_ssim = float(np.mean([row["ssim"] for row in datasets.values()])) if datasets else float(overall_ssim)
    mean_mse = float(np.mean([row["mse"] for row in datasets.values()])) if datasets else float(overall_mse)
    return {
        "kind": "reconstruction",
        "ssim": float(overall_ssim),
        "mse": float(overall_mse),
        "mean_ssim": mean_ssim,
        "mean_mse": mean_mse,
        "n": float(n),
        "datasets": datasets,
    }


def reconstruction_eval_pair(
    outputs: dict,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """优先用 RevIN 归一化空间的重建对，避免原始幅度把 SSIM/MSE 撑爆。"""
    pred = outputs.get("recon_norm")
    target = outputs.get("patch_targets_norm")
    if pred is None or target is None:
        pred = outputs["mae_pred"]
        target = outputs["patch_targets"]
    mask = outputs.get("recon_mask")
    if mask is None:
        mask = outputs.get("target_mask")
    if mask is None:
        mask = outputs.get("suffix_mask")
    if mask is None:
        mask = outputs.get("span_mask")
    if mask is None:
        mask = outputs.get("mae_mask")
    if mask is None:
        mask = outputs.get("patch_mask")
    return pred, target, mask


def masked_patch_mse(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """对 patch 重建做 masked MSE；无 mask 时对全部元素取均值。"""
    n = min(pred.shape[1], target.shape[1])
    pred = pred[:, :n].float()
    target = target[:, :n].float()
    err = (pred - target).square()
    while err.ndim > 2:
        err = err.mean(dim=-1)
    if mask is None:
        return err.mean()
    mask = mask[:, :n]
    err = err.masked_fill(~mask, 0.0)
    denom = mask.sum().clamp_min(1).to(dtype=err.dtype)
    return err.sum() / denom


def _ssim_1d_batched(x: torch.Tensor, y: torch.Tensor, window: int = 11, sigma: float = 1.5) -> torch.Tensor:
    """``x,y`` 形状 ``[N,L]``，返回每条 1D 序列的 SSIM ``[N]``。"""
    x = x.float()
    y = y.float()
    length = int(x.shape[-1])
    if length < window:
        err = (x - y).square().mean(dim=-1)
        return 1.0 - err / (err + 1.0)
    coords = torch.arange(window, device=x.device, dtype=x.dtype) - window // 2
    kernel = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    kernel = (kernel / kernel.sum()).view(1, 1, -1)

    def filter1d(t: torch.Tensor) -> torch.Tensor:
        return F.conv1d(t.unsqueeze(1), kernel, padding=window // 2).squeeze(1)

    data_range = torch.maximum((x.amax(dim=-1) - x.amin(dim=-1)).abs(), (y.amax(dim=-1) - y.amin(dim=-1)).abs()).clamp_min(1.0)
    c1 = ((0.01 * data_range) ** 2).unsqueeze(-1)
    c2 = ((0.03 * data_range) ** 2).unsqueeze(-1)
    mu_x, mu_y = filter1d(x), filter1d(y)
    sigma_x = (filter1d(x * x) - mu_x ** 2).clamp_min(0.0)
    sigma_y = (filter1d(y * y) - mu_y ** 2).clamp_min(0.0)
    sigma_xy = filter1d(x * y) - mu_x * mu_y
    num = (2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)
    den = (mu_x ** 2 + mu_y ** 2 + c1) * (sigma_x + sigma_y + c2)
    ssim_map = (num / den.clamp_min(1.0e-8)).clamp(-1.0, 1.0)
    return ssim_map.mean(dim=-1)


def _ssim_1d(x: torch.Tensor, y: torch.Tensor, window: int = 11, sigma: float = 1.5) -> torch.Tensor:
    return _ssim_1d_batched(x.unsqueeze(0), y.unsqueeze(0), window=window, sigma=sigma).squeeze(0)


def ssim_iq(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> float:
    """Compute mean SSIM over I/Q channels on masked patch reconstructions."""
    total, count = ssim_iq_accumulate(pred, target, mask)
    if count == 0:
        return 0.0
    return float(total / count)


def ssim_iq_accumulate(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> tuple[float, int]:
    """返回 (SSIM 分数之和, patch×channel 计数)，用于按样本加权聚合。"""
    if pred.ndim == 4:
        b, p, c, l = pred.shape
        pred = pred.reshape(b * p, c, l)
        target = target.reshape(b * p, c, l)
        if mask is not None:
            mask = mask.reshape(b * p)
    if pred.ndim != 3 or pred.shape[1] != 2:
        raise ValueError(f"期望 pred 形状 [N,2,L] 或 [B,P,2,L]，实际 {tuple(pred.shape)}")
    if mask is not None:
        pred = pred[mask]
        target = target[mask]
    if pred.shape[0] == 0:
        return 0.0, 0
    n = int(pred.shape[0])
    x = pred.reshape(n * 2, pred.shape[-1]).float()
    y = target.reshape(n * 2, target.shape[-1]).float()
    scores = _ssim_1d_batched(x, y)
    return float(scores.sum().item()), int(scores.numel())


def _finite_binary(y_unknown: torch.Tensor | np.ndarray, scores: torch.Tensor | np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    y = _to_numpy(y_unknown).astype(bool).reshape(-1)
    s = _to_numpy(scores).astype(np.float64).reshape(-1)
    n = min(y.size, s.size)
    y, s = y[:n], s[:n]
    keep = np.isfinite(s)
    return y[keep], s[keep]


def auroc_score(y_unknown: torch.Tensor | np.ndarray, scores: torch.Tensor | np.ndarray) -> float:
    from sklearn.metrics import roc_auc_score

    y, s = _finite_binary(y_unknown, scores)
    if y.size == 0 or y.min() == y.max():
        return float("nan")
    return float(roc_auc_score(y.astype(int), s))


def aupr_score(y_unknown: torch.Tensor | np.ndarray, scores: torch.Tensor | np.ndarray) -> float:
    from sklearn.metrics import average_precision_score

    y, s = _finite_binary(y_unknown, scores)
    if y.size == 0 or y.min() == y.max():
        return float("nan")
    return float(average_precision_score(y.astype(int), s))


def fpr95_score(y_unknown: torch.Tensor | np.ndarray, scores: torch.Tensor | np.ndarray, *, tpr_level: float = 0.95) -> float:
    """未知检测：TPR 达到 ``tpr_level`` 时的 FPR（越高分越像未知）。"""
    y, s = _finite_binary(y_unknown, scores)
    n_pos = float(y.sum())
    n_neg = float((~y).sum())
    if y.size == 0 or n_pos <= 0 or n_neg <= 0:
        return float("nan")
    order = np.argsort(-s)
    y_sorted = y[order]
    tp = np.cumsum(y_sorted)
    fp = np.cumsum(~y_sorted)
    tpr = tp / n_pos
    fpr = fp / n_neg
    hits = np.where(tpr >= float(tpr_level))[0]
    if hits.size == 0:
        return float(fpr[-1])
    return float(fpr[hits[0]])


def oscr_score(
    y_unknown: torch.Tensor | np.ndarray,
    scores: torch.Tensor | np.ndarray,
    *,
    correct: torch.Tensor | np.ndarray | None = None,
) -> float:
    """AUOSCR：按未知分从低到高逐步判为已知，对 CCR–FPR 曲线积分。"""
    y, s = _finite_binary(y_unknown, scores)
    if y.size == 0 or y.min() == y.max():
        return float("nan")
    if correct is None:
        ok = np.ones(y.shape[0], dtype=bool)
    else:
        ok = _to_numpy(correct).astype(bool).reshape(-1)
        ok = ok[: y.shape[0]]
        if ok.shape[0] != y.shape[0]:
            ok = np.ones(y.shape[0], dtype=bool)
    n_known = float((~y).sum())
    n_unknown = float(y.sum())
    if n_known <= 0 or n_unknown <= 0:
        return float("nan")
    order = np.argsort(s)
    y_ord = y[order]
    known_ord = ~y_ord
    ok_ord = ok[order]
    ccr = np.zeros(y.shape[0] + 1, dtype=np.float64)
    fpr = np.zeros(y.shape[0] + 1, dtype=np.float64)
    known_hit = 0.0
    unknown_hit = 0.0
    for i, idx in enumerate(order, start=1):
        del idx
        if known_ord[i - 1] and ok_ord[i - 1]:
            known_hit += 1.0
        if y_ord[i - 1]:
            unknown_hit += 1.0
        ccr[i] = known_hit / n_known
        fpr[i] = unknown_hit / n_unknown
    return float(np.trapezoid(ccr, fpr) if hasattr(np, "trapezoid") else np.trapz(ccr, fpr))


def expected_calibration_error(
    confidences: torch.Tensor | np.ndarray,
    correct: torch.Tensor | np.ndarray,
    *,
    n_bins: int = 15,
) -> float:
    conf = _to_numpy(confidences).astype(np.float64).reshape(-1)
    ok = _to_numpy(correct).astype(np.float64).reshape(-1)
    n = min(conf.size, ok.size)
    conf, ok = conf[:n], ok[:n]
    keep = np.isfinite(conf)
    conf, ok = conf[keep], ok[keep]
    if conf.size == 0:
        return float("nan")
    bins = np.linspace(0.0, 1.0, int(n_bins) + 1)
    ece = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (conf >= lo) & (conf < hi if hi < 1.0 else conf <= hi)
        if not mask.any():
            continue
        ece += float(mask.mean()) * abs(float(ok[mask].mean()) - float(conf[mask].mean()))
    return float(ece)


def openset_detection_metrics(
    y_unknown: torch.Tensor | np.ndarray,
    scores: torch.Tensor | np.ndarray,
    *,
    confidences: torch.Tensor | np.ndarray | None = None,
    correct: torch.Tensor | np.ndarray | None = None,
) -> dict[str, float]:
    out = {
        "auroc": auroc_score(y_unknown, scores),
        "aupr": aupr_score(y_unknown, scores),
        "fpr95": fpr95_score(y_unknown, scores),
        "oscr": oscr_score(y_unknown, scores, correct=correct),
    }
    if confidences is not None and correct is not None:
        out["ece"] = expected_calibration_error(confidences, correct)
    else:
        out["ece"] = float("nan")
    return out


def openset_four_quadrant_metrics(
    *,
    known_id_scores: torch.Tensor | np.ndarray,
    known_ood_scores: torch.Tensor | np.ndarray,
    unknown_id_scores: torch.Tensor | np.ndarray,
    unknown_ood_scores: torch.Tensor | np.ndarray,
    known_id_correct: torch.Tensor | np.ndarray | None = None,
    known_ood_correct: torch.Tensor | np.ndarray | None = None,
    known_id_conf: torch.Tensor | np.ndarray | None = None,
    known_ood_conf: torch.Tensor | np.ndarray | None = None,
) -> dict[str, Any]:
    """已知/未知 × ID/OOD 四象限开集评测。"""

    def _arr(value: torch.Tensor | np.ndarray | None) -> np.ndarray:
        if value is None:
            return np.zeros((0,), dtype=np.float64)
        return _to_numpy(value).astype(np.float64).reshape(-1)

    known_id = _arr(known_id_scores)
    known_ood = _arr(known_ood_scores)
    unk_id = _arr(unknown_id_scores)
    unk_ood = _arr(unknown_ood_scores)
    known = np.concatenate([known_id, known_ood])
    unknown = np.concatenate([unk_id, unk_ood])
    y = np.concatenate([np.zeros(known.size, dtype=bool), np.ones(unknown.size, dtype=bool)])
    scores = np.concatenate([known, unknown])
    correct_parts = []
    conf_parts = []
    if known_id_correct is not None:
        correct_parts.append(_arr(known_id_correct))
    else:
        correct_parts.append(np.ones(known_id.size, dtype=np.float64))
    if known_ood_correct is not None:
        correct_parts.append(_arr(known_ood_correct))
    else:
        correct_parts.append(np.ones(known_ood.size, dtype=np.float64))
    correct = np.concatenate([*correct_parts, np.zeros(unknown.size, dtype=np.float64)])
    if known_id_conf is not None or known_ood_conf is not None:
        conf_parts.append(_arr(known_id_conf) if known_id_conf is not None else np.full(known_id.size, np.nan))
        conf_parts.append(_arr(known_ood_conf) if known_ood_conf is not None else np.full(known_ood.size, np.nan))
        confidences = np.concatenate([*conf_parts, np.full(unknown.size, np.nan)])
    else:
        confidences = None
    report = openset_detection_metrics(y, scores, confidences=confidences, correct=correct)
    quadrants = {
        "known_id": {"mean_score": float(known_id.mean()) if known_id.size else float("nan"), "n": int(known_id.size)},
        "known_ood": {"mean_score": float(known_ood.mean()) if known_ood.size else float("nan"), "n": int(known_ood.size)},
        "unknown_id": {"mean_score": float(unk_id.mean()) if unk_id.size else float("nan"), "n": int(unk_id.size)},
        "unknown_ood": {"mean_score": float(unk_ood.mean()) if unk_ood.size else float("nan"), "n": int(unk_ood.size)},
    }
    report["quadrants"] = quadrants
    return report
