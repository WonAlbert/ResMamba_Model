from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score, normalized_mutual_info_score


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


def macro_f1(preds: torch.Tensor | np.ndarray, labels: torch.Tensor | np.ndarray, mask: torch.Tensor | np.ndarray | None = None) -> float:
    preds_np = _to_numpy(preds)
    labels_np = _to_numpy(labels)
    valid = _valid_mask(labels_np, mask)
    if valid.sum() == 0:
        return 0.0
    return float(f1_score(labels_np[valid], preds_np[valid], average="macro", zero_division=0))


def nmi_score(pred_clusters: torch.Tensor | np.ndarray, true_labels: torch.Tensor | np.ndarray, mask: torch.Tensor | np.ndarray | None = None) -> float:
    pred_np = _to_numpy(pred_clusters)
    true_np = _to_numpy(true_labels)
    valid = _valid_mask(true_np, mask)
    if valid.sum() < 2:
        return 0.0
    return float(normalized_mutual_info_score(true_np[valid], pred_np[valid]))


def _ssim_1d(x: torch.Tensor, y: torch.Tensor, window: int = 11, sigma: float = 1.5) -> torch.Tensor:
    if x.numel() < window:
        return F.mse_loss(x, y, reduction="none").new_tensor(1.0) - F.mse_loss(x, y)
    coords = torch.arange(window, device=x.device, dtype=x.dtype) - window // 2
    kernel = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    kernel = (kernel / kernel.sum()).view(1, 1, -1)

    def filter1d(t: torch.Tensor) -> torch.Tensor:
        return F.conv1d(t.unsqueeze(0).unsqueeze(0), kernel, padding=window // 2).squeeze()

    c1, c2 = 0.01 ** 2, 0.03 ** 2
    mu_x, mu_y = filter1d(x), filter1d(y)
    sigma_x = filter1d(x * x) - mu_x ** 2
    sigma_y = filter1d(y * y) - mu_y ** 2
    sigma_xy = filter1d(x * y) - mu_x * mu_y
    num = (2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)
    den = (mu_x ** 2 + mu_y ** 2 + c1) * (sigma_x + sigma_y + c2)
    ssim_map = num / den.clamp_min(1.0e-8)
    return ssim_map.mean()


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

    scores: list[torch.Tensor] = []
    for i in range(pred.shape[0]):
        if mask is not None and not bool(mask[i]):
            continue
        for ch in range(2):
            scores.append(_ssim_1d(pred[i, ch].float(), target[i, ch].float()))
    if not scores:
        return 0.0, 0
    stacked = torch.stack(scores)
    return float(stacked.sum().item()), int(stacked.numel())
