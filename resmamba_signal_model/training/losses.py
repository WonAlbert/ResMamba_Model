from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment

from resmamba_signal_model.models.domain import DomainDiscriminator, GradientReversal
from resmamba_signal_model.models.physics import physics_constraint_loss, safe_complex_abs
from resmamba_signal_model.training.clustering_labels import resolve_modulation_labels
from resmamba_signal_model.training.continual import confidence_masked_distillation_loss, old_prototype_anchor_loss
from resmamba_signal_model.training.emitter_labels import global_emitter_labels

DEFAULT_CONTRASTIVE_TEMPERATURE = 0.2
MAX_CONTRASTIVE_SAMPLES = 512
MIN_CONTRASTIVE_SAMPLES = 4
MAX_SINGLE_LOSS = 10.0
PREDICTION_MAE_LOSS_SCALE = 10.0
NEGCOS_TAU_MAX = 0.5
NEGCOS_TAU_MIN = 0.05

__all__ = [
    "DomainDiscriminator",
    "GradientReversal",
    "NEGCOS_TAU_MAX",
    "NEGCOS_TAU_MIN",
    "PREDICTION_MAE_LOSS_SCALE",
    "assign_unique_prototypes",
    "clustering_prototype_alignment_loss",
    "domain_adversarial_loss",
    "downstream_task_loss",
    "foundation_pretrain_losses",
    "latent_prediction_loss",
    "vicreg_loss",
    "vicreg_token_loss",
    "modulation_hierarchical_metric_loss",
    "moe_load_balance_loss",
    "negcos_temperature",
    "physics_constraint_loss",
    "mae_reconstruction_loss",
    "RECON_MONITOR_WEIGHTS",
    "resolve_recon_mask",
    "reconstruction_monitor_loss",
    "safe_cross_entropy",
    "safe_l2_normalize",
    "sinkhorn_balanced_assignment",
    "structure_preserving_loss",
    "supervised_contrastive_loss",
    "unsupervised_clustering_loss",
    "weighted_pretrain_loss",
]


def _first_present(outputs: dict[str, torch.Tensor], *keys: str) -> torch.Tensor | None:
    for key in keys:
        if key in outputs and outputs[key] is not None:
            return outputs[key]
    return None


def _zero_like(base: torch.Tensor) -> torch.Tensor:
    return base.new_tensor(0.0)


def _clamp_loss(loss: torch.Tensor, max_val: float = MAX_SINGLE_LOSS) -> torch.Tensor:
    if not torch.is_tensor(loss):
        return loss
    posinf = float(max_val) if max_val > 0 else 0.0
    loss = torch.nan_to_num(loss, nan=0.0, posinf=posinf, neginf=0.0)
    if max_val <= 0:
        return loss
    return torch.clamp(loss, min=0.0, max=max_val)


def safe_l2_normalize(x: torch.Tensor, dim: int = -1, eps: float = 1.0e-6) -> torch.Tensor:
    """零向量 / 非有限值不会变成 NaN。"""
    x = torch.nan_to_num(x.float(), nan=0.0, posinf=0.0, neginf=0.0)
    return F.normalize(x, p=2.0, dim=dim, eps=eps)


def _subsample_indices(count: int, max_samples: int, device: torch.device) -> torch.Tensor:
    if count <= max_samples:
        return torch.arange(count, device=device)
    return torch.randperm(count, device=device)[:max_samples]


def negcos_temperature(
    progress: float,
    *,
    tau_max: float = NEGCOS_TAU_MAX,
    tau_min: float = NEGCOS_TAU_MIN,
) -> float:
    """零参数负余弦温度调度：progress∈[0,1] 从 tau_max 平滑降到 tau_min。"""
    p = min(max(float(progress), 0.0), 1.0)
    return float(tau_min + 0.5 * (tau_max - tau_min) * (1.0 + math.cos(math.pi * p)))


def safe_cross_entropy(
    logits: torch.Tensor,
    labels: torch.Tensor,
    *,
    label_smoothing: float = 0.0,
) -> torch.Tensor:
    valid = labels >= 0
    if valid.sum() == 0:
        return logits.new_tensor(0.0)
    logits = logits[valid].float()
    labels = labels[valid].long()
    num_classes = logits.shape[-1]
    in_range = (labels >= 0) & (labels < num_classes)
    if not in_range.all():
        logits = logits[in_range]
        labels = labels[in_range]
    if labels.numel() == 0:
        return logits.new_tensor(0.0)
    logits = torch.clamp(logits, -50.0, 50.0)
    smoothing = min(max(float(label_smoothing), 0.0), 0.5)
    return F.cross_entropy(logits, labels, label_smoothing=smoothing)


def _masked_per_sample(elem: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
    """将 [B, T, ...] 误差收成每样本标量 [B]。"""
    if mask is None:
        dims = tuple(range(1, elem.ndim))
        return elem.mean(dim=dims) if dims else elem
    n = min(elem.shape[1], mask.shape[1])
    elem = elem[:, :n]
    mask = mask[:, :n]
    while elem.ndim > mask.ndim:
        elem = elem.mean(dim=-1)
    elem = elem.masked_fill(~mask, 0.0)
    denom = mask.sum(dim=-1).clamp_min(1).to(dtype=elem.dtype)
    return elem.sum(dim=-1) / denom


def _mean_clamped(per: torch.Tensor, valid: torch.Tensor | None = None, max_val: float = MAX_SINGLE_LOSS) -> torch.Tensor:
    if valid is not None:
        if not valid.any():
            return per.new_tensor(0.0)
        per = per[valid]
    if per.numel() == 0:
        return per.new_tensor(0.0)
    if max_val > 0:
        per = per.clamp(max=max_val)
    return per.mean()


def mae_reconstruction_loss(pred: torch.Tensor, target: torch.Tensor, mae_mask: torch.Tensor) -> torch.Tensor:
    if mae_mask is None or mae_mask.sum() == 0:
        return pred.new_tensor(0.0)
    n = min(pred.shape[1], target.shape[1], mae_mask.shape[1])
    pred = pred[:, :n]
    target = target[:, :n]
    mask = mae_mask[:, :n]
    elem = F.smooth_l1_loss(pred, target, reduction="none")
    per = _masked_per_sample(elem, mask)
    return _mean_clamped(per, mask.any(dim=-1))


def _intersect_masks(mask: torch.Tensor, extra: torch.Tensor | None) -> torch.Tensor:
    if extra is None:
        return mask
    n = min(mask.shape[1], extra.shape[1])
    return mask[:, :n] & extra[:, :n].to(dtype=torch.bool)


def resolve_recon_mask(outputs: dict[str, Any], kind: str) -> torch.Tensor | None:
    """按任务选取无泄漏目标掩码；旧 checkpoint 仅有 mae_mask 时回退。"""
    target = outputs.get("target_mask")
    if kind == "prediction":
        if "suffix_mask" in outputs and outputs["suffix_mask"] is not None:
            mask = outputs["suffix_mask"]
        else:
            mask = outputs.get("mae_mask")
    elif kind == "imputation":
        if "span_mask" in outputs and outputs["span_mask"] is not None:
            mask = outputs["span_mask"]
        else:
            mask = outputs.get("mae_mask")
    elif kind == "mae":
        mask = outputs.get("mae_mask")
        if mask is not None and not bool(mask.any()) and outputs.get("span_mask") is not None:
            mask = outputs["span_mask"]
    elif kind in ("query", "target"):
        mask = outputs.get("target_mask", outputs.get("recon_mask", outputs.get("mae_mask")))
    else:
        mask = outputs.get("mae_mask")
    if mask is None:
        return None
    if kind in ("prediction", "imputation", "mae") and target is not None:
        mask = _intersect_masks(mask, target)
    return mask


def domain_adversarial_loss(domain_logits: torch.Tensor, dataset_id: torch.Tensor) -> torch.Tensor:
    return safe_cross_entropy(domain_logits, dataset_id)


def supervised_contrastive_loss(
    features: torch.Tensor,
    labels: torch.Tensor,
    temperature: float = DEFAULT_CONTRASTIVE_TEMPERATURE,
    *,
    max_samples: int = MAX_CONTRASTIVE_SAMPLES,
    min_samples: int = MIN_CONTRASTIVE_SAMPLES,
) -> torch.Tensor:
    valid = labels >= 0
    features = features[valid]
    labels = labels[valid]
    if features.shape[0] < min_samples:
        return features.new_tensor(0.0) if features.numel() else _zero_like(features)
    idx = _subsample_indices(features.shape[0], max_samples, features.device)
    features = features[idx]
    labels = labels[idx]
    features = F.normalize(features.float(), dim=-1)
    logits = features @ features.T / max(float(temperature), 1e-6)
    logits = logits - logits.max(dim=1, keepdim=True).values.detach()
    self_mask = torch.eye(labels.shape[0], dtype=torch.bool, device=labels.device)
    positive_mask = labels[:, None].eq(labels[None, :]) & ~self_mask
    if not positive_mask.any():
        return features.new_tensor(0.0)
    anchor_has_pos = positive_mask.any(dim=1)
    if anchor_has_pos.sum() < 2:
        return features.new_tensor(0.0)
    logits = logits.masked_fill(self_mask, float("-inf"))
    log_prob = logits - torch.logsumexp(logits, dim=1, keepdim=True)
    per_anchor = -(log_prob.masked_fill(~positive_mask, 0.0).sum(dim=1) / positive_mask.sum(dim=1).clamp_min(1))
    return _clamp_loss(per_anchor[anchor_has_pos].mean())


def modulation_hierarchical_metric_loss(
    features: torch.Tensor,
    labels: torch.Tensor,
    dataset_id: torch.Tensor,
    temperature: float = DEFAULT_CONTRASTIVE_TEMPERATURE,
    weak_weight: float = 0.25,
) -> torch.Tensor:
    valid = labels >= 0
    if valid.sum() < MIN_CONTRASTIVE_SAMPLES:
        return features.new_tensor(0.0)
    strong = supervised_contrastive_loss(
        features[valid],
        labels[valid] * 100000 + dataset_id[valid].clamp_min(0),
        temperature,
    )
    weak = supervised_contrastive_loss(features[valid], labels[valid], temperature)
    return _clamp_loss(strong + weak_weight * weak)


def assign_unique_prototypes(class_mean_logits: torch.Tensor) -> torch.Tensor:
    if class_mean_logits.ndim != 2:
        raise ValueError(f"class_mean_logits 期望 [C, K]，当前 shape={tuple(class_mean_logits.shape)}")
    n_cls, n_proto = class_mean_logits.shape
    if n_cls == 0:
        return torch.empty(0, dtype=torch.long, device=class_mean_logits.device)
    if n_cls == 1:
        return class_mean_logits[0].detach().argmax().view(1).to(dtype=torch.long)
    cost = (-class_mean_logits.detach().float()).cpu().numpy()
    row_ind, col_ind = linear_sum_assignment(cost)
    assigned = torch.full((n_cls,), -1, dtype=torch.long, device=class_mean_logits.device)
    taken: set[int] = set()
    for row, col in zip(row_ind.tolist(), col_ind.tolist()):
        assigned[row] = int(col)
        taken.add(int(col))
    if int((assigned < 0).sum().item()) == 0:
        return assigned
    scores = class_mean_logits.detach()
    for cls_idx in range(n_cls):
        if assigned[cls_idx] >= 0:
            continue
        order = scores[cls_idx].argsort(descending=True).tolist()
        chosen = int(order[0])
        for proto_idx in order:
            if int(proto_idx) not in taken:
                chosen = int(proto_idx)
                taken.add(chosen)
                break
        assigned[cls_idx] = chosen
    return assigned


def clustering_prototype_alignment_loss(
    embedding: torch.Tensor,
    logits: torch.Tensor,
    labels: torch.Tensor,
    *,
    separation_weight: float = 1.0,
    collapse_weight: float = 0.1,
) -> torch.Tensor:
    valid = labels >= 0
    if int(valid.sum().item()) < 2:
        return logits.float().sum() * 0.0 + embedding.float().sum() * 0.0
    logits = torch.clamp(logits[valid].float(), -50.0, 50.0)
    labels = labels[valid]
    embedding = safe_l2_normalize(embedding[valid].float(), dim=-1)
    unique_labels = labels.unique()
    class_mean_logits: list[torch.Tensor] = []
    class_means: list[torch.Tensor] = []
    class_masks: list[torch.Tensor] = []
    for label in unique_labels:
        mask = labels == label
        class_masks.append(mask)
        class_mean_logits.append(logits[mask].mean(dim=0))
        class_means.append(safe_l2_normalize(embedding[mask].mean(dim=0), dim=0))
    mean_logits = torch.stack(class_mean_logits, dim=0)
    proto_ids = assign_unique_prototypes(mean_logits)
    proto_targets = torch.empty(labels.shape[0], dtype=torch.long, device=labels.device)
    for mask, proto_id in zip(class_masks, proto_ids.tolist()):
        proto_targets[mask] = int(proto_id)
    align = F.cross_entropy(logits, proto_targets)
    sep = logits.new_tensor(0.0)
    if separation_weight > 0 and len(class_means) >= 2:
        means = torch.stack(class_means, dim=0)
        sim = means @ means.t()
        n_cls = means.shape[0]
        off_diag = sim - torch.eye(n_cls, device=sim.device, dtype=sim.dtype)
        sep = off_diag.square().sum() / float(max(n_cls * (n_cls - 1), 1))
    collapse = logits.new_tensor(0.0)
    if collapse_weight > 0:
        usage = F.softmax(logits, dim=-1).mean(dim=0).clamp_min(1.0e-8)
        entropy = -(usage * usage.log()).sum()
        max_entropy = torch.log(logits.new_tensor(float(logits.shape[-1])))
        collapse = (max_entropy - entropy) / max_entropy.clamp_min(1.0e-8)
    return _clamp_loss(align + float(separation_weight) * sep + float(collapse_weight) * collapse)


@torch.no_grad()
def sinkhorn_balanced_assignment(
    logits: torch.Tensor,
    *,
    epsilon: float = 0.05,
    n_iters: int = 3,
) -> torch.Tensor:
    """SwAV 风格均衡分配（log-space），返回每样本对原型的软目标。"""
    scores = torch.nan_to_num(logits.detach().float(), nan=0.0, posinf=0.0, neginf=0.0)
    eps = max(float(epsilon), 1.0e-6)
    log_q = (scores / eps).transpose(0, 1).contiguous()
    log_q = log_q - torch.logsumexp(log_q.reshape(-1), dim=0)
    n_proto, batch = log_q.shape
    log_r = log_q.new_full((n_proto,), -math.log(float(max(n_proto, 1))))
    log_c = log_q.new_full((batch,), -math.log(float(max(batch, 1))))
    for _ in range(max(int(n_iters), 1)):
        log_q = log_q + (log_r - torch.logsumexp(log_q, dim=1)).unsqueeze(1)
        log_q = log_q + (log_c - torch.logsumexp(log_q, dim=0)).unsqueeze(0)
    log_q = log_q + math.log(float(max(batch, 1)))
    return torch.exp(log_q).transpose(0, 1).contiguous()


def _soft_ce(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    log_prob = F.log_softmax(logits.float(), dim=-1)
    return -(targets.float() * log_prob).sum(dim=-1).mean()


def _prototype_utilization(probs: torch.Tensor) -> torch.Tensor:
    usage = probs.float().mean(dim=0).clamp_min(1.0e-8)
    entropy = -(usage * usage.log()).sum()
    max_entropy = torch.log(probs.new_tensor(float(probs.shape[-1])))
    return (max_entropy - entropy) / max_entropy.clamp_min(1.0e-8)


def unsupervised_clustering_loss(
    embedding: torch.Tensor,
    logits: torch.Tensor,
    *,
    embedding_alt: torch.Tensor | None = None,
    logits_alt: torch.Tensor | None = None,
    prototypes: torch.Tensor | None = None,
    temperature: float = 0.1,
    utilization_weight: float = 0.15,
    consistency_weight: float = 1.0,
    balance_mix: float = 0.7,
    sinkhorn_epsilon: float = 0.1,
    sinkhorn_iters: int = 3,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """跨视图原型一致性；Sinkhorn 均衡可混合，避免硬分配塌到少数原型。

    ``balance_mix``∈[0,1]：1=纯 Sinkhorn 均摊；0=仅用 softmax 目标（更易塌缩）。
    默认 0.7 + ``utilization_weight=0.15``，让硬 argmax 占用更多原型槽。
    """
    parts: dict[str, torch.Tensor] = {}
    z1 = safe_l2_normalize(embedding.float(), dim=-1)
    if embedding_alt is None:
        noise = torch.randn_like(z1) * 0.05
        z2 = safe_l2_normalize(z1 + noise, dim=-1)
    else:
        z2 = safe_l2_normalize(embedding_alt.float(), dim=-1)
    tau = max(float(temperature), 1.0e-6)
    if logits is None and prototypes is not None:
        proto = safe_l2_normalize(prototypes.float(), dim=-1)
        logits = z1 @ proto.t() / tau
    if logits_alt is None:
        if prototypes is not None:
            proto = safe_l2_normalize(prototypes.float(), dim=-1)
            logits_alt = z2 @ proto.t() / tau
        else:
            logits_alt = logits
    logits_a = torch.clamp(logits.float(), -50.0, 50.0)
    logits_b = torch.clamp(logits_alt.float(), -50.0, 50.0)
    if z1.shape[0] < 2:
        zero = logits_a.sum() * 0.0 + z1.sum() * 0.0
        parts["cluster_consistency"] = zero
        parts["cluster_utilization"] = zero
        return zero, parts
    mix = float(min(1.0, max(0.0, balance_mix)))
    soft_a = F.softmax(logits_a, dim=-1)
    soft_b = F.softmax(logits_b, dim=-1)
    if mix > 0.0:
        bal_a = sinkhorn_balanced_assignment(
            logits_a, epsilon=float(sinkhorn_epsilon), n_iters=int(sinkhorn_iters)
        )
        bal_b = sinkhorn_balanced_assignment(
            logits_b, epsilon=float(sinkhorn_epsilon), n_iters=int(sinkhorn_iters)
        )
        q1 = mix * bal_a + (1.0 - mix) * soft_a
        q2 = mix * bal_b + (1.0 - mix) * soft_b
    else:
        q1, q2 = soft_a, soft_b
    consistency = 0.5 * (_soft_ce(logits_a, q2) + _soft_ce(logits_b, q1))
    usage_probs = 0.5 * (q1 + q2)
    utilization = _prototype_utilization(usage_probs)
    parts["cluster_consistency"] = _clamp_loss(consistency)
    parts["cluster_utilization"] = _clamp_loss(utilization)
    total = float(consistency_weight) * parts["cluster_consistency"] + float(utilization_weight) * parts["cluster_utilization"]
    return _clamp_loss(total), parts


def _batch_has_complex_pair(outputs: dict[str, Any], batch: dict[str, Any] | None) -> bool:
    if outputs.get("complex_pair") is False:
        return False
    if outputs.get("complex_pair") is True:
        return True
    if batch:
        if batch.get("complex_pair") is False:
            return False
        spec = batch.get("signal_spec")
        if spec is not None:
            pairs = getattr(spec, "complex_pairs", None)
            if pairs is None and isinstance(spec, dict):
                pairs = spec.get("complex_pairs")
            if pairs:
                return True
            modality = getattr(spec, "modality", None) or getattr(spec, "modality_id", None)
            if isinstance(spec, dict):
                modality = spec.get("modality", spec.get("modality_id", modality))
            if str(modality or "").lower() in ("imu", "navigation", "sonar", "acoustic"):
                return False
    pred = outputs.get("recon_norm", outputs.get("mae_pred"))
    return bool(pred is not None and pred.ndim == 4 and pred.shape[2] == 2)


def _as_complex_last(patches: torch.Tensor) -> torch.Tensor:
    return torch.complex(patches[:, :, 0].float(), patches[:, :, 1].float())


def _masked_mean(elem: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
    valid = None if mask is None else mask[:, : min(elem.shape[1], mask.shape[1])].any(dim=-1)
    return _mean_clamped(_masked_per_sample(elem, mask), valid)


def structure_preserving_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor | None = None,
    *,
    complex_pair: bool = False,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """时域 SmoothL1 + 对数频谱幅度；相位仅 complex-pair 模态。

    返回的主损失只含 time+spectrum。``1-相干`` 常年停在 ~0.9，若与时频等权
    会盖过 MAE；相位项单独放进 ``structure_phase``，由配置降权。
    """
    if mask is not None:
        n = min(pred.shape[1], target.shape[1], mask.shape[1])
        mask = mask[:, :n]
        valid = mask.any(dim=-1)
    else:
        n = min(pred.shape[1], target.shape[1])
        valid = None
    pred = pred[:, :n]
    target = target[:, :n]
    time_elem = F.smooth_l1_loss(pred.float(), target.float(), reduction="none")
    time_per = _masked_per_sample(time_elem, mask)
    spec_per = time_per.new_zeros(time_per.shape)
    phase_per = time_per.new_zeros(time_per.shape)
    if pred.ndim == 4 and pred.shape[2] == 2 and target.shape[:3] == pred.shape[:3]:
        pred_c = _as_complex_last(pred)
        target_c = _as_complex_last(target)
        pred_spec = torch.fft.fft(pred_c, dim=-1)
        target_spec = torch.fft.fft(target_c, dim=-1)
        mag_elem = F.smooth_l1_loss(
            torch.log1p(safe_complex_abs(pred_spec)),
            torch.log1p(safe_complex_abs(target_spec)),
            reduction="none",
        ).mean(dim=-1)
        spec_per = _masked_per_sample(mag_elem, mask)
        if complex_pair:
            pred_u = pred_spec / safe_complex_abs(pred_spec)
            target_u = target_spec / safe_complex_abs(target_spec)
            coherence = pred_u.real * target_u.real + pred_u.imag * target_u.imag
            phase_elem = (1.0 - coherence).mean(dim=-1)
            phase_per = _masked_per_sample(phase_elem, mask)
    elif pred.ndim >= 3:
        pred_f = torch.log1p(torch.fft.rfft(pred.float(), dim=-1).abs())
        target_f = torch.log1p(torch.fft.rfft(target.float(), dim=-1).abs())
        mag_elem = F.smooth_l1_loss(pred_f, target_f, reduction="none")
        spec_per = _masked_per_sample(mag_elem, mask)
    parts = {
        "structure_time": _mean_clamped(time_per, valid),
        "structure_spectrum": _mean_clamped(spec_per, valid),
        "structure_phase": _mean_clamped(phase_per, valid),
    }
    return _mean_clamped(time_per + spec_per, valid), parts


def _cosine_latent(student: torch.Tensor, teacher: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
    s = F.normalize(student.float(), dim=-1)
    t = F.normalize(teacher.detach().float(), dim=-1)
    if s.shape != t.shape:
        n = min(s.shape[1], t.shape[1]) if s.ndim > 1 and t.ndim > 1 and s.ndim == t.ndim else None
        if n is None:
            return student.new_tensor(0.0)
        s = s[:, :n]
        t = t[:, :n]
    sim = (s * t).sum(dim=-1)
    loss = 1.0 - sim
    if mask is not None and loss.ndim >= 1 and mask.ndim == loss.ndim:
        n = min(loss.shape[-1] if loss.ndim > 1 else loss.shape[0], mask.shape[-1] if mask.ndim > 1 else mask.shape[0])
        if loss.ndim == 1:
            loss = loss[:n]
            keep = mask.reshape(-1)[:n]
            if not keep.any():
                return student.new_tensor(0.0)
            return loss[keep].mean()
        loss = loss[:, :n]
        keep = mask[:, :n]
        if not keep.any():
            return student.new_tensor(0.0)
        return loss.masked_fill(~keep, 0.0).sum() / keep.sum().clamp_min(1).to(dtype=loss.dtype)
    return loss.mean()


def latent_prediction_loss(
    student_z: torch.Tensor,
    teacher_z: torch.Tensor,
    student_h: torch.Tensor | None = None,
    teacher_h: torch.Tensor | None = None,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    pooled = _cosine_latent(student_z, teacher_z)
    if student_h is None or teacher_h is None:
        return _clamp_loss(pooled)
    token = _cosine_latent(student_h, teacher_h, mask)
    return _clamp_loss(pooled + token)


def vicreg_loss(
    z: torch.Tensor,
    teacher_z: torch.Tensor | None = None,
    *,
    var_weight: float = 25.0,
    cov_weight: float = 1.0,
    inv_weight: float = 1.0,
    gamma: float = 1.0,
    eps: float = 1.0e-4,
) -> torch.Tensor:
    """VICReg 抗塌缩：方差下界 + 协方差去相关；若有 teacher 则加不变性。

    默认作用在 encoder 池化 ``z_enc``。不经 ``MAX_SINGLE_LOSS`` 硬截断，
    否则原值常 >10、clamp 后梯度为 0、TensorBoard 会卡在常数。
    """
    if z is None or not torch.is_tensor(z) or z.ndim != 2 or z.shape[0] < 2:
        ref = z if torch.is_tensor(z) else torch.zeros(())
        return ref.new_tensor(0.0)
    zf = torch.nan_to_num(z.float(), nan=0.0, posinf=0.0, neginf=0.0)
    std = torch.sqrt(zf.var(dim=0, unbiased=False) + eps)
    var_loss = torch.mean(F.relu(float(gamma) - std))
    zc = zf - zf.mean(dim=0, keepdim=True)
    n = float(max(zf.shape[0] - 1, 1))
    d = float(max(zf.shape[1], 1))
    cov = (zc.T @ zc) / n
    # 论文形式：off-diag 平方和 / d；再除以 d 使宽表征尺度与 d 近似无关
    off = cov.pow(2).sum() - cov.diagonal().pow(2).sum()
    cov_loss = off / (d * d)
    total = float(var_weight) * var_loss + float(cov_weight) * cov_loss
    if (
        teacher_z is not None
        and torch.is_tensor(teacher_z)
        and teacher_z.shape == zf.shape
        and float(inv_weight) > 0.0
    ):
        inv = 1.0 - F.cosine_similarity(zf, teacher_z.detach().float(), dim=-1).mean()
        total = total + float(inv_weight) * inv
    return torch.nan_to_num(total, nan=0.0, posinf=0.0, neginf=0.0)


def vicreg_token_loss(
    h: torch.Tensor,
    mask: torch.Tensor | None = None,
    *,
    var_weight: float = 25.0,
    cov_weight: float = 1.0,
    gamma: float = 1.0,
    eps: float = 1.0e-4,
) -> torch.Tensor:
    """可见 encoder token 上的 VICReg，防止细粒度 token 塌缩。"""
    if h is None or h.ndim != 3 or h.shape[0] < 1:
        return h.new_tensor(0.0) if torch.is_tensor(h) else torch.tensor(0.0)
    if mask is not None:
        keep = mask.to(dtype=torch.bool)
        flat = h[keep]
    else:
        flat = h.reshape(-1, h.shape[-1])
    if flat.shape[0] < 2:
        return flat.new_tensor(0.0)
    return vicreg_loss(
        flat,
        var_weight=var_weight,
        cov_weight=cov_weight,
        inv_weight=0.0,
        gamma=gamma,
        eps=eps,
    )


def moe_load_balance_loss(outputs: dict[str, Any]) -> torch.Tensor:
    """聚合 tokenizer / encoder / decoder MoE 负载均衡损失。"""
    lb = outputs.get("moe_load_balance")
    if lb is None:
        ref = outputs.get("z_enc", outputs.get("z"))
        return ref.new_tensor(0.0) if torch.is_tensor(ref) else torch.tensor(0.0)
    if not torch.is_tensor(lb):
        ref = outputs.get("z_enc", outputs.get("z"))
        return ref.new_tensor(0.0) if torch.is_tensor(ref) else torch.tensor(0.0)
    return _clamp_loss(lb)


def foundation_pretrain_losses(
    outputs: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor] | None = None,
    *,
    include: set[str] | None = None,
) -> dict[str, torch.Tensor]:
    wanted = include
    losses: dict[str, torch.Tensor] = {}

    def _need(name: str) -> bool:
        return wanted is None or name in wanted

    pred = outputs.get("recon_norm", outputs["mae_pred"])
    target = outputs.get("patch_targets_norm", outputs["patch_targets"])
    if _need("mae"):
        mae_mask = resolve_recon_mask(outputs, "mae")
        if mae_mask is None:
            losses["mae"] = pred.new_tensor(0.0)
        else:
            losses["mae"] = _clamp_loss(mae_reconstruction_loss(pred, target, mae_mask))
    if _need("impute"):
        span = resolve_recon_mask(outputs, "imputation")
        if span is None or not bool(span.any()):
            losses["impute"] = pred.new_tensor(0.0)
        else:
            losses["impute"] = _clamp_loss(mae_reconstruction_loss(pred, target, span))
    if _need("physical"):
        mask = outputs.get("recon_mask", outputs.get("target_mask", outputs.get("mae_mask")))
        losses["physical"] = _clamp_loss(physics_constraint_loss(pred, target, mask))
    if _need("readout"):
        losses["readout"] = _clamp_loss(
            F.smooth_l1_loss(outputs["global_phys_pred"].float(), outputs["global_phys_target"].float())
        )
    if _need("domain"):
        dataset_id = outputs.get("dataset_id")
        if dataset_id is None and batch is not None:
            dataset_id = batch.get("dataset_id")
        domain_logits = outputs.get("domain_logits")
        if dataset_id is None or domain_logits is None:
            losses["domain"] = pred.new_tensor(0.0)
        else:
            losses["domain"] = domain_adversarial_loss(domain_logits, dataset_id)
    if _need("structure") or _need("structure_phase"):
        struct_mask = outputs.get("target_mask", outputs.get("recon_mask", outputs.get("mae_mask")))
        structure, struct_parts = structure_preserving_loss(
            pred,
            target,
            struct_mask,
            complex_pair=_batch_has_complex_pair(outputs, batch),
        )
        losses["structure"] = structure
        losses.update(struct_parts)
    if _need("latent"):
        teacher_z = outputs.get("teacher_z")
        # 旧 EMA 余弦默认打在 decoder z 上易塌缩；仅当显式提供 teacher 时保留。
        student_z = outputs.get("z_recon", outputs.get("z_general", outputs.get("z")))
        if teacher_z is None or student_z is None:
            losses["latent"] = pred.new_tensor(0.0)
        else:
            losses["latent"] = latent_prediction_loss(
                student_z,
                teacher_z,
                outputs.get("h_recon", outputs.get("h_general", outputs.get("patch_h"))),
                outputs.get("teacher_h", outputs.get("teacher_tokens")),
                outputs.get("target_mask"),
            )
    if _need("vicreg"):
        student = outputs.get("z_enc", outputs.get("z_general", outputs.get("z")))
        if student is None:
            losses["vicreg"] = pred.new_tensor(0.0)
        else:
            losses["vicreg"] = vicreg_loss(
                student,
                outputs.get("teacher_z_enc", outputs.get("teacher_z")),
                var_weight=float(outputs.get("vicreg_var_weight", 25.0) or 25.0),
                cov_weight=float(outputs.get("vicreg_cov_weight", 1.0) or 1.0),
                inv_weight=float(outputs.get("vicreg_inv_weight", 0.0) or 0.0),
            )
    if _need("vicreg_token"):
        h_enc = outputs.get("h_enc")
        visible = outputs.get("visible", outputs.get("patch_mask"))
        if h_enc is None or visible is None:
            losses["vicreg_token"] = pred.new_tensor(0.0)
        else:
            losses["vicreg_token"] = vicreg_token_loss(
                h_enc,
                visible & outputs.get("patch_mask", visible),
                var_weight=float(outputs.get("vicreg_var_weight", 25.0) or 25.0),
                cov_weight=float(outputs.get("vicreg_cov_weight", 1.0) or 1.0),
            )
    return losses


# 验证/选 ckpt 用的重建项：排除 domain、近常数 phase、以及依赖 teacher 的对齐项。
RECON_MONITOR_WEIGHTS: dict[str, float] = {
    "mae": 1.0,
    "impute": 0.2,
    "structure": 0.2,
}


def reconstruction_monitor_loss(
    parts: dict[str, torch.Tensor],
    weights: dict[str, float] | None = None,
) -> torch.Tensor:
    """由已算好的分项拼重建监控标量（不进 domain / structure_phase）。"""
    active = {k: float(v) for k, v in (weights or RECON_MONITOR_WEIGHTS).items() if float(v) > 0.0}
    if not active:
        active = dict(RECON_MONITOR_WEIGHTS)
    ref = next(iter(parts.values())) if parts else None
    total = ref.new_tensor(0.0) if ref is not None else torch.tensor(0.0)
    for name, weight in active.items():
        value = parts.get(name)
        if value is None:
            continue
        total = total + float(weight) * value
    return total


def weighted_pretrain_loss(
    outputs: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
    weights: dict[str, float],
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    active = {k: float(v) for k, v in weights.items() if float(v) > 0.0}
    if not active:
        active = {"mae": 1.0}
    losses = foundation_pretrain_losses(outputs, batch, include=set(active))
    total = outputs["mae_pred"].new_tensor(0.0)
    logged: dict[str, torch.Tensor] = {}
    for name, value in losses.items():
        weight = float(active.get(name, 0.0))
        if weight > 0.0:
            total = total + weight * value
            logged[name] = value
        elif name not in weights:
            # structure_time 等诊断子项：父项在 active 时一并返回供监控
            logged[name] = value
    return total, logged


def downstream_task_loss(
    outputs: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
    task: str,
    *,
    emitter_offset_lookup: torch.Tensor | None = None,
    modulation_compact_lookup: torch.Tensor | None = None,
    emitter_contrastive_weight: float = 0.0,
    modulation_contrastive_weight: float = 0.0,
    z_contrastive_weight: float = 0.0,
    recon_weight: float = 0.1,
    phys_weight: float = 0.1,
    domain_weight: float = 0.1,
    task_kind: str | None = None,
    task_catalog: dict[str, Any] | None = None,
    label_field: str | None = None,
    supervised_clustering: bool = False,
    distill_weight: float = 0.0,
    distill_temperature: float = 2.0,
    distill_confidence: float = 0.5,
    prototype_anchor_weight: float = 0.0,
    z_probe_weight: float = 1.0,
    emitter_label_smoothing: float = 0.0,
    cluster_utilization_weight: float = 0.15,
    cluster_consistency_weight: float = 1.0,
    cluster_balance_mix: float = 0.7,
    cluster_sinkhorn_epsilon: float = 0.1,
    cluster_sinkhorn_iters: int = 3,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    parts: dict[str, torch.Tensor] = {}
    loss = outputs["z"].new_tensor(0.0) if "z" in outputs else outputs["mae_pred"].new_tensor(0.0)
    feat = outputs.get("task_pooled", outputs.get("z"))
    kind = task_kind
    if kind is None:
        from resmamba_signal_model.training.task_catalog import resolve_task_catalog

        kind = resolve_task_catalog(task_catalog).kind(task) if task_catalog is not None else None
    if kind is None:
        if task == "modulation":
            kind = "classification"
        elif task in ("emitter", "clustering", "prediction", "imputation"):
            kind = task
        elif "cluster_logits" in outputs:
            kind = "clustering"
        elif "pred_patches" in outputs or task.endswith("imputation"):
            kind = "imputation" if "impute" in task or task.endswith("imputation") else "prediction"
        else:
            kind = "classification"

    if kind == "classification":
        logits = _first_present(outputs, "task_logits", f"{task}_logits", "modulation_logits")
        if logits is None:
            raise KeyError(f"分类任务 {task!r} 缺少 logits")
        raw_labels = batch.get(label_field or "mod_label_id", batch.get("source_label_id"))
        if raw_labels is None:
            raise KeyError(f"分类任务 {task!r} 缺少标签列")
        labels = resolve_modulation_labels(raw_labels, batch.get("source_label_id"))
        if modulation_compact_lookup is not None:
            from resmamba_signal_model.training.modulation_labels import remap_modulation_labels

            labels = remap_modulation_labels(labels, modulation_compact_lookup)
        ce = safe_cross_entropy(logits, labels)
        parts["task_ce"] = ce
        loss = loss + ce
        if "z_probe_logits" in outputs and outputs["z_probe_logits"] is not None:
            probe_ce = safe_cross_entropy(outputs["z_probe_logits"], labels)
            parts["z_probe_ce"] = probe_ce
            loss = loss + float(z_probe_weight) * probe_ce
        if modulation_contrastive_weight > 0:
            contrastive = modulation_hierarchical_metric_loss(
                feat,
                labels,
                batch.get("dataset_id", torch.zeros_like(labels)),
            )
            parts["modulation_contrastive"] = contrastive
            loss = loss + float(modulation_contrastive_weight) * contrastive
        if float(z_contrastive_weight) > 0:
            z_feat = outputs.get("z_enc", outputs.get("z_general", outputs.get("z")))
            if z_feat is not None and torch.is_tensor(z_feat):
                z_contrastive = modulation_hierarchical_metric_loss(
                    z_feat,
                    labels,
                    batch.get("dataset_id", torch.zeros_like(labels)),
                )
                parts["z_contrastive"] = z_contrastive
                loss = loss + float(z_contrastive_weight) * z_contrastive
    elif kind == "emitter":
        # 紧凑标签：必须用局部 emitter_id + offset，不能直接加在 namespace global_id 上。
        if emitter_offset_lookup is not None:
            local = batch.get("emitter_id")
            if local is None:
                raise KeyError("紧凑个体标签需要 batch['emitter_id']")
            labels = global_emitter_labels(batch["dataset_id"], local, emitter_offset_lookup)
        else:
            labels = batch.get(label_field or "emitter_id", batch["emitter_id"])
        logits = _first_present(outputs, "task_logits", "emitter_logits")
        ce = safe_cross_entropy(logits, labels, label_smoothing=emitter_label_smoothing)
        parts["task_ce"] = ce
        loss = loss + ce
        if "z_probe_logits" in outputs and outputs["z_probe_logits"] is not None:
            probe_ce = safe_cross_entropy(outputs["z_probe_logits"], labels)
            parts["z_probe_ce"] = probe_ce
            loss = loss + float(z_probe_weight) * probe_ce
        if emitter_contrastive_weight > 0:
            contrastive_feat = feat
            contrastive = supervised_contrastive_loss(contrastive_feat, labels)
            parts["emitter_contrastive"] = contrastive
            loss = loss + float(emitter_contrastive_weight) * contrastive
        if float(z_contrastive_weight) > 0:
            z_feat = outputs.get("z_enc", outputs.get("z_general", outputs.get("z")))
            if z_feat is not None and torch.is_tensor(z_feat):
                z_contrastive = supervised_contrastive_loss(z_feat, labels)
                parts["z_contrastive"] = z_contrastive
                loss = loss + float(z_contrastive_weight) * z_contrastive
    elif kind == "clustering":
        # 训练路径默认无监督；global_label_id 不得进入 loss。
        if supervised_clustering:
            labels = batch.get(label_field or "global_label_id")
            if labels is None:
                parts["cluster_align"] = outputs["cluster_logits"].new_tensor(0.0)
            else:
                align = clustering_prototype_alignment_loss(outputs["cluster_embedding"], outputs["cluster_logits"], labels)
                parts["cluster_align"] = align
                loss = loss + align
        else:
            cluster_loss, cluster_parts = unsupervised_clustering_loss(
                outputs["cluster_embedding"],
                outputs["cluster_logits"],
                embedding_alt=outputs.get("cluster_embedding_view2"),
                logits_alt=outputs.get("cluster_logits_view2"),
                prototypes=outputs.get("cluster_prototypes"),
                utilization_weight=float(cluster_utilization_weight),
                consistency_weight=float(cluster_consistency_weight),
                balance_mix=float(cluster_balance_mix),
                sinkhorn_epsilon=float(cluster_sinkhorn_epsilon),
                sinkhorn_iters=int(cluster_sinkhorn_iters),
            )
            parts.update(cluster_parts)
            parts["cluster_unsupervised"] = cluster_loss
            loss = loss + cluster_loss
    elif kind in ("prediction", "imputation"):
        pred = outputs.get("pred_patches")
        if pred is None:
            pred = outputs.get("recon_norm")
        if pred is None:
            pred = outputs["mae_pred"]
        target = outputs.get("patch_targets_norm")
        if target is None:
            target = outputs["patch_targets"]
        mask = resolve_recon_mask(outputs, kind)
        if mask is None:
            mask = outputs.get("mae_mask")
        mae = mae_reconstruction_loss(pred, target, mask)
        scaled = mae * float(PREDICTION_MAE_LOSS_SCALE)
        parts["mae"] = mae
        parts["mae_scaled"] = scaled
        loss = loss + scaled
    else:
        raise ValueError(f"未知下游任务 {task!r} kind={kind!r}")

    if domain_weight > 0 and "domain_logits" in outputs and batch.get("dataset_id") is not None:
        domain = domain_adversarial_loss(outputs["domain_logits"], batch["dataset_id"])
        parts["domain"] = domain
        loss = loss + float(domain_weight) * domain
    if recon_weight > 0 and kind in ("classification", "emitter", "clustering") and "mae_pred" in outputs:
        pred = outputs.get("recon_norm", outputs["mae_pred"])
        target = outputs.get("patch_targets_norm", outputs["patch_targets"])
        recon = mae_reconstruction_loss(pred, target, outputs.get("patch_mask", outputs.get("mae_mask")))
        parts["mae"] = recon
        loss = loss + float(recon_weight) * recon
    if phys_weight > 0 and "mae_pred" in outputs and outputs["mae_pred"].ndim == 4:
        pred = outputs.get("recon_norm", outputs["mae_pred"])
        target = outputs.get("patch_targets_norm", outputs["patch_targets"])
        phys = physics_constraint_loss(pred, target, outputs.get("patch_mask"))
        parts["physical"] = phys
        loss = loss + float(phys_weight) * phys
    if distill_weight > 0:
        student_logits = _first_present(outputs, "task_logits", "cluster_logits", "modulation_logits", "emitter_logits")
        teacher_logits = _first_present(outputs, "teacher_logits", "teacher_task_logits")
        if student_logits is not None and teacher_logits is not None:
            distill = confidence_masked_distillation_loss(
                student_logits,
                teacher_logits,
                temperature=distill_temperature,
                confidence_threshold=distill_confidence,
            )
            parts["distill"] = distill
            loss = loss + float(distill_weight) * distill
    if prototype_anchor_weight > 0 and "prototype_mean" in outputs and "frozen_prototype_mean" in outputs:
        anchor = old_prototype_anchor_loss(
            outputs["prototype_mean"],
            outputs["frozen_prototype_mean"],
            outputs.get("prototype_count"),
        )
        parts["prototype_anchor"] = anchor
        loss = loss + float(prototype_anchor_weight) * anchor
    return loss, parts
