from __future__ import annotations

import torch
import torch.nn.functional as F

from resmamba_signal_model.training.clustering_labels import resolve_modulation_labels
from resmamba_signal_model.training.emitter_labels import global_emitter_labels

# 对比类 loss 的稳定化默认值
DEFAULT_CONTRASTIVE_TEMPERATURE = 0.2
MAX_CONTRASTIVE_SAMPLES = 512
MIN_CONTRASTIVE_SAMPLES = 4
MAX_SINGLE_LOSS = 10.0


def _zero_like(base: torch.Tensor) -> torch.Tensor:
    return base.new_tensor(0.0)


def _clamp_loss(loss: torch.Tensor, max_val: float = MAX_SINGLE_LOSS) -> torch.Tensor:
    if max_val <= 0:
        return loss
    return torch.clamp(loss, max=max_val)


def _subsample_indices(count: int, max_samples: int, device: torch.device) -> torch.Tensor:
    if count <= max_samples:
        return torch.arange(count, device=device)
    return torch.randperm(count, device=device)[:max_samples]


def safe_cross_entropy(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """AMP 友好的分类 CE：float32 logits、过滤越界标签、限制 logit 幅度。"""
    valid = labels >= 0
    if valid.sum() == 0:
        return logits.new_tensor(0.0)
    logits = logits[valid].float()
    labels = labels[valid].long()
    num_classes = logits.shape[-1]
    in_range = labels < num_classes
    if not in_range.all():
        logits = logits[in_range]
        labels = labels[in_range]
    if labels.numel() == 0:
        return logits.new_tensor(0.0)
    logits = torch.clamp(logits, -50.0, 50.0)
    return F.cross_entropy(logits, labels)


def mae_reconstruction_loss(pred: torch.Tensor, target: torch.Tensor, mae_mask: torch.Tensor) -> torch.Tensor:
    if mae_mask.sum() == 0:
        return pred.new_tensor(0.0)
    return F.smooth_l1_loss(pred[mae_mask], target[mae_mask])


def physical_preservation_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return F.smooth_l1_loss(pred.float(), target.float())


def future_latent_infonce_loss(
    pred: torch.Tensor | None,
    target: torch.Tensor | None,
    valid_mask: torch.Tensor | None,
    temperature: float = DEFAULT_CONTRASTIVE_TEMPERATURE,
    max_samples: int = MAX_CONTRASTIVE_SAMPLES,
) -> torch.Tensor:
    if pred is None or target is None or valid_mask is None or valid_mask.sum() < 2:
        base = pred if pred is not None else target
        if base is None:
            return torch.tensor(0.0)
        return base.new_tensor(0.0)

    p = pred[valid_mask].reshape(-1, pred.shape[-1])
    t = target[valid_mask].reshape(-1, target.shape[-1])
    if p.shape[0] < 2:
        return p.new_tensor(0.0)

    idx = _subsample_indices(p.shape[0], max_samples, p.device)
    p = F.normalize(p[idx].float(), dim=-1)
    t = F.normalize(t[idx].float(), dim=-1)
    logits = p @ t.T / max(float(temperature), 1.0e-6)
    labels = torch.arange(logits.shape[0], device=logits.device)
    return _clamp_loss(F.cross_entropy(logits, labels))


def vicreg_loss(
    x: torch.Tensor,
    y: torch.Tensor,
    sim_weight: float = 1.0,
    var_weight: float = 1.0,
    cov_weight: float = 0.04,
) -> torch.Tensor:
    if x.shape[0] < 2:
        return x.new_tensor(0.0)
    x = x.float()
    y = y.float()
    sim = F.mse_loss(x, y)
    std_x = torch.sqrt(x.var(dim=0) + 1.0e-4)
    std_y = torch.sqrt(y.var(dim=0) + 1.0e-4)
    var = torch.mean(F.relu(1.0 - std_x)) + torch.mean(F.relu(1.0 - std_y))
    x = x - x.mean(dim=0)
    y = y - y.mean(dim=0)
    cov_x = (x.T @ x) / (x.shape[0] - 1)
    cov_y = (y.T @ y) / (y.shape[0] - 1)
    off_x = cov_x.flatten()[:-1].view(cov_x.shape[0] - 1, cov_x.shape[0] + 1)[:, 1:].flatten()
    off_y = cov_y.flatten()[:-1].view(cov_y.shape[0] - 1, cov_y.shape[0] + 1)[:, 1:].flatten()
    cov = (off_x.square().sum() + off_y.square().sum()) / x.shape[1]
    return _clamp_loss(sim_weight * sim + var_weight * var + cov_weight * cov)


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


def emitter_namespace_labels(dataset_id: torch.Tensor, emitter_id: torch.Tensor, namespace_size: int = 100000) -> torch.Tensor:
    valid = (dataset_id >= 0) & (emitter_id >= 0)
    return torch.where(valid, dataset_id.long() * namespace_size + emitter_id.long(), torch.full_like(emitter_id.long(), -1))


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


def multi_space_consistency_loss(spaces: dict[str, torch.Tensor]) -> torch.Tensor:
    shared = spaces["long_context_shared"]
    cross = spaces["cross_domain_shared"]
    return F.mse_loss(F.normalize(shared, dim=-1), F.normalize(cross, dim=-1))


def orthogonality_loss(spaces: dict[str, torch.Tensor]) -> torch.Tensor:
    pairs = [("mod_specific", "emitter_specific"), ("mod_specific", "cross_domain_shared"), ("emitter_specific", "cross_domain_shared")]
    losses = []
    for a, b in pairs:
        za = F.normalize(spaces[a].float(), dim=-1)
        zb = F.normalize(spaces[b].float(), dim=-1)
        losses.append((za * zb).sum(dim=-1).square().mean())
    return torch.stack(losses).mean()


def weighted_pretrain_loss(
    outputs: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
    weights: dict[str, float],
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    losses = pretrain_smoke_losses(outputs, batch)
    total = outputs["mae_pred"].new_tensor(0.0)
    for name, value in losses.items():
        total = total + float(weights.get(name, 0.0)) * value
    return total, losses


def stage2_task_loss(
    outputs: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
    task: str,
    *,
    emitter_offset_lookup: torch.Tensor | None = None,
    emitter_contrastive_weight: float = 0.0,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    if task == "modulation":
        labels = resolve_modulation_labels(
            batch["mod_label_id"],
            batch.get("source_label_id"),
        )
        logits = outputs["modulation_logits"]
        loss = safe_cross_entropy(logits, labels)
        return loss, {"task_ce": loss}

    if task == "emitter":
        raw_labels = batch["emitter_id"]
        if emitter_offset_lookup is not None:
            labels = global_emitter_labels(batch["dataset_id"], raw_labels, emitter_offset_lookup)
        else:
            labels = raw_labels
        logits = outputs["emitter_logits"]
        valid = labels >= 0
        if valid.sum() == 0:
            loss = logits.new_tensor(0.0)
            parts = {"task_ce": loss}
        else:
            ce = safe_cross_entropy(logits, labels)
            parts = {"task_ce": ce}
            loss = ce
            if emitter_contrastive_weight > 0 and "emitter_repr" in outputs:
                contrastive = supervised_contrastive_loss(outputs["emitter_repr"], labels)
                parts["emitter_contrastive"] = contrastive
                loss = loss + float(emitter_contrastive_weight) * contrastive
        return loss, parts

    if task == "clustering":
        labels = batch.get("global_label_id")
        embedding = outputs["cluster_embedding"]
        if labels is None:
            loss = (embedding ** 2).mean() * 1.0e-3
        else:
            valid = labels >= 0
            if valid.sum() < MIN_CONTRASTIVE_SAMPLES:
                loss = (embedding ** 2).mean() * 1.0e-3
            else:
                loss = supervised_contrastive_loss(embedding[valid], labels[valid])
                if not loss.requires_grad:
                    loss = (embedding ** 2).mean() * 1.0e-3
        return loss, {"cluster_contrastive": loss}

    if task == "prediction":
        loss = mae_reconstruction_loss(outputs["mae_pred"], outputs["patch_targets"], outputs["mae_mask"])
        return loss, {"mae": loss}

    raise ValueError(f"未知 stage2 任务 {task!r}")


def pretrain_smoke_losses(outputs: dict[str, torch.Tensor], batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    spaces = outputs["spaces"]
    emitter_labels = emitter_namespace_labels(
        batch.get("dataset_id", torch.zeros_like(batch["task_type_id"])),
        batch.get("emitter_id", torch.full_like(batch["task_type_id"], -1)),
    )
    mod_labels = batch.get("mod_label_id", torch.full_like(batch["task_type_id"], -1))
    return {
        "mae": mae_reconstruction_loss(outputs["mae_pred"], outputs["patch_targets"], outputs["mae_mask"]),
        "physical": physical_preservation_loss(outputs["physical_pred"], outputs["physical_targets"]),
        "future": future_latent_infonce_loss(outputs["future_pred"], outputs["future_targets"], outputs["future_mask"]),
        "mod_metric": modulation_hierarchical_metric_loss(
            outputs["modulation_repr"],
            mod_labels,
            batch.get("dataset_id", torch.zeros_like(mod_labels)),
        ),
        "emitter_metric": supervised_contrastive_loss(outputs["emitter_repr"], emitter_labels),
        "multi_consistency": multi_space_consistency_loss(spaces),
        "orthogonality": orthogonality_loss(spaces),
        "vicreg": vicreg_loss(spaces["long_context_shared"], spaces["cross_domain_shared"]),
    }
