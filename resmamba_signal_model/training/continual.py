from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from resmamba_signal_model.models.prototypes import PrototypeRegistry


def confidence_masked_distillation_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    *,
    temperature: float = 2.0,
    confidence_threshold: float = 0.5,
) -> torch.Tensor:
    """旧模型置信样本上的 KL 蒸馏；低置信位置不回传。"""
    if student_logits.shape != teacher_logits.shape or student_logits.shape[0] == 0:
        return student_logits.new_tensor(0.0)
    tau = max(float(temperature), 1.0e-6)
    teacher = teacher_logits.detach().float()
    student = student_logits.float()
    teacher_prob = F.softmax(teacher / tau, dim=-1)
    confidence = teacher_prob.max(dim=-1).values
    keep = confidence >= float(confidence_threshold)
    if not keep.any():
        return student.new_tensor(0.0)
    log_student = F.log_softmax(student[keep] / tau, dim=-1)
    distill = F.kl_div(log_student, teacher_prob[keep], reduction="batchmean") * (tau * tau)
    return distill


def old_prototype_anchor_loss(
    current_mean: torch.Tensor,
    frozen_mean: torch.Tensor,
    counts: torch.Tensor | None = None,
    *,
    min_count: float = 1.0,
) -> torch.Tensor:
    """约束已被吸收的旧原型不要漂移过远。"""
    n = min(current_mean.shape[0], frozen_mean.shape[0])
    if n == 0:
        return current_mean.new_tensor(0.0)
    cur = F.normalize(current_mean[:n].float(), dim=-1)
    old = F.normalize(frozen_mean[:n].detach().float(), dim=-1)
    per = (1.0 - (cur * old).sum(dim=-1)).clamp_min(0.0)
    if counts is not None:
        mass = counts[:n].to(dtype=per.dtype)
        keep = mass >= float(min_count)
        if not keep.any():
            return current_mean.new_tensor(0.0)
        return (per[keep] * mass[keep]).sum() / mass[keep].sum().clamp_min(1.0)
    return per.mean()


def absorb_unknown_embeddings(
    registry: PrototypeRegistry,
    embedding: torch.Tensor,
    *,
    namespace: str,
    unknown_mask: torch.Tensor | None = None,
    min_count: float = 2.0,
) -> int:
    if unknown_mask is not None:
        embedding = embedding[unknown_mask.to(dtype=torch.bool)]
    if embedding.shape[0] < int(min_count):
        return 0
    return int(registry.absorb(namespace, embedding, min_count=min_count))


def apply_continual_freeze(model: nn.Module) -> None:
    """只打开共享低秩 adapter、原型与任务头；不新建 per-task LoRA。"""
    for param in model.parameters():
        param.requires_grad = False
    for name in ("shared_adapter", "prototype_registry"):
        module = getattr(model, name, None)
        if isinstance(module, nn.Module):
            for param in module.parameters():
                param.requires_grad = True
    extra = getattr(model, "extra_task_heads", None)
    heads = []
    for attr in ("modulation_head", "emitter_head", "clustering_head", "prediction_head", "imputation_head"):
        head = getattr(model, attr, None)
        if isinstance(head, nn.Module):
            heads.append(head)
    if isinstance(extra, nn.ModuleDict):
        heads.extend(list(extra.values()))
    for head in heads:
        for param in head.parameters():
            param.requires_grad = True
    fingerprint = getattr(model, "emitter_fingerprint", None)
    if isinstance(fingerprint, nn.Module):
        for param in fingerprint.parameters():
            param.requires_grad = True


def continual_parameter_budget(model: nn.Module) -> dict[str, Any]:
    trainable = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    n_params = int(sum(p.numel() for _, p in trainable))
    lora_task = [n for n, _ in trainable if "lora_" in n and ".shared" not in n]
    return {
        "trainable_params": n_params,
        "has_shared_adapter": any(n.startswith("shared_adapter.") for n, _ in trainable),
        "per_task_lora": lora_task,
    }


def resolve_continual_sessions(train_cfg: dict[str, Any], *, stage: str | None = None) -> list[dict[str, Any]]:
    raw = train_cfg.get("continual_sessions")
    if isinstance(raw, list) and raw:
        sessions: list[dict[str, Any]] = []
        for item in raw:
            if isinstance(item, dict):
                sessions.append(dict(item))
            else:
                sessions.append({"name": str(item)})
        return sessions
    if str(stage or "") == "continual" or bool(train_cfg.get("continual")):
        return [{"name": "default", "epochs": int(train_cfg.get("epochs", 1))}]
    return []


def session_end_epochs(sessions: list[dict[str, Any]], *, default_epochs: int = 1) -> set[int]:
    """0-indexed epoch 边界（每个会话最后一轮）。"""
    cursor = -1
    ends: set[int] = set()
    for session in sessions:
        cursor += int(session.get("epochs", default_epochs) or default_epochs)
        ends.add(cursor)
    return ends


def total_continual_epochs(sessions: list[dict[str, Any]], *, default_epochs: int = 1) -> int:
    if not sessions:
        return int(default_epochs)
    return int(sum(int(session.get("epochs", default_epochs) or default_epochs) for session in sessions))


def absorb_unknown_from_module(pl_module: Any, train_cfg: dict[str, Any] | None = None) -> int:
    """用最近一次 val 收集的未知 embedding 写入低计数原型槽。"""
    cfg = train_cfg or getattr(pl_module, "train_cfg", {}) or {}
    if not bool(cfg.get("absorb_unknown", True if str(getattr(pl_module, "stage", "")) == "continual" else False)):
        return 0
    registry = getattr(getattr(pl_module, "model", None), "prototype_registry", None)
    if registry is None:
        return 0
    embeds = getattr(pl_module, "_val_absorb_embeds", None) or []
    scores = getattr(pl_module, "_val_absorb_scores", None) or []
    if not embeds:
        return 0
    embedding = torch.cat([item if torch.is_tensor(item) else torch.as_tensor(item) for item in embeds], dim=0)
    unknown_mask = None
    threshold = cfg.get("openset_absorb_threshold")
    if scores and threshold is not None:
        score = torch.cat([item if torch.is_tensor(item) else torch.as_tensor(item) for item in scores], dim=0)
        n = min(int(embedding.shape[0]), int(score.shape[0]))
        unknown_mask = score[:n] >= float(threshold)
        embedding = embedding[:n]
    device = next(registry.parameters()).device
    embedding = embedding.to(device=device)
    if unknown_mask is not None:
        unknown_mask = unknown_mask.to(device=device)
    ns = str(cfg.get("absorb_namespace") or "modulation")
    n_slots = absorb_unknown_embeddings(
        registry,
        embedding,
        namespace=ns,
        unknown_mask=unknown_mask,
        min_count=float(cfg.get("absorb_min_count", 2.0)),
    )
    if hasattr(pl_module, "_val_absorb_embeds"):
        pl_module._val_absorb_embeds = []
        pl_module._val_absorb_scores = []
    return int(n_slots)


try:
    from lightning.pytorch.callbacks import Callback
except ImportError:  # pragma: no cover
    Callback = object  # type: ignore[misc, assignment]


class ContinualSessionCallback(Callback):
    """会话边界吸收未知簇，并刷新蒸馏 teacher / 冻结原型。"""

    def __init__(self, sessions: list[dict[str, Any]], train_cfg: dict[str, Any]) -> None:
        self.sessions = list(sessions)
        self.train_cfg = train_cfg
        self._ends = session_end_epochs(sessions, default_epochs=int(train_cfg.get("epochs", 1)))
        self._done: set[int] = set()

    def _run(self, pl_module: Any, *, tag: str, epoch: int) -> None:
        n = absorb_unknown_from_module(pl_module, self.train_cfg)
        refresh = getattr(pl_module, "refresh_continual_teacher", None)
        if callable(refresh):
            refresh()
        import logging

        logging.getLogger("resmamba").info("continual %s epoch=%s absorbed_slots=%s", tag, epoch, n)

    def on_validation_epoch_end(self, trainer: Any, pl_module: Any) -> None:
        if getattr(trainer, "sanity_checking", False):
            return
        epoch = int(getattr(trainer, "current_epoch", 0) or 0)
        if epoch not in self._ends or epoch in self._done:
            return
        self._run(pl_module, tag="session_end", epoch=epoch)
        self._done.add(epoch)

    def on_fit_end(self, trainer: Any, pl_module: Any) -> None:
        epoch = int(getattr(trainer, "current_epoch", 0) or 0)
        if epoch in self._done:
            return
        self._run(pl_module, tag="fit_end", epoch=epoch)
        self._done.add(epoch)
