from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from resmamba_signal_model.models.task_interface import TaskFeatures

LEGACY_RECOGNITION_SHARED = "recognition_heads.shared"


def _shared_residual_mlp(
    d_model: int,
    dropout: float,
    *,
    low_rank_prototype: bool = False,
    prototype_rank: int = 64,
) -> nn.Sequential:
    if low_rank_prototype:
        rank = max(1, min(int(prototype_rank), int(d_model)))
        return nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, rank),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(rank, d_model),
        )
    return nn.Sequential(
        nn.LayerNorm(d_model),
        nn.Linear(d_model, d_model * 2),
        nn.GELU(),
        nn.Dropout(dropout),
        nn.Linear(d_model * 2, d_model),
    )


def remap_legacy_recognition_shared(state: dict[str, Any]) -> dict[str, Any]:
    """旧 checkpoint 的 ``recognition_heads.shared.*`` 不再映射到新头；仅删除遗留键。"""
    remapped = dict(state)
    for key in list(state):
        if key == LEGACY_RECOGNITION_SHARED or key.startswith(LEGACY_RECOGNITION_SHARED + "."):
            remapped.pop(key, None)
    return remapped


def remap_recognition_heads_to_task_heads(state: dict[str, Any]) -> dict[str, Any]:
    """旧 ``modulation_head`` / ``emitter_head`` → 新任务头命名（加载 init 时兼容）。"""
    remapped = dict(state)
    legacy_map = (
        ("modulation_head.", "tx_modulation_head."),
        ("emitter_head.", "ld_model_head."),
        ("clustering_head.", "ld_clustering_head."),
    )
    for old_prefix, new_prefix in legacy_map:
        for key in list(state):
            if key.startswith(old_prefix):
                new_key = new_prefix + key[len(old_prefix) :]
                if new_key not in remapped:
                    remapped[new_key] = state[key]
    return remapped


def remap_task_head_checkpoints(state: dict[str, Any]) -> dict[str, Any]:
    """旧 RecognitionHeads 权重 → 新五头命名（先拆 shared，再改前缀）。"""
    return remap_recognition_heads_to_task_heads(remap_legacy_recognition_shared(state))


class ResidualMLPBlock(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, hidden_dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden_dim, dim), nn.Dropout(dropout))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x.float())


class MLPHead(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        hidden_dim: int | None = None,
        dropout: float = 0.1,
        depth: int = 2,
        low_rank_prototype: bool = False,
        prototype_rank: int = 64,
    ) -> None:
        super().__init__()
        self.low_rank_prototype = bool(low_rank_prototype)
        if self.low_rank_prototype:
            rank = max(1, int(prototype_rank))
            self.net = nn.Sequential(
                nn.LayerNorm(in_dim),
                nn.Linear(in_dim, rank),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(rank, out_dim),
            )
            return
        h = hidden_dim or max(in_dim * 2, 256)
        blocks = [ResidualMLPBlock(in_dim, h, dropout=dropout) for _ in range(depth)]
        self.net = nn.Sequential(*blocks, nn.LayerNorm(in_dim), nn.Linear(in_dim, h), nn.GELU(), nn.Dropout(dropout), nn.Linear(h, out_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x.float())


class CosineClassifierHead(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        hidden_dim: int | None = None,
        dropout: float = 0.1,
        depth: int = 2,
        scale: float = 16.0,
        low_rank_prototype: bool = False,
        prototype_rank: int = 64,
    ) -> None:
        super().__init__()
        self.low_rank_prototype = bool(low_rank_prototype)
        if self.low_rank_prototype:
            rank = max(1, int(prototype_rank))
            self.features = nn.Sequential(
                nn.LayerNorm(in_dim),
                nn.Linear(in_dim, rank),
                nn.GELU(),
                nn.Dropout(dropout),
            )
            self.weight = nn.Parameter(torch.empty(out_dim, rank))
        else:
            h = hidden_dim or max(in_dim * 2, 256)
            self.features = nn.Sequential(
                *[ResidualMLPBlock(in_dim, h, dropout=dropout) for _ in range(depth)],
                nn.LayerNorm(in_dim),
                nn.Linear(in_dim, h),
                nn.GELU(),
                nn.Dropout(dropout),
            )
            self.weight = nn.Parameter(torch.empty(out_dim, h))
        self.scale = nn.Parameter(torch.tensor(float(scale)))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = F.normalize(self.features(x.float()), dim=-1)
        weight = F.normalize(self.weight, dim=-1)
        return self.scale.clamp_min(1.0) * features @ weight.t()


def _as_pooled(h: torch.Tensor | TaskFeatures) -> torch.Tensor:
    return h.pooled if isinstance(h, TaskFeatures) else h


def _as_tokens(h: torch.Tensor | TaskFeatures) -> torch.Tensor:
    if isinstance(h, TaskFeatures):
        if h.readout == "query" and h.query is not None:
            return h.query
        return h.tokens
    return h


def _safe_l2_normalize(x: torch.Tensor, dim: int = -1, eps: float = 1.0e-6) -> torch.Tensor:
    x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    return F.normalize(x, p=2.0, dim=dim, eps=eps)


class RecognitionHeads(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_mod_classes: int = 31,
        num_emitters: int = 100,
        dropout: float = 0.1,
        num_datasets: int = 32,
        use_dataset_bias: bool = True,
        low_rank_prototype: bool = False,
        prototype_rank: int = 64,
    ) -> None:
        super().__init__()
        self.use_dataset_bias = use_dataset_bias
        # 调制要抑制硬件指纹、个体要保留，禁止共用同一层 residual MLP。
        self.shared_modulation = _shared_residual_mlp(
            d_model, dropout, low_rank_prototype=low_rank_prototype, prototype_rank=prototype_rank
        )
        self.shared_emitter = _shared_residual_mlp(
            d_model, dropout, low_rank_prototype=low_rank_prototype, prototype_rank=prototype_rank
        )
        self.modulation = CosineClassifierHead(
            d_model,
            num_mod_classes,
            hidden_dim=d_model * 2,
            dropout=dropout,
            depth=2,
            low_rank_prototype=low_rank_prototype,
            prototype_rank=prototype_rank,
        )
        self.emitter = CosineClassifierHead(
            d_model,
            num_emitters,
            hidden_dim=d_model * 4,
            dropout=dropout,
            depth=3,
            low_rank_prototype=low_rank_prototype,
            prototype_rank=prototype_rank,
        )
        if use_dataset_bias:
            self.modulation_dataset_bias = nn.Embedding(num_datasets, num_mod_classes)
            nn.init.zeros_(self.modulation_dataset_bias.weight)

    def forward(
        self,
        h: torch.Tensor,
        dataset_id: torch.Tensor | None = None,
        *,
        heads: str = "both",
    ) -> dict[str, torch.Tensor]:
        out: dict[str, torch.Tensor] = {}
        if heads in ("both", "modulation"):
            h_mod = h + self.shared_modulation(h)
            modulation_logits = self.modulation(h_mod)
            if self.use_dataset_bias and dataset_id is not None:
                dataset_id = dataset_id.long().clamp(0, self.modulation_dataset_bias.num_embeddings - 1)
                modulation_logits = modulation_logits + self.modulation_dataset_bias(dataset_id)
            out["modulation_logits"] = modulation_logits
        if heads in ("both", "emitter"):
            h_emit = h + self.shared_emitter(h)
            out["emitter_logits"] = self.emitter(h_emit)
        return out


class ClassificationHead(nn.Module):
    """通用 Cosine 分类头；每任务独立实例。"""

    def __init__(
        self,
        d_model: int,
        num_classes: int,
        dropout: float = 0.1,
        logits_key: str | None = None,
        num_datasets: int = 32,
        use_dataset_bias: bool = False,
        low_rank_prototype: bool = False,
        prototype_rank: int = 64,
    ) -> None:
        super().__init__()
        self.use_dataset_bias = use_dataset_bias
        self.logits_key = str(logits_key or "task_logits")
        self.shared = _shared_residual_mlp(
            d_model, dropout, low_rank_prototype=low_rank_prototype, prototype_rank=prototype_rank
        )
        self.classifier = CosineClassifierHead(
            d_model,
            num_classes,
            hidden_dim=d_model * 2,
            dropout=dropout,
            depth=2,
            low_rank_prototype=low_rank_prototype,
            prototype_rank=prototype_rank,
        )
        if use_dataset_bias:
            self.dataset_bias = nn.Embedding(num_datasets, num_classes)
            nn.init.zeros_(self.dataset_bias.weight)
        self.dataset_class_mask: torch.Tensor | None = None

    def set_dataset_class_mask(self, mask: torch.Tensor | None) -> None:
        self.dataset_class_mask = mask.to(dtype=torch.bool) if mask is not None else None

    def forward(
        self,
        features: torch.Tensor | TaskFeatures,
        dataset_id: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        h = _as_pooled(features)
        logits = self.classifier(h + self.shared(h))
        if self.use_dataset_bias and dataset_id is not None:
            dataset_id = dataset_id.long().clamp(0, self.dataset_bias.num_embeddings - 1)
            logits = logits + self.dataset_bias(dataset_id)
        if self.dataset_class_mask is not None:
            logits = apply_emitter_dataset_mask(logits, dataset_id, self.dataset_class_mask)
        return {"task_logits": logits, self.logits_key: logits}


class ModulationHead(nn.Module):
    """浅 Cosine 分类；可选 dataset bias。消费 UTI pooled。"""

    def __init__(
        self,
        d_model: int,
        num_mod_classes: int = 31,
        dropout: float = 0.1,
        num_datasets: int = 32,
        use_dataset_bias: bool = False,
        logits_key: str = "modulation_logits",
        low_rank_prototype: bool = False,
        prototype_rank: int = 64,
    ) -> None:
        super().__init__()
        self.use_dataset_bias = use_dataset_bias
        self.logits_key = str(logits_key)
        self.shared = _shared_residual_mlp(
            d_model, dropout, low_rank_prototype=low_rank_prototype, prototype_rank=prototype_rank
        )
        self.classifier = CosineClassifierHead(
            d_model,
            num_mod_classes,
            hidden_dim=d_model * 2,
            dropout=dropout,
            depth=2,
            low_rank_prototype=low_rank_prototype,
            prototype_rank=prototype_rank,
        )
        if use_dataset_bias:
            self.dataset_bias = nn.Embedding(num_datasets, num_mod_classes)
            nn.init.zeros_(self.dataset_bias.weight)

    def forward(self, features: torch.Tensor | TaskFeatures, dataset_id: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        h = _as_pooled(features)
        logits = self.classifier(h + self.shared(h))
        if self.use_dataset_bias and dataset_id is not None:
            dataset_id = dataset_id.long().clamp(0, self.dataset_bias.num_embeddings - 1)
            logits = logits + self.dataset_bias(dataset_id)
        out = {"task_logits": logits, self.logits_key: logits}
        if self.logits_key != "modulation_logits":
            out["modulation_logits"] = logits
        return out


def apply_dataset_class_mask(
    logits: torch.Tensor,
    dataset_id: torch.Tensor | None,
    class_mask: torch.Tensor | None,
    *,
    blocked_value: float = -1.0e4,
) -> torch.Tensor:
    """按 ``dataset_id`` 只保留该数据集的类；全空行不掩，避免 -inf argmax。"""
    if dataset_id is None or class_mask is None or logits.ndim != 2:
        return logits
    idx = dataset_id.long().reshape(-1)
    if idx.numel() != int(logits.shape[0]):
        return logits
    n_ds, n_cls = int(class_mask.shape[0]), int(class_mask.shape[1])
    if int(logits.shape[-1]) != n_cls or n_ds < 1:
        return logits
    idx = idx.clamp(min=0, max=n_ds - 1)
    allowed = class_mask.to(device=logits.device, dtype=torch.bool)[idx]
    empty = ~allowed.any(dim=-1, keepdim=True)
    allowed = allowed | empty
    return logits.masked_fill(~allowed, logits.new_tensor(blocked_value))


def apply_emitter_dataset_mask(
    logits: torch.Tensor,
    dataset_id: torch.Tensor | None,
    class_mask: torch.Tensor | None,
    *,
    blocked_value: float = -1.0e4,
) -> torch.Tensor:
    return apply_dataset_class_mask(logits, dataset_id, class_mask, blocked_value=blocked_value)


class ClassificationHead(nn.Module):
    """通用 Cosine 分类头；可选 dataset bias / dataset class mask。"""

    def __init__(
        self,
        d_model: int,
        num_classes: int = 31,
        dropout: float = 0.1,
        num_datasets: int = 32,
        use_dataset_bias: bool = False,
        logits_key: str = "task_logits",
        low_rank_prototype: bool = False,
        prototype_rank: int = 64,
        hidden_depth: int = 2,
        hidden_scale: int = 2,
    ) -> None:
        super().__init__()
        self.use_dataset_bias = use_dataset_bias
        self.logits_key = str(logits_key)
        self.num_classes = int(num_classes)
        self.shared = _shared_residual_mlp(
            d_model, dropout, low_rank_prototype=low_rank_prototype, prototype_rank=prototype_rank
        )
        self.classifier = CosineClassifierHead(
            d_model,
            num_classes,
            hidden_dim=d_model * hidden_scale,
            dropout=dropout,
            depth=hidden_depth,
            low_rank_prototype=low_rank_prototype,
            prototype_rank=prototype_rank,
        )
        if use_dataset_bias:
            self.dataset_bias = nn.Embedding(num_datasets, num_classes)
            nn.init.zeros_(self.dataset_bias.weight)

    def set_dataset_class_mask(self, mask: torch.Tensor | None) -> None:
        if "dataset_class_mask" in self._buffers:
            del self._buffers["dataset_class_mask"]
        if mask is None:
            return
        self.register_buffer("dataset_class_mask", mask.to(dtype=torch.bool), persistent=False)

    def forward(self, features: torch.Tensor | TaskFeatures, dataset_id: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        h = _as_pooled(features)
        logits = self.classifier(h + self.shared(h))
        logits = apply_dataset_class_mask(
            logits, dataset_id, getattr(self, "dataset_class_mask", None)
        )
        if self.use_dataset_bias and dataset_id is not None:
            dataset_id = dataset_id.long().clamp(0, self.dataset_bias.num_embeddings - 1)
            logits = logits + self.dataset_bias(dataset_id)
        return {"task_logits": logits, self.logits_key: logits}


class EmitterHead(nn.Module):
    """低秩原型分类（默认）或深层 Cosine 头；禁止与调制共用 MLP。消费 UTI ``source`` 视图。"""

    def __init__(
        self,
        d_model: int,
        num_emitters: int = 100,
        dropout: float = 0.1,
        low_rank_prototype: bool = False,
        prototype_rank: int = 64,
        uti_logit_scale: float = 0.05,
    ) -> None:
        super().__init__()
        self.shared = _shared_residual_mlp(
            d_model, dropout, low_rank_prototype=low_rank_prototype, prototype_rank=prototype_rank
        )
        self.classifier = CosineClassifierHead(
            d_model,
            num_emitters,
            hidden_dim=d_model * 4,
            dropout=dropout,
            depth=3,
            low_rank_prototype=low_rank_prototype,
            prototype_rank=prototype_rank,
        )
        self.uti_logit_scale = nn.Parameter(torch.tensor(float(uti_logit_scale)))

    def set_dataset_class_mask(self, mask: torch.Tensor | None) -> None:
        if "dataset_class_mask" in self._buffers:
            del self._buffers["dataset_class_mask"]
        if mask is None:
            return
        self.register_buffer("dataset_class_mask", mask.to(dtype=torch.bool), persistent=False)

    def forward(
        self,
        features: torch.Tensor | TaskFeatures,
        dataset_id: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        h = _as_pooled(features)
        logits = self.classifier(h + self.shared(h))
        logits = apply_emitter_dataset_mask(
            logits, dataset_id, getattr(self, "dataset_class_mask", None)
        )
        return {"emitter_logits": logits, "task_logits": logits}


class PrototypeClusteringHead(nn.Module):
    def __init__(
        self,
        d_model: int,
        proj_dim: int = 128,
        num_prototypes: int = 32,
        temperature: float = 0.1,
        view_dropout: float = 0.1,
        low_rank_prototype: bool = False,
        prototype_rank: int = 64,
        namespace: str = "clustering",
    ) -> None:
        super().__init__()
        self.namespace = str(namespace)
        self.proj = MLPHead(
            d_model,
            proj_dim,
            hidden_dim=d_model * 2,
            low_rank_prototype=low_rank_prototype,
            prototype_rank=prototype_rank,
        )
        self.prototypes = nn.Parameter(torch.randn(num_prototypes, proj_dim) * 0.02)
        self.temperature = float(temperature)
        self.view_dropout = nn.Dropout(view_dropout)

    def forward(
        self,
        h: torch.Tensor | TaskFeatures,
        registry: Any | None = None,
        namespace: str | None = None,
        temperature: float | None = None,
    ) -> dict[str, torch.Tensor]:
        h = _as_pooled(h)
        tau = max(float(self.temperature if temperature is None else temperature), 1.0e-6)
        z = _safe_l2_normalize(self.proj(h))
        proto = _safe_l2_normalize(self.prototypes)
        logits = z @ proto.t() / tau
        out: dict[str, torch.Tensor] = {
            "cluster_embedding": z,
            "cluster_logits": logits,
            "cluster_probs": logits.softmax(dim=-1),
            "cluster_prototypes": proto,
            "openset_energy": -tau * torch.logsumexp(logits.float(), dim=-1),
        }
        if self.training:
            z2 = _safe_l2_normalize(self.proj(self.view_dropout(h)))
            out["cluster_embedding_view2"] = z2
            out["cluster_logits_view2"] = z2 @ proto.t() / tau
        ns = namespace or self.namespace
        if registry is not None and z.shape[-1] == getattr(registry, "dim", -1):
            out.update(registry.score(ns, z, logits, temperature=tau))
        return out


class _PatchIQHead(nn.Module):
    def __init__(
        self,
        d_model: int,
        patch_size: int,
        dropout: float = 0.1,
        low_rank_prototype: bool = False,
        prototype_rank: int = 64,
    ) -> None:
        super().__init__()
        self.patch_size = patch_size
        if low_rank_prototype:
            rank = max(1, min(int(prototype_rank), int(d_model)))
            self.net = nn.Sequential(
                nn.LayerNorm(d_model),
                nn.Linear(d_model, rank),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(rank, 2 * patch_size),
            )
        else:
            self.net = nn.Sequential(
                nn.LayerNorm(d_model),
                nn.Linear(d_model, d_model),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d_model, 2 * patch_size),
            )

    def project(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.net(tokens.float()).view(*tokens.shape[:-1], 2, self.patch_size)


class PredictionHead(_PatchIQHead):
    """token MLP，在 suffix 位置出 IQ patch（损失侧再 mask）。"""

    def __init__(
        self,
        d_model: int,
        patch_size: int,
        dropout: float = 0.1,
        low_rank_prototype: bool = False,
        prototype_rank: int = 64,
    ) -> None:
        super().__init__(
            d_model,
            patch_size,
            dropout=dropout,
            low_rank_prototype=low_rank_prototype,
            prototype_rank=prototype_rank,
        )

    def forward(self, features: torch.Tensor | TaskFeatures) -> dict[str, torch.Tensor]:
        pred = self.project(_as_tokens(features))
        return {"pred_patches": pred, "prediction_patches": pred}


class ImputationHead(_PatchIQHead):
    """token MLP + mask embedding，在 span 位置出 IQ patch。"""

    def __init__(
        self,
        d_model: int,
        patch_size: int,
        dropout: float = 0.1,
        low_rank_prototype: bool = False,
        prototype_rank: int = 64,
    ) -> None:
        super().__init__(
            d_model,
            patch_size,
            dropout=dropout,
            low_rank_prototype=low_rank_prototype,
            prototype_rank=prototype_rank,
        )
        self.mask_embed = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.normal_(self.mask_embed, std=0.02)

    def forward(
        self,
        features: torch.Tensor | TaskFeatures,
        span_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        tokens = _as_tokens(features)
        if span_mask is None and isinstance(features, TaskFeatures):
            span_mask = features.mask
        if span_mask is not None:
            tokens = tokens + span_mask.to(dtype=tokens.dtype).unsqueeze(-1) * self.mask_embed.to(dtype=tokens.dtype)
        pred = self.project(tokens)
        return {"pred_patches": pred, "imputation_patches": pred}


TASK_HEAD_REGISTRY: dict[str, type[nn.Module]] = {
    "ld_intrapulse": ClassificationHead,
    "ld_model": ClassificationHead,
    "tx_modulation": ClassificationHead,
    "ld_clustering": PrototypeClusteringHead,
    "tx_clustering": PrototypeClusteringHead,
    "prediction": PredictionHead,
}


def register_task(name: str, head_cls: type[nn.Module]) -> None:
    """新领域拓展：注册任务名与头类型（模型侧再加 UTI 行 / adapter）。"""
    TASK_HEAD_REGISTRY[str(name)] = head_cls
