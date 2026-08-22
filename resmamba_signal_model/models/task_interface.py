from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

DEFAULT_TASKS: tuple[str, ...] = (
    "modulation",
    "emitter",
    "clustering",
    "prediction",
    "imputation",
)

TASK_TO_SOURCE: dict[str, str] = {
    "modulation": "classification",
    "emitter": "emitter",
    "clustering": "clustering",
    "prediction": "prediction",
    "imputation": "imputation",
}

SOURCE_TO_TASK: dict[str, str] = {
    "classification": "modulation",
    "clustering": "clustering",
    "prediction": "prediction",
    "imputation": "imputation",
    "modulation": "modulation",
    "emitter": "emitter",
}


FAMILY_TO_ID = {
    "classification": 0,
    "clustering": 1,
    "generation": 2,
    "sequence_regression": 3,
    "dense_prediction": 4,
}
READOUT_TO_ID = {"pooled": 0, "token": 1, "query": 2}
VIEW_TO_ID = {"general": 0, "semantic": 1, "source": 2, "context": 3, "mixed": 4}
MODALITY_TO_ID = {
    "generic": 0,
    "rf": 1,
    "sonar": 2,
    "acoustic": 2,
    "imu": 3,
    "navigation": 3,
}
SPECIALIST_VIEWS = ("semantic", "source", "context")


@dataclass
class TaskSpec:
    """模型侧任务契约；不依赖训练目录，可由字典或旧 task name 构造。"""

    name: str
    family: str = "classification"
    readout: str = "pooled"
    view: str | tuple[str, ...] = "general"
    modality: str = "generic"
    metadata: dict[str, float] = field(default_factory=dict)
    invariant_views: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        self.name = str(self.name)
        self.family = str(self.family).lower()
        self.readout = str(self.readout).lower()
        self.modality = str(self.modality).lower()
        if isinstance(self.view, str):
            self.view = self.view.lower()
        else:
            self.view = tuple(str(item).lower() for item in self.view)
        if isinstance(self.invariant_views, str):
            self.invariant_views = (self.invariant_views.lower(),)
        else:
            self.invariant_views = tuple(str(item).lower() for item in self.invariant_views)
        if self.family not in FAMILY_TO_ID:
            raise ValueError(f"未知 task family {self.family!r}")
        if self.readout not in READOUT_TO_ID:
            raise ValueError(f"未知 task readout {self.readout!r}")
        requested = (self.view,) if isinstance(self.view, str) else self.view
        unknown = [name for name in requested if name not in VIEW_TO_ID]
        if unknown:
            raise ValueError(f"未知 task view: {unknown}")
        unknown_inv = [name for name in self.invariant_views if name not in VIEW_TO_ID]
        if unknown_inv:
            raise ValueError(f"未知 invariant view: {unknown_inv}")


def default_task_spec(name: str, kind: str | None = None) -> TaskSpec:
    key = str(name)
    kind = str(kind or TASK_TO_SOURCE.get(key, "classification")).lower()
    if key == "modulation":
        return TaskSpec(key, "classification", "pooled", "semantic", "rf", invariant_views=("semantic",))
    if key == "emitter":
        return TaskSpec(key, "classification", "pooled", "source", "rf")
    if key == "clustering":
        return TaskSpec(key, "clustering", "pooled", "mixed", "rf", invariant_views=("semantic",))
    if key == "pretrain" or kind in ("pretrain", "mae"):
        return TaskSpec(key, "generation", "query", "general", "generic", invariant_views=("semantic",))
    if kind in ("prediction", "imputation", "generation"):
        return TaskSpec(key, "generation", "query", "general", "rf" if key in DEFAULT_TASKS else "generic")
    if kind in ("sequence_regression", "regression"):
        return TaskSpec(key, "sequence_regression", "token", "general", "generic")
    if kind in ("dense_prediction", "dense"):
        return TaskSpec(key, "dense_prediction", "token", "general", "generic")
    return TaskSpec(key, "classification", "pooled", "semantic", "generic")


@dataclass
class TaskFeatures:
    """UTI 标准读出；前三个字段保持旧构造 API。"""

    pooled: torch.Tensor
    tokens: torch.Tensor
    mask: torch.Tensor
    query: torch.Tensor | None = None
    readout: str = "pooled"
    task_vec: torch.Tensor | None = None
    views: dict[str, torch.Tensor] | None = None


class HoulsbyStem(nn.Module):
    """Houlsby 瓶颈残差：LN → Linear(d, r) → GELU → Linear(r, d)。"""

    def __init__(self, d_model: int, rank: int) -> None:
        super().__init__()
        rank = max(1, min(int(rank), int(d_model)))
        self.net = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, rank),
            nn.GELU(),
            nn.Linear(rank, d_model),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x.float()).to(dtype=x.dtype)


class LowRankResidualView(nn.Module):
    """不覆盖 general，仅学习低秩残差专家。"""

    def __init__(self, d_model: int, rank: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.down = nn.Linear(d_model, rank)
        self.up = nn.Linear(rank, d_model)
        nn.init.normal_(self.up.weight, std=0.01)
        nn.init.zeros_(self.up.bias)

    def forward(self, general: torch.Tensor) -> torch.Tensor:
        residual = self.up(F.gelu(self.down(self.norm(general.float())))).to(dtype=general.dtype)
        return general + residual


class UniversalTaskInterfaceV2(nn.Module):
    """由任务契约组合条件，并统一提供 pooled/token/query 三种读出。"""

    def __init__(
        self,
        d_model: int,
        *,
        rank: int = 64,
        num_task_types: int = 8,
        dropout: float = 0.0,
        task_names: tuple[str, ...] | list[str] = DEFAULT_TASKS,
        task_specs: Mapping[str, TaskSpec | Mapping[str, Any]] | None = None,
        metadata_dim: int = 8,
        legacy_mode: bool = False,
        use_specialist_views: bool = True,
        domain_prompt_size: int = 6,
        num_datasets: int = 32,
    ) -> None:
        super().__init__()
        self.d_model = int(d_model)
        self.rank = max(1, min(int(rank), int(d_model)))
        self.metadata_dim = max(1, int(metadata_dim))
        self.num_task_types = max(int(num_task_types), len(task_names))
        self.legacy_mode = bool(legacy_mode)
        self.use_specialist_views = bool(use_specialist_views)
        self.task_to_id: dict[str, int] = {str(name): i for i, name in enumerate(task_names)}
        self.id_to_task: list[str] = [str(name) for name in task_names]
        self.task_specs: dict[str, TaskSpec] = {
            str(name): default_task_spec(str(name)) for name in task_names
        }
        for name, spec in dict(task_specs or {}).items():
            self.task_specs[str(name)] = self._coerce_spec(spec, name=str(name))

        # stem/task_embed 保留旧 key；v2 不使用无语义 task_embed 做条件。
        self.stem = HoulsbyStem(d_model, self.rank)
        self.task_embed = nn.Embedding(self.num_task_types, d_model)
        nn.init.normal_(self.task_embed.weight, std=0.02)

        self.family_embed = nn.Embedding(len(FAMILY_TO_ID), self.rank)
        self.readout_embed = nn.Embedding(len(READOUT_TO_ID), self.rank)
        self.view_embed = nn.Embedding(len(VIEW_TO_ID), self.rank)
        self.modality_embed = nn.Embedding(max(MODALITY_TO_ID.values()) + 1, self.rank)
        self.meta_mlp = nn.Sequential(
            nn.Linear(self.metadata_dim, self.rank),
            nn.SiLU(),
            nn.Linear(self.rank, self.rank),
        )
        prompt_size = max(4, min(int(domain_prompt_size), 8))
        self.domain_prompt_size = prompt_size
        self.domain_prompts = nn.Parameter(torch.randn(num_datasets, prompt_size, self.rank) * 0.02)
        self.condition_norm = nn.LayerNorm(self.rank)
        self.feature_down = nn.Linear(d_model, self.rank)
        self.condition_film = nn.Linear(self.rank, self.rank * 2)
        self.feature_up = nn.Linear(self.rank, d_model)
        self.view_gate = nn.Linear(self.rank, len(SPECIALIST_VIEWS))
        self.pool_gate = nn.Linear(self.rank, 1)
        self.coord_down = nn.Linear(2, self.rank)
        self.coord_up = nn.Linear(self.rank, d_model)
        self.view_adapters = nn.ModuleDict(
            {
                name: LowRankResidualView(d_model, self.rank)
                for name in SPECIALIST_VIEWS
            }
        )
        # 慢视图：对 token 做因果指数衰减再均值，与 semantic/source 的聚合不同
        self.register_buffer("_context_decay", torch.tensor(0.95), persistent=False)
        self.dropout = nn.Dropout(dropout)

        # 旧 UTI v1 路径仅在兼容开关下构建，旧 checkpoint 可无损复现。
        if self.legacy_mode:
            from resmamba_signal_model.models.decoder import AttentionPooling

            self.film = nn.Linear(d_model, d_model * 2)
            self.token_pool = AttentionPooling(d_model, num_heads=min(4, max(1, d_model // 16) or 1))
            self.fuse = nn.Linear(d_model * 2, d_model)
            nn.init.zeros_(self.film.bias)

    @staticmethod
    def _coerce_spec(
        spec: TaskSpec | Mapping[str, Any] | Any,
        *,
        name: str | None = None,
    ) -> TaskSpec:
        if isinstance(spec, TaskSpec):
            return spec
        if isinstance(spec, Mapping):
            raw = dict(spec)
        else:
            raw = {
                key: getattr(spec, key)
                for key in ("name", "family", "kind", "readout", "view", "modality", "metadata")
                if hasattr(spec, key)
            }
            if hasattr(spec, "extra") and isinstance(getattr(spec, "extra"), Mapping):
                raw.update(dict(getattr(spec, "extra")))
        extra = raw.get("extra")
        if isinstance(extra, Mapping):
            raw = {**dict(extra), **raw}
        spec_name = str(raw.get("name", name or "task"))
        family = str(raw.get("family", raw.get("kind", ""))).lower()
        base = default_task_spec(spec_name, family or None)
        if not family:
            return base
        if family in ("prediction", "imputation"):
            family = "generation"
        elif family in ("emitter", "modulation"):
            family = "classification"
        elif family == "regression":
            family = "sequence_regression"
        elif family == "dense":
            family = "dense_prediction"
        inv = raw.get("invariant_views", base.invariant_views)
        return TaskSpec(
            name=spec_name,
            family=family,
            readout=str(raw.get("readout", base.readout)),
            view=raw.get("view", base.view),
            modality=str(raw.get("modality", base.modality)),
            metadata=dict(raw.get("metadata") or base.metadata),
            invariant_views=tuple(inv) if inv is not None else (),
        )

    def _expand_task_embed(self, new_n: int) -> None:
        old = self.task_embed
        new_n = max(int(new_n), int(old.num_embeddings) + 1)
        expanded = nn.Embedding(new_n, old.embedding_dim)
        expanded = expanded.to(device=old.weight.device, dtype=old.weight.dtype)
        with torch.no_grad():
            expanded.weight[: old.num_embeddings].copy_(old.weight)
            nn.init.normal_(expanded.weight[old.num_embeddings :], std=0.02)
        self.task_embed = expanded
        self.num_task_types = new_n

    def add_task(
        self,
        name: str,
        spec: TaskSpec | Mapping[str, Any] | Any | None = None,
    ) -> int:
        name = str(name)
        if name in self.task_to_id:
            if spec is not None:
                self.task_specs[name] = self._coerce_spec(spec, name=name)
            return self.task_to_id[name]
        idx = len(self.task_to_id)
        if idx >= self.task_embed.num_embeddings:
            self._expand_task_embed(max(idx + 4, self.task_embed.num_embeddings * 2))
        self.task_to_id[name] = idx
        self.id_to_task.append(name)
        self.task_specs[name] = self._coerce_spec(spec, name=name) if spec is not None else default_task_spec(name)
        return idx

    def task_index(self, task: str | int | torch.Tensor | TaskSpec) -> torch.Tensor | int:
        if isinstance(task, TaskSpec):
            task = task.name
        if isinstance(task, torch.Tensor):
            return task.long()
        if isinstance(task, int):
            return task
        if task not in self.task_to_id:
            raise KeyError(f"未知任务 {task!r}，已注册: {list(self.task_to_id)}")
        return self.task_to_id[task]

    def resolve_spec(
        self,
        task: str | int | TaskSpec | Mapping[str, Any],
    ) -> TaskSpec:
        if isinstance(task, TaskSpec):
            return task
        if isinstance(task, Mapping):
            return self._coerce_spec(task)
        if isinstance(task, int):
            idx = max(0, min(int(task), len(self.id_to_task) - 1))
            task = self.id_to_task[idx]
        if task not in self.task_specs:
            raise KeyError(f"任务 {task!r} 尚未注册 TaskSpec")
        return self.task_specs[str(task)]

    def _specs_for_batch(
        self,
        task: str | int | torch.Tensor | TaskSpec | Mapping[str, Any],
        batch: int,
    ) -> list[TaskSpec]:
        if isinstance(task, torch.Tensor):
            ids = task.detach().long().view(-1)
            if ids.numel() == 1:
                ids = ids.expand(batch)
            if ids.numel() != batch:
                raise ValueError(f"task id 数量 {ids.numel()} 与 batch={batch} 不一致")
            return [self.resolve_spec(int(idx)) for idx in ids.tolist()]
        return [self.resolve_spec(task)] * batch

    def _metadata_tensor(
        self,
        specs: Sequence[TaskSpec],
        metadata: torch.Tensor | Mapping[str, Any] | None,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        batch = len(specs)
        if metadata is None:
            rows = []
            for spec in specs:
                values = [float(spec.metadata[key]) for key in sorted(spec.metadata)]
                values = values[: self.metadata_dim] + [0.0] * max(0, self.metadata_dim - len(values))
                rows.append(values)
            return torch.tensor(rows, device=device, dtype=dtype)
        if isinstance(metadata, Mapping):
            columns: list[torch.Tensor] = []
            for key in sorted(metadata):
                value = metadata[key]
                tensor = torch.as_tensor(value, device=device, dtype=dtype).view(-1)
                if tensor.numel() == 1:
                    tensor = tensor.expand(batch)
                columns.append(tensor[:batch])
            result = torch.stack(columns, dim=-1) if columns else torch.zeros(batch, 0, device=device, dtype=dtype)
        else:
            result = metadata.to(device=device, dtype=dtype)
            if result.ndim == 1:
                result = result.unsqueeze(0)
            if result.shape[0] == 1 and batch > 1:
                result = result.expand(batch, -1)
        if result.shape[0] != batch:
            raise ValueError(f"metadata batch={result.shape[0]}，期望 {batch}")
        if result.shape[-1] < self.metadata_dim:
            result = F.pad(result, (0, self.metadata_dim - result.shape[-1]))
        return result[:, : self.metadata_dim]

    def _domain_prompt(self, dataset_id: torch.Tensor | None, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        if dataset_id is None:
            return torch.zeros(0, self.rank, device=device, dtype=dtype)
        ids = dataset_id.long().reshape(-1)
        if ids.numel() == 0:
            return torch.zeros(0, self.rank, device=device, dtype=dtype)
        ids = ids.clamp(0, self.domain_prompts.shape[0] - 1)
        prompts = self.domain_prompts.index_select(0, ids).to(device=device, dtype=dtype)
        return prompts.mean(dim=1)

    def condition_vector(
        self,
        task: str | int | torch.Tensor | TaskSpec | Mapping[str, Any],
        batch: int,
        *,
        device: torch.device,
        dtype: torch.dtype,
        metadata: torch.Tensor | Mapping[str, Any] | None = None,
        dataset_id: torch.Tensor | None = None,
    ) -> torch.Tensor:
        specs = self._specs_for_batch(task, batch)
        family_ids = torch.tensor(
            [FAMILY_TO_ID[spec.family] for spec in specs], device=device, dtype=torch.long
        )
        readout_ids = torch.tensor(
            [READOUT_TO_ID[spec.readout] for spec in specs], device=device, dtype=torch.long
        )
        view_ids = torch.tensor(
            [
                VIEW_TO_ID[
                    spec.view
                    if isinstance(spec.view, str)
                    else ("mixed" if len(spec.view) != 1 else spec.view[0])
                ]
                for spec in specs
            ],
            device=device,
            dtype=torch.long,
        )
        modality_ids = torch.tensor(
            [MODALITY_TO_ID.get(spec.modality, MODALITY_TO_ID["generic"]) for spec in specs],
            device=device,
            dtype=torch.long,
        )
        meta = self._metadata_tensor(specs, metadata, device=device, dtype=dtype)
        condition = (
            self.family_embed(family_ids)
            + self.readout_embed(readout_ids)
            + self.view_embed(view_ids)
            + self.modality_embed(modality_ids)
            + self.meta_mlp(meta)
        )
        if dataset_id is None and isinstance(metadata, Mapping) and "dataset_id" in metadata:
            dataset_id = metadata["dataset_id"]
        if dataset_id is not None:
            domain = self._domain_prompt(dataset_id, device=device, dtype=dtype)
            if domain.shape[0] == 1 and batch > 1:
                domain = domain.expand(batch, -1)
            if domain.shape[0] == batch:
                condition = condition + domain
        return self.condition_norm(condition).to(dtype=dtype)

    @staticmethod
    def _masked_mean(tokens: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        weights = mask.unsqueeze(-1).to(dtype=tokens.dtype)
        denom = weights.sum(dim=1).clamp_min(1.0)
        return (tokens * weights).sum(dim=1) / denom

    def _context_pool(self, tokens: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """长时上下文：时间指数衰减加权均值（相对 semantic 的均匀均值更偏慢变量）。"""
        batch, length, _ = tokens.shape
        decay = float(self._context_decay.item()) if hasattr(self, "_context_decay") else 0.95
        idx = torch.arange(length, device=tokens.device, dtype=tokens.dtype)
        # 近端权重大：w_t ∝ decay^(L-1-t)
        raw = decay ** (float(length - 1) - idx)
        weights = raw.view(1, length, 1).expand(batch, -1, -1) * mask.unsqueeze(-1).to(dtype=tokens.dtype)
        denom = weights.sum(dim=1).clamp_min(1.0e-6)
        return (tokens * weights).sum(dim=1) / denom

    def build_views(
        self,
        z_general: torch.Tensor,
        h_general: torch.Tensor,
        *,
        patch_mask: torch.Tensor | None = None,
    ) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
        """构造 specialist 视图。

        不再是 ``adapter(z_general)`` 的三份拷贝：semantic 为 ``z_enc`` 低秩残差；
        source 为去均值 token 残差（先 adapter 再池化，并加回 ``z_enc``）；
        context 为慢衰减池化后再低秩残差。
        ``general`` 仍对应 encoder 全局读出 ``z_general`` / ``h_general``。
        """
        views: dict[str, tuple[torch.Tensor, torch.Tensor]] = {
            "general": (z_general, h_general)
        }
        if not self.use_specialist_views:
            return views
        if patch_mask is None:
            patch_mask = torch.ones(
                h_general.shape[0],
                h_general.shape[1],
                dtype=torch.bool,
                device=h_general.device,
            )
        else:
            patch_mask = patch_mask.to(device=h_general.device, dtype=torch.bool)
            if patch_mask.shape[:2] != h_general.shape[:2]:
                n = min(patch_mask.shape[1], h_general.shape[1])
                patch_mask = patch_mask[:, :n]
                h_general = h_general[:, :n]

        z_sem = self.view_adapters["semantic"](z_general)
        h_sem = self.view_adapters["semantic"](h_general)

        # source：去均值 token → 低秩残差 → 再池化，并残差加回 z_enc。
        # 旧实现先 pool(h-mean) 再 adapter：均匀 mask 下 mean(h-mean)≡0，
        # z_src 退化为与样本无关的常数，个体头只能落到 1/C 随机水平。
        z_mean = self._masked_mean(h_general, patch_mask)
        h_resid = h_general - z_mean.unsqueeze(1)
        h_src = self.view_adapters["source"](h_resid)
        z_src = self.view_adapters["source"](z_general) + self._masked_mean(h_src, patch_mask)

        z_ctx = self._context_pool(h_general, patch_mask)
        h_ctx = self.view_adapters["context"](h_general)
        z_ctx = self.view_adapters["context"](z_ctx)

        views["semantic"] = (z_sem, h_sem)
        views["source"] = (z_src, h_src)
        views["context"] = (z_ctx, h_ctx)
        return views

    def _allowed_views(self, specs: Sequence[TaskSpec], device: torch.device) -> torch.Tensor:
        allowed = torch.zeros(len(specs), len(SPECIALIST_VIEWS), device=device, dtype=torch.bool)
        for row, spec in enumerate(specs):
            requested = (spec.view,) if isinstance(spec.view, str) else tuple(spec.view)
            if "mixed" in requested:
                allowed[row] = True
                continue
            for name in requested:
                if name in SPECIALIST_VIEWS:
                    allowed[row, SPECIALIST_VIEWS.index(name)] = True
        return allowed

    def _mix_views(
        self,
        general: torch.Tensor,
        views: Mapping[str, torch.Tensor],
        condition: torch.Tensor,
        specs: Sequence[TaskSpec],
    ) -> torch.Tensor:
        allowed = self._allowed_views(specs, general.device)
        if not allowed.any():
            return general
        logits = self.view_gate(condition).masked_fill(~allowed, -1.0e4)
        weights = logits.softmax(dim=-1) * allowed.to(dtype=logits.dtype)
        weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1.0)
        result = general
        for idx, name in enumerate(SPECIALIST_VIEWS):
            if name not in views:
                continue
            weight = weights[:, idx]
            while weight.ndim < general.ndim:
                weight = weight.unsqueeze(1)
            result = result + weight * (views[name] - general)
        return result

    def _modulate(self, h: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        base = self.stem(h)
        low = self.feature_down(base.float())
        gamma, beta = self.condition_film(condition.float()).chunk(2, dim=-1)
        while gamma.ndim < low.ndim:
            gamma = gamma.unsqueeze(1)
            beta = beta.unsqueeze(1)
        residual = self.feature_up((1.0 + gamma.tanh()) * low + beta)
        return base + residual.to(dtype=base.dtype)

    def _legacy_forward(
        self,
        z: torch.Tensor,
        patch_h: torch.Tensor,
        patch_mask: torch.Tensor,
        task_id: str | int | torch.Tensor | TaskSpec,
    ) -> TaskFeatures:
        batch = z.shape[0]
        idx = self.task_index(task_id)
        if isinstance(idx, int):
            ids = torch.full((batch,), int(idx), device=z.device, dtype=torch.long)
        else:
            ids = idx.to(device=z.device).long().view(-1)
            if ids.numel() == 1 and batch > 1:
                ids = ids.expand(batch)
        ids = ids.clamp(0, self.task_embed.num_embeddings - 1)
        task_vec = self.task_embed(ids)
        gamma, beta = self.film(task_vec).chunk(2, dim=-1)

        def film(h: torch.Tensor) -> torch.Tensor:
            g, b = gamma, beta
            while g.ndim < h.ndim:
                g = g.unsqueeze(1)
                b = b.unsqueeze(1)
            return (1.0 + g.tanh()) * h + b

        z_h = film(self.stem(z))
        tok_h = film(self.stem(patch_h))
        tok_h = tok_h.masked_fill(~patch_mask.unsqueeze(-1), 0.0)
        pooled_tok = self.token_pool(tok_h, key_padding_mask=~patch_mask)
        pooled = self.fuse(torch.cat([z_h, pooled_tok], dim=-1))
        pooled = self.dropout(F.normalize(pooled.float(), dim=-1).to(dtype=z.dtype) + z_h)
        return TaskFeatures(
            pooled=pooled,
            tokens=tok_h,
            mask=patch_mask,
            query=tok_h,
            task_vec=task_vec,
        )

    def forward(
        self,
        z: torch.Tensor,
        patch_h: torch.Tensor,
        patch_mask: torch.Tensor,
        task_id: str | int | torch.Tensor | TaskSpec | Mapping[str, Any],
        recon_norm: torch.Tensor | None = None,
        *,
        views: Mapping[str, tuple[torch.Tensor, torch.Tensor]] | None = None,
        metadata: torch.Tensor | Mapping[str, Any] | None = None,
        query_coords: torch.Tensor | None = None,
    ) -> TaskFeatures:
        del recon_norm
        if self.legacy_mode:
            return self._legacy_forward(z, patch_h, patch_mask, task_id)
        batch = z.shape[0]
        specs = self._specs_for_batch(task_id, batch)
        condition = self.condition_vector(
            task_id,
            batch,
            device=z.device,
            dtype=z.dtype,
            metadata=metadata,
            dataset_id=(
                metadata.get("dataset_id")
                if isinstance(metadata, Mapping)
                else None
            ),
        )
        all_views = dict(views or self.build_views(z, patch_h, patch_mask=patch_mask))
        z_views = {name: pair[0] for name, pair in all_views.items()}
        h_views = {name: pair[1] for name, pair in all_views.items()}
        z_mix = self._mix_views(z, z_views, condition, specs)
        h_mix = self._mix_views(patch_h, h_views, condition, specs)
        z_h = self._modulate(z_mix, condition)
        tok_h = self._modulate(h_mix, condition)
        tok_h = tok_h.masked_fill(~patch_mask.unsqueeze(-1), 0.0)

        denom = patch_mask.sum(dim=1, keepdim=True).clamp_min(1).to(dtype=tok_h.dtype)
        pooled_tok = (tok_h * patch_mask.unsqueeze(-1)).sum(dim=1) / denom
        pool_gate = torch.sigmoid(self.pool_gate(condition)).to(dtype=tok_h.dtype)
        pooled = z_h + pool_gate * pooled_tok
        pooled = self.dropout(F.normalize(pooled.float(), dim=-1).to(dtype=z.dtype) + z_h)

        n_tokens = patch_h.shape[1]
        if query_coords is None:
            pos = torch.linspace(0.0, 1.0, n_tokens, device=z.device, dtype=z.dtype)
            pos = pos.view(1, n_tokens, 1).expand(batch, -1, -1)
            query_coords = torch.cat([pos, torch.zeros_like(pos)], dim=-1)
        elif query_coords.ndim == 2:
            query_coords = query_coords.unsqueeze(-1)
        query_coords = query_coords.to(device=z.device, dtype=z.dtype)
        if query_coords.shape[-1] == 1:
            query_coords = torch.cat([query_coords, torch.zeros_like(query_coords)], dim=-1)
        query = tok_h + self.coord_up(F.silu(self.coord_down(query_coords[..., :2].float()))).to(dtype=tok_h.dtype)
        query = query.masked_fill(~patch_mask.unsqueeze(-1), 0.0)
        readout = specs[0].readout if all(spec.readout == specs[0].readout for spec in specs) else "mixed"
        return TaskFeatures(
            pooled=pooled,
            tokens=tok_h,
            mask=patch_mask,
            query=query,
            readout=readout,
            task_vec=condition,
            views={name: value for name, value in h_views.items()},
        )


class UniversalTaskInterface(UniversalTaskInterfaceV2):
    """向后兼容名称；默认启用组合式 UTI v2。"""
