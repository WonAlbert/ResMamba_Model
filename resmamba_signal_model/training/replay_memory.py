from __future__ import annotations

import logging
import random
from collections import defaultdict
from typing import Any, Iterable, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from resmamba_signal_model.data.sampling import dataset_class_ids, n_tokens_for_length
from resmamba_signal_model.training.clustering_labels import resolve_modulation_labels
from resmamba_signal_model.training.data_module import _collate_with_source, train_sampler_label_fields
from resmamba_signal_model.training.task_catalog import TaskCatalog, resolve_task_catalog

_LOG = logging.getLogger("resmamba")


def resolve_replay_strategy(train_cfg: dict[str, Any]) -> str:
    raw = str(train_cfg.get("replay_strategy", "class_center") or "class_center").strip().lower()
    if raw in ("class_center", "center", "exemplar", "icarl"):
        return "class_center"
    if raw in ("random", "uniform", "none"):
        return "random"
    raise ValueError(f"未知 replay_strategy={raw!r}，可选: class_center | random")


def count_replay_classes(dataset: Any) -> int:
    """统计 replay 源有效类别数（``label >= 0``）。"""
    class_ids = dataset_class_ids(dataset)
    valid = {int(x) for x in class_ids if int(x) >= 0}
    return max(1, len(valid))


def resolve_replay_memory_total(train_cfg: dict[str, Any]) -> int:
    """回放记忆库总 exemplar 上限；不随 replay 任务数增长。"""
    explicit = int(train_cfg.get("replay_memory_total_exemplars", 0) or 0)
    if explicit > 0:
        return explicit
    base_per_class = max(1, int(train_cfg.get("replay_samples_per_class", 20) or 20))
    baseline_classes = max(1, int(train_cfg.get("replay_baseline_classes_per_task", 11) or 11))
    return base_per_class * baseline_classes


def resolve_dynamic_samples_per_class(
    train_cfg: dict[str, Any],
    *,
    num_replay_tasks: int,
    num_classes_in_task: int,
) -> tuple[int, int, int]:
    """按 replay 任务数均分总 budget，再在任务内按类别均分。

    返回 ``(samples_per_class, task_budget, total_budget)``。
    """
    n_tasks = max(1, int(num_replay_tasks))
    n_classes = max(1, int(num_classes_in_task))
    total = resolve_replay_memory_total(train_cfg)
    task_budget = max(1, int(total) // n_tasks)
    per_class = max(1, int(task_budget) // n_classes)
    return per_class, task_budget, total


def stratified_scan_indices(class_ids: Sequence[int], max_samples: int, *, seed: int = 0) -> list[int]:
    """按类分层抽取待扫描下标，避免大类独占 scan budget。"""
    n = len(class_ids)
    if max_samples <= 0 or max_samples >= n:
        return list(range(n))
    by_class: dict[int, list[int]] = defaultdict(list)
    unlabeled: list[int] = []
    for idx, label in enumerate(class_ids):
        cid = int(label)
        if cid < 0:
            unlabeled.append(idx)
        else:
            by_class[cid].append(idx)
    if not by_class:
        rng = random.Random(int(seed))
        order = list(range(n))
        rng.shuffle(order)
        return order[:max_samples]
    per_class = max(1, int(max_samples) // max(1, len(by_class)))
    rng = random.Random(int(seed))
    picked: list[int] = []
    for idxs in by_class.values():
        pool = list(idxs)
        rng.shuffle(pool)
        picked.extend(pool[:per_class])
    if len(picked) < max_samples and unlabeled:
        extra = list(unlabeled)
        rng.shuffle(extra)
        picked.extend(extra[: max(0, max_samples - len(picked))])
    if len(picked) < max_samples:
        remain = [i for i in range(n) if i not in set(picked)]
        rng.shuffle(remain)
        picked.extend(remain[: max(0, max_samples - len(picked))])
    return picked[:max_samples]


def select_nearest_class_center_exemplars(
    embeddings: torch.Tensor,
    labels: torch.Tensor,
    sample_indices: torch.Tensor,
    *,
    samples_per_class: int,
) -> list[int]:
    """每类取 embedding 距类均值（归一化余弦）最近的 ``samples_per_class`` 条 exemplar。"""
    if embeddings.ndim != 2 or labels.ndim != 1 or sample_indices.ndim != 1:
        raise ValueError("embeddings/labels/sample_indices 维度不匹配")
    if embeddings.shape[0] != labels.shape[0] or embeddings.shape[0] != sample_indices.shape[0]:
        raise ValueError("embeddings/labels/sample_indices 长度不一致")
    k = max(1, int(samples_per_class))
    selected: list[int] = []
    valid = labels >= 0
    if not valid.any():
        return []
    for cls in torch.unique(labels[valid]).tolist():
        cls = int(cls)
        mask = valid & (labels == cls)
        if not mask.any():
            continue
        cls_emb = embeddings[mask].float()
        cls_idx = sample_indices[mask]
        center = F.normalize(cls_emb.mean(dim=0), dim=0)
        cls_n = F.normalize(cls_emb, dim=-1)
        dist = 1.0 - (cls_n @ center)
        order = dist.argsort()
        take = min(k, int(order.numel()))
        selected.extend(int(x) for x in cls_idx[order[:take]].tolist())
    return selected


def _labels_from_batch(
    batch: dict[str, Any],
    *,
    task: str,
    catalog: TaskCatalog,
) -> torch.Tensor:
    spec = catalog.get(task)
    kind = spec.kind if spec is not None else task
    if kind == "classification" or task == "modulation":
        raw = batch.get(spec.label_field if spec else "canonical_mod_label_id", batch.get("mod_label_id"))
        if raw is None:
            raw = batch.get("source_label_id")
        if raw is None:
            raise KeyError(f"replay 扫描缺少调制标签 task={task!r}")
        return resolve_modulation_labels(raw, batch.get("source_label_id")).view(-1).long()
    if kind == "emitter":
        field = spec.label_field if spec else "global_emitter_id"
        raw = batch.get(field, batch.get("emitter_id"))
        if raw is None:
            raise KeyError(f"replay 扫描缺少 emitter 标签 task={task!r}")
        return raw.view(-1).long()
    fields = train_sampler_label_fields(task)
    for name in fields:
        raw = batch.get(name)
        if raw is None:
            continue
        labels = raw.view(-1).long()
        if bool((labels >= 0).any()):
            return labels
    dataset_id = batch.get("dataset_id")
    if dataset_id is not None:
        return dataset_id.view(-1).long().clamp_min(0)
    raise KeyError(f"replay 扫描无法解析类别 task={task!r}")


def _pack_scan_indices(
    scan_indices: list[int],
    lengths: list[int],
    *,
    patch_size: int,
    max_tokens: int,
) -> list[list[int]]:
    """把扫描下标按 token budget 分组，避免一次 forward 过大。"""
    budget = max(1, int(max_tokens))
    batches: list[list[int]] = []
    cur: list[int] = []
    used = 0
    for idx in scan_indices:
        cost = n_tokens_for_length(int(lengths[idx]), int(patch_size))
        if cur and used + cost > budget:
            batches.append(cur)
            cur = []
            used = 0
        cur.append(int(idx))
        used += cost
        if used >= budget:
            batches.append(cur)
            cur = []
            used = 0
    if cur:
        batches.append(cur)
    return batches


@torch.no_grad()
def build_class_center_replay_memory(
    teacher: nn.Module,
    dataset: Any,
    lengths: list[int],
    *,
    task: str,
    source_name: str,
    device: torch.device,
    patch_size: int,
    train_cfg: dict[str, Any],
    catalog: TaskCatalog | None = None,
    samples_per_class: int | None = None,
) -> list[int]:
    """用 frozen teacher 的 ``task_pooled`` 为 replay 源挑选最近类中心 exemplar 下标。"""
    catalog = catalog or resolve_task_catalog(train_cfg)
    if samples_per_class is None:
        n_classes = count_replay_classes(dataset)
        samples_per_class, _, _ = resolve_dynamic_samples_per_class(
            train_cfg,
            num_replay_tasks=1,
            num_classes_in_task=n_classes,
        )
    samples_per_class = max(1, int(samples_per_class))
    max_scan = int(train_cfg.get("replay_scan_max_samples", 4096) or 4096)
    scan_token_budget = max(256, int(train_cfg.get("replay_scan_token_budget", 2048) or 2048))
    seed = int(train_cfg.get("seed", 0))

    class_ids = dataset_class_ids(dataset)
    scan_indices = stratified_scan_indices(class_ids, max_scan, seed=seed + hash(source_name) % 9973)
    if not scan_indices:
        return []

    batches = _pack_scan_indices(
        scan_indices,
        lengths,
        patch_size=int(patch_size),
        max_tokens=scan_token_budget,
    )
    collate = _collate_with_source(source_name, task)

    emb_parts: list[torch.Tensor] = []
    label_parts: list[torch.Tensor] = []
    index_parts: list[torch.Tensor] = []

    teacher.eval()
    for batch_indices in batches:
        samples = [dataset[int(i)] for i in batch_indices]
        batch = collate(samples)
        batch = _to_device(batch, device)
        out = teacher(batch, mode="task", task=task)
        pooled = out.get("task_pooled", out.get("uti_pooled"))
        if pooled is None:
            continue
        labels = _labels_from_batch(batch, task=task, catalog=catalog)
        n = min(int(pooled.shape[0]), int(labels.shape[0]), len(batch_indices))
        if n <= 0:
            continue
        emb_parts.append(pooled[:n].detach().float().cpu())
        label_parts.append(labels[:n].detach().cpu())
        index_parts.append(torch.tensor(batch_indices[:n], dtype=torch.long))

    if not emb_parts:
        _LOG.warning("replay memory empty after scan source=%s task=%s", source_name, task)
        return []

    embeddings = torch.cat(emb_parts, dim=0)
    labels = torch.cat(label_parts, dim=0)
    sample_indices = torch.cat(index_parts, dim=0)
    selected = select_nearest_class_center_exemplars(
        embeddings,
        labels,
        sample_indices,
        samples_per_class=samples_per_class,
    )
    _LOG.info(
        "replay memory source=%s task=%s scanned=%s selected=%s classes=%s per_class=%s",
        source_name,
        task,
        len(scan_indices),
        len(selected),
        len({int(x) for x in labels.tolist() if int(x) >= 0}),
        samples_per_class,
    )
    return selected


def plan_replay_samples_per_class(
    datamodule: Any,
    replay_tasks: Iterable[str],
    *,
    train_cfg: dict[str, Any],
    catalog: TaskCatalog | None = None,
) -> dict[str, int]:
    """为每个 replay 任务计算动态 ``samples_per_class``。"""
    catalog = catalog or resolve_task_catalog(train_cfg)
    tasks = [str(t) for t in replay_tasks]
    n_tasks = max(1, len(tasks))
    train_sets = getattr(datamodule, "_train_sets", {})
    out: dict[str, int] = {}
    for task in tasks:
        spec = catalog.get(task)
        source = spec.source if spec is not None else task
        dataset = train_sets.get(source)
        if dataset is None:
            continue
        n_classes = count_replay_classes(dataset)
        per_class, task_budget, total = resolve_dynamic_samples_per_class(
            train_cfg,
            num_replay_tasks=n_tasks,
            num_classes_in_task=n_classes,
        )
        out[task] = per_class
        _LOG.info(
            "replay budget task=%s source=%s classes=%s per_class=%s task_budget=%s total=%s n_tasks=%s",
            task,
            source,
            n_classes,
            per_class,
            task_budget,
            total,
            n_tasks,
        )
    return out


def _to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in batch.items():
        if torch.is_tensor(value):
            out[key] = value.to(device)
        elif isinstance(value, list) and value and torch.is_tensor(value[0]):
            out[key] = [item.to(device) for item in value]
        else:
            out[key] = value
    return out


def build_replay_memories_for_tasks(
    teacher: nn.Module,
    datamodule: Any,
    replay_tasks: Iterable[str],
    *,
    train_cfg: dict[str, Any],
    catalog: TaskCatalog | None = None,
    device: torch.device | None = None,
) -> dict[str, list[int]]:
    """为多个 replay 任务源构建 exemplar 索引表 ``{source_name: indices}``。"""
    if datamodule is None:
        return {}
    catalog = catalog or resolve_task_catalog(train_cfg)
    device = device or next(teacher.parameters()).device
    patch_size = int(getattr(datamodule, "patch_size", train_cfg.get("patch_size", 8)))
    per_class_plan = plan_replay_samples_per_class(
        datamodule,
        replay_tasks,
        train_cfg=train_cfg,
        catalog=catalog,
    )
    out: dict[str, list[int]] = {}
    for task in replay_tasks:
        spec = catalog.get(str(task))
        source = spec.source if spec is not None else str(task)
        dataset = getattr(datamodule, "_train_sets", {}).get(source)
        lengths_map = getattr(datamodule, "_train_lengths", {})
        if dataset is None or source not in lengths_map:
            continue
        indices = build_class_center_replay_memory(
            teacher,
            dataset,
            list(lengths_map[source]),
            task=str(task),
            source_name=str(source),
            device=device,
            patch_size=patch_size,
            train_cfg=train_cfg,
            catalog=catalog,
            samples_per_class=per_class_plan.get(str(task)),
        )
        if indices:
            out[str(source)] = indices
    return out
