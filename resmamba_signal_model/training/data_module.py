from __future__ import annotations

import math
import os
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader, Dataset, Sampler, Subset

from resmamba_signal_model.data.contracts import CAPTURE_METADATA_KEYS, MISSING_METADATA
from resmamba_signal_model.data.rfdata import (
    RFDataH5Dataset,
    RFDataPoolDataset,
    build_rfdata_pool,
    rfdata_dataloader_worker_init,
    rfdata_loader_worker_kwargs,
    variable_length_collate,
)
from resmamba_signal_model.data.sampling import (
    FixedBatchSampler,
    HomogeneousTokenBudgetSampler,
    LengthBucketBalancedBatchSampler,
    LengthBucketDynamicBatchSampler,
    LengthBucketPKBatchSampler,
    TokenBudgetSampler,
    build_dataset_balanced_sampler,
    build_uniform_sampler,
    dataset_class_ids,
    plan_fixed_token_budget_batches,
    pool_sample_lengths,
    regroup_token_batches_by_length,
    resolve_balanced_sampling_strategy,
    resolve_length_bucket_weight_mode,
    resolve_pk_sampling_params,
)
from resmamba_signal_model.models.task_interface import SOURCE_TO_TASK, TASK_TO_SOURCE
from resmamba_signal_model.training.mix import DynamicRatioScheduler, resolve_mix_strategy
from resmamba_signal_model.training.task_catalog import resolve_task_catalog

try:
    from lightning.pytorch import LightningDataModule
    from lightning.pytorch.utilities import CombinedLoader

    from resmamba_signal_model.training.logging_utils import silence_third_party_warnings

    silence_third_party_warnings()
except ImportError:  # pragma: no cover
    LightningDataModule = object  # type: ignore[misc, assignment]
    CombinedLoader = None  # type: ignore[misc, assignment]


if CombinedLoader is not None:
    class SizedCombinedLoader(CombinedLoader):
        """CombinedLoader 在 ``iter()`` 前不能 ``len()``，进度条会变成 ``n/?``。"""

        def __init__(self, iterables, mode: str = "max_size_cycle", *, length: int) -> None:
            super().__init__(iterables, mode)
            self._known_length = max(1, int(length))

        def __len__(self) -> int:
            if self._iterator is None:
                return self._known_length
            try:
                value = super().__len__()
            except (RuntimeError, TypeError, NotImplementedError):
                return self._known_length
            if isinstance(value, float) and value == float("inf"):
                return self._known_length
            return int(value)
else:  # pragma: no cover
    SizedCombinedLoader = None  # type: ignore[misc, assignment]


LABEL_TENSOR_KEYS = (
    "length",
    "dataset_id",
    "task_type_id",
    "mod_label_id",
    "canonical_mod_label_id",
    "emitter_id",
    "global_emitter_id",
    "source_label_id",
    "global_label_id",
)

CONTRACT_LIST_KEYS = (
    "modality_id",
    "coordinate_unit",
    *CAPTURE_METADATA_KEYS,
)

PRETRAIN_BLOCKED_KEYS = frozenset(
    {
        "mod_label_id",
        "canonical_mod_label_id",
        "emitter_id",
        "source_label_id",
        "global_emitter_id",
        "global_label_id",
        "dataset_id",
        "task_type_id",
        *CAPTURE_METADATA_KEYS,
        "h5_path",
    }
)


def is_pretrain_blocked_key(key: str) -> bool:
    name = str(key)
    return name in PRETRAIN_BLOCKED_KEYS or name.startswith("global_")


def pretrain_collate_firewall(batch: dict[str, Any]) -> dict[str, Any]:
    """预训练模型 batch 不得含标签 / dataset_id / 采集元数据 / 文件身份。"""
    return {key: value for key, value in batch.items() if not is_pretrain_blocked_key(key)}

# 训练 sampler 不得使用 global_label_id（仅 val 聚类指标）。
TRAIN_SAMPLER_LABEL_FIELDS: dict[str, tuple[str, ...]] = {
    "ld_intrapulse": ("canonical_mod_label_id", "mod_label_id"),
    "ld_model": ("global_emitter_id", "emitter_id"),
    "tx_modulation": ("canonical_mod_label_id", "mod_label_id"),
    "ld_clustering": ("dataset_id",),
    "tx_clustering": ("dataset_id",),
    "prediction": ("dataset_id",),
    "classification": ("canonical_mod_label_id", "mod_label_id"),
    "emitter": ("global_emitter_id", "emitter_id"),
    "clustering": ("dataset_id",),
    "imputation": ("dataset_id",),
}


def train_sampler_label_fields(task: str | None) -> tuple[str, ...]:
    if not task:
        return ("canonical_mod_label_id", "mod_label_id", "global_emitter_id", "emitter_id", "dataset_id")
    return TRAIN_SAMPLER_LABEL_FIELDS.get(str(task), ("dataset_id",))


def physical_iq_view2(iq: torch.Tensor) -> torch.Tensor:
    """聚类第二视图：相位旋转 + 循环时移。``iq`` 为 ``[C,L]`` 或 ``[B,C,L]``。"""
    if iq.ndim == 3:
        return torch.stack([physical_iq_view2(item) for item in iq], dim=0)
    if iq.ndim != 2:
        return iq
    view = iq.clone()
    n_ch, length = int(view.shape[0]), int(view.shape[-1])
    if n_ch >= 2:
        theta = view.new_empty(()).uniform_(-math.pi, math.pi)
        cosine, sine = torch.cos(theta), torch.sin(theta)
        i_ch, q_ch = view[0], view[1]
        view[0] = i_ch * cosine - q_ch * sine
        view[1] = i_ch * sine + q_ch * cosine
    max_shift = max(1, length // 16)
    shift = int(torch.randint(-max_shift, max_shift + 1, (1,), device=view.device).item())
    if shift:
        view = torch.roll(view, shifts=shift, dims=-1)
    return view


def attach_clustering_view2(batch: dict[str, Any]) -> dict[str, Any]:
    """给 clustering batch 补物理合法 ``view2``；已有则保持。"""
    if batch.get("view2") is not None:
        return batch
    iq = batch.get("iq", batch.get("values"))
    if iq is None:
        return batch
    if isinstance(iq, list):
        batch["view2"] = [physical_iq_view2(item) for item in iq]
    else:
        batch["view2"] = physical_iq_view2(iq)
    return batch


class SyntheticIQDataset(Dataset):
    """无 H5 时的合成变长 I/Q，供 tiny / CI。"""

    def __init__(
        self,
        n: int = 32,
        lengths: tuple[int, ...] = (128, 256, 512),
        n_datasets: int = 4,
        seed: int = 0,
        source_id: int = 0,
        task: str | None = None,
    ) -> None:
        self.n = int(n)
        self.lengths = tuple(int(x) for x in lengths)
        self.n_datasets = int(n_datasets)
        self.source_id = int(source_id)
        self.task = str(task) if task else None
        g = torch.Generator().manual_seed(seed + source_id)
        self._len_idx = torch.randint(0, len(self.lengths), (self.n,), generator=g)
        self._ds = torch.randint(0, self.n_datasets, (self.n,), generator=g)
        self._mod = torch.randint(0, 8, (self.n,), generator=g)
        self._model = torch.randint(0, 12, (self.n,), generator=g)

    def class_labels(self) -> list[int]:
        return [int(self._ds[i]) * 100000 + int(self._mod[i]) for i in range(self.n)]

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, idx: int) -> dict[str, Any]:
        length = self.lengths[int(self._len_idx[idx])]
        mod_id = int(self._mod[idx])
        model_id = int(self._model[idx])
        ds_id = int(self._ds[idx])
        iq = torch.randn(2, length)
        item: dict[str, Any] = {
            "iq": iq,
            "values": iq,
            "length": length,
            "dataset_id": ds_id,
            "task_type_id": 0,
            "mod_label_id": mod_id,
            "canonical_mod_label_id": mod_id,
            "model_label_id": model_id,
            "emitter_id": mod_id,
            "global_emitter_id": mod_id,
            "source_label_id": -1,
            "global_label_id": ds_id * 100000 + mod_id,
            "modality_id": "rf",
            "receiver_id": MISSING_METADATA,
            "session_id": MISSING_METADATA,
            "channel_id": MISSING_METADATA,
            "capture_id": MISSING_METADATA,
        }
        return item


def merge_source_batches(batches: dict[str, Any] | list[Any]) -> dict[str, Any]:
    """把 CombinedLoader 各源 batch 拼成一次 packed forward（list[Tensor]）。"""
    if isinstance(batches, dict):
        items = [(name, batch) for name, batch in batches.items() if batch is not None]
    else:
        items = [(str(i), batch) for i, batch in enumerate(batches) if batch is not None]
    if not items:
        raise ValueError("combine_then_pack 收到空 batch")
    iq: list[torch.Tensor] = []
    view2: list[torch.Tensor] = []
    has_view2 = False
    source_name: list[str] = []
    stacked: dict[str, list[torch.Tensor]] = {k: [] for k in LABEL_TENSOR_KEYS}
    lists: dict[str, list[Any]] = {k: [] for k in CONTRACT_LIST_KEYS}
    per_source_tokens: dict[str, int] = {}
    for name, batch in items:
        batch_iq = batch.get("iq", batch.get("values"))
        if batch_iq is None:
            raise KeyError(f"源 {name!r} 缺少 iq/values")
        if isinstance(batch_iq, list):
            iq.extend(batch_iq)
            n = len(batch_iq)
        else:
            iq.extend([batch_iq[i] for i in range(batch_iq.shape[0])])
            n = int(batch_iq.shape[0])
        batch_view2 = batch.get("view2")
        if batch_view2 is not None:
            has_view2 = True
            if isinstance(batch_view2, list):
                view2.extend(batch_view2)
            else:
                view2.extend([batch_view2[i] for i in range(batch_view2.shape[0])])
        else:
            view2.extend([iq[-n + i] for i in range(n)])
        source_name.extend([name] * n)
        n_tok = 0
        for tensor in iq[-n:]:
            n_tok += max(1, int(tensor.shape[-1]))
        per_source_tokens[name] = n_tok
        for key in LABEL_TENSOR_KEYS:
            value = batch.get(key)
            if value is None:
                stacked[key].append(
                    torch.full((n,), -1 if key.endswith("_id") or "label" in key else 0, dtype=torch.long)
                )
            else:
                stacked[key].append(value.long() if torch.is_tensor(value) else torch.as_tensor(value, dtype=torch.long))
        for key in CONTRACT_LIST_KEYS:
            value = batch.get(key)
            if value is None:
                lists[key].extend([MISSING_METADATA if key in CAPTURE_METADATA_KEYS else "rf"] * n)
            elif isinstance(value, (list, tuple)):
                lists[key].extend(list(value))
            else:
                lists[key].extend([value] * n)
    out: dict[str, Any] = {
        "iq": iq,
        "values": iq,
        "source_name": source_name,
        "per_source_tokens": per_source_tokens,
    }
    if has_view2:
        out["view2"] = view2
    for key, parts in stacked.items():
        out[key] = torch.cat(parts, dim=0)
    for key, values in lists.items():
        out[key] = values
    return out


def _dataset_family(h5_name: str) -> str:
    stem = Path(h5_name).name.replace("_train.h5", "").replace("_val.h5", "").replace("_test.h5", "")
    for prefix in ("radcom", "rml2016", "rml2018", "wifi", "adsb", "radar", "cjr"):
        if stem.startswith(prefix):
            return prefix
    return stem.split("_")[0] if "_" in stem else stem


def split_pool_by_groups(
    pool: RFDataPoolDataset,
    groups: dict[str, list[str]] | None,
) -> dict[str, RFDataPoolDataset]:
    if not pool.datasets:
        return {}
    if not groups:
        buckets: dict[str, list] = {}
        for sub in pool.datasets:
            family = _dataset_family(sub.h5_path.name)
            buckets.setdefault(family, []).append(sub)
        return {name: RFDataPoolDataset(subs, pool_name=f"{pool.pool_name}:{name}") for name, subs in buckets.items()}
    assigned: dict[str, list] = {name: [] for name in groups}
    assigned["other"] = []
    for sub in pool.datasets:
        stem = Path(sub.h5_path).name.replace("_train.h5", "").replace("_val.h5", "").replace("_test.h5", "")
        hit = None
        for name, keys in groups.items():
            if stem in keys or any(stem.startswith(k) for k in keys):
                hit = name
                break
        assigned[hit or "other"].append(sub)
    return {
        name: RFDataPoolDataset(subs, pool_name=f"{pool.pool_name}:{name}")
        for name, subs in assigned.items()
        if subs
    }


def resolve_dataset_h5(rfdata_root: str | Path, dataset: str, split: str = "val") -> Path:
    """按 split 解析 H5：默认 val（评估）；test 现为阶段二/三训练集，缺失则回退。"""
    root = Path(rfdata_root)
    split_key = str(split).strip().lower()
    order = {
        "test": (f"{dataset}_test.h5", f"{dataset}_val.h5", f"{dataset}_train.h5"),
        "val": (f"{dataset}_val.h5", f"{dataset}_test.h5", f"{dataset}_train.h5"),
        "train": (f"{dataset}_train.h5", f"{dataset}_val.h5", f"{dataset}_test.h5"),
    }.get(split_key)
    if order is None:
        raise ValueError(f"未知 split={split!r}，可选: train | val | test")
    for name in order:
        for candidate in (root / "h5" / name, root / name):
            if candidate.is_file():
                return candidate
    raise FileNotFoundError(f"找不到数据集 H5: {dataset} split={split_key}（在 {root}）")


def _collate_with_source(
    source_name: str | None,
    task: str | None = None,
    *,
    attach_view2: bool = False,
    stage: str | None = None,
):
    def _collate(samples: list[Any]) -> dict[str, Any]:
        batch = variable_length_collate(samples)
        if source_name:
            batch["source_name"] = [source_name] * len(samples)
        if task:
            batch["task"] = task
        if attach_view2:
            attach_clustering_view2(batch)
        if stage == "pretrain":
            batch = pretrain_collate_firewall(batch)
        return batch

    return _collate


def _composed_worker_init(worker_id: int) -> None:
    rfdata_dataloader_worker_init(worker_id)
    if os.environ.get("PL_SEED_WORKERS") == "1":
        try:
            from lightning.fabric.utilities.seed import pl_worker_init_function

            pl_worker_init_function(worker_id)
        except Exception:
            pass


def _plan_fixed_eval_batches(
    dataset: Dataset,
    *,
    token_budget: int,
    patch_size: int,
    num_batches: int,
    seed: int,
) -> tuple[list[int], list[list[int]]]:
    if isinstance(dataset, RFDataPoolDataset):
        lengths = pool_sample_lengths(dataset)
    else:
        lengths = [int(dataset[i]["length"]) for i in range(len(dataset))]
    raw = plan_fixed_token_budget_batches(
        lengths,
        token_budget=max(1, int(token_budget)),
        patch_size=int(patch_size),
        num_batches=max(1, int(num_batches)),
        seed=int(seed),
        class_ids=dataset_class_ids(dataset),
    )
    plan = regroup_token_batches_by_length(
        raw,
        lengths,
        token_budget=max(1, int(token_budget)),
        patch_size=int(patch_size),
    )
    return lengths, plan


def _make_loader(
    dataset: Dataset,
    *,
    lengths: list[int],
    token_budget: int,
    patch_size: int,
    num_batches: int,
    num_workers: int,
    seed: int | None,
    pin_memory: bool,
    batch_plan: list[list[int]] | None = None,
    prefetch_factor: int = 4,
    persistent_workers: bool = True,
    source_name: str | None = None,
    task: str | None = None,
    batch_sampler: Sampler[list[int]] | None = None,
    attach_view2: bool = False,
    stage: str | None = None,
) -> DataLoader:
    if batch_sampler is not None:
        sampler: Sampler[list[int]] = batch_sampler
    elif batch_plan is not None:
        sampler = FixedBatchSampler(batch_plan)
    else:
        sampler = TokenBudgetSampler(
            lengths,
            token_budget=max(1, int(token_budget)),
            patch_size=patch_size,
            num_batches=num_batches,
            seed=seed,
        )
    kwargs = rfdata_loader_worker_kwargs(
        num_workers,
        prefetch_factor=prefetch_factor,
        persistent_workers=persistent_workers,
    )
    if num_workers > 0:
        kwargs["worker_init_fn"] = _composed_worker_init
    return DataLoader(
        dataset,
        batch_sampler=sampler,
        collate_fn=_collate_with_source(source_name, task, attach_view2=attach_view2, stage=stage),
        num_workers=num_workers,
        pin_memory=pin_memory,
        **kwargs,
    )


def make_fixed_eval_loader(
    dataset: Dataset,
    *,
    token_budget: int,
    patch_size: int,
    num_batches: int,
    seed: int,
    num_workers: int = 0,
    pin_memory: bool = False,
    source_name: str | None = None,
    task: str | None = None,
    prefetch_factor: int = 4,
    attach_view2: bool | None = None,
) -> DataLoader:
    """与验证相同的固定 token-budget / seed 协议，供 val 与 infer 复用。"""
    _lengths, plan = _plan_fixed_eval_batches(
        dataset,
        token_budget=token_budget,
        patch_size=patch_size,
        num_batches=num_batches,
        seed=seed,
    )
    return _make_loader(
        dataset,
        lengths=[],
        token_budget=token_budget,
        patch_size=patch_size,
        num_batches=max(1, int(num_batches)),
        num_workers=num_workers,
        seed=int(seed),
        pin_memory=pin_memory,
        batch_plan=plan,
        prefetch_factor=prefetch_factor,
        persistent_workers=bool(num_workers > 0),
        source_name=source_name,
        task=task,
        attach_view2=bool(attach_view2) if attach_view2 is not None else False,
    )


def _pool_has_label(pool: RFDataPoolDataset, field: str) -> bool:
    for sub in pool.datasets:
        labels = getattr(sub, "_labels", None) or {}
        if field in labels:
            return True
    return False


class _EpochBatchSampler(Sampler[list[int]]):
    """把逐样本 sampler 切成固定步数的 batch。"""

    def __init__(self, index_sampler: Sampler[int], *, batch_size: int, num_batches: int) -> None:
        self.index_sampler = index_sampler
        self.batch_size = max(1, int(batch_size))
        self.num_batches = max(1, int(num_batches))

    def __len__(self) -> int:
        return self.num_batches

    def __iter__(self):
        iterator = iter(self.index_sampler)
        for _ in range(self.num_batches):
            batch: list[int] = []
            for _ in range(self.batch_size):
                try:
                    batch.append(int(next(iterator)))
                except StopIteration:
                    iterator = iter(self.index_sampler)
                    batch.append(int(next(iterator)))
            yield batch


def build_train_batch_sampler(
    dataset: Dataset,
    *,
    lengths: list[int],
    token_budget: int,
    patch_size: int,
    num_batches: int,
    seed: int,
    train_cfg: dict[str, Any],
    task: str | None = None,
    stage: str | None = None,
) -> Sampler[list[int]]:
    homo_cfg = train_cfg.get("homogeneous_batch")
    use_homo = stage == "pretrain" if homo_cfg is None else bool(homo_cfg)
    if use_homo and isinstance(dataset, RFDataPoolDataset):
        return HomogeneousTokenBudgetSampler(
            dataset,
            token_budget=max(1, int(token_budget)),
            patch_size=int(patch_size),
            num_batches=int(num_batches),
            seed=int(seed),
            lengths=lengths,
            source_groups=train_cfg.get("source_groups"),
            family_quotas=train_cfg.get("family_quotas"),
        )
    enabled = bool(train_cfg.get("balanced_sampling", False))
    strategy = resolve_balanced_sampling_strategy(
        enabled=enabled,
        strategy=train_cfg.get("balanced_sampling_strategy"),
    )
    if strategy == "none" or not isinstance(dataset, RFDataPoolDataset):
        return TokenBudgetSampler(
            lengths,
            token_budget=max(1, int(token_budget)),
            patch_size=int(patch_size),
            num_batches=int(num_batches),
            seed=int(seed),
        )
    batch_size = int(train_cfg.get("batch_size", 32))
    bucket_mode = resolve_length_bucket_weight_mode(strategy, train_cfg)
    infer_emitter = bool(
        task == "emitter" or (strategy in ("length_bucket_class", "length_bucket_pk") and task not in ("clustering", None))
    )
    fields = train_sampler_label_fields(task)
    if strategy == "uniform":
        return _EpochBatchSampler(
            build_uniform_sampler(dataset),
            batch_size=batch_size,
            num_batches=num_batches,
        )
    if strategy == "dataset":
        return _EpochBatchSampler(
            build_dataset_balanced_sampler(dataset),
            batch_size=batch_size,
            num_batches=num_batches,
        )
    if strategy == "length_bucket_dynamic":
        return LengthBucketDynamicBatchSampler(
            dataset,
            schedule=train_cfg.get("dynamic_batch_size_schedule") or [],
            num_batches=num_batches,
            seed=int(seed),
            bucket_weight_mode=bucket_mode,
            infer_emitter_classes=infer_emitter,
        )
    if strategy == "length_bucket_pk":
        p, k = resolve_pk_sampling_params(train_cfg, batch_size)
        label_field = next((name for name in fields if _pool_has_label(dataset, name)), fields[0])
        if label_field == "global_label_id":
            label_field = "dataset_id"
        return LengthBucketPKBatchSampler(
            dataset,
            batch_size=batch_size,
            pk_num_classes=p,
            pk_samples_per_class=k,
            num_batches=num_batches,
            seed=int(seed),
            bucket_weight_mode=bucket_mode,
            label_field=label_field,
        )
    return LengthBucketBalancedBatchSampler(
        dataset,
        batch_size=batch_size,
        num_batches=num_batches,
        seed=int(seed),
        bucket_weight_mode=bucket_mode,
        infer_emitter_classes=infer_emitter,
    )


class SignalDataModule(LightningDataModule):
    def __init__(self, train_cfg: dict[str, Any], *, stage: str = "pretrain") -> None:
        super().__init__()
        self._log_hyperparams = False
        self.train_cfg = train_cfg
        self.stage = stage
        self.patch_size = int(train_cfg.get("patch_size") or train_cfg.get("model", {}).get("patch_size", 16))
        self.token_budget = int(train_cfg.get("token_budget", 4096))
        self.steps_per_epoch = int(train_cfg.get("steps_per_epoch", 100))
        self.val_batches = int(train_cfg.get("val_batches", 50))
        self.val_seed = int(train_cfg.get("val_seed", train_cfg.get("seed", 0)))
        self.num_workers = int(train_cfg.get("num_workers", 0))
        self.val_num_workers = int(train_cfg.get("val_num_workers", self.num_workers))
        self.prefetch_factor = int(train_cfg.get("prefetch_factor", 4))
        self.val_prefetch_factor = int(train_cfg.get("val_prefetch_factor", self.prefetch_factor))
        self.pin_memory = bool(train_cfg.get("pin_memory", torch.cuda.is_available()))
        self.cache_iq_in_memory = train_cfg.get("cache_iq_in_memory", False)
        self.synthetic = bool(train_cfg.get("synthetic", False))
        self.source_groups = train_cfg.get("source_groups")
        self.mix = None
        self.source_names: list[str] = []
        self._train_sets: dict[str, Dataset] = {}
        self._val_sets: dict[str, Dataset] = {}
        self._train_lengths: dict[str, list[int]] = {}
        self._val_batch_plan: dict[str, list[list[int]]] = {}
        self._cached_val_loader = None
        self._active_train_sources: list[str] | None = None
        self._active_replay_sources: list[str] | None = None
        self._replay_memory: dict[str, list[int]] = {}

    @property
    def val_source_names(self) -> list[str]:
        return list(self._filtered_val_sets()) or list(self.source_names)

    def set_active_train_filter(
        self,
        sources: list[str] | None,
        *,
        replay_sources: list[str] | None = None,
    ) -> None:
        """限制本轮 train/val dataloader 只用这些源；``None`` 表示全部。

        单任务 / ``task_schedule`` 阶段：验证只跑当前任务；训练可额外混入 ``replay_sources``。
        """
        prev = (tuple(self._active_train_sources or ()), tuple(self._active_replay_sources or ()))
        if sources is None:
            self._active_train_sources = None
        else:
            pool = set(self._train_sets) | set(self._val_sets)
            allowed = [str(name) for name in sources if str(name) in pool]
            if not allowed and pool:
                raise ValueError(f"active_train_sources={sources!r} 与已加载源 {sorted(pool)} 无交集")
            self._active_train_sources = allowed
        if replay_sources is None:
            self._active_replay_sources = None
        else:
            pool = set(self._train_sets)
            replay = [str(name) for name in replay_sources if str(name) in pool]
            self._active_replay_sources = replay or None
        if (tuple(self._active_train_sources or ()), tuple(self._active_replay_sources or ())) != prev:
            self._cached_val_loader = None

    def set_replay_memory(self, source: str, indices: list[int] | None) -> None:
        key = str(source)
        if indices:
            self._replay_memory[key] = [int(i) for i in indices]
        elif key in self._replay_memory:
            del self._replay_memory[key]

    def update_replay_memory(self, memory: dict[str, list[int]] | None) -> None:
        """批量更新 exemplar 索引；仅影响 replay 源训练采样。"""
        for source, indices in dict(memory or {}).items():
            self.set_replay_memory(source, list(indices))

    def _train_dataset_and_lengths(
        self,
        name: str,
        dataset: Dataset,
        *,
        train: bool,
    ) -> tuple[Dataset, list[int]]:
        lengths = list(self._train_lengths[name])
        if not train:
            return dataset, lengths
        from resmamba_signal_model.training.replay_memory import resolve_replay_strategy

        if resolve_replay_strategy(self.train_cfg) != "class_center":
            return dataset, lengths
        replay_set = set(self._active_replay_sources or [])
        if str(name) not in replay_set:
            return dataset, lengths
        mem = self._replay_memory.get(str(name))
        if not mem:
            return dataset, lengths
        base = self._train_sets.get(str(name), dataset)
        subset = Subset(base, mem)
        return subset, [lengths[int(i)] for i in mem]

    def _filtered_train_sets(self) -> dict[str, Dataset]:
        if not self._active_train_sources:
            return dict(self._train_sets)
        out: dict[str, Dataset] = {
            name: self._train_sets[name] for name in self._active_train_sources if name in self._train_sets
        }
        if self._active_replay_sources:
            for name in self._active_replay_sources:
                if name in self._train_sets and name not in out:
                    out[name] = self._train_sets[name]
        return out

    def _filtered_val_sets(self) -> dict[str, Dataset]:
        if not self._active_train_sources:
            return dict(self._val_sets)
        return {name: self._val_sets[name] for name in self._active_train_sources if name in self._val_sets}

    def train_sampler_seed(self, source_index: int = 0) -> int:
        base = int(self.train_cfg.get("seed", 0))
        epoch = 0
        rank = 0
        trainer = getattr(self, "trainer", None)
        if trainer is not None:
            epoch = int(getattr(trainer, "current_epoch", 0) or 0)
            rank = int(getattr(trainer, "global_rank", 0) or 0)
        return int(base) + 1_000_003 * epoch + 97 * rank + int(source_index)

    @staticmethod
    def _split_token_budget(names: list[str], budget: int) -> dict[str, int]:
        if not names:
            return {}
        n = len(names)
        base = max(1, int(budget) // n)
        rem = max(0, int(budget) - base * n)
        return {name: base + (1 if i < rem else 0) for i, name in enumerate(names)}

    def _token_shares(self, names: list[str], *, train: bool) -> dict[str, int]:
        if not train:
            return {name: self.token_budget for name in names}
        replay_ratio = float(self.train_cfg.get("replay_mix_ratio", 0.0) or 0.0)
        replay_set = set(self._active_replay_sources or [])
        if replay_ratio > 0 and replay_set:
            current = [name for name in names if name not in replay_set]
            replay = [name for name in names if name in replay_set]
            if current and replay:
                replay_budget = max(1, int(round(self.token_budget * replay_ratio)))
                current_budget = max(1, self.token_budget - replay_budget)
                return {
                    **self._split_token_budget(current, current_budget),
                    **self._split_token_budget(replay, replay_budget),
                }
        strategy = resolve_mix_strategy(self.train_cfg.get("mix_strategy"))
        if strategy == "equal" or self.mix is None:
            n = max(1, len(names))
            base = max(1, self.token_budget // n)
            rem = max(0, self.token_budget - base * n)
            shares = {}
            for i, name in enumerate(names):
                shares[name] = base + (1 if i < rem else 0)
            return shares
        return self.mix.token_shares(self.token_budget)

    def _build_val_plans(self) -> None:
        """setup 时从 val 集按类别分层均匀抽出固定 token-budget batch，之后每个 epoch 复用。"""
        self._cached_val_loader = None
        self._val_batch_plan = {}
        for name, dataset in self._val_sets.items():
            _lengths, plan = _plan_fixed_eval_batches(
                dataset,
                token_budget=self.token_budget,
                patch_size=self.patch_size,
                num_batches=max(1, self.val_batches),
                seed=self.val_seed,
            )
            self._val_batch_plan[name] = plan

    def setup(self, stage: str | None = None) -> None:
        if self.source_names and self.mix is not None:
            return
        if self.synthetic:
            catalog = resolve_task_catalog(self.train_cfg)
            if self.stage in ("stage2", "joint"):
                default_names = [spec.source for spec in catalog.specs]
            elif self.stage == "stage3":
                task = str(self.train_cfg.get("task") or "ld_intrapulse")
                spec = catalog.get(task)
                default_names = [spec.source if spec is not None else TASK_TO_SOURCE.get(task, task)]
            else:
                default_names = ["src_a", "src_b"]
            names = list(self.train_cfg.get("synthetic_sources") or default_names)
            self.source_names = names
            synth_seed = int(self.train_cfg.get("seed", 0))
            source_to_task = {spec.source: spec.name for spec in catalog.specs}
            for i, name in enumerate(names):
                task_name = source_to_task.get(name)
                self._train_sets[name] = SyntheticIQDataset(
                    n=64, lengths=(128, 256, 512), source_id=i, seed=synth_seed, task=task_name
                )
                self._val_sets[name] = SyntheticIQDataset(
                    n=16, lengths=(128, 256), source_id=100 + i, seed=synth_seed, task=task_name
                )
                self._train_lengths[name] = [
                    int(self._train_sets[name][j]["length"]) for j in range(len(self._train_sets[name]))
                ]
            self.mix = DynamicRatioScheduler(
                self.source_names,
                min_ratio=float(self.train_cfg.get("min_ratio", 0.05)),
                value_clip=float(self.train_cfg.get("mix_value_clip", 2.0)),
            )
            self._build_val_plans()
            return

        root = self.train_cfg.get("rfdata_root") or "dataset"
        pool_cache: dict[tuple[str, bool | None], Dataset] = {}

        def get_pool(pool_name: str, *, use_labels: bool | None = None) -> Dataset:
            cached = pool_cache.get((pool_name, use_labels))
            if cached is not None:
                return cached
            label_flag = use_labels if use_labels is not None else (self.stage != "pretrain")
            pool = build_rfdata_pool(
                root,
                pool_name,
                use_labels=label_flag,
                iq_normalize=self.train_cfg.get("iq_normalize", "none"),
                cache_iq_in_memory=self.cache_iq_in_memory,
            )
            pool_cache[(pool_name, use_labels)] = pool
            return pool

        if self.stage == "pretrain":
            pool = get_pool(self.train_cfg.get("pool", "pretrain_train"), use_labels=False)
            val_pool = get_pool(self.train_cfg.get("val_pool", "pretrain_val"), use_labels=False)
            self._train_sets = {"pretrain": pool}
            self._val_sets = {"pretrain": val_pool}
        else:
            task_pools = self.train_cfg.get("task_pools") or {
                "ld_intrapulse": ("downstream_radar_modulation_train", "downstream_radar_modulation_val"),
                "ld_model": ("downstream_radar_model_train", "downstream_radar_model_val"),
                "tx_modulation": ("downstream_comm_modulation_train", "downstream_comm_modulation_val"),
                "ld_clustering": ("clustering_radar_train", "clustering_radar_val"),
                "tx_clustering": ("clustering_comm_train", "clustering_comm_val"),
                "prediction": ("prediction_train", "prediction_val"),
            }
            catalog = resolve_task_catalog(self.train_cfg)
            if self.stage == "stage3":
                task = str(self.train_cfg.get("task") or "")
                spec = catalog.get(task)
                keep = spec.source if spec is not None else TASK_TO_SOURCE.get(task, task)
                if keep and keep in task_pools:
                    task_pools = {keep: task_pools[keep]}
            elif self.stage in ("stage2", "joint") and not self.train_cfg.get("task_schedule"):
                # 无 task_schedule 时，``tasks:`` / ``--tasks`` 只加载对应源，避免名存实亡的混训。
                keep_sources = {spec.source for spec in catalog.specs}
                if keep_sources:
                    task_pools = {name: pair for name, pair in task_pools.items() if name in keep_sources}
            for name, pair in task_pools.items():
                train_name, val_name = pair[0], pair[1]
                try:
                    self._train_sets[name] = get_pool(train_name)
                    self._val_sets[name] = get_pool(val_name)
                except (KeyError, FileNotFoundError):
                    continue
        self.source_names = list(self._train_sets)
        for name, dataset in self._train_sets.items():
            if isinstance(dataset, RFDataPoolDataset):
                self._train_lengths[name] = pool_sample_lengths(dataset)
            else:
                self._train_lengths[name] = [int(dataset[i]["length"]) for i in range(len(dataset))]
        if not self.source_names:
            raise RuntimeError("没有可用数据源；可设 synthetic: true 做冒烟")
        self.mix = DynamicRatioScheduler(
                self.source_names,
                min_ratio=float(self.train_cfg.get("min_ratio", 0.05)),
                value_clip=float(self.train_cfg.get("mix_value_clip", 2.0)),
            )
        self._build_val_plans()

    def _task_for_loader_name(self, name: str) -> str | None:
        if self.stage == "pretrain":
            return None
        catalog = resolve_task_catalog(self.train_cfg)
        if name in catalog.source_to_task:
            return catalog.source_to_task[name]
        if name in catalog.by_name:
            return name
        return SOURCE_TO_TASK.get(name)

    def _loaders(self, sets: dict[str, Dataset], *, train: bool) -> dict[str, DataLoader]:
        assert self.mix is not None
        shares = self._token_shares(list(sets), train=train)
        loaders: dict[str, DataLoader] = {}
        for source_index, (name, dataset) in enumerate(sets.items()):
            if train:
                dataset, lengths = self._train_dataset_and_lengths(name, dataset, train=True)
                num_batches = self.steps_per_epoch
                workers = self.num_workers
                plan = None
                prefetch = self.prefetch_factor
                persistent = True
                seed = self.train_sampler_seed(source_index)
                sampler = build_train_batch_sampler(
                    dataset,
                    lengths=lengths,
                    token_budget=shares.get(name, 1),
                    patch_size=self.patch_size,
                    num_batches=num_batches,
                    seed=seed,
                    train_cfg=self.train_cfg,
                    task=self._task_for_loader_name(name),
                    stage=self.stage,
                )
            else:
                lengths = []
                num_batches = max(1, self.val_batches)
                workers = self.val_num_workers
                plan = self._val_batch_plan[name]
                prefetch = self.val_prefetch_factor
                persistent = True
                seed = self.val_seed
                sampler = None
            task_name = self._task_for_loader_name(name)
            task_kind = None
            if task_name and task_name in resolve_task_catalog(self.train_cfg).by_name:
                task_kind = resolve_task_catalog(self.train_cfg).kind(task_name)
            attach_view2 = bool(
                train
                and task_kind == "clustering"
                and bool(self.train_cfg.get("clustering_view2", True))
            )
            pin_memory = self.pin_memory and not (workers > 0 and persistent)
            loaders[name] = _make_loader(
                dataset,
                lengths=lengths,
                token_budget=shares.get(name, 1),
                patch_size=self.patch_size,
                num_batches=num_batches,
                num_workers=workers,
                seed=seed,
                pin_memory=pin_memory,
                batch_plan=plan,
                prefetch_factor=prefetch,
                persistent_workers=persistent,
                source_name=name,
                task=task_name,
                batch_sampler=sampler if train else None,
                attach_view2=attach_view2,
                stage=self.stage,
            )
        return loaders

    def train_dataloader(self):
        if CombinedLoader is None:
            raise ImportError("需要 lightning>=2.4 以使用 CombinedLoader")
        train_sets = self._filtered_train_sets()
        if not train_sets:
            raise RuntimeError("没有可训练数据源（检查 task_schedule / active_train_sources）")
        # 单任务阶段重建 mix，避免仍按五源份额拆 token_budget。
        if self._active_train_sources is not None:
            self.mix = DynamicRatioScheduler(
                list(train_sets),
                min_ratio=float(self.train_cfg.get("min_ratio", 0.05)),
                value_clip=float(self.train_cfg.get("mix_value_clip", 2.0)),
            )
            self.source_names = list(train_sets)
        loaders = self._loaders(train_sets, train=True)
        if SizedCombinedLoader is None:
            raise ImportError("需要 lightning>=2.4 以使用 CombinedLoader")
        return SizedCombinedLoader(loaders, mode="max_size_cycle", length=self.steps_per_epoch)

    def val_dataloader(self):
        if CombinedLoader is None or SizedCombinedLoader is None:
            raise ImportError("需要 lightning>=2.4 以使用 CombinedLoader")
        val_sets = self._filtered_val_sets()
        if not val_sets:
            raise RuntimeError("没有可验证数据源（检查 task_schedule / active_train_sources）")
        # 单任务阶段必须重建，否则会缓存「全任务」val loader。
        cache_key = tuple(val_sets.keys())
        if self._cached_val_loader is None or getattr(self, "_cached_val_key", None) != cache_key:
            loaders = self._loaders(val_sets, train=False)
            val_steps = max(1, self.val_batches) * max(1, len(loaders))
            self._cached_val_loader = SizedCombinedLoader(loaders, mode="sequential", length=val_steps)
            self._cached_val_key = cache_key
        return self._cached_val_loader


def build_infer_pool(rfdata_root: str | Path, datasets: list[str], *, split: str = "val") -> RFDataPoolDataset:
    root = Path(rfdata_root)
    parts = [
        RFDataH5Dataset(resolve_dataset_h5(root, name, split), iq_normalize="none")
        for name in datasets
    ]
    return RFDataPoolDataset(parts, pool_name=f"infer_{split}")
