from __future__ import annotations

import random
from collections import defaultdict
from dataclasses import dataclass
from typing import Iterator, Literal

import h5py
import numpy as np
import torch
from torch.utils.data import Sampler, WeightedRandomSampler

from resmamba_signal_model.data.rfdata import RFDataPoolDataset

BalancedSamplingStrategy = Literal[
    "none",
    "uniform",
    "dataset",
    "length_bucket",
    "length_bucket_proportional",
    "length_bucket_class",
    "length_bucket_pk",
    "length_bucket_dynamic",
]
LengthBucketWeightMode = Literal["equal", "proportional", "class_count"]


@dataclass(frozen=True)
class DynamicBatchSizeRule:
    """长度上限 → batch_size；max_length=None 表示兜底（任意更长序列）。"""

    max_length: int | None
    batch_size: int


def parse_dynamic_batch_size_schedule(schedule: list[dict] | None) -> list[DynamicBatchSizeRule]:
    if not schedule:
        raise ValueError("length_bucket_dynamic 需要非空 dynamic_batch_size_schedule")
    rules: list[DynamicBatchSizeRule] = []
    for i, item in enumerate(schedule):
        if not isinstance(item, dict):
            raise ValueError(f"dynamic_batch_size_schedule[{i}] 必须为 dict，当前为 {type(item).__name__}")
        if "batch_size" not in item:
            raise ValueError(f"dynamic_batch_size_schedule[{i}] 缺少 batch_size")
        batch_size = int(item["batch_size"])
        if batch_size < 1:
            raise ValueError(f"dynamic_batch_size_schedule[{i}].batch_size 必须 >= 1，当前为 {batch_size}")
        raw_max = item.get("max_length", None)
        max_length = None if raw_max is None else int(raw_max)
        if max_length is not None and max_length < 1:
            raise ValueError(f"dynamic_batch_size_schedule[{i}].max_length 必须 >= 1 或 null")
        rules.append(DynamicBatchSizeRule(max_length=max_length, batch_size=batch_size))
    finite = [r for r in rules if r.max_length is not None]
    catch_all = [r for r in rules if r.max_length is None]
    if len(catch_all) > 1:
        raise ValueError("dynamic_batch_size_schedule 至多一条 max_length: null 兜底规则")
    finite_sorted = sorted(finite, key=lambda r: int(r.max_length))  # type: ignore[arg-type]
    for prev, cur in zip(finite_sorted, finite_sorted[1:]):
        if prev.max_length == cur.max_length:
            raise ValueError(f"dynamic_batch_size_schedule 存在重复 max_length={prev.max_length}")
    return finite_sorted + catch_all


def resolve_dynamic_batch_size(signal_length: int, schedule: list[DynamicBatchSizeRule]) -> int:
    if signal_length < 1:
        raise ValueError(f"signal_length 必须 >= 1，当前为 {signal_length}")
    for rule in schedule:
        if rule.max_length is None or signal_length <= rule.max_length:
            return rule.batch_size
    raise ValueError(
        f"信号长度 {signal_length} 未命中 dynamic_batch_size_schedule；请增加 max_length: null 兜底项"
    )


@dataclass(frozen=True)
class _DatasetSegment:
    offset: int
    size: int
    signal_length: int
    h5_name: str
    num_classes: int = 0

    @property
    def dataset_equal_weight(self) -> float:
        """桶内选 H5 时等权：每个子数据集被选中的概率相同。"""
        return 1.0 if self.size > 0 else 0.0

    @property
    def weight(self) -> float:
        return self.dataset_equal_weight


def segment_num_classes(h5_name: str, segment_size: int) -> int:
    """从 H5 文件名推断 emitter 类别数（用于桶权重）；非 emitter 文件返回样本数。"""
    stem = h5_name
    for suffix in ("_train.h5", "_val.h5", "_test.h5"):
        if stem.endswith(suffix):
            stem = stem[: -len(suffix)]
            break
    known = {"adsb2": 100, "wifi150": 150, "radar_emitters": 10, "communication_emitters": 30}
    return known.get(stem, max(segment_size, 1))


def pool_segments(pool: RFDataPoolDataset, *, infer_emitter_classes: bool = False) -> list[_DatasetSegment]:
    segments: list[_DatasetSegment] = []
    offset = 0
    for sub in pool.datasets:
        n = len(sub)
        segments.append(
            _DatasetSegment(
                offset=offset,
                size=n,
                signal_length=int(sub.signal_length),
                h5_name=sub.h5_path.name,
                num_classes=segment_num_classes(sub.h5_path.name, n) if infer_emitter_classes else 0,
            )
        )
        offset += n
    if offset != len(pool):
        raise ValueError(f"pool {pool.pool_name!r} segment 总长度 {offset} != {len(pool)}")
    return segments


def build_uniform_sampler(pool: RFDataPoolDataset) -> WeightedRandomSampler:
    """每条样本被看到概率相同：P(i) = 1 / len(pool)。"""
    if len(pool) <= 0:
        raise ValueError(f"pool {pool.pool_name!r} 为空，无法构建 uniform 采样器")
    weights = torch.ones(len(pool), dtype=torch.double)
    return WeightedRandomSampler(weights, num_samples=len(pool), replacement=True)


def build_dataset_balanced_sampler(pool: RFDataPoolDataset) -> WeightedRandomSampler:
    """按子 H5 数据集等权采样：每个 dataset 的期望贡献接近 1 / num_datasets。"""
    weights = torch.zeros(len(pool), dtype=torch.double)
    offset = 0
    for sub in pool.datasets:
        n = len(sub)
        if n > 0:
            weights[offset : offset + n] = 1.0 / n
        offset += n
    if offset != len(pool) or weights.sum() <= 0:
        raise ValueError(f"无法为 pool {pool.pool_name!r} 构建等权采样权重")
    return WeightedRandomSampler(weights, num_samples=len(pool), replacement=True)


def bucket_weights_for_segments(
    segs: list[_DatasetSegment],
    mode: LengthBucketWeightMode,
) -> float:
    if not segs:
        return 0.0
    if mode == "equal":
        return 1.0
    if mode == "proportional":
        return float(sum(seg.size for seg in segs))
    if mode == "class_count":
        return float(sum(max(seg.num_classes, 1) for seg in segs))
    raise ValueError(f"未知 length bucket 权重模式 {mode!r}")


def _sample_bucket_batch(
    rng: random.Random,
    segs: list[_DatasetSegment],
    seg_weights: list[float],
    batch_size: int,
) -> list[int]:
    """一次抽 batch_size 个 segment 下标，避免 Python 层 800 次 rng.choices(k=1)。"""
    if not segs:
        raise ValueError("长度桶为空，无法采样")
    if len(segs) == 1:
        seg = segs[0]
        return [seg.offset + rng.randrange(seg.size) for _ in range(batch_size)]
    picks = rng.choices(range(len(segs)), weights=seg_weights, k=batch_size)
    return [segs[i].offset + rng.randrange(segs[i].size) for i in picks]


class LengthBucketBalancedBatchSampler(Sampler[list[int]]):
    """同一 batch 内 I/Q 长度一致；桶间按权重选长度；桶内各子 H5 等权。"""

    def __init__(
        self,
        pool: RFDataPoolDataset,
        batch_size: int,
        *,
        num_batches: int | None = None,
        seed: int | None = None,
        within_bucket_dataset_equal: bool = True,
        bucket_weight_mode: LengthBucketWeightMode = "equal",
        infer_emitter_classes: bool = False,
    ) -> None:
        if batch_size < 1:
            raise ValueError(f"batch_size 必须 >= 1，当前为 {batch_size}")
        self.batch_size = batch_size
        self.within_bucket_dataset_equal = within_bucket_dataset_equal
        self.bucket_weight_mode = bucket_weight_mode
        self._segments = pool_segments(pool, infer_emitter_classes=infer_emitter_classes)
        self._num_batches = (
            num_batches if num_batches is not None else max(1, (len(pool) + batch_size - 1) // batch_size)
        )

        by_length: dict[int, list[_DatasetSegment]] = defaultdict(list)
        for seg in self._segments:
            if seg.size > 0:
                by_length[seg.signal_length].append(seg)
        if not by_length:
            raise ValueError(f"pool {pool.pool_name!r} 无有效样本")
        self._by_length = dict(sorted(by_length.items()))
        self._lengths = list(self._by_length.keys())
        self._length_weights = [
            bucket_weights_for_segments(self._by_length[length], bucket_weight_mode) for length in self._lengths
        ]
        self._seed = seed

    @property
    def length_buckets(self) -> dict[int, list[str]]:
        return {length: [seg.h5_name for seg in segs] for length, segs in self._by_length.items()}

    @property
    def length_bucket_weights(self) -> dict[int, float]:
        return {
            length: weight
            for length, weight in zip(self._lengths, self._length_weights, strict=True)
        }

    def __len__(self) -> int:
        return self._num_batches

    def _segment_weights(self, segs: list[_DatasetSegment]) -> list[float]:
        if self.within_bucket_dataset_equal:
            return [seg.dataset_equal_weight for seg in segs]
        return [float(seg.size) for seg in segs]

    def __iter__(self) -> Iterator[list[int]]:
        rng = random.Random(self._seed)
        lengths = self._lengths
        weights = self._length_weights
        by_length = self._by_length
        batch_size = self.batch_size
        for _ in range(self._num_batches):
            length = rng.choices(lengths, weights=weights, k=1)[0]
            segs = by_length[length]
            yield _sample_bucket_batch(rng, segs, self._segment_weights(segs), batch_size)


class LengthBucketDynamicBatchSampler(Sampler[list[int]]):
    """同长度桶采样，并按信号长度查表动态调整 batch_size（短序列大 batch / 长序列小 batch）。"""

    def __init__(
        self,
        pool: RFDataPoolDataset,
        schedule: list[dict] | list[DynamicBatchSizeRule],
        *,
        num_batches: int | None = None,
        seed: int | None = None,
        within_bucket_dataset_equal: bool = True,
        bucket_weight_mode: LengthBucketWeightMode = "equal",
        infer_emitter_classes: bool = False,
    ) -> None:
        if schedule and isinstance(schedule[0], DynamicBatchSizeRule):
            self._schedule = list(schedule)  # type: ignore[arg-type]
        else:
            self._schedule = parse_dynamic_batch_size_schedule(schedule)  # type: ignore[arg-type]
        self.within_bucket_dataset_equal = within_bucket_dataset_equal
        self.bucket_weight_mode = bucket_weight_mode
        self._segments = pool_segments(pool, infer_emitter_classes=infer_emitter_classes)

        by_length: dict[int, list[_DatasetSegment]] = defaultdict(list)
        for seg in self._segments:
            if seg.size > 0:
                by_length[seg.signal_length].append(seg)
        if not by_length:
            raise ValueError(f"pool {pool.pool_name!r} 无有效样本")
        self._by_length = dict(sorted(by_length.items()))
        self._lengths = list(self._by_length.keys())
        self._length_weights = [
            bucket_weights_for_segments(self._by_length[length], bucket_weight_mode) for length in self._lengths
        ]
        self._batch_size_by_length = {
            length: resolve_dynamic_batch_size(length, self._schedule) for length in self._lengths
        }
        if num_batches is not None:
            self._num_batches = max(1, int(num_batches))
        else:
            estimated = 0
            for length, segs in self._by_length.items():
                n = sum(seg.size for seg in segs)
                bs = self._batch_size_by_length[length]
                estimated += max(1, (n + bs - 1) // bs)
            self._num_batches = max(1, estimated)
        self._seed = seed

    @property
    def schedule(self) -> list[DynamicBatchSizeRule]:
        return list(self._schedule)

    @property
    def batch_size_by_length(self) -> dict[int, int]:
        return dict(self._batch_size_by_length)

    @property
    def length_buckets(self) -> dict[int, list[str]]:
        return {length: [seg.h5_name for seg in segs] for length, segs in self._by_length.items()}

    @property
    def length_bucket_weights(self) -> dict[int, float]:
        return {
            length: weight
            for length, weight in zip(self._lengths, self._length_weights, strict=True)
        }

    def __len__(self) -> int:
        return self._num_batches

    def _segment_weights(self, segs: list[_DatasetSegment]) -> list[float]:
        if self.within_bucket_dataset_equal:
            return [seg.dataset_equal_weight for seg in segs]
        return [float(seg.size) for seg in segs]

    def __iter__(self) -> Iterator[list[int]]:
        rng = random.Random(self._seed)
        lengths = self._lengths
        weights = self._length_weights
        by_length = self._by_length
        for _ in range(self._num_batches):
            length = rng.choices(lengths, weights=weights, k=1)[0]
            segs = by_length[length]
            yield _sample_bucket_batch(
                rng, segs, self._segment_weights(segs), self._batch_size_by_length[length]
            )


def build_segment_class_indices(
    pool: RFDataPoolDataset,
    *,
    label_field: str = "emitter_id",
) -> list[dict[int, list[int]]]:
    """各子 H5 segment 内 local label -> local index 列表（仅 label >= 0）。"""
    per_segment: list[dict[int, list[int]]] = []
    for sub in pool.datasets:
        with h5py.File(sub.h5_path, "r") as f:
            if label_field not in f:
                raise KeyError(f"{sub.h5_path.name} 缺少标签字段 {label_field!r}，无法构建 PK 采样索引")
            labels = np.asarray(f[label_field][:], dtype=np.int64)
        by_class: dict[int, list[int]] = defaultdict(list)
        for local_idx, label in enumerate(labels):
            label_int = int(label)
            if label_int >= 0:
                by_class[label_int].append(local_idx)
        if not by_class:
            raise ValueError(f"{sub.h5_path.name} 无有效 {label_field} 标签，无法构建 PK 采样索引")
        per_segment.append(dict(by_class))
    return per_segment


class LengthBucketPKBatchSampler(Sampler[list[int]]):
    """同长度桶 + PK 采样：每 batch 抽 P 个类、每类 K 条，batch_size = P × K。"""

    def __init__(
        self,
        pool: RFDataPoolDataset,
        batch_size: int,
        *,
        pk_num_classes: int,
        pk_samples_per_class: int,
        num_batches: int | None = None,
        seed: int | None = None,
        within_bucket_dataset_equal: bool = True,
        bucket_weight_mode: LengthBucketWeightMode = "class_count",
        label_field: str = "emitter_id",
    ) -> None:
        if pk_num_classes < 2:
            raise ValueError(f"pk_num_classes 必须 >= 2，当前为 {pk_num_classes}")
        if pk_samples_per_class < 2:
            raise ValueError(f"pk_samples_per_class 必须 >= 2（对比学习需同类正对），当前为 {pk_samples_per_class}")
        expected_batch = pk_num_classes * pk_samples_per_class
        if batch_size != expected_batch:
            raise ValueError(
                f"batch_size={batch_size} 与 PK 配置不一致："
                f"pk_num_classes({pk_num_classes}) × pk_samples_per_class({pk_samples_per_class}) = {expected_batch}"
            )
        self.batch_size = batch_size
        self.pk_num_classes = pk_num_classes
        self.pk_samples_per_class = pk_samples_per_class
        self.within_bucket_dataset_equal = within_bucket_dataset_equal
        self.bucket_weight_mode = bucket_weight_mode
        self.label_field = label_field
        self._segments = pool_segments(pool, infer_emitter_classes=True)
        self._class_indices = build_segment_class_indices(pool, label_field=label_field)
        if len(self._class_indices) != len(self._segments):
            raise RuntimeError("segment 与 class 索引数量不一致")
        self._class_by_offset = {
            seg.offset: class_map for seg, class_map in zip(self._segments, self._class_indices, strict=True)
        }
        self._num_batches = (
            num_batches if num_batches is not None else max(1, (len(pool) + batch_size - 1) // batch_size)
        )

        by_length: dict[int, list[_DatasetSegment]] = defaultdict(list)
        for seg in self._segments:
            if seg.size > 0:
                by_length[seg.signal_length].append(seg)
        if not by_length:
            raise ValueError(f"pool {pool.pool_name!r} 无有效样本")
        self._by_length = dict(sorted(by_length.items()))
        self._lengths = list(self._by_length.keys())
        self._length_weights = [
            bucket_weights_for_segments(self._by_length[length], bucket_weight_mode) for length in self._lengths
        ]
        self._seed = seed

    @property
    def length_buckets(self) -> dict[int, list[str]]:
        return {length: [seg.h5_name for seg in segs] for length, segs in self._by_length.items()}

    @property
    def length_bucket_weights(self) -> dict[int, float]:
        return {
            length: weight
            for length, weight in zip(self._lengths, self._length_weights, strict=True)
        }

    def __len__(self) -> int:
        return self._num_batches

    def _segment_weights(self, segs: list[_DatasetSegment]) -> list[float]:
        if self.within_bucket_dataset_equal:
            return [seg.dataset_equal_weight for seg in segs]
        return [float(seg.size) for seg in segs]

    def __iter__(self) -> Iterator[list[int]]:
        rng = random.Random(self._seed)
        lengths = self._lengths
        weights = self._length_weights
        by_length = self._by_length
        for _ in range(self._num_batches):
            length = rng.choices(lengths, weights=weights, k=1)[0]
            segs = by_length[length]
            seg = rng.choices(segs, weights=self._segment_weights(segs), k=1)[0]
            class_map = self._class_by_offset[seg.offset]
            all_classes = list(class_map.keys())
            if len(all_classes) >= self.pk_num_classes:
                chosen_classes = rng.sample(all_classes, self.pk_num_classes)
            else:
                chosen_classes = rng.choices(all_classes, k=self.pk_num_classes)
            batch: list[int] = []
            for cls in chosen_classes:
                local_indices = class_map[cls]
                for _ in range(self.pk_samples_per_class):
                    batch.append(seg.offset + rng.choice(local_indices))
            yield batch


def resolve_pk_sampling_params(
    train_cfg: dict | None,
    batch_size: int,
    *,
    pk_num_classes: int | None = None,
    pk_samples_per_class: int | None = None,
) -> tuple[int, int]:
    cfg = train_cfg or {}
    p = pk_num_classes if pk_num_classes is not None else cfg.get("pk_num_classes")
    k = pk_samples_per_class if pk_samples_per_class is not None else cfg.get("pk_samples_per_class")
    if p is None and k is None:
        k = 8
        p = max(2, batch_size // int(k))
    elif p is None:
        p = max(2, batch_size // int(k))
    elif k is None:
        k = max(2, batch_size // int(p))
    p_int = int(p)
    k_int = int(k)
    if p_int * k_int != batch_size:
        raise ValueError(
            f"PK 采样要求 batch_size == pk_num_classes × pk_samples_per_class，"
            f"当前 {batch_size} != {p_int} × {k_int}"
        )
    return p_int, k_int


def resolve_length_bucket_weight_mode(
    strategy: BalancedSamplingStrategy,
    train_cfg: dict | None = None,
) -> LengthBucketWeightMode:
    if train_cfg:
        explicit = train_cfg.get("length_bucket_weight")
        if explicit:
            mode = str(explicit).strip().lower()
            if mode in ("equal", "proportional", "class_count"):
                return mode  # type: ignore[return-value]
            raise ValueError(f"未知 length_bucket_weight={explicit!r}")
    if strategy == "length_bucket_proportional":
        return "proportional"
    if strategy in ("length_bucket_class", "length_bucket_pk"):
        return "class_count"
    return "equal"


def resolve_balanced_sampling_strategy(
    *,
    enabled: bool,
    strategy: str | None,
) -> BalancedSamplingStrategy:
    if not enabled:
        return "none"
    normalized = (strategy or "length_bucket").strip().lower()
    if normalized in ("uniform", "sample", "sample_equal"):
        return "uniform"
    if normalized in ("dataset", "dataset_equal"):
        return "dataset"
    if normalized in ("length_bucket", "length-bucket", "bucket"):
        return "length_bucket"
    if normalized in ("length_bucket_proportional", "length-bucket-proportional", "bucket_proportional"):
        return "length_bucket_proportional"
    if normalized in ("length_bucket_class", "length-bucket-class", "bucket_class", "emitter_class"):
        return "length_bucket_class"
    if normalized in ("length_bucket_pk", "length-bucket-pk", "bucket_pk", "pk", "class_balanced_pk"):
        return "length_bucket_pk"
    if normalized in ("length_bucket_dynamic", "length-bucket-dynamic", "bucket_dynamic", "dynamic"):
        return "length_bucket_dynamic"
    raise ValueError(
        "未知 balanced_sampling_strategy="
        f"{strategy!r}，可选: uniform | dataset | length_bucket | "
        "length_bucket_proportional | length_bucket_class | length_bucket_pk | length_bucket_dynamic"
    )


def format_sampling_plan(
    pool: RFDataPoolDataset,
    strategy: BalancedSamplingStrategy,
    *,
    bucket_weight_mode: LengthBucketWeightMode = "equal",
    infer_emitter_classes: bool = False,
    dynamic_batch_size_schedule: list[dict] | list[DynamicBatchSizeRule] | None = None,
) -> str:
    lines = [f"balanced_sampling_strategy={strategy}"]
    if strategy == "none":
        return "\n".join(lines)

    segments = pool_segments(pool, infer_emitter_classes=infer_emitter_classes)
    by_length: dict[int, list[_DatasetSegment]] = defaultdict(list)
    for seg in segments:
        if seg.size > 0:
            by_length[seg.signal_length].append(seg)

    bucket_weight_by_length = {
        length: bucket_weights_for_segments(segs, bucket_weight_mode) for length, segs in by_length.items()
    }
    total_bucket_weight = sum(bucket_weight_by_length.values())
    dynamic_bs: dict[int, int] | None = None
    if strategy == "length_bucket_dynamic":
        rules = (
            list(dynamic_batch_size_schedule)  # type: ignore[arg-type]
            if dynamic_batch_size_schedule and isinstance(dynamic_batch_size_schedule[0], DynamicBatchSizeRule)
            else parse_dynamic_batch_size_schedule(dynamic_batch_size_schedule)  # type: ignore[arg-type]
        )
        dynamic_bs = {length: resolve_dynamic_batch_size(length, rules) for length in by_length}

    for length in sorted(by_length):
        segs = by_length[length]
        names = ", ".join(f"{s.h5_name} ({s.size:,})" for s in segs)
        bucket_w = bucket_weight_by_length[length]
        share = bucket_w / max(total_bucket_weight, 1.0)
        bs_note = f", batch_size={dynamic_bs[length]}" if dynamic_bs is not None else ""
        if strategy.startswith("length_bucket") and len(segs) > 1:
            lines.append(
                f"length={length}: {names}  [桶权重={bucket_w:g}, 期望占比≈{share:.1%}{bs_note}, 桶内 {len(segs)} 个子 H5 等权]"
            )
        elif strategy.startswith("length_bucket"):
            detail = f"classes={segs[0].num_classes}" if segs[0].num_classes else f"samples={segs[0].size:,}"
            lines.append(f"length={length}: {names}  [桶权重={bucket_w:g}, 期望占比≈{share:.1%}{bs_note}, {detail}]")
        else:
            lines.append(f"length={length}: {names}")

    for seg in segments:
        if seg.size > 0:
            extra = f", classes={seg.num_classes}" if seg.num_classes else ""
            lines.append(f"{seg.h5_name}: {seg.size:,} samples, L={seg.signal_length}{extra}")

    if strategy == "length_bucket_pk":
        mode_desc = {
            "equal": "桶间均匀",
            "proportional": "桶间按样本数加权",
            "class_count": "桶间按类别数加权（个体识别推荐）",
        }[bucket_weight_mode]
        lines.append(
            f"每个 batch 仅从同一 length 桶采样；{mode_desc}；"
            "桶内 PK 采样（P 类 × K 样本/类），用于加强监督对比学习。"
        )
    elif strategy == "length_bucket_dynamic":
        mode_desc = {
            "equal": "桶间均匀",
            "proportional": "桶间按样本数加权",
            "class_count": "桶间按类别数加权（个体识别推荐）",
        }[bucket_weight_mode]
        lines.append(
            f"每个 batch 仅从同一 length 桶采样；{mode_desc}；桶内各子 H5 等权；"
            "batch_size 按 dynamic_batch_size_schedule 随长度自适应（短序列大 batch / 长序列小 batch）。"
        )
    elif strategy.startswith("length_bucket"):
        mode_desc = {
            "equal": "桶间均匀",
            "proportional": "桶间按样本数加权",
            "class_count": "桶间按类别数加权（个体识别推荐）",
        }[bucket_weight_mode]
        lines.append(f"每个 batch 仅从同一 length 桶采样；{mode_desc}；桶内各子 H5 等权。")
    elif strategy == "uniform":
        lines.append("WeightedRandomSampler：每条样本等概率 1/N（batch 内长度可能混合）。")
    elif strategy == "dataset":
        lines.append("WeightedRandomSampler：各子 H5 等权（batch 内长度可能混合）。")

    return "\n".join(lines)


def n_tokens_for_length(length: int, patch_size: int) -> int:
    return max(1, (int(length) + int(patch_size) - 1) // int(patch_size))


def pool_sample_lengths(pool: RFDataPoolDataset) -> list[int]:
    lengths: list[int] = []
    for sub in pool.datasets:
        per_sample = getattr(sub, "_length", None)
        if per_sample is None:
            lengths.extend([int(sub.signal_length)] * len(sub))
        else:
            lengths.extend(int(x) for x in per_sample.tolist())
    if len(lengths) != len(pool):
        raise ValueError(f"pool {pool.pool_name!r} 长度列表 {len(lengths)} != {len(pool)}")
    return lengths


# 分层类别：优先 global_label_id，再规范化调制 / 局部调制 / 信源 / 全局辐射源。
CLASS_ID_FIELDS = (
    "global_label_id",
    "canonical_mod_label_id",
    "mod_label_id",
    "source_label_id",
    "global_emitter_id",
    "emitter_id",
)


def resolve_sample_class_id(sample: dict) -> int:
    """从 sample dict 解析分层用的 class id；无有效标签时退回 dataset_id。"""
    for key in CLASS_ID_FIELDS:
        value = sample.get(key)
        if value is None:
            continue
        class_id = int(value.item() if hasattr(value, "item") else value)
        if class_id >= 0:
            return class_id
    dataset_id = sample.get("dataset_id", 0)
    if dataset_id is None:
        return 0
    return max(0, int(dataset_id.item() if hasattr(dataset_id, "item") else dataset_id))


def pool_sample_class_ids(pool: RFDataPoolDataset) -> list[int]:
    """各子 H5 列上解析 class id，避免为了分层去读 I/Q。"""
    ids: list[int] = []
    for sub in pool.datasets:
        n = len(sub)
        labels = getattr(sub, "_labels", None) or {}
        chosen = np.full(n, -1, dtype=np.int64)
        for key in CLASS_ID_FIELDS:
            col = labels.get(key)
            if col is None:
                continue
            col = np.asarray(col, dtype=np.int64)
            if col.shape[0] != n:
                continue
            fill = (chosen < 0) & (col >= 0)
            chosen[fill] = col[fill]
            if bool(np.all(chosen >= 0)):
                break
        if np.any(chosen < 0):
            dataset_ids = getattr(sub, "_dataset_id", None)
            if dataset_ids is not None:
                dataset_ids = np.maximum(np.asarray(dataset_ids, dtype=np.int64), 0)
                chosen = np.where(chosen >= 0, chosen, dataset_ids)
            else:
                chosen = np.where(chosen >= 0, chosen, 0)
        ids.extend(int(x) for x in chosen.tolist())
    if len(ids) != len(pool):
        raise ValueError(f"pool {pool.pool_name!r} class id 列表 {len(ids)} != {len(pool)}")
    return ids


def dataset_class_ids(dataset) -> list[int]:
    class_labels_fn = getattr(dataset, "class_labels", None)
    if callable(class_labels_fn):
        labels = [int(x) for x in class_labels_fn()]
        if len(labels) != len(dataset):
            raise ValueError(f"class_labels 长度 {len(labels)} != {len(dataset)}")
        return labels
    if isinstance(dataset, RFDataPoolDataset):
        return pool_sample_class_ids(dataset)
    return [resolve_sample_class_id(dataset[i]) for i in range(len(dataset))]


class TokenBudgetSampler(Sampler[list[int]]):
    """按 ``sum(ceil(L_i / patch_size)) <= token_budget`` 组 batch；超预算的单条单独成批，不静默截断。"""

    def __init__(
        self,
        lengths: list[int] | torch.Tensor,
        *,
        token_budget: int,
        patch_size: int,
        num_batches: int | None = None,
        seed: int | None = None,
    ) -> None:
        if token_budget < 1:
            raise ValueError(f"token_budget 必须 >= 1，当前 {token_budget}")
        if isinstance(lengths, torch.Tensor):
            lengths = [int(x) for x in lengths.tolist()]
        self.lengths = [int(x) for x in lengths]
        if not self.lengths:
            raise ValueError("TokenBudgetSampler 需要非空 lengths")
        self.token_budget = int(token_budget)
        self.patch_size = int(patch_size)
        self._tokens = [n_tokens_for_length(length, patch_size) for length in self.lengths]
        total_tokens = sum(self._tokens)
        self._num_batches = num_batches if num_batches is not None else max(1, (total_tokens + token_budget - 1) // token_budget)
        self._seed = seed

    def __len__(self) -> int:
        return self._num_batches

    def __iter__(self) -> Iterator[list[int]]:
        rng = random.Random(self._seed)
        order = list(range(len(self.lengths)))
        rng.shuffle(order)
        pos = 0
        n = len(order)
        for _ in range(self._num_batches):
            batch: list[int] = []
            used = 0
            while True:
                if pos >= n:
                    rng.shuffle(order)
                    pos = 0
                idx = order[pos]
                cost = self._tokens[idx]
                if not batch:
                    batch.append(idx)
                    used += cost
                    pos += 1
                    if used >= self.token_budget:
                        break
                    continue
                if used + cost > self.token_budget:
                    break
                batch.append(idx)
                used += cost
                pos += 1
                if used >= self.token_budget:
                    break
            yield batch


def _stratified_round_robin_order(labels: list[int], seed: int) -> list[int]:
    """各类内部 shuffle，再 round-robin 交错，使抽取在类别间尽量均匀。"""
    by_class: dict[int, list[int]] = defaultdict(list)
    for idx, label in enumerate(labels):
        by_class[int(label)].append(idx)
    rng = random.Random(int(seed))
    keys = sorted(by_class)
    rng.shuffle(keys)
    queues: list[list[int]] = []
    for key in keys:
        queue = list(by_class[key])
        rng.shuffle(queue)
        queues.append(queue)
    order: list[int] = []
    cursors = [0] * len(queues)
    active = list(range(len(queues)))
    while active:
        nxt: list[int] = []
        for qid in active:
            cursor = cursors[qid]
            queue = queues[qid]
            order.append(queue[cursor])
            cursor += 1
            cursors[qid] = cursor
            if cursor < len(queue):
                nxt.append(qid)
        active = nxt
    return order


def _pack_order_into_token_batches(
    order: list[int],
    tokens: list[int],
    token_budget: int,
    num_batches: int,
) -> list[list[int]]:
    batches: list[list[int]] = []
    pos = 0
    n = len(order)
    while len(batches) < max(1, int(num_batches)) and pos < n:
        batch: list[int] = []
        used = 0
        while pos < n:
            idx = order[pos]
            cost = tokens[idx]
            if batch and used + cost > token_budget:
                break
            batch.append(idx)
            used += cost
            pos += 1
            if used >= token_budget:
                break
        if batch:
            batches.append(batch)
    if not batches:
        raise ValueError("plan_fixed_token_budget_batches 得不到任何 batch")
    return batches


def regroup_token_batches_by_length(
    batches: list[list[int]],
    lengths: list[int],
    *,
    token_budget: int,
    patch_size: int,
) -> list[list[int]]:
    """把已抽出的 val 下标按信号长度重组成 batch，避免验证时长短混 pad 后逐条前向。"""
    selected: list[int] = []
    seen: set[int] = set()
    for batch in batches:
        for raw in batch:
            idx = int(raw)
            if idx in seen:
                continue
            seen.add(idx)
            selected.append(idx)
    if not selected:
        return list(batches)
    tokens = [n_tokens_for_length(int(length), patch_size) for length in lengths]
    by_len: dict[int, list[int]] = defaultdict(list)
    for idx in selected:
        by_len[int(lengths[idx])].append(idx)
    packed_groups = [
        _pack_order_into_token_batches(idxs, tokens, token_budget, num_batches=max(1, len(idxs)))
        for idxs in by_len.values()
        if idxs
    ]
    out: list[list[int]] = []
    max_n = max((len(group) for group in packed_groups), default=0)
    for i in range(max_n):
        for group in packed_groups:
            if i < len(group):
                out.append(group[i])
    return out or list(batches)


def plan_fixed_token_budget_batches(
    lengths: list[int] | torch.Tensor,
    *,
    token_budget: int,
    patch_size: int,
    num_batches: int,
    seed: int = 0,
    class_ids: list[int] | torch.Tensor | None = None,
) -> list[list[int]]:
    """从 val 下标固定抽出最多 ``num_batches`` 个 token-budget batch。

    不回头、不重洗：样本用尽即停。同一 ``seed`` 永远得到同一组下标。
    传入 ``class_ids`` 时按类别分层：每类内部 shuffle，再 round-robin 交错，
    多数类不会占满；某一类用尽就跳过，不回头重复抽。
    """
    if int(token_budget) < 1:
        raise ValueError(f"token_budget 必须 >= 1，当前 {token_budget}")
    if isinstance(lengths, torch.Tensor):
        lengths = [int(x) for x in lengths.tolist()]
    else:
        lengths = [int(x) for x in lengths]
    if not lengths:
        raise ValueError("plan_fixed_token_budget_batches 需要非空 lengths")
    if isinstance(class_ids, torch.Tensor):
        class_ids = [int(x) for x in class_ids.tolist()]
    if class_ids is not None and len(class_ids) != len(lengths):
        raise ValueError(f"class_ids 长度 {len(class_ids)} 与 lengths {len(lengths)} 不一致")
    token_budget = int(token_budget)
    tokens = [n_tokens_for_length(int(length), patch_size) for length in lengths]
    if class_ids is None:
        order = list(range(len(tokens)))
        random.Random(int(seed)).shuffle(order)
    else:
        order = _stratified_round_robin_order([int(x) for x in class_ids], seed)
    return _pack_order_into_token_batches(order, tokens, token_budget, num_batches)


class FixedBatchSampler(Sampler[list[int]]):
    """重复产出预先算好的 batch 下标，用于固定验证子集。"""

    def __init__(self, batches: list[list[int]]) -> None:
        if not batches:
            raise ValueError("FixedBatchSampler 需要非空 batches")
        self.batches = [list(batch) for batch in batches]

    def __len__(self) -> int:
        return len(self.batches)

    def __iter__(self) -> Iterator[list[int]]:
        for batch in self.batches:
            yield list(batch)

