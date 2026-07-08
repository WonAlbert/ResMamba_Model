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
]
LengthBucketWeightMode = Literal["equal", "proportional", "class_count"]


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
            seg_weights = self._segment_weights(segs)
            batch: list[int] = []
            for _ in range(batch_size):
                seg = rng.choices(segs, weights=seg_weights, k=1)[0]
                batch.append(seg.offset + rng.randrange(seg.size))
            yield batch


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
    raise ValueError(
        "未知 balanced_sampling_strategy="
        f"{strategy!r}，可选: uniform | dataset | length_bucket | "
        "length_bucket_proportional | length_bucket_class | length_bucket_pk"
    )


def format_sampling_plan(
    pool: RFDataPoolDataset,
    strategy: BalancedSamplingStrategy,
    *,
    bucket_weight_mode: LengthBucketWeightMode = "equal",
    infer_emitter_classes: bool = False,
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

    for length in sorted(by_length):
        segs = by_length[length]
        names = ", ".join(f"{s.h5_name} ({s.size:,})" for s in segs)
        bucket_w = bucket_weight_by_length[length]
        share = bucket_w / max(total_bucket_weight, 1.0)
        if strategy.startswith("length_bucket") and len(segs) > 1:
            lines.append(
                f"length={length}: {names}  [桶权重={bucket_w:g}, 期望占比≈{share:.1%}, 桶内 {len(segs)} 个子 H5 等权]"
            )
        elif strategy.startswith("length_bucket"):
            detail = f"classes={segs[0].num_classes}" if segs[0].num_classes else f"samples={segs[0].size:,}"
            lines.append(f"length={length}: {names}  [桶权重={bucket_w:g}, 期望占比≈{share:.1%}, {detail}]")
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
