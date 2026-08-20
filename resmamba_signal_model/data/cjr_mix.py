from __future__ import annotations

from dataclasses import dataclass
import glob
import os
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch
from torch.utils.data import ConcatDataset, Dataset

from resmamba_signal_model.data.contracts import MISSING_METADATA, SignalSpec

try:
    import pyarrow.parquet as pq
except ImportError as exc:  # pragma: no cover - optional dependency
    pq = None
    _PYARROW_IMPORT_ERROR = exc
else:
    _PYARROW_IMPORT_ERROR = None

SplitName = Literal["train", "val"]
NormMethod = Literal["abs", "joint_power", "none"]

DEFAULT_DATASET_DIRS = ("CJR-mix", "CJR_mix")
COLUMNS = ("iq", "infer_class", "snr", "fs", "dataset_name")


def require_pyarrow() -> None:
    if pq is None:
        raise ImportError(
            "读取 CJR-mix parquet 需要 pyarrow，请执行：pip install pyarrow"
        ) from _PYARROW_IMPORT_ERROR


def resolve_cjr_mix_root(rfdata_root: str | Path | None = None) -> Path:
    root = Path(rfdata_root or os.environ.get("RFDATA_ROOT", "dataset")).expanduser()
    if not root.is_absolute():
        root = (Path.cwd() / root).resolve()
    for name in DEFAULT_DATASET_DIRS:
        candidate = root / name
        if candidate.is_dir():
            return candidate
    raise FileNotFoundError(
        f"未找到 CJR-mix 数据目录，请在 {root} 下放置 CJR-mix/ 或 CJR_mix/（含 train/、val/ 子目录）"
    )


def list_parquet_files(data_root: str | Path, split: SplitName) -> list[Path]:
    require_pyarrow()
    root = Path(data_root)
    subdir = "val" if split == "val" else "train"
    pattern = str(root / subdir / "*.parquet")
    files = sorted(Path(path) for path in glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"未找到 parquet 文件：{pattern}")
    return files


def parse_iq_array(raw: Any) -> np.ndarray:
    """将 parquet 中的 iq 字段转为 float32 [2, L]。"""
    arr = np.asarray(raw, dtype=np.float32)
    if arr.ndim == 2 and arr.shape[-1] == 2:
        return np.transpose(arr, (1, 0))
    if arr.ndim == 3 and arr.shape[0] == 2 and arr.shape[-1] == 1:
        return arr[:, :, 0]
    if arr.ndim == 2 and arr.shape[0] == 2:
        return arr
    raise ValueError(f"不支持的 iq 形状：{arr.shape}")


def normalize_iq_absmax(iq: np.ndarray) -> np.ndarray:
    """EMind 默认：按样本 max-abs 归一化并裁剪到 [-5, 5]。"""
    scale = np.max(np.abs(iq), axis=(-2, -1), keepdims=True)
    out = iq / (scale + 1e-6)
    return np.clip(out, -5.0, 5.0).astype(np.float32, copy=False)


def normalize_iq_joint_power(iq: np.ndarray) -> np.ndarray:
    power = np.square(iq).sum(axis=0).mean()
    scale = float(np.sqrt(max(power, 1.0e-8)))
    out = iq / scale
    return np.clip(out, -5.0, 5.0).astype(np.float32, copy=False)


def normalize_iq_array(iq: np.ndarray, method: NormMethod) -> np.ndarray:
    if method == "none":
        return iq.astype(np.float32, copy=False)
    if method == "abs":
        return normalize_iq_absmax(iq)
    if method == "joint_power":
        return normalize_iq_joint_power(iq)
    raise ValueError(f"未知归一化方式 {method!r}，可选 abs、joint_power、none")


@dataclass(frozen=True)
class ParquetRowRef:
    file_index: int
    row_index: int


class _ParquetFileView:
    def __init__(self, path: Path) -> None:
        require_pyarrow()
        self.path = path
        self._pf = pq.ParquetFile(path)
        self.num_rows = int(self._pf.metadata.num_rows)
        self._row_group_rows = [
            int(self._pf.metadata.row_group(i).num_rows) for i in range(self._pf.num_row_groups)
        ]
        self._row_group_starts = np.cumsum([0] + self._row_group_rows[:-1], dtype=np.int64)

    def read_row(self, row_index: int, columns: tuple[str, ...] = COLUMNS) -> dict[str, Any]:
        if row_index < 0 or row_index >= self.num_rows:
            raise IndexError(f"行索引越界：{row_index} / {self.num_rows} @ {self.path}")
        group_idx = int(np.searchsorted(self._row_group_starts, row_index, side="right") - 1)
        offset = int(row_index - self._row_group_starts[group_idx])
        table = self._pf.read_row_group(group_idx, columns=list(columns))
        row = table.slice(offset, 1).to_pydict()
        return {key: row[key][0] for key in columns}


class CJRMixParquetDataset(Dataset):
    """直接读取 CJR-mix parquet（对齐 EMind 目录结构）。"""

    def __init__(
        self,
        data_root: str | Path,
        *,
        split: SplitName = "train",
        label_field: str = "infer_class",
        iq_normalize: NormMethod = "abs",
        snr_threshold: float | None = None,
        dataset_id: int = 0,
        task_type_id: int = 0,
        max_samples: int | None = None,
    ) -> None:
        super().__init__()
        require_pyarrow()
        self.data_root = Path(data_root)
        self.split = split
        self.label_field = label_field
        self.iq_normalize = iq_normalize
        self.snr_threshold = snr_threshold
        self.dataset_id = int(dataset_id)
        self.task_type_id = int(task_type_id)

        self.parquet_files = list_parquet_files(self.data_root, split)
        self._file_views = [_ParquetFileView(path) for path in self.parquet_files]
        self._index: list[ParquetRowRef] = []
        for file_index, view in enumerate(self._file_views):
            for row_index in range(view.num_rows):
                self._index.append(ParquetRowRef(file_index, row_index))
                if max_samples is not None and len(self._index) >= max_samples:
                    break
            if max_samples is not None and len(self._index) >= max_samples:
                break

        if not self._index:
            raise ValueError(f"{self.data_root}/{split} 为空，无法构建数据集")

    def __len__(self) -> int:
        return len(self._index)

    def _passes_filters(self, row: dict[str, Any]) -> bool:
        if self.snr_threshold is None:
            return True
        snr = row.get("snr")
        if snr is None:
            return False
        return float(snr) >= float(self.snr_threshold)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        ref = self._index[idx]
        row = self._file_views[ref.file_index].read_row(ref.row_index)
        if not self._passes_filters(row):
            return self.__getitem__((idx + 1) % len(self))

        iq = normalize_iq_array(parse_iq_array(row["iq"]), self.iq_normalize)
        length = int(iq.shape[-1])
        label = int(row[self.label_field])
        snr = float(row["snr"]) if row.get("snr") is not None else float("nan")
        fs = float(row["fs"]) if row.get("fs") is not None else float("nan")
        dataset_name = row.get("dataset_name", "")
        if isinstance(dataset_name, (bytes, bytearray)):
            dataset_name = dataset_name.decode("utf-8", errors="replace")

        return {
            "iq": torch.from_numpy(iq),
            "length": length,
            "modality_id": "rf",
            "complex_pairs": ((0, 1),),
            "signal_spec": SignalSpec.rf(
                num_channels=2,
                sample_rate_hz=fs if np.isfinite(fs) and fs > 0 else None,
            ),
            "dataset_id": self.dataset_id,
            "task_type_id": self.task_type_id,
            "snr": snr,
            "sample_rate_hz": fs,
            "mod_label_id": label,
            "canonical_mod_label_id": -1,
            "source_label_id": label,
            "emitter_id": -1,
            "global_emitter_id": -1,
            "global_label_id": -1,
            "receiver_id": MISSING_METADATA,
            "session_id": MISSING_METADATA,
            "channel_id": MISSING_METADATA,
            "capture_id": MISSING_METADATA,
            "dataset_name": dataset_name,
            "split": self.split,
            "parquet_path": str(self.parquet_files[ref.file_index]),
            "row_index": ref.row_index,
        }


class CJRMixPoolDataset(ConcatDataset):
    def __init__(self, datasets: list[CJRMixParquetDataset], *, pool_name: str) -> None:
        super().__init__(datasets)
        self.pool_name = pool_name


def build_cjr_mix_dataset(
    rfdata_root: str | Path | None = None,
    *,
    split: SplitName = "train",
    **kwargs: Any,
) -> CJRMixParquetDataset:
    data_root = resolve_cjr_mix_root(rfdata_root)
    return CJRMixParquetDataset(data_root, split=split, **kwargs)


def summarize_cjr_mix(rfdata_root: str | Path | None = None) -> dict[str, Any]:
    data_root = resolve_cjr_mix_root(rfdata_root)
    summary: dict[str, Any] = {"data_root": str(data_root), "splits": {}}
    for split in ("train", "val"):
        files = list_parquet_files(data_root, split)  # type: ignore[arg-type]
        total_rows = 0
        for path in files:
            total_rows += int(pq.ParquetFile(path).metadata.num_rows)
        summary["splits"][split] = {
            "num_files": len(files),
            "num_rows": total_rows,
            "files": [path.name for path in files],
        }
    return summary
