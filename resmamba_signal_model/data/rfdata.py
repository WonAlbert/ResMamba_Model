from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import bisect
import json
import math
import os
import threading
import time
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch
from torch.utils.data import ConcatDataset, Dataset

from resmamba_signal_model.data.contracts import (
    CAPTURE_METADATA_KEYS,
    MISSING_METADATA,
    SIGNAL_CONTRACT_VERSION,
    SignalSpec,
    collate_signal_batch,
)
from resmamba_signal_model.data.labels import build_emitter_namespace, build_modulation_ontology
from resmamba_signal_model.data.splits import parse_split_filename


@dataclass(frozen=True)
class RFDataSample:
    iq: torch.Tensor
    length: int
    dataset_id: int
    task_type_id: int
    snr: float
    labels: dict[str, int]


FLOAT_METADATA_KEYS = (
    "snr", "sampling_rate", "sample_rate_hz", "center_freq", "center_frequency_hz",
    "carrier_freq", "carrier_frequency_hz", "bandwidth", "bandwidth_hz", "symbol_rate",
    "symbol_rate_hz", "tx_power_dbm", "rx_power_dbm", "distance_m", "latitude", "longitude", "altitude_m",
)
STRING_METADATA_KEYS = ("capture_date", "raw_dtype", "raw_layout", "scale_policy", "modulation_name")
_TRAIN_INT_KEYS = (
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
_LABEL_KEYS = (
    "mod_label_id",
    "canonical_mod_label_id",
    "emitter_id",
    "global_emitter_id",
    "source_label_id",
    "global_label_id",
)
_NUMPY_NORM_MODES = frozenset({"joint_power", "complex_absmax"})
# 每个 H5 handle 的 chunk cache；worker × 文件数较多，不宜太大
_H5_RDCC_NBYTES = 16 * 1024 * 1024
_H5_RDCC_NSLOTS = 10007
# 进程内按文件共享的 I/Q 缓存。必须在 DataLoader fork 之前填好，worker 只读、写时拷页。
_IQ_RAM_CACHE: dict[str, np.ndarray] = {}
_IQ_RAM_LOCK = threading.Lock()
_LABEL_NAMESPACE_CACHE: dict[str, tuple[float, dict[str, Any]]] = {}


def remove_iq_dc(iq: torch.Tensor) -> torch.Tensor:
    return iq - iq.mean(dim=-1, keepdim=True)


def center_spectral_peak(iq: torch.Tensor) -> torch.Tensor:
    if iq.ndim != 2 or iq.shape[0] != 2:
        raise ValueError("center_spectral_peak 仅适用于 [2,L] 复数 I/Q")
    if iq.shape[-1] < 8:
        return iq
    z = torch.complex(iq[0].float(), iq[1].float())
    spectrum = torch.fft.fft(z)
    peak = int(torch.argmax(spectrum.abs()).item())
    length = int(z.shape[-1])
    centered_peak = peak if peak <= length // 2 else peak - length
    if centered_peak == 0:
        return iq
    n = torch.arange(length, device=iq.device, dtype=torch.float32)
    rot = torch.exp(-2j * math.pi * float(centered_peak) * n / float(length))
    shifted = z * rot
    return torch.stack([shifted.real, shifted.imag], dim=0).to(dtype=iq.dtype)


def normalize_iq_numpy(iq: np.ndarray, mode: str = "joint_power") -> np.ndarray:
    """CPU worker 热路径：避免在 DataLoader 子进程里建 Torch 张量。"""
    if mode == "none":
        return np.asarray(iq, dtype=np.float32)
    x = np.asarray(iq, dtype=np.float32)
    if mode == "joint_power":
        power = float(np.mean(np.square(x).sum(axis=0)))
        scale = math.sqrt(max(power, 1.0e-8))
        return np.clip(x / scale, -5.0, 5.0).astype(np.float32, copy=False)
    if mode == "complex_absmax":
        scale = float(np.linalg.norm(x, axis=0).max())
        scale = max(scale, 1.0e-8)
        return (x / scale).astype(np.float32, copy=False)
    raise ValueError("未知 I/Q 归一化方式 %r，可选：joint_power、complex_absmax、emitter_iq_balance_peak、none" % mode)


def normalize_iq(iq: torch.Tensor, mode: str = "joint_power") -> torch.Tensor:
    if mode == "none":
        return iq
    if iq.device.type == "cpu" and mode in _NUMPY_NORM_MODES:
        return torch.from_numpy(np.ascontiguousarray(normalize_iq_numpy(iq.detach().contiguous().numpy(), mode)))
    if mode == "joint_power":
        power = iq.float().square().sum(dim=0).mean().clamp_min(1.0e-8)
        return (iq / torch.sqrt(power)).clamp(-5.0, 5.0)
    if mode == "complex_absmax":
        scale = torch.linalg.vector_norm(iq.float(), dim=0).max().clamp_min(1.0e-8)
        return iq / scale
    if mode == "emitter_iq_balance_peak":
        if iq.ndim != 2 or iq.shape[0] != 2:
            raise ValueError("emitter_iq_balance_peak 仅适用于 [2,L] 复数 I/Q")
        iq = center_spectral_peak(remove_iq_dc(iq))
        rms = torch.sqrt(iq.square().mean(dim=-1, keepdim=True) + 1.0e-8)
        iq = iq / rms.clamp_min(1.0e-4)
        scale = torch.linalg.vector_norm(iq.float(), dim=0).max().clamp_min(1.0e-8)
        return iq / scale
    raise ValueError("未知 I/Q 归一化方式 %r，可选：joint_power、complex_absmax、emitter_iq_balance_peak、none" % mode)


def open_rfdata_h5(path: str | Path) -> h5py.File:
    return h5py.File(
        path,
        "r",
        rdcc_nbytes=_H5_RDCC_NBYTES,
        rdcc_nslots=_H5_RDCC_NSLOTS,
    )


def _available_ram_bytes() -> int:
    try:
        with open("/proc/meminfo", encoding="utf-8") as fh:
            for line in fh:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) * 1024
    except OSError:
        pass
    try:
        return int(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_AVPHYS_PAGES"))
    except (ValueError, OSError):
        return 0


def resolve_cache_iq_in_memory(value: Any, *, nbytes: int) -> bool:
    if isinstance(value, str):
        value = value.strip().lower()
    if value in (False, "off", "false", "0", 0, None):
        return False
    if value in (True, "eager", "on", "true", "1", 1):
        return True
    avail = _available_ram_bytes()
    if avail <= 0:
        return nbytes <= 8 * 1024**3
    return (nbytes + 4 * 1024**3) < (avail * 0.5)


def clear_iq_ram_cache() -> None:
    with _IQ_RAM_LOCK:
        _IQ_RAM_CACHE.clear()


def iq_ram_cache_stats() -> dict[str, Any]:
    with _IQ_RAM_LOCK:
        items = tuple(_IQ_RAM_CACHE.items())
    total = int(sum(arr.nbytes for _key, arr in items))
    return {
        "files": len(items),
        "bytes": total,
        "entries": [
            {"path": key, "shape": tuple(arr.shape), "bytes": int(arr.nbytes)}
            for key, arr in items
        ],
    }


def format_iq_ram_cache() -> str:
    stats = iq_ram_cache_stats()
    gib = stats["bytes"] / 1024**3
    if stats["files"] == 0:
        return "iq_ram_cache: off (0 files)"
    names = ", ".join(Path(entry["path"]).name for entry in stats["entries"])
    return f"iq_ram_cache: {stats['files']} unique files, {gib:.2f} GiB [{names}]"


def get_shared_iq_array(path: str | Path) -> np.ndarray:
    """顺序读入整个信号数据集；同路径多 pool 共用一份。"""
    key = str(Path(path).resolve())
    with _IQ_RAM_LOCK:
        cached = _IQ_RAM_CACHE.get(key)
        if cached is not None:
            return cached
    started = time.perf_counter()
    print(f"iq_ram_cache: loading {Path(key).name} ...", flush=True)
    with open_rfdata_h5(key) as handle:
        arr = np.asarray(handle["iq"][:], dtype=np.float32)
        channel_axis = int(handle.attrs.get("channel_axis", 1))
    if arr.ndim != 3:
        raise ValueError(f"期望 iq 形状为 [N,C,L] 或 [N,L,C]，实际得到 {tuple(arr.shape)}，文件为 {key}")
    if channel_axis in (-1, 2) or (channel_axis == 1 and arr.shape[1] != 2 and arr.shape[2] == 2):
        arr = np.swapaxes(arr, 1, 2)
    arr = np.ascontiguousarray(arr, dtype=np.float32)
    with _IQ_RAM_LOCK:
        existing = _IQ_RAM_CACHE.get(key)
        if existing is not None:
            return existing
        _IQ_RAM_CACHE[key] = arr
    elapsed = time.perf_counter() - started
    print(
        f"iq_ram_cache: {Path(key).name} {arr.nbytes / 1024**3:.2f} GiB "
        f"shape={tuple(arr.shape)} in {elapsed:.1f}s"
    )
    return arr


def apply_iq_augmentation(iq: torch.Tensor, mode: str = "none") -> torch.Tensor:
    if mode == "none":
        return iq
    if mode == "emitter_freq_shift":
        if iq.ndim != 2 or iq.shape[0] != 2:
            raise ValueError("emitter_freq_shift 仅适用于 [2,L] 复数 I/Q")
        max_offset = 0.04
        offset = torch.empty((), dtype=torch.float32, device=iq.device).uniform_(-max_offset, max_offset)
        n = torch.arange(iq.shape[-1], device=iq.device, dtype=torch.float32)
        angle = 2.0 * math.pi * offset * n
        cos_angle = torch.cos(angle).to(dtype=iq.dtype)
        sin_angle = torch.sin(angle).to(dtype=iq.dtype)
        i = iq[0] * cos_angle - iq[1] * sin_angle
        q = iq[0] * sin_angle + iq[1] * cos_angle
        return torch.stack((i, q), dim=0).contiguous()
    raise ValueError("未知 I/Q 增强方式 %r，可选：none、emitter_freq_shift" % mode)


def _read_int_column(f: h5py.File, name: str, n: int) -> np.ndarray | None:
    if name not in f:
        return None
    data = f[name]
    if int(data.shape[0]) != n:
        return None
    return np.asarray(data[:], dtype=np.int64)


def _load_label_maps_for_h5(h5_path: Path) -> dict[str, Any] | None:
    if h5_path.parent.name != "h5":
        return None
    maps_path = h5_path.parent.parent / "label_maps.json"
    if not maps_path.is_file():
        return None
    key = str(maps_path.resolve())
    mtime = maps_path.stat().st_mtime
    cached = _LABEL_NAMESPACE_CACHE.get(key)
    if cached is not None and cached[0] == mtime:
        return cached[1]
    with maps_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        return None
    _LABEL_NAMESPACE_CACHE[key] = (mtime, payload)
    return payload


class RFDataH5Dataset(Dataset):
    def __init__(
        self,
        h5_path: str | Path,
        *,
        use_labels: bool = True,
        iq_normalize: str = "none",
        iq_augment: str = "none",
        include_extra_metadata: bool = False,
        cache_iq_in_memory: Any = False,
    ) -> None:
        super().__init__()
        self.h5_path = Path(h5_path)
        self.use_labels = use_labels
        self.iq_normalize = iq_normalize
        self.iq_augment = iq_augment
        self.include_extra_metadata = include_extra_metadata
        self._file: h5py.File | None = None
        self._iq_ram: np.ndarray | None = None
        self._length: np.ndarray | None = None
        self._dataset_id: np.ndarray | None = None
        self._task_type_id: np.ndarray | None = None
        self._labels: dict[str, np.ndarray] = {}
        self._extra_float: dict[str, np.ndarray] = {}
        self._extra_string: dict[str, np.ndarray] = {}
        self._capture_metadata: dict[str, np.ndarray] = {}
        self._sample_rate_hz: np.ndarray | None = None
        self._default_sample_rate_hz = float("nan")
        with open_rfdata_h5(self.h5_path) as f:
            self._len = int(f["iq"].shape[0])
            iq_shape = f["iq"].shape
            if len(iq_shape) != 3:
                raise ValueError(f"期望 iq 形状为 [N,C,L] 或 [N,L,C]，实际为 {tuple(iq_shape)}")
            self._channel_axis = int(f.attrs.get("channel_axis", 1))
            if self._channel_axis not in (-1, 1, 2):
                raise ValueError(f"非法 channel_axis={self._channel_axis}，文件为 {self.h5_path}")
            if self._channel_axis in (-1, 2):
                self._num_channels = int(iq_shape[2])
                self._signal_length = int(iq_shape[1])
            elif iq_shape[1] != 2 and iq_shape[2] == 2:
                # 无 attrs 的旧 [N,L,2] 文件。
                self._channel_axis = 2
                self._num_channels = int(iq_shape[2])
                self._signal_length = int(iq_shape[1])
            else:
                self._channel_axis = 1
                self._num_channels = int(iq_shape[1])
                self._signal_length = int(iq_shape[2])
            iq_nbytes = int(np.prod(iq_shape)) * 4
            length_arr = _read_int_column(f, "length", self._len)
            if length_arr is not None and int(length_arr.min()) == int(length_arr.max()) == self._signal_length:
                self._length = None
            else:
                self._length = length_arr
            self._dataset_id = _read_int_column(f, "dataset_id", self._len)
            self._task_type_id = _read_int_column(f, "task_type_id", self._len)
            if use_labels:
                for key in _LABEL_KEYS:
                    col = _read_int_column(f, key, self._len)
                    if col is not None:
                        self._labels[key] = col
                self._fill_semantic_labels_from_maps()
            for key in CAPTURE_METADATA_KEYS:
                if key in f and int(f[key].shape[0]) == self._len:
                    self._capture_metadata[key] = np.asarray(f[key][:])
            for key in ("sample_rate_hz", "sampling_rate"):
                if key in f and int(f[key].shape[0]) == self._len:
                    self._sample_rate_hz = np.asarray(f[key][:], dtype=np.float32)
                    break
            for key in ("sample_rate_hz", "sampling_rate"):
                if key in f.attrs:
                    try:
                        value = float(f.attrs[key])
                    except (TypeError, ValueError):
                        continue
                    if math.isfinite(value) and value > 0:
                        self._default_sample_rate_hz = value
                        break
            if include_extra_metadata:
                for key in FLOAT_METADATA_KEYS:
                    if key in f:
                        self._extra_float[key] = np.asarray(f[key][:], dtype=np.float32)
                for key in STRING_METADATA_KEYS:
                    if key in f:
                        self._extra_string[key] = np.asarray(f[key][:])
        spec_rate = self._default_sample_rate_hz if math.isfinite(self._default_sample_rate_hz) else None
        self.signal_spec = SignalSpec.rf(
            num_channels=self._num_channels,
            sample_rate_hz=spec_rate,
        )
        if resolve_cache_iq_in_memory(cache_iq_in_memory, nbytes=iq_nbytes):
            self._iq_ram = get_shared_iq_array(self.h5_path)

    def __len__(self) -> int:
        return self._len

    @property
    def signal_length(self) -> int:
        return self._signal_length

    @property
    def num_channels(self) -> int:
        return self._num_channels

    def _ensure_file(self) -> h5py.File:
        if self._file is None:
            self._file = open_rfdata_h5(self.h5_path)
        return self._file

    def close(self) -> None:
        if self._file is not None:
            self._file.close()
            self._file = None

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass

    def _int_at(self, column: np.ndarray | None, idx: int, default: int) -> int:
        if column is None:
            return default
        return int(column[idx])

    @staticmethod
    def _decode_string(value: Any, default: str = "") -> str:
        if value is None:
            return default
        if hasattr(value, "item"):
            value = value.item()
        if isinstance(value, bytes):
            return value.decode("utf-8", errors="replace")
        return str(value)

    def _iq_tensor_from_raw(self, raw: np.ndarray, length: int) -> torch.Tensor:
        if raw.ndim != 2:
            raise ValueError(f"期望 iq 形状为 [C,L]，实际得到 {tuple(raw.shape)}，文件为 {self.h5_path}")
        if raw.shape[0] != self._num_channels and raw.shape[1] == self._num_channels:
            raw = np.ascontiguousarray(raw.T)
        if raw.shape[0] != self._num_channels:
            raise ValueError(
                f"信号通道数不一致：期望 {self._num_channels}，实际 {tuple(raw.shape)}，文件为 {self.h5_path}"
            )
        if length < raw.shape[-1]:
            raw = raw[:, :length]
        if self.iq_normalize in _NUMPY_NORM_MODES:
            raw = normalize_iq_numpy(raw, self.iq_normalize)
            iq = torch.from_numpy(np.ascontiguousarray(raw))
        else:
            iq = torch.from_numpy(np.ascontiguousarray(raw))
            if self.iq_normalize != "none":
                iq = normalize_iq(iq, self.iq_normalize)
        if self.iq_augment != "none":
            iq = apply_iq_augmentation(iq, self.iq_augment)
        return iq

    def _load_iq(self, idx: int, length: int) -> torch.Tensor:
        if self._iq_ram is not None:
            row = self._iq_ram[idx]
            raw = np.array(row[:, :length] if row.shape[-1] >= length else row, dtype=np.float32, copy=True)
        else:
            raw = np.asarray(self._ensure_file()["iq"][idx], dtype=np.float32)
        return self._iq_tensor_from_raw(raw, length)

    def _gather_iq_rows(self, idx_arr: np.ndarray) -> np.ndarray:
        """按原下标顺序取出 iq 行；磁盘路径先排序再读，避免逐条随机 seek。"""
        if self._iq_ram is not None:
            return self._iq_ram[idx_arr]
        if idx_arr.size == 0:
            return np.empty((0, self._num_channels, self._signal_length), dtype=np.float32)
        order = np.argsort(idx_arr, kind="mergesort")
        sorted_idx = idx_arr[order]
        unique, inverse = np.unique(sorted_idx, return_inverse=True)
        handle = self._ensure_file()["iq"]
        start = int(unique[0])
        end = int(unique[-1]) + 1
        if end - start == int(unique.size):
            raw_unique = np.asarray(handle[start:end], dtype=np.float32)
        else:
            raw_unique = np.asarray(handle[unique.tolist()], dtype=np.float32)
        raw_sorted = raw_unique[inverse]
        gathered = np.empty_like(raw_sorted)
        gathered[order] = raw_sorted
        return gathered

    def _fill_semantic_labels_from_maps(self) -> None:
        """H5 尚未 stamp 时，用 label_maps ontology 回填 canonical / global emitter。"""
        need_canonical = "canonical_mod_label_id" not in self._labels
        need_emitter = "global_emitter_id" not in self._labels
        if not need_canonical and not need_emitter:
            return
        maps = _load_label_maps_for_h5(self.h5_path)
        if maps is None:
            return
        ontology = build_modulation_ontology(
            maps.get("modulations", {}),
            existing=maps.get("modulation_ontology"),
        )
        emitter_ns = build_emitter_namespace(
            maps.get("emitters", {}),
            existing=maps.get("emitter_namespace"),
        )
        id_to_name = {
            int(dataset_id): str(name)
            for dataset_id, name in dict(maps.get("datasets") or {}).items()
        }
        try:
            fallback_name, _split = parse_split_filename(self.h5_path.name)
        except ValueError:
            fallback_name = None
        n = int(self._len)
        dataset_ids = (
            np.asarray(self._dataset_id, dtype=np.int32)
            if self._dataset_id is not None
            else np.zeros(n, dtype=np.int32)
        )
        local_mod = self._labels.get("mod_label_id")
        if local_mod is None:
            local_mod = np.full(n, -1, dtype=np.int32)
        local_emitter = self._labels.get("emitter_id")
        if local_emitter is None:
            local_emitter = np.full(n, -1, dtype=np.int32)
        if need_canonical:
            canonical = np.full(n, -1, dtype=np.int32)
            for dataset_id in np.unique(dataset_ids):
                name = id_to_name.get(int(dataset_id), fallback_name)
                if not name:
                    continue
                choose = dataset_ids == dataset_id
                canonical[choose] = ontology.map_local(name, local_mod[choose])
            self._labels["canonical_mod_label_id"] = canonical
        if need_emitter:
            global_emitter = np.full(n, -1, dtype=np.int32)
            for dataset_id in np.unique(dataset_ids):
                name = id_to_name.get(int(dataset_id), fallback_name)
                if not name:
                    continue
                choose = dataset_ids == dataset_id
                global_emitter[choose] = emitter_ns.map_local(name, local_emitter[choose])
            self._labels["global_emitter_id"] = global_emitter

    def _items_from_iq_rows(self, idx_list: list[int], gathered: np.ndarray) -> list[dict[str, Any]]:
        if gathered.ndim != 3:
            return [self[i] for i in idx_list]
        if gathered.shape[1] != self._num_channels and gathered.shape[2] == self._num_channels:
            gathered = np.swapaxes(gathered, 1, 2)
        items: list[dict[str, Any]] = []
        for row, idx in zip(gathered, idx_list, strict=True):
            length = self._int_at(self._length, idx, self._signal_length)
            length = max(1, min(length, self._signal_length))
            items.append(self._sample_dict(idx, self._iq_tensor_from_raw(row, length), length))
        return items

    def _sample_dict(self, idx: int, iq: torch.Tensor, length: int) -> dict[str, Any]:
        out: dict[str, Any] = {
            "iq": iq,
            "values": iq,
            "length": length,
            "dataset_id": max(0, self._int_at(self._dataset_id, idx, 0)),
            "task_type_id": self._int_at(self._task_type_id, idx, 0),
            "modality_id": "rf",
            "complex_pairs": self.signal_spec.complex_pairs,
            "signal_spec": self.signal_spec,
            "sample_rate_hz": (
                float(self._sample_rate_hz[idx])
                if self._sample_rate_hz is not None
                else self._default_sample_rate_hz
            ),
        }
        if self.use_labels:
            for key in _LABEL_KEYS:
                out[key] = self._int_at(self._labels.get(key), idx, -1)
        for key in CAPTURE_METADATA_KEYS:
            column = self._capture_metadata.get(key)
            out[key] = self._decode_string(
                None if column is None else column[idx],
                MISSING_METADATA,
            )
            if not out[key]:
                out[key] = MISSING_METADATA
        if self.include_extra_metadata:
            out["h5_path"] = str(self.h5_path)
            for key, col in self._extra_float.items():
                out[key] = float(col[idx])
            for key in FLOAT_METADATA_KEYS:
                out.setdefault(key, float("nan") if key != "snr" else -999.0)
            for key, col in self._extra_string.items():
                out[key] = self._decode_string(col[idx], "")
            for key in STRING_METADATA_KEYS:
                out.setdefault(key, "")
        return out

    def __getitem__(self, idx: int) -> dict[str, Any]:
        length = self._int_at(self._length, idx, self._signal_length)
        length = max(1, min(length, self._signal_length))
        return self._sample_dict(idx, self._load_iq(idx, length), length)

    def __getitems__(self, indices: list[int]) -> list[dict[str, Any]]:
        idx_list = [int(i) for i in indices]
        if len(idx_list) <= 1:
            return [self[i] for i in idx_list]
        try:
            gathered = self._gather_iq_rows(np.asarray(idx_list, dtype=np.int64))
            return self._items_from_iq_rows(idx_list, gathered)
        except (OSError, ValueError, TypeError, IndexError):
            return [self[i] for i in idx_list]


class RFDataPoolDataset(ConcatDataset):
    def __init__(self, datasets: list[RFDataH5Dataset], *, pool_name: str) -> None:
        super().__init__(datasets)
        self.pool_name = pool_name

    def __getitems__(self, indices: list[int]) -> list[dict[str, Any]]:
        grouped: dict[int, list[tuple[int, int]]] = defaultdict(list)
        n = len(self)
        for pos, raw_idx in enumerate(indices):
            idx = int(raw_idx)
            if idx < 0:
                idx = n + idx
            dataset_idx = bisect.bisect_right(self.cumulative_sizes, idx)
            sample_idx = idx if dataset_idx == 0 else idx - self.cumulative_sizes[dataset_idx - 1]
            grouped[dataset_idx].append((pos, sample_idx))
        out: list[dict[str, Any] | None] = [None] * len(indices)
        for dataset_idx, pairs in grouped.items():
            dataset = self.datasets[dataset_idx]
            local = [sample_idx for _pos, sample_idx in pairs]
            getter = getattr(dataset, "__getitems__", None)
            items = getter(local) if callable(getter) else [dataset[i] for i in local]
            for (pos, _sample_idx), item in zip(pairs, items, strict=True):
                out[pos] = item
        return out  # type: ignore[return-value]


def close_rfdata_handles(dataset: Dataset) -> None:
    from torch.utils.data import ConcatDataset, Subset
    if isinstance(dataset, RFDataH5Dataset):
        dataset.close()
    elif isinstance(dataset, Subset):
        close_rfdata_handles(dataset.dataset)
    elif isinstance(dataset, ConcatDataset):
        for child in dataset.datasets:
            close_rfdata_handles(child)


def rfdata_dataloader_worker_init(_worker_id: int) -> None:
    import os

    import torch.utils.data as torch_data

    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[key] = "1"
    try:
        import torch

        torch.set_num_threads(1)
    except Exception:
        pass
    worker_info = torch_data.get_worker_info()
    if worker_info is not None:
        close_rfdata_handles(worker_info.dataset)


def load_task_pool(rfdata_root: str | Path, pool_name: str) -> list[str]:
    """加载 task pool 对应的 H5 文件名列表。

    数据划分约定（H5 文件名后缀，白名单见 ``configs/datasets.yaml``）：
    - ``*_train.h5``：MAE 预训练（``pretrain_train``）或下游训练（``downstream_*_train``、``clustering_train``）
    - ``*_val.h5``：各阶段唯一验证集（早停 / 选模 / 指标报告）
    """
    root = Path(rfdata_root)
    with (root / "label_maps.json").open("r", encoding="utf-8") as f:
        label_maps = json.load(f)
    pools = label_maps.get("task_pools", {})
    if pool_name not in pools:
        raise KeyError(f"未知 RFData pool {pool_name!r}，可选项为：{sorted(pools)}")
    return list(pools[pool_name])


def build_rfdata_pool(
    rfdata_root: str | Path,
    pool_name: str,
    *,
    use_labels: bool = True,
    iq_normalize: str = "none",
    iq_augment: str = "none",
    include_extra_metadata: bool = False,
    cache_iq_in_memory: Any = False,
) -> RFDataPoolDataset:
    root = Path(rfdata_root)
    return RFDataPoolDataset([
        RFDataH5Dataset(
            root / "h5" / name,
            use_labels=use_labels,
            iq_normalize=iq_normalize,
            iq_augment=iq_augment,
            include_extra_metadata=include_extra_metadata,
            cache_iq_in_memory=cache_iq_in_memory,
        )
        for name in load_task_pool(root, pool_name)
    ], pool_name=pool_name)


def rfdata_loader_worker_kwargs(
    num_workers: int,
    *,
    prefetch_factor: int = 4,
    persistent_workers: bool = True,
) -> dict[str, Any]:
    if num_workers <= 0:
        return {}
    return {
        "prefetch_factor": max(1, int(prefetch_factor)),
        "worker_init_fn": rfdata_dataloader_worker_init,
        "persistent_workers": bool(persistent_workers),
    }


def _collate_metadata(batch: list[dict[str, Any]], out: dict[str, Any]) -> dict[str, Any]:
    first = batch[0]
    for key in _TRAIN_INT_KEYS:
        if key in first and key not in out:
            default = -1 if key in _LABEL_KEYS else 0
            out[key] = torch.as_tensor([int(item.get(key, default)) for item in batch], dtype=torch.long)
    for key in FLOAT_METADATA_KEYS:
        if key in first:
            out[key] = torch.as_tensor([float(item.get(key, float("nan"))) for item in batch], dtype=torch.float32)
    for key in STRING_METADATA_KEYS:
        if key in first:
            out[key] = [str(item.get(key, "")) for item in batch]
    for key in CAPTURE_METADATA_KEYS:
        if key in first and key not in out:
            out[key] = [str(item.get(key, MISSING_METADATA)) for item in batch]
    if "h5_path" in first:
        out["h5_path"] = [str(item["h5_path"]) for item in batch]
    return out


def pad_iq_collate(batch: list[dict[str, Any]]) -> dict[str, Any]:
    """兼容旧 ``iq`` 键的任意通道 padding collate。"""
    signal_batch = collate_signal_batch(batch)
    return _collate_metadata(batch, signal_batch.to_legacy_dict())


def variable_length_collate(batch: list[dict[str, Any]]) -> dict[str, Any]:
    """不 pad I/Q；等长时 stack 成一张量，便于 pin_memory / H2D。"""
    lengths = [int(item["length"]) for item in batch]
    iqs = [item["iq"] for item in batch]
    same_len = bool(lengths) and all(
        length == lengths[0]
        and int(tensor.shape[-1]) == lengths[0]
        and int(tensor.shape[0]) == int(iqs[0].shape[0])
        for length, tensor in zip(lengths, iqs)
    )
    if same_len:
        out = collate_signal_batch(batch).to_legacy_dict()
    else:
        signal_specs: list[dict[str, Any]] = []
        for item, tensor in zip(batch, iqs, strict=True):
            raw_spec = item.get("signal_spec")
            if isinstance(raw_spec, SignalSpec):
                signal_specs.append(raw_spec.to_model_dict())
            elif isinstance(raw_spec, dict):
                signal_specs.append(dict(raw_spec))
            else:
                signal_specs.append(
                    SignalSpec.rf(num_channels=int(tensor.shape[0])).to_model_dict()
                )
        out = {
            "iq": iqs,
            "values": iqs,
            "length": torch.as_tensor(lengths, dtype=torch.long),
            "channel_mask": [
                torch.ones(int(tensor.shape[0]), dtype=torch.bool) for tensor in iqs
            ],
            "modality_id": [str(item.get("modality_id", "rf")) for item in batch],
            "signal_spec": signal_specs,
            "signal_specs": signal_specs,
            "signal_contract_version": SIGNAL_CONTRACT_VERSION,
        }
    return _collate_metadata(batch, out)
