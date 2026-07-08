from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any

import h5py
import torch
from torch.utils.data import ConcatDataset, Dataset


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


def remove_iq_dc(iq: torch.Tensor) -> torch.Tensor:
    return iq - iq.mean(dim=-1, keepdim=True)


def center_spectral_peak(iq: torch.Tensor) -> torch.Tensor:
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


def normalize_iq(iq: torch.Tensor, mode: str = "joint_power") -> torch.Tensor:
    if mode == "none":
        return iq
    if mode == "joint_power":
        power = iq.float().square().sum(dim=0).mean().clamp_min(1.0e-8)
        return (iq / torch.sqrt(power)).clamp(-5.0, 5.0)
    if mode == "complex_absmax":
        scale = torch.linalg.vector_norm(iq.float(), dim=0).max().clamp_min(1.0e-8)
        return iq / scale
    if mode == "emitter_iq_balance_peak":
        iq = center_spectral_peak(remove_iq_dc(iq))
        rms = torch.sqrt(iq.square().mean(dim=-1, keepdim=True) + 1.0e-8)
        iq = iq / rms.clamp_min(1.0e-4)
        scale = torch.linalg.vector_norm(iq.float(), dim=0).max().clamp_min(1.0e-8)
        return iq / scale
    raise ValueError("未知 I/Q 归一化方式 %r，可选：joint_power、complex_absmax、emitter_iq_balance_peak、none" % mode)


def apply_iq_augmentation(iq: torch.Tensor, mode: str = "none") -> torch.Tensor:
    if mode == "none":
        return iq
    if mode == "emitter_freq_shift":
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


class RFDataH5Dataset(Dataset):
    def __init__(self, h5_path: str | Path, *, use_labels: bool = True, iq_normalize: str = "none", iq_augment: str = "none") -> None:
        super().__init__()
        self.h5_path = Path(h5_path)
        self.use_labels = use_labels
        self.iq_normalize = iq_normalize
        self.iq_augment = iq_augment
        self._file: h5py.File | None = None
        with h5py.File(self.h5_path, "r") as f:
            self._len = int(f["iq"].shape[0])
            self._signal_length = int(f["iq"].shape[2])

    def __len__(self) -> int:
        return self._len

    @property
    def signal_length(self) -> int:
        return self._signal_length

    def _ensure_file(self) -> h5py.File:
        if self._file is None:
            self._file = h5py.File(self.h5_path, "r")
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

    @staticmethod
    def _scalar(f: h5py.File, name: str, idx: int, default: int | float) -> int | float:
        if name not in f:
            return default
        value = f[name][idx]
        return value.item() if hasattr(value, "item") else value

    @staticmethod
    def _string_scalar(f: h5py.File, name: str, idx: int, default: str = "") -> str:
        if name not in f:
            return default
        value = f[name][idx]
        if hasattr(value, "item"):
            value = value.item()
        if isinstance(value, bytes):
            return value.decode("utf-8", errors="replace")
        return str(value)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        f = self._ensure_file()
        iq = torch.as_tensor(f["iq"][idx], dtype=torch.float32)
        if iq.ndim != 2:
            raise ValueError(f"期望 iq 形状为 [2,L]，实际得到 {tuple(iq.shape)}，文件为 {self.h5_path}")
        if iq.shape[0] != 2:
            iq = iq.t().contiguous()
        length = int(self._scalar(f, "length", idx, iq.shape[-1]))
        length = max(1, min(length, iq.shape[-1]))
        iq = apply_iq_augmentation(normalize_iq(iq[:, :length].contiguous(), self.iq_normalize), self.iq_augment)
        dataset_id = int(self._scalar(f, "dataset_id", idx, 0))
        out: dict[str, Any] = {
            "iq": iq,
            "length": length,
            "dataset_id": max(0, dataset_id),
            "task_type_id": int(self._scalar(f, "task_type_id", idx, 0)),
            "snr": float(self._scalar(f, "snr", idx, -999.0)),
            "h5_path": str(self.h5_path),
        }
        for key in FLOAT_METADATA_KEYS:
            if key != "snr":
                out[key] = float(self._scalar(f, key, idx, float("nan")))
        for key in STRING_METADATA_KEYS:
            out[key] = self._string_scalar(f, key, idx, "")
        if self.use_labels:
            for key in ("mod_label_id", "emitter_id", "source_label_id", "global_label_id"):
                out[key] = int(self._scalar(f, key, idx, -1))
        return out


class RFDataPoolDataset(ConcatDataset):
    def __init__(self, datasets: list[RFDataH5Dataset], *, pool_name: str) -> None:
        super().__init__(datasets)
        self.pool_name = pool_name


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
    import torch.utils.data as torch_data
    worker_info = torch_data.get_worker_info()
    if worker_info is not None:
        close_rfdata_handles(worker_info.dataset)


def load_task_pool(rfdata_root: str | Path, pool_name: str) -> list[str]:
    """加载 task pool 对应的 H5 文件名列表。

    数据划分约定（H5 文件名后缀）：
    - ``*_train.h5``：仅 MAE 预训练（``pretrain_train``）
    - ``*_val.h5``：各阶段唯一验证集（早停 / 选模 / 指标报告）
    - ``*_test.h5``：下游头与微调的训练数据（``downstream_*_train``、``clustering_train`` 等 pool），不作评测

    ``label_maps.json`` 中不再提供 ``*_test`` task pool。
    """
    root = Path(rfdata_root)
    with (root / "label_maps.json").open("r", encoding="utf-8") as f:
        label_maps = json.load(f)
    pools = label_maps.get("task_pools", {})
    if pool_name not in pools:
        raise KeyError(f"未知 RFData pool {pool_name!r}，可选项为：{sorted(pools)}")
    return list(pools[pool_name])


def build_rfdata_pool(rfdata_root: str | Path, pool_name: str, *, use_labels: bool = True, iq_normalize: str = "none", iq_augment: str = "none") -> RFDataPoolDataset:
    root = Path(rfdata_root)
    return RFDataPoolDataset([
        RFDataH5Dataset(root / "h5" / name, use_labels=use_labels, iq_normalize=iq_normalize, iq_augment=iq_augment)
        for name in load_task_pool(root, pool_name)
    ], pool_name=pool_name)


def _collate_metadata(batch: list[dict[str, Any]], out: dict[str, Any]) -> dict[str, Any]:
    numeric_keys = ["length", "dataset_id", "task_type_id", "mod_label_id", "emitter_id", "source_label_id", "global_label_id"]
    numeric_keys.extend(key for key in FLOAT_METADATA_KEYS if key not in numeric_keys)
    for key in numeric_keys:
        if key in batch[0]:
            dtype = torch.float32 if key in FLOAT_METADATA_KEYS else torch.long
            default = float("nan") if dtype == torch.float32 else -1
            out[key] = torch.as_tensor([item.get(key, default) for item in batch], dtype=dtype)
    for key in STRING_METADATA_KEYS:
        if key in batch[0]:
            out[key] = [str(item.get(key, "")) for item in batch]
    if "h5_path" in batch[0]:
        out["h5_path"] = [str(item["h5_path"]) for item in batch]
    return out


def pad_iq_collate(batch: list[dict[str, Any]]) -> dict[str, torch.Tensor | list[str]]:
    max_len = max(int(x["length"]) for x in batch)
    iq = torch.zeros(len(batch), 2, max_len, dtype=torch.float32)
    sample_mask = torch.zeros(len(batch), max_len, dtype=torch.bool)
    out: dict[str, Any] = {"iq": iq, "sample_mask": sample_mask}
    for i, item in enumerate(batch):
        length = int(item["length"])
        iq[i, :, :length] = item["iq"][:, :length]
        sample_mask[i, :length] = True
    return _collate_metadata(batch, out)


def variable_length_collate(batch: list[dict[str, Any]]) -> dict[str, Any]:
    """不 pad I/Q；每条样本保留原生长度，供 sequence packing 使用。"""
    out: dict[str, Any] = {
        "iq": [item["iq"].clone() for item in batch],
        "length": torch.as_tensor([int(item["length"]) for item in batch], dtype=torch.long),
    }
    return _collate_metadata(batch, out)
