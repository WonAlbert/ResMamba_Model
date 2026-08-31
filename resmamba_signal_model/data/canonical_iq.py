"""H5 写入前的统一 I/Q 预处理：float32、可选去 DC / 谱峰居中、定长对齐。

不做 joint_power / abs 幅度归一化；幅度语义留给模型内 RevIN（``iq_normalize: none``）。
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from scipy.signal import resample

CANONICAL_SCALE_POLICY = "canonical_iq_no_amp_norm"

# 预训练池默认定长档位（patch_size=8 对齐）
DEFAULT_DATASET_LENGTHS: dict[str, int] = {
  # short
    "rml2016_04c": 128,
    "rml2016_10a": 128,
    "rml2016_10b": 128,
    "radcom_awgn": 128,
    "radcom_dynamic": 128,
    "radcom_ota": 128,
    # mid
    "radchar": 512,
    # long
    "radar_mod15": 1024,
    "cjr_mix": 1024,
    "xidian14": 1024,
    "rml2018_1a": 1024,
    "wisig": 1024,
    "adsb2": 1024,
    # xlong → 1024
    "panoradio_hf": 1024,
    "communication_emitters": 1024,
    "electromagnetic_0926": 1024,
    "radar_emitters": 1024,
}

DEFAULT_LENGTH_TIERS: tuple[int, ...] = (128, 512, 1024)


def dataset_name_from_h5_path(path: str | Path) -> str:
    stem = Path(path).stem
    for suffix in ("_train", "_val", "_test"):
        if stem.endswith(suffix):
            return stem[: -len(suffix)]
    return stem


def _as_batch_iq(iq: np.ndarray) -> np.ndarray:
    x = np.asarray(iq)
    if x.ndim == 2 and x.shape[0] == 2:
        return x[np.newaxis, ...]
    if x.ndim != 3 or x.shape[1] != 2:
        raise ValueError(f"canonical iq 期望 [B,2,L] 或 [2,L]，当前 {x.shape}")
    return x


def remove_dc_numpy(iq: np.ndarray) -> np.ndarray:
    """``[B,2,L]`` 或 ``[2,L]``：逐样本逐通道去均值。"""
    x = np.asarray(iq, dtype=np.float32)
    if x.ndim == 2:
        return x - x.mean(axis=-1, keepdims=True)
    return x - x.mean(axis=-1, keepdims=True)


def center_spectral_peak_numpy(iq: np.ndarray) -> np.ndarray:
    """``[B,2,L]`` 或 ``[2,L]``：复谱主峰旋至零频（与 ``rfdata.center_spectral_peak`` 一致）。"""
    batch = _as_batch_iq(iq)
    out = np.empty_like(batch, dtype=np.float32)
    for i in range(batch.shape[0]):
        sample = batch[i]
        if sample.shape[-1] < 8:
            out[i] = sample
            continue
        z = sample[0] + 1j * sample[1]
        spectrum = np.fft.fft(z)
        peak = int(np.argmax(np.abs(spectrum)))
        length = int(z.shape[-1])
        centered_peak = peak if peak <= length // 2 else peak - length
        if centered_peak == 0:
            out[i] = sample
            continue
        n = np.arange(length, dtype=np.float64)
        rot = np.exp(-2j * math.pi * float(centered_peak) * n / float(length))
        shifted = z * rot
        out[i] = np.stack([shifted.real, shifted.imag], axis=0).astype(np.float32, copy=False)
    if iq.ndim == 2:
        return out[0]
    return out


def _resize_one(iq2l: np.ndarray, target: int) -> np.ndarray:
    if target <= 0:
        raise ValueError(f"target length 必须 > 0，当前 {target}")
    iq2l = np.asarray(iq2l, dtype=np.float32)
    length = int(iq2l.shape[-1])
    if length == target:
        return iq2l
    if length > target:
        z = iq2l[0] + 1j * iq2l[1]
        z_out = resample(z, target)
        return np.stack([z_out.real, z_out.imag], axis=0).astype(np.float32, copy=False)
    pad_total = target - length
    left = pad_total // 2
    right = pad_total - left
    return np.pad(iq2l, ((0, 0), (left, right)), mode="constant").astype(np.float32, copy=False)


def resize_iq_length(iq: np.ndarray, target: int) -> np.ndarray:
    """定长：长则 FFT 重采样，短则居中零填充。"""
    batch = _as_batch_iq(iq)
    out = np.stack([_resize_one(batch[i], target) for i in range(batch.shape[0])], axis=0)
    if iq.ndim == 2:
        return out[0]
    return out


@dataclass
class CanonicalIQConfig:
    enabled: bool = False
    remove_dc: bool = True
    center_spectral_peak: bool = True
    dataset_lengths: dict[str, int] = field(default_factory=lambda: dict(DEFAULT_DATASET_LENGTHS))
    length_tiers: tuple[int, ...] = DEFAULT_LENGTH_TIERS

    def resolve_length(self, dataset_name: str, native_length: int) -> int:
        name = str(dataset_name).strip()
        if name in self.dataset_lengths:
            return int(self.dataset_lengths[name])
        native = int(native_length)
        tiers = sorted(int(t) for t in self.length_tiers if int(t) > 0)
        if not tiers:
            return native
        for tier in tiers:
            if native <= tier:
                return tier
        return tiers[-1]

    def preprocess(self, iq: np.ndarray) -> np.ndarray:
        """去 DC + 谱峰居中，不定长。"""
        if not self.enabled:
            return np.asarray(iq, dtype=np.float32)
        x = np.asarray(iq, dtype=np.float32)
        if self.remove_dc:
            x = remove_dc_numpy(x)
        if self.center_spectral_peak:
            x = center_spectral_peak_numpy(x)
        return x

    def apply(self, iq: np.ndarray, target_length: int) -> np.ndarray:
        if not self.enabled:
            return np.asarray(iq, dtype=np.float32)
        return resize_iq_length(self.preprocess(iq), int(target_length))

    def attrs(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "remove_dc": self.remove_dc,
            "center_spectral_peak": self.center_spectral_peak,
            "scale_policy": CANONICAL_SCALE_POLICY,
            "dataset_lengths": dict(self.dataset_lengths),
            "length_tiers": list(self.length_tiers),
        }

    def to_json(self) -> str:
        return json.dumps(self.attrs(), sort_keys=True)

    @classmethod
    def from_json(cls, raw: str) -> CanonicalIQConfig:
        payload = json.loads(raw)
        if not isinstance(payload, dict):
            raise TypeError("canonical_iq JSON 必须是 object")
        lengths = payload.get("dataset_lengths") or {}
        tiers = payload.get("length_tiers") or DEFAULT_LENGTH_TIERS
        return CanonicalIQConfig(
            enabled=bool(payload.get("enabled", True)),
            remove_dc=bool(payload.get("remove_dc", True)),
            center_spectral_peak=bool(payload.get("center_spectral_peak", True)),
            dataset_lengths={str(k): int(v) for k, v in dict(lengths).items()},
            length_tiers=tuple(int(x) for x in tiers),
        )

    @classmethod
    def from_cli(
        cls,
        *,
        enabled: bool,
        remove_dc: bool = True,
        center_spectral_peak: bool = True,
    ) -> CanonicalIQConfig:
        if not enabled:
            return CanonicalIQConfig(enabled=False)
        return CanonicalIQConfig(
            enabled=True,
            remove_dc=remove_dc,
            center_spectral_peak=center_spectral_peak,
            dataset_lengths=dict(DEFAULT_DATASET_LENGTHS),
            length_tiers=DEFAULT_LENGTH_TIERS,
        )
