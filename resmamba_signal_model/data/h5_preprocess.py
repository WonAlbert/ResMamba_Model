"""预训练 H5 v2：离线 joint_energy、采样率 / SNR 入库；purge 旧 H5。"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import h5py
import numpy as np

from resmamba_signal_model.data.canonical_iq import dataset_name_from_h5_path
from resmamba_signal_model.data.contracts import SIGNAL_CONTRACT_VERSION
from resmamba_signal_model.models.revin import batch_joint_energy_preprocess

H5_SCHEMA_VERSION = 2
H5_SCALE_POLICY = "joint_energy_h5_v2"

# 预训练 7 库整库常量 fs（radar_mod15 逐样本见 CSV / sample_rates.json）
PRETRAIN_DATASET_FS_HZ: dict[str, float] = {
    "rml2016_04c": 1_000_000.0,
    "rml2016_10a": 1_000_000.0,
    "rml2016_10b": 1_000_000.0,
    "radchar": 3_200_000.0,
    "panoradio_hf": 6_000.0,
    "cjr_mix": 20_000_000.0,
}

PRETRAIN_STEMS: tuple[str, ...] = (
    "rml2016_04c",
    "rml2016_10a",
    "rml2016_10b",
    "radchar",
    "radar_mod15",
    "cjr_mix",
    "panoradio_hf",
)

# 列级拷贝（copy_records v2）：禁止二次 joint_energy
PRETRAIN_V2_COPY_KEYS: tuple[str, ...] = (
    "iq",
    "revin_mean",
    "snr",
    "sample_rate_hz",
    "mod_label_id",
    "source_label_id",
    "canonical_mod_label_id",
    "norm_scale",
    "log_scale",
    "log_peak",
    "papr_preclip",
    "scale_gap",
)

# v1 Writer 列 + v2 预计算列（不含 iq）；供 refine / rebalance 等读回 H5
_LEGACY_H5_META_KEYS: tuple[str, ...] = (
    "length",
    "dataset_id",
    "task_type_id",
    "snr",
    "mod_label_id",
    "canonical_mod_label_id",
    "emitter_id",
    "global_emitter_id",
    "source_label_id",
    "global_label_id",
)
H5_RECORD_META_KEYS: tuple[str, ...] = tuple(
    dict.fromkeys([k for k in PRETRAIN_V2_COPY_KEYS if k != "iq"] + list(_LEGACY_H5_META_KEYS))
)


@dataclass
class PretrainH5Config:
    precompute_joint_energy: bool = True
    device: str = "cpu"
    chunk_size: int = 256
    std_min: float = 0.01
    winsorize_top_frac: float = 0.01
    peak_papr_clip: float = 16.0
    revin_clip: float = 8.0

    def to_json(self) -> str:
        return json.dumps(asdict(self), sort_keys=True)

    @classmethod
    def from_json(cls, raw: str) -> PretrainH5Config:
        payload = json.loads(raw)
        if not isinstance(payload, dict):
            raise TypeError("PretrainH5Config JSON 必须是 object")
        return cls(
            precompute_joint_energy=bool(payload.get("precompute_joint_energy", True)),
            device=str(payload.get("device", "cpu")),
            chunk_size=int(payload.get("chunk_size", 256)),
            std_min=float(payload.get("std_min", 0.01)),
            winsorize_top_frac=float(payload.get("winsorize_top_frac", 0.01)),
            peak_papr_clip=float(payload.get("peak_papr_clip", 16.0)),
            revin_clip=float(payload.get("revin_clip", 8.0)),
        )


def resolve_pretrain_fs_hz(dataset_name: str, per_sample: np.ndarray | None = None) -> np.ndarray | float:
    name = str(dataset_name).strip()
    if per_sample is not None:
        return np.asarray(per_sample, dtype=np.float32)
    if name in PRETRAIN_DATASET_FS_HZ:
        return float(PRETRAIN_DATASET_FS_HZ[name])
    return float("nan")


def _fs_vector(n: int, dataset_name: str, values: np.ndarray | None = None) -> np.ndarray:
    if values is not None:
        return np.asarray(values, dtype=np.float32).reshape(-1)[:n]
    fs = resolve_pretrain_fs_hz(dataset_name)
    if not np.isfinite(fs):
        return np.full(n, np.nan, dtype=np.float32)
    return np.full(n, float(fs), dtype=np.float32)


def infer_radar_mod15_csv_fs(csv_path: Path) -> float:
    """从 round7 CSV 的 ``time(s)`` 列推断采样率（Hz）。"""
    import csv

    times: list[float] = []
    with csv_path.open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            times.append(float(row["time(s)"]))
    t = np.asarray(times, dtype=np.float64)
    if t.size < 2:
        return float("nan")
    return float(1.0 / np.median(np.diff(t)))


def purge_h5_for_stems(
    h5_dir: Path,
    stems: Iterable[str],
    *,
    dry_run: bool = False,
) -> list[str]:
    """删除待重建数据集的 H5（**不**动源数据）。返回已删文件名。"""
    h5_dir = Path(h5_dir)
    removed: list[str] = []
    for stem in stems:
        for path in sorted(h5_dir.glob(f"{stem}_*.h5")):
            removed.append(path.name)
            if not dry_run:
                path.unlink(missing_ok=True)
    return removed


class PretrainH5Writer:
    """紧凑预训练 H5 v2：iq 可预 joint_energy；snr/fs/RevIN 统计入库。"""

    FIELDS = {
        "snr": ("f2", np.nan),
        "sample_rate_hz": ("f4", np.nan),
        "mod_label_id": ("i2", -1),
        "source_label_id": ("i2", -1),
        "canonical_mod_label_id": ("i2", -1),
        "norm_scale": ("f2", np.nan),
        "log_scale": ("f2", np.nan),
        "log_peak": ("f2", np.nan),
        "papr_preclip": ("f2", np.nan),
        "scale_gap": ("f2", np.nan),
    }
    MEAN_DATASET = "revin_mean"  # [N,2] f2

    def __init__(
        self,
        path: Path,
        length: int,
        dataset_id: int,
        task_id: int,
        source: Path,
        *,
        dataset_name: str,
        cfg: PretrainH5Config | None = None,
    ) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self.length = int(length)
        self.dataset_id = int(dataset_id)
        self.task_id = int(task_id)
        self.dataset_name = str(dataset_name)
        self.cfg = cfg or PretrainH5Config()
        self.count = 0

        self.f = h5py.File(path, "w")
        chunk = max(1, min(256, (8 << 20) // max(1, 2 * self.length * 4)))
        self.iq = self.f.create_dataset(
            "iq",
            (0, 2, self.length),
            maxshape=(None, 2, self.length),
            dtype=np.float32,
            chunks=(chunk, 2, self.length),
            compression="lzf",
        )
        self.ds = {
            key: self.f.create_dataset(
                key,
                (0,),
                maxshape=(None,),
                dtype=spec[0],
                chunks=(max(1024, chunk),),
                fillvalue=spec[1],
            )
            for key, spec in self.FIELDS.items()
        }
        self.revin_mean = self.f.create_dataset(
            self.MEAN_DATASET,
            (0, 2),
            maxshape=(None, 2),
            dtype="f2",
            chunks=(max(1024, chunk), 2),
            fillvalue=0.0,
        )
        self.f.attrs.update(
            {
                "h5_schema_version": H5_SCHEMA_VERSION,
                "iq_preprocessed": "joint_energy" if self.cfg.precompute_joint_energy else "none",
                "scale_policy": H5_SCALE_POLICY if self.cfg.precompute_joint_energy else "none",
                "source_path": str(source),
                "signal_length": self.length,
                "dataset_id": self.dataset_id,
                "task_type_id": self.task_id,
                "dataset_name": self.dataset_name,
                "channel_axis": 1,
                "signal_contract_version": SIGNAL_CONTRACT_VERSION,
            }
        )
        fs_const = PRETRAIN_DATASET_FS_HZ.get(self.dataset_name)
        if fs_const is not None:
            self.f.attrs["sampling_rate_hz"] = float(fs_const)

    def _write_block(self, iq: np.ndarray, meta: dict[str, Any]) -> None:
        iq = np.asarray(iq, dtype=np.float32)
        if iq.ndim != 3 or iq.shape[1] != 2:
            raise ValueError(f"PretrainH5Writer 期望 [B,2,L]，实际 {iq.shape}")
        n = int(iq.shape[0])
        if n <= 0:
            return

        pre_meta: dict[str, np.ndarray] = {}
        if self.cfg.precompute_joint_energy:
            iq, pre_meta = batch_joint_energy_preprocess(
                iq,
                device=self.cfg.device,
                std_min=self.cfg.std_min,
                winsorize_top_frac=self.cfg.winsorize_top_frac,
                peak_papr_clip=self.cfg.peak_papr_clip,
                clip=self.cfg.revin_clip,
                chunk_size=self.cfg.chunk_size,
            )

        start, end = self.count, self.count + n
        self.iq.resize(end, axis=0)
        self.iq[start:end] = iq

        combined: dict[str, np.ndarray] = dict(pre_meta)
        for key, val in meta.items():
            combined[key] = np.asarray(val).reshape(-1)[:n]
        if "sample_rate_hz" not in combined:
            combined["sample_rate_hz"] = _fs_vector(n, self.dataset_name)

        for key, ds in self.ds.items():
            ds.resize(end, axis=0)
            if key in combined:
                ds[start:end] = np.asarray(combined[key], dtype=ds.dtype)

        if self.cfg.precompute_joint_energy:
            self.revin_mean.resize(end, axis=0)
            self.revin_mean[start:end] = pre_meta["revin_mean"].astype(np.float16, copy=False)

        self.count = end

    def append_raw(self, iq: np.ndarray, **meta: Any) -> None:
        self._write_block(iq, meta)

    def append_preprocessed(self, iq: np.ndarray, **meta: Any) -> None:
        """列级拷贝：``iq`` 已 joint_energy，跳过离线预计算。"""
        iq = np.asarray(iq, dtype=np.float32)
        if iq.ndim != 3 or iq.shape[1] != 2:
            raise ValueError(f"PretrainH5Writer 期望 [B,2,L]，实际 {iq.shape}")
        n = int(iq.shape[0])
        if n <= 0:
            return
        if "revin_mean" not in meta:
            raise ValueError("append_preprocessed 需要 meta['revin_mean']")
        start, end = self.count, self.count + n
        self.iq.resize(end, axis=0)
        self.iq[start:end] = iq
        combined: dict[str, np.ndarray] = {}
        for key, val in meta.items():
            arr = np.asarray(val)
            if key == self.MEAN_DATASET:
                combined[key] = arr[:n]
            else:
                combined[key] = arr.reshape(-1)[:n]
        if "sample_rate_hz" not in combined:
            combined["sample_rate_hz"] = _fs_vector(n, self.dataset_name)
        for key, ds in self.ds.items():
            ds.resize(end, axis=0)
            if key in combined:
                ds[start:end] = np.asarray(combined[key], dtype=ds.dtype)
        self.revin_mean.resize(end, axis=0)
        self.revin_mean[start:end] = np.asarray(combined["revin_mean"], dtype=np.float16)
        self.count = end

    def append_clean(self, iq: np.ndarray, removed, **meta: Any) -> None:
        del removed
        self._write_block(iq, meta)

    def close(self) -> None:
        self.f.attrs["sample_count"] = self.count
        self.f.close()
