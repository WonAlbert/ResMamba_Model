#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from resmamba_signal_model.thread_env import normalize_thread_env

normalize_thread_env()

import argparse
import ast
from collections import Counter, defaultdict
import csv
import json
import os
import pickle
from typing import Any
from concurrent.futures import ProcessPoolExecutor, as_completed

import h5py
import numpy as np
import yaml
from scipy.signal import hilbert

from resmamba_signal_model.data.h5_preprocess import (
    PRETRAIN_STEMS,
    H5_SCHEMA_VERSION,
    H5_RECORD_META_KEYS,
    PRETRAIN_V2_COPY_KEYS,
    PretrainH5Config,
    PretrainH5Writer,
    purge_h5_for_stems,
    infer_radar_mod15_csv_fs,
    resolve_pretrain_fs_hz,
)
from resmamba_signal_model.data.canonical_iq import (
    CANONICAL_SCALE_POLICY,
    CanonicalIQConfig,
    dataset_name_from_h5_path,
    resize_iq_length,
)
from resmamba_signal_model.data.quality import quality_mask, stratified_split_indices
from resmamba_signal_model.data.contracts import (
    CAPTURE_METADATA_KEYS,
    MISSING_METADATA,
    SIGNAL_CONTRACT_VERSION,
)
from resmamba_signal_model.data.labels import (
    build_emitter_namespace,
    build_modulation_ontology,
)
from resmamba_signal_model.data.splits import (
    ImmutableManifestError,
    assert_manifest_files_unchanged,
    build_split_manifest,
    load_manifest,
    write_immutable_manifest,
)
from resmamba_signal_model.training.clustering_labels import GLOBAL_LABEL_NAMESPACE, global_cluster_labels
from resmamba_signal_model.training.emitter_labels import h5_dataset_name
from resmamba_signal_model.data.wisig_manytx import (
    WiSigBlockSink,
    emitter_labels,
    stream_manytx_blocks,
)
from resmamba_signal_model.training.pool_filters import (
    filter_dataset_pool,
    filter_excluded_dataset_pool,
    load_clustering_comm_datasets,
    load_clustering_radar_datasets,
    load_downstream_comm_modulation_datasets,
    load_downstream_modulation_extra_datasets,
    load_downstream_radar_modulation_datasets,
    load_downstream_radar_model_datasets,
    load_prediction_datasets,
    load_excluded_datasets,
    load_pretrain_datasets,
)


ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = ROOT.parent
NON_EMITTER = WORKSPACE / "辐射源（非个体）识别数据"
EMITTER = WORKSPACE / "个体辐射源数据"
EXTERNAL = ROOT / "dataset" / "external"
PANORADIO_MODES = [
    "morse", "psk31", "psk63", "qpsk31", "rtty45_170", "rtty50_170", "rtty100_850",
    "olivia8_250", "olivia16_500", "olivia16_1000", "olivia32_1000", "dominoex11",
    "mt63_1000", "navtex", "usb", "lsb", "am", "fax",
]
# 全库统一：有限 snr 且 snr >= UNIFIED_MIN_SNR_DB 才入库（radar_mod15 名义 10 dB 自然通过）
UNIFIED_MIN_SNR_DB = 8.0
RADCOM_VARIANTS = (
    ("radcom_dynamic", 13, "RadComDynamic.hdf5", UNIFIED_MIN_SNR_DB),
    ("radcom_awgn", 14, "RadComAWGN.hdf5", UNIFIED_MIN_SNR_DB),
    ("radcom_ota", 15, "RadComOta2.45GHz.hdf5", UNIFIED_MIN_SNR_DB),
)
RML2016_ORDER = ["8PSK", "AM-DSB", "AM-SSB", "BPSK", "CPFSK", "GFSK", "PAM4", "QAM16", "QAM64", "QPSK", "WBFM"]
RML2018_ORDER = [
    "32PSK", "16APSK", "32QAM", "FM", "GMSK", "32APSK", "OQPSK", "8ASK",
    "BPSK", "8PSK", "AM-SSB-SC", "4ASK", "16PSK", "64APSK", "128QAM",
    "128APSK", "AM-DSB-SC", "AM-SSB-WC", "64QAM", "QPSK", "256QAM",
    "AM-DSB-WC", "OOK", "16QAM",
]
EVAL_QUALITY_DATASETS = frozenset({"rml2018_1a", "adsb2", "wifi150", "xidian14"})
EVAL_QUALITY_SPLITS = ("val", "test")
# 仅 val/test、不做类别均衡的数据集（不参与 MAE 预训练时的旧约定；radar_mod15 已改为 train/val）
VAL_TEST_ONLY_DATASETS = frozenset()
VAL_TEST_SPLIT_RATIO = 0.8
RADCHAR_SIGNAL_TYPE_NAMES = (
    "coherent_pulse_train",
    "barker_code",
    "polyphase_barker_code",
    "frank_code",
    "linear_frequency_modulated",
)
RADCHAR_SAMPLE_RATE_HZ = 3_200_000.0
RADAR_MOD15_NOMINAL_SNR_DB = 10.0
RADCHAR_DEFAULT_SOURCE = WORKSPACE / "RadChar-Small.h5"
RML2018_EVAL_MIN_SNR = UNIFIED_MIN_SNR_DB
MISSING_TEST_SPLIT_SEED = 20260822
_SNR_LT_MIN_REASON = f"snr_lt_{int(UNIFIED_MIN_SNR_DB)}"


def snr_keep_mask(
    snr: np.ndarray | float,
    *,
    min_db: float = UNIFIED_MIN_SNR_DB,
) -> np.ndarray:
    """有限且 ``snr >= min_db`` 为 True；标量输入返回 shape=(1,) 的 bool 数组。"""
    arr = np.asarray(snr, dtype=np.float64)
    scalar = arr.ndim == 0
    if scalar:
        arr = arr.reshape(1)
    out = np.isfinite(arr) & (arr >= float(min_db))
    return out
MISSING_TEST_FRACTION = 0.2
_MISSING_TEST_LABEL_FIELDS = (
    "emitter_id",
    "mod_label_id",
    "source_label_id",
    "canonical_mod_label_id",
    "global_label_id",
)

_CANONICAL_CFG: CanonicalIQConfig | None = None
_CANONICAL_ENV = "RFDATA_CANONICAL_IQ"
_PRETRAIN_H5_CFG: PretrainH5Config | None = None
_PRETRAIN_H5_ENV = "RFDATA_PRETRAIN_H5_CFG"
PRETRAIN_STRATIFIED_RATIOS = (0.7, 0.2, 0.1)
PRETRAIN_STRATIFIED_SEED = 20260629


def _make_writer(path: Path, length: int, dtype, dataset_id: int, task_id: int, source: Path):
    name = dataset_name_from_h5_path(path)
    if _PRETRAIN_H5_CFG is not None:
        return PretrainWriter(
            path,
            int(length),
            int(dataset_id),
            int(task_id),
            source,
            dataset_name=name,
            cfg=_PRETRAIN_H5_CFG,
        )
    return Writer(path, length, dtype, dataset_id, task_id, source)


def _fs_meta_array(n: int, dataset_name: str, values: np.ndarray | None = None) -> np.ndarray:
    if values is not None:
        return np.asarray(values, dtype=np.float32).reshape(-1)[:n]
    fs = resolve_pretrain_fs_hz(dataset_name)
    if not np.isfinite(fs):
        return np.full(n, np.nan, dtype=np.float32)
    return np.full(n, float(fs), dtype=np.float32)


def _active_canonical_config() -> CanonicalIQConfig | None:
    return _CANONICAL_CFG


def _apply_canonical_preprocess(iq: np.ndarray) -> np.ndarray:
    cfg = _active_canonical_config()
    if cfg is None or not cfg.enabled:
        return np.asarray(iq, dtype=np.float32)
    return cfg.preprocess(iq)


def _apply_canonical_resize(iq: np.ndarray, target_length: int) -> np.ndarray:
    cfg = _active_canonical_config()
    if cfg is None or not cfg.enabled:
        return np.asarray(iq, dtype=np.float32)
    return resize_iq_length(iq, int(target_length))


def _apply_canonical_iq(iq: np.ndarray, target_length: int) -> np.ndarray:
    return _apply_canonical_resize(_apply_canonical_preprocess(iq), target_length)


def _preprocess_canonical_iq(iq: np.ndarray) -> np.ndarray:
    cfg = _active_canonical_config()
    if cfg is None or not cfg.enabled:
        return np.asarray(iq, dtype=np.float32)
    return cfg.preprocess(iq)


def _resize_canonical_iq(iq: np.ndarray, target_length: int) -> np.ndarray:
    cfg = _active_canonical_config()
    if cfg is None or not cfg.enabled:
        return np.asarray(iq, dtype=np.float32)
    return resize_iq_length(iq, int(target_length))


def _resolve_target_length(dataset_name: str, native_length: int) -> int:
    cfg = _active_canonical_config()
    if cfg is not None and cfg.enabled:
        return int(cfg.resolve_length(dataset_name, int(native_length)))
    return int(native_length)


def _iq_at_quality_stage(iq: np.ndarray, target_length: int) -> np.ndarray:
    """去 DC + 谱峰居中后；长序列先 FFT resample，短序列保持原长（pad 前检测）。"""
    iq = _preprocess_canonical_iq(iq)
    if int(iq.shape[-1]) > int(target_length):
        return _resize_canonical_iq(iq, target_length)
    return iq


def _quality_label_field(meta: dict) -> np.ndarray | None:
    for field in ("emitter_id", "mod_label_id", "source_label_id", "global_label_id"):
        if field in meta:
            return np.asarray(meta[field]).reshape(-1)
    return None


def stratified_split_indices(
    labels: np.ndarray,
    ratios: tuple[float, float, float] = PRETRAIN_STRATIFIED_RATIOS,
    seed: int = PRETRAIN_STRATIFIED_SEED,
) -> dict[str, np.ndarray]:
    """按类分层 7:2:1（默认），不做最小类下采样。"""
    labels = np.asarray(labels).reshape(-1)
    rng = np.random.default_rng(int(seed))
    buckets: dict[str, list[np.ndarray]] = {"train": [], "test": [], "val": []}
    for cls in np.unique(labels):
        if int(cls) < 0:
            continue
        idx = np.flatnonzero(labels == cls)
        rng.shuffle(idx)
        m = int(idx.size)
        if m <= 0:
            continue
        n_train = int(round(float(ratios[0]) * m))
        n_test = int(round(float(ratios[1]) * m))
        n_val = max(0, m - n_train - n_test)
        if m >= 3:
            if n_train <= 0:
                n_train = 1
            if n_test <= 0 and m - n_train > 1:
                n_test = 1
            if n_val <= 0 and m - n_train - n_test > 0:
                n_val = 1
            total = n_train + n_test + n_val
            if total > m:
                while total > m and n_train > 1:
                    n_train -= 1
                    total -= 1
                while total > m and n_test > 1:
                    n_test -= 1
                    total -= 1
                while total > m and n_val > 1:
                    n_val -= 1
                    total -= 1
            elif total < m:
                n_train += m - total
        buckets["train"].append(idx[:n_train])
        buckets["test"].append(idx[n_train:n_train + n_test])
        buckets["val"].append(idx[n_train + n_test:n_train + n_test + n_val])
    return {
        key: np.concatenate(parts).astype(np.int64, copy=False) if parts else np.array([], dtype=np.int64)
        for key, parts in buckets.items()
    }


def _h5_task_id(f: h5py.File) -> int:
    if "task_type_id" in f and f["task_type_id"].shape[0] > 0:
        return int(f["task_type_id"][0])
    return int(f.attrs.get("task_type_id", 0))


def _h5_dataset_id(f: h5py.File) -> int:
    if "dataset_id" in f and f["dataset_id"].shape[0] > 0:
        return int(f["dataset_id"][0])
    return int(f.attrs.get("dataset_id", 0))


def _is_h5_v2(f: h5py.File) -> bool:
    return int(f.attrs.get("h5_schema_version", 0)) == H5_SCHEMA_VERSION


def write_pretrain_stratified_splits(
    ctx: Context,
    name: str,
    dataset_id: int,
    source: Path,
    length: int,
    dtype,
    iq_all: np.ndarray,
    labels_all: np.ndarray,
    labels_map: dict,
    removed: Counter,
    raw: int,
    *,
    label_field: str = "mod_label_id",
    extra_meta: dict[str, np.ndarray] | None = None,
    split_seed: int = PRETRAIN_STRATIFIED_SEED,
    writer_attrs: dict[str, Any] | None = None,
    snr_removed: dict[str, int] | None = None,
) -> None:
    """quality 过滤 + 定长后的样本，分层写 train/test/val。"""
    meta = dict(extra_meta or {})
    meta[label_field] = np.asarray(labels_all).reshape(-1)
    split_idx = stratified_split_indices(
        meta[label_field],
        ratios=PRETRAIN_STRATIFIED_RATIOS,
        seed=split_seed + (sum(name.encode()) & 0xFFFF),
    )
    entry = ctx.report["datasets"].setdefault(name, {"labels": labels_map, "splits": {}})
    entry["labels"] = labels_map
    entry.pop("balanced_split", None)
    entry.pop("val_test_split", None)
    split_removed = dict(removed)
    if snr_removed:
        split_removed.update(snr_removed)
    for split_name in ("train", "test", "val"):
        idx = split_idx[split_name]
        out_path = ctx.h5 / f"{name}_{split_name}.h5"
        writer = _make_writer(out_path, length, dtype, dataset_id, 0, source)
        if writer_attrs and hasattr(writer, "f"):
            writer.f.attrs.update(writer_attrs)
        if idx.size:
            chunk_meta = {key: np.asarray(val)[idx] for key, val in meta.items()}
            if isinstance(writer, PretrainWriter):
                writer.append_prepared(iq_all[idx], **chunk_meta)
            else:
                writer.append_raw(iq_all[idx], **chunk_meta)
        writer.close()
        ctx.pool[split_name].append((out_path.name, writer.task_id))
        entry["splits"][split_name] = {
            "raw": int(raw),
            "kept": int(idx.size),
            "removed": dict(split_removed),
            "file": out_path.name,
        }
    entry["stratified_split"] = {
        "label_field": label_field,
        "ratios": list(PRETRAIN_STRATIFIED_RATIOS),
        "seed": int(split_seed),
        "no_class_balance": True,
        "total_kept": int(len(labels_all)),
        "splits": {
            split: {"kept": entry["splits"][split]["kept"], "file": entry["splits"][split]["file"]}
            for split in ("train", "test", "val")
        },
    }
    ctx.touched.add(name)


def aggregate_quality_filter_report(ctx: Context) -> dict[str, Any]:
    by_reason: Counter = Counter()
    by_dataset: dict[str, dict[str, Any]] = {}
    total_raw = total_kept = total_removed = 0
    for name in PRETRAIN_STEMS:
        entry = ctx.report.get("datasets", {}).get(name)
        if not isinstance(entry, dict):
            continue
        ds_raw = ds_kept = ds_removed = 0
        ds_reasons: Counter = Counter()
        for split in ("train", "test", "val"):
            split_info = entry.get("splits", {}).get(split)
            if not isinstance(split_info, dict):
                continue
            raw = int(split_info.get("raw", 0))
            kept = int(split_info.get("kept", 0))
            ds_raw = max(ds_raw, raw)
            ds_kept += kept
            removed_info = split_info.get("removed", {})
            if isinstance(removed_info, dict) and split == "train":
                for reason, count in removed_info.items():
                    if str(reason).startswith("snr_"):
                        continue
                    c = int(count)
                    ds_reasons[reason] = c
                    by_reason[reason] += c
        ds_removed = max(0, ds_raw - ds_kept) if ds_raw else sum(ds_reasons.values())
        total_raw += ds_raw
        total_kept += ds_kept
        total_removed += ds_removed
        by_dataset[name] = {
            "raw": ds_raw,
            "kept": ds_kept,
            "removed": ds_removed,
            "by_reason": dict(sorted(ds_reasons.items())),
        }
    return {
        "total_raw": int(total_raw),
        "total_kept": int(total_kept),
        "total_removed": int(total_removed),
        "by_reason": dict(sorted(by_reason.items())),
        "by_dataset": by_dataset,
    }


def print_quality_filter_summary(report: dict[str, Any]) -> None:
    reasons = sorted(
        set(report.get("by_reason", {}))
        | {
            reason
            for info in report.get("by_dataset", {}).values()
            for reason in info.get("by_reason", {})
        }
    )
    if not reasons and not report.get("by_dataset"):
        return
    header = ["dataset", "raw", "kept", "removed", *reasons]
    rows: list[list[str]] = []
    for name, info in sorted(report.get("by_dataset", {}).items()):
        row = [
            name,
            str(info.get("raw", 0)),
            str(info.get("kept", 0)),
            str(info.get("removed", 0)),
        ]
        by_reason = info.get("by_reason", {})
        row.extend(str(by_reason.get(reason, 0)) for reason in reasons)
        rows.append(row)
    totals = report.get("by_reason", {})
    rows.append(
        ["TOTAL", str(report.get("total_raw", 0)), str(report.get("total_kept", 0)), str(report.get("total_removed", 0))]
        + [str(totals.get(reason, 0)) for reason in reasons]
    )
    widths = [max(len(header[i]), *(len(row[i]) for row in rows)) for i in range(len(header))]
    line = " | ".join(h.ljust(widths[i]) for i, h in enumerate(header))
    print("[quality-filter]", flush=True)
    print(line, flush=True)
    print("-+-".join("-" * w for w in widths), flush=True)
    for row in rows:
        print(" | ".join(row[i].ljust(widths[i]) for i in range(len(header))), flush=True)


def persist_quality_filter_report(ctx: Context) -> dict[str, Any]:
    payload = aggregate_quality_filter_report(ctx)
    if not payload.get("by_dataset"):
        return payload
    ctx.report["quality_filter"] = payload
    report_path = ctx.output / "quality_filter_report.json"
    report_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print_quality_filter_summary(payload)
    return payload


def as_iq(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x)
    if np.iscomplexobj(x):
        return np.stack((x.real, x.imag), axis=-2)
    if x.ndim == 1:
        z = hilbert(x, axis=-1)
        return np.stack((z.real, z.imag), axis=0)
    if x.shape[-2] == 2:
        return x
    if x.shape[-1] == 2:
        return np.swapaxes(x, -1, -2)
    z = hilbert(x, axis=-1)
    return np.stack((z.real, z.imag), axis=-2)


class Writer:
    FIELDS = {
        "length": ("i4", 0), "dataset_id": ("i4", 0), "task_type_id": ("i1", 0),
        "snr": ("f4", np.nan), "mod_label_id": ("i4", -1), "canonical_mod_label_id": ("i4", -1),
        "emitter_id": ("i4", -1), "global_emitter_id": ("i4", -1),
        "source_label_id": ("i4", -1), "global_label_id": ("i4", -1),
    }
    STRING_FIELDS = {
        **{key: MISSING_METADATA for key in CAPTURE_METADATA_KEYS},
        "capture_date": MISSING_METADATA,
    }
    ALL_FIELDS = tuple(FIELDS) + tuple(STRING_FIELDS)

    def __init__(self, path: Path, length: int, dtype, dataset_id: int, task_id: int, source: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        cfg = _active_canonical_config()
        self._dataset_name = dataset_name_from_h5_path(path)
        if cfg is not None and cfg.enabled:
            length = cfg.resolve_length(self._dataset_name, int(length))
        self.path, self.length, self.dataset_id, self.task_id = path, length, dataset_id, task_id
        self.f = h5py.File(path, "w")
        chunk = max(1, min(256, (8 << 20) // max(1, 2 * length * np.dtype(dtype).itemsize)))
        self.iq = self.f.create_dataset(
            "iq", (0, 2, length), maxshape=(None, 2, length), dtype=dtype,
            chunks=(chunk, 2, length), compression="lzf",
        )
        self.ds = {
            key: self.f.create_dataset(key, (0,), maxshape=(None,), dtype=spec[0], chunks=(max(1024, chunk),), fillvalue=spec[1])
            for key, spec in self.FIELDS.items()
        }
        string_dtype = h5py.string_dtype(encoding="utf-8")
        self.string_ds = {
            key: self.f.create_dataset(
                key,
                (0,),
                maxshape=(None,),
                dtype=string_dtype,
                chunks=(max(1024, chunk),),
                fillvalue=default,
            )
            for key, default in self.STRING_FIELDS.items()
        }
        scale_policy = "none"
        attrs: dict[str, Any] = {
            "source_path": str(source),
            "scale_policy": scale_policy,
            "quality_filter": "finite, non-silent, robust-power-8MAD, PAPR<=100, lag1-correlation>=1e-3",
            "channel_axis": 1,
            "signal_contract_version": SIGNAL_CONTRACT_VERSION,
            "missing_metadata_marker": MISSING_METADATA,
        }
        if cfg is not None and cfg.enabled:
            attrs["scale_policy"] = CANONICAL_SCALE_POLICY
            attrs["canonical_iq"] = json.dumps(cfg.attrs(), sort_keys=True)
        self.f.attrs.update(attrs)
        self.count = 0
        self._known_group_metadata = Counter()

    @staticmethod
    def record_load_keys() -> tuple[str, ...]:
        return tuple(dict.fromkeys(list(H5_RECORD_META_KEYS) + list(Writer.ALL_FIELDS)))

    @staticmethod
    def _select_meta(value, keep: np.ndarray | None):
        value = np.asarray(value)
        if keep is not None and value.ndim and value.shape[0] == keep.shape[0]:
            return value[keep]
        return value

    @staticmethod
    def _string_meta(value):
        array = np.asarray(value)
        if array.ndim == 0:
            item = array.item()
            if isinstance(item, bytes):
                item = item.decode("utf-8", errors="replace")
            return str(item)
        return np.asarray([
            item.decode("utf-8", errors="replace") if isinstance(item, bytes) else str(item)
            for item in array.reshape(-1)
        ], dtype=object).reshape(array.shape)

    def _append_metadata(self, start: int, end: int, keep: np.ndarray | None, meta: dict, defaults: dict) -> None:
        for key, ds in self.ds.items():
            ds.resize(end, axis=0)
            value = meta.get(key, defaults.get(key, ds.fillvalue))
            ds[start:end] = self._select_meta(value, keep)
        for key, ds in self.string_ds.items():
            ds.resize(end, axis=0)
            value = meta.get(key, self.STRING_FIELDS[key])
            selected = self._string_meta(self._select_meta(value, keep))
            ds[start:end] = selected
            if key in CAPTURE_METADATA_KEYS:
                if np.asarray(selected).ndim == 0:
                    known = str(np.asarray(selected).item()) not in ("", MISSING_METADATA)
                    self._known_group_metadata[key] += (end - start) if known else 0
                else:
                    self._known_group_metadata[key] += sum(
                        str(item) not in ("", MISSING_METADATA)
                        for item in np.asarray(selected).reshape(-1)
                    )

    def append_clean(self, iq: np.ndarray, removed: Counter, **meta) -> None:
        iq = as_iq(iq)
        per_class_labels = _quality_label_field(meta)
        iq_prep = _preprocess_canonical_iq(iq)
        iq_check = _iq_at_quality_stage(iq, self.length)
        keep, reasons = quality_mask(
            iq_check,
            labels=per_class_labels,
            dataset_name=self._dataset_name,
        )
        removed.update(reasons)
        iq = _resize_canonical_iq(iq_prep[keep], self.length)
        if not len(iq):
            return
        filtered_meta = {key: Writer._select_meta(value, keep) for key, value in meta.items()}
        start, end = self.count, self.count + len(iq)
        self.iq.resize(end, axis=0)
        self.iq[start:end] = iq
        defaults = {"length": self.length, "dataset_id": self.dataset_id, "task_type_id": self.task_id}
        self._append_metadata(start, end, keep, filtered_meta, defaults)
        self.count = end

    def append_raw(self, iq: np.ndarray, **meta) -> None:
        if not len(iq):
            return
        iq = _apply_canonical_iq(as_iq(iq), self.length)
        start, end = self.count, self.count + len(iq)
        self.iq.resize(end, axis=0)
        self.iq[start:end] = iq
        defaults = {"length": self.length, "dataset_id": self.dataset_id, "task_type_id": self.task_id}
        self._append_metadata(start, end, None, meta, defaults)
        self.count = end

    def close(self):
        self.f.attrs["sample_count"] = self.count
        self.f.attrs["group_metadata_known_counts"] = json.dumps(
            {key: int(self._known_group_metadata[key]) for key in CAPTURE_METADATA_KEYS},
            sort_keys=True,
        )
        self.f.close()


class PretrainWriter(PretrainH5Writer):
    """带 quality_mask / canonical_iq 的预训练 H5 v2 Writer。"""

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
        canon = _active_canonical_config()
        if canon is not None and canon.enabled:
            length = canon.resolve_length(dataset_name, int(length))
        super().__init__(
            path,
            length,
            dataset_id,
            task_id,
            source,
            dataset_name=dataset_name,
            cfg=cfg,
        )

    def append_clean(self, iq: np.ndarray, removed: Counter, **meta) -> None:
        iq = as_iq(iq)
        per_class_labels = _quality_label_field(meta)
        iq_prep = _preprocess_canonical_iq(iq)
        iq_check = _iq_at_quality_stage(iq, self.length)
        keep, reasons = quality_mask(
            iq_check,
            labels=per_class_labels,
            dataset_name=self.dataset_name,
        )
        removed.update(reasons)
        iq = _resize_canonical_iq(iq_prep[keep], self.length)
        if not len(iq):
            return
        filtered = {key: Writer._select_meta(value, keep) for key, value in meta.items()}
        self._write_block(iq, filtered)

    def append_prepared(self, iq: np.ndarray, **meta) -> None:
        if not len(iq):
            return
        self._write_block(as_iq(iq), meta)

    def append_raw(self, iq: np.ndarray, **meta) -> None:
        if not len(iq):
            return
        iq = _apply_canonical_iq(as_iq(iq), self.length)
        self._write_block(iq, meta)


class Context:
    def __init__(self, output: Path):
        self.output, self.h5 = output, output / "h5"
        self.maps = self._load_json(output / "label_maps.json", {
            "datasets": {}, "modulations": {}, "emitters": {}, "sources": {},
            "receivers": {}, "sessions": {}, "task_pools": {},
        })
        for key in ("datasets", "modulations", "emitters", "sources", "receivers", "sessions", "task_pools"):
            self.maps.setdefault(key, {})
        self.report = self._load_json(output / "cleaning_report.json", None)
        if self.report is None:
            self.report = {
                "policy": {
                    "normalization": "none",
                    "rml_snr": f"SNR >= {UNIFIED_MIN_SNR_DB:g} dB (unified)",
                    "real_signal_iq": "Hilbert analytic signal; real part is original signal and imaginary part is Hilbert transform",
                    "extreme_filter": "finite/non-silent, per-class robust power 4 MAD, PAPR <= 30",
                    "npy_label_maps": "numeric Y_*.npy labels are inferred from data; string labels use configured name maps",
                    "unlabelled_snr": "stricter lag-1 correlation proxy; SNR remains NaN",
                    "unified_min_snr_db": UNIFIED_MIN_SNR_DB,
                    "snr_policy": (
                        f"all datasets with finite snr: keep snr >= {UNIFIED_MIN_SNR_DB:g} dB; "
                        f"radar_mod15 nominal {RADAR_MOD15_NOMINAL_SNR_DB:g} dB"
                    ),
                    "split": (
                        "preserve verified capture groups when available; otherwise mark group metadata "
                        "unavailable and use the dataset's documented sample-level split"
                    ),
                    "group_metadata": (
                        f"receiver_id/session_id/channel_id/capture_id are always present; "
                        f"{MISSING_METADATA!r} means unavailable from source"
                    ),
                    "group_split_claims": (
                        "Only datasets whose immutable split manifest says verified_group_held_out "
                        "may be reported as receiver/session/capture held out"
                    ),
                    "split_usage": "*_train.h5: MAE pretrain only; *_val.h5: all validation and model selection; *_test.h5: downstream/finetune training only (not for evaluation)",
                    "eval_quality_datasets": sorted(EVAL_QUALITY_DATASETS),
                    "eval_quality_filter": "val/test only: per-class power 3 MAD, PAPR <= 20, lag1-correlation >= 0.05",
                    "rml2018_eval_snr": f"val/test only: SNR >= {RML2018_EVAL_MIN_SNR:g} dB",
                },
                "datasets": {},
                "excluded": {
                    "ADSB-1 and ADSB-3": "SHA256 confirms duplicate copies; ADSB-2 is used to avoid leakage",
                    "WIFIDATASET/62ft.rar": "RAR archive only; no RAR extractor is installed",
                    "wisig/ManyTx.pkl.zip": "superseded when ManyTx.pkl is present; use --datasets wisig",
                    "XSRPdatav1": "capture dates are present but emitter identities are absent",
                },
            }
        else:
            self.report.setdefault("policy", {})
            self.report["policy"].update({
                "extreme_filter": "finite/non-silent, per-class robust power 4 MAD, PAPR <= 30",
                "npy_label_maps": "numeric Y_*.npy labels are inferred from data; string labels use configured name maps",
                "rml_snr": f"SNR >= {UNIFIED_MIN_SNR_DB:g} dB (unified)",
                "unified_min_snr_db": UNIFIED_MIN_SNR_DB,
                "snr_policy": (
                    f"all datasets with finite snr: keep snr >= {UNIFIED_MIN_SNR_DB:g} dB; "
                    f"radar_mod15 nominal {RADAR_MOD15_NOMINAL_SNR_DB:g} dB"
                ),
                "split": (
                    "preserve verified capture groups when available; otherwise mark group metadata "
                    "unavailable and use the dataset's documented sample-level split"
                ),
                "group_metadata": (
                    f"receiver_id/session_id/channel_id/capture_id are always present; "
                    f"{MISSING_METADATA!r} means unavailable from source"
                ),
                "group_split_claims": (
                    "Only datasets whose immutable split manifest says verified_group_held_out "
                    "may be reported as receiver/session/capture held out"
                ),
                "split_usage": "*_train.h5: MAE pretrain only; *_val.h5: all validation and model selection; *_test.h5: downstream/finetune training only (not for evaluation)",
                "eval_quality_datasets": sorted(EVAL_QUALITY_DATASETS),
                "eval_quality_filter": "val/test only: per-class power 3 MAD, PAPR <= 20, lag1-correlation >= 0.05",
                "rml2018_eval_snr": f"val/test only: SNR >= {RML2018_EVAL_MIN_SNR:g} dB",
            })
        self.pool = defaultdict(list)
        self.final_label_fields = {}
        self.touched: set[str] = set()

    @staticmethod
    def _load_json(path: Path, default):
        if not path.exists():
            return default() if callable(default) else default
        return json.loads(path.read_text(encoding="utf-8"))

    def done(self, name: str, split: str, writer: Writer, raw: int, removed: Counter, labels: dict):
        self.touched.add(name)
        writer.close()
        self.pool[split].append((writer.path.name, writer.task_id))
        entry = self.report["datasets"].setdefault(name, {"labels": labels, "splits": {}})
        entry["splits"][split] = {"raw": raw, "kept": writer.count, "removed": dict(removed), "file": writer.path.name}


def _skip_eval_refine_for_pretrain(ctx: Context, name: str) -> bool:
    """预训练 7 库 convert 已 quality + 分层划分，finalize 不再 refine val/test。"""
    if name not in PRETRAIN_STEMS:
        return False
    if ctx.report.get("policy", {}).get("h5_schema") == "pretrain_v2_joint_energy":
        return True
    entry = ctx.report.get("datasets", {}).get(name)
    if not isinstance(entry, dict):
        return False
    if entry.get("stratified_split"):
        return True
    splits = entry.get("splits", {})
    return all(splits.get(split, {}).get("file") for split in ("train", "test", "val"))


def resolve_y_path(root: Path, source_split: str) -> Path | None:
    for candidate in (root / f"Y_{source_split}.npy", root / source_split / "label.npy"):
        if candidate.exists():
            return candidate
    return None


def infer_npy_labels(root: Path, pairs: list[tuple[str, str]], fallback: dict | None = None) -> dict:
    """从 Y_*.npy / label.npy 推断标签表；数值类标签以数据为准，字符串类标签使用 fallback 名称映射。"""
    fallback = dict(fallback or {})
    numeric_values: set[int] = set()
    string_names: set[str] = set()
    for source_split, _output_split in pairs:
        yp = resolve_y_path(root, source_split)
        if yp is None:
            continue
        y = np.asarray(np.load(yp, mmap_mode="r")).reshape(-1)
        if y.size == 0:
            continue
        if y.dtype.kind in "OUS":
            string_names.update(str(v) for v in np.unique(y))
        else:
            numeric_values.update(int(v) for v in np.unique(y.astype(np.int64)))
    if string_names:
        if not fallback:
            raise ValueError(f"{root} 含字符串标签但未提供名称映射")
        unknown = string_names - set(fallback)
        if unknown:
            raise ValueError(f"{root} 存在未配置的标签名: {sorted(unknown)}")
        return dict(fallback)
    if numeric_values:
        return {str(v): v for v in sorted(numeric_values)}
    if fallback:
        return dict(fallback)
    raise ValueError(f"{root} 未找到可用的 Y_*.npy / label.npy 标签文件")


def npy_dataset(
    ctx: Context,
    name: str,
    dataset_id: int,
    root: Path,
    task: int,
    label_field: str,
    labels: dict | None,
    pairs: list[tuple[str, str]],
    *,
    mirror_mod_label_id: bool = False,
):
    labels = infer_npy_labels(root, pairs, labels)
    ctx.maps["datasets"][str(dataset_id)] = name
    if task:
        ctx.maps["emitters"][name] = labels
    elif label_field == "source_label_id":
        ctx.maps["sources"][name] = labels
        if mirror_mod_label_id:
            ctx.maps["modulations"][name] = labels
    else:
        ctx.maps["modulations"][name] = labels
    for source_split, output_split in pairs:
        xp, yp = root / f"X_{source_split}.npy", root / f"Y_{source_split}.npy"
        if not xp.exists():
            xp, yp = root / source_split / "data.npy", root / source_split / "label.npy"
        x, y = np.load(xp, mmap_mode="r"), np.asarray(np.load(yp, mmap_mode="r")).reshape(-1)
        if y.dtype.kind in "OUS":
            y = np.asarray([labels[str(v)] for v in y], dtype=np.int32)
        else:
            y = y.astype(np.int32)
        writer = _make_writer(ctx.h5 / f"{name}_{output_split}.h5", as_iq(x[:1]).shape[-1], x.dtype, dataset_id, task, root)
        removed = Counter()
        for start in range(0, len(x), 2048):
            block = as_iq(np.asarray(x[start:start + 2048]))
            block_y = y[start:start + len(block)]
            meta = {label_field: block_y}
            if mirror_mod_label_id:
                meta["mod_label_id"] = block_y
            writer.append_clean(block, removed, **meta)
        ctx.done(name, output_split, writer, len(x), removed, labels)


def load_pickle(path: Path):
    with path.open("rb") as f:
        return pickle.load(f, encoding="latin1")


def rml_pickle(ctx: Context, name: str, dataset_id: int, path: Path):
    data = load_pickle(path)
    mods = sorted({str(k[0]) for k in data})
    labels = {m: RML2016_ORDER.index(m) if m in RML2016_ORDER else i for i, m in enumerate(mods)}
    ctx.maps["datasets"][str(dataset_id)], ctx.maps["modulations"][name] = name, labels
    snr_removed = sum(len(v) for (m, s), v in data.items() if float(s) < UNIFIED_MIN_SNR_DB)
    if _PRETRAIN_H5_CFG is not None:
        removed = Counter()
        raw = 0
        iq_parts: list[np.ndarray] = []
        label_parts: list[np.ndarray] = []
        snr_parts: list[np.ndarray] = []
        native_length: int | None = None
        dtype = np.float32
        for (_mod, snr), values in sorted(data.items(), key=lambda x: (str(x[0][0]), float(x[0][1]))):
            if float(snr) < UNIFIED_MIN_SNR_DB:
                continue
            block = as_iq(np.asarray(values))
            raw += len(block)
            native_length = int(block.shape[-1])
            dtype = block.dtype
            target = _resolve_target_length(name, native_length)
            block_prep = _preprocess_canonical_iq(block)
            keep, reasons = quality_mask(
                _iq_at_quality_stage(block, target),
                labels=np.full(len(block), labels[str(_mod)], dtype=np.int32),
                dataset_name=name,
            )
            removed.update(reasons)
            if not np.any(keep):
                continue
            block_out = _resize_canonical_iq(block_prep[keep], target)
            iq_parts.append(block_out)
            label_parts.append(np.full(int(keep.sum()), labels[str(_mod)], dtype=np.int32))
            snr_parts.append(np.full(int(keep.sum()), float(snr), dtype=np.float32))
        if not iq_parts:
            raise ValueError(f"{name} 清洗后无可用样本")
        iq_all = np.concatenate(iq_parts, axis=0)
        labels_all = np.concatenate(label_parts, axis=0)
        write_pretrain_stratified_splits(
            ctx,
            name,
            dataset_id,
            path,
            int(_resolve_target_length(name, int(native_length or iq_all.shape[-1]))),
            dtype,
            iq_all,
            labels_all,
            labels,
            removed,
            raw,
            label_field="mod_label_id",
            extra_meta={
                "snr": np.concatenate(snr_parts, axis=0),
                "sample_rate_hz": _fs_meta_array(len(labels_all), name),
            },
            snr_removed={_SNR_LT_MIN_REASON: int(snr_removed)},
        )
        return
    groups = {"train": [], "val": []}
    for (mod, snr), values in sorted(data.items(), key=lambda x: (str(x[0][0]), float(x[0][1]))):
        if float(snr) < UNIFIED_MIN_SNR_DB:
            continue
        values = as_iq(np.asarray(values))
        cut = int(len(values) * 0.8)
        groups["train"].append((values[:cut], mod, snr))
        groups["val"].append((values[cut:], mod, snr))
    for split, items in groups.items():
        writer = _make_writer(ctx.h5 / f"{name}_{split}.h5", items[0][0].shape[-1], items[0][0].dtype, dataset_id, 0, path)
        removed, raw = Counter({_SNR_LT_MIN_REASON: snr_removed}), 0
        for values, mod, snr in items:
            raw += len(values)
            writer.append_clean(
                values,
                removed,
                mod_label_id=np.full(len(values), labels[str(mod)], dtype=np.int32),
                snr=np.full(len(values), snr, dtype=np.float32),
                sample_rate_hz=_fs_meta_array(len(values), name),
            )
        ctx.done(name, split, writer, raw, removed, labels)


def rml2018(ctx: Context, dataset_id: int):
    name = "rml2018_1a"
    path = NON_EMITTER / "RML2018.1A" / "GOLD_XYZ_OSC.0001_1024.hdf5"
    labels = {m: i for i, m in enumerate(RML2018_ORDER)}
    ctx.maps["datasets"][str(dataset_id)], ctx.maps["modulations"][name] = name, labels
    with h5py.File(path, "r") as f:
        writers = {s: _make_writer(ctx.h5 / f"{name}_{s}.h5", 1024, f["X"].dtype, dataset_id, 0, path) for s in ("train", "val")}
        removed, raw, sequence = {s: Counter() for s in writers}, Counter(), 0
        for start in range(0, len(f["X"]), 2048):
            snr = np.asarray(f["Z"][start:start + 2048]).reshape(-1)
            allowed = snr_keep_mask(snr)
            block = as_iq(np.asarray(f["X"][start:start + 2048])[allowed])
            labels_block = np.argmax(np.asarray(f["Y"][start:start + 2048])[allowed], axis=1)
            snr_block = snr[allowed]
            removed["train"][_SNR_LT_MIN_REASON] += int((~allowed).sum())
            idx = np.arange(sequence, sequence + len(block))
            sequence += len(block)
            for split, choose in (("train", idx % 5 != 0), ("val", idx % 5 == 0)):
                raw[split] += int(choose.sum())
                writers[split].append_clean(block[choose], removed[split], mod_label_id=labels_block[choose], snr=snr_block[choose])
        for split in writers:
            ctx.done(name, split, writers[split], raw[split], removed[split], labels)


def dat_emitters(ctx: Context, name: str, dataset_id: int, root: Path, window: int, real: bool):
    files, labels = sorted(root.glob("*.dat")), {}
    ctx.maps["datasets"][str(dataset_id)] = name
    output_dtype = np.float32 if real else np.dtype("<i2")
    writers = {s: _make_writer(ctx.h5 / f"{name}_{s}.h5", window, output_dtype, dataset_id, 1, root) for s in ("train", "val")}
    removed, raw = {s: Counter() for s in writers}, Counter()
    for path in files:
        parts, data = path.stem.split("#"), np.memmap(path, dtype="<i2", mode="r")
        emitter_id = int(parts[0])
        labels[parts[1]] = emitter_id
        if real:
            n = len(data) // window
            values = np.asarray(data[:n * window]).reshape(n, window)
            iq = as_iq(values).astype(np.float32)
        else:
            n = len(data) // (2 * window)
            iq = np.asarray(data[:n * 2 * window]).reshape(n, window, 2).transpose(0, 2, 1)
        for start in range(0, len(iq), 2048):
            block, idx = iq[start:start + 2048], np.arange(start, min(start + 2048, len(iq)))
            for split, choose in (("train", idx % 5 != 0), ("val", idx % 5 == 0)):
                raw[split] += int(choose.sum())
                writers[split].append_clean(block[choose], removed[split], emitter_id=np.full(choose.sum(), emitter_id))
    ctx.maps["emitters"][name] = labels
    for split in writers:
        ctx.done(name, split, writers[split], raw[split], removed[split], labels)


def choose_label_field(files: list[Path]) -> str:
    for field in ("emitter_id", "mod_label_id", "source_label_id", "global_label_id"):
        for path in files:
            with h5py.File(path, "r") as f:
                if field in f and np.any(np.asarray(f[field][: min(len(f[field]), 4096)]) >= 0):
                    return field
    raise ValueError(f"没有找到可用于类别均衡的标签字段: {[str(p) for p in files]}")


def copy_records(inputs: list[Path], selected: np.ndarray, writer: Writer) -> None:
    selected = np.asarray(selected, dtype=np.int64)
    if selected.size == 0:
        return
    selected.sort()
    offset = 0
    cursor = 0
    for path in inputs:
        with h5py.File(path, "r") as f:
            n = len(f["iq"])
            hi = offset + n
            lo_pos = np.searchsorted(selected, offset, side="left", sorter=None)
            hi_pos = np.searchsorted(selected, hi, side="left", sorter=None)
            if hi_pos > lo_pos:
                local = selected[lo_pos:hi_pos] - offset
                v2_source = _is_h5_v2(f)
                v2_writer = isinstance(writer, PretrainH5Writer)
                for start in range(0, len(local), 2048):
                    idx = local[start:start + 2048]
                    if v2_source and v2_writer:
                        meta = {
                            key: np.asarray(f[key][idx])
                            for key in PRETRAIN_V2_COPY_KEYS
                            if key in f and key != "iq"
                        }
                        writer.append_preprocessed(np.asarray(f["iq"][idx]), **meta)
                    else:
                        meta = {key: np.asarray(f[key][idx]) for key in Writer.ALL_FIELDS if key in f}
                        writer.append_raw(np.asarray(f["iq"][idx]), **meta)
                cursor = hi_pos
            offset = hi
    if cursor != len(selected):
        raise RuntimeError("部分均衡索引未能复制到最终 H5")


def rebalance_dataset(ctx: Context, name: str, entry: dict) -> tuple[list[tuple[str, int]], dict]:
    source_files = [ctx.h5 / info["file"] for info in entry["splits"].values()]
    source_files = [p for p in source_files if p.exists()]
    if not source_files:
        return [], {}
    with h5py.File(source_files[0], "r") as first:
        length = int(first["iq"].shape[-1])
        dtype = first["iq"].dtype
        if _is_h5_v2(first):
            dataset_id = _h5_dataset_id(first)
            task_id = _h5_task_id(first)
        else:
            dataset_id = int(first["dataset_id"][0])
            task_id = int(first["task_type_id"][0])
    label_field = choose_label_field(source_files)
    class_indices: dict[int, list[int]] = defaultdict(list)
    offset = 0
    for path in source_files:
        with h5py.File(path, "r") as f:
            labels = np.asarray(f[label_field])
            for cls in np.unique(labels[labels >= 0]):
                local = np.flatnonzero(labels == cls)
                class_indices[int(cls)].extend((local + offset).tolist())
            offset += len(labels)
    if not class_indices:
        raise ValueError(f"{name} 清洗后没有可用类别")
    min_count = min(len(v) for v in class_indices.values())
    # Make each class divisible enough for a clean 8:1:1 split.
    min_count = (min_count // 10) * 10
    if min_count <= 0:
        raise ValueError(f"{name} 类别均衡后样本数为 0")
    rng = np.random.default_rng(20260629)
    split_indices = {"train": [], "val": [], "test": []}
    for cls in sorted(class_indices):
        chosen = rng.choice(np.asarray(class_indices[cls], dtype=np.int64), size=min_count, replace=False)
        rng.shuffle(chosen)
        n_train, n_val = int(0.8 * min_count), int(0.1 * min_count)
        split_indices["train"].append(chosen[:n_train])
        split_indices["val"].append(chosen[n_train:n_train + n_val])
        split_indices["test"].append(chosen[n_train + n_val:])
    final_files: list[tuple[str, int]] = []
    final_report = {
        "label_field": label_field,
        "classes": len(class_indices),
        "candidate_counts_by_class": {str(k): len(v) for k, v in sorted(class_indices.items())},
        "balanced_per_class": min_count,
        "splits": {},
    }
    temp_paths = {}
    for split, chunks in split_indices.items():
        idx = np.concatenate(chunks)
        rng.shuffle(idx)
        tmp_path = ctx.h5 / f"{name}__balanced_{split}.h5"
        writer = _make_writer(tmp_path, length, dtype, dataset_id, task_id, source_files[0])
        copy_records(source_files, idx, writer)
        writer.close()
        temp_paths[split] = tmp_path
        final_report["splits"][split] = {"kept": int(len(idx)), "file": f"{name}_{split}.h5"}
    for path in source_files:
        path.unlink(missing_ok=True)
    for split, tmp_path in temp_paths.items():
        final_path = ctx.h5 / f"{name}_{split}.h5"
        final_path.unlink(missing_ok=True)
        tmp_path.rename(final_path)
        final_files.append((final_path.name, task_id))
    return final_files, final_report


def collect_label_ids_from_h5(files: list[Path], label_field: str) -> list[int]:
    ids: set[int] = set()
    for path in files:
        with h5py.File(path, "r") as f:
            if label_field not in f:
                continue
            labels = np.asarray(f[label_field])
            ids.update(int(v) for v in np.unique(labels[labels >= 0]))
    return sorted(ids)


def sync_label_maps_from_balanced(ctx: Context) -> None:
    field_to_map = {
        "emitter_id": "emitters",
        "mod_label_id": "modulations",
        "source_label_id": "sources",
    }
    for name, entry in ctx.report.get("datasets", {}).items():
        split_meta = (
            entry.get("balanced_split")
            or entry.get("val_test_split")
            or entry.get("group_heldout_split")
            or {}
        )
        label_field = split_meta.get("label_field")
        map_key = field_to_map.get(label_field or "")
        if not map_key:
            continue
        existing = ctx.maps.get(map_key, {}).get(name, entry.get("labels", {}))
        if split_meta.get("candidate_counts_by_class"):
            ids = sorted(int(k) for k in split_meta["candidate_counts_by_class"])
        else:
            files = [ctx.h5 / info["file"] for info in split_meta.get("splits", entry.get("splits", {})).values()]
            files = [p for p in files if p.exists()]
            if not files:
                continue
            ids = collect_label_ids_from_h5(files, label_field)
        if not ids:
            continue
        if existing and any(not str(k).isdigit() for k in existing):
            synced = dict(existing)
            missing = set(ids) - set(synced.values())
            if missing:
                raise ValueError(f"{name} 的 H5 标签 {sorted(missing)} 未出现在名称映射中")
        else:
            synced = {str(i): i for i in ids}
        ctx.maps.setdefault(map_key, {})[name] = synced
        entry["labels"] = synced


def register_val_test_only(ctx: Context, name: str, entry: dict) -> None:
    """注册仅 val/test 划分的数据集（跳过 rebalance）。"""
    (ctx.h5 / f"{name}_train.h5").unlink(missing_ok=True)
    split_paths: list[Path] = []
    for split in ("test", "val"):
        info = entry.get("splits", {}).get(split)
        if info is None:
            continue
        path = ctx.h5 / info["file"]
        if not path.is_file():
            continue
        split_paths.append(path)
        with h5py.File(path, "r") as f:
            task_id = int(f["task_type_id"][0])
            label_field = choose_label_field([path])
        ctx.final_label_fields[path.name] = label_field
        ctx.pool[split].append((path.name, task_id))
    entry.pop("balanced_split", None)
    if split_paths and "val_test_split" not in entry:
        entry["val_test_split"] = {
            "label_field": choose_label_field(split_paths),
            "test_ratio": VAL_TEST_SPLIT_RATIO,
            "no_class_balance": True,
            "splits": {
                split: {"kept": entry["splits"][split]["kept"], "file": entry["splits"][split]["file"]}
                for split in ("test", "val")
                if split in entry.get("splits", {})
            },
        }


def register_group_heldout(ctx: Context, name: str, entry: dict) -> None:
    """保留源 capture 划分，禁止 rebalance 再次按样本随机拆分。"""
    split_paths: list[Path] = []
    split_report: dict[str, dict] = {}
    label_field: str | None = None
    for split in ("train", "val", "test"):
        info = entry.get("splits", {}).get(split)
        if info is None:
            continue
        path = ctx.h5 / str(info["file"])
        if not path.is_file():
            continue
        with h5py.File(path, "r") as f:
            if int(f["iq"].shape[0]) == 0:
                continue
            task_id = int(f["task_type_id"][0])
        current_label_field = choose_label_field([path])
        if label_field is None:
            label_field = current_label_field
        elif current_label_field != label_field:
            raise ValueError(
                f"{name} 的 group-held-out split 标签字段不一致: "
                f"{label_field!r} != {current_label_field!r}"
            )
        ctx.final_label_fields[path.name] = current_label_field
        ctx.pool[split].append((path.name, task_id))
        split_paths.append(path)
        split_report[split] = {
            "kept": int(info.get("kept", 0)),
            "file": path.name,
        }
    if len(split_paths) < 2:
        raise ValueError(f"{name} 声称 group-held-out，但少于两个非空 split")
    entry.pop("balanced_split", None)
    entry["group_heldout_split"] = {
        "label_field": label_field,
        "preserve_capture_groups": True,
        "splits": split_report,
    }


def register_pretrain_stem(ctx: Context, name: str, entry: dict) -> None:
    """预训练 7 库：convert 已写 train/test/val，跳过 rebalance。"""
    label_field: str | None = None
    split_report: dict[str, dict] = {}
    for split in ("train", "test", "val"):
        info = entry.get("splits", {}).get(split)
        if info is None:
            continue
        path = ctx.h5 / str(info["file"])
        if not path.is_file():
            continue
        with h5py.File(path, "r") as f:
            if int(f["iq"].shape[0]) == 0:
                continue
            task_id = _h5_task_id(f)
        current_label_field = choose_label_field([path])
        if label_field is None:
            label_field = current_label_field
        elif current_label_field != label_field:
            raise ValueError(
                f"{name} 预训练 split 标签字段不一致: {label_field!r} != {current_label_field!r}"
            )
        ctx.final_label_fields[path.name] = current_label_field
        ctx.pool[split].append((path.name, task_id))
        split_report[split] = {
            "kept": int(info.get("kept", 0)),
            "file": path.name,
        }
    if len(split_report) < 2:
        raise ValueError(f"{name} 预训练 split 少于两个非空文件")
    entry.pop("balanced_split", None)
    entry.setdefault(
        "stratified_split",
        {
            "label_field": label_field,
            "ratios": list(PRETRAIN_STRATIFIED_RATIOS),
            "no_class_balance": True,
            "splits": split_report,
        },
    )


def rebalance_all(ctx: Context) -> None:
    ctx.pool = defaultdict(list)
    dataset_names = sorted(ctx.touched) if ctx.touched else sorted(ctx.report.get("datasets", {}))
    for name in dataset_names:
        entry = ctx.report.get("datasets", {}).get(name)
        if entry is None:
            continue
        if name in PRETRAIN_STEMS:
            splits = entry.get("splits", {})
            if entry.get("stratified_split") or all(
                splits.get(s, {}).get("file") for s in ("train", "test", "val")
            ):
                register_pretrain_stem(ctx, name, entry)
                continue
        if entry.get("split_provenance", {}).get("claim_group_held_out"):
            register_group_heldout(ctx, name, entry)
            continue
        if name in VAL_TEST_ONLY_DATASETS:
            register_val_test_only(ctx, name, entry)
            continue
        final_files, final_report = rebalance_dataset(ctx, name, entry)
        entry["balanced_split"] = final_report
        for filename, task_id in final_files:
            ctx.final_label_fields[filename] = final_report.get("label_field", "")
            if filename.endswith("_train.h5"):
                ctx.pool["train"].append((filename, task_id))
            elif filename.endswith("_val.h5"):
                ctx.pool["val"].append((filename, task_id))
            elif filename.endswith("_test.h5"):
                ctx.pool["test"].append((filename, task_id))


def _load_h5_records(path: Path) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    load_keys = Writer.record_load_keys()
    with h5py.File(path, "r") as f:
        n = int(f["iq"].shape[0])
        iq_parts: list[np.ndarray] = []
        meta: dict[str, list[np.ndarray]] = {key: [] for key in load_keys}
        for start in range(0, n, 2048):
            end = min(start + 2048, n)
            iq_parts.append(np.asarray(f["iq"][start:end]))
            for key in load_keys:
                if key in f:
                    meta[key].append(np.asarray(f[key][start:end]))
        iq = np.concatenate(iq_parts, axis=0)
        merged = {key: np.concatenate(parts) if parts else None for key, parts in meta.items()}
    return iq, merged


def refine_eval_split_h5(
    path: Path,
    label_field: str,
    dataset_name: str,
    rng: np.random.Generator,
) -> dict:
    iq, meta = _load_h5_records(path)
    if meta.get(label_field) is None:
        label_field = choose_label_field([path])
    labels = np.asarray(meta[label_field]).reshape(-1)
    keep, reasons = quality_mask(iq, labels=labels, strict=True, dataset_name=dataset_name)
    if dataset_name == "rml2018_1a" and meta.get("snr") is not None:
        snr = np.asarray(meta["snr"]).reshape(-1)
        snr_ok = snr_keep_mask(snr, min_db=RML2018_EVAL_MIN_SNR)
        reasons[_SNR_LT_MIN_REASON] = int((keep & ~snr_ok).sum())
        keep &= snr_ok
    class_indices: dict[int, list[int]] = defaultdict(list)
    for i in np.flatnonzero(keep):
        cls = int(labels[i])
        if cls >= 0:
            class_indices[cls].append(int(i))
    if not class_indices:
        raise ValueError(f"{path} refine 后无可用类别")
    min_count = min(len(v) for v in class_indices.values())
    selected: list[np.ndarray] = []
    for cls in sorted(class_indices):
        idx = np.asarray(class_indices[cls], dtype=np.int64)
        if len(idx) > min_count:
            idx = rng.choice(idx, size=min_count, replace=False)
        selected.append(idx)
    selected_idx = np.concatenate(selected)
    rng.shuffle(selected_idx)
    load_keys = Writer.record_load_keys()
    with h5py.File(path, "r") as f:
        length = int(f["iq"].shape[-1])
        dtype = f["iq"].dtype
        schema_v2 = _is_h5_v2(f)
        dataset_id = _h5_dataset_id(f)
        task_id = _h5_task_id(f)
        source = Path(str(f.attrs.get("source_path", path)))
    tmp_path = path.with_suffix(".refine.h5")
    if schema_v2:
        writer: Writer | PretrainWriter = PretrainWriter(
            tmp_path,
            length,
            dataset_id,
            task_id,
            source,
            dataset_name=dataset_name,
            cfg=PretrainH5Config(precompute_joint_energy=False),
        )
    else:
        writer = _make_writer(tmp_path, length, dtype, dataset_id, task_id, source)
    quality_note = "eval-refined: strict per-class power 3 MAD, PAPR<=20, lag1-correlation>=0.05"
    if dataset_name == "rml2018_1a":
        quality_note += f", SNR>={RML2018_EVAL_MIN_SNR:g}"
    writer.f.attrs["quality_filter"] = quality_note
    for start in range(0, len(selected_idx), 2048):
        chunk_idx = selected_idx[start:start + 2048]
        chunk_meta = {
            key: meta[key][chunk_idx]
            for key in load_keys
            if meta.get(key) is not None
        }
        if schema_v2:
            writer.append_preprocessed(iq[chunk_idx], **chunk_meta)
        else:
            writer.append_raw(iq[chunk_idx], **chunk_meta)
    writer.close()
    path.unlink(missing_ok=True)
    tmp_path.rename(path)
    return {
        "raw": int(len(iq)),
        "after_quality": int(keep.sum()),
        "kept": int(len(selected_idx)),
        "removed": dict(reasons),
        "balanced_per_class": int(min_count),
        "classes": len(class_indices),
    }


def refine_eval_splits(ctx: Context, datasets: set[str] | None = None) -> None:
    rng = np.random.default_rng(20260703)
    targets = datasets if datasets is not None else set(EVAL_QUALITY_DATASETS)
    for name in sorted(targets):
        if _skip_eval_refine_for_pretrain(ctx, name):
            continue
        entry = ctx.report.get("datasets", {}).get(name)
        if entry is None:
            continue
        balanced = entry.setdefault("balanced_split", {})
        label_field = balanced.get("label_field")
        split_paths = [ctx.h5 / f"{name}_{split}.h5" for split in EVAL_QUALITY_SPLITS]
        existing = [p for p in split_paths if p.exists()]
        if not existing:
            continue
        if not label_field:
            label_field = choose_label_field(existing)
        refined: dict[str, dict] = {}
        for split in EVAL_QUALITY_SPLITS:
            path = ctx.h5 / f"{name}_{split}.h5"
            if not path.exists():
                continue
            refined[split] = refine_eval_split_h5(path, label_field, name, rng)
            balanced.setdefault("splits", {})[split] = {
                "kept": refined[split]["kept"],
                "file": path.name,
            }
        if not refined:
            continue
        entry["eval_quality_refine"] = refined
        balanced["balanced_per_class"] = min(r["balanced_per_class"] for r in refined.values())
        entry["balanced_split"] = balanced


def rml10b(ctx: Context, dataset_id: int):
    name, root = "rml2016_10b", NON_EMITTER / "RML2016.10b" / "2016.10b"
    files = sorted(root.glob("*.txt"))
    mods = sorted({p.stem.rsplit(" ", 1)[0] for p in files})
    labels = {m: RML2016_ORDER.index(m) if m in RML2016_ORDER else i for i, m in enumerate(mods)}
    ctx.maps["datasets"][str(dataset_id)], ctx.maps["modulations"][name] = name, labels
    snr_removed = Counter(
        {_SNR_LT_MIN_REASON: sum(float(p.stem.rsplit(" ", 1)[1]) < UNIFIED_MIN_SNR_DB for p in files)}
    )
    removed = Counter()
    raw = 0
    malformed = 0
    iq_parts: list[np.ndarray] = []
    label_parts: list[np.ndarray] = []
    snr_parts: list[np.ndarray] = []
    target_length = _resolve_target_length(name, 128)
    for path in files:
        mod, snr_text = path.stem.rsplit(" ", 1)
        snr = float(snr_text)
        if snr < UNIFIED_MIN_SNR_DB:
            continue
        batch: list[np.ndarray] = []
        with path.open(encoding="utf-8") as f:
            for line in f:
                values = np.fromstring(line.replace("(", "").replace(")", ""), dtype=np.complex64, sep=" ")
                if len(values) != 128:
                    malformed += 1
                    continue
                batch.append(values)
                if len(batch) == 2048:
                    block_raw = as_iq(np.asarray(batch))
                    raw += len(block_raw)
                    mod_labels = np.full(len(block_raw), labels[mod], dtype=np.int32)
                    block_prep = _preprocess_canonical_iq(block_raw)
                    keep, reasons = quality_mask(
                        _iq_at_quality_stage(block_raw, target_length),
                        labels=mod_labels,
                        dataset_name=name,
                    )
                    removed.update(reasons)
                    if np.any(keep):
                        iq_parts.append(_resize_canonical_iq(block_prep[keep], target_length))
                        label_parts.append(mod_labels[keep])
                        snr_parts.append(np.full(int(keep.sum()), snr, dtype=np.float32))
                    batch.clear()
            if batch:
                block_raw = as_iq(np.asarray(batch))
                raw += len(block_raw)
                mod_labels = np.full(len(block_raw), labels[mod], dtype=np.int32)
                block_prep = _preprocess_canonical_iq(block_raw)
                keep, reasons = quality_mask(
                    _iq_at_quality_stage(block_raw, target_length),
                    labels=mod_labels,
                    dataset_name=name,
                )
                removed.update(reasons)
                if np.any(keep):
                    iq_parts.append(_resize_canonical_iq(block_prep[keep], target_length))
                    label_parts.append(mod_labels[keep])
                    snr_parts.append(np.full(int(keep.sum()), snr, dtype=np.float32))
    if malformed:
        removed["malformed"] += malformed
    if not iq_parts:
        raise ValueError(f"{name} 清洗后无可用样本")
    iq_all = np.concatenate(iq_parts, axis=0)
    labels_all = np.concatenate(label_parts, axis=0)
    write_pretrain_stratified_splits(
        ctx,
        name,
        dataset_id,
        root,
        target_length,
        np.float32,
        iq_all,
        labels_all,
        labels,
        removed,
        raw,
        label_field="mod_label_id",
        extra_meta={
            "snr": np.concatenate(snr_parts, axis=0),
            "sample_rate_hz": _fs_meta_array(len(labels_all), name),
        },
        snr_removed=dict(snr_removed),
    )


def wisig_manytx(ctx: Context, dataset_id: int, *, equalized: int = 0) -> None:
    """WiSig ManyTx 个体识别：按完整 Rx/Day capture 做 group-held-out。

    旧 ``X_train/X_val/X_test.npy`` 不携带 receiver/capture sidecar，无法证明
    group-held-out，因此不再把它当作可信划分输入。
    """
    name = "wisig"
    root = EMITTER / "wisig"
    npy_root = root / "npy_data" / ("equalized_data_0" if equalized == 0 else "equalized_data_1")
    pkl_path = root / "ManyTx.pkl"
    if not pkl_path.is_file():
        has_legacy_npy = all(
            (npy_root / f"X_{split}.npy").is_file()
            for split in ("train", "val", "test")
        )
        if has_legacy_npy:
            raise ValueError(
                f"{npy_root} 只有旧的样本级预拆分 NPY，且缺少 receiver/session/capture sidecar；"
                "无法验证 group-held-out。请提供原始 ManyTx.pkl 后重新转换。"
            )
        raise FileNotFoundError(
            f"未找到 {pkl_path}；可信 WiSig 划分必须从原始 pickle 恢复 Rx/Day 元数据"
        )
    ctx.maps["datasets"][str(dataset_id)] = name
    writers = {
        split: _make_writer(ctx.h5 / f"{name}_{split}.h5", 256, np.float32, dataset_id, 1, pkl_path)
        for split in ("train", "val", "test")
    }
    removed = {split: Counter() for split in writers}
    raw = Counter()

    def on_batch(
        split: str,
        iq: np.ndarray,
        labels: np.ndarray,
        capture_metadata: dict[str, str],
    ) -> None:
        raw[split] += len(iq)
        writers[split].append_clean(
            iq.astype(np.float32, copy=False),
            removed[split],
            emitter_id=labels,
            **capture_metadata,
        )

    sink = WiSigBlockSink(equalized=equalized, on_batch=on_batch)
    meta = stream_manytx_blocks(
        pkl_path,
        sink,
    )
    labels = emitter_labels(meta)
    ctx.maps["emitters"][name] = labels
    ctx.maps["receivers"][name] = dict(meta.receiver_ids)
    ctx.maps["sessions"][name] = dict(meta.session_ids)
    split_group_counts = Counter(meta.group_assignments.values())
    for split in writers:
        writers[split].f.attrs["split_strategy"] = meta.split_strategy
        writers[split].f.attrs["group_field"] = "capture_id"
        writers[split].f.attrs["receiver_id_map"] = json.dumps(meta.receiver_ids, ensure_ascii=False)
        writers[split].f.attrs["session_id_map"] = json.dumps(meta.session_ids, ensure_ascii=False)
        ctx.done(name, split, writers[split], raw[split], removed[split], labels)
    entry = ctx.report["datasets"][name]
    entry["split_provenance"] = {
        "strategy": meta.split_strategy,
        "group_fields": ["receiver_id", "session_id", "capture_id"],
        "claim_group_held_out": True,
        "verification": "pending immutable split manifest validation",
        "groups_by_split": {
            split: int(split_group_counts.get(split, 0))
            for split in ("train", "val", "test")
        },
    }


def _csv_field(row: dict[str, str], *names: str) -> str:
    for name in names:
        if name in row and row[name].strip():
            return row[name].strip()
        spaced = f" {name.strip()}"
        if spaced in row and row[spaced].strip():
            return row[spaced].strip()
    raise KeyError(f"缺少字段 {names}，可用列: {list(row)}")


def radcom_waveform_to_iq(waveform: np.ndarray) -> np.ndarray:
    waveform = np.asarray(waveform, dtype=np.float32)
    return np.stack([waveform[:128], waveform[128:]], axis=0)


def parse_radcom_key(key: str) -> tuple[str, str, float, object]:
    mod, sig, snr, sample_idx = ast.literal_eval(key)
    return str(mod), str(sig), float(snr), sample_idx


def build_radcom_label_maps(path: Path) -> tuple[dict[str, int], dict[str, int]]:
    mods: set[str] = set()
    sigs: set[str] = set()
    with h5py.File(path, "r") as f:
        for key in f.keys():
            mod, sig, _, _ = parse_radcom_key(key)
            mods.add(mod)
            sigs.add(sig)
    mod_labels = {name: idx for idx, name in enumerate(sorted(mods))}
    sig_labels = {name: idx for idx, name in enumerate(sorted(sigs))}
    return mod_labels, sig_labels


def radcom_hdf5(
    ctx: Context,
    name: str,
    dataset_id: int,
    path: Path,
    *,
    snr_min: float | None = UNIFIED_MIN_SNR_DB,
    jobs: int = 1,
) -> None:
    if jobs > 1:
        radcom_hdf5_parallel(ctx, name, dataset_id, path, snr_min=snr_min, jobs=jobs)
        return
    if not path.is_file():
        raise FileNotFoundError(f"未找到 RadarComm HDF5: {path}")
    mod_labels, sig_labels = build_radcom_label_maps(path)
    ctx.maps["datasets"][str(dataset_id)] = name
    ctx.maps["modulations"][name] = mod_labels
    ctx.maps["sources"][name] = sig_labels
    writer = _make_writer(ctx.h5 / f"{name}_train.h5", 128, np.float32, dataset_id, 0, path)
    removed, raw = Counter(), 0
    batch_iq: list[np.ndarray] = []
    batch_mod: list[int] = []
    batch_sig: list[int] = []
    batch_snr: list[float] = []

    def flush() -> None:
        nonlocal batch_iq, batch_mod, batch_sig, batch_snr
        if not batch_iq:
            return
        block = np.stack(batch_iq, axis=0)
        writer.append_clean(
            block,
            removed,
            mod_label_id=np.asarray(batch_mod, dtype=np.int32),
            source_label_id=np.asarray(batch_sig, dtype=np.int32),
            snr=np.asarray(batch_snr, dtype=np.float32),
        )
        batch_iq, batch_mod, batch_sig, batch_snr = [], [], [], []

    with h5py.File(path, "r") as f:
        for key in f.keys():
            mod, sig, snr, _ = parse_radcom_key(key)
            if snr_min is not None and snr < snr_min:
                removed[_SNR_LT_MIN_REASON] += 1
                continue
            raw += 1
            batch_iq.append(radcom_waveform_to_iq(f[key][:]))
            batch_mod.append(mod_labels[mod])
            batch_sig.append(sig_labels[sig])
            batch_snr.append(snr)
            if len(batch_iq) >= 2048:
                flush()
    flush()
    ctx.done(name, "train", writer, raw, removed, mod_labels)


def _radcom_hdf5_part_worker(
    keys: list[str],
    src_path: str,
    part_path: str,
    dataset_id: int,
    mod_labels: dict[str, int],
    sig_labels: dict[str, int],
    snr_min: float | None,
) -> tuple[str, int, int, dict[str, int]]:
    removed: Counter = Counter()
    raw = 0
    writer = _make_writer(
        Path(part_path),
        128,
        np.float32,
        dataset_id,
        0,
        Path(src_path),
    )
    batch_iq: list[np.ndarray] = []
    batch_mod: list[int] = []
    batch_sig: list[int] = []
    batch_snr: list[float] = []

    def flush() -> None:
        nonlocal batch_iq, batch_mod, batch_sig, batch_snr
        if not batch_iq:
            return
        block = np.stack(batch_iq, axis=0)
        writer.append_clean(
            block,
            removed,
            mod_label_id=np.asarray(batch_mod, dtype=np.int32),
            source_label_id=np.asarray(batch_sig, dtype=np.int32),
            snr=np.asarray(batch_snr, dtype=np.float32),
        )
        batch_iq, batch_mod, batch_sig, batch_snr = [], [], [], []

    with h5py.File(src_path, "r") as f:
        for key in keys:
            mod, sig, snr, _ = parse_radcom_key(key)
            if snr_min is not None and snr < snr_min:
                removed[_SNR_LT_MIN_REASON] += 1
                continue
            raw += 1
            batch_iq.append(radcom_waveform_to_iq(f[key][:]))
            batch_mod.append(mod_labels[mod])
            batch_sig.append(sig_labels[sig])
            batch_snr.append(snr)
            if len(batch_iq) >= 2048:
                flush()
    flush()
    writer.close()
    return part_path, raw, writer.count, dict(removed)


def _merge_h5_parts(parts: list[Path], out_path: Path, dataset_id: int, task_id: int, source: Path) -> int:
    if not parts:
        return 0
    with h5py.File(parts[0], "r") as first:
        length = int(first["iq"].shape[-1])
        dtype = first["iq"].dtype
    writer = _make_writer(out_path, length, dtype, dataset_id, task_id, source)
    total = 0
    for part in parts:
        with h5py.File(part, "r") as f:
            n = int(f["iq"].shape[0])
            if n == 0:
                continue
            idx = np.arange(n, dtype=np.int64)
            copy_records([part], idx, writer)
            total += n
    writer.close()
    return total


def radcom_hdf5_parallel(
    ctx: Context,
    name: str,
    dataset_id: int,
    path: Path,
    *,
    snr_min: float | None = UNIFIED_MIN_SNR_DB,
    jobs: int = 25,
) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"未找到 RadarComm HDF5: {path}")
    jobs = max(1, int(jobs))
    mod_labels, sig_labels = build_radcom_label_maps(path)
    ctx.maps["datasets"][str(dataset_id)] = name
    ctx.maps["modulations"][name] = mod_labels
    ctx.maps["sources"][name] = sig_labels

    with h5py.File(path, "r") as f:
        keys = list(f.keys())
    if snr_min is not None:
        keys = [k for k in keys if parse_radcom_key(k)[2] >= snr_min]

    parts_dir = ctx.h5 / f"{name}__parts"
    parts_dir.mkdir(parents=True, exist_ok=True)
    for old in parts_dir.glob("part_*.h5"):
        old.unlink(missing_ok=True)

    chunks: list[list[str]] = [[] for _ in range(min(jobs, len(keys)))]
    for i, key in enumerate(keys):
        chunks[i % len(chunks)].append(key)

    print(f"[parallel-radcom] {name} jobs={len(chunks)} keys={len(keys)}", flush=True)
    removed: Counter = Counter()
    raw = 0
    part_paths: list[Path] = []
    with ProcessPoolExecutor(max_workers=len(chunks)) as pool:
        futures = []
        for i, chunk in enumerate(chunks):
            if not chunk:
                continue
            part_path = parts_dir / f"part_{i:03d}.h5"
            futures.append(
                pool.submit(
                    _radcom_hdf5_part_worker,
                    chunk,
                    str(path),
                    str(part_path),
                    dataset_id,
                    mod_labels,
                    sig_labels,
                    snr_min,
                )
            )
        for fut in as_completed(futures):
            part_file, part_raw, part_kept, part_removed = fut.result()
            raw += part_raw
            removed.update(part_removed)
            if part_kept > 0:
                part_paths.append(Path(part_file))
            print(f"[parallel-radcom] part done kept={part_kept:,} raw={part_raw:,}", flush=True)

    part_paths.sort()
    out_train = ctx.h5 / f"{name}_train.h5"
    out_train.unlink(missing_ok=True)
    kept = _merge_h5_parts(part_paths, out_train, dataset_id, 0, path)
    for part in part_paths:
        part.unlink(missing_ok=True)
    if parts_dir.exists():
        parts_dir.rmdir()

    ctx.touched.add(name)
    entry = ctx.report["datasets"].setdefault(name, {"labels": mod_labels, "splits": {}})
    entry["labels"] = mod_labels
    entry["splits"]["train"] = {
        "raw": raw,
        "kept": kept,
        "removed": dict(removed),
        "file": out_train.name,
    }
    print(f"[parallel-radcom] {name} merged kept={kept:,} raw={raw:,}", flush=True)


def panoradio_hf(ctx: Context, dataset_id: int) -> None:
    name = "panoradio_hf"
    root = EXTERNAL / "panoradio_hf"
    npy_path = root / "dataset_panoradio_hf.npy"
    tags_path = root / "dataset_panoradio_hf_tags.csv"
    if not npy_path.is_file() or not tags_path.is_file():
        raise FileNotFoundError(f"缺少 Panoradio HF 数据: {root}")
    labels = {mode: idx for idx, mode in enumerate(PANORADIO_MODES)}
    ctx.maps["datasets"][str(dataset_id)] = name
    ctx.maps["modulations"][name] = labels

    modes: list[str] = []
    snrs: list[float] = []
    with tags_path.open(encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            mode = _csv_field(row, "mode")
            if mode not in labels:
                raise ValueError(f"未知调制类型 {mode!r}")
            modes.append(mode)
            snrs.append(float(_csv_field(row, "snr")))

    x = np.load(npy_path, mmap_mode="r")
    if len(x) != len(modes):
        raise ValueError(f"样本数不一致: npy={len(x)} tags={len(modes)}")
    y = np.asarray([labels[m] for m in modes], dtype=np.int32)
    snr_arr = np.asarray(snrs, dtype=np.float32)
    snr_ok = snr_keep_mask(snr_arr)
    snr_removed = Counter({_SNR_LT_MIN_REASON: int((~snr_ok).sum())})
    raw = int(len(x))
    x = x[snr_ok]
    y = y[snr_ok]
    snr_arr = snr_arr[snr_ok]

    target_length = _resolve_target_length(name, int(x.shape[-1]))
    removed = Counter()
    iq_parts: list[np.ndarray] = []
    label_parts: list[np.ndarray] = []
    snr_parts: list[np.ndarray] = []
    for start in range(0, len(x), 2048):
        end = min(start + 2048, len(x))
        block = _preprocess_canonical_iq(as_iq(np.asarray(x[start:end])).astype(np.float32, copy=False))
        block_y = y[start:end]
        keep, reasons = quality_mask(block, labels=block_y, dataset_name=name)
        removed.update(reasons)
        if not np.any(keep):
            continue
        iq_parts.append(_resize_canonical_iq(block[keep], target_length))
        label_parts.append(block_y[keep])
        snr_parts.append(snr_arr[start:end][keep])
    if not iq_parts:
        raise ValueError(f"{name} 清洗后无可用样本")
    iq_all = np.concatenate(iq_parts, axis=0)
    labels_all = np.concatenate(label_parts, axis=0)
    write_pretrain_stratified_splits(
        ctx,
        name,
        dataset_id,
        root,
        target_length,
        np.float32,
        iq_all,
        labels_all,
        labels,
        removed,
        raw,
        label_field="mod_label_id",
        extra_meta={
            "snr": np.concatenate(snr_parts, axis=0),
            "sample_rate_hz": _fs_meta_array(len(labels_all), name),
        },
        snr_removed=dict(snr_removed),
    )

def cjr_mix(ctx: Context, dataset_id: int) -> None:
    """CJR-mix：train parquet → 分层 train/test/val H5。"""
    from resmamba_signal_model.data.cjr_mix import (
        list_parquet_files,
        normalize_iq_array,
        parse_iq_array,
        require_pyarrow,
        resolve_cjr_mix_root,
    )

    import pyarrow.parquet as pq

    require_pyarrow()
    name = "cjr_mix"
    data_root = resolve_cjr_mix_root(ctx.output)
    ctx.maps["datasets"][str(dataset_id)] = name

    removed = Counter()
    raw = 0
    label_ids: set[int] = set()
    iq_parts: list[np.ndarray] = []
    label_parts: list[np.ndarray] = []
    snr_parts: list[np.ndarray] = []
    fs_parts: list[np.ndarray] = []
    native_length: int | None = None
    use_canonical = bool(_active_canonical_config() and _active_canonical_config().enabled)

    for path in list_parquet_files(data_root, "train"):
        pf = pq.ParquetFile(path)
        for rg in range(pf.num_row_groups):
            table = pf.read_row_group(rg, columns=["iq", "infer_class", "snr", "fs"])
            infer = table.column("infer_class").to_numpy(zero_copy_only=False)
            snrs = table.column("snr").to_numpy(zero_copy_only=False)
            fs_col = table.column("fs").to_numpy(zero_copy_only=False)
            iq_col = table.column("iq")
            batch_iq: list[np.ndarray] = []
            batch_mod: list[int] = []
            batch_snr: list[float] = []
            batch_fs: list[float] = []

            def flush_batch() -> None:
                nonlocal batch_iq, batch_mod, batch_snr, batch_fs, native_length
                if not batch_iq:
                    return
                block = np.stack(batch_iq, axis=0)
                block = _preprocess_canonical_iq(block)
                mod_labels = np.asarray(batch_mod, dtype=np.int32)
                keep, reasons = quality_mask(block, labels=mod_labels, dataset_name=name)
                snr_arr = np.asarray(batch_snr, dtype=np.float32)
                snr_ok = snr_keep_mask(snr_arr)
                reasons[_SNR_LT_MIN_REASON] = int((~snr_ok).sum())
                keep = keep & snr_ok
                removed.update(reasons)
                if not np.any(keep):
                    batch_iq, batch_mod, batch_snr, batch_fs = [], [], [], []
                    return
                if native_length is None:
                    native_length = int(block.shape[-1])
                target = _resolve_target_length(name, native_length)
                iq_parts.append(_resize_canonical_iq(block[keep], target))
                label_parts.append(mod_labels[keep])
                snr_parts.append(np.asarray(batch_snr, dtype=np.float32)[keep])
                fs_parts.append(np.asarray(batch_fs, dtype=np.float32)[keep])
                batch_iq, batch_mod, batch_snr, batch_fs = [], [], [], []

            for i in range(table.num_rows):
                raw += 1
                iq = parse_iq_array(iq_col[i].as_py())
                if not use_canonical:
                    iq = normalize_iq_array(iq, "abs")
                label = int(infer[i])
                label_ids.add(label)
                batch_iq.append(iq)
                batch_mod.append(label)
                batch_snr.append(float(snrs[i]))
                batch_fs.append(float(fs_col[i]))
                if len(batch_iq) >= 256:
                    flush_batch()
            flush_batch()

    if not iq_parts:
        raise RuntimeError("CJR-mix train 为空")
    labels = {str(v): v for v in sorted(label_ids)}
    ctx.maps["modulations"][name] = labels
    iq_all = np.concatenate(iq_parts, axis=0)
    labels_all = np.concatenate(label_parts, axis=0)
    target_length = _resolve_target_length(name, int(native_length or iq_all.shape[-1]))
    writer_attrs = {"source_path": str(data_root / "train")}
    if _PRETRAIN_H5_CFG is None:
        writer_attrs["scale_policy"] = "abs_max_clip5_at_convert"
    write_pretrain_stratified_splits(
        ctx,
        name,
        dataset_id,
        data_root / "train",
        target_length,
        np.float32,
        iq_all,
        labels_all,
        labels,
        removed,
        raw,
        label_field="mod_label_id",
        extra_meta={
            "source_label_id": labels_all,
            "snr": np.concatenate(snr_parts, axis=0),
            "sample_rate_hz": np.concatenate(fs_parts, axis=0),
        },
        writer_attrs=writer_attrs,
    )

def radar_mod15(ctx: Context, dataset_id: int) -> None:
    """open_realData round7 CSV → RFData H5；分层 train/test/val 7:2:1。"""
    name = "radar_mod15"
    root = NON_EMITTER / "open_realData" / "outputv2" / "round7_dataset"
    summary = json.loads((root / "manifests" / "summary.json").read_text(encoding="utf-8"))
    labels = {
        info["source_label"]: int(cls_str)
        for cls_str, info in summary["mapping"].items()
    }
    ctx.maps["datasets"][str(dataset_id)] = name
    ctx.maps["sources"][name] = labels
    ctx.maps["modulations"][name] = labels

    for suffix in ("train", "val", "test"):
        (ctx.h5 / f"{name}_{suffix}.h5").unlink(missing_ok=True)

    class_dirs = sorted(
        (p for p in root.iterdir() if p.is_dir() and p.name.isdigit()),
        key=lambda p: int(p.name),
    )
    if not class_dirs:
        raise FileNotFoundError(f"{root} 下未找到类别目录")
    sample_amp = np.loadtxt(next(class_dirs[0].glob("*.csv")), delimiter=",", skiprows=1, usecols=1)
    native_length = int(len(sample_amp))
    target_length = _resolve_target_length(name, native_length)

    iq_parts: list[np.ndarray] = []
    label_parts: list[np.ndarray] = []
    fs_parts: list[np.ndarray] = []
    removed: Counter = Counter()
    raw = 0

    for cls_dir in class_dirs:
        cls_id = int(cls_dir.name)
        for csv_path in sorted(cls_dir.glob("*.csv")):
            raw += 1
            amp = np.loadtxt(csv_path, delimiter=",", skiprows=1, usecols=1, dtype=np.float64)
            if len(amp) != native_length:
                removed["malformed"] += 1
                continue
            iq_raw = as_iq(amp).astype(np.float32)[None]
            keep, reasons = quality_mask(
                _iq_at_quality_stage(iq_raw, target_length),
                labels=np.asarray([cls_id], dtype=np.int32),
                dataset_name=name,
            )
            removed.update(reasons)
            if bool(keep[0]):
                iq_parts.append(_resize_canonical_iq(_preprocess_canonical_iq(iq_raw), target_length)[0])
                label_parts.append(cls_id)
                fs_parts.append(infer_radar_mod15_csv_fs(csv_path))

    if not iq_parts:
        raise ValueError(f"{name} 清洗后无可用样本")

    iq_all = np.stack(iq_parts, axis=0)
    labels_all = np.asarray(label_parts, dtype=np.int32)
    fs_all = np.asarray(fs_parts, dtype=np.float32)
    write_pretrain_stratified_splits(
        ctx,
        name,
        dataset_id,
        root,
        target_length,
        np.float32,
        iq_all,
        labels_all,
        labels,
        removed,
        raw,
        label_field="source_label_id",
        extra_meta={
            "mod_label_id": labels_all,
            "snr": np.full(len(labels_all), RADAR_MOD15_NOMINAL_SNR_DB, dtype=np.float32),
            "sample_rate_hz": fs_all,
        },
        writer_attrs={
            "snr_policy": "nominal_10dB",
            "snr_db_nominal": RADAR_MOD15_NOMINAL_SNR_DB,
        },
    )

def resolve_radchar_source(source: str | Path | None = None) -> Path:
    candidates: list[Path] = []
    if source is not None:
        candidates.append(Path(source))
    env = os.environ.get("RADCHAR_H5") or os.environ.get("RADCHAR_SOURCE")
    if env:
        candidates.append(Path(env))
    candidates.extend(
        (
            RADCHAR_DEFAULT_SOURCE,
            EXTERNAL / "radchar" / "RadChar-Small.h5",
            ROOT / "dataset" / "external" / "radchar" / "RadChar-Small.h5",
        )
    )
    for path in candidates:
        if path.is_file():
            return path.resolve()
    raise FileNotFoundError(
        "未找到 RadChar-Small.h5。请放到 /root/autodl-tmp/RadChar-Small.h5，"
        "或设置环境变量 RADCHAR_H5。"
    )


def radchar(ctx: Context, dataset_id: int, source: str | Path | None = None) -> None:
    """RadChar-Small：5 类雷达脉内调制 → 分层 train/test/val 7:2:1。"""
    name = "radchar"
    src = resolve_radchar_source(source)
    labels = {str(kind): int(idx) for idx, kind in enumerate(RADCHAR_SIGNAL_TYPE_NAMES)}
    ctx.maps["datasets"][str(dataset_id)] = name
    ctx.maps["modulations"][name] = labels
    ctx.maps["sources"][name] = labels

    for suffix in ("train", "val", "test"):
        (ctx.h5 / f"{name}_{suffix}.h5").unlink(missing_ok=True)

    with h5py.File(src, "r") as handle:
        n_total = int(handle["iq"].shape[0])
        native_length = int(handle["iq"].shape[1])
        lab = handle["labels"][:]
    target_length = _resolve_target_length(name, native_length)
    signal_type = np.asarray(lab["signal_type"], dtype=np.int32)
    snr = np.asarray(lab["signal_to_noise_ratio"], dtype=np.float32)
    if int(signal_type.min()) < 0 or int(signal_type.max()) >= len(RADCHAR_SIGNAL_TYPE_NAMES):
        raise ValueError(f"{src} 的 signal_type 超出 0..{len(RADCHAR_SIGNAL_TYPE_NAMES) - 1}")

    removed: Counter = Counter()
    iq_parts: list[np.ndarray] = []
    label_parts: list[np.ndarray] = []
    snr_parts: list[np.ndarray] = []
    chunk = 4096
    with h5py.File(src, "r") as handle:
        for start in range(0, n_total, chunk):
            end = min(start + chunk, n_total)
            iq_raw = as_iq(handle["iq"][start:end]).astype(np.float32, copy=False)
            y = signal_type[start:end]
            s = snr[start:end]
            iq_prep = _preprocess_canonical_iq(iq_raw)
            keep, reasons = quality_mask(
                _iq_at_quality_stage(iq_raw, target_length),
                labels=y,
                dataset_name=name,
            )
            snr_ok = snr_keep_mask(s)
            reasons[_SNR_LT_MIN_REASON] = int((keep & ~snr_ok).sum())
            keep &= snr_ok
            removed.update(reasons)
            if not np.any(keep):
                continue
            kept = _resize_canonical_iq(iq_prep[keep], target_length)
            iq_parts.append(kept)
            label_parts.append(y[keep])
            snr_parts.append(s[keep])

    if not iq_parts:
        raise ValueError(f"{name} 清洗后无可用样本")

    iq_all = np.concatenate(iq_parts, axis=0)
    labels_all = np.concatenate(label_parts, axis=0)
    snr_all = np.concatenate(snr_parts, axis=0)
    write_pretrain_stratified_splits(
        ctx,
        name,
        dataset_id,
        src,
        target_length,
        np.float32,
        iq_all,
        labels_all,
        labels,
        removed,
        n_total,
        label_field="mod_label_id",
        extra_meta={
            "source_label_id": labels_all,
            "snr": snr_all,
        },
        writer_attrs={
            "sampling_rate_hz": RADCHAR_SAMPLE_RATE_HZ,
            "radchar_signal_types": json.dumps(list(RADCHAR_SIGNAL_TYPE_NAMES)),
        },
    )
    ctx.report["datasets"][name]["source_path"] = str(src)

def rebuild_pools_from_h5(ctx: Context) -> None:
    ctx.pool = defaultdict(list)
    ctx.final_label_fields = {}
    if not ctx.h5.exists():
        return
    for path in sorted(ctx.h5.glob("*.h5")):
        if "__balanced_" in path.name:
            continue
        with h5py.File(path, "r") as f:
            task_id = _h5_task_id(f)
            label_field = choose_label_field([path])
            if "dataset_id" in f and f["dataset_id"].shape[0] > 0:
                dataset_id = int(f["dataset_id"][0])
            elif "dataset_id" in f.attrs:
                dataset_id = int(f.attrs["dataset_id"])
            else:
                dataset_id = -1
            if dataset_id >= 0:
                stem = path.name.rsplit("_", 1)[0]
                if stem:
                    ctx.maps.setdefault("datasets", {})[str(dataset_id)] = stem
        ctx.final_label_fields[path.name] = label_field
        if path.name.endswith("_train.h5"):
            ctx.pool["train"].append((path.name, task_id))
        elif path.name.endswith("_val.h5"):
            ctx.pool["val"].append((path.name, task_id))
        elif path.name.endswith("_test.h5"):
            ctx.pool["test"].append((path.name, task_id))


def pool_files(
    split_files: list[tuple[str, int]],
    label_fields: dict[str, str],
    *,
    task_id: int | None = None,
    label_field: str | None = None,
) -> list[str]:
    out: list[str] = []
    for filename, tid in split_files:
        if task_id is not None and tid != task_id:
            continue
        if label_field is not None and label_fields.get(filename) != label_field:
            continue
        out.append(filename)
    return sorted(out)


def _read_label_array(f: h5py.File, key: str, length: int) -> np.ndarray:
    if key in f:
        return np.asarray(f[key][:])
    return np.full(length, -1, dtype=np.int32)


def _write_int_column(f: h5py.File, key: str, values: np.ndarray) -> None:
    values = np.asarray(values, dtype=np.int32)
    n = int(values.shape[0])
    if key in f and tuple(f[key].shape) == (n,):
        f[key][:] = values
        return
    if key in f:
        del f[key]
    f.create_dataset(
        key,
        data=values,
        maxshape=(None,),
        chunks=(max(1, min(max(n, 1), 4096)),),
        fillvalue=-1,
    )


def _h5_paths_for_datasets(ctx: Context, only_datasets: set[str] | None = None) -> list[Path]:
    if not ctx.h5.exists():
        return []
    paths: list[Path] = []
    for path in sorted(ctx.h5.glob("*.h5")):
        if "__balanced_" in path.name:
            continue
        if only_datasets is not None:
            try:
                if h5_dataset_name(path.name) not in only_datasets:
                    continue
            except ValueError:
                continue
        paths.append(path)
    return paths


def stamp_semantic_namespaces(
    ctx: Context,
    *,
    only_datasets: set[str] | None = None,
) -> dict[str, dict[str, int]]:
    """写入 canonical modulation 与 namespaced emitter 的连续全局 ID。"""
    ontology = build_modulation_ontology(
        ctx.maps.get("modulations", {}),
        existing=ctx.maps.get("modulation_ontology"),
    )
    emitter_namespace = build_emitter_namespace(
        ctx.maps.get("emitters", {}),
        existing=ctx.maps.get("emitter_namespace"),
    )
    ctx.maps["modulation_ontology"] = ontology.to_dict()
    ctx.maps["canonical_modulations"] = dict(ontology.canonical_to_id)
    ctx.maps["emitter_namespace"] = emitter_namespace.to_dict()
    dataset_names = {
        int(dataset_id): str(dataset_name)
        for dataset_id, dataset_name in ctx.maps.get("datasets", {}).items()
    }
    canonical_stats: dict[str, int] = {}
    emitter_stats: dict[str, int] = {}
    if not ctx.h5.exists():
        return {"canonical_mod_label_id": canonical_stats, "global_emitter_id": emitter_stats}
    for path in _h5_paths_for_datasets(ctx, only_datasets):
        with h5py.File(path, "r+") as f:
            n = int(f["iq"].shape[0])
            schema_v2 = _is_h5_v2(f)
            if schema_v2:
                dataset_id = int(f.attrs.get("dataset_id", -1))
                dataset_ids = np.full(n, dataset_id, dtype=np.int32)
            else:
                dataset_ids = _read_label_array(f, "dataset_id", n)
            local_mod = _read_label_array(f, "mod_label_id", n)
            local_emitter = _read_label_array(f, "emitter_id", n)
            canonical = np.full(n, -1, dtype=np.int32)
            global_emitter = np.full(n, -1, dtype=np.int32)
            for dataset_id in np.unique(dataset_ids):
                dataset_name = dataset_names.get(int(dataset_id))
                if dataset_name is None:
                    continue
                choose = dataset_ids == dataset_id
                canonical[choose] = ontology.map_local(dataset_name, local_mod[choose])
                if not schema_v2:
                    global_emitter[choose] = emitter_namespace.map_local(
                        dataset_name,
                        local_emitter[choose],
                    )
            _write_int_column(f, "canonical_mod_label_id", canonical)
            if not schema_v2:
                _write_int_column(f, "global_emitter_id", global_emitter)
            canonical_stats[path.name] = int(np.sum(canonical >= 0))
            if not schema_v2:
                emitter_stats[path.name] = int(np.sum(global_emitter >= 0))
            f.attrs["modulation_ontology_version"] = ontology.version
            if not schema_v2:
                f.attrs["emitter_namespace_version"] = emitter_namespace.version
    return {
        "canonical_mod_label_id": canonical_stats,
        "global_emitter_id": emitter_stats,
    }


def stamp_global_label_ids(
    ctx: Context,
    *,
    only_datasets: set[str] | None = None,
) -> dict[str, int]:
    """为每个 H5 样本写入跨数据集唯一的 global_label_id（用于 clustering）。"""
    if not ctx.h5.exists():
        return {}
    stats: dict[str, int] = {}
    for path in _h5_paths_for_datasets(ctx, only_datasets):
        with h5py.File(path, "r+") as f:
            n = int(f["iq"].shape[0])
            if _is_h5_v2(f):
                dataset_id = np.full(n, int(f.attrs.get("dataset_id", -1)), dtype=np.int32)
            else:
                dataset_id = np.asarray(f["dataset_id"][:])
            mod_label_id = _read_label_array(f, "mod_label_id", n)
            emitter_id = _read_label_array(f, "emitter_id", n)
            source_label_id = _read_label_array(f, "source_label_id", n)
            global_id = global_cluster_labels(dataset_id, mod_label_id, emitter_id, source_label_id)
            if "global_label_id" in f:
                f["global_label_id"].resize((n,))
                f["global_label_id"][:] = global_id
            else:
                f.create_dataset(
                    "global_label_id",
                    data=global_id,
                    maxshape=(None,),
                    chunks=(max(1024, min(n, 4096)),),
                    fillvalue=-1,
                )
            stats[path.name] = int(np.sum(global_id >= 0))
    ctx.maps["global_label_namespace"] = int(GLOBAL_LABEL_NAMESPACE)
    return stats


def stamp_mod_label_ids(ctx: Context, dataset_names: list[str] | None = None) -> dict[str, int]:
    """将 source_label_id 镜像写入 mod_label_id（用于辐射源数据集参与调制下游）。"""
    if dataset_names is None:
        dataset_names = load_downstream_modulation_extra_datasets(
            config_path=ROOT / "configs" / "datasets.yaml"
        )
    allowed = set(dataset_names)
    stats: dict[str, int] = {}
    if not ctx.h5.exists():
        return stats
    for path in sorted(ctx.h5.glob("*.h5")):
        if "__balanced_" in path.name or h5_dataset_name(path.name) not in allowed:
            continue
        with h5py.File(path, "r+") as f:
            if "source_label_id" not in f:
                continue
            source = np.asarray(f["source_label_id"][:], dtype=np.int32)
            valid = source >= 0
            if not np.any(valid):
                continue
            n = int(source.shape[0])
            mod = np.full(n, -1, dtype=np.int32)
            if "mod_label_id" in f:
                mod = np.asarray(f["mod_label_id"][:], dtype=np.int32)
            mod[valid] = source[valid]
            if "mod_label_id" in f:
                f["mod_label_id"].resize((n,))
                f["mod_label_id"][:] = mod
            else:
                f.create_dataset(
                    "mod_label_id",
                    data=mod,
                    maxshape=(None,),
                    chunks=(max(1024, min(n, 4096)),),
                    fillvalue=-1,
                )
            stats[path.name] = int(np.sum(valid))
    for name in allowed:
        source_labels = ctx.maps.get("sources", {}).get(name)
        if source_labels:
            ctx.maps.setdefault("modulations", {})[name] = dict(source_labels)
    return stats


def pool_files_with_global_labels(ctx: Context, split_files: list[tuple[str, int]]) -> list[str]:
    """clustering pool：保留可产生聚类标签的 H5（已写入 global_label_id 或含 mod/emitter/source 标签）。"""
    out: list[str] = []
    for filename, _tid in split_files:
        path = ctx.h5 / filename
        if not path.is_file():
            continue
        with h5py.File(path, "r") as f:
            n = int(f["iq"].shape[0])
            if "global_label_id" in f and np.any(np.asarray(f["global_label_id"][:]) >= 0):
                out.append(filename)
                continue
            for key in ("mod_label_id", "emitter_id", "source_label_id"):
                if key in f and np.any(np.asarray(f[key][:]) >= 0):
                    out.append(filename)
                    break
    return sorted(out)


def _holdout_indices(
    n: int,
    labels: np.ndarray | None,
    frac: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """从 n 条样本中切 ``frac`` 到 test；有类别则分层，且每类至少留 1 条在 train。"""
    if n < 2:
        return np.array([], dtype=np.int64)
    frac = float(frac)

    def _random_holdout() -> np.ndarray:
        n_test = int(round(frac * n))
        n_test = min(max(n_test, 1), n - 1)
        return np.sort(rng.choice(n, size=n_test, replace=False).astype(np.int64))

    if labels is None:
        return _random_holdout()
    labels = np.asarray(labels).reshape(-1)
    if labels.shape[0] != n or not np.any(labels >= 0):
        return _random_holdout()
    chosen: list[int] = []
    for cls in np.unique(labels):
        if int(cls) < 0:
            continue
        idx = np.flatnonzero(labels == cls)
        rng.shuffle(idx)
        if idx.size < 2:
            continue
        k = int(round(frac * int(idx.size)))
        k = min(max(k, 0), int(idx.size) - 1)
        if k > 0:
            chosen.extend(int(x) for x in idx[:k].tolist())
    unlabeled = np.flatnonzero(labels < 0)
    if unlabeled.size >= 2:
        rng.shuffle(unlabeled)
        k = int(round(frac * int(unlabeled.size)))
        k = min(max(k, 0), int(unlabeled.size) - 1)
        if k > 0:
            chosen.extend(int(x) for x in unlabeled[:k].tolist())
    if not chosen:
        return _random_holdout()
    return np.sort(np.unique(np.asarray(chosen, dtype=np.int64)))


def _subset_h5_rows(src: Path, dst: Path, keep: np.ndarray) -> None:
    """按行号复制 H5 中所有长度为 N 的列，保留 attrs 与额外字段。"""
    keep = np.asarray(keep, dtype=np.int64)
    if keep.size <= 0:
        raise ValueError(f"{src.name} keep 为空")
    dst = Path(dst)
    if dst.exists():
        dst.unlink()
    dst.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(src, "r") as fin, h5py.File(dst, "w") as fout:
        n = int(fin["iq"].shape[0])
        if int(keep.min()) < 0 or int(keep.max()) >= n:
            raise ValueError(f"{src.name} keep 越界")
        for attr_key, attr_val in fin.attrs.items():
            fout.attrs[attr_key] = attr_val
        fout.attrs["sample_count"] = int(keep.shape[0])
        for key, obj in fin.items():
            if not isinstance(obj, h5py.Dataset):
                continue
            kwargs: dict[str, Any] = {}
            if obj.compression:
                kwargs["compression"] = obj.compression
                if obj.compression_opts is not None:
                    kwargs["compression_opts"] = obj.compression_opts
            if obj.shape and int(obj.shape[0]) == n:
                fout.create_dataset(key, data=obj[keep], **kwargs)
            else:
                fout.create_dataset(key, data=obj[()], **kwargs)


def apply_unified_min_snr_to_h5(
    ctx: Context,
    *,
    min_db: float = UNIFIED_MIN_SNR_DB,
    only_datasets: set[str] | None = None,
    only_splits: set[str] | None = None,
) -> dict[str, dict[str, int]]:
    """对现有 H5 就地过滤：只保留有限且 ``snr >= min_db`` 的行。

    - ``only_datasets``：按 stem 白名单（如 ``rml2016_10a``）
    - ``only_splits``：仅处理 ``train`` / ``val`` / ``test`` 后缀
    - 无 ``snr`` 列、或 snr 全非有限：跳过（不删样本）
    - 全部已满足则不动。完成后调用方应 ``refresh_split_manifest``。
    """
    stats: dict[str, dict[str, int]] = {}
    if not ctx.h5.exists():
        return stats
    split_allow = {str(s).lower() for s in only_splits} if only_splits else None
    for path in sorted(ctx.h5.glob("*.h5")):
        if "__balanced_" in path.name or "__tmp" in path.name or "__parts" in path.name:
            continue
        if "_" not in path.name or not path.name.endswith(".h5"):
            continue
        stem, split = path.name[: -len(".h5")].rsplit("_", 1)
        if only_datasets is not None and stem not in only_datasets:
            continue
        if split_allow is not None and split.lower() not in split_allow:
            continue
        with h5py.File(path, "r") as f:
            if "snr" not in f or "iq" not in f:
                stats[path.name] = {"before": -1, "after": -1, "dropped": 0, "skipped": "no_snr"}
                continue
            n = int(f["iq"].shape[0])
            snr = np.asarray(f["snr"][:], dtype=np.float64).reshape(-1)
            if snr.shape[0] != n:
                raise ValueError(f"{path.name}: snr 长度 {snr.shape[0]} != iq {n}")
            if not np.isfinite(snr).any():
                stats[path.name] = {"before": n, "after": n, "dropped": 0, "skipped": "no_finite_snr"}
                print(f"[apply-min-snr] {path.name}: skip (no finite snr), n={n}", flush=True)
                continue
            mask = snr_keep_mask(snr, min_db=min_db)
            kept = int(mask.sum())
            dropped = n - kept
            if dropped <= 0:
                stats[path.name] = {"before": n, "after": n, "dropped": 0}
                print(f"[apply-min-snr] {path.name}: already ok n={n} min_db={min_db:g}", flush=True)
                continue
            if kept <= 0:
                raise ValueError(f"{path.name}: SNR>={min_db:g} 后无样本")
            keep_idx = np.flatnonzero(mask).astype(np.int64)
        tmp = path.with_name(path.name + ".__snr_tmp.h5")
        try:
            _subset_h5_rows(path, tmp, keep_idx)
            with h5py.File(tmp, "r+") as fout:
                fout.attrs["snr_policy"] = f"unified_min_{min_db:g}dB"
                fout.attrs["unified_min_snr_db"] = float(min_db)
            tmp.replace(path)
        finally:
            tmp.unlink(missing_ok=True)
        stats[path.name] = {"before": n, "after": kept, "dropped": dropped}
        print(
            f"[apply-min-snr] {path.name}: {n} -> {kept} (drop {dropped}, min_db={min_db:g})",
            flush=True,
        )
    policy = ctx.report.setdefault("policy", {})
    policy["rml_snr"] = f"SNR >= {min_db:g} dB (unified)"
    policy["unified_min_snr_db"] = float(min_db)
    policy["snr_policy"] = (
        f"all datasets with finite snr: keep snr >= {min_db:g} dB; "
        f"radar_mod15 nominal {RADAR_MOD15_NOMINAL_SNR_DB:g} dB"
    )
    policy["rml2018_eval_snr"] = f"val/test only: SNR >= {min_db:g} dB"
    ctx.report["apply_min_snr"] = {
        "min_db": float(min_db),
        "only_splits": sorted(split_allow) if split_allow else None,
        "files": stats,
    }
    return stats


def ensure_missing_test_splits(
    ctx: Context,
    *,
    seed: int = MISSING_TEST_SPLIT_SEED,
    frac: float = MISSING_TEST_FRACTION,
    only_datasets: set[str] | None = None,
) -> dict[str, int]:
    """只处理「有 train、无 test」；不动已有 test/val（WiSig group-held-out 保持原样）。

    按类别从 ``*_train`` 切 ``frac``（默认 20%）写 ``*_test.h5``，并从 train 删除这些行。
    无可用类别字段则随机切。固定 seed。
    """
    stats: dict[str, int] = {}
    if not ctx.h5.exists():
        return stats
    for train_path in sorted(ctx.h5.glob("*_train.h5")):
        if "__balanced_" in train_path.name:
            continue
        base = train_path.name[: -len("_train.h5")]
        if only_datasets is not None and base not in only_datasets:
            continue
        test_path = ctx.h5 / f"{base}_test.h5"
        val_path = ctx.h5 / f"{base}_val.h5"
        if test_path.is_file():
            continue
        with h5py.File(train_path, "r") as f:
            n = int(f["iq"].shape[0])
            if n < 2:
                continue
            labels = None
            for key in _MISSING_TEST_LABEL_FIELDS:
                if key not in f:
                    continue
                col = np.asarray(f[key][:], dtype=np.int64)
                if np.any(col >= 0):
                    labels = col
                    break
        ds_rng = np.random.default_rng(int(seed) + (sum(base.encode()) & 0xFFFFFFFF))
        test_idx = _holdout_indices(n, labels, frac, ds_rng)
        keep_train = np.setdiff1d(np.arange(n, dtype=np.int64), test_idx, assume_unique=False)
        if test_idx.size <= 0 or keep_train.size <= 0:
            continue
        tmp_test = ctx.h5 / f"{base}_test.__tmp.h5"
        tmp_train = ctx.h5 / f"{base}_train.__tmp.h5"
        try:
            _subset_h5_rows(train_path, tmp_test, test_idx)
            _subset_h5_rows(train_path, tmp_train, keep_train)
            tmp_test.replace(test_path)
            tmp_train.replace(train_path)
        finally:
            tmp_test.unlink(missing_ok=True)
            tmp_train.unlink(missing_ok=True)
        stats[base] = int(test_idx.shape[0])
        entry = ctx.report.get("datasets", {}).get(base)
        if isinstance(entry, dict):
            splits = entry.setdefault("splits", {})
            splits["test"] = {
                "kept": int(test_idx.size),
                "file": test_path.name,
                "carved_from_train": True,
            }
            if isinstance(splits.get("train"), dict):
                splits["train"]["kept"] = int(keep_train.size)
            if val_path.is_file() and "val" not in splits:
                splits["val"] = {"file": val_path.name}
            entry["missing_test_carve"] = {
                "fraction": float(frac),
                "seed": int(seed),
                "test_count": int(test_idx.size),
                "train_remaining": int(keep_train.size),
            }
    return stats


def finalize_task_pools(ctx: Context) -> None:
    """构建 task_pools。

    H5 后缀约定：
    - *_train.h5：MAE 预训练（pretrain_train）
    - *_val.h5：各阶段验证（早停 / 选模 / 报告指标）
    - *_test.h5：阶段二/三有标签训练（downstream_*_train / clustering_train）
    白名单见 configs/datasets.yaml（pretrain / downstream_* / clustering_* / prediction）。
    """
    rebuild_pools_from_h5(ctx)
    train = sorted(ctx.pool["train"])
    val = sorted(ctx.pool["val"])
    test = sorted(ctx.pool["test"])
    fields = ctx.final_label_fields
    pf = lambda split, **kw: pool_files(split, fields, **kw)
    config_path = ROOT / "configs" / "datasets.yaml"
    pretrain_datasets = load_pretrain_datasets(config_path=config_path)
    radar_mod_datasets = load_downstream_radar_modulation_datasets(config_path=config_path)
    radar_model_datasets = load_downstream_radar_model_datasets(config_path=config_path)
    comm_mod_datasets = load_downstream_comm_modulation_datasets(config_path=config_path)
    clustering_radar_datasets = load_clustering_radar_datasets(config_path=config_path)
    clustering_comm_datasets = load_clustering_comm_datasets(config_path=config_path)
    prediction_datasets = load_prediction_datasets(config_path=config_path)
    excluded = load_excluded_datasets(config_path=config_path)

    def _pool(split_files: list[tuple[str, int]], **kwargs: object) -> list[str]:
        return filter_excluded_dataset_pool(pf(split_files, **kwargs), excluded)

    def _whitelist_pool(
        split_files: list[tuple[str, int]],
        allowed: list[str],
        **kwargs: object,
    ) -> list[str]:
        return filter_dataset_pool(_pool(split_files, **kwargs), allowed)

    radar_mod_train = _whitelist_pool(
        test, radar_mod_datasets, task_id=0, label_field="mod_label_id"
    )
    radar_mod_val = _whitelist_pool(
        val, radar_mod_datasets, task_id=0, label_field="mod_label_id"
    )
    radar_model_train = _whitelist_pool(
        test, radar_model_datasets, task_id=0, label_field="mod_label_id"
    )
    radar_model_val = _whitelist_pool(
        val, radar_model_datasets, task_id=0, label_field="mod_label_id"
    )
    comm_mod_train = _whitelist_pool(
        test, comm_mod_datasets, task_id=0, label_field="mod_label_id"
    )
    comm_mod_val = _whitelist_pool(
        val, comm_mod_datasets, task_id=0, label_field="mod_label_id"
    )
    prediction_train = _whitelist_pool(test, prediction_datasets)
    prediction_val = _whitelist_pool(val, prediction_datasets)
    clustering_radar_train = pool_files_with_global_labels(
        ctx,
        [
            (filename, tid)
            for filename, tid in test
            if filename in set(_whitelist_pool(test, clustering_radar_datasets))
        ],
    )
    clustering_radar_val = pool_files_with_global_labels(
        ctx,
        [
            (filename, tid)
            for filename, tid in val
            if filename in set(_whitelist_pool(val, clustering_radar_datasets))
        ],
    )
    clustering_comm_train = pool_files_with_global_labels(
        ctx,
        [
            (filename, tid)
            for filename, tid in test
            if filename in set(_whitelist_pool(test, clustering_comm_datasets))
        ],
    )
    clustering_comm_val = pool_files_with_global_labels(
        ctx,
        [
            (filename, tid)
            for filename, tid in val
            if filename in set(_whitelist_pool(val, clustering_comm_datasets))
        ],
    )
    clustering_train = sorted(set(clustering_radar_train) | set(clustering_comm_train))
    clustering_val = sorted(set(clustering_radar_val) | set(clustering_comm_val))

    ctx.maps["task_pools"] = {
        "pretrain_train": _whitelist_pool(train, pretrain_datasets),
        "pretrain_val": _whitelist_pool(val, pretrain_datasets),
        "downstream_radar_modulation_train": radar_mod_train,
        "downstream_radar_modulation_val": radar_mod_val,
        "downstream_radar_model_train": radar_model_train,
        "downstream_radar_model_val": radar_model_val,
        "downstream_comm_modulation_train": comm_mod_train,
        "downstream_comm_modulation_val": comm_mod_val,
        # 兼容旧 pool 名（通信调制）
        "downstream_modulation_train": comm_mod_train,
        "downstream_modulation_val": comm_mod_val,
        "downstream_emitter_train": [],
        "downstream_emitter_val": [],
        "downstream_source_train": _whitelist_pool(
            test, comm_mod_datasets, task_id=0, label_field="source_label_id"
        ),
        "downstream_source_val": _whitelist_pool(
            val, comm_mod_datasets, task_id=0, label_field="source_label_id"
        ),
        "downstream_prediction_train": prediction_train,
        "downstream_prediction_val": prediction_val,
        "clustering_radar_train": clustering_radar_train,
        "clustering_radar_val": clustering_radar_val,
        "clustering_comm_train": clustering_comm_train,
        "clustering_comm_val": clustering_comm_val,
        "clustering_train": clustering_train,
        "clustering_val": clustering_val,
    }
    ctx.maps["pretrain_datasets"] = list(pretrain_datasets)
    ctx.maps["downstream_radar_modulation_datasets"] = list(radar_mod_datasets)
    ctx.maps["downstream_radar_model_datasets"] = list(radar_model_datasets)
    ctx.maps["downstream_comm_modulation_datasets"] = list(comm_mod_datasets)
    ctx.maps["downstream_modulation_datasets"] = list(comm_mod_datasets)
    ctx.maps["emitter_downstream_datasets"] = []
    ctx.maps["clustering_radar_datasets"] = list(clustering_radar_datasets)
    ctx.maps["clustering_comm_datasets"] = list(clustering_comm_datasets)
    ctx.maps["prediction_datasets"] = list(prediction_datasets)
    ctx.maps["downstream_shared_datasets"] = sorted(
        set(radar_model_datasets) | set(comm_mod_datasets) | set(clustering_radar_datasets) | set(clustering_comm_datasets)
    )
    ctx.maps["excluded_datasets"] = list(excluded)


def claimed_group_splits(ctx: Context) -> dict[str, str]:
    claims: dict[str, str] = {}
    for name, entry in ctx.report.get("datasets", {}).items():
        provenance = entry.get("split_provenance", {})
        if provenance.get("claim_group_held_out"):
            claims[str(name)] = str(provenance.get("strategy", "group_held_out"))
    return claims


def finalize_split_manifest(ctx: Context) -> dict:
    payload = build_split_manifest(
        ctx.h5,
        group_split_claims=claimed_group_splits(ctx),
    )
    digest = write_immutable_manifest(ctx.output / "split_manifest.json", payload)
    for name, manifest_entry in payload.get("datasets", {}).items():
        report_entry = ctx.report.get("datasets", {}).get(name)
        if report_entry is None:
            continue
        provenance = report_entry.get("split_provenance")
        if provenance is not None:
            provenance["verification"] = manifest_entry["group_integrity"]["status"]
    ctx.maps["split_manifest"] = {
        "file": "split_manifest.json",
        "sha256": digest,
        "immutable": True,
    }
    ctx.report["split_manifest"] = {
        "file": "split_manifest.json",
        "sha256": digest,
        "immutable": True,
        "datasets": {
            name: entry["group_integrity"]
            for name, entry in payload.get("datasets", {}).items()
        },
    }
    return payload


def persist_maps_and_report(ctx: Context) -> None:
    ctx.output.mkdir(parents=True, exist_ok=True)
    (ctx.output / "label_maps.json").write_text(
        json.dumps(ctx.maps, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (ctx.output / "cleaning_report.json").write_text(
        json.dumps(ctx.report, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def refresh_split_manifest(ctx: Context) -> dict:
    """carve 之后允许重写 split_manifest（原文件为不可变 444）。"""
    path = ctx.output / "split_manifest.json"
    if path.exists():
        path.chmod(0o644)
        path.unlink()
    return finalize_split_manifest(ctx)


def finalize(ctx: Context, *, refine_eval: bool = True) -> None:
    rebalance_all(ctx)
    if refine_eval:
        refine_eval_splits(ctx)
    ensure_missing_test_splits(ctx)
    sync_label_maps_from_balanced(ctx)
    stamp_semantic_namespaces(ctx)
    stamp_global_label_ids(ctx)
    finalize_task_pools(ctx)
    finalize_split_manifest(ctx)
    persist_quality_filter_report(ctx)
    persist_maps_and_report(ctx)


def build_registry() -> dict[str, object]:
  registry: dict[str, object] = {
      "electromagnetic_0926": lambda ctx: npy_dataset(ctx, "electromagnetic_0926", 0, NON_EMITTER / "0926电磁数据", 0, "source_label_id", {x: i for i, x in enumerate(["ADSB", "AIS", "AM", "FM", "GPS", "Iridium", "P25"])}, [("train", "train"), ("test", "val")], mirror_mod_label_id=True),
      "xidian14": lambda ctx: npy_dataset(ctx, "xidian14", 1, NON_EMITTER / "xidian_npy_cls14", 0, "mod_label_id", None, [("train", "train"), ("val", "val")]),
      "rml2016_04c": lambda ctx: rml_pickle(ctx, "rml2016_04c", 2, NON_EMITTER / "2016.04C.multisnr.pkl"),
      "rml2016_10a": lambda ctx: rml_pickle(ctx, "rml2016_10a", 3, NON_EMITTER / "RML2016.10a_dict.pkl"),
      "rml2016_10b": lambda ctx: rml10b(ctx, 4),
      "rml2018_1a": lambda ctx: rml2018(ctx, 5),
      "adsb2": lambda ctx: npy_dataset(ctx, "adsb2", 6, EMITTER / "ADSB-2", 1, "emitter_id", None, [("train", "train"), ("val", "val")]),
      "wifi150": lambda ctx: npy_dataset(ctx, "wifi150", 7, EMITTER / "wifi_cls150", 1, "emitter_id", None, [("train", "train"), ("val", "val"), ("test", "test")]),
      "communication_emitters": lambda ctx: dat_emitters(ctx, "communication_emitters", 8, EMITTER / "通信辐射源个体识别数据集", 2048, False),
      "radar_emitters": lambda ctx: dat_emitters(ctx, "radar_emitters", 9, EMITTER / "雷达辐射源个体识别数据集", 1000, True),
      "radar_mod15": lambda ctx: radar_mod15(ctx, 10),
      "wisig": lambda ctx: wisig_manytx(ctx, 11),
      "panoradio_hf": lambda ctx: panoradio_hf(ctx, 12),
      "cjr_mix": lambda ctx: cjr_mix(ctx, 31),
      "radchar": lambda ctx: radchar(ctx, 16),
  }
  for ds_name, ds_id, filename, snr_min in RADCOM_VARIANTS:
      registry[ds_name] = (
          lambda ctx, name=ds_name, dataset_id=ds_id, h5_name=filename, min_snr=snr_min: radcom_hdf5(
              ctx, name, dataset_id, EXTERNAL / "radarcommdataset" / h5_name, snr_min=min_snr
          )
      )
  return registry


RADCOM_ALIASES = frozenset({"radarcomm", "radarcommdataset"})


def resolve_selected_builds(selected: set[str], registry: dict[str, object]) -> list[str]:
    if "all" in selected:
        return sorted(registry)
    names: list[str] = []
    if selected & RADCOM_ALIASES:
        names.extend(name for name, _, _, _ in RADCOM_VARIANTS)
    for name in selected:
        if name in registry:
            names.append(name)
    unknown = selected - set(names) - RADCOM_ALIASES - {"all"}
    if unknown:
        raise ValueError(f"未知数据集: {sorted(unknown)}；可选: {sorted(registry)}")
    return sorted(set(names))


def build_cache_dir(output: Path) -> Path:
    path = output / ".build_cache"
    path.mkdir(parents=True, exist_ok=True)
    return path


def clear_dataset_h5(ctx: Context, name: str) -> None:
    for suffix in ("train", "val", "test", "__balanced_train", "__balanced_val", "__balanced_test"):
        (ctx.h5 / f"{name}_{suffix}.h5").unlink(missing_ok=True)


def write_build_cache(ctx: Context, name: str) -> None:
    entry = ctx.report["datasets"].get(name)
    if entry is None:
        raise RuntimeError(f"{name} 构建后缺少 cleaning_report 条目")
    cache = {
        "dataset_name": name,
        "datasets_id_map": {ds_id: ds_name for ds_id, ds_name in ctx.maps.get("datasets", {}).items() if ds_name == name},
        "modulations": ctx.maps.get("modulations", {}).get(name),
        "sources": ctx.maps.get("sources", {}).get(name),
        "emitters": ctx.maps.get("emitters", {}).get(name),
        "receivers": ctx.maps.get("receivers", {}).get(name),
        "sessions": ctx.maps.get("sessions", {}).get(name),
        "report_entry": entry,
    }
    path = build_cache_dir(ctx.output) / f"{name}.json"
    path.write_text(json.dumps(cache, ensure_ascii=False, indent=2), encoding="utf-8")


def merge_build_caches(ctx: Context, names: list[str] | None = None) -> list[str]:
    cache_dir = build_cache_dir(ctx.output)
    merged: list[str] = []
    for path in sorted(cache_dir.glob("*.json")):
        if names is not None and path.stem not in names:
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        name = str(payload["dataset_name"])
        ctx.touched.add(name)
        ctx.report["datasets"][name] = payload["report_entry"]
        for ds_id, ds_name in payload.get("datasets_id_map", {}).items():
            ctx.maps.setdefault("datasets", {})[str(ds_id)] = ds_name
        for map_key in ("modulations", "sources", "emitters", "receivers", "sessions"):
            value = payload.get(map_key)
            if value:
                ctx.maps.setdefault(map_key, {})[name] = value
        merged.append(name)
    return merged


def run_build(ctx: Context, name: str, registry: dict[str, object]) -> None:
    clear_dataset_h5(ctx, name)
    print(f"[build] {name}", flush=True)
    registry[name](ctx)
    write_build_cache(ctx, name)


def _parallel_build_worker(output: str, dataset_name: str) -> str:
    global _CANONICAL_CFG, _PRETRAIN_H5_CFG
    cfg_json = os.environ.get(_CANONICAL_ENV, "")
    _CANONICAL_CFG = CanonicalIQConfig.from_json(cfg_json) if cfg_json else None
    pretrain_json = os.environ.get(_PRETRAIN_H5_ENV, "")
    _PRETRAIN_H5_CFG = PretrainH5Config.from_json(pretrain_json) if pretrain_json else None
    registry = build_registry()
    ctx = Context(Path(output))
    run_build(ctx, dataset_name, registry)
    return dataset_name


def run_builds_parallel(
    output: Path,
    names: list[str],
    jobs: int,
    *,
    canonical_cfg: CanonicalIQConfig | None = None,
    pretrain_cfg: PretrainH5Config | None = None,
) -> None:
    global _CANONICAL_CFG, _PRETRAIN_H5_CFG
    _CANONICAL_CFG = canonical_cfg
    _PRETRAIN_H5_CFG = pretrain_cfg
    if jobs > 1:
        if canonical_cfg is not None and canonical_cfg.enabled:
            os.environ[_CANONICAL_ENV] = canonical_cfg.to_json()
        else:
            os.environ.pop(_CANONICAL_ENV, None)
        if pretrain_cfg is not None:
            os.environ[_PRETRAIN_H5_ENV] = pretrain_cfg.to_json()
        else:
            os.environ.pop(_PRETRAIN_H5_ENV, None)
    if jobs <= 1 or len(names) <= 1:
        registry = build_registry()
        ctx = Context(output)
        for name in names:
            run_build(ctx, name, registry)
        merge_build_caches(ctx, names)
        return
    print(f"[parallel] jobs={min(jobs, len(names))} datasets={names}", flush=True)
    with ProcessPoolExecutor(max_workers=min(jobs, len(names))) as pool:
        futures = {pool.submit(_parallel_build_worker, str(output), name): name for name in names}
        for future in as_completed(futures):
            name = futures[future]
            future.result()
            print(f"[built] {name}", flush=True)
    merge_build_caches(Context(output), names)


def manifested_dataset_names(ctx: Context) -> set[str]:
    manifest_path = ctx.output / "split_manifest.json"
    if not manifest_path.is_file():
        return set()
    payload = load_manifest(manifest_path)
    return {str(name) for name in payload.get("datasets", {})}


def refuse_manifested_h5_mutation(
    ctx: Context,
    operation: str,
    *,
    allow_new: set[str] | None = None,
) -> None:
    locked = manifested_dataset_names(ctx)
    if not locked:
        return
    if allow_new is not None:
        overlap = sorted(set(allow_new) & locked)
        if not overlap:
            return
        raise ImmutableManifestError(
            f"{operation} 会覆盖已锁定数据集 {overlap}；请使用新的 --output 目录"
        )
    manifest_path = ctx.output / "split_manifest.json"
    raise ImmutableManifestError(
        f"{operation} 会修改已有 split 对应的 H5，但 {manifest_path} 已锁定该数据版本；"
        "请使用新的 --output 目录"
    )


def finalize_new_datasets(ctx: Context, names: list[str]) -> None:
    """在已锁定 split_manifest 上追加或按请求重建指定数据集：不 rebalance、不改写其它 H5。"""
    only = set(names)
    stamp_semantic_namespaces(ctx, only_datasets=only)
    stamp_global_label_ids(ctx, only_datasets=only)
    ensure_missing_test_splits(ctx, only_datasets=only)
    finalize_task_pools(ctx)
    persist_quality_filter_report(ctx)
    refresh_split_manifest(ctx)
    persist_maps_and_report(ctx)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT / "dataset")
    parser.add_argument("--datasets", nargs="*", default=["all"])
    parser.add_argument(
        "--sync-label-maps",
        action="store_true",
        help="仅根据现有 H5 / cleaning_report 同步 label_maps.json，不重新清洗数据",
    )
    parser.add_argument(
        "--stamp-global-labels",
        action="store_true",
        help="仅为现有 H5 写入 global_label_id 并刷新 clustering pool（可与 --sync-label-maps 联用）",
    )
    parser.add_argument(
        "--stamp-mod-labels",
        action="store_true",
        help="将 downstream_modulation_extra 数据集的 source_label_id 镜像为 mod_label_id（可与 --sync-label-maps 联用）",
    )
    parser.add_argument(
        "--stamp-semantic-labels",
        action="store_true",
        help="写入 canonical_mod_label_id/global_emitter_id 并刷新标签命名空间",
    )
    parser.add_argument(
        "--verify-splits",
        action="store_true",
        help="验证不可变 split manifest、H5 哈希及 capture/group 跨 split 重叠",
    )
    parser.add_argument(
        "--refine-eval-splits",
        action="store_true",
        help="仅对 rml2018_1a/adsb2/wifi150/xidian14 的 val/test 应用更严格质量过滤并刷新 task pool",
    )
    parser.add_argument(
        "--build-only",
        action="store_true",
        help="仅构建原始 H5 与 .build_cache，不执行 rebalance / label_maps 收尾",
    )
    parser.add_argument(
        "--rebuild-task-pools",
        action="store_true",
        help="根据现有 H5 重建 task_pools 并写入 label_maps.json；不 rebalance、不改 H5",
    )
    parser.add_argument(
        "--rebuild-pools-only",
        action="store_true",
        help="同 --rebuild-task-pools（兼容别名）",
    )
    parser.add_argument(
        "--apply-min-snr",
        action="store_true",
        help=(
            f"对现有 H5 就地过滤：只保留 snr>={UNIFIED_MIN_SNR_DB:g} dB（可用 --min-snr-db 覆盖）；"
            "需配合 --allow-locked-rebuild；随后刷新 split_manifest 与 task pools"
        ),
    )
    parser.add_argument(
        "--min-snr-db",
        type=float,
        default=UNIFIED_MIN_SNR_DB,
        help=f"--apply-min-snr 的阈值（默认 {UNIFIED_MIN_SNR_DB:g}）",
    )
    parser.add_argument(
        "--splits",
        default=None,
        help="配合 --apply-min-snr：只处理指定 split，逗号分隔，如 val,test",
    )
    parser.add_argument(
        "--allow-locked-rebuild",
        action="store_true",
        help="允许覆盖 split_manifest 已锁定的数据集（只重建 --datasets，随后刷新 manifest，不 rebalance 其它库）",
    )
    parser.add_argument(
        "--ensure-missing-test-splits",
        action="store_true",
        help="仅为「有 train 无 test」按类切 20% 写 *_test.h5 并从 train 删除；不碰已有 test/val，不 rebalance",
    )
    parser.add_argument(
        "--split-seed",
        type=int,
        default=MISSING_TEST_SPLIT_SEED,
        help="ensure_missing_test_splits 固定种子",
    )
    parser.add_argument(
        "--split-fraction",
        type=float,
        default=MISSING_TEST_FRACTION,
        help="ensure_missing_test_splits 每类切分比例（默认 0.2）",
    )
    parser.add_argument(
        "--finalize-only",
        action="store_true",
        help="合并 .build_cache 并执行 rebalance、global_label_id、task_pools",
    )
    parser.add_argument(
        "--jobs",
        type=int,
        default=1,
        help="并行构建数据集的工作进程数（>1 时各数据集独立构建）",
    )
    parser.add_argument(
        "--canonical-iq",
        action="store_true",
        help="统一预处理：float32 + 可选去 DC/谱峰居中 + 定长档位（不做 joint_power；幅度仍由模型 RevIN）",
    )
    parser.add_argument(
        "--canonical-no-dc",
        action="store_true",
        help="与 --canonical-iq 联用：跳过去 DC",
    )
    parser.add_argument(
        "--canonical-no-center-peak",
        action="store_true",
        help="与 --canonical-iq 联用：跳过谱峰居中",
    )
    parser.add_argument(
        "--pretrain-h5-v2",
        action="store_true",
        help="预训练 7 库紧凑 H5 v2：离线 joint_energy + snr/fs/RevIN 统计入库（训练跳过在线 RevIN 统计）",
    )
    parser.add_argument(
        "--purge-h5",
        action="store_true",
        help="构建前删除待重建数据集的旧 H5（仅 dataset/h5/*.h5，不删除源数据）",
    )
    parser.add_argument(
        "--device",
        choices=("cpu", "cuda"),
        default="cpu",
        help=(
            "--pretrain-h5-v2 时 joint_energy 预计算设备（cuda 需 PyTorch GPU）；"
            "与 --jobs > 1 联用时自动回退 cpu（避免 fork+CUDA）"
        ),
    )
    parser.add_argument(
        "--precompute-chunk",
        type=int,
        default=256,
        help="--pretrain-h5-v2 预计算 chunk 大小",
    )
    args = parser.parse_args()
    ctx = Context(args.output)
    if args.apply_min_snr:
        if not args.allow_locked_rebuild:
            raise SystemExit("--apply-min-snr 会改写已锁定 H5，请加 --allow-locked-rebuild")
        only = None
        selected = {str(x) for x in (args.datasets or []) if str(x) != "all"}
        if selected:
            only = selected
        only_splits = None
        if args.splits:
            only_splits = {part.strip().lower() for part in str(args.splits).split(",") if part.strip()}
        stats = apply_unified_min_snr_to_h5(
            ctx,
            min_db=float(args.min_snr_db),
            only_datasets=only,
            only_splits=only_splits,
        )
        dropped_total = sum(int(v.get("dropped", 0)) for v in stats.values())
        print(
            f"[apply-min-snr] files={len(stats)} dropped_rows={dropped_total} min_db={float(args.min_snr_db):g}",
            flush=True,
        )
        # 行子集已保留既有 label 列；勿 stamp_global（可能对非 chunked 列 resize 失败）
        finalize_task_pools(ctx)
        refresh_split_manifest(ctx)
        persist_quality_filter_report(ctx)
        persist_maps_and_report(ctx)
        print(f"[apply-min-snr] done {ctx.output}", flush=True)
        return
    if args.verify_splits:
        manifest_path = ctx.output / "split_manifest.json"
        if manifest_path.exists():
            payload = load_manifest(manifest_path)
            assert_manifest_files_unchanged(payload, ctx.h5)
        else:
            payload = finalize_split_manifest(ctx)
            (ctx.output / "label_maps.json").write_text(
                json.dumps(ctx.maps, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            (ctx.output / "cleaning_report.json").write_text(
                json.dumps(ctx.report, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        print(f"[verify-splits] {manifest_path} sha256={payload['manifest_sha256']}", flush=True)
        return
    if args.ensure_missing_test_splits or args.rebuild_task_pools or args.rebuild_pools_only:
        if args.ensure_missing_test_splits:
            split_stats = ensure_missing_test_splits(
                ctx, seed=int(args.split_seed), frac=float(args.split_fraction)
            )
            if split_stats:
                print(f"[ensure-test-splits] {split_stats}", flush=True)
                refresh_split_manifest(ctx)
        finalize_task_pools(ctx)
        persist_maps_and_report(ctx)
        print(f"[rebuild-task-pools] {ctx.output}", flush=True)
        return
    if args.finalize_only:
        refuse_manifested_h5_mutation(ctx, "--finalize-only")
        merged = merge_build_caches(ctx)
        if merged:
            print(f"[merge-cache] {merged}", flush=True)
        finalize(ctx)
        print(f"[done] {ctx.output}", flush=True)
        return
    if args.refine_eval_splits:
        refuse_manifested_h5_mutation(ctx, "--refine-eval-splits")
        refine_eval_splits(ctx)
        stamp_semantic_namespaces(ctx)
        stamp_global_label_ids(ctx)
        finalize_task_pools(ctx)
        finalize_split_manifest(ctx)
        ctx.output.mkdir(parents=True, exist_ok=True)
        (ctx.output / "label_maps.json").write_text(json.dumps(ctx.maps, ensure_ascii=False, indent=2), encoding="utf-8")
        (ctx.output / "cleaning_report.json").write_text(json.dumps(ctx.report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[refine-eval-splits] {ctx.output}", flush=True)
        return
    if (
        args.sync_label_maps
        or args.stamp_global_labels
        or args.stamp_mod_labels
        or args.stamp_semantic_labels
    ):
        if args.stamp_global_labels or args.stamp_mod_labels or args.stamp_semantic_labels:
            refuse_manifested_h5_mutation(ctx, "label stamping")
        if args.sync_label_maps:
            sync_label_maps_from_balanced(ctx)
        mod_stamped: dict[str, int] = {}
        if args.stamp_mod_labels:
            mod_stamped = stamp_mod_label_ids(ctx)
        namespace_stats: dict[str, dict[str, int]] = {}
        if args.stamp_semantic_labels or args.stamp_mod_labels:
            namespace_stats = stamp_semantic_namespaces(ctx)
        stamped: dict[str, int] = {}
        if args.stamp_global_labels:
            stamped = stamp_global_label_ids(ctx)
        finalize_task_pools(ctx)
        finalize_split_manifest(ctx)
        ctx.output.mkdir(parents=True, exist_ok=True)
        (ctx.output / "label_maps.json").write_text(json.dumps(ctx.maps, ensure_ascii=False, indent=2), encoding="utf-8")
        (ctx.output / "cleaning_report.json").write_text(json.dumps(ctx.report, ensure_ascii=False, indent=2), encoding="utf-8")
        if mod_stamped:
            labeled = sum(mod_stamped.values())
            print(f"[stamp-mod-labels] {len(mod_stamped)} files, {labeled} labeled samples", flush=True)
        if stamped:
            labeled = sum(stamped.values())
            print(f"[stamp-global-labels] {len(stamped)} files, {labeled} labeled samples", flush=True)
        if namespace_stats:
            mod_count = sum(namespace_stats["canonical_mod_label_id"].values())
            emitter_count = sum(namespace_stats["global_emitter_id"].values())
            print(
                f"[stamp-semantic-labels] canonical_mod={mod_count}, global_emitter={emitter_count}",
                flush=True,
            )
        print(f"[sync-label-maps] {ctx.output}", flush=True)
        return
    selected = set(args.datasets)
    registry = build_registry()
    build_names = resolve_selected_builds(selected, registry)
    if not build_names:
        raise SystemExit("未选择任何数据集")
    locked_now = manifested_dataset_names(ctx)
    rebuild_locked = bool(args.allow_locked_rebuild) and bool(set(build_names) & locked_now)
    if rebuild_locked:
        print(f"[allow-locked-rebuild] {sorted(set(build_names) & locked_now)}", flush=True)
    else:
        refuse_manifested_h5_mutation(ctx, "dataset build", allow_new=set(build_names))
    jobs = max(1, int(args.jobs))
    if args.pretrain_h5_v2 and jobs == 1:
        jobs = min(22, os.cpu_count() or 1)
        print(f"[pretrain-h5-v2] auto jobs={jobs}", flush=True)
    if args.pretrain_h5_v2 and jobs > 1 and args.device == "cuda":
        print("[pretrain-h5-v2] --jobs > 1 与 CUDA fork 不兼容，回退 --device cpu", flush=True)
        args.device = "cpu"
    global _PRETRAIN_H5_CFG
    if args.pretrain_h5_v2:
        device = str(args.device)
        if device == "cuda":
            import torch

            if not torch.cuda.is_available():
                print("[pretrain-h5-v2] CUDA 不可用，回退 CPU", flush=True)
                device = "cpu"
        _PRETRAIN_H5_CFG = PretrainH5Config(
            precompute_joint_energy=True,
            device=device,
            chunk_size=max(1, int(args.precompute_chunk)),
        )
        ctx.report.setdefault("policy", {})["h5_schema"] = "pretrain_v2_joint_energy"
        if not args.canonical_iq:
            args.canonical_iq = True
    if args.purge_h5 or args.pretrain_h5_v2:
        purge_stems = [n for n in build_names if n in PRETRAIN_STEMS] if args.pretrain_h5_v2 else build_names
        removed = purge_h5_for_stems(ctx.h5, purge_stems)
        if removed:
            print(f"[purge-h5] removed {len(removed)} files from {ctx.h5}", flush=True)
    canonical_cfg = CanonicalIQConfig.from_cli(
        enabled=bool(args.canonical_iq),
        remove_dc=not bool(args.canonical_no_dc),
        center_spectral_peak=not bool(args.canonical_no_center_peak),
    )
    if canonical_cfg.enabled:
        ctx.report.setdefault("policy", {})["normalization"] = CANONICAL_SCALE_POLICY
        ctx.report["policy"]["canonical_iq"] = canonical_cfg.attrs()
    run_builds_parallel(
        args.output,
        build_names,
        jobs,
        canonical_cfg=canonical_cfg,
        pretrain_cfg=_PRETRAIN_H5_CFG,
    )
    if args.build_only:
        print(f"[build-only] {build_names}", flush=True)
        return
    merge_build_caches(ctx, build_names)
    locked = manifested_dataset_names(ctx)
    if rebuild_locked or (locked and not (set(build_names) & locked)):
        finalize_new_datasets(ctx, build_names)
        print(f"[done] {'rebuilt' if rebuild_locked else 'appended'} {build_names} -> {ctx.output}", flush=True)
        return
    finalize(ctx)
    print(f"[done] {ctx.output}", flush=True)


if __name__ == "__main__":
    main()
