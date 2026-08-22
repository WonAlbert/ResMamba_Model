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
from resmamba_signal_model.training.emitter_labels import (
    filter_emitter_downstream_pool,
    h5_dataset_name,
    load_emitter_downstream_datasets,
)
from resmamba_signal_model.data.wisig_manytx import (
    WiSigBlockSink,
    emitter_labels,
    stream_manytx_blocks,
)
from resmamba_signal_model.training.pool_filters import (
    filter_dataset_pool,
    filter_excluded_dataset_pool,
    load_downstream_modulation_datasets,
    load_downstream_modulation_extra_datasets,
    load_downstream_shared_datasets,
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
RADCOM_VARIANTS = (
    ("radcom_dynamic", 13, "RadComDynamic.hdf5", 0.0),
    ("radcom_awgn", 14, "RadComAWGN.hdf5", 0.0),
    ("radcom_ota", 15, "RadComOta2.45GHz.hdf5", None),
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
# 仅 val/test、不做类别均衡的数据集（不参与 MAE 预训练时的旧约定；open_real_data 已改为 train/val）
VAL_TEST_ONLY_DATASETS = frozenset()
VAL_TEST_SPLIT_RATIO = 0.8
RML2018_EVAL_MIN_SNR = 6.0
MISSING_TEST_SPLIT_SEED = 20260822
MISSING_TEST_FRACTION = 0.2
_MISSING_TEST_LABEL_FIELDS = (
    "emitter_id",
    "mod_label_id",
    "source_label_id",
    "canonical_mod_label_id",
    "global_label_id",
)


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


def quality_mask(
    iq: np.ndarray,
    labels: np.ndarray | None = None,
    *,
    strict: bool = False,
) -> tuple[np.ndarray, Counter]:
    if labels is None:
        return _quality_mask_global(iq, strict=strict)
    labels = np.asarray(labels).reshape(-1)
    keep = np.zeros(len(iq), dtype=bool)
    reasons: Counter = Counter()
    for cls in np.unique(labels):
        idx = labels == cls
        cls_keep, cls_reasons = _quality_mask_global(iq[idx], strict=strict)
        keep[idx] = cls_keep
        reasons.update(cls_reasons)
    return keep, reasons


def _quality_mask_global(iq: np.ndarray, *, strict: bool = False) -> tuple[np.ndarray, Counter]:
    x = iq.astype(np.float64, copy=False)
    finite = np.isfinite(x).all(axis=(-2, -1))
    power = np.mean(np.square(x), axis=(-2, -1))
    positive = power > np.finfo(np.float32).tiny
    log_power = np.log10(np.maximum(power, 1e-30))
    valid = log_power[finite & positive]
    mad_scale = 3.0 if strict else 4.0
    if len(valid) >= 32:
        median = np.median(valid)
        spread = max(1.4826 * np.median(np.abs(valid - median)), 0.15)
        power_ok = np.abs(log_power - median) <= mad_scale * spread
    else:
        power_ok = positive
    complex_power = np.square(x).sum(axis=-2)
    papr = complex_power.max(axis=-1) / np.maximum(complex_power.mean(axis=-1), 1e-30)
    papr_limit = 20.0 if strict else 30.0
    impulse_ok = papr <= papr_limit
    z = x[..., 0, :] + 1j * x[..., 1, :]
    corr = np.abs(np.sum(z[..., 1:] * np.conj(z[..., :-1]), axis=-1))
    corr /= np.maximum(
        np.sqrt(np.sum(np.abs(z[..., 1:]) ** 2, axis=-1) * np.sum(np.abs(z[..., :-1]) ** 2, axis=-1)),
        1e-30,
    )
    corr_floor = 0.05 if strict else 1e-2
    structure_ok = corr >= corr_floor
    keep = finite & positive & power_ok & impulse_ok & structure_ok
    return keep, Counter({
        "non_finite": int((~finite).sum()),
        "silent": int((finite & ~positive).sum()),
        "power_outlier": int((finite & positive & ~power_ok).sum()),
        "impulsive": int((finite & positive & power_ok & ~impulse_ok).sum()),
        "low_quality_proxy": int((finite & positive & power_ok & impulse_ok & ~structure_ok).sum()),
    })


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
        self.f.attrs.update({
            "source_path": str(source), "scale_policy": "none",
            "quality_filter": "finite, non-silent, robust-power-8MAD, PAPR<=100, lag1-correlation>=1e-3",
            "channel_axis": 1,
            "signal_contract_version": SIGNAL_CONTRACT_VERSION,
            "missing_metadata_marker": MISSING_METADATA,
        })
        self.count = 0
        self._known_group_metadata = Counter()

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
        per_class_labels = None
        for field in ("emitter_id", "mod_label_id", "source_label_id", "global_label_id"):
            if field in meta:
                per_class_labels = np.asarray(meta[field]).reshape(-1)
                break
        keep, reasons = quality_mask(iq, labels=per_class_labels)
        removed.update(reasons)
        iq = iq[keep]
        if not len(iq):
            return
        start, end = self.count, self.count + len(iq)
        self.iq.resize(end, axis=0)
        self.iq[start:end] = iq
        defaults = {"length": self.length, "dataset_id": self.dataset_id, "task_type_id": self.task_id}
        self._append_metadata(start, end, keep, meta, defaults)
        self.count = end

    def append_raw(self, iq: np.ndarray, **meta) -> None:
        if not len(iq):
            return
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
                    "rml_snr": "SNR > 0 dB",
                    "real_signal_iq": "Hilbert analytic signal; real part is original signal and imaginary part is Hilbert transform",
                    "extreme_filter": "finite/non-silent, per-class robust power 4 MAD, PAPR <= 30",
                    "npy_label_maps": "numeric Y_*.npy labels are inferred from data; string labels use configured name maps",
                    "unlabelled_snr": "stricter lag-1 correlation proxy; SNR remains NaN",
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
                    "rml2018_eval_snr": f"val/test only: SNR >= {RML2018_EVAL_MIN_SNR} dB",
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
                "rml2018_eval_snr": f"val/test only: SNR >= {RML2018_EVAL_MIN_SNR} dB",
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
        writer = Writer(ctx.h5 / f"{name}_{output_split}.h5", as_iq(x[:1]).shape[-1], x.dtype, dataset_id, task, root)
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
    groups = {"train": [], "val": []}
    snr_removed = sum(len(v) for (m, s), v in data.items() if float(s) <= 0)
    for (mod, snr), values in sorted(data.items(), key=lambda x: (str(x[0][0]), float(x[0][1]))):
        if float(snr) <= 0:
            continue
        values = as_iq(np.asarray(values))
        cut = int(len(values) * 0.8)
        groups["train"].append((values[:cut], mod, snr))
        groups["val"].append((values[cut:], mod, snr))
    for split, items in groups.items():
        writer = Writer(ctx.h5 / f"{name}_{split}.h5", items[0][0].shape[-1], items[0][0].dtype, dataset_id, 0, path)
        removed, raw = Counter({"snr_le_0_source_samples": snr_removed}), 0
        for values, mod, snr in items:
            raw += len(values)
            writer.append_clean(values, removed, mod_label_id=np.full(len(values), labels[str(mod)]), snr=np.full(len(values), snr))
        ctx.done(name, split, writer, raw, removed, labels)


def rml2018(ctx: Context, dataset_id: int):
    name = "rml2018_1a"
    path = NON_EMITTER / "RML2018.1A" / "GOLD_XYZ_OSC.0001_1024.hdf5"
    labels = {m: i for i, m in enumerate(RML2018_ORDER)}
    ctx.maps["datasets"][str(dataset_id)], ctx.maps["modulations"][name] = name, labels
    with h5py.File(path, "r") as f:
        writers = {s: Writer(ctx.h5 / f"{name}_{s}.h5", 1024, f["X"].dtype, dataset_id, 0, path) for s in ("train", "val")}
        removed, raw, sequence = {s: Counter() for s in writers}, Counter(), 0
        for start in range(0, len(f["X"]), 2048):
            snr = np.asarray(f["Z"][start:start + 2048]).reshape(-1)
            allowed = snr > 0
            block = as_iq(np.asarray(f["X"][start:start + 2048])[allowed])
            labels_block = np.argmax(np.asarray(f["Y"][start:start + 2048])[allowed], axis=1)
            snr_block = snr[allowed]
            removed["train"]["snr_le_0_source_samples"] += int((~allowed).sum())
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
    writers = {s: Writer(ctx.h5 / f"{name}_{s}.h5", window, output_dtype, dataset_id, 1, root) for s in ("train", "val")}
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
                for start in range(0, len(local), 2048):
                    idx = local[start:start + 2048]
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
        writer = Writer(tmp_path, length, dtype, dataset_id, task_id, source_files[0])
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


def rebalance_all(ctx: Context) -> None:
    ctx.pool = defaultdict(list)
    dataset_names = sorted(ctx.touched) if ctx.touched else sorted(ctx.report.get("datasets", {}))
    for name in dataset_names:
        entry = ctx.report.get("datasets", {}).get(name)
        if entry is None:
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
    with h5py.File(path, "r") as f:
        n = int(f["iq"].shape[0])
        iq_parts: list[np.ndarray] = []
        meta: dict[str, list[np.ndarray]] = {key: [] for key in Writer.ALL_FIELDS}
        for start in range(0, n, 2048):
            end = min(start + 2048, n)
            iq_parts.append(np.asarray(f["iq"][start:end]))
            for key in Writer.ALL_FIELDS:
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
    labels = np.asarray(meta[label_field]).reshape(-1)
    keep, reasons = quality_mask(iq, labels=labels, strict=True)
    if dataset_name == "rml2018_1a" and meta.get("snr") is not None:
        snr = np.asarray(meta["snr"]).reshape(-1)
        snr_ok = np.isfinite(snr) & (snr >= RML2018_EVAL_MIN_SNR)
        reasons["snr_lt_6"] = int((keep & ~snr_ok).sum())
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
    with h5py.File(path, "r") as f:
        length = int(f["iq"].shape[-1])
        dtype = f["iq"].dtype
        dataset_id = int(f["dataset_id"][0])
        task_id = int(f["task_type_id"][0])
        source = Path(str(f.attrs.get("source_path", path)))
    tmp_path = path.with_suffix(".refine.h5")
    writer = Writer(tmp_path, length, dtype, dataset_id, task_id, source)
    quality_note = "eval-refined: strict per-class power 3 MAD, PAPR<=20, lag1-correlation>=0.05"
    if dataset_name == "rml2018_1a":
        quality_note += f", SNR>={RML2018_EVAL_MIN_SNR}"
    writer.f.attrs["quality_filter"] = quality_note
    for start in range(0, len(selected_idx), 2048):
        chunk_idx = selected_idx[start:start + 2048]
        chunk_meta = {
            key: meta[key][chunk_idx]
            for key in Writer.ALL_FIELDS
            if meta.get(key) is not None
        }
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
    writers = {s: Writer(ctx.h5 / f"{name}_{s}.h5", 128, np.float32, dataset_id, 0, root) for s in ("train", "val")}
    removed, raw = {s: Counter({"snr_le_0_source_files": sum(float(p.stem.rsplit(" ", 1)[1]) <= 0 for p in files)}) for s in writers}, Counter()
    for path in files:
        mod, snr_text = path.stem.rsplit(" ", 1)
        snr = float(snr_text)
        if snr <= 0:
            continue
        batch, line_no = [], 0
        with path.open(encoding="utf-8") as f:
            for line in f:
                values = np.fromstring(line.replace("(", "").replace(")", ""), dtype=np.complex64, sep=" ")
                if len(values) != 128:
                    removed["train"]["malformed"] += 1
                    continue
                batch.append(values)
                if len(batch) == 2048:
                    block = as_iq(np.asarray(batch))
                    idx = np.arange(line_no, line_no + len(block))
                    for split, choose in (("train", idx % 5 != 0), ("val", idx % 5 == 0)):
                        raw[split] += int(choose.sum())
                        writers[split].append_clean(block[choose], removed[split], mod_label_id=np.full(choose.sum(), labels[mod]), snr=np.full(choose.sum(), snr))
                    line_no += len(block)
                    batch.clear()
            if batch:
                block, idx = as_iq(np.asarray(batch)), np.arange(line_no, line_no + len(batch))
                for split, choose in (("train", idx % 5 != 0), ("val", idx % 5 == 0)):
                    raw[split] += int(choose.sum())
                    writers[split].append_clean(block[choose], removed[split], mod_label_id=np.full(choose.sum(), labels[mod]), snr=np.full(choose.sum(), snr))
    for split in writers:
        ctx.done(name, split, writers[split], raw[split], removed[split], labels)


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
        split: Writer(ctx.h5 / f"{name}_{split}.h5", 256, np.float32, dataset_id, 1, pkl_path)
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
    snr_min: float | None = 0.0,
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
    writer = Writer(ctx.h5 / f"{name}_train.h5", 128, np.float32, dataset_id, 0, path)
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
            if snr_min is not None and snr <= snr_min:
                removed["snr_le_min"] += 1
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
    writer = Writer(
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
            if snr_min is not None and snr <= snr_min:
                removed["snr_le_min"] += 1
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
    writer = Writer(out_path, length, dtype, dataset_id, task_id, source)
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
    snr_min: float | None = 0.0,
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
        keys = [k for k in keys if parse_radcom_key(k)[2] > snr_min]

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
    keep = snr_arr > 0
    removed = Counter({"snr_le_0_source_samples": int((~keep).sum())})
    x = x[keep]
    y = y[keep]
    snr_arr = snr_arr[keep]

    writer = Writer(ctx.h5 / f"{name}_train.h5", int(x.shape[-1]), np.float32, dataset_id, 0, root)
    raw = len(x)
    for start in range(0, len(x), 2048):
        end = min(start + 2048, len(x))
        block = as_iq(np.asarray(x[start:end])).astype(np.float32, copy=False)
        writer.append_clean(
            block,
            removed,
            mod_label_id=y[start:end],
            snr=snr_arr[start:end],
        )
    ctx.done(name, "train", writer, raw, removed, labels)


def cjr_mix(ctx: Context, dataset_id: int) -> None:
    """CJR-mix：仅转换 train parquet → RFData H5。"""
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

    writer: Writer | None = None
    removed = Counter()
    raw = 0
    label_ids: set[int] = set()
    batch_iq: list[np.ndarray] = []
    batch_mod: list[int] = []
    batch_snr: list[float] = []

    def flush() -> None:
        nonlocal batch_iq, batch_mod, batch_snr
        if not batch_iq or writer is None:
            return
        block = np.stack(batch_iq, axis=0)
        writer.append_clean(
            block,
            removed,
            mod_label_id=np.asarray(batch_mod, dtype=np.int32),
            source_label_id=np.asarray(batch_mod, dtype=np.int32),
            snr=np.asarray(batch_snr, dtype=np.float32),
        )
        batch_iq, batch_mod, batch_snr = [], [], []

    for path in list_parquet_files(data_root, "train"):
        pf = pq.ParquetFile(path)
        for rg in range(pf.num_row_groups):
            table = pf.read_row_group(rg, columns=["iq", "infer_class", "snr"])
            infer = table.column("infer_class").to_numpy(zero_copy_only=False)
            snrs = table.column("snr").to_numpy(zero_copy_only=False)
            iq_col = table.column("iq")
            for i in range(table.num_rows):
                raw += 1
                iq = normalize_iq_array(parse_iq_array(iq_col[i].as_py()), "abs")
                label = int(infer[i])
                label_ids.add(label)
                if writer is None:
                    labels = {str(v): v for v in sorted(label_ids)}
                    ctx.maps["modulations"][name] = labels
                    writer = Writer(
                        ctx.h5 / f"{name}_train.h5",
                        int(iq.shape[-1]),
                        np.float32,
                        dataset_id,
                        0,
                        data_root / "train",
                    )
                batch_iq.append(iq)
                batch_mod.append(label)
                batch_snr.append(float(snrs[i]))
                if len(batch_iq) >= 256:
                    flush()

    flush()
    if writer is None:
        raise RuntimeError("CJR-mix train 为空")
    labels = {str(v): v for v in sorted(label_ids)}
    ctx.maps["modulations"][name] = labels
    writer.f.attrs["scale_policy"] = "abs_max_clip5_at_convert"
    writer.f.attrs["source_path"] = str(data_root / "train")
    ctx.done(name, "train", writer, raw, removed, labels)


def open_real_data(ctx: Context, dataset_id: int) -> None:
    """open_realData round7 CSV → RFData H5；train/val（8:2），不做类别均衡。"""
    name = "open_real_data"
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
    length = int(len(sample_amp))

    kept_iq: list[np.ndarray] = []
    kept_labels: list[int] = []
    removed: Counter = Counter()
    raw = 0

    for cls_dir in class_dirs:
        cls_id = int(cls_dir.name)
        for csv_path in sorted(cls_dir.glob("*.csv")):
            raw += 1
            amp = np.loadtxt(csv_path, delimiter=",", skiprows=1, usecols=1, dtype=np.float64)
            if len(amp) != length:
                removed["malformed"] += 1
                continue
            iq = as_iq(amp).astype(np.float32)
            keep, reasons = quality_mask(iq[None], labels=np.asarray([cls_id], dtype=np.int32))
            removed.update(reasons)
            if bool(keep[0]):
                kept_iq.append(iq)
                kept_labels.append(cls_id)

    if not kept_iq:
        raise ValueError(f"{name} 清洗后无可用样本")

    iq_all = np.stack(kept_iq, axis=0)
    labels_all = np.asarray(kept_labels, dtype=np.int32)
    rng = np.random.default_rng(20260629)
    order = rng.permutation(len(iq_all))
    iq_all = iq_all[order]
    labels_all = labels_all[order]

    n = len(iq_all)
    n_train = int(n * VAL_TEST_SPLIT_RATIO)
    split_blocks = [
        ("train", iq_all[:n_train], labels_all[:n_train]),
        ("val", iq_all[n_train:], labels_all[n_train:]),
    ]

    ctx.touched.add(name)
    entry = ctx.report["datasets"].setdefault(name, {"labels": labels, "splits": {}})
    entry["labels"] = labels
    entry.pop("balanced_split", None)
    entry["splits"].pop("train", None)

    for split_name, iq_block, lab_block in split_blocks:
        out_path = ctx.h5 / f"{name}_{split_name}.h5"
        writer = Writer(out_path, length, np.float32, dataset_id, 0, root)
        for start in range(0, len(iq_block), 2048):
            end = start + 2048
            writer.append_raw(
                iq_block[start:end],
                source_label_id=lab_block[start:end],
                mod_label_id=lab_block[start:end],
            )
        writer.close()
        entry["splits"][split_name] = {
            "raw": raw,
            "kept": writer.count,
            "removed": dict(removed) if split_name == "train" else {},
            "file": out_path.name,
        }

    entry["val_test_split"] = {
        "label_field": "source_label_id",
        "train_ratio": VAL_TEST_SPLIT_RATIO,
        "no_class_balance": True,
        "total_kept": n,
        "classes": len(set(kept_labels)),
        "splits": {
            split: {"kept": entry["splits"][split]["kept"], "file": entry["splits"][split]["file"]}
            for split in ("train", "val")
        },
    }


def rebuild_pools_from_h5(ctx: Context) -> None:
    ctx.pool = defaultdict(list)
    ctx.final_label_fields = {}
    if not ctx.h5.exists():
        return
    for path in sorted(ctx.h5.glob("*.h5")):
        if "__balanced_" in path.name:
            continue
        with h5py.File(path, "r") as f:
            task_id = int(f["task_type_id"][0])
            label_field = choose_label_field([path])
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


def stamp_semantic_namespaces(ctx: Context) -> dict[str, dict[str, int]]:
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
    for path in sorted(ctx.h5.glob("*.h5")):
        if "__balanced_" in path.name:
            continue
        with h5py.File(path, "r+") as f:
            n = int(f["iq"].shape[0])
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
                global_emitter[choose] = emitter_namespace.map_local(
                    dataset_name,
                    local_emitter[choose],
                )
            _write_int_column(f, "canonical_mod_label_id", canonical)
            _write_int_column(f, "global_emitter_id", global_emitter)
            canonical_stats[path.name] = int(np.sum(canonical >= 0))
            emitter_stats[path.name] = int(np.sum(global_emitter >= 0))
            f.attrs["modulation_ontology_version"] = ontology.version
            f.attrs["emitter_namespace_version"] = emitter_namespace.version
    return {
        "canonical_mod_label_id": canonical_stats,
        "global_emitter_id": emitter_stats,
    }


def stamp_global_label_ids(ctx: Context) -> dict[str, int]:
    """为每个 H5 样本写入跨数据集唯一的 global_label_id（用于 clustering）。"""
    if not ctx.h5.exists():
        return {}
    stats: dict[str, int] = {}
    for path in sorted(ctx.h5.glob("*.h5")):
        if "__balanced_" in path.name:
            continue
        with h5py.File(path, "r+") as f:
            n = int(f["iq"].shape[0])
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


def ensure_missing_test_splits(
    ctx: Context,
    *,
    seed: int = MISSING_TEST_SPLIT_SEED,
    frac: float = MISSING_TEST_FRACTION,
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
    白名单见 configs/datasets.yaml（pretrain / downstream_modulation / emitter_downstream）。
    """
    rebuild_pools_from_h5(ctx)
    train = sorted(ctx.pool["train"])
    val = sorted(ctx.pool["val"])
    test = sorted(ctx.pool["test"])
    fields = ctx.final_label_fields
    pf = lambda split, **kw: pool_files(split, fields, **kw)
    config_path = ROOT / "configs" / "datasets.yaml"
    pretrain_datasets = load_pretrain_datasets(config_path=config_path)
    modulation_datasets = load_downstream_modulation_datasets(config_path=config_path)
    emitter_downstream = load_emitter_downstream_datasets(config_path=config_path)
    shared_datasets = load_downstream_shared_datasets(config_path=config_path)
    excluded = load_excluded_datasets(config_path=config_path)

    def _pool(split_files: list[tuple[str, int]], **kwargs: object) -> list[str]:
        return filter_excluded_dataset_pool(pf(split_files, **kwargs), excluded)

    def _whitelist_pool(
        split_files: list[tuple[str, int]],
        allowed: list[str],
        **kwargs: object,
    ) -> list[str]:
        return filter_dataset_pool(_pool(split_files, **kwargs), allowed)

    modulation_train = _whitelist_pool(
        test, modulation_datasets, task_id=0, label_field="mod_label_id"
    )
    modulation_val = _whitelist_pool(
        val, modulation_datasets, task_id=0, label_field="mod_label_id"
    )
    emitter_train = filter_emitter_downstream_pool(
        _whitelist_pool(test, emitter_downstream, task_id=1, label_field="emitter_id"),
        emitter_downstream,
    )
    emitter_val = filter_emitter_downstream_pool(
        _whitelist_pool(val, emitter_downstream, task_id=1, label_field="emitter_id"),
        emitter_downstream,
    )
    shared_train = sorted(set(modulation_train) | set(emitter_train))
    shared_val = sorted(set(modulation_val) | set(emitter_val))
    shared_train = filter_dataset_pool(shared_train, shared_datasets)
    shared_val = filter_dataset_pool(shared_val, shared_datasets)

    ctx.maps["task_pools"] = {
        "pretrain_train": _whitelist_pool(train, pretrain_datasets),
        "pretrain_val": _whitelist_pool(val, pretrain_datasets),
        "downstream_modulation_train": modulation_train,
        "downstream_modulation_val": modulation_val,
        "downstream_source_train": _whitelist_pool(test, modulation_datasets, task_id=0, label_field="source_label_id"),
        "downstream_source_val": _whitelist_pool(val, modulation_datasets, task_id=0, label_field="source_label_id"),
        "downstream_emitter_train": emitter_train,
        "downstream_emitter_val": emitter_val,
        "downstream_prediction_train": shared_train,
        "downstream_prediction_val": shared_val,
        "clustering_train": pool_files_with_global_labels(
            ctx, [(filename, tid) for filename, tid in test if filename in set(shared_train)]
        ),
        "clustering_val": pool_files_with_global_labels(
            ctx, [(filename, tid) for filename, tid in val if filename in set(shared_val)]
        ),
    }
    ctx.maps["pretrain_datasets"] = list(pretrain_datasets)
    ctx.maps["downstream_modulation_datasets"] = list(modulation_datasets)
    ctx.maps["emitter_downstream_datasets"] = list(emitter_downstream)
    ctx.maps["downstream_shared_datasets"] = list(shared_datasets)
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
      "open_real_data": lambda ctx: open_real_data(ctx, 10),
      "wisig": lambda ctx: wisig_manytx(ctx, 11),
      "panoradio_hf": lambda ctx: panoradio_hf(ctx, 12),
      "cjr_mix": lambda ctx: cjr_mix(ctx, 31),
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
    registry = build_registry()
    ctx = Context(Path(output))
    run_build(ctx, dataset_name, registry)
    return dataset_name


def run_builds_parallel(output: Path, names: list[str], jobs: int) -> None:
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


def refuse_manifested_h5_mutation(ctx: Context, operation: str) -> None:
    manifest_path = ctx.output / "split_manifest.json"
    if manifest_path.exists():
        raise ImmutableManifestError(
            f"{operation} 会修改已有 split 对应的 H5，但 {manifest_path} 已锁定该数据版本；"
            "请使用新的 --output 目录"
        )


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
    args = parser.parse_args()
    ctx = Context(args.output)
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
            split_stats = ensure_missing_test_splits(ctx, seed=int(args.split_seed))
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
    refuse_manifested_h5_mutation(ctx, "dataset build")
    registry = build_registry()
    build_names = resolve_selected_builds(selected, registry)
    if not build_names:
        raise SystemExit("未选择任何数据集")
    jobs = max(1, int(args.jobs))
    run_builds_parallel(args.output, build_names, jobs)
    if args.build_only:
        print(f"[build-only] {build_names}", flush=True)
        return
    merge_build_caches(ctx, build_names)
    finalize(ctx)
    print(f"[done] {ctx.output}", flush=True)


if __name__ == "__main__":
    main()
