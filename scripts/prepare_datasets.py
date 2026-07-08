#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from resmamba_signal_model.thread_env import normalize_thread_env

normalize_thread_env()

import argparse
from collections import Counter, defaultdict
import json
import pickle

import h5py
import numpy as np
import yaml
from scipy.signal import hilbert

from resmamba_signal_model.training.clustering_labels import GLOBAL_LABEL_NAMESPACE, global_cluster_labels
from resmamba_signal_model.training.emitter_labels import (
    filter_emitter_downstream_pool,
    h5_dataset_name,
    load_emitter_downstream_datasets,
)
from resmamba_signal_model.training.pool_filters import (
    filter_excluded_dataset_pool,
    load_downstream_excluded_datasets,
    load_downstream_modulation_extra_datasets,
    load_excluded_datasets,
)


ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = ROOT.parent
NON_EMITTER = WORKSPACE / "辐射源（非个体）识别数据"
EMITTER = WORKSPACE / "个体辐射源数据"
RML2016_ORDER = ["8PSK", "AM-DSB", "AM-SSB", "BPSK", "CPFSK", "GFSK", "PAM4", "QAM16", "QAM64", "QPSK", "WBFM"]
RML2018_ORDER = [
    "32PSK", "16APSK", "32QAM", "FM", "GMSK", "32APSK", "OQPSK", "8ASK",
    "BPSK", "8PSK", "AM-SSB-SC", "4ASK", "16PSK", "64APSK", "128QAM",
    "128APSK", "AM-DSB-SC", "AM-SSB-WC", "64QAM", "QPSK", "256QAM",
    "AM-DSB-WC", "OOK", "16QAM",
]
EVAL_QUALITY_DATASETS = frozenset({"rml2018_1a", "adsb2", "wifi150", "xidian14"})
EVAL_QUALITY_SPLITS = ("val", "test")
# 仅 val/test、不做类别均衡的数据集（test:val = 8:2，不参与 MAE 预训练）
VAL_TEST_ONLY_DATASETS = frozenset({"open_real_data"})
VAL_TEST_SPLIT_RATIO = 0.8
RML2018_EVAL_MIN_SNR = 6.0


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
        "snr": ("f4", np.nan), "mod_label_id": ("i4", -1), "emitter_id": ("i4", -1),
        "source_label_id": ("i4", -1), "global_label_id": ("i4", -1),
    }

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
        self.f.attrs.update({
            "source_path": str(source), "scale_policy": "none",
            "quality_filter": "finite, non-silent, robust-power-8MAD, PAPR<=100, lag1-correlation>=1e-3",
        })
        self.count = 0

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
        for key, ds in self.ds.items():
            ds.resize(end, axis=0)
            value = meta.get(key, defaults.get(key, ds.fillvalue))
            value = np.asarray(value)
            ds[start:end] = value[keep] if value.ndim else value
        self.count = end

    def append_raw(self, iq: np.ndarray, **meta) -> None:
        if not len(iq):
            return
        start, end = self.count, self.count + len(iq)
        self.iq.resize(end, axis=0)
        self.iq[start:end] = iq
        defaults = {"length": self.length, "dataset_id": self.dataset_id, "task_type_id": self.task_id}
        for key, ds in self.ds.items():
            ds.resize(end, axis=0)
            value = meta.get(key, defaults.get(key, ds.fillvalue))
            ds[start:end] = value
        self.count = end

    def close(self):
        self.f.attrs["sample_count"] = self.count
        self.f.close()


class Context:
    def __init__(self, output: Path):
        self.output, self.h5 = output, output / "h5"
        self.maps = self._load_json(output / "label_maps.json", {
            "datasets": {}, "modulations": {}, "emitters": {}, "sources": {}, "task_pools": {},
        })
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
                    "split": "after cleaning, balance classes inside each dataset, then stratified 8:1:1 train/val/test",
                    "split_usage": "*_train.h5: MAE pretrain only; *_val.h5: all validation and model selection; *_test.h5: downstream/finetune training only (not for evaluation)",
                    "eval_quality_datasets": sorted(EVAL_QUALITY_DATASETS),
                    "eval_quality_filter": "val/test only: per-class power 3 MAD, PAPR <= 20, lag1-correlation >= 0.05",
                    "rml2018_eval_snr": f"val/test only: SNR >= {RML2018_EVAL_MIN_SNR} dB",
                },
                "datasets": {},
                "excluded": {
                    "ADSB-1 and ADSB-3": "SHA256 confirms duplicate copies; ADSB-2 is used to avoid leakage",
                    "WIFIDATASET/62ft.rar": "RAR archive only; no RAR extractor is installed",
                    "wisig/ManyTx.pkl.zip": "compressed archive only; not expanded automatically",
                    "XSRPdatav1": "capture dates are present but emitter identities are absent",
                },
            }
        else:
            self.report.setdefault("policy", {})
            self.report["policy"].update({
                "extreme_filter": "finite/non-silent, per-class robust power 4 MAD, PAPR <= 30",
                "npy_label_maps": "numeric Y_*.npy labels are inferred from data; string labels use configured name maps",
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
                    meta = {key: np.asarray(f[key][idx]) for key in Writer.FIELDS if key in f}
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
        split_meta = entry.get("balanced_split") or entry.get("val_test_split") or {}
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


def rebalance_all(ctx: Context) -> None:
    ctx.pool = defaultdict(list)
    dataset_names = sorted(ctx.touched) if ctx.touched else sorted(ctx.report.get("datasets", {}))
    for name in dataset_names:
        entry = ctx.report.get("datasets", {}).get(name)
        if entry is None:
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
        meta: dict[str, list[np.ndarray]] = {key: [] for key in Writer.FIELDS}
        for start in range(0, n, 2048):
            end = min(start + 2048, n)
            iq_parts.append(np.asarray(f["iq"][start:end]))
            for key in Writer.FIELDS:
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
            for key in Writer.FIELDS
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


def open_real_data(ctx: Context, dataset_id: int) -> None:
    """open_realData round7 CSV → RFData H5；仅 test/val（8:2），不做类别均衡，不参与 MAE 预训练。"""
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
    n_test = int(n * VAL_TEST_SPLIT_RATIO)
    split_blocks = [
        ("test", iq_all[:n_test], labels_all[:n_test]),
        ("val", iq_all[n_test:], labels_all[n_test:]),
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
            "removed": dict(removed) if split_name == "test" else {},
            "file": out_path.name,
        }

    entry["val_test_split"] = {
        "label_field": "source_label_id",
        "test_ratio": VAL_TEST_SPLIT_RATIO,
        "no_class_balance": True,
        "total_kept": n,
        "classes": len(set(kept_labels)),
        "splits": {
            split: {"kept": entry["splits"][split]["kept"], "file": entry["splits"][split]["file"]}
            for split in ("test", "val")
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
            config_path=ROOT / "configs" / "downstream_modulation_extra_datasets.yaml"
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


def finalize_task_pools(ctx: Context) -> None:
    """构建 task_pools。

    H5 后缀约定（pool 名中的 train/val 指训练流程角色，不等于 H5 文件名后缀）：
    - *_train.h5  → 仅 pretrain_train（MAE 预训练）
    - *_val.h5    → 各阶段唯一验证集（早停 / 选模 / 报告指标）
    - *_test.h5   → 下游头与微调的训练数据（经 downstream_*_train、clustering_train 等 pool 读取，不作评测）
    """
    rebuild_pools_from_h5(ctx)
    train = sorted(ctx.pool["train"])
    val = sorted(ctx.pool["val"])
    test = sorted(ctx.pool["test"])
    fields = ctx.final_label_fields
    pf = lambda split, **kw: pool_files(split, fields, **kw)
    emitter_downstream = load_emitter_downstream_datasets(config_path=ROOT / "configs" / "emitter_downstream.yaml")
    excluded = load_excluded_datasets(config_path=ROOT / "configs" / "excluded_datasets.yaml")
    downstream_excluded = load_downstream_excluded_datasets(
        config_path=ROOT / "configs" / "downstream_excluded_datasets.yaml"
    )
    modulation_extra = load_downstream_modulation_extra_datasets(
        config_path=ROOT / "configs" / "downstream_modulation_extra_datasets.yaml"
    )

    def _pool(split_files: list[tuple[str, int]], **kwargs: object) -> list[str]:
        return filter_excluded_dataset_pool(pf(split_files, **kwargs), excluded)

    def _downstream_pool(split_files: list[tuple[str, int]], **kwargs: object) -> list[str]:
        return filter_excluded_dataset_pool(_pool(split_files, **kwargs), downstream_excluded)

    def _modulation_extra_pool(split_files: list[tuple[str, int]]) -> list[str]:
        extra = set(modulation_extra)
        return _downstream_pool(
            [(filename, tid) for filename, tid in split_files if tid == 0 and h5_dataset_name(filename) in extra],
        )

    allowed_test = set(_pool(test))
    allowed_val = set(_pool(val))
    pretrain_excluded = set(excluded) | VAL_TEST_ONLY_DATASETS
    downstream_test = set(_downstream_pool(test))
    downstream_val = set(_downstream_pool(val))
    modulation_train = sorted(
        set(_downstream_pool(test, task_id=0, label_field="mod_label_id")) | set(_modulation_extra_pool(test))
    )
    modulation_val = sorted(
        set(_downstream_pool(val, task_id=0, label_field="mod_label_id")) | set(_modulation_extra_pool(val))
    )

    ctx.maps["task_pools"] = {
        "pretrain_train": filter_excluded_dataset_pool(_pool(train), pretrain_excluded),
        "pretrain_val": sorted(filter_excluded_dataset_pool(list(allowed_val), pretrain_excluded)),
        "downstream_modulation_train": modulation_train,
        "downstream_modulation_val": modulation_val,
        "downstream_source_train": _downstream_pool(test, task_id=0, label_field="source_label_id"),
        "downstream_source_val": _downstream_pool(val, task_id=0, label_field="source_label_id"),
        "downstream_emitter_train": filter_emitter_downstream_pool(
            _downstream_pool(test, task_id=1, label_field="emitter_id"),
            emitter_downstream,
        ),
        "downstream_emitter_val": filter_emitter_downstream_pool(
            _downstream_pool(val, task_id=1, label_field="emitter_id"),
            emitter_downstream,
        ),
        "downstream_prediction_train": sorted(downstream_test),
        "downstream_prediction_val": sorted(downstream_val),
        "clustering_train": pool_files_with_global_labels(
            ctx, [(filename, tid) for filename, tid in test if filename in downstream_test]
        ),
        "clustering_val": pool_files_with_global_labels(
            ctx, [(filename, tid) for filename, tid in val if filename in downstream_val]
        ),
    }
    ctx.maps["emitter_downstream_datasets"] = list(emitter_downstream)
    ctx.maps["excluded_datasets"] = list(excluded)
    ctx.maps["downstream_excluded_datasets"] = list(downstream_excluded)
    ctx.maps["downstream_modulation_extra_datasets"] = list(modulation_extra)


def finalize(ctx: Context, *, refine_eval: bool = True) -> None:
    rebalance_all(ctx)
    if refine_eval:
        refine_eval_splits(ctx)
    sync_label_maps_from_balanced(ctx)
    stamp_global_label_ids(ctx)
    finalize_task_pools(ctx)
    ctx.output.mkdir(parents=True, exist_ok=True)
    (ctx.output / "label_maps.json").write_text(json.dumps(ctx.maps, ensure_ascii=False, indent=2), encoding="utf-8")
    (ctx.output / "cleaning_report.json").write_text(json.dumps(ctx.report, ensure_ascii=False, indent=2), encoding="utf-8")


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
        "--refine-eval-splits",
        action="store_true",
        help="仅对 rml2018_1a/adsb2/wifi150/xidian14 的 val/test 应用更严格质量过滤并刷新 task pool",
    )
    args = parser.parse_args()
    ctx = Context(args.output)
    if args.refine_eval_splits:
        refine_eval_splits(ctx)
        stamp_global_label_ids(ctx)
        finalize_task_pools(ctx)
        ctx.output.mkdir(parents=True, exist_ok=True)
        (ctx.output / "label_maps.json").write_text(json.dumps(ctx.maps, ensure_ascii=False, indent=2), encoding="utf-8")
        (ctx.output / "cleaning_report.json").write_text(json.dumps(ctx.report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[refine-eval-splits] {ctx.output}", flush=True)
        return
    if args.sync_label_maps or args.stamp_global_labels or args.stamp_mod_labels:
        if args.sync_label_maps:
            sync_label_maps_from_balanced(ctx)
        mod_stamped: dict[str, int] = {}
        if args.stamp_mod_labels:
            mod_stamped = stamp_mod_label_ids(ctx)
        stamped: dict[str, int] = {}
        if args.stamp_global_labels:
            stamped = stamp_global_label_ids(ctx)
        finalize_task_pools(ctx)
        ctx.output.mkdir(parents=True, exist_ok=True)
        (ctx.output / "label_maps.json").write_text(json.dumps(ctx.maps, ensure_ascii=False, indent=2), encoding="utf-8")
        (ctx.output / "cleaning_report.json").write_text(json.dumps(ctx.report, ensure_ascii=False, indent=2), encoding="utf-8")
        if mod_stamped:
            labeled = sum(mod_stamped.values())
            print(f"[stamp-mod-labels] {len(mod_stamped)} files, {labeled} labeled samples", flush=True)
        if stamped:
            labeled = sum(stamped.values())
            print(f"[stamp-global-labels] {len(stamped)} files, {labeled} labeled samples", flush=True)
        print(f"[sync-label-maps] {ctx.output}", flush=True)
        return
    selected = set(args.datasets)
    jobs = [
        ("electromagnetic_0926", lambda: npy_dataset(ctx, "electromagnetic_0926", 0, NON_EMITTER / "0926电磁数据", 0, "source_label_id", {x: i for i, x in enumerate(["ADSB", "AIS", "AM", "FM", "GPS", "Iridium", "P25"])}, [("train", "train"), ("test", "val")], mirror_mod_label_id=True)),
        ("xidian14", lambda: npy_dataset(ctx, "xidian14", 1, NON_EMITTER / "xidian_npy_cls14", 0, "mod_label_id", None, [("train", "train"), ("val", "val")])),
        ("rml2016_04c", lambda: rml_pickle(ctx, "rml2016_04c", 2, NON_EMITTER / "2016.04C.multisnr.pkl")),
        ("rml2016_10a", lambda: rml_pickle(ctx, "rml2016_10a", 3, NON_EMITTER / "RML2016.10a_dict.pkl")),
        ("rml2016_10b", lambda: rml10b(ctx, 4)),
        ("rml2018_1a", lambda: rml2018(ctx, 5)),
        ("adsb2", lambda: npy_dataset(ctx, "adsb2", 6, EMITTER / "ADSB-2", 1, "emitter_id", None, [("train", "train"), ("val", "val")])),
        ("wifi150", lambda: npy_dataset(ctx, "wifi150", 7, EMITTER / "wifi_cls150", 1, "emitter_id", None, [("train", "train"), ("val", "val"), ("test", "test")])),
        ("communication_emitters", lambda: dat_emitters(ctx, "communication_emitters", 8, EMITTER / "通信辐射源个体识别数据集", 2048, False)),
        ("radar_emitters", lambda: dat_emitters(ctx, "radar_emitters", 9, EMITTER / "雷达辐射源个体识别数据集", 1000, True)),
        ("open_real_data", lambda: open_real_data(ctx, 10)),
    ]
    for name, job in jobs:
        if "all" in selected or name in selected:
            print(f"[build] {name}", flush=True)
            job()
    finalize(ctx)
    print(f"[done] {ctx.output}", flush=True)


if __name__ == "__main__":
    main()
