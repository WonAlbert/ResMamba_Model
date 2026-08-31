#!/usr/bin/env python3
"""从 round7 CSV 推断 radar_mod15 每条 H5 样本的采样率，写入 JSON。"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.prepare_datasets import NON_EMITTER, as_iq, quality_mask  # noqa: E402

DEFAULT_SOURCE = NON_EMITTER / "open_realData" / "outputv2" / "round7_dataset"
DEFAULT_H5_DIR = ROOT / "dataset" / "h5"
DEFAULT_OUTPUT = ROOT / "dataset" / "radar_mod15_sample_rates.json"
SPLITS = ("train", "val", "test")


def infer_fs_duration(path: Path) -> tuple[float, float]:
    times: list[float] = []
    with path.open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            times.append(float(row["time(s)"]))
    t = np.asarray(times, dtype=np.float64)
    if t.size < 2:
        raise ValueError(f"{path} 时间列不足")
    fs_hz = float(1.0 / np.median(np.diff(t)))
    duration_us = float((t[-1] - t[0]) * 1e6)
    return fs_hz, duration_us


def build_source_index(source_root: Path, class_names: dict[int, str]) -> dict[bytes, dict]:
    class_dirs = sorted(
        (p for p in source_root.iterdir() if p.is_dir() and p.name.isdigit()),
        key=lambda p: int(p.name),
    )
    if not class_dirs:
        raise FileNotFoundError(f"{source_root} 下未找到类别目录")

    expected_len = int(
        len(np.loadtxt(next(class_dirs[0].glob("*.csv")), delimiter=",", skiprows=1, usecols=1))
    )
    index: dict[bytes, dict] = {}
    duplicates = 0

    for cls_dir in class_dirs:
        cls_id = int(cls_dir.name)
        for csv_path in sorted(cls_dir.glob("*.csv")):
            amp = np.loadtxt(csv_path, delimiter=",", skiprows=1, usecols=1, dtype=np.float64)
            if len(amp) != expected_len:
                continue
            iq = as_iq(amp).astype(np.float32)
            keep, _ = quality_mask(
                iq[None],
                labels=np.asarray([cls_id], dtype=np.int32),
                dataset_name="radar_mod15",
            )
            if not bool(keep[0]):
                continue
            fs_hz, duration_us = infer_fs_duration(csv_path)
            rel_csv = csv_path.relative_to(source_root.parent.parent).as_posix()
            meta = {
                "class_index": cls_id,
                "class_name": class_names[cls_id],
                "source_csv": rel_csv,
                "signal_length": expected_len,
                "sample_rate_hz": fs_hz,
                "duration_us": duration_us,
            }
            key = iq.tobytes()
            if key in index:
                duplicates += 1
            index[key] = meta

    if duplicates:
        print(f"warning: {duplicates} duplicate IQ fingerprints in source index", flush=True)
    return index


def index_h5_split(h5_path: Path, source_index: dict[bytes, dict]) -> list[dict]:
    rows: list[dict] = []
    missing = 0
    with h5py.File(h5_path, "r") as handle:
        iq = handle["iq"]
        labels = np.asarray(handle["mod_label_id"][:], dtype=np.int32)
        n = int(iq.shape[0])
        for idx in range(n):
            key = np.asarray(iq[idx], dtype=np.float32).tobytes()
            meta = source_index.get(key)
            if meta is None:
                missing += 1
                rows.append(
                    {
                        "index": idx,
                        "class_index": int(labels[idx]),
                        "class_name": None,
                        "source_csv": None,
                        "signal_length": int(iq.shape[-1]),
                        "sample_rate_hz": None,
                        "duration_us": None,
                        "match_status": "missing",
                    }
                )
                continue
            if int(meta["class_index"]) != int(labels[idx]):
                raise RuntimeError(
                    f"{h5_path.name}[{idx}]: H5 label {labels[idx]} != source {meta['class_index']}"
                )
            rows.append({**meta, "index": idx, "match_status": "ok"})
    if missing:
        raise RuntimeError(f"{h5_path.name}: {missing}/{n} 样本无法匹配源 CSV")
    return rows


def build_payload(source_root: Path, h5_dir: Path) -> dict:
    summary = json.loads((source_root / "manifests/summary.json").read_text(encoding="utf-8"))
    class_names = {
        int(cls_str): info["source_label"] for cls_str, info in summary["mapping"].items()
    }
    source_index = build_source_index(source_root, class_names)
    splits: dict[str, list[dict]] = {}
    for split in SPLITS:
        h5_path = h5_dir / f"radar_mod15_{split}.h5"
        if not h5_path.is_file():
            raise FileNotFoundError(h5_path)
        splits[split] = index_h5_split(h5_path, source_index)

    return {
        "dataset": "radar_mod15",
        "source_root": str(source_root.resolve()),
        "h5_dir": str(h5_dir.resolve()),
        "inference_method": "sample_rate_hz = 1 / median(diff(time(s))) from source CSV",
        "index_method": "match H5 iq row to source CSV via float32 IQ fingerprint",
        "sample_counts": {split: len(rows) for split, rows in splits.items()},
        "splits": splits,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="生成 radar_mod15 逐样本采样率 JSON")
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--h5-dir", type=Path, default=DEFAULT_H5_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = build_payload(args.source_root, args.h5_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    counts = payload["sample_counts"]
    print(
        f"Wrote {args.output}  train={counts['train']} val={counts['val']} test={counts['test']}",
        flush=True,
    )


if __name__ == "__main__":
    main()
