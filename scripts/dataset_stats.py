#!/usr/bin/env python3
"""统计模型所用 H5 数据集的样本数、长度、SNR 与类别分布，导出 CSV。"""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
import sys

import h5py
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from resmamba_signal_model.thread_env import normalize_thread_env

normalize_thread_env()

SPLITS = ("train", "test", "val")
CHUNK = 65536

# 各库主类别字段（与下游任务语义对齐）
PRIMARY_LABEL_FIELD = {
    "radar_mod15": "source_label_id",
    "radchar": "mod_label_id",
    "cjr_mix": "mod_label_id",
    "rml2016_04c": "mod_label_id",
    "rml2016_10a": "mod_label_id",
    "rml2016_10b": "mod_label_id",
    "panoradio_hf": "mod_label_id",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="导出数据集样本/长度/SNR/类别统计 CSV")
    parser.add_argument(
        "--rfdata-root",
        type=Path,
        default=None,
        help="RFData 根目录（默认 RFDATA_ROOT 或项目 dataset/）",
    )
    parser.add_argument(
        "--h5-dir",
        type=Path,
        default=None,
        help="H5 目录（默认 <rfdata-root>/h5）",
    )
    parser.add_argument(
        "--datasets-config",
        type=Path,
        default=ROOT / "configs" / "datasets.yaml",
        help="数据集白名单 YAML",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="CSV 输出目录（默认 <rfdata-root>/stats）",
    )
    parser.add_argument(
        "--include-excluded",
        action="store_true",
        help="同时统计 excluded 中的库（若 H5 存在）",
    )
    return parser.parse_args()


def resolve_rfdata_root(cli: Path | None) -> Path:
    if cli is not None:
        return cli.resolve()
    import os

    env = os.environ.get("RFDATA_ROOT")
    if env:
        return Path(env).resolve()
    return (ROOT / "dataset").resolve()


def load_whitelist(cfg_path: Path, *, include_excluded: bool) -> list[str]:
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
    names: list[str] = []
    seen: set[str] = set()
    for section, payload in cfg.items():
        if not isinstance(payload, dict):
            continue
        if section == "excluded" and not include_excluded:
            continue
        for name in payload.get("datasets") or []:
            stem = str(name).strip()
            if not stem or stem in seen:
                continue
            seen.add(stem)
            names.append(stem)
    return names


def invert_name_map(raw: dict | None) -> dict[int, str]:
    if not isinstance(raw, dict):
        return {}
    out: dict[int, str] = {}
    for name, idx in raw.items():
        try:
            out[int(idx)] = str(name)
        except (TypeError, ValueError):
            continue
    return out


def class_name_lookup(label_maps: dict, dataset: str, field: str) -> dict[int, str]:
    if field == "source_label_id":
        return invert_name_map(label_maps.get("sources", {}).get(dataset))
    if field in {"mod_label_id", "canonical_mod_label_id"}:
        local = invert_name_map(label_maps.get("modulations", {}).get(dataset))
        if local:
            return local
        return invert_name_map(label_maps.get("sources", {}).get(dataset))
    return {}


def parse_stem_split(filename: str) -> tuple[str, str] | None:
    stem = Path(filename).stem
    for split in SPLITS:
        suffix = f"_{split}"
        if stem.endswith(suffix):
            return stem[: -len(suffix)], split
    return None


def _snr_key(value: float) -> str:
    # float16 读出可能有细微误差；整数 SNR 归一成整数字符串
    if abs(value - round(value)) < 1e-3:
        return str(int(round(value)))
    return f"{value:.4g}"


def summarize_file(
    path: Path,
    *,
    dataset: str,
    split: str,
    label_maps: dict,
) -> dict:
    primary_field = PRIMARY_LABEL_FIELD.get(dataset, "mod_label_id")
    name_map = class_name_lookup(label_maps, dataset, primary_field)

    with h5py.File(path, "r") as handle:
        iq = handle["iq"]
        n = int(iq.shape[0])
        length = int(iq.shape[-1]) if iq.ndim >= 3 else int(handle.attrs.get("signal_length", -1))
        attrs_length = int(handle.attrs.get("signal_length", length))
        sample_rate = handle.attrs.get("sampling_rate_hz", None)
        snr_policy = str(handle.attrs.get("snr_policy", ""))
        dataset_id = int(handle.attrs.get("dataset_id", -1))

        class_counts: Counter[int] = Counter()
        snr_counts: Counter[str] = Counter()
        snr_known = 0
        snr_missing = 0
        snr_min = float("inf")
        snr_max = float("-inf")
        snr_sum = 0.0

        has_primary = primary_field in handle and int(handle[primary_field].shape[0]) == n
        has_snr = "snr" in handle and int(handle["snr"].shape[0]) == n

        for start in range(0, n, CHUNK):
            end = min(start + CHUNK, n)
            if has_primary:
                labels = np.asarray(handle[primary_field][start:end], dtype=np.int64)
                class_counts.update(int(v) for v in labels[labels >= 0])
            if has_snr:
                snr_values = np.asarray(handle["snr"][start:end], dtype=np.float64)
                valid = np.isfinite(snr_values) & (snr_values > -900)
                finite = snr_values[valid]
                snr_known += int(valid.sum())
                snr_missing += int((~valid).sum())
                if finite.size:
                    snr_min = min(snr_min, float(finite.min()))
                    snr_max = max(snr_max, float(finite.max()))
                    snr_sum += float(finite.sum())
                    snr_counts.update(_snr_key(float(v)) for v in finite)
            else:
                snr_missing += end - start

    return {
        "dataset": dataset,
        "split": split,
        "file": path.name,
        "n_samples": n,
        "signal_length": attrs_length if attrs_length > 0 else length,
        "iq_shape": f"(N, 2, {length})",
        "dataset_id": dataset_id,
        "sample_rate_hz": float(sample_rate) if sample_rate is not None else "",
        "snr_policy": snr_policy,
        "primary_label_field": primary_field,
        "n_classes": len(class_counts),
        "class_counts": class_counts,
        "class_name_map": name_map,
        "snr_known": snr_known,
        "snr_missing": snr_missing,
        "snr_min": snr_min if snr_known else "",
        "snr_max": snr_max if snr_known else "",
        "snr_mean": (snr_sum / snr_known) if snr_known else "",
        "snr_counts": snr_counts,
        "bytes": int(path.stat().st_size),
    }


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})


def main() -> None:
    args = parse_args()
    rfdata_root = resolve_rfdata_root(args.rfdata_root)
    h5_dir = (args.h5_dir or (rfdata_root / "h5")).resolve()
    out_dir = (args.output_dir or (rfdata_root / "stats")).resolve()
    label_maps_path = rfdata_root / "label_maps.json"
    label_maps = json.loads(label_maps_path.read_text(encoding="utf-8")) if label_maps_path.exists() else {}

    whitelist = load_whitelist(args.datasets_config, include_excluded=args.include_excluded)
    if not whitelist:
        raise SystemExit(f"未在 {args.datasets_config} 解析到任何数据集")

    # 按白名单顺序；同一库三个 split
    summaries: list[dict] = []
    missing_files: list[str] = []
    for dataset in whitelist:
        for split in SPLITS:
            path = h5_dir / f"{dataset}_{split}.h5"
            if not path.exists():
                missing_files.append(str(path))
                continue
            summaries.append(summarize_file(path, dataset=dataset, split=split, label_maps=label_maps))

    if not summaries:
        raise SystemExit(f"在 {h5_dir} 未找到白名单内任何 H5")

    # --- 1) 按文件 / split 汇总 ---
    summary_rows = []
    for s in summaries:
        snr_levels = sorted(s["snr_counts"].keys(), key=lambda x: float(x))
        summary_rows.append(
            {
                "dataset": s["dataset"],
                "split": s["split"],
                "file": s["file"],
                "n_samples": s["n_samples"],
                "signal_length": s["signal_length"],
                "iq_shape": s["iq_shape"],
                "dataset_id": s["dataset_id"],
                "sample_rate_hz": s["sample_rate_hz"],
                "primary_label_field": s["primary_label_field"],
                "n_classes": s["n_classes"],
                "snr_policy": s["snr_policy"],
                "snr_known": s["snr_known"],
                "snr_missing": s["snr_missing"],
                "snr_min_db": s["snr_min"],
                "snr_max_db": s["snr_max"],
                "snr_mean_db": f"{s['snr_mean']:.4f}" if s["snr_mean"] != "" else "",
                "snr_levels_db": ";".join(snr_levels),
                "file_bytes": s["bytes"],
            }
        )

    # --- 2) 按数据集合计（三 split 合并） ---
    by_ds: dict[str, list[dict]] = defaultdict(list)
    for s in summaries:
        by_ds[s["dataset"]].append(s)

    dataset_total_rows = []
    for dataset in whitelist:
        parts = by_ds.get(dataset) or []
        if not parts:
            continue
        n_total = sum(p["n_samples"] for p in parts)
        class_counts: Counter[int] = Counter()
        snr_counts: Counter[str] = Counter()
        snr_known = 0
        snr_missing = 0
        snr_min = float("inf")
        snr_max = float("-inf")
        snr_sum = 0.0
        for p in parts:
            class_counts.update(p["class_counts"])
            snr_counts.update(p["snr_counts"])
            snr_known += p["snr_known"]
            snr_missing += p["snr_missing"]
            if p["snr_known"]:
                snr_min = min(snr_min, float(p["snr_min"]))
                snr_max = max(snr_max, float(p["snr_max"]))
                snr_sum += float(p["snr_mean"]) * int(p["snr_known"])
        lengths = sorted({int(p["signal_length"]) for p in parts})
        snr_levels = sorted(snr_counts.keys(), key=lambda x: float(x))
        split_counts = {p["split"]: p["n_samples"] for p in parts}
        dataset_total_rows.append(
            {
                "dataset": dataset,
                "n_samples_total": n_total,
                "n_train": split_counts.get("train", 0),
                "n_test": split_counts.get("test", 0),
                "n_val": split_counts.get("val", 0),
                "signal_length": lengths[0] if len(lengths) == 1 else ";".join(map(str, lengths)),
                "n_classes": len(class_counts),
                "primary_label_field": parts[0]["primary_label_field"],
                "snr_policy": parts[0]["snr_policy"],
                "snr_known": snr_known,
                "snr_missing": snr_missing,
                "snr_min_db": snr_min if snr_known else "",
                "snr_max_db": snr_max if snr_known else "",
                "snr_mean_db": f"{(snr_sum / snr_known):.4f}" if snr_known else "",
                "snr_levels_db": ";".join(snr_levels),
                "sample_rate_hz": parts[0]["sample_rate_hz"],
            }
        )

    # --- 3) 类别分布（按 split + 合计） ---
    class_rows = []
    for s in summaries:
        name_map = s["class_name_map"]
        for class_id, count in sorted(s["class_counts"].items()):
            class_rows.append(
                {
                    "dataset": s["dataset"],
                    "split": s["split"],
                    "label_field": s["primary_label_field"],
                    "class_id": class_id,
                    "class_name": name_map.get(class_id, str(class_id)),
                    "n_samples": count,
                    "pct_of_split": f"{100.0 * count / s['n_samples']:.4f}" if s["n_samples"] else "0",
                }
            )
    # 合计行
    for dataset in whitelist:
        parts = by_ds.get(dataset) or []
        if not parts:
            continue
        merged: Counter[int] = Counter()
        for p in parts:
            merged.update(p["class_counts"])
        n_total = sum(p["n_samples"] for p in parts)
        name_map = parts[0]["class_name_map"]
        field = parts[0]["primary_label_field"]
        for class_id, count in sorted(merged.items()):
            class_rows.append(
                {
                    "dataset": dataset,
                    "split": "all",
                    "label_field": field,
                    "class_id": class_id,
                    "class_name": name_map.get(class_id, str(class_id)),
                    "n_samples": count,
                    "pct_of_split": f"{100.0 * count / n_total:.4f}" if n_total else "0",
                }
            )

    # --- 4) SNR 分布 ---
    snr_rows = []
    for s in summaries:
        for snr_db, count in sorted(s["snr_counts"].items(), key=lambda kv: float(kv[0])):
            snr_rows.append(
                {
                    "dataset": s["dataset"],
                    "split": s["split"],
                    "snr_db": snr_db,
                    "n_samples": count,
                    "pct_of_split": f"{100.0 * count / s['n_samples']:.4f}" if s["n_samples"] else "0",
                }
            )
    for dataset in whitelist:
        parts = by_ds.get(dataset) or []
        if not parts:
            continue
        merged: Counter[str] = Counter()
        for p in parts:
            merged.update(p["snr_counts"])
        n_total = sum(p["n_samples"] for p in parts)
        for snr_db, count in sorted(merged.items(), key=lambda kv: float(kv[0])):
            snr_rows.append(
                {
                    "dataset": dataset,
                    "split": "all",
                    "snr_db": snr_db,
                    "n_samples": count,
                    "pct_of_split": f"{100.0 * count / n_total:.4f}" if n_total else "0",
                }
            )

    # --- 5) 类别 × SNR 交叉（合计） ---
    # 为控制体积，仅输出 all split；若需要可再扫一遍 H5
    cross_rows = []
    for dataset in whitelist:
        parts = by_ds.get(dataset) or []
        if not parts:
            continue
        primary_field = parts[0]["primary_label_field"]
        name_map = parts[0]["class_name_map"]
        cross: Counter[tuple[int, str]] = Counter()
        for p in parts:
            path = h5_dir / p["file"]
            with h5py.File(path, "r") as handle:
                n = int(handle["iq"].shape[0])
                has_primary = primary_field in handle
                has_snr = "snr" in handle
                for start in range(0, n, CHUNK):
                    end = min(start + CHUNK, n)
                    if not (has_primary and has_snr):
                        continue
                    labels = np.asarray(handle[primary_field][start:end], dtype=np.int64)
                    snrs = np.asarray(handle["snr"][start:end], dtype=np.float64)
                    valid = (labels >= 0) & np.isfinite(snrs) & (snrs > -900)
                    for lab, snr in zip(labels[valid], snrs[valid], strict=True):
                        cross[(int(lab), _snr_key(float(snr)))] += 1
        for (class_id, snr_db), count in sorted(cross.items(), key=lambda kv: (kv[0][0], float(kv[0][1]))):
            cross_rows.append(
                {
                    "dataset": dataset,
                    "split": "all",
                    "label_field": primary_field,
                    "class_id": class_id,
                    "class_name": name_map.get(class_id, str(class_id)),
                    "snr_db": snr_db,
                    "n_samples": count,
                }
            )

    write_csv(
        out_dir / "dataset_summary_by_split.csv",
        summary_rows,
        [
            "dataset",
            "split",
            "file",
            "n_samples",
            "signal_length",
            "iq_shape",
            "dataset_id",
            "sample_rate_hz",
            "primary_label_field",
            "n_classes",
            "snr_policy",
            "snr_known",
            "snr_missing",
            "snr_min_db",
            "snr_max_db",
            "snr_mean_db",
            "snr_levels_db",
            "file_bytes",
        ],
    )
    write_csv(
        out_dir / "dataset_summary_total.csv",
        dataset_total_rows,
        [
            "dataset",
            "n_samples_total",
            "n_train",
            "n_test",
            "n_val",
            "signal_length",
            "n_classes",
            "primary_label_field",
            "snr_policy",
            "snr_known",
            "snr_missing",
            "snr_min_db",
            "snr_max_db",
            "snr_mean_db",
            "snr_levels_db",
            "sample_rate_hz",
        ],
    )
    write_csv(
        out_dir / "dataset_class_counts.csv",
        class_rows,
        ["dataset", "split", "label_field", "class_id", "class_name", "n_samples", "pct_of_split"],
    )
    write_csv(
        out_dir / "dataset_snr_counts.csv",
        snr_rows,
        ["dataset", "split", "snr_db", "n_samples", "pct_of_split"],
    )
    write_csv(
        out_dir / "dataset_class_snr_counts.csv",
        cross_rows,
        ["dataset", "split", "label_field", "class_id", "class_name", "snr_db", "n_samples"],
    )

    meta = {
        "rfdata_root": str(rfdata_root),
        "h5_dir": str(h5_dir),
        "datasets_config": str(args.datasets_config.resolve()),
        "whitelist": whitelist,
        "n_files_summarized": len(summaries),
        "missing_files": missing_files,
        "outputs": [
            "dataset_summary_by_split.csv",
            "dataset_summary_total.csv",
            "dataset_class_counts.csv",
            "dataset_snr_counts.csv",
            "dataset_class_snr_counts.csv",
        ],
    }
    (out_dir / "dataset_stats_meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print(f"Wrote {len(summaries)} H5 summaries → {out_dir}")
    for row in dataset_total_rows:
        print(
            f"  {row['dataset']}: N={row['n_samples_total']} "
            f"L={row['signal_length']} classes={row['n_classes']} "
            f"SNR=[{row['snr_min_db']}, {row['snr_max_db']}]"
        )
    if missing_files:
        print(f"Missing {len(missing_files)} expected files (see dataset_stats_meta.json)")


if __name__ == "__main__":
    main()
