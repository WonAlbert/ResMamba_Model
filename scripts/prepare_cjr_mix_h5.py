#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import h5py
import numpy as np

from resmamba_signal_model.data.cjr_mix import (
    build_cjr_mix_dataset,
    list_parquet_files,
    parse_iq_array,
    require_pyarrow,
    resolve_cjr_mix_root,
)
from resmamba_signal_model.thread_env import normalize_thread_env

normalize_thread_env()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="将 CJR-mix parquet 转为 ResMamba RFData H5（默认仅 train）"
    )
    parser.add_argument(
        "--splits",
        nargs="*",
        default=["train"],
        choices=("train", "val"),
        help="要转换的 split，默认仅 train",
    )
    parser.add_argument("--rfdata-root", default=None, help="默认 dataset/")
    parser.add_argument("--output", default=None, help="H5 输出目录，默认 <rfdata-root>/h5")
    parser.add_argument("--dataset-id", type=int, default=31, help="写入 H5 的 dataset_id")
    parser.add_argument("--task-type-id", type=int, default=0, help="调制任务 task_type_id=0")
    parser.add_argument("--iq-normalize", choices=("abs", "joint_power", "none"), default="abs")
    parser.add_argument("--snr-threshold", type=float, default=None)
    parser.add_argument("--max-samples", type=int, default=None, help="每个 split 最多转换样本数（调试用）")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--skip-quality-filter", action="store_true", help="跳过简易质量过滤")
    return parser.parse_args()


def quality_mask(iq: np.ndarray) -> np.ndarray:
    finite = np.isfinite(iq).all(axis=(-2, -1))
    power = np.square(iq).sum(axis=-2).mean(axis=-1)
    return finite & (power > np.finfo(np.float32).tiny)


class H5Writer:
    def __init__(self, path: Path, signal_length: int, dataset_id: int, task_type_id: int) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self.signal_length = signal_length
        self.dataset_id = dataset_id
        self.task_type_id = task_type_id
        self.count = 0
        self.f = h5py.File(path, "w")
        chunk = max(1, min(256, (8 << 20) // max(1, 2 * signal_length * 4)))
        self.iq = self.f.create_dataset(
            "iq",
            (0, 2, signal_length),
            maxshape=(None, 2, signal_length),
            dtype="f4",
            chunks=(chunk, 2, signal_length),
            compression="lzf",
        )
        self.fields = {
            "length": self.f.create_dataset("length", (0,), maxshape=(None,), dtype="i4", chunks=(1024,)),
            "dataset_id": self.f.create_dataset("dataset_id", (0,), maxshape=(None,), dtype="i4", chunks=(1024,)),
            "task_type_id": self.f.create_dataset("task_type_id", (0,), maxshape=(None,), dtype="i1", chunks=(1024,)),
            "snr": self.f.create_dataset("snr", (0,), maxshape=(None,), dtype="f4", chunks=(1024,)),
            "mod_label_id": self.f.create_dataset("mod_label_id", (0,), maxshape=(None,), dtype="i4", chunks=(1024,)),
            "emitter_id": self.f.create_dataset("emitter_id", (0,), maxshape=(None,), dtype="i4", chunks=(1024,)),
            "source_label_id": self.f.create_dataset("source_label_id", (0,), maxshape=(None,), dtype="i4", chunks=(1024,)),
            "global_label_id": self.f.create_dataset("global_label_id", (0,), maxshape=(None,), dtype="i4", chunks=(1024,)),
        }
        self.f.attrs.update(
            {
                "source_dataset": "LapplandSaluzzo/CJR-mix",
                "scale_policy": "abs_or_joint_power_at_convert",
            }
        )

    def append(self, batch: list[dict]) -> None:
        if not batch:
            return
        iq = np.stack([item["iq"] for item in batch], axis=0).astype(np.float32, copy=False)
        start, end = self.count, self.count + len(batch)
        self.iq.resize(end, axis=0)
        self.iq[start:end] = iq
        for key, ds in self.fields.items():
            ds.resize(end, axis=0)
            if key == "length":
                ds[start:end] = [int(item["length"]) for item in batch]
            elif key == "dataset_id":
                ds[start:end] = self.dataset_id
            elif key == "task_type_id":
                ds[start:end] = self.task_type_id
            elif key == "snr":
                ds[start:end] = [float(item["snr"]) for item in batch]
            elif key == "mod_label_id":
                ds[start:end] = [int(item["mod_label_id"]) for item in batch]
            elif key == "source_label_id":
                ds[start:end] = [int(item["source_label_id"]) for item in batch]
            elif key in ("emitter_id", "global_label_id"):
                ds[start:end] = -1
        self.count = end

    def close(self) -> None:
        self.f.close()


def convert_split(
    *,
    data_root: Path,
    split: str,
    output_dir: Path,
    dataset_id: int,
    task_type_id: int,
    iq_normalize: str,
    snr_threshold: float | None,
    max_samples: int | None,
    batch_size: int,
    skip_quality_filter: bool,
) -> dict[str, int | str]:
    require_pyarrow()
    ds = build_cjr_mix_dataset(
        data_root.parent,
        split=split,  # type: ignore[arg-type]
        iq_normalize=iq_normalize,  # type: ignore[arg-type]
        snr_threshold=snr_threshold,
        dataset_id=dataset_id,
        task_type_id=task_type_id,
        max_samples=max_samples,
    )
    if not len(ds):
        raise RuntimeError(f"{split} 数据集为空")

    probe = ds[0]
    signal_length = int(probe["length"])
    out_path = output_dir / f"cjr_mix_{split}.h5"
    writer = H5Writer(out_path, signal_length, dataset_id, task_type_id)
    removed = Counter()
    batch: list[dict] = []

    for idx in range(len(ds)):
        sample = ds[idx]
        iq = sample["iq"].numpy()
        if not skip_quality_filter and not quality_mask(iq).all():
            removed["quality"] += 1
            continue
        batch.append(
            {
                "iq": iq,
                "length": int(sample["length"]),
                "snr": float(sample["snr"]),
                "mod_label_id": int(sample["mod_label_id"]),
                "source_label_id": int(sample["source_label_id"]),
            }
        )
        if len(batch) >= batch_size:
            writer.append(batch)
            batch.clear()

    writer.append(batch)
    writer.close()
    return {
        "split": split,
        "path": str(out_path),
        "written": writer.count,
        "removed": dict(removed),
        "signal_length": signal_length,
        "parquet_files": len(list_parquet_files(data_root, split)),  # type: ignore[arg-type]
    }


def main() -> None:
    args = parse_args()
    data_root = resolve_cjr_mix_root(args.rfdata_root)
    output_dir = Path(args.output or (data_root.parent / "h5")).resolve()
    reports = []
    for split in args.splits:
        reports.append(
            convert_split(
                data_root=data_root,
                split=split,
                output_dir=output_dir,
                dataset_id=args.dataset_id,
                task_type_id=args.task_type_id,
                iq_normalize=args.iq_normalize,
                snr_threshold=args.snr_threshold,
                max_samples=args.max_samples,
                batch_size=args.batch_size,
                skip_quality_filter=args.skip_quality_filter,
            )
        )
    print("CJR-mix -> H5 完成：")
    for item in reports:
        print(
            f"  {item['split']}: {item['written']:,} samples -> {item['path']} "
            f"(L={item['signal_length']}, parquet_files={item['parquet_files']}, removed={item['removed']})"
        )


if __name__ == "__main__":
    main()
