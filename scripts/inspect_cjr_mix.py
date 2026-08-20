#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from resmamba_signal_model.data.cjr_mix import (
    build_cjr_mix_dataset,
    resolve_cjr_mix_root,
    summarize_cjr_mix,
)
from resmamba_signal_model.thread_env import normalize_thread_env

normalize_thread_env()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="检查 CJR-mix parquet 数据集概况与样本字段")
    parser.add_argument(
        "--rfdata-root",
        default=None,
        help="RFData 根目录，默认读取环境变量 RFDATA_ROOT 或项目 dataset/",
    )
    parser.add_argument("--split", choices=("train", "val"), default="train")
    parser.add_argument("--num-samples", type=int, default=3, help="打印前 N 条样本摘要")
    parser.add_argument("--snr-threshold", type=float, default=None, help="与训练脚本一致的 SNR 过滤阈值")
    parser.add_argument("--iq-normalize", choices=("abs", "joint_power", "none"), default="abs")
    parser.add_argument("--json", action="store_true", help="仅输出 JSON 摘要")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = summarize_cjr_mix(args.rfdata_root)
    if args.json:
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return

    data_root = resolve_cjr_mix_root(args.rfdata_root)
    print(f"数据根目录: {data_root}")
    for split_name, info in summary["splits"].items():
        print(f"  [{split_name}] files={info['num_files']} rows={info['num_rows']:,}")

    ds = build_cjr_mix_dataset(
        args.rfdata_root,
        split=args.split,  # type: ignore[arg-type]
        iq_normalize=args.iq_normalize,  # type: ignore[arg-type]
        snr_threshold=args.snr_threshold,
        max_samples=max(args.num_samples, 1),
    )
    labels: Counter[int] = Counter()
    lengths: list[int] = []
    snrs: list[float] = []

    for i in range(min(len(ds), args.num_samples)):
        sample = ds[i]
        labels[int(sample["mod_label_id"])] += 1
        lengths.append(int(sample["length"]))
        snrs.append(float(sample["snr"]))
        print(
            f"sample[{i}] iq={tuple(sample['iq'].shape)} "
            f"label={sample['mod_label_id']} snr={sample['snr']:.1f} "
            f"fs={sample['sample_rate_hz']:.0f} name={sample['dataset_name']!r}"
        )

    print(
        f"抽样标签分布: {dict(sorted(labels.items()))} | "
        f"长度={lengths} | SNR={[round(x, 1) for x in snrs]}"
    )


if __name__ == "__main__":
    main()
