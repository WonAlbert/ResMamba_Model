#!/usr/bin/env python3
"""25 核并行转换 radcom_ota，并按 8:1:1 划分 train/val/test。"""
from __future__ import annotations

import argparse
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from resmamba_signal_model.thread_env import normalize_thread_env

normalize_thread_env()

_PREPARE = ROOT / "scripts" / "prepare_datasets.py"
_spec = importlib.util.spec_from_file_location("prepare_datasets", _PREPARE)
pd = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(pd)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="多核并行转换 radcom_ota 并 8:1:1 划分")
    parser.add_argument("--output", type=Path, default=ROOT / "dataset")
    parser.add_argument("--jobs", type=int, default=25, help="并行子进程数")
    parser.add_argument(
        "--skip-split",
        action="store_true",
        help="仅构建 train H5，不执行 8:1:1 rebalance",
    )
    return parser.parse_args()


def run_parallel_parts(output: Path, jobs: int) -> None:
    name = "radcom_ota"
    parts_dir = output / "h5" / f"{name}__parts"
    parts_dir.mkdir(parents=True, exist_ok=True)
    for old in parts_dir.glob("part_*.h5"):
        old.unlink(missing_ok=True)

    build_script = ROOT / "scripts" / "radcom_ota_build_part.py"
    procs: list[subprocess.Popen] = []
    print(f"[parallel-radcom] launching {jobs} workers", flush=True)
    for part in range(jobs):
        cmd = [
            sys.executable,
            str(build_script),
            "--output",
            str(output),
            "--part",
            str(part),
            "--parts",
            str(jobs),
        ]
        procs.append(subprocess.Popen(cmd))

    failed = 0
    for i, proc in enumerate(procs):
        code = proc.wait()
        if code != 0:
            failed += 1
            print(f"[parallel-radcom] part {i} failed exit={code}", flush=True)
    if failed:
        raise RuntimeError(f"{failed}/{jobs} 分片构建失败")


def merge_and_register(ctx: pd.Context, name: str, dataset_id: int, src: Path) -> None:
    mod_labels, sig_labels = pd.build_radcom_label_maps(src)
    ctx.maps["datasets"][str(dataset_id)] = name
    ctx.maps["modulations"][name] = mod_labels
    ctx.maps["sources"][name] = sig_labels

    parts_dir = ctx.h5 / f"{name}__parts"
    part_paths = sorted(parts_dir.glob("part_*.h5"))
    out_train = ctx.h5 / f"{name}_train.h5"
    out_train.unlink(missing_ok=True)
    kept = pd._merge_h5_parts(part_paths, out_train, dataset_id, 0, src)
    for part in part_paths:
        part.unlink(missing_ok=True)
    if parts_dir.exists():
        parts_dir.rmdir()

    raw = 567_000
    removed: dict[str, int] = {}
    ctx.touched.add(name)
    entry = ctx.report["datasets"].setdefault(name, {"labels": mod_labels, "splits": {}})
    entry["labels"] = mod_labels
    entry["splits"]["train"] = {
        "raw": raw,
        "kept": kept,
        "removed": removed,
        "file": out_train.name,
    }
    print(f"[parallel-radcom] merged kept={kept:,}", flush=True)


def main() -> None:
    args = parse_args()
    name = "radcom_ota"
    dataset_id = 15
    src = pd.EXTERNAL / "radarcommdataset" / "RadComOta2.45GHz.hdf5"
    jobs = max(1, int(args.jobs))

    ctx = pd.Context(args.output)
    pd.clear_dataset_h5(ctx, name)
    for suffix in ("train", "val", "test"):
        (ctx.h5 / f"{name}_{suffix}.h5").unlink(missing_ok=True)

    run_parallel_parts(args.output, jobs)
    merge_and_register(ctx, name, dataset_id, src)
    pd.write_build_cache(ctx, name)

    if not args.skip_split:
        entry = ctx.report["datasets"][name]
        _final_files, final_report = pd.rebalance_dataset(ctx, name, entry)
        entry["balanced_split"] = final_report
        print(
            f"[split 8:1:1] classes={final_report['classes']} "
            f"per_class={final_report['balanced_per_class']} "
            f"train={final_report['splits']['train']['kept']} "
            f"val={final_report['splits']['val']['kept']} "
            f"test={final_report['splits']['test']['kept']}",
            flush=True,
        )

    pd.sync_label_maps_from_balanced(ctx)
    pd.stamp_global_label_ids(ctx)
    pd.finalize_task_pools(ctx)
    ctx.output.mkdir(parents=True, exist_ok=True)
    (ctx.output / "label_maps.json").write_text(
        json.dumps(ctx.maps, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (ctx.output / "cleaning_report.json").write_text(
        json.dumps(ctx.report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"[done] {ctx.output / 'h5' / name}_*.h5", flush=True)


if __name__ == "__main__":
    main()
