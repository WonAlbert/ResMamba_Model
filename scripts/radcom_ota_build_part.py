#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
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
    parser = argparse.ArgumentParser(description="构建 radcom_ota 分片 H5")
    parser.add_argument("--output", type=Path, default=ROOT / "dataset")
    parser.add_argument("--part", type=int, required=True)
    parser.add_argument("--parts", type=int, default=25)
    parser.add_argument("--dataset-id", type=int, default=15)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    name = "radcom_ota"
    src = pd.EXTERNAL / "radarcommdataset" / "RadComOta2.45GHz.hdf5"
    mod_labels, sig_labels = pd.build_radcom_label_maps(src)
    with __import__("h5py").File(src, "r") as f:
        keys = list(f.keys())

    n_parts = max(1, args.parts)
    part = args.part
    if part < 0 or part >= n_parts:
        raise SystemExit(f"part 须在 [0, {n_parts}) 内")
    chunk = [k for i, k in enumerate(keys) if i % n_parts == part]

    parts_dir = args.output / "h5" / f"{name}__parts"
    parts_dir.mkdir(parents=True, exist_ok=True)
    part_path = parts_dir / f"part_{part:03d}.h5"
    part_path.unlink(missing_ok=True)

    out_path, raw, kept, removed = pd._radcom_hdf5_part_worker(
        chunk,
        str(src),
        str(part_path),
        args.dataset_id,
        mod_labels,
        sig_labels,
        None,
    )
    print(f"[part {part}/{n_parts}] raw={raw} kept={kept} removed={removed} -> {out_path}", flush=True)


if __name__ == "__main__":
    main()
