#!/usr/bin/env python3
"""将 WiSig ManyTx.pkl 流式转换为 npy（供 prepare_datasets.py --datasets wisig 使用）。"""
from __future__ import annotations

import importlib.util
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PKL = ROOT.parent / "个体辐射源数据" / "wisig" / "ManyTx.pkl"
OUT = ROOT.parent / "个体辐射源数据" / "wisig" / "npy_data" / "equalized_data_0"


def _load_wisig_module():
    spec = importlib.util.spec_from_file_location(
        "wisig_manytx",
        ROOT / "resmamba_signal_model" / "data" / "wisig_manytx.py",
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _count_samples(wm):
    counts: Counter = Counter()
    meta_holder: dict[str, object] = {}

    def on_batch(split: str, iq: np.ndarray, labels: np.ndarray) -> None:
        counts[split] += len(iq)

    meta_holder["meta"] = wm.stream_manytx_blocks(
        PKL, wm.WiSigBlockSink(equalized=0, on_batch=on_batch)
    )
    return counts, meta_holder["meta"]


def _write_npy(wm, counts: Counter) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    cursors = {split: 0 for split in counts}
    mmaps = {
        split: {
            "iq": np.lib.format.open_memmap(
                OUT / f"X_{split}.npy",
                mode="w+",
                dtype=np.float32,
                shape=(counts[split], 2, 256),
            ),
            "y": np.lib.format.open_memmap(
                OUT / f"Y_{split}.npy",
                mode="w+",
                dtype=np.int32,
                shape=(counts[split],),
            ),
        }
        for split in counts
    }

    def on_batch(split: str, iq: np.ndarray, labels: np.ndarray) -> None:
        n = len(iq)
        start = cursors[split]
        end = start + n
        mmaps[split]["iq"][start:end] = iq
        mmaps[split]["y"][start:end] = labels
        cursors[split] = end

    wm.stream_manytx_blocks(PKL, wm.WiSigBlockSink(equalized=0, on_batch=on_batch))
    for split, end in cursors.items():
        if end != counts[split]:
            raise RuntimeError(f"{split} 写入数量不一致: {end} != {counts[split]}")
        mmaps[split]["iq"].flush()
        mmaps[split]["y"].flush()


def main() -> None:
    if not PKL.is_file():
        raise FileNotFoundError(f"未找到 {PKL}")

    wm = _load_wisig_module()
    print(f"[convert] pass-1 count: {PKL}", flush=True)
    counts, meta = _count_samples(wm)
    print(f"[convert] counts: {dict(counts)}", flush=True)
    if not counts:
        raise RuntimeError("未解析到任何样本")

    print(f"[convert] pass-2 write: {OUT}", flush=True)
    _write_npy(wm, counts)
    labels = wm.emitter_labels(meta)
    print(f"[convert] done -> {OUT}")
    print(f"[convert] tx={len(meta.tx_list)} rx={len(meta.rx_list)} labels={len(labels)}", flush=True)
    print("[convert] 下一步: python scripts/prepare_datasets.py --output dataset --datasets wisig", flush=True)


if __name__ == "__main__":
    main()
