#!/usr/bin/env python3
"""使用原始 genNPY 逻辑将 ManyTx.pkl 转为 npy（需约 8GB+ 内存）。"""
from __future__ import annotations

import os
import pickle
import random
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PKL = ROOT.parent / "个体辐射源数据" / "wisig" / "ManyTx.pkl"
OUT = ROOT.parent / "个体辐射源数据" / "wisig" / "npy_data"


def ditc_concate(dict_name: dict) -> dict:
    dict_name["train_data"] = np.concatenate(dict_name["train_data"], axis=0)
    dict_name["train_label"] = np.concatenate(dict_name["train_label"], axis=0)
    dict_name["val_data"] = np.concatenate(dict_name["val_data"], axis=0)
    dict_name["val_label"] = np.concatenate(dict_name["val_label"], axis=0)
    dict_name["test_data"] = np.concatenate(dict_name["test_data"], axis=0)
    dict_name["test_label"] = np.concatenate(dict_name["test_label"], axis=0)
    return dict_name


def savenpy(savepath: Path, dict_name: dict) -> None:
    savepath.mkdir(parents=True, exist_ok=True)
    np.save(savepath / "X_train.npy", dict_name["train_data"])
    np.save(savepath / "Y_train.npy", dict_name["train_label"])
    np.save(savepath / "X_val.npy", dict_name["val_data"])
    np.save(savepath / "Y_val.npy", dict_name["val_label"])
    np.save(savepath / "X_test.npy", dict_name["test_data"])
    np.save(savepath / "Y_test.npy", dict_name["test_label"])


def splitdata(data: dict, savepath: Path) -> None:
    tx_list = data["tx_list"]
    rx_list = data["rx_list"]
    capture_date_list = data["capture_date_list"]
    signal = data["data"]

    buckets = {
        0: {
            "train_data": [],
            "train_label": [],
            "val_data": [],
            "val_label": [],
            "test_data": [],
            "test_label": [],
        },
        1: {
            "train_data": [],
            "train_label": [],
            "val_data": [],
            "val_label": [],
            "test_data": [],
            "test_label": [],
        },
    }

    for tx_idx, _tx_name in enumerate(tx_list):
        label_name = tx_idx
        for _rx_idx, _rx_name in enumerate(rx_list):
            for _date_idx, _date in enumerate(capture_date_list):
                for eq_idx in range(2):
                    block = signal[tx_idx][_rx_idx][_date_idx][eq_idx]
                    num = block.shape[0]
                    if num == 0:
                        continue
                    block = np.transpose(block, (0, 2, 1))

                    indices = list(range(num))
                    random.shuffle(indices)
                    num_train = int(num * 0.6)
                    num_val = int(num * 0.2)
                    train_idx = indices[:num_train]
                    val_idx = indices[num_train : num_train + num_val]
                    test_idx = indices[num_train + num_val :]

                    bucket = buckets[eq_idx]
                    bucket["train_data"].append(block[train_idx])
                    bucket["train_label"].append(np.full((len(train_idx), 1), label_name))
                    bucket["val_data"].append(block[val_idx])
                    bucket["val_label"].append(np.full((len(val_idx), 1), label_name))
                    bucket["test_data"].append(block[test_idx])
                    bucket["test_label"].append(np.full((len(test_idx), 1), label_name))

    for eq_idx, bucket in buckets.items():
        merged = ditc_concate(bucket)
        print(
            f"[genNPY] equalized_data_{eq_idx}: "
            f"train={merged['train_data'].shape} val={merged['val_data'].shape} test={merged['test_data'].shape}",
            flush=True,
        )
        savenpy(savepath / f"equalized_data_{eq_idx}", merged)


def main() -> None:
    if not PKL.is_file():
        raise FileNotFoundError(f"未找到 {PKL}")
    print(f"[genNPY] loading {PKL}", flush=True)
    with PKL.open("rb") as f:
        data = pickle.load(f, encoding="latin1")
    random.seed(0)
    splitdata(data, OUT)
    print(f"[genNPY] done -> {OUT}")
    print("[genNPY] 下一步: python scripts/prepare_datasets.py --output dataset --datasets wisig", flush=True)


if __name__ == "__main__":
    main()
