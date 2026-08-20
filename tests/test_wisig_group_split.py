from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pytest

from resmamba_signal_model.data.contracts import MISSING_METADATA
from resmamba_signal_model.data.wisig_manytx import (
    WISIG_GROUP_SPLIT_STRATEGY,
    WiSigBlockSink,
)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import prepare_datasets


def test_wisig_assigns_whole_receiver_day_group_to_one_split() -> None:
    group_a = WiSigBlockSink.group_id(1, 2, 0)
    group_b = WiSigBlockSink.group_id(3, 4, 0)
    calls: list[tuple[str, np.ndarray, np.ndarray, dict[str, str]]] = []

    def on_batch(
        split: str,
        iq: np.ndarray,
        labels: np.ndarray,
        metadata: dict[str, str],
    ) -> None:
        calls.append((split, iq, labels, metadata))

    sink = WiSigBlockSink(
        on_batch=on_batch,
        group_assignments={group_a: "train", group_b: "val"},
    )
    block = np.arange(5 * 16 * 2, dtype=np.float32).reshape(5, 16, 2)
    sink.on_block(block, tx_i=0, rx_i=1, day_i=2, eq_i=0)
    # 不同 Tx 但同一 Rx/Day/capture 仍必须进入同一个 split。
    sink.on_block(block, tx_i=7, rx_i=1, day_i=2, eq_i=0)
    sink.on_block(block[:3], tx_i=2, rx_i=3, day_i=4, eq_i=0)

    assert [call[0] for call in calls] == ["train", "train", "val"]
    assert [len(call[1]) for call in calls] == [5, 5, 3]
    assert calls[0][1].shape == (5, 2, 16)
    assert calls[0][2].tolist() == [0] * 5
    assert calls[1][2].tolist() == [7] * 5
    assert calls[0][3]["capture_id"] == group_a
    assert calls[0][3]["receiver_id"] == "wisig:rx=1"
    assert calls[0][3]["session_id"] == "wisig:day=2"
    assert calls[0][3]["channel_id"] == MISSING_METADATA
    assert sink.group_assignments[group_a] == "train"


def test_wisig_split_is_deterministic_and_old_callback_still_works() -> None:
    first: list[str] = []
    second: list[str] = []

    def old_callback(split: str, _iq: np.ndarray, _labels: np.ndarray) -> None:
        first.append(split)

    def second_callback(split: str, _iq: np.ndarray, _labels: np.ndarray) -> None:
        second.append(split)

    block = np.ones((2, 8, 2), dtype=np.float32)
    sink_a = WiSigBlockSink(seed=123, on_batch=old_callback)
    sink_b = WiSigBlockSink(seed=123, on_batch=second_callback)
    for rx_i, day_i in ((0, 0), (1, 0), (1, 1), (9, 3)):
        sink_a.on_block(block, tx_i=0, rx_i=rx_i, day_i=day_i, eq_i=0)
        sink_b.on_block(block, tx_i=0, rx_i=rx_i, day_i=day_i, eq_i=0)
    assert first == second
    assert sink_a.group_assignments == sink_b.group_assignments
    assert WISIG_GROUP_SPLIT_STRATEGY == "receiver_day_group_held_out_v1"


def test_prepare_rejects_legacy_wisig_npy_without_group_metadata(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    emitter_root = tmp_path / "emitters"
    npy_root = emitter_root / "wisig" / "npy_data" / "equalized_data_0"
    npy_root.mkdir(parents=True)
    for split in ("train", "val", "test"):
        (npy_root / f"X_{split}.npy").touch()
    monkeypatch.setattr(prepare_datasets, "EMITTER", emitter_root)
    ctx = prepare_datasets.Context(tmp_path / "output")
    with pytest.raises(ValueError, match="无法验证 group-held-out"):
        prepare_datasets.wisig_manytx(ctx, dataset_id=11)
