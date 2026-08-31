from __future__ import annotations

import sys
from pathlib import Path

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from prepare_datasets import (  # noqa: E402
    RADCHAR_SIGNAL_TYPE_NAMES,
    Context,
    radchar,
)


def _write_toy_radchar(path: Path, *, n: int = 50, length: int = 64) -> None:
    rng = np.random.default_rng(0)
    t = np.arange(length, dtype=np.float64)
    iq = np.empty((n, length), dtype=np.complex128)
    dtype = np.dtype(
        [
            ("index", "<i8"),
            ("signal_type", "<i8"),
            ("number_of_pulses", "<i8"),
            ("pulse_width", "<f8"),
            ("time_delay", "<f8"),
            ("pulse_repetition_interval", "<f8"),
            ("signal_to_noise_ratio", "<i8"),
        ]
    )
    labels = np.zeros(n, dtype=dtype)
    for i in range(n):
        cls = i % 5
        freq = 4 + cls
        tone = np.exp(2j * np.pi * freq * t / length)
        noise = 0.03 * (rng.normal(size=length) + 1j * rng.normal(size=length))
        iq[i] = tone + noise
        labels[i] = (i, cls, 2, 1.2e-5, 2e-6, 2e-5, 12)
    with h5py.File(path, "w") as handle:
        handle.create_dataset("iq", data=iq)
        handle.create_dataset("labels", data=labels)


def test_radchar_converts_to_stratified_train_test_val_h5(tmp_path: Path) -> None:
    src = tmp_path / "RadChar-Small.h5"
    _write_toy_radchar(src)
    ctx = Context(tmp_path)
    radchar(ctx, 16, source=src)

    paths = {split: tmp_path / "h5" / f"radchar_{split}.h5" for split in ("train", "test", "val")}
    for path in paths.values():
        assert path.is_file()

    counts: dict[str, int] = {}
    for split, path in paths.items():
        with h5py.File(path, "r") as handle:
            counts[split] = int(handle["iq"].shape[0])
            assert handle["iq"].shape[1:] == (2, 64)
            assert handle["iq"].dtype == np.float32
            if "dataset_id" in handle:
                assert int(handle["dataset_id"][0]) == 16
            mods = set(np.unique(handle["mod_label_id"][:]).tolist())
            assert mods <= {0, 1, 2, 3, 4}
            assert np.all(np.isfinite(handle["snr"][:]))
            if "sampling_rate_hz" in handle.attrs:
                assert float(handle.attrs["sampling_rate_hz"]) == 3_200_000.0

    assert sum(counts.values()) == 50
    assert counts["train"] >= counts["test"] >= 1
    assert counts["val"] >= 1
    entry = ctx.report["datasets"]["radchar"]
    for split in ("train", "test", "val"):
        assert "removed" in entry["splits"][split]
    assert ctx.maps["datasets"]["16"] == "radchar"
    assert ctx.maps["modulations"]["radchar"] == {
        name: idx for idx, name in enumerate(RADCHAR_SIGNAL_TYPE_NAMES)
    }
