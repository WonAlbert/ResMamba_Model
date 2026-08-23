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


def test_radchar_converts_to_train_val_h5(tmp_path: Path) -> None:
    src = tmp_path / "RadChar-Small.h5"
    _write_toy_radchar(src)
    ctx = Context(tmp_path)
    radchar(ctx, 16, source=src)

    train = tmp_path / "h5" / "radchar_train.h5"
    val = tmp_path / "h5" / "radchar_val.h5"
    assert train.is_file()
    assert val.is_file()
    assert not (tmp_path / "h5" / "radchar_test.h5").is_file()

    with h5py.File(train, "r") as handle:
        n_train = int(handle["iq"].shape[0])
        assert handle["iq"].shape[1:] == (2, 64)
        assert handle["iq"].dtype == np.float32
        assert int(handle["dataset_id"][0]) == 16
        mods = set(np.unique(handle["mod_label_id"][:]).tolist())
        assert mods <= {0, 1, 2, 3, 4}
        assert np.all(np.isfinite(handle["snr"][:]))
        assert float(handle.attrs["sampling_rate_hz"]) == 3_200_000.0
    with h5py.File(val, "r") as handle:
        n_val = int(handle["iq"].shape[0])
        assert handle["iq"].shape[1:] == (2, 64)

    assert n_train + n_val == 50
    assert n_train > n_val
    assert ctx.maps["datasets"]["16"] == "radchar"
    assert ctx.maps["modulations"]["radchar"] == {
        name: idx for idx, name in enumerate(RADCHAR_SIGNAL_TYPE_NAMES)
    }
