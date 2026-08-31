from __future__ import annotations

import numpy as np

from resmamba_signal_model.data.canonical_iq import (
    CanonicalIQConfig,
    center_spectral_peak_numpy,
    dataset_name_from_h5_path,
    remove_dc_numpy,
    resize_iq_length,
)


def test_dataset_name_from_h5_path() -> None:
    assert dataset_name_from_h5_path("dataset/h5/rml2016_10a_train.h5") == "rml2016_10a"
    assert dataset_name_from_h5_path("radchar_val.h5") == "radchar"


def test_resize_pad_and_resample() -> None:
    iq = np.ones((2, 64), dtype=np.float32)
    padded = resize_iq_length(iq, 128)
    assert padded.shape == (2, 128)
    assert np.count_nonzero(padded[:, 32:96]) == 2 * 64

    long_iq = np.random.randn(2, 256).astype(np.float32)
    short = resize_iq_length(long_iq, 128)
    assert short.shape == (2, 128)


def test_remove_dc() -> None:
    iq = np.ones((2, 32), dtype=np.float32) * 3.0
    out = remove_dc_numpy(iq)
    assert np.allclose(out.mean(axis=-1), 0.0, atol=1e-6)


def test_canonical_config_resolve_length() -> None:
    cfg = CanonicalIQConfig.from_cli(enabled=True)
    assert cfg.resolve_length("rml2016_10a", 128) == 128
    assert cfg.resolve_length("panoradio_hf", 2048) == 1024
    assert cfg.resolve_length("unknown_set", 600) == 1024


def test_canonical_apply_batch() -> None:
    cfg = CanonicalIQConfig.from_cli(enabled=True)
    batch = np.random.randn(4, 2, 200).astype(np.float32)
    out = cfg.apply(batch, 128)
    assert out.shape == (4, 2, 128)
    assert out.dtype == np.float32


def test_center_spectral_peak_runs() -> None:
    iq = np.random.randn(2, 64).astype(np.float32)
    out = center_spectral_peak_numpy(iq)
    assert out.shape == iq.shape
