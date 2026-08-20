from __future__ import annotations

import numpy as np

from resmamba_signal_model.data.cjr_mix import normalize_iq_absmax, normalize_iq_joint_power, parse_iq_array


def test_parse_iq_from_tl2():
    raw = np.random.randn(1024, 2).astype(np.float32)
    iq = parse_iq_array(raw)
    assert iq.shape == (2, 1024)


def test_parse_iq_from_2t1():
    raw = np.random.randn(2, 1024, 1).astype(np.float32)
    iq = parse_iq_array(raw)
    assert iq.shape == (2, 1024)


def test_normalize_methods():
    iq = np.random.randn(2, 128).astype(np.float32)
    abs_out = normalize_iq_absmax(iq)
    joint_out = normalize_iq_joint_power(iq)
    assert abs_out.shape == iq.shape
    assert joint_out.shape == iq.shape
    assert np.all(abs_out <= 5.0) and np.all(abs_out >= -5.0)
