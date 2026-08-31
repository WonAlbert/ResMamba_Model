"""quality_mask 扩展过滤：尖峰 / 长期静默 / 削波卡死 / 族感知。"""

from __future__ import annotations

import numpy as np

from resmamba_signal_model.data.quality import quality_mask, quality_mask_global


def _base_iq(length: int = 128, scale: float = 1.0) -> np.ndarray:
    rng = np.random.default_rng(0)
    z = rng.standard_normal(length) + 1j * rng.standard_normal(length)
    x = np.stack([z.real, z.imag], axis=0).astype(np.float32) * scale
    return x[None, ...]


def test_spike_detected() -> None:
    iq = _base_iq()
    iq[0, 0, 64] *= 50.0
    iq[0, 1, 64] *= 50.0
    keep, reasons = quality_mask_global(iq, family="comm")
    assert not bool(keep[0])
    assert reasons["spike"] >= 1


def test_stuck_channel_detected() -> None:
    iq = _base_iq()
    iq[0, 0, :] = 1.0
    keep, reasons = quality_mask_global(iq, family="comm")
    assert not bool(keep[0])
    assert reasons["stuck_or_clip"] >= 1


def test_long_silence_comm_drops() -> None:
    iq = _base_iq(length=256)
    iq[0, :, :180] = 0.0
    keep, reasons = quality_mask_global(iq, family="comm")
    assert not bool(keep[0])
    assert reasons["long_silence"] >= 1


def test_radar_low_duty_kept() -> None:
    iq = np.zeros((1, 2, 256), dtype=np.float32)
    iq[0, :, 210:240] = _base_iq(length=30)[0]
    keep, reasons = quality_mask_global(iq, family="radar")
    assert bool(keep[0])
    assert reasons.get("long_silence", 0) == 0


def test_radar_empty_window_dropped() -> None:
    iq = np.zeros((1, 2, 256), dtype=np.float32)
    keep, reasons = quality_mask_global(iq, family="radar")
    assert not bool(keep[0])
    assert reasons["long_silence"] >= 1 or reasons["silent"] >= 1


def test_family_aware_via_dataset_name() -> None:
    iq = np.zeros((1, 2, 256), dtype=np.float32)
    iq[0, :, 210:240] = _base_iq(length=30)[0]
    iq_comm = iq.copy()
    iq_comm[0, :, :180] = 0.0
    keep_comm, _ = quality_mask(iq_comm, dataset_name="rml2016_04c")
    keep_radar, _ = quality_mask(iq, dataset_name="radchar")
    assert not bool(keep_comm[0])
    assert bool(keep_radar[0])


def test_radar_mod15_keeps_pulse_edge_overshoot() -> None:
    """宽脉冲 + 脉沿 1 点过冲：默认 spike 会杀，radar_mod15 只筛极端毛刺。"""
    iq = np.ones((1, 2, 256), dtype=np.float32)
    iq[0, :, :40] = 0.05
    iq[0, :, 200:] = 0.05
    iq[0, :, 40] = 3.0
    keep_default, reasons_default = quality_mask_global(iq, family="radar")
    keep_mod15, reasons_mod15 = quality_mask(iq, dataset_name="radar_mod15")
    assert not bool(keep_default[0])
    assert reasons_default["spike"] >= 1
    assert bool(keep_mod15[0])
    assert reasons_mod15["spike"] == 0


def test_radar_mod15_drops_extreme_one_sample_glitch() -> None:
    iq = np.ones((1, 2, 256), dtype=np.float32)
    iq[0, :, 80] = 8.0
    keep, reasons = quality_mask(iq, dataset_name="radar_mod15")
    assert not bool(keep[0])
    assert reasons["spike"] >= 1


def test_per_class_masking() -> None:
    iq = np.concatenate([_base_iq(), np.zeros((1, 2, 128), dtype=np.float32)], axis=0)
    labels = np.array([0, 1], dtype=np.int32)
    keep, reasons = quality_mask(iq, labels=labels, dataset_name="rml2016_10a")
    assert keep.tolist() == [True, False]
    assert reasons["silent"] >= 1