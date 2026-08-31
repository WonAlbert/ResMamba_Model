"""波形质量过滤：尖峰 / 长期静默 / 削波卡死；族感知阈值（通信 vs 雷达）。"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Literal

import numpy as np

QualityFamily = Literal["comm", "radar"]

_COMM_STEMS = frozenset({"panoradio_hf"})
_RADAR_STEMS = frozenset({"radchar", "radar_mod15", "cjr_mix"})

REASON_ORDER = (
    "non_finite",
    "silent",
    "stuck_or_clip",
    "long_silence",
    "spike",
    "power_outlier",
    "impulsive",
    "low_quality_proxy",
)


def resolve_quality_family(dataset_name: str | None) -> QualityFamily:
    name = str(dataset_name or "").strip()
    if name in _RADAR_STEMS:
        return "radar"
    if name.startswith("rml2016") or name in _COMM_STEMS:
        return "comm"
    return "comm"


def stratified_split_indices(
    labels: np.ndarray,
    ratios: tuple[float, float, float] = (0.7, 0.2, 0.1),
    seed: int = 20260629,
) -> dict[str, np.ndarray]:
    """按类分层划分 train / test / val，类内不跨 split。"""
    labels = np.asarray(labels).reshape(-1)
    if labels.shape[0] == 0:
        return {
            "train": np.array([], dtype=np.int64),
            "test": np.array([], dtype=np.int64),
            "val": np.array([], dtype=np.int64),
        }
    ratio_train, ratio_test, ratio_val = (float(r) for r in ratios)
    total = ratio_train + ratio_test + ratio_val
    ratio_train /= total
    ratio_test /= total
    ratio_val /= total
    rng = np.random.default_rng(int(seed))
    buckets: dict[str, list[np.ndarray]] = {"train": [], "test": [], "val": []}
    for cls in np.unique(labels):
        idx = np.flatnonzero(labels == cls).astype(np.int64)
        rng.shuffle(idx)
        n = int(idx.size)
        if n == 0:
            continue
        if n == 1:
            buckets["train"].append(idx)
            continue
        if n == 2:
            buckets["train"].append(idx[:1])
            buckets["test"].append(idx[1:])
            continue
        n_train = int(round(ratio_train * n))
        n_test = int(round(ratio_test * n))
        n_train = max(1, min(n_train, n - 2))
        n_test = max(1, min(n_test, n - n_train - 1))
        n_val = n - n_train - n_test
        if n_val < 1:
            n_val = 1
            if n_train > 1:
                n_train -= 1
            elif n_test > 1:
                n_test -= 1
        buckets["train"].append(idx[:n_train])
        buckets["test"].append(idx[n_train : n_train + n_test])
        buckets["val"].append(idx[n_train + n_test :])
    out: dict[str, np.ndarray] = {}
    for split, parts in buckets.items():
        out[split] = np.concatenate(parts).astype(np.int64) if parts else np.array([], dtype=np.int64)
        rng.shuffle(out[split])
    return out


def _as_batch(iq: np.ndarray) -> np.ndarray:
    x = np.asarray(iq)
    if x.ndim == 2 and x.shape[0] == 2:
        return x[np.newaxis, ...]
    if x.ndim != 3 or x.shape[1] != 2:
        raise ValueError(f"quality_mask 期望 [B,2,L] 或 [2,L]，当前 {x.shape}")
    return x


def _per_sample_power(x: np.ndarray) -> np.ndarray:
    return np.mean(np.square(x), axis=(-2, -1))


def _is_silent_power(power: np.ndarray) -> np.ndarray:
    return power <= float(np.finfo(np.float32).tiny)


def _instant_power(x: np.ndarray) -> np.ndarray:
    return np.square(x).sum(axis=-2)


def _low_power_threshold(instant: np.ndarray) -> np.ndarray:
    """逐样本稳健低功率阈值。"""
    med = np.median(instant, axis=-1)
    mad = np.median(np.abs(instant - med[..., None]), axis=-1)
    noise = np.maximum(1.4826 * mad, med * 0.01 + 1e-12)
    return np.maximum(0.01 * np.maximum(med, 1e-12), noise)


def _max_run_fraction_1d(row: np.ndarray) -> float:
    row = np.asarray(row, dtype=bool).reshape(-1)
    length = int(row.size)
    if length == 0 or not np.any(row):
        return 0.0
    padded = np.concatenate(([False], row, [False])).astype(np.int8)
    diffs = np.diff(padded)
    starts = np.flatnonzero(diffs == 1)
    ends = np.flatnonzero(diffs == -1)
    if starts.size == 0:
        return 0.0
    return float(np.max(ends - starts)) / float(length)


def _max_run_fraction(mask: np.ndarray) -> np.ndarray:
    """mask True=低功率；返回最长连续低功率段占比 [B]。"""
    b, length = mask.shape
    out = np.zeros(b, dtype=np.float64)
    for i in range(b):
        row = mask[i]
        if not np.any(row):
            out[i] = 0.0
            continue
        padded = np.concatenate(([False], row, [False])).astype(np.int8)
        diffs = np.diff(padded)
        starts = np.flatnonzero(diffs == 1)
        ends = np.flatnonzero(diffs == -1)
        if starts.size == 0:
            out[i] = 0.0
        else:
            out[i] = float(np.max(ends - starts)) / float(length)
    return out


def _stuck_or_clip_mask(x: np.ndarray, *, active: np.ndarray, power: np.ndarray) -> np.ndarray:
    i_ch = x[..., 0, :]
    q_ch = x[..., 1, :]
    i_std = np.std(i_ch, axis=-1)
    q_std = np.std(q_ch, axis=-1)
    meaningful = active & (power > 1e-8)
    stuck = meaningful & (
        ((i_std < 1e-7) & (q_std > 1e-4))
        | ((q_std < 1e-7) & (i_std > 1e-4))
    )
    amp = np.sqrt(np.maximum(_instant_power(x), 0.0))
    amp_std = np.std(amp, axis=-1)
    peak = np.max(amp, axis=-1, keepdims=True)
    near_peak = peak > 1e-3
    clip_frac = np.mean((amp >= 0.98 * peak) & near_peak, axis=-1)
    clipped = active & (clip_frac > 0.25) & (amp_std > 1e-4)
    return stuck | clipped


@dataclass(frozen=True)
class SpikePolicy:
    """瞬时功率 MAD z-score 尖峰策略。

    ``isolated_run_max=0`` 关闭短 run click；``min_peak_energy_frac>0`` 时
    孤立短峰还须占全序列能量足够份额才判坏（避免 Hilbert 脉沿误杀）。
    """

    z_limit: float
    frac_limit: float
    scattered_max_run: float
    isolated_run_max: int
    min_peak_energy_frac: float = 0.0


def resolve_spike_policy(dataset_name: str | None, *, strict: bool = False) -> SpikePolicy:
    name = str(dataset_name or "").strip()
    if name == "radar_mod15":
        # 幅度 CSV→Hilbert 的脉沿/脉组会被默认 1–3 点 click 与 scattered-hot 误杀。
        # 只丢：盐胡椒噪声，或单点能量占比过高的极端毛刺。
        return SpikePolicy(
            z_limit=24.0,
            frac_limit=0.20,
            scattered_max_run=0.02,
            isolated_run_max=1,
            min_peak_energy_frac=0.15,
        )
    if strict:
        return SpikePolicy(z_limit=6.0, frac_limit=0.02, scattered_max_run=0.05, isolated_run_max=3)
    return SpikePolicy(z_limit=8.0, frac_limit=0.05, scattered_max_run=0.05, isolated_run_max=3)


def _spike_mask(
    x: np.ndarray,
    *,
    strict: bool = False,
    policy: SpikePolicy | None = None,
) -> np.ndarray:
    policy = policy or resolve_spike_policy(None, strict=strict)
    instant = _instant_power(x)
    med = np.median(instant, axis=-1, keepdims=True)
    mad = np.median(np.abs(instant - med), axis=-1, keepdims=True)
    spread = np.maximum(1.4826 * mad, med * 0.05 + 1e-12)
    z = (instant - med) / spread
    hot = z > float(policy.z_limit)
    totals = np.maximum(instant.sum(axis=-1), 1e-30)
    b, length = hot.shape
    bad = np.zeros(b, dtype=bool)
    frac_limit = float(policy.frac_limit)
    isolated_max = int(policy.isolated_run_max)
    min_frac = float(policy.min_peak_energy_frac)
    for i in range(b):
        row = hot[i]
        if not np.any(row):
            continue
        if float(row.sum()) / float(length) > frac_limit:
            max_hot_run = _max_run_fraction_1d(row)
            if max_hot_run <= float(policy.scattered_max_run):
                bad[i] = True
            continue
        if isolated_max < 1:
            continue
        padded = np.concatenate(([False], row, [False])).astype(np.int8)
        diffs = np.diff(padded)
        starts = np.flatnonzero(diffs == 1)
        ends = np.flatnonzero(diffs == -1)
        for s, e in zip(starts, ends, strict=True):
            run = int(e - s)
            if 1 <= run <= isolated_max and 3 <= s and (e + 3) <= length:
                if min_frac > 0.0:
                    run_frac = float(instant[i, s:e].sum()) / float(totals[i])
                    if run_frac < min_frac:
                        continue
                bad[i] = True
                break
    return bad


def _long_silence_mask(x: np.ndarray, family: QualityFamily) -> np.ndarray:
    instant = _instant_power(x)
    thresh = _low_power_threshold(instant)[..., None]
    low = instant <= thresh
    duty = 1.0 - np.mean(low, axis=-1)
    max_run = _max_run_fraction(low)
    if family == "radar":
        return (max_run > 0.95) | (duty < 0.02)
    return (max_run > 0.50) | (duty < 0.25)


def _legacy_segment_masks(
    x: np.ndarray,
    *,
    strict: bool,
    family: QualityFamily = "comm",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    power = _per_sample_power(x)
    positive = power > float(np.finfo(np.float32).tiny)
    log_power = np.log10(np.maximum(power, 1e-30))
    valid = log_power[positive]
    mad_scale = 3.0 if strict else 4.0
    if valid.size >= 32:
        median = float(np.median(valid))
        spread = max(1.4826 * float(np.median(np.abs(valid - median))), 0.15)
        power_ok = np.abs(log_power - median) <= mad_scale * spread
    else:
        power_ok = positive.copy()
    complex_power = _instant_power(x)
    papr = complex_power.max(axis=-1) / np.maximum(complex_power.mean(axis=-1), 1e-30)
    papr_limit = 20.0 if strict else 30.0
    if family == "radar":
        papr_limit *= 4.0
    impulse_ok = papr <= papr_limit
    z = x[..., 0, :] + 1j * x[..., 1, :]
    corr = np.abs(np.sum(z[..., 1:] * np.conj(z[..., :-1]), axis=-1))
    corr /= np.maximum(
        np.sqrt(np.sum(np.abs(z[..., 1:]) ** 2, axis=-1) * np.sum(np.abs(z[..., :-1]) ** 2, axis=-1)),
        1e-30,
    )
    corr_floor = 0.05 if strict else 1e-2
    structure_ok = corr >= corr_floor
    return power_ok, impulse_ok, structure_ok


def quality_mask_global(
    iq: np.ndarray,
    *,
    strict: bool = False,
    family: QualityFamily = "comm",
    dataset_name: str | None = None,
) -> tuple[np.ndarray, Counter]:
    """互斥原因桶；返回 keep 与 removed 计数。"""
    x = _as_batch(iq).astype(np.float64, copy=False)
    n = int(x.shape[0])
    assigned = np.zeros(n, dtype=bool)
    reasons: Counter = Counter()
    spike_policy = resolve_spike_policy(dataset_name, strict=strict)

    def _mark(mask: np.ndarray, name: str) -> None:
        nonlocal assigned
        pick = mask & ~assigned
        reasons[name] += int(pick.sum())
        assigned |= pick

    finite = np.isfinite(x).all(axis=(-2, -1))
    _mark(~finite, "non_finite")

    power = _per_sample_power(x)
    silent = _is_silent_power(power)
    _mark(finite & silent, "silent")
    positive = finite & ~silent

    active = positive & ~assigned
    if np.any(active):
        _mark(active & _stuck_or_clip_mask(x, active=active, power=power), "stuck_or_clip")
        active = finite & positive & ~assigned
        _mark(active & _long_silence_mask(x, family), "long_silence")
        active = finite & positive & ~assigned
        _mark(active & _spike_mask(x, policy=spike_policy), "spike")

    power_ok, impulse_ok, structure_ok = _legacy_segment_masks(x, strict=strict, family=family)
    active = finite & positive & ~assigned
    _mark(active & ~power_ok, "power_outlier")
    active = finite & positive & ~assigned
    _mark(active & ~impulse_ok, "impulsive")
    active = finite & positive & ~assigned
    _mark(active & ~structure_ok, "low_quality_proxy")

    keep = finite & positive & ~assigned
    for key in REASON_ORDER:
        reasons.setdefault(key, 0)
    return keep, reasons


def quality_mask(
    iq: np.ndarray,
    labels: np.ndarray | None = None,
    *,
    strict: bool = False,
    dataset_name: str | None = None,
    family: QualityFamily | None = None,
) -> tuple[np.ndarray, Counter]:
    fam = family or resolve_quality_family(dataset_name)
    if labels is None:
        return quality_mask_global(iq, strict=strict, family=fam, dataset_name=dataset_name)
    labels = np.asarray(labels).reshape(-1)
    batch = _as_batch(iq)
    keep = np.zeros(batch.shape[0], dtype=bool)
    reasons: Counter = Counter()
    for cls in np.unique(labels):
        idx = labels == cls
        cls_keep, cls_reasons = quality_mask_global(
            iq[idx], strict=strict, family=fam, dataset_name=dataset_name
        )
        keep[idx] = cls_keep
        reasons.update(cls_reasons)
    return keep, reasons
