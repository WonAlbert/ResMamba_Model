#!/usr/bin/env python3
"""从 RML2016.10a 的 IQ 样本估计瞬时幅度、相位、频率。"""
from __future__ import annotations

import pickle
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PKL = Path("/root/autodl-tmp/辐射源（非个体）识别数据/RML2016.10a_dict.pkl")
FS = 200e3  # RadioML 2016.10a GNU Radio 采样率
MARK = 64  # 打印“某一刻”的采样点
EXAMPLES = [
    ("BPSK", 18, 0),
    ("QPSK", 18, 0),
    ("QAM16", 18, 0),
    ("CPFSK", 18, 0),
    ("AM-DSB", 18, 0),
    ("WBFM", 18, 976),
]


def setup_font() -> None:
    for path in (
        Path("/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc"),
        Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"),
    ):
        if path.is_file():
            font_manager.fontManager.addfont(str(path))
            name = font_manager.FontProperties(fname=str(path)).get_name()
            plt.rcParams["font.sans-serif"] = [name, "DejaVu Sans"]
            plt.rcParams["font.family"] = "sans-serif"
            plt.rcParams["axes.unicode_minus"] = False
            return


def iq_to_complex(iq: np.ndarray) -> np.ndarray:
    """(2, L) 或 (L, 2) -> 复基带 z[n]."""
    x = np.asarray(iq, dtype=np.float64)
    if x.shape[0] == 2:
        return x[0] + 1j * x[1]
    if x.shape[-1] == 2:
        return x[..., 0] + 1j * x[..., 1]
    raise ValueError(f"无法识别 IQ 形状 {x.shape}")


def instantaneous(z: np.ndarray, fs: float = FS) -> dict[str, np.ndarray]:
    """由复基带 z[n] 得到瞬时幅度、缠绕/解缠相位、瞬时频率。"""
    amp = np.abs(z)
    phase_wrapped = np.angle(z)  # atan2(Q, I), (-π, π]
    phase = np.unwrap(phase_wrapped)
    dphi = np.diff(phase, prepend=phase[0])
    dphi = (dphi + np.pi) % (2 * np.pi) - np.pi  # 抑制残留 2π 跳变
    freq_hz = dphi * fs / (2 * np.pi)
    freq_hz[0] = freq_hz[1] if len(freq_hz) > 1 else 0.0
    return {
        "amp": amp,
        "phase_wrapped": phase_wrapped,
        "phase": phase,
        "freq_hz": freq_hz,
        "freq_rad_per_sample": dphi,
    }


def main() -> int:
    setup_font()
    pkl = DEFAULT_PKL
    if not pkl.is_file():
        print(f"找不到 {pkl}", file=sys.stderr)
        return 1
    with pkl.open("rb") as f:
        data = pickle.load(f, encoding="latin1")

    out_dir = ROOT / "figures" / "rml2016_10a_instantaneous"
    out_dir.mkdir(parents=True, exist_ok=True)

    n_ex = len(EXAMPLES)
    fig, axes = plt.subplots(n_ex, 4, figsize=(16.5, 2.15 * n_ex), squeeze=False)
    col_titles = ["I / Q", "瞬时幅度 |z|", "解缠相位", "瞬时频率"]

    print(f"{'调制':8s} {'n':>4s} {'I':>10s} {'Q':>10s} {'幅度':>10s} {'相位(rad)':>12s} {'相位(°)':>10s} {'频率(Hz)':>12s}")
    t = None
    for row, (mod, snr, idx) in enumerate(EXAMPLES):
        iq = np.asarray(data[(mod, snr)][idx], dtype=np.float32)
        z = iq_to_complex(iq)
        est = instantaneous(z, FS)
        if t is None:
            t = np.arange(z.size)
        n = MARK
        print(
            f"{mod:8s} {n:4d} {z.real[n]:10.5f} {z.imag[n]:10.5f} "
            f"{est['amp'][n]:10.5f} {est['phase'][n]:12.4f} "
            f"{np.degrees(est['phase_wrapped'][n]):10.2f} {est['freq_hz'][n]:12.1f}"
        )

        ax_iq, ax_a, ax_p, ax_f = axes[row]
        ax_iq.plot(t, z.real, color="#1f77b4", lw=1.05, label="I")
        ax_iq.plot(t, z.imag, color="#2ca02c", lw=1.0, alpha=0.9, label="Q")
        ax_a.plot(t, est["amp"], color="#d62728", lw=1.1)
        ax_p.plot(t, est["phase"], color="#9467bd", lw=1.1)
        ax_f.plot(t, est["freq_hz"] / 1e3, color="#ff7f0e", lw=1.1)
        for ax in (ax_iq, ax_a, ax_p, ax_f):
            ax.axvline(n, color="#444444", ls="--", lw=0.8, alpha=0.7)
            ax.grid(True, alpha=0.28)
            ax.tick_params(labelsize=8)
        ax_iq.set_ylabel(f"{mod}\nSNR {snr:+d} dB", fontsize=8)
        if row == 0:
            ax_iq.legend(loc="upper right", fontsize=7, ncol=2)
            for ax, title in zip((ax_iq, ax_a, ax_p, ax_f), col_titles):
                ax.set_title(title, fontsize=10)
        if row == n_ex - 1:
            for ax in (ax_iq, ax_a, ax_p, ax_f):
                ax.set_xlabel("采样点 n", fontsize=8)
        ax_a.set_ylabel("|z|", fontsize=8)
        ax_p.set_ylabel("rad", fontsize=8)
        ax_f.set_ylabel("kHz", fontsize=8)

    fig.suptitle(
        f"RML2016.10a：由 IQ 估计瞬时幅度 / 相位 / 频率（虚线为 n={MARK}，fs={FS/1e3:.0f} kHz）",
        fontsize=13,
        y=1.01,
    )
    fig.tight_layout()
    path = out_dir / "amp_phase_freq.png"
    fig.savefig(path, dpi=170, bbox_inches="tight")
    plt.close(fig)
    print(f"写入 {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
