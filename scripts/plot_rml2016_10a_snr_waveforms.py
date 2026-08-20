#!/usr/bin/env python3
"""绘制 RML2016.10a 同一调制类别、不同信噪比的 I/Q 波形。

每类选取 5 个代表性 SNR（默认 -18/-8/0/8/18 dB），各取 1 个样本。
"""
from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PKL = Path("/root/autodl-tmp/辐射源（非个体）识别数据/RML2016.10a_dict.pkl")
MOD_ORDER = [
    "8PSK",
    "AM-DSB",
    "AM-SSB",
    "BPSK",
    "CPFSK",
    "GFSK",
    "PAM4",
    "QAM16",
    "QAM64",
    "QPSK",
    "WBFM",
]
DEFAULT_SNRS = (-18, -8, 0, 8, 18)
I_COLOR = "#1f77b4"
Q_COLOR = "#2ca02c"


def setup_font() -> None:
    candidates = [
        Path("/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc"),
        Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"),
        ROOT / "assets/fonts/NotoSansCJKsc-Regular.otf",
    ]
    for path in candidates:
        if path.is_file():
            font_manager.fontManager.addfont(str(path))
            name = font_manager.FontProperties(fname=str(path)).get_name()
            plt.rcParams["font.sans-serif"] = [name, "DejaVu Sans"]
            plt.rcParams["font.family"] = "sans-serif"
            plt.rcParams["axes.unicode_minus"] = False
            return


def load_rml2016_10a(path: Path) -> dict[tuple[str, int], np.ndarray]:
    with path.open("rb") as f:
        raw = pickle.load(f, encoding="latin1")
    out: dict[tuple[str, int], np.ndarray] = {}
    for (mod, snr), values in raw.items():
        out[(str(mod), int(snr))] = np.asarray(values, dtype=np.float32)
    return out


def pick_sample(iq: np.ndarray, index: int) -> np.ndarray:
    """iq: (N, 2, L) -> (2, L)."""
    n = iq.shape[0]
    return iq[int(index) % n]


def _ylim_from_iq(iq: np.ndarray, pad_frac: float = 0.12) -> tuple[float, float]:
    lo, hi = float(iq.min()), float(iq.max())
    span = max(hi - lo, 1e-6)
    pad = pad_frac * span
    return lo - pad, hi + pad


def plot_class_examples(
    data: dict[tuple[str, int], np.ndarray],
    mod: str,
    snrs: tuple[int, ...],
    sample_index: int,
    out_path: Path,
) -> None:
    n = len(snrs)
    fig, axes = plt.subplots(n, 2, figsize=(12.8, 2.15 * n), squeeze=False)
    for row, snr in enumerate(snrs):
        iq = pick_sample(data[(mod, snr)], sample_index)
        t = np.arange(iq.shape[-1])
        ax_w, ax_c = axes[row]

        ax_w.plot(t, iq[0], color=I_COLOR, linewidth=1.15, label="I")
        ax_w.plot(t, iq[1], color=Q_COLOR, linewidth=1.05, alpha=0.9, label="Q")
        ax_w.set_xlim(0, iq.shape[-1] - 1)
        ax_w.set_ylim(*_ylim_from_iq(iq))
        ax_w.set_ylabel("幅度")
        ax_w.grid(True, alpha=0.28)
        ax_w.set_title(f"{mod}  ·  SNR = {snr:+d} dB  ·  样本 #{sample_index}", fontsize=10)
        if row == 0:
            ax_w.legend(loc="upper right", fontsize=8, ncol=2, framealpha=0.92)

        ax_c.scatter(iq[0], iq[1], s=10, c="#4c78a8", alpha=0.72, linewidths=0)
        ax_c.set_aspect("equal", adjustable="box")
        lim = max(abs(iq).max() * 1.15, 1e-3)
        ax_c.set_xlim(-lim, lim)
        ax_c.set_ylim(-lim, lim)
        ax_c.set_xlabel("I")
        ax_c.set_ylabel("Q")
        ax_c.set_title("星座图", fontsize=10)
        ax_c.grid(True, alpha=0.28)
        ax_c.axhline(0, color="#888888", linewidth=0.6, alpha=0.5)
        ax_c.axvline(0, color="#888888", linewidth=0.6, alpha=0.5)

        ax_w.set_xlabel("采样点")

    fig.suptitle(
        f"RML2016.10a  ·  {mod}  同一类别不同信噪比（每类 5 例）",
        fontsize=13,
        y=1.01,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=170, bbox_inches="tight")
    plt.close(fig)


def plot_overview_grid(
    data: dict[tuple[str, int], np.ndarray],
    mods: list[str],
    snrs: tuple[int, ...],
    sample_index: int,
    out_path: Path,
) -> None:
    n_row, n_col = len(mods), len(snrs)
    fig, axes = plt.subplots(
        n_row,
        n_col,
        figsize=(3.15 * n_col, 1.72 * n_row),
        sharex=True,
        squeeze=False,
    )
    for i, mod in enumerate(mods):
        for j, snr in enumerate(snrs):
            ax = axes[i, j]
            iq = pick_sample(data[(mod, snr)], sample_index)
            t = np.arange(iq.shape[-1])
            ax.plot(t, iq[0], color=I_COLOR, linewidth=0.85)
            ax.plot(t, iq[1], color=Q_COLOR, linewidth=0.8, alpha=0.88)
            ax.set_xlim(0, iq.shape[-1] - 1)
            ax.set_ylim(*_ylim_from_iq(iq, pad_frac=0.18))
            ax.grid(True, alpha=0.22)
            ax.tick_params(labelsize=7)
            if i == 0:
                ax.set_title(f"SNR {snr:+d} dB", fontsize=10)
            if j == 0:
                ax.set_ylabel(mod, fontsize=9)
            if i == n_row - 1:
                ax.set_xlabel("采样点", fontsize=8)

    handles = [
        Line2D([0], [0], color=I_COLOR, linewidth=1.4, label="I"),
        Line2D([0], [0], color=Q_COLOR, linewidth=1.4, label="Q"),
    ]
    fig.legend(handles=handles, loc="upper right", ncol=2, fontsize=9, frameon=True)
    fig.suptitle(
        "RML2016.10a：同一调制类别在不同信噪比下的 I/Q 波形（每类 5 例，同一样本序号）",
        fontsize=13,
        y=0.995,
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.97))
    fig.savefig(out_path, dpi=170, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="绘制 RML2016.10a 同类不同 SNR 波形")
    p.add_argument("--pkl", type=Path, default=DEFAULT_PKL)
    p.add_argument("--out-dir", type=Path, default=ROOT / "figures" / "rml2016_10a_snr_waveforms")
    p.add_argument("--snrs", type=int, nargs="+", default=list(DEFAULT_SNRS))
    p.add_argument("--sample-index", type=int, default=0, help="每个 (调制, SNR) 桶内的样本序号")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    setup_font()
    pkl = args.pkl
    if not pkl.is_file():
        print(f"找不到数据文件: {pkl}", file=sys.stderr)
        return 1
    snrs = tuple(int(s) for s in args.snrs)
    if len(snrs) != 5:
        print("请提供恰好 5 个 SNR，作为每类的 5 个例子。", file=sys.stderr)
        return 1

    print(f"加载 {pkl} ...")
    data = load_rml2016_10a(pkl)
    missing = [(m, s) for m in MOD_ORDER for s in snrs if (m, s) not in data]
    if missing:
        print(f"缺失键: {missing[:8]}{'...' if len(missing) > 8 else ''}", file=sys.stderr)
        return 1

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    overview = out_dir / "overview_all_classes.png"
    plot_overview_grid(data, MOD_ORDER, snrs, args.sample_index, overview)
    print(f"写入 {overview}")

    for mod in MOD_ORDER:
        path = out_dir / f"{mod.replace('/', '-')}.png"
        plot_class_examples(data, mod, snrs, args.sample_index, path)
        print(f"写入 {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
