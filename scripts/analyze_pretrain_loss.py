#!/usr/bin/env python3
"""从 Lightning CSV 诊断预训练 step loss 波动：分 H5 stem、与 seq_len 相关性。"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _find_metrics_csv(run_dir: Path) -> Path | None:
    candidates = [
        run_dir / "csv" / "version_0" / "metrics.csv",
        run_dir / "csv" / "metrics.csv",
    ]
    for path in candidates:
        if path.is_file():
            return path
    for path in sorted(run_dir.glob("csv/**/metrics.csv")):
        if path.is_file():
            return path
    return None


def _load_csv_rows(path: Path) -> list[dict[str, str]]:
    import csv

    with path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        return list(reader)


def _finite_float(raw: str | None) -> float | None:
    if raw is None or str(raw).strip() == "":
        return None
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(value):
        return None
    return value


def _pearson(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) < 3 or len(xs) != len(ys):
        return None
    mx = sum(xs) / len(xs)
    my = sum(ys) / len(ys)
    cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    vx = sum((x - mx) ** 2 for x in xs)
    vy = sum((y - my) ** 2 for y in ys)
    if vx <= 0 or vy <= 0:
        return None
    return cov / math.sqrt(vx * vy)


def _resolve_mse_column(columns: list[str]) -> str | None:
    for pattern in (
        lambda c: c.endswith("loss/mse") and "ema" not in c,
        lambda c: "loss/mse" in c and "stem" not in c and "ema" not in c,
        lambda c: c.endswith("loss/mae") and "ema" not in c,
        lambda c: "loss/mae" in c and "stem" not in c and "ema" not in c,
    ):
        col = next((c for c in columns if pattern(c)), None)
        if col:
            return col
    return None


def analyze_metrics(rows: list[dict[str, str]]) -> dict[str, Any]:
    if not rows:
        return {"error": "metrics.csv 为空"}

    columns = list(rows[0].keys())
    mse_col = _resolve_mse_column(columns)
    total_col = next((c for c in columns if c.endswith("loss/total") and "ema" not in c), None)
    total_ema_col = next((c for c in columns if "loss/total_ema" in c), None)
    stem_norm_col = next((c for c in columns if "loss/stem_norm" in c), None)
    seq_col = next((c for c in columns if "seq_len/mean" in c), None)
    mask_col = next((c for c in columns if "mask/ratio_mean" in c), None)

    stem_re = re.compile(r"loss/mse_stem/(.+)_step$")
    legacy_stem_re = re.compile(r"loss/mae_stem/(.+)_step$")
    stem_cols: dict[str, str] = {}
    for c in columns:
        match = stem_re.match(c) or legacy_stem_re.match(c)
        if match:
            stem_cols[match.group(1)] = c

    paired_mse: list[float] = []
    paired_len: list[float] = []
    paired_mask: list[float] = []
    mse_vals: list[float] = []
    total_vals: list[float] = []

    for row in rows:
        mse = _finite_float(row.get(mse_col) if mse_col else None)
        total = _finite_float(row.get(total_col) if total_col else None)
        slen = _finite_float(row.get(seq_col) if seq_col else None)
        mask_r = _finite_float(row.get(mask_col) if mask_col else None)
        if mse is not None:
            mse_vals.append(mse)
            if slen is not None:
                paired_mse.append(mse)
                paired_len.append(slen)
            if mask_r is not None:
                paired_mask.append(mask_r)
        if total is not None:
            total_vals.append(total)

    total_ema_vals: list[float] = []
    stem_norm_vals: list[float] = []
    for row in rows:
        t_ema = _finite_float(row.get(total_ema_col) if total_ema_col else None)
        if t_ema is not None:
            total_ema_vals.append(t_ema)
        sn = _finite_float(row.get(stem_norm_col) if stem_norm_col else None)
        if sn is not None:
            stem_norm_vals.append(sn)

    def stats(vals: list[float]) -> dict[str, float | None]:
        if not vals:
            return {"n": 0, "mean": None, "std": None, "min": None, "max": None}
        mean = sum(vals) / len(vals)
        var = sum((v - mean) ** 2 for v in vals) / len(vals)
        return {
            "n": len(vals),
            "mean": mean,
            "std": math.sqrt(var),
            "min": min(vals),
            "max": max(vals),
        }

    stem_stats: dict[str, Any] = {}
    for stem, col in sorted(stem_cols.items()):
        vals = [_finite_float(row.get(col)) for row in rows]
        clean = [v for v in vals if v is not None]
        stem_stats[stem] = stats(clean)

    report: dict[str, Any] = {
        "columns": {
            "mse": mse_col,
            "total": total_col,
            "seq_len_mean": seq_col,
            "mask_ratio_mean": mask_col,
            "stem_columns": len(stem_cols),
        },
        "loss_mse": stats(mse_vals),
        "loss_total": stats(total_vals),
        "loss_total_ema": stats(total_ema_vals),
        "loss_stem_norm": stats(stem_norm_vals),
        "correlation": {
            "mse_vs_seq_len": _pearson(paired_mse, paired_len),
            "mse_vs_mask_ratio": _pearson(paired_mse, paired_mask) if paired_mask else None,
        },
        "mse_stem": stem_stats,
    }
    return report


def format_report(report: dict[str, Any], run_dir: Path) -> str:
    lines = [f"run: {run_dir}", ""]
    if "error" in report:
        lines.append(f"error: {report['error']}")
        return "\n".join(lines)

    mse = report.get("loss_mse") or {}
    total = report.get("loss_total") or {}
    lines.append(
        f"loss/mse: n={mse.get('n')} mean={mse.get('mean'):.4f} std={mse.get('std'):.4f} "
        f"min={mse.get('min'):.4f} max={mse.get('max'):.4f}"
        if mse.get("mean") is not None
        else "loss/mse: (missing)"
    )
    lines.append(
        f"loss/total: n={total.get('n')} mean={total.get('mean'):.4f} std={total.get('std'):.4f}"
        if total.get("mean") is not None
        else "loss/total: (missing)"
    )
    corr = report.get("correlation") or {}
    if corr.get("mse_vs_seq_len") is not None:
        lines.append(f"corr(mse, seq_len/mean): {corr['mse_vs_seq_len']:.3f}")
    if corr.get("mse_vs_mask_ratio") is not None:
        lines.append(f"corr(mse, mask/ratio_mean): {corr['mse_vs_mask_ratio']:.3f}")
    lines.append("")
    lines.append("loss/mse_stem (mean ± std):")
    for stem, row in sorted((report.get("mse_stem") or {}).items()):
        if row.get("mean") is None:
            continue
        lines.append(
            f"  {stem}: {row['mean']:.4f} ± {row['std']:.4f} (n={row['n']})"
        )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path, help="Lightning run 目录（含 csv/metrics.csv）")
    parser.add_argument("--json", action="store_true", help="输出 JSON")
    args = parser.parse_args()

    csv_path = _find_metrics_csv(args.run_dir)
    if csv_path is None:
        print(f"未找到 metrics.csv: {args.run_dir}", file=sys.stderr)
        sys.exit(1)
    rows = _load_csv_rows(csv_path)
    report = analyze_metrics(rows)
    if args.json:
        print(json.dumps(report, indent=2, ensure_ascii=False))
    else:
        print(format_report(report, args.run_dir))


if __name__ == "__main__":
    main()
