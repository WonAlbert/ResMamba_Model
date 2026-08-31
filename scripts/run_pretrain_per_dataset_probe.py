#!/usr/bin/env python
"""逐预训练库从零各训 N epoch（默认 2），每 epoch 验证；库与库之间不续训。

示例：
  python scripts/run_pretrain_per_dataset_probe.py
  python scripts/run_pretrain_per_dataset_probe.py --stems xidian14,rml2016_04c --epochs 2
  python scripts/run_pretrain_per_dataset_probe.py --dry-run
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_STEMS = [
    "cjr_mix",
    "panoradio_hf",
    "radar_mod15",
    "radchar",
    "rml2016_04c",
    "rml2016_10a",
    "rml2016_10b",
    "xidian14",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Per-dataset pretrain probe (fresh init each stem)")
    p.add_argument("--config", default="configs/pretrain_per_dataset_probe.yaml")
    p.add_argument("--stems", default=None, help="逗号分隔；默认 pretrain 八库")
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--tag", default=None, help="run 名前缀时间戳；默认当前时间")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument(
        "--continue-on-error",
        action="store_true",
        help="单库失败时继续后续库（默认遇错即停）",
    )
    p.add_argument(
        "--summary-only",
        default=None,
        help="只汇总已有 runs：传入 probe 根目录或 tag（如 20260826_201500）",
    )
    return p.parse_args()


def resolve_stems(raw: str | None) -> list[str]:
    if not raw:
        return list(DEFAULT_STEMS)
    stems = [s.strip() for s in str(raw).split(",") if s.strip()]
    if not stems:
        raise SystemExit("--stems 为空")
    return stems


def run_dir_for(tag: str, stem: str) -> Path:
    return ROOT / "runs" / "experiments" / f"pretrain_probe_{tag}_{stem}"


def train_one(stem: str, *, config: str, epochs: int, seed: int, tag: str, dry_run: bool) -> int:
    run_name = f"pretrain_probe_{tag}_{stem}"
    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "train.py"),
        "--stage",
        "pretrain",
        "--config",
        config,
        "--pretrain-stem",
        stem,
        "--max-epochs",
        str(int(epochs)),
        "--seed",
        str(int(seed)),
        "--run-name",
        run_name,
    ]
    # 刻意不传 --init-from / --resume：每库从随机初始化开始
    print("=" * 72, flush=True)
    print(f"[probe] stem={stem} run={run_name}", flush=True)
    print("[probe] cmd:", " ".join(cmd), flush=True)
    if dry_run:
        return 0
    env = dict(**{k: v for k, v in __import__("os").environ.items()})
    env["PYTHONPATH"] = f"{ROOT}:{env.get('PYTHONPATH', '')}"
    env.setdefault("RFDATA_ROOT", str(ROOT / "dataset"))
    proc = subprocess.run(cmd, cwd=str(ROOT), env=env)
    return int(proc.returncode)


def _finite(v: Any) -> float | None:
    if v is None or v == "":
        return None
    try:
        x = float(v)
    except (TypeError, ValueError):
        return None
    if x != x:
        return None
    return x


def parse_val_epochs(run_dir: Path) -> list[dict[str, Any]]:
    metrics = run_dir / "csv" / "version_0" / "metrics.csv"
    if not metrics.is_file():
        # lightning 可能用 version_N
        csv_root = run_dir / "csv"
        if csv_root.is_dir():
            cands = sorted(csv_root.glob("version_*/metrics.csv"))
            metrics = cands[-1] if cands else metrics
    if not metrics.is_file():
        return []
    rows: list[dict[str, Any]] = []
    with metrics.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            mse = _finite(row.get("val/recon_mse"))
            if mse is None:
                continue
            rows.append(
                {
                    "epoch": int(float(row["epoch"])) if row.get("epoch") not in (None, "") else None,
                    "step": int(float(row["step"])) if row.get("step") not in (None, "") else None,
                    "val/recon_mse": mse,
                    "val/recon_mae": _finite(row.get("val/recon_mae")),
                    "val/mae": _finite(row.get("val/mae")),
                    "val/monitor": _finite(row.get("val/monitor")),
                    "val/loss": _finite(row.get("val/loss")),
                    "val/structure": _finite(row.get("val/structure")),
                    "val/structure_phase": _finite(row.get("val/structure_phase")),
                    "val/vicreg_token": _finite(row.get("val/vicreg_token")),
                    "val/ssim": _finite(row.get("val/ssim")),
                }
            )
    return rows


def parse_train_log_val_blocks(run_dir: Path) -> list[dict[str, Any]]:
    """回退：从 train.log 的 ========== val epoch N ========== 块解析。"""
    log_path = run_dir / "train.log"
    if not log_path.is_file():
        return []
    text = log_path.read_text(encoding="utf-8", errors="replace")
    blocks = re.split(r"={5,}\s*val epoch\s+(\d+)\s*={5,}", text)
    out: list[dict[str, Any]] = []
    # split → [pre, ep0, body0, ep1, body1, ...]
    i = 1
    while i + 1 < len(blocks):
        epoch = int(blocks[i])
        body = blocks[i + 1]
        m = re.search(
            r"pretrain\s+mse=([0-9.eE+-]+)\s+mae=([0-9.eE+-]+)\s+n=(\d+)",
            body,
        )
        loss_m = re.search(
            r"loss=([0-9.eE+-]+).*?monitor=([0-9.eE+-]+).*?recon=([0-9.eE+-]+)",
            body,
        )
        row: dict[str, Any] = {"epoch": epoch}
        if m:
            row["val/recon_mse"] = float(m.group(1))
            row["val/recon_mae"] = float(m.group(2))
            row["n"] = int(m.group(3))
        if loss_m:
            row["val/loss"] = float(loss_m.group(1))
            row["val/monitor"] = float(loss_m.group(2))
        out.append(row)
        i += 2
    return out


def summarize(tag: str, stems: list[str]) -> dict[str, Any]:
    report: dict[str, Any] = {"tag": tag, "stems": {}, "ranking": []}
    ranking: list[tuple[str, float | None, float | None]] = []
    for stem in stems:
        run_dir = run_dir_for(tag, stem)
        epochs = parse_val_epochs(run_dir)
        if not epochs:
            epochs = parse_train_log_val_blocks(run_dir)
        first = epochs[0] if epochs else None
        last = epochs[-1] if epochs else None
        first_mse = first.get("val/recon_mse") if first else None
        last_mse = last.get("val/recon_mse") if last else None
        delta = None
        if first_mse is not None and last_mse is not None:
            delta = float(last_mse) - float(first_mse)
        entry = {
            "run_dir": str(run_dir),
            "epochs": epochs,
            "first_mse": first_mse,
            "last_mse": last_mse,
            "delta_mse": delta,
            "ok": run_dir.is_dir() and bool(epochs),
        }
        report["stems"][stem] = entry
        ranking.append((stem, last_mse, delta))
    ranking.sort(key=lambda t: (t[1] is None, t[1] if t[1] is not None else 0.0))
    report["ranking"] = [
        {"stem": s, "last_mse": mse, "delta_mse": d} for s, mse, d in ranking
    ]
    return report


def print_report(report: dict[str, Any]) -> None:
    print("=" * 72)
    print(f"[probe summary] tag={report['tag']}")
    print(f"{'stem':<16} {'ep0_mse':>10} {'last_mse':>10} {'delta':>10} {'status':>8}")
    for stem, entry in report["stems"].items():
        ep0 = entry.get("first_mse")
        last = entry.get("last_mse")
        delta = entry.get("delta_mse")
        status = "ok" if entry.get("ok") else "missing"
        print(
            f"{stem:<16} "
            f"{(f'{ep0:.4f}' if ep0 is not None else '—'):>10} "
            f"{(f'{last:.4f}' if last is not None else '—'):>10} "
            f"{(f'{delta:+.4f}' if delta is not None else '—'):>10} "
            f"{status:>8}"
        )
    print("\n难度排序（last val/recon_mse 升序=更容易）：")
    for i, row in enumerate(report.get("ranking") or [], 1):
        mse = row.get("last_mse")
        print(f"  {i}. {row['stem']:<16} mse={mse if mse is not None else '—'}")


def main() -> None:
    args = parse_args()
    stems = resolve_stems(args.stems)
    if args.summary_only:
        tag = str(args.summary_only)
        # 允许传入完整目录前缀
        m = re.search(r"pretrain_probe_([^/]+?)_(?:" + "|".join(map(re.escape, DEFAULT_STEMS)) + r")/?$", tag)
        if m:
            tag = m.group(1)
        elif tag.startswith("pretrain_probe_"):
            # pretrain_probe_<tag>_* → 取中间 tag：用第一个 stem 试探
            for stem in stems:
                suffix = f"_{stem}"
                name = tag[len("pretrain_probe_") :]
                if name.endswith(suffix):
                    tag = name[: -len(suffix)]
                    break
        report = summarize(tag, stems)
        print_report(report)
        out = ROOT / "runs" / "experiments" / f"pretrain_probe_{tag}_summary.json"
        out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"wrote {out}")
        return

    tag = args.tag or datetime.now().strftime("%Y%m%d_%H%M%S")
    print(f"[probe] tag={tag} stems={stems} epochs={args.epochs} seed={args.seed}", flush=True)
    failures: list[str] = []
    for stem in stems:
        code = train_one(
            stem,
            config=args.config,
            epochs=args.epochs,
            seed=args.seed,
            tag=tag,
            dry_run=args.dry_run,
        )
        if code != 0:
            failures.append(stem)
            print(f"[probe] FAILED stem={stem} exit={code}", flush=True)
            if not args.continue_on_error:
                break
    if not args.dry_run:
        report = summarize(tag, stems)
        print_report(report)
        out = ROOT / "runs" / "experiments" / f"pretrain_probe_{tag}_summary.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"wrote {out}", flush=True)
    if failures:
        raise SystemExit(f"probe 失败: {failures}")


if __name__ == "__main__":
    main()
