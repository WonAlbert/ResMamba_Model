#!/usr/bin/env python
"""72h / 6×5090 SOTA 实验门控入口。默认 dry-run，不启动正式训练。"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from resmamba_signal_model.training.sota_gate import main


if __name__ == "__main__":
    raise SystemExit(main())
