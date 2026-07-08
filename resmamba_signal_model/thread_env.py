from __future__ import annotations

import os


def normalize_thread_env(default: str | None = None) -> None:
    """libgomp 不接受 OMP_NUM_THREADS=0；在导入 numpy/torch 前调用。"""
    if default is None:
        default = str(min(8, os.cpu_count() or 1))
    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        if os.environ.get(key, "0") in ("", "0"):
            os.environ[key] = default
