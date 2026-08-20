from __future__ import annotations

from pathlib import Path
import sys

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from resmamba_signal_model.models.mamba_backbone import (
    _build_mamba2,
    build_mamba_block,
    get_mamba_runtime_info,
    mamba2_available,
)
from resmamba_signal_model.models.mamba_kernel_check import run_mamba_kernel_smoke_test


def test_get_mamba_runtime_info() -> None:
    info = get_mamba_runtime_info(scan_direction="bidirectional")
    assert info["mamba_operator"] == "mamba2"
    assert info["scan_direction"] == "bidirectional"
    assert info["fallback"] == (not mamba2_available())


def test_require_kernel_raises_without_cuda_kernel(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("resmamba_signal_model.models.mamba_backbone._MAMBA2_AVAILABLE", False)
    with pytest.raises(RuntimeError, match="Mamba2 CUDA kernel is required"):
        _build_mamba2(32, d_state=8, d_conv=4, expand=2, headdim=16, require_mamba_kernel=True, allow_fallback_mamba=False)


def test_allow_fallback_when_not_required(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("resmamba_signal_model.models.mamba_backbone._MAMBA2_AVAILABLE", False)
    block = _build_mamba2(32, d_state=8, d_conv=4, expand=2, headdim=16, require_mamba_kernel=False, allow_fallback_mamba=True)
    x = torch.randn(2, 16, 32)
    y = block(x)
    assert y.shape == x.shape
    assert torch.isfinite(y).all()


def test_build_mamba_block_unidirectional() -> None:
    block = build_mamba_block(
        "unidirectional",
        32,
        d_state=8,
        headdim=16,
        require_mamba_kernel=False,
        allow_fallback_mamba=True,
    )
    x = torch.randn(2, 16, 32)
    y = block(x)
    assert y.shape == x.shape


def test_cuda_smoke_when_available() -> None:
    if not torch.cuda.is_available() or not mamba2_available():
        pytest.skip("CUDA mamba kernel unavailable")
    run_mamba_kernel_smoke_test(
        d_model=32,
        seq_len=64,
        d_state=8,
        headdim=16,
        require_mamba_kernel=False,
        allow_fallback_mamba=True,
    )
