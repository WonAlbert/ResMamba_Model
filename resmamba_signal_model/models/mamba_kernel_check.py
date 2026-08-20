from __future__ import annotations

import torch

from resmamba_signal_model.models.mamba_backbone import build_mamba_block, mamba2_available


def run_mamba_kernel_smoke_test(
    *,
    d_model: int = 64,
    seq_len: int = 128,
    d_state: int = 16,
    headdim: int = 32,
    scan_direction: str = "bidirectional",
    require_mamba_kernel: bool = True,
    allow_fallback_mamba: bool = False,
    device: str | torch.device = "cuda",
    dtype: torch.dtype = torch.bfloat16,
) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for mamba kernel smoke test.")
    if require_mamba_kernel and not mamba2_available():
        raise RuntimeError("Mamba2 CUDA kernel is required but unavailable.")

    block = build_mamba_block(
        scan_direction,
        d_model,
        d_state=d_state,
        headdim=headdim,
        require_mamba_kernel=require_mamba_kernel,
        allow_fallback_mamba=allow_fallback_mamba,
    ).to(device=device, dtype=dtype)
    block.eval()
    x = torch.randn(2, seq_len, d_model, device=device, dtype=dtype)
    with torch.no_grad():
        y = block(x)
    if y.shape != x.shape:
        raise RuntimeError(f"Mamba smoke test shape mismatch: {y.shape} vs {x.shape}")
    if not torch.isfinite(y).all():
        raise RuntimeError("Mamba smoke test produced non-finite values.")
