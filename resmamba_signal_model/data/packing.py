from __future__ import annotations

import torch


def build_cu_seqlens(lengths: list[int] | torch.Tensor, *, device: torch.device, dtype: torch.dtype = torch.int32) -> torch.Tensor:
    if isinstance(lengths, torch.Tensor):
        lengths = lengths.tolist()
    if not lengths:
        raise ValueError("lengths 不能为空")
    cu = [0]
    for length in lengths:
        if length < 0:
            raise ValueError(f"非法序列长度 {length}")
        cu.append(cu[-1] + int(length))
    return torch.tensor(cu, device=device, dtype=dtype)


def build_seq_idx(lengths: list[int] | torch.Tensor, *, device: torch.device, dtype: torch.dtype = torch.int32) -> torch.Tensor:
    if isinstance(lengths, torch.Tensor):
        lengths = lengths.tolist()
    total = sum(int(x) for x in lengths)
    seq_idx = torch.empty(total, device=device, dtype=dtype)
    offset = 0
    for sample_id, length in enumerate(lengths):
        if length > 0:
            seq_idx[offset : offset + length] = sample_id
            offset += int(length)
    return seq_idx.unsqueeze(0)


def segment_slices(cu_seqlens: torch.Tensor) -> list[tuple[int, int]]:
    bounds = cu_seqlens.tolist()
    return [(int(start), int(end)) for start, end in zip(bounds[:-1], bounds[1:])]


def flip_segments(x: torch.Tensor, cu_seqlens: torch.Tensor, *, dim: int = 1) -> torch.Tensor:
    out = x.clone()
    for start, end in segment_slices(cu_seqlens):
        if end > start:
            out.narrow(dim, start, end - start).copy_(torch.flip(x.narrow(dim, start, end - start), dims=[dim]))
    return out


def apply_segments(x: torch.Tensor, cu_seqlens: torch.Tensor, fn, *, dim: int = 1) -> torch.Tensor:
    parts: list[torch.Tensor] = []
    for start, end in segment_slices(cu_seqlens):
        seg = x.narrow(dim, start, end - start)
        parts.append(fn(seg))
    return torch.cat(parts, dim=dim)


def pad_sequence_list(tensors: list[torch.Tensor], *, pad: int | float = 0) -> tuple[torch.Tensor, torch.Tensor]:
    if not tensors:
        raise ValueError("tensors 不能为空")
    max_len = max(t.shape[0] for t in tensors)
    device = tensors[0].device
    dtype = tensors[0].dtype
    out = torch.full((len(tensors), max_len, *tensors[0].shape[1:]), pad, device=device, dtype=dtype)
    mask = torch.zeros(len(tensors), max_len, dtype=torch.bool, device=device)
    for i, tensor in enumerate(tensors):
        n = tensor.shape[0]
        out[i, :n] = tensor
        mask[i, :n] = True
    return out, mask


def segment_lengths(cu_seqlens: torch.Tensor) -> list[int]:
    return [end - start for start, end in segment_slices(cu_seqlens)]


def unpack_packed_tokens(x: torch.Tensor, cu_seqlens: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """(1, total, D) -> (B, max_len, D) + valid mask。"""
    if x.ndim != 3 or x.shape[0] != 1:
        raise ValueError(f"unpack_packed_tokens 期望 (1, total, D)，当前 {tuple(x.shape)}")
    slices = segment_slices(cu_seqlens)
    lengths = [end - start for start, end in slices]
    max_len = max(lengths) if lengths else 0
    padded = x.new_zeros(len(slices), max_len, x.shape[-1])
    mask = torch.zeros(len(slices), max_len, dtype=torch.bool, device=x.device)
    for i, (start, end) in enumerate(slices):
        n = end - start
        if n > 0:
            padded[i, :n] = x[0, start:end]
            mask[i, :n] = True
    return padded, mask


def repack_tokens(y: torch.Tensor, cu_seqlens: torch.Tensor) -> torch.Tensor:
    """(B, max_len, D) -> (1, total, D)。"""
    slices = segment_slices(cu_seqlens)
    parts = [y[i, : end - start].unsqueeze(0) for i, (start, end) in enumerate(slices)]
    if not parts:
        return y.new_zeros(1, 0, y.shape[-1])
    return torch.cat(parts, dim=1)
