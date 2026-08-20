from __future__ import annotations

import math

import torch
import torch.nn.functional as F


def n_patches(length: int, patch_size: int) -> int:
    return max(1, int(math.ceil(int(length) / int(patch_size))))


def pad_time_to_patch(iq: torch.Tensor, sample_mask: torch.Tensor | None, patch_size: int) -> tuple[torch.Tensor, torch.Tensor]:
    """右 pad 到 patch_size 的整数倍；pad 位 sample_mask=False。"""
    length = int(iq.shape[-1])
    pad = (patch_size - length % patch_size) % patch_size
    if sample_mask is None:
        sample_mask = torch.ones(iq.shape[0], length, dtype=torch.bool, device=iq.device)
    if pad:
        iq = F.pad(iq, (0, pad))
        sample_mask = F.pad(sample_mask, (0, pad), value=False)
    return iq, sample_mask


def patchify_iq(
    iq: torch.Tensor,
    sample_mask: torch.Tensor | None,
    patch_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """``[B,2,L]`` → patches ``[B,N,2,P]`` 与 patch_mask（半截 patch 仍有效）。"""
    iq, sample_mask = pad_time_to_patch(iq, sample_mask, patch_size)
    patches = iq.unfold(-1, patch_size, patch_size).transpose(1, 2).contiguous()
    patch_mask = sample_mask.unfold(-1, patch_size, patch_size).any(dim=-1)
    return patches, patch_mask


def patches_to_iq(patches: torch.Tensor, length: int) -> torch.Tensor:
    """``[B,N,2,P]`` → ``[B,2,L]``（裁到原始长度）。"""
    batch, num, channels, patch = patches.shape
    wave = patches.permute(0, 2, 1, 3).reshape(batch, channels, num * patch)
    return wave[..., :length]


def assert_min_length(length: int, l_min: int) -> None:
    if int(length) < int(l_min):
        raise ValueError(f"信号长度 {length} < L_min={l_min}，拒绝静默丢弃")


def pad_iq_list(iq_list: list[torch.Tensor], l_min: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """变长 I/Q 列表 pad 成 ``[B,2,Lmax]``，返回 lengths。"""
    if not iq_list:
        raise ValueError("iq 列表为空")
    lengths = []
    for tensor in iq_list:
        length = int(tensor.shape[-1])
        assert_min_length(length, l_min)
        lengths.append(length)
    max_len = max(lengths)
    device = iq_list[0].device
    dtype = iq_list[0].dtype
    iq = torch.zeros(len(iq_list), 2, max_len, device=device, dtype=dtype)
    sample_mask = torch.zeros(len(iq_list), max_len, dtype=torch.bool, device=device)
    for i, tensor in enumerate(iq_list):
        n = lengths[i]
        iq[i, :, :n] = tensor[:, :n]
        sample_mask[i, :n] = True
    return iq, sample_mask, torch.tensor(lengths, device=device, dtype=torch.long)


def apply_truncation_aug(
    iq: torch.Tensor,
    sample_mask: torch.Tensor,
    *,
    p_trunc: float,
    l_min: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """以 ``p_trunc`` 随机取前缀/中段/后缀窗，窗长 ``U(L_min, L_valid)``。"""
    if p_trunc <= 0 or iq.shape[0] == 0:
        return iq, sample_mask
    batch, _, max_len = iq.shape
    out_iq = iq
    out_mask = sample_mask
    copied = False
    for b in range(batch):
        valid = int(sample_mask[b].sum().item())
        if valid < l_min:
            assert_min_length(valid, l_min)
        if float(torch.rand((), device=iq.device)) >= p_trunc:
            continue
        if not copied:
            out_iq = iq.clone()
            out_mask = sample_mask.clone()
            copied = True
        win = int(torch.randint(l_min, valid + 1, (), device=iq.device).item())
        mode = int(torch.randint(0, 3, (), device=iq.device).item())
        if mode == 0:
            start = 0
        elif mode == 1:
            start = valid - win
        else:
            start = int(torch.randint(0, valid - win + 1, (), device=iq.device).item())
        end = start + win
        new_mask = torch.zeros_like(out_mask[b])
        new_mask[start:end] = True
        out_mask[b] = new_mask
        out_iq[b] = out_iq[b].masked_fill(~new_mask.unsqueeze(0), 0.0)
    return out_iq, out_mask


def chunk_starts(length: int, chunk_len: int, overlap: int) -> list[int]:
    if length <= chunk_len:
        return [0]
    step = max(1, chunk_len - overlap)
    starts = list(range(0, length - chunk_len + 1, step))
    last = length - chunk_len
    if starts[-1] != last:
        starts.append(last)
    return starts


def overlap_weights(chunk_len: int, overlap: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    w = torch.ones(chunk_len, device=device, dtype=dtype)
    if overlap <= 0:
        return w
    ramp = torch.linspace(0.0, 1.0, overlap, device=device, dtype=dtype)
    w[:overlap] = ramp
    w[-overlap:] = ramp.flip(0)
    return w.clamp_min(1.0e-4)


def random_train_chunk(
    iq: torch.Tensor,
    sample_mask: torch.Tensor,
    chunk_len: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """训练时对超长样本随机抽一块。"""
    batch, _, max_len = iq.shape
    if max_len <= chunk_len:
        return iq, sample_mask
    out_iq = iq.new_zeros(batch, 2, chunk_len)
    out_mask = sample_mask.new_zeros(batch, chunk_len)
    for b in range(batch):
        valid = int(sample_mask[b].sum().item())
        if valid <= chunk_len:
            n = min(valid, chunk_len)
            out_iq[b, :, :n] = iq[b, :, :n]
            out_mask[b, :n] = sample_mask[b, :n]
            continue
        start = int(torch.randint(0, valid - chunk_len + 1, (), device=iq.device).item())
        sl = slice(start, start + chunk_len)
        out_iq[b] = iq[b, :, sl]
        out_mask[b] = sample_mask[b, sl]
    return out_iq, out_mask
