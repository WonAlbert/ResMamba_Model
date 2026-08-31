from __future__ import annotations

import torch

from resmamba_signal_model.models.elastic_sampler import (
    ElasticRoVSampler,
    compute_rov_weights,
    gather_elastic_patches,
)
from resmamba_signal_model.models.tokenizer import TimeFreqTokenizer, TimeFreqTokenizerConfig


def test_elastic_rov_sampler_shape() -> None:
    sampler = ElasticRoVSampler(patch_size=8, length_scale=0.5)
    iq = torch.randn(2, 2, 64)
    patches, mask, starts = sampler(iq, torch.ones(2, 64, dtype=torch.bool))
    assert patches.shape == (2, 8, 2, 8)
    assert mask.shape == (2, 8)
    assert starts.shape == (2, 8)
    assert torch.isfinite(patches).all()


def test_elastic_tokenizer_matches_fixed_token_count() -> None:
    tok = TimeFreqTokenizer(
        TimeFreqTokenizerConfig(
            d_model=32,
            patch_size=8,
            stem_channels=8,
            freq_bands=4,
            tokenization_mode="elastic_rov",
        )
    )
    iq = torch.randn(2, 2, 64)
    out = tok(iq)
    assert out["tokens"].shape[1] == 8
    assert "elastic_start_indices" in out
    assert out["iq_patch_targets"].shape == (2, 8, 2, 8)


def test_rov_weights_peak_on_jump() -> None:
    t = torch.arange(32, dtype=torch.float32)
    wave = torch.zeros(32)
    wave[16:] = 1.0
    iq = torch.stack([wave, torch.zeros_like(wave)], dim=0).unsqueeze(0)
    rov = compute_rov_weights(iq, None)
    assert rov[0, 15] > rov[0, :5].mean()


def test_gather_elastic_patches() -> None:
    iq = torch.randn(1, 2, 32)
    starts = torch.tensor([[0, 8, 16, 24]])
    patches, mask = gather_elastic_patches(iq, starts, patch_size=8)
    assert patches.shape == (1, 4, 2, 8)
    assert mask.shape == (1, 4)
