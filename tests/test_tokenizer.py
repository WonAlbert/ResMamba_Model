from __future__ import annotations

import torch

from resmamba_signal_model.models.tokenizer import TimeFreqTokenizer, TimeFreqTokenizerConfig


def test_tokenizer_variable_lengths() -> None:
    tok = TimeFreqTokenizer(TimeFreqTokenizerConfig(d_model=32, patch_size=8, stem_channels=8, freq_bands=4, l_min=16))
    for length in (16, 20, 33, 128):
        iq = torch.randn(2, 2, length)
        out = tok(iq)
        n = (length + 7) // 8
        assert out["tokens"].shape[0] == 2
        assert out["tokens"].shape[1] == n
        assert out["tokens"].shape[2] == 32
        assert out["patch_mask"].shape == (2, n)
        assert "max_tokens" not in out


def test_time_freq_token_length_match() -> None:
    tok = TimeFreqTokenizer(TimeFreqTokenizerConfig(d_model=32, patch_size=8, stem_channels=8, freq_bands=4))
    iq = torch.randn(1, 2, 80)
    out = tok(iq)
    assert out["tokens"].shape[1] == out["iq_patch_targets"].shape[1] == out["patch_physics"].shape[1]


def test_tokenizer_no_task_or_max_tokens() -> None:
    tok = TimeFreqTokenizer(TimeFreqTokenizerConfig(d_model=16, patch_size=8, stem_channels=8, freq_bands=4))
    assert not hasattr(tok, "dataset_embed")
    assert not hasattr(tok, "task_embed")
    assert not hasattr(tok, "pos")
    out = tok(torch.randn(1, 2, 200))
    assert out["tokens"].shape[1] == 25


def test_phase_plugin_is_opt_in_and_changes_tokens() -> None:
    torch.manual_seed(0)
    iq = torch.randn(2, 2, 64)
    off = TimeFreqTokenizer(TimeFreqTokenizerConfig(d_model=16, patch_size=8, stem_channels=8, freq_bands=4, phase_plugin=False))
    on = TimeFreqTokenizer(TimeFreqTokenizerConfig(d_model=16, patch_size=8, stem_channels=8, freq_bands=4, phase_plugin=True))
    assert off.phase_proj is None
    assert on.phase_proj is not None
    with torch.no_grad():
        on.stem.weight.copy_(off.stem.weight)
        on.stem.bias.copy_(off.stem.bias)
        for a, b in zip(on.time_branches, off.time_branches):
            a.weight.copy_(b.weight)
            if a.bias is not None and b.bias is not None:
                a.bias.copy_(b.bias)
        on.time_fuse.weight.copy_(off.time_fuse.weight)
        on.time_fuse.bias.copy_(off.time_fuse.bias)
        on.freq_proj.weight.copy_(off.freq_proj.weight)
        on.freq_proj.bias.copy_(off.freq_proj.bias)
        on.gate.weight.copy_(off.gate.weight)
        on.gate.bias.copy_(off.gate.bias)
        if on.physics_proj is not None and off.physics_proj is not None:
            on.physics_proj.weight.copy_(off.physics_proj.weight)
            on.physics_proj.bias.copy_(off.physics_proj.bias)
        on.norm.weight.copy_(off.norm.weight)
        on.norm.bias.copy_(off.norm.bias)
    out_off = off(iq)
    out_on = on(iq)
    assert out_on["tokens"].shape == out_off["tokens"].shape
    assert not torch.allclose(out_on["tokens"], out_off["tokens"])

