from __future__ import annotations

import torch

from resmamba_signal_model.models.model import SignalFoundationModel, SignalModelConfig
from resmamba_signal_model.models.signal_adapter import (
    ChannelProjectionAdapter,
    SignalAdapterRegistry,
    SignalSpec,
)


def _cfg() -> SignalModelConfig:
    return SignalModelConfig(
        d_model=32,
        encoder_mamba_layers=1,
        mamba_d_state=8,
        mamba_headdim=16,
        require_mamba_kernel=False,
        allow_fallback_mamba=True,
        attn_num_heads=4,
        patch_size=8,
        stem_channels=8,
        freq_bands=4,
        dropout=0.0,
        p_trunc=0.0,
        build_task_heads=False,
    )


def test_unregistered_signal_specs_have_safe_arbitrary_channel_fallback() -> None:
    registry = SignalAdapterRegistry()
    iq = torch.randn(2, 2, 32)
    assert torch.equal(registry(iq, SignalSpec(name="rf", num_channels=2)), iq)

    mono = torch.randn(2, 1, 32)
    mono_out = registry(mono, SignalSpec(name="sonar", num_channels=1, modality="sonar"))
    assert mono_out.shape == (2, 2, 32)
    assert torch.equal(mono_out[:, 0], mono[:, 0])
    assert torch.count_nonzero(mono_out[:, 1]) == 0

    imu = torch.randn(2, 6, 32)
    channel_mask = torch.tensor([[1, 1, 1, 1, 1, 1], [1, 1, 1, 0, 0, 0]], dtype=torch.bool)
    imu_out = registry(
        imu,
        SignalSpec(name="imu", num_channels=6, modality="imu"),
        channel_mask,
    )
    assert imu_out.shape == (2, 2, 32)
    assert torch.isfinite(imu_out).all()


def test_registered_learnable_channel_projection() -> None:
    registry = SignalAdapterRegistry()
    spec = SignalSpec(name="imu6", num_channels=6, modality="imu")
    adapter = registry.register(spec)
    assert isinstance(adapter, ChannelProjectionAdapter)
    values = torch.randn(3, 6, 24, requires_grad=True)
    projected = registry(values, spec)
    projected.square().mean().backward()
    assert projected.shape == (3, 2, 24)
    assert adapter.proj.weight.grad is not None


def test_model_accepts_values_protocol_without_data_proxy_dependency() -> None:
    model = SignalFoundationModel(_cfg()).eval()
    sample_mask = torch.ones(2, 64, dtype=torch.bool)
    sonar = {
        "values": torch.randn(2, 1, 64),
        "sample_mask": sample_mask,
        "signal_spec": SignalSpec("sonar_mono", 1, "sonar"),
    }
    sonar_out = model(sonar, mode="encode")
    assert sonar_out["z_general"].shape == (2, 32)

    imu_out = model(
        values := torch.randn(2, 6, 64),
        sample_mask,
        mode="encode",
        signal_spec=SignalSpec("imu6", 6, "imu"),
        channel_mask=torch.ones(2, 6, dtype=torch.bool),
    )
    assert values.shape[1] == 6
    assert imu_out["h_general"].shape[0] == 2
