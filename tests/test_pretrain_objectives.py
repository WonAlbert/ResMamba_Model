from __future__ import annotations

import torch
import torch.nn.functional as F

from resmamba_signal_model.models.ema import EMATeacher
from resmamba_signal_model.models.model import SignalFoundationModel, SignalModelConfig
from resmamba_signal_model.training.losses import foundation_pretrain_losses, latent_prediction_loss


def _tiny(**kwargs) -> SignalFoundationModel:
    payload = dict(
        d_model=32,
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
        num_datasets=4,
        build_task_heads=False,
        build_task_interface=True,
        build_prototype_registry=False,
    )
    payload.update(kwargs)
    return SignalFoundationModel(SignalModelConfig(**payload))


def _batch(n: int = 4, length: int = 64) -> dict[str, torch.Tensor]:
    return {
        "iq": torch.randn(n, 2, length),
        "sample_mask": torch.ones(n, length, dtype=torch.bool),
        "dataset_id": torch.arange(n) % 3,
    }


def test_pretrain_three_objectives_and_uti_readouts() -> None:
    torch.manual_seed(0)
    model = _tiny()
    model.train()
    batch = _batch()
    out = model(batch, mode="pretrain")
    assert "uti_pooled" in out and "uti_tokens" in out and "uti_query" in out
    assert "z_enc" in out and "z_recon" in out
    assert out["uti_pooled"].shape[0] == 4
    # latent 对齐 decoder 重建读出；vicreg 作用在 encoder 池化
    out["teacher_z"] = out["z_recon"].detach() + 0.01
    out["teacher_h"] = out.get("h_recon", out["h_general"]).detach()
    out["teacher_pooled"] = out["uti_pooled"].detach()
    out["teacher_tokens"] = out["uti_tokens"].detach()
    out["teacher_query"] = out["uti_query"].detach()
    losses = foundation_pretrain_losses(
        out,
        batch,
        include={"mae", "structure", "latent", "vicreg", "uti_pooled", "uti_token", "uti_query"},
    )
    for key in ("mae", "structure", "latent", "vicreg", "uti_pooled", "uti_token", "uti_query"):
        assert key in losses
        assert torch.isfinite(losses[key])
    assert float(losses["structure"].detach()) > 0.0
    assert float(losses["latent"].detach()) >= 0.0
    assert float(losses["vicreg"].detach()) >= 0.0
    total = sum(losses.values())
    total.backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters() if p.requires_grad)


def test_latent_prediction_stops_teacher_grad() -> None:
    student = torch.randn(3, 8, requires_grad=True)
    teacher = torch.randn(3, 8, requires_grad=True)
    loss = latent_prediction_loss(student, teacher)
    loss.backward()
    assert student.grad is not None and student.grad.abs().sum() > 0
    assert teacher.grad is None


def test_grl_does_not_flow_into_z_general() -> None:
    torch.manual_seed(0)
    model = _tiny()
    model.train()
    batch = _batch(n=4, length=32)
    out = model(batch, mode="pretrain")
    out["z"].retain_grad()
    loss = F.cross_entropy(out["domain_logits"], batch["dataset_id"].clamp(max=model.cfg.num_datasets - 1))
    loss.backward()
    z_grad = out["z"].grad
    assert z_grad is None or float(z_grad.abs().sum()) < 1.0e-8
    adapter = model.task_interface.view_adapters["semantic"]
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in adapter.parameters())


def test_ema_teacher_exists_only_as_shadow_copy() -> None:
    student = _tiny()
    teacher = EMATeacher(student, momentum=0.5)
    n_student = sum(p.numel() for p in student.parameters())
    n_teacher = sum(p.numel() for p in teacher.parameters())
    assert n_teacher > 0
    before = next(teacher.model.encoder.parameters()).detach().clone()
    with torch.no_grad():
        for param in student.parameters():
            param.add_(0.1)
    teacher.update(student)
    after = next(teacher.model.encoder.parameters())
    assert not torch.equal(before, after)
    batch = _batch(n=2, length=32)
    out = teacher.forward_unmasked(batch)
    assert out["z"].requires_grad is False
    assert "mae_mask" in out
    assert not bool(out["mae_mask"].any())



def test_quiet_observed_masked_pulse_mae_stays_below_clamp() -> None:
    torch.manual_seed(0)
    model = _tiny(revin_clip=8.0, revin_std_min=1.0e-2)
    model.eval()
    iq = torch.full((2, 2, 64), 1.0e-4)
    iq[:, :, 48:] = 3.0
    sample_mask = torch.ones(2, 64, dtype=torch.bool)
    out = model(iq, sample_mask, mode="pretrain", mask_mode="suffix")
    assert out["suffix_mask"].any()
    assert float(out["patch_targets_norm"].detach().abs().max()) <= 8.0 + 1.0e-5
    losses = foundation_pretrain_losses(
        out,
        include={"mae", "impute", "physical", "structure"},
    )
    assert float(losses["mae"].detach()) < 10.0
    assert float(losses["structure"].detach()) <= 10.0
    assert float(losses["structure_spectrum"].detach()) <= 10.0
    assert float(losses["structure_time"].detach()) <= 10.0
    assert all(torch.isfinite(v.detach()) for v in losses.values())



def test_real_scale_iq_two_steps_stay_finite() -> None:
    torch.manual_seed(0)
    model = _tiny(revin_clip=8.0, revin_std_min=1.0e-2)
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=1.0e-4)
    iq = torch.zeros(4, 2, 128)
    iq[:, :, :16] = 1.0e-4
    iq[:, :, 96:] = 5.0e4
    batch = {
        "iq": iq,
        "sample_mask": torch.ones(4, 128, dtype=torch.bool),
        "dataset_id": torch.arange(4) % 3,
    }
    for _ in range(2):
        opt.zero_grad(set_to_none=True)
        out = model(batch, mode="pretrain", mask_mode="suffix")
        losses = foundation_pretrain_losses(out, batch)
        total = sum(losses.values())
        assert torch.isfinite(total), {k: float(v.detach()) for k, v in losses.items()}
        total.backward()
        for name, param in model.named_parameters():
            if param.grad is not None:
                assert torch.isfinite(param.grad).all(), name
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        for name, param in model.named_parameters():
            assert torch.isfinite(param).all(), name
