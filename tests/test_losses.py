import torch

from resmamba_signal_model.training.losses import safe_cross_entropy


def test_safe_cross_entropy_filters_invalid_labels() -> None:
    logits = torch.tensor([[10.0, 0.0, -5.0], [0.0, 8.0, 1.0], [1.0, 2.0, 3.0]])
    labels = torch.tensor([0, 5, -1])
    loss = safe_cross_entropy(logits, labels)
    assert torch.isfinite(loss)
    assert loss.item() > 0


def test_safe_cross_entropy_clamps_large_logits() -> None:
    logits = torch.tensor([[1.0e4, -1.0e4]])
    labels = torch.tensor([0])
    loss = safe_cross_entropy(logits, labels)
    assert torch.isfinite(loss)
