from __future__ import annotations

from typing import Any


class EarlyStopping:
    """在验证指标长期无改善时触发停止。"""

    def __init__(self, *, patience: int, min_delta: float = 0.0, mode: str = "min") -> None:
        if patience < 0:
            raise ValueError("patience 不能为负数")
        if mode not in ("min", "max"):
            raise ValueError(f"mode 必须为 min 或 max，当前为 {mode!r}")
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.best_score: float | None = None
        self.counter = 0

    @property
    def enabled(self) -> bool:
        return self.patience > 0

    def step(self, score: float) -> bool:
        """更新状态并返回是否应停止训练。"""
        if not self.enabled:
            return False

        if self.best_score is None:
            improved = True
        elif self.mode == "min":
            improved = score < self.best_score - self.min_delta
        else:
            improved = score > self.best_score + self.min_delta

        if improved:
            self.best_score = score
            self.counter = 0
            return False

        self.counter += 1
        return self.counter >= self.patience

    def state_dict(self) -> dict[str, float | int | str | None]:
        return {
            "patience": self.patience,
            "min_delta": self.min_delta,
            "mode": self.mode,
            "best_score": self.best_score,
            "counter": self.counter,
        }

    def load_state_dict(self, state: dict[str, float | int | str | None]) -> None:
        self.best_score = state.get("best_score")
        if self.best_score is not None:
            self.best_score = float(self.best_score)
        self.counter = int(state.get("counter", 0))
        mode = state.get("mode")
        if mode in ("min", "max"):
            self.mode = str(mode)


def make_early_stopping_callback(
    *,
    monitor: str,
    mode: str,
    patience: int,
    min_delta: float = 0.0,
) -> Any | None:
    """patience<=0 时关闭；否则监控与 best ckpt 相同的验证指标。"""
    if int(patience) <= 0:
        return None
    if mode not in ("min", "max"):
        raise ValueError(f"early stopping mode 必须为 min 或 max，当前为 {mode!r}")
    try:
        from lightning.pytorch.callbacks import EarlyStopping as LightningEarlyStopping
    except ImportError:  # pragma: no cover
        return None
    return LightningEarlyStopping(
        monitor=monitor,
        mode=mode,
        patience=int(patience),
        min_delta=float(min_delta),
        verbose=True,
        check_on_train_epoch_end=False,
    )
