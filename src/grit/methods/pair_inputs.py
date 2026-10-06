"""The only pair information a pair method receives: endpoint features and a label."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True, slots=True)
class PairInputs:
    """Canonically oriented endpoints (red/land first) and the pair's shared label.

    There are deliberately no color, digit, clean-label, background, or group fields;
    dataset builders drop them before a method can see them.
    """

    left: torch.Tensor
    right: torch.Tensor
    labels: torch.Tensor

    def __post_init__(self) -> None:
        left = self.left.detach().cpu().to(torch.float32).clone()
        right = self.right.detach().cpu().to(torch.float32).clone()
        labels = self.labels.detach().cpu().to(torch.int64).clone()
        if (
            left.ndim != 2
            or left.shape[0] == 0
            or left.shape != right.shape
            or not bool(torch.isfinite(left).all() & torch.isfinite(right).all())
        ):
            raise ValueError("pair endpoints must be finite, non-empty and aligned")
        if labels.shape != (int(left.shape[0]),) or bool((labels < 0).any()):
            raise ValueError("pair labels must be non-negative and align with pairs")
        object.__setattr__(self, "left", left)
        object.__setattr__(self, "right", right)
        object.__setattr__(self, "labels", labels)

    @property
    def differences(self) -> torch.Tensor:
        return self.left - self.right
