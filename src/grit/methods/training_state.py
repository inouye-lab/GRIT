"""Shared linear-probe predictor state."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True, slots=True)
class LinearProbeState:
    weight: torch.Tensor
    bias: torch.Tensor
    # None means a historical state whose caller supplies the fitted projection.
    # Modern vanilla predictors embed an empty basis, explicitly clearing projection.
    projection_basis: torch.Tensor | None = None
