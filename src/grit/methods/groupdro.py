"""GroupDRO objective and deterministic group-balanced epoch sampling."""

from __future__ import annotations

import torch

CMNIST_GROUP_COUNT = 4


def cmnist_group_ids(targets: torch.Tensor, colors: torch.Tensor) -> torch.Tensor:
    """Map binary ``(target, color)`` pairs to the fixed order 00, 01, 10, 11."""

    targets = targets.detach().cpu().to(torch.int64)
    colors = colors.detach().cpu().to(torch.int64)
    if targets.shape != colors.shape or targets.ndim != 1:
        raise ValueError("CMNIST targets and colors must be aligned vectors")
    if torch.any((targets < 0) | (targets > 1)) or torch.any(
        (colors < 0) | (colors > 1)
    ):
        raise ValueError("CMNIST GroupDRO requires binary targets and colors")
    return targets * 2 + colors


def group_balanced_epoch_indices(
    group_ids: torch.Tensor,
    *,
    group_count: int,
    generator: torch.Generator,
) -> torch.Tensor:
    """Draw one replacement epoch with inverse-frequency sample weights."""

    groups = group_ids.detach().cpu().to(torch.int64)
    if groups.ndim != 1 or groups.numel() == 0:
        raise ValueError("GroupDRO group IDs must be a non-empty vector")
    if torch.any((groups < 0) | (groups >= group_count)):
        raise ValueError("GroupDRO group ID is outside the configured group range")
    counts = torch.bincount(groups, minlength=group_count)
    if torch.any(counts == 0):
        raise ValueError("GroupDRO training data must contain every configured group")
    weights = counts.sum().to(torch.float64) / counts.to(torch.float64)
    return torch.multinomial(
        weights[groups],
        num_samples=int(groups.numel()),
        replacement=True,
        generator=generator,
    )


class GroupDroObjective:
    """Maintain adversarial group probabilities and reduce per-example losses."""

    def __init__(self, *, group_count: int, step_size: float) -> None:
        if group_count <= 0 or step_size <= 0.0:
            raise ValueError("GroupDRO group count and step size must be positive")
        self._group_count = group_count
        self._step_size = step_size
        self._probabilities = torch.full(
            (group_count,), 1.0 / group_count, dtype=torch.float32
        )

    @property
    def probabilities(self) -> torch.Tensor:
        return self._probabilities.clone()

    def __call__(
        self, per_example_losses: torch.Tensor, group_ids: torch.Tensor
    ) -> torch.Tensor:
        losses = per_example_losses
        groups = group_ids.detach().cpu().to(torch.int64)
        if losses.ndim != 1 or groups.shape != losses.shape:
            raise ValueError("GroupDRO losses and group IDs must be aligned vectors")
        if torch.any((groups < 0) | (groups >= self._group_count)):
            raise ValueError("GroupDRO group ID is outside the configured group range")
        group_map = torch.nn.functional.one_hot(
            groups, num_classes=self._group_count
        ).to(losses.dtype)
        group_counts = group_map.sum(dim=0)
        denominators = group_counts + (group_counts == 0).to(losses.dtype)
        group_losses = group_map.transpose(0, 1).matmul(losses) / denominators
        with torch.no_grad():
            updated = self._probabilities * torch.exp(
                self._step_size * group_losses.detach()
            )
            self._probabilities = updated / updated.sum()
        return group_losses.dot(self._probabilities)
