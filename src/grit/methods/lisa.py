"""LISA minibatch construction and feature-space mixup on frozen features."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as functional


@dataclass(frozen=True, slots=True)
class LisaBatchPlan:
    """One planned minibatch: anchor rows, partner rows, and per-row mixing weights."""

    rows: torch.Tensor
    partner_rows: torch.Tensor
    intra_label: bool
    mixing_weights: torch.Tensor


def _rows_by_group(
    group_ids: torch.Tensor, group_count: int
) -> tuple[torch.Tensor, ...]:
    groups = group_ids.detach().cpu().to(torch.int64)
    if groups.ndim != 1 or groups.numel() == 0:
        raise ValueError("LISA group IDs must be a non-empty vector")
    if torch.any((groups < 0) | (groups >= group_count)):
        raise ValueError("LISA group ID is outside the configured group range")
    rows = tuple(
        torch.nonzero(groups == group, as_tuple=False).flatten()
        for group in range(group_count)
    )
    if any(int(item.numel()) == 0 for item in rows):
        raise ValueError("LISA training data must contain every configured group")
    return rows


def _sample_rows(
    pool: torch.Tensor, count: int, generator: torch.Generator
) -> torch.Tensor:
    """Sample without replacement when the pool allows it, as the WILDS sampler does."""

    size = int(pool.numel())
    if size >= count:
        return pool[torch.randperm(size, generator=generator)[:count]]
    return pool[torch.randint(size, (count,), generator=generator)]


def beta_2_2(count: int, generator: torch.Generator) -> torch.Tensor:
    """Draw Beta(2, 2) weights as the median of three uniforms from the generator."""

    uniforms = torch.rand((count, 3), generator=generator, dtype=torch.float32)
    return uniforms.sort(dim=1).values[:, 1]


def plan_lisa_epoch(
    group_ids: torch.Tensor,
    *,
    group_count: int,
    batch_size: int,
    selection_prob: float,
    generator: torch.Generator,
) -> tuple[LisaBatchPlan, ...]:
    """Plan one epoch of single-group minibatches with their mixing partners.

    Groups are ``target * 2 + color``. Intra-label batches mix with the group that
    shares the target and flips the color; intra-domain batches mix with the group
    that shares the color and flips the target.
    """

    if batch_size <= 0:
        raise ValueError("LISA batch size must be positive")
    if not 0.0 <= selection_prob <= 1.0:
        raise ValueError("LISA selection probability must lie in [0, 1]")
    if group_count != 4:
        raise ValueError("CMNIST LISA requires the four target-color groups")
    rows_by_group = _rows_by_group(group_ids, group_count)
    batch_count = int(group_ids.numel()) // batch_size
    if batch_count == 0:
        raise ValueError("LISA epoch requires at least one full minibatch")
    plans: list[LisaBatchPlan] = []
    for _ in range(batch_count):
        group = int(torch.randint(group_count, (1,), generator=generator).item())
        rows = _sample_rows(rows_by_group[group], batch_size, generator)
        intra_label = bool(
            torch.rand((1,), generator=generator).item() < selection_prob
        )
        target, color = divmod(group, 2)
        partner_group = (
            target * 2 + (1 - color) if intra_label else (1 - target) * 2 + color
        )
        partner_rows = _sample_rows(rows_by_group[partner_group], batch_size, generator)
        plans.append(
            LisaBatchPlan(
                rows=rows,
                partner_rows=partner_rows,
                intra_label=intra_label,
                mixing_weights=beta_2_2(batch_size, generator),
            )
        )
    return tuple(plans)


def mix_features_and_targets(
    features: torch.Tensor,
    targets: torch.Tensor,
    plan: LisaBatchPlan,
    *,
    num_classes: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return mixed features and mixed one-hot targets for one planned batch."""

    weights = plan.mixing_weights.to(torch.float32).unsqueeze(1)
    anchor = features[plan.rows].to(torch.float32)
    partner = features[plan.partner_rows].to(torch.float32)
    mixed_features = weights * anchor + (1.0 - weights) * partner
    anchor_targets = functional.one_hot(
        targets[plan.rows].to(torch.int64), num_classes=num_classes
    ).to(torch.float32)
    partner_targets = functional.one_hot(
        targets[plan.partner_rows].to(torch.int64), num_classes=num_classes
    ).to(torch.float32)
    mixed_targets = weights * anchor_targets + (1.0 - weights) * partner_targets
    return mixed_features, mixed_targets


def soft_cross_entropy(
    logits: torch.Tensor, soft_targets: torch.Tensor
) -> torch.Tensor:
    """Mean over the batch of the cross-entropy against soft target distributions."""

    if logits.shape != soft_targets.shape:
        raise ValueError("soft cross-entropy requires aligned logits and targets")
    return -(functional.log_softmax(logits, dim=1) * soft_targets).sum(dim=1).mean()
