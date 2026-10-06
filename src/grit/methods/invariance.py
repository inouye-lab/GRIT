"""Environment-aware minibatches and shared frozen-feature training objectives."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as functional


def environment_balanced_epoch_batches(
    environment_ids: torch.Tensor,
    *,
    environment_count: int,
    batch_size: int,
    generator: torch.Generator,
) -> tuple[torch.Tensor, ...]:
    """Balanced minibatches that exhaust the largest environment once per epoch.

    Every environment is shuffled independently and drawn without replacement within a
    pass. The largest environment is sliced exactly once (its final batch is the
    remainder); every other environment contributes the same count per batch and
    starts a fresh shuffled pass whenever it runs out. Equal-sized environments
    therefore reduce to the original equal-slice rule with one remainder batch.
    """

    ids = environment_ids.detach().cpu().to(torch.int64)
    if ids.ndim != 1 or ids.numel() == 0:
        raise ValueError("environment IDs must be a non-empty vector")
    if environment_count <= 1:
        raise ValueError("invariant training requires at least two environments")
    if batch_size <= 0 or batch_size % environment_count != 0:
        raise ValueError(
            "environment-balanced batch size must be positive and divisible by "
            "the environment count"
        )
    if bool(((ids < 0) | (ids >= environment_count)).any()) or any(
        not bool((ids == environment).any()) for environment in range(environment_count)
    ):
        raise ValueError("invariant training requires every contiguous environment ID")
    rows_by_environment = tuple(
        torch.nonzero(ids == environment, as_tuple=False).flatten()
        for environment in range(environment_count)
    )
    per_environment = batch_size // environment_count
    largest = max(int(rows.numel()) for rows in rows_by_environment)
    batch_counts = tuple(
        min(per_environment, largest - start)
        for start in range(0, largest, per_environment)
    )
    streams: list[torch.Tensor] = []
    needed = sum(batch_counts)
    for rows in rows_by_environment:
        passes: list[torch.Tensor] = []
        drawn = 0
        while drawn < needed:
            passes.append(rows[torch.randperm(int(rows.numel()), generator=generator)])
            drawn += int(rows.numel())
        streams.append(torch.cat(passes, dim=0)[:needed])
    batches: list[torch.Tensor] = []
    offset = 0
    for count in batch_counts:
        batches.append(
            torch.cat([stream[offset : offset + count] for stream in streams], dim=0)
        )
        offset += count
    return tuple(batches)


def environment_balanced_epoch_batch_count(
    environment_ids: torch.Tensor,
    *,
    environment_count: int,
    batch_size: int,
) -> int:
    """How many batches one balanced epoch yields, without drawing them."""

    ids = environment_ids.detach().cpu().to(torch.int64)
    if environment_count <= 1:
        raise ValueError("invariant training requires at least two environments")
    if batch_size <= 0 or batch_size % environment_count != 0:
        raise ValueError(
            "environment-balanced batch size must be positive and divisible by "
            "the environment count"
        )
    per_environment = batch_size // environment_count
    largest = max(
        int((ids == environment).sum()) for environment in range(environment_count)
    )
    return len(range(0, largest, per_environment))


def require_active_penalty_updates(
    *,
    warm_up_updates: int,
    updates_per_epoch: int,
    max_epochs: int,
    label: str,
) -> None:
    """Refuse a run whose warm-up would swallow every update.

    A warmed-up method that never activates its penalty is plain ERM wearing the
    method's name, which would be reported as that method's result.
    """

    total = updates_per_epoch * max_epochs
    if warm_up_updates >= total:
        raise ValueError(
            f"{label} warm-up of {warm_up_updates} updates leaves no penalized "
            f"update in a run of {total} ({updates_per_epoch} per epoch over "
            f"{max_epochs} epochs); shorten the warm-up or lengthen the run"
        )


def environment_mean_losses(
    per_example_losses: torch.Tensor,
    environment_ids: torch.Tensor,
    *,
    environment_count: int,
) -> torch.Tensor:
    """Return one mean loss for each required training environment."""

    if per_example_losses.ndim != 1:
        raise ValueError("per-example losses must be a vector")
    ids = environment_ids.to(device=per_example_losses.device, dtype=torch.int64)
    if ids.shape != per_example_losses.shape:
        raise ValueError("environment IDs must align with per-example losses")
    if environment_count <= 1:
        raise ValueError("invariant training requires at least two environments")
    risks: list[torch.Tensor] = []
    for environment in range(environment_count):
        mask = ids == environment
        if not bool(mask.any()):
            raise ValueError(
                "every invariant-training batch must contain every environment"
            )
        risks.append(per_example_losses[mask].mean())
    if bool(((ids < 0) | (ids >= environment_count)).any()):
        raise ValueError("invariant-training environment ID is out of range")
    return torch.stack(risks)


def annealed_penalty_weight(
    configured_weight: float,
    *,
    anneal_updates: int,
    update_count: int,
) -> float:
    if not math.isfinite(configured_weight) or configured_weight <= 0.0:
        raise ValueError("penalty weight must be positive and finite")
    if anneal_updates < 0 or update_count < 0:
        raise ValueError("anneal updates and update count must be non-negative")
    return configured_weight if update_count >= anneal_updates else 1.0


def vrex_objective(
    per_example_losses: torch.Tensor,
    environment_ids: torch.Tensor,
    *,
    environment_count: int,
    penalty_weight: float,
) -> torch.Tensor:
    """Compute mean environment risk plus population risk variance."""

    risks = environment_mean_losses(
        per_example_losses,
        environment_ids,
        environment_count=environment_count,
    )
    mean_risk = risks.mean()
    penalty = ((risks - mean_risk) ** 2).mean()
    objective = mean_risk + penalty_weight * penalty
    if penalty_weight > 1.0:
        objective = objective / penalty_weight
    return objective


def irmv1_objective(
    logits: torch.Tensor,
    targets: torch.Tensor,
    environment_ids: torch.Tensor,
    *,
    environment_count: int,
    penalty_weight: float,
) -> torch.Tensor:
    """Compute the original scalar dummy-classifier IRMv1 objective."""

    if logits.ndim != 2 or int(logits.shape[0]) == 0:
        raise ValueError("IRMv1 logits must have shape [N, classes]")
    if targets.ndim != 1 or int(targets.shape[0]) != int(logits.shape[0]):
        raise ValueError("IRMv1 targets must align with logits")
    per_example = functional.cross_entropy(logits, targets, reduction="none")
    risks = environment_mean_losses(
        per_example,
        environment_ids,
        environment_count=environment_count,
    )
    ids = environment_ids.to(device=logits.device, dtype=torch.int64)
    penalties: list[torch.Tensor] = []
    for environment in range(environment_count):
        mask = ids == environment
        scale = torch.ones(
            (), device=logits.device, dtype=logits.dtype
        ).requires_grad_()
        scaled_risk = functional.cross_entropy(
            logits[mask] * scale,
            targets[mask],
        )
        gradient = torch.autograd.grad(
            scaled_risk,
            scale,
            create_graph=True,
        )[0]
        penalties.append(gradient.square())
    objective = risks.mean() + penalty_weight * torch.stack(penalties).mean()
    if penalty_weight > 1.0:
        objective = objective / penalty_weight
    return objective


def spectral_decoupling_objective(
    logits: torch.Tensor,
    targets: torch.Tensor,
    *,
    penalty_weight: float,
) -> torch.Tensor:
    """Mean cross-entropy plus the mean squared raw logit.

    The penalty averages over examples and classes and is applied to the raw logits,
    without centering or a softmax, from the first update.
    """

    if logits.ndim != 2 or int(logits.shape[0]) == 0:
        raise ValueError("SD logits must have shape [N, classes]")
    if targets.ndim != 1 or int(targets.shape[0]) != int(logits.shape[0]):
        raise ValueError("SD targets must align with logits")
    return functional.cross_entropy(logits, targets) + penalty_weight * logits.pow(
        2
    ).mean()
