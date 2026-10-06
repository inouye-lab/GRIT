"""RDM: match the worst training environment's risk distribution to the pooled one."""

from __future__ import annotations

import torch

from grit.methods.invariance import environment_mean_losses

# The reference's summed Gaussian kernel, keyed by inverse bandwidth.
RDM_KERNEL_GAMMAS: tuple[float, ...] = (
    0.0001,
    0.001,
    0.01,
    0.1,
    1.0,
    10.0,
    100.0,
    1000.0,
)
RDM_MINIMUM_ENVIRONMENT_ROWS = 2


def _squared_distances(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    left_norm = left.pow(2).sum(dim=-1, keepdim=True)
    right_norm = right.pow(2).sum(dim=-1, keepdim=True)
    distances = torch.addmm(
        right_norm.transpose(-2, -1), left, right.transpose(-2, -1), alpha=-2
    ) + left_norm
    return distances.clamp_min(1e-30)


def _summed_gaussian_kernel(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    distances = _squared_distances(left, right)
    # Accumulated in the reference's order, which matters at float32 precision.
    kernel = torch.zeros_like(distances)
    for gamma in RDM_KERNEL_GAMMAS:
        kernel = kernel + torch.exp(distances * -gamma)
    return kernel


def gaussian_mmd(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    """The reference biased MMD estimator; diagonal kernel entries are kept."""

    return (
        _summed_gaussian_kernel(left, left).mean()
        + _summed_gaussian_kernel(right, right).mean()
        - 2.0 * _summed_gaussian_kernel(left, right).mean()
    )


def require_rdm_environment_rows(
    counts: tuple[int, ...],
) -> None:
    """Reject a batch the sample-variance penalty cannot be evaluated on."""

    if any(count < RDM_MINIMUM_ENVIRONMENT_ROWS for count in counts):
        raise ValueError(
            "RDM requires at least "
            f"{RDM_MINIMUM_ENVIRONMENT_ROWS} examples per environment in every "
            f"minibatch, including the epoch remainder; got {counts}"
        )


def rdm_objective(
    per_example_losses: torch.Tensor,
    environment_ids: torch.Tensor,
    *,
    environment_count: int,
    penalty_weight: float,
    variance_weight: float,
) -> torch.Tensor:
    """Mean environment risk plus the matching MMD and the risk-variance penalty."""

    risks = environment_mean_losses(
        per_example_losses,
        environment_ids,
        environment_count=environment_count,
    )
    ids = environment_ids.to(device=per_example_losses.device, dtype=torch.int64)
    require_rdm_environment_rows(
        tuple(
            int((ids == environment).sum()) for environment in range(environment_count)
        )
    )
    if penalty_weight == 0.0 and variance_weight == 0.0:
        return risks.mean()
    # Ties take the lowest environment ID, which is what argmax already returns.
    worst = per_example_losses[ids == int(torch.argmax(risks.detach()))]
    matching = gaussian_mmd(worst.unsqueeze(1), per_example_losses.unsqueeze(1))
    variance = per_example_losses.var(correction=1) + worst.var(correction=1)
    return risks.mean() + penalty_weight * matching + variance_weight * variance
