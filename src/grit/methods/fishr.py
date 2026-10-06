"""Fishr: analytic per-example classifier gradients and their variance penalty.

The reference implementation (DomainBed ``b93c22a``) reads per-example classifier
gradients out of BackPACK. A linear probe over frozen features needs no autograd
extension: for one example with prepared feature vector ``z``, softmax probabilities
``p``, and one-hot target ``q``, the gradients of that example's cross-entropy are
exactly ``g_W = (p - q) z^T`` and ``g_b = p - q``. They are written here in closed
form so they stay differentiable with respect to the classifier.
"""

from __future__ import annotations

import torch
import torch.nn.functional as functional


def fishr_example_gradients(
    logits: torch.Tensor,
    targets: torch.Tensor,
    features: torch.Tensor,
) -> torch.Tensor:
    """Per-example weight and bias gradients, flattened to ``[N, classes * (D + 1)]``.

    These are gradients of the individual losses, without a minibatch-mean factor,
    and they retain their dependence on the classifier through ``logits``.
    """

    if logits.ndim != 2:
        raise ValueError("Fishr logits must have shape [N, classes]")
    if features.ndim != 2 or int(features.shape[0]) != int(logits.shape[0]):
        raise ValueError("Fishr features must align with logits")
    if targets.ndim != 1 or int(targets.shape[0]) != int(logits.shape[0]):
        raise ValueError("Fishr targets must align with logits")
    num_classes = int(logits.shape[1])
    residual = functional.softmax(logits, dim=1) - functional.one_hot(
        targets, num_classes=num_classes
    ).to(logits.dtype)
    weight_gradients = residual.unsqueeze(2) * features.unsqueeze(1)
    return torch.cat(
        (weight_gradients.flatten(start_dim=1), residual),
        dim=1,
    )


class FishrGradientVarianceEma:
    """The reference moving average with its ``1 / (1 - ema)`` gradient correction.

    One state vector per environment holds the concatenated weight and bias
    coordinates; the reference keeps one per parameter tensor, which is the same
    arithmetic because every operation is elementwise.
    """

    def __init__(self, *, environment_count: int, decay: float) -> None:
        if environment_count <= 1:
            raise ValueError("Fishr requires at least two environments")
        if not 0.0 <= decay < 1.0:
            raise ValueError("Fishr EMA decay must lie in [0, 1)")
        self._decay = decay
        self._state: list[torch.Tensor | None] = [None] * environment_count

    @property
    def update_count(self) -> int:
        return sum(1 for state in self._state if state is not None)

    def state(self, environment: int) -> torch.Tensor | None:
        """The detached historical average; exposed so tests can read it."""

        return self._state[environment]

    def update(self, environment: int, variances: torch.Tensor) -> torch.Tensor:
        previous = self._state[environment]
        if previous is None:
            previous = torch.zeros_like(variances.detach())
        averaged = self._decay * previous + (1.0 - self._decay) * variances
        self._state[environment] = averaged.detach().clone()
        return averaged / (1.0 - self._decay)


def fishr_penalty(
    logits: torch.Tensor,
    targets: torch.Tensor,
    features: torch.Tensor,
    environment_ids: torch.Tensor,
    *,
    environment_count: int,
    ema: FishrGradientVarianceEma,
) -> torch.Tensor:
    """Mean squared distance from each environment's gradient variance to their mean.

    The EMA is advanced on every call, including warm-up updates, matching the
    reference, which always computes the penalty and only zeroes its weight.
    """

    gradients = fishr_example_gradients(logits, targets, features)
    ids = environment_ids.to(device=logits.device, dtype=torch.int64)
    if ids.shape != targets.shape:
        raise ValueError("Fishr environment IDs must align with the minibatch")
    corrected: list[torch.Tensor] = []
    for environment in range(environment_count):
        mask = ids == environment
        if int(mask.sum()) == 0:
            raise ValueError("every Fishr minibatch must contain every environment")
        environment_gradients = gradients[mask]
        centered = environment_gradients - environment_gradients.mean(
            dim=0, keepdim=True
        )
        corrected.append(ema.update(environment, centered.pow(2).mean(dim=0)))
    mean_variance = torch.stack(corrected, dim=0).mean(dim=0)
    return torch.stack(
        [(variance - mean_variance).pow(2).mean() for variance in corrected]
    ).mean()
