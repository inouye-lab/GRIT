"""LISA, SD, Fishr, and RDM for the shared linear-probe trainer."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from functools import partial
from typing import Literal

import torch

from grit.config import (
    FishrAlgorithmConfig,
    LinearProbeTrainingConfig,
    RdmAlgorithmConfig,
)
from grit.methods.fishr import FishrGradientVarianceEma, fishr_penalty
from grit.methods.invariance import (
    environment_balanced_epoch_batch_count,
    environment_balanced_epoch_batches,
    require_active_penalty_updates,
    spectral_decoupling_objective,
)
from grit.methods.lisa import (
    LisaBatchPlan,
    mix_features_and_targets,
    plan_lisa_epoch,
    soft_cross_entropy,
)
from grit.methods.rdm import rdm_objective, require_rdm_environment_rows
from grit.methods.training import (
    LinearProbeAlgorithm,
    LinearProbeMethodDefaults,
    random_epoch_batches,
    require_aligned_ids,
)
from grit.methods.types import METHOD_LABELS


@dataclass(slots=True)
class LisaLinearProbeMethod(LinearProbeMethodDefaults):
    """Single-group minibatches mixed with intra-label or intra-domain partners."""

    group_ids: torch.Tensor
    group_count: int
    selection_prob: float
    num_classes: int = 2
    method_id: Literal["lisa"] = field(init=False, default="lisa")
    projection: None = field(init=False, default=None)
    projection_rank: None = field(init=False, default=None)
    _planned: deque[LisaBatchPlan] = field(
        init=False, default_factory=deque, repr=False
    )

    def __post_init__(self) -> None:
        self.group_ids = self.group_ids.detach().cpu().to(torch.int64)

    def epoch_batches(
        self,
        *,
        row_count: int,
        batch_size: int,
        generator: torch.Generator,
    ) -> tuple[torch.Tensor, ...]:
        require_aligned_ids(self.group_ids, row_count, "group")
        plans = plan_lisa_epoch(
            self.group_ids,
            group_count=self.group_count,
            batch_size=batch_size,
            selection_prob=self.selection_prob,
            generator=generator,
        )
        self._planned = deque(plans)
        return tuple(plan.rows for plan in plans)

    def update(
        self,
        algorithm: LinearProbeAlgorithm,
        features: torch.Tensor,
        targets: torch.Tensor,
        rows: torch.Tensor,
    ) -> float:
        if not self._planned:
            raise ValueError("LISA update requested outside a planned epoch")
        plan = self._planned.popleft()
        if not torch.equal(plan.rows, rows):
            raise ValueError("LISA update rows do not match the planned minibatch")
        mixed_features, mixed_targets = mix_features_and_targets(
            features, targets, plan, num_classes=self.num_classes
        )
        return algorithm.update_with_logits_objective(
            mixed_features,
            targets[rows],
            lambda logits, _targets: soft_cross_entropy(logits, mixed_targets),
        )


@dataclass(frozen=True, slots=True)
class SpectralDecouplingLinearProbeMethod(LinearProbeMethodDefaults):
    """ERM minibatches with the raw-logit magnitude penalty, active from update 0."""

    penalty_weight: float
    method_id: Literal["sd"] = field(init=False, default="sd")
    projection: None = field(init=False, default=None)
    projection_rank: None = field(init=False, default=None)

    def epoch_batches(
        self,
        *,
        row_count: int,
        batch_size: int,
        generator: torch.Generator,
    ) -> tuple[torch.Tensor, ...]:
        return random_epoch_batches(row_count, batch_size, generator)

    def update(
        self,
        algorithm: LinearProbeAlgorithm,
        features: torch.Tensor,
        targets: torch.Tensor,
        rows: torch.Tensor,
    ) -> float:
        return algorithm.update_with_logits_objective(
            features[rows],
            targets[rows],
            partial(spectral_decoupling_objective, penalty_weight=self.penalty_weight),
        )


@dataclass(slots=True)
class _WarmedUpPenalty:
    """Shared warm-up bookkeeping: the penalty starts, and Adam restarts, together."""

    penalty_weight: float
    warm_up_updates: int
    _update_count: int = field(init=False, default=0, repr=False)

    def begin_update(self, algorithm: LinearProbeAlgorithm) -> float:
        """Return this update's penalty weight, resetting Adam once on activation."""

        if self._update_count == self.warm_up_updates:
            algorithm.reset_optimizer()
        return 0.0 if self._update_count < self.warm_up_updates else self.penalty_weight

    def end_update(self) -> None:
        self._update_count += 1

    @property
    def update_count(self) -> int:
        return self._update_count


@dataclass(slots=True)
class FishrLinearProbeMethod(LinearProbeMethodDefaults):
    """Environment-balanced epochs matching per-environment gradient variances."""

    environment_ids: torch.Tensor
    environment_count: int
    penalty_weight: float
    penalty_anneal_updates: int
    ema_decay: float
    method_id: Literal["fishr"] = field(init=False, default="fishr")
    projection: None = field(init=False, default=None)
    projection_rank: None = field(init=False, default=None)
    _schedule: _WarmedUpPenalty = field(init=False, repr=False)
    _ema: FishrGradientVarianceEma = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if self.penalty_anneal_updates < 0:
            raise ValueError("Fishr warm-up length must be non-negative")
        self.environment_ids = self.environment_ids.detach().cpu().to(torch.int64)
        self._schedule = _WarmedUpPenalty(
            penalty_weight=self.penalty_weight,
            warm_up_updates=self.penalty_anneal_updates,
        )
        self._ema = FishrGradientVarianceEma(
            environment_count=self.environment_count,
            decay=self.ema_decay,
        )

    @property
    def ema(self) -> FishrGradientVarianceEma:
        """Training-only state; evaluation never touches it."""

        return self._ema

    def epoch_batches(
        self,
        *,
        row_count: int,
        batch_size: int,
        generator: torch.Generator,
    ) -> tuple[torch.Tensor, ...]:
        require_aligned_ids(self.environment_ids, row_count, "environment")
        return environment_balanced_epoch_batches(
            self.environment_ids,
            environment_count=self.environment_count,
            batch_size=batch_size,
            generator=generator,
        )

    def update(
        self,
        algorithm: LinearProbeAlgorithm,
        features: torch.Tensor,
        targets: torch.Tensor,
        rows: torch.Tensor,
    ) -> float:
        penalty_weight = self._schedule.begin_update(algorithm)
        batch_environments = self.environment_ids[rows]
        prepared = algorithm.prepare_features(features[rows])

        def objective(
            logits: torch.Tensor, prepared_targets: torch.Tensor
        ) -> torch.Tensor:
            # The EMA advances during warm-up too, so the penalty it later reports
            # already reflects the whole run rather than restarting at activation.
            penalty = fishr_penalty(
                logits,
                prepared_targets,
                prepared,
                batch_environments,
                environment_count=self.environment_count,
                ema=self._ema,
            )
            cross_entropy = torch.nn.functional.cross_entropy(logits, prepared_targets)
            if penalty_weight == 0.0:
                return cross_entropy
            return cross_entropy + penalty_weight * penalty

        loss = algorithm.update_with_logits_objective(
            features[rows], targets[rows], objective
        )
        self._schedule.end_update()
        return loss


@dataclass(slots=True)
class RdmLinearProbeMethod(LinearProbeMethodDefaults):
    """Environment-balanced epochs matching the worst environment's risk spread."""

    environment_ids: torch.Tensor
    environment_count: int
    penalty_weight: float
    penalty_anneal_updates: int
    variance_weight: float
    method_id: Literal["rdm"] = field(init=False, default="rdm")
    projection: None = field(init=False, default=None)
    projection_rank: None = field(init=False, default=None)
    _schedule: _WarmedUpPenalty = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if self.penalty_anneal_updates < 0:
            raise ValueError("RDM warm-up length must be non-negative")
        if self.variance_weight < 0.0:
            raise ValueError("RDM variance weight must be non-negative")
        self.environment_ids = self.environment_ids.detach().cpu().to(torch.int64)
        self._schedule = _WarmedUpPenalty(
            penalty_weight=self.penalty_weight,
            warm_up_updates=self.penalty_anneal_updates,
        )

    def epoch_batches(
        self,
        *,
        row_count: int,
        batch_size: int,
        generator: torch.Generator,
    ) -> tuple[torch.Tensor, ...]:
        require_aligned_ids(self.environment_ids, row_count, "environment")
        batches = environment_balanced_epoch_batches(
            self.environment_ids,
            environment_count=self.environment_count,
            batch_size=batch_size,
            generator=generator,
        )
        # Reject the run before its first update rather than quietly evaluating the
        # sample variance on a one-example remainder.
        for rows in batches:
            environments = self.environment_ids[rows]
            require_rdm_environment_rows(
                tuple(
                    int((environments == environment).sum())
                    for environment in range(self.environment_count)
                )
            )
        return batches

    def update(
        self,
        algorithm: LinearProbeAlgorithm,
        features: torch.Tensor,
        targets: torch.Tensor,
        rows: torch.Tensor,
    ) -> float:
        penalty_weight = self._schedule.begin_update(algorithm)
        active = penalty_weight != 0.0
        batch_environments = self.environment_ids[rows]
        loss = algorithm.update_with_objective(
            features[rows],
            targets[rows],
            partial(
                rdm_objective,
                environment_ids=batch_environments,
                environment_count=self.environment_count,
                penalty_weight=penalty_weight,
                variance_weight=self.variance_weight if active else 0.0,
            ),
        )
        self._schedule.end_update()
        return loss


def bind_risk_matching_method(
    algorithm: FishrAlgorithmConfig | RdmAlgorithmConfig,
    *,
    environment_ids: torch.Tensor,
    environment_count: int,
    training: LinearProbeTrainingConfig,
) -> FishrLinearProbeMethod | RdmLinearProbeMethod:
    """Bind Fishr or RDM for any dataset that supplies aligned environment IDs.

    Only here are the training row counts, the batch size, and the epoch count all
    known, so this is where a warm-up that would never activate is caught, before the
    first update.
    """

    require_active_penalty_updates(
        warm_up_updates=int(algorithm.penalty_anneal_updates),
        updates_per_epoch=environment_balanced_epoch_batch_count(
            environment_ids,
            environment_count=environment_count,
            batch_size=int(training.batch_size),
        ),
        max_epochs=int(training.max_epochs),
        label=METHOD_LABELS[algorithm.kind],
    )
    if isinstance(algorithm, FishrAlgorithmConfig):
        return FishrLinearProbeMethod(
            environment_ids=environment_ids,
            environment_count=environment_count,
            penalty_weight=float(algorithm.penalty_weight),
            penalty_anneal_updates=int(algorithm.penalty_anneal_updates),
            ema_decay=float(algorithm.ema),
        )
    return RdmLinearProbeMethod(
        environment_ids=environment_ids,
        environment_count=environment_count,
        penalty_weight=float(algorithm.penalty_weight),
        penalty_anneal_updates=int(algorithm.penalty_anneal_updates),
        variance_weight=float(algorithm.variance_weight),
    )
