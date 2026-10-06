"""Pair interventions composed with an unchanged base objective and sampler."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from grit.config import LinearProbeTrainingConfig
from grit.methods.projection import FittedLinearProjection
from grit.methods.training import (
    LinearProbeAlgorithm,
    LinearProbeState,
    LinearProbeTrainingMethod,
    prediction_consistency_penalty,
)
from grit.methods.types import MethodId, pair_intervention

__all__ = [
    "COMPOSABLE_BASES",
    "ComposedLinearProbeMethod",
    "prediction_consistency_penalty",
]

# Base objectives whose losses reach the shared logits callback, so a pair term can be
# added after their own supervised statistics and schedules.
COMPOSABLE_BASES: frozenset[MethodId] = frozenset(
    {"erm", "rex", "irm", "fishr", "groupdro", "sd", "rdm", "lisa"}
)


@dataclass(slots=True)
class ComposedLinearProbeMethod:
    """Reuse base sampling, objective schedules, optimizer resets and checkpoints.

    The predictor supplies its actual classifier inputs to explicit gradient
    statistics. It applies projection or adds the selected pair term outside
    supervised statistics (GroupDRO's group update, RDM's risk distribution, SD's
    logit penalty, LISA's mixed soft labels), including throughout Fishr and RDM
    warm-up. LISA mixes raw features, so GRIT projects the mixture.
    """

    base: LinearProbeTrainingMethod
    method_id: MethodId
    projection: FittedLinearProjection | None = None
    projection_rank: int | None = None
    pair_differences: torch.Tensor | None = None
    consistency_weight: float = 0.0

    def __post_init__(self) -> None:
        if self.base.method_id not in COMPOSABLE_BASES:
            raise ValueError(
                "pair interventions support ERM, V-REx, IRMv1, Fishr, GroupDRO, SD, "
                "RDM and LISA"
            )
        if self.base.projection is not None:
            raise ValueError("base objective must not already have a projection")
        if (self.projection is None) != (self.projection_rank is None):
            raise ValueError(
                "a fitted projection and its rank must be supplied together"
            )
        intervention = pair_intervention(self.method_id)
        expected = (
            "grit"
            if intervention == "grit" and self.base.method_id == "erm"
            else f"{self.base.method_id}_{intervention}"
        )
        if intervention == "vanilla" or self.method_id != expected:
            raise ValueError("composed method identity does not match its intervention")
        if (self.projection is not None) != (intervention == "grit"):
            raise ValueError("only GRIT uses a fitted projection")
        if intervention == "grit" and self.pair_differences is not None:
            raise ValueError("projection and consistency are exclusive interventions")
        if intervention != "grit" and self.pair_differences is None:
            raise ValueError(f"{intervention} requires training-pair differences")
        if self.consistency_weight != 0.0 and intervention != "consistency":
            raise ValueError(f"{self.method_id} does not use consistency strength")

    def epoch_batches(
        self,
        *,
        row_count: int,
        batch_size: int,
        generator: torch.Generator,
    ) -> tuple[torch.Tensor, ...]:
        return self.base.epoch_batches(
            row_count=row_count, batch_size=batch_size, generator=generator
        )

    def update(
        self,
        algorithm: LinearProbeAlgorithm,
        features: torch.Tensor,
        targets: torch.Tensor,
        rows: torch.Tensor,
    ) -> float:
        return self.base.update(algorithm, features, targets, rows)

    def build_algorithm(
        self,
        config: LinearProbeTrainingConfig,
        *,
        model_seed: int,
        num_classes: int,
    ) -> LinearProbeAlgorithm:
        return LinearProbeAlgorithm(
            config,
            model_seed=model_seed,
            projection=self.projection,
            num_classes=num_classes,
            pair_differences=self.pair_differences,
            consistency_weight=self.consistency_weight,
        )

    def checkpoint_state(self, algorithm: LinearProbeAlgorithm) -> LinearProbeState:
        return self.base.checkpoint_state(algorithm)
