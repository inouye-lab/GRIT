"""Pooled OOD-validation accuracy selection, with no test-metric input field."""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import Field, model_validator

from grit.schemas import StrictBoundaryModel
from grit.selection.cmnist import CheckpointIdentity


class AccuracyObservation(StrictBoundaryModel):
    split: Literal["val"] = "val"
    epoch: Annotated[int, Field(gt=0)]
    correct: Annotated[int, Field(ge=0)]
    count: Annotated[int, Field(gt=0)]
    train_objective: float

    @model_validator(mode="after")
    def _counts(self) -> AccuracyObservation:
        if self.correct > self.count:
            raise ValueError("correct count exceeds validation size")
        return self

    @property
    def accuracy(self) -> float:
        return self.correct / self.count


def select_accuracy_checkpoint(
    observations: tuple[AccuracyObservation, ...],
) -> AccuracyObservation:
    if not observations or any(
        type(v) is not AccuracyObservation for v in observations
    ):
        raise ValueError("selection requires OOD validation observations only")
    if len({v.epoch for v in observations}) != len(observations):
        raise ValueError("duplicate validation epoch")
    if len({v.count for v in observations}) != 1:
        raise ValueError("validation population changed between checkpoints")
    return min(observations, key=lambda v: (-v.accuracy, v.epoch))


class FrozenAccuracyCheckpoint(StrictBoundaryModel):
    """Written before final feature access, binding selection to cache and weights."""

    stage: Literal["final"]
    seed: int
    plan_digest: str
    feature_manifest_digest: str
    winner_digest: str
    checkpoint: CheckpointIdentity
    checkpoint_manifest_digest: str
    validation: AccuracyObservation

    @model_validator(mode="after")
    def _epoch(self) -> FrozenAccuracyCheckpoint:
        if self.validation.epoch != self.checkpoint.epoch:
            raise ValueError("frozen checkpoint does not match validation selection")
        return self
