"""Straightforward final-test gate for the in-memory lifecycle."""

from __future__ import annotations

from typing import Literal, cast

from pydantic import FiniteFloat, PositiveInt

from grit.data.views import FinalTestHandle, FinalTestView
from grit.methods.checkpoints import RestorationReceipt
from grit.schemas import SeedStage
from grit.selection.cmnist import (
    FinalTestMetricRecord,
    FrozenCandidateSelection,
    FrozenCheckpointSelection,
)


def open_final_test(
    handle: FinalTestHandle,
    candidate: FrozenCandidateSelection,
    checkpoint: FrozenCheckpointSelection,
    restoration: RestorationReceipt,
) -> FinalTestView:
    """Open final-test records only after matching selection and restoration."""

    return handle.open(candidate, checkpoint, restoration)


def record_final_accuracy(
    view: FinalTestView,
    *,
    record_id: str,
    value: FiniteFloat,
    sample_count: PositiveInt,
) -> FinalTestMetricRecord:
    """Construct a final metric only from a gate-authorized final-test view."""

    return FinalTestMetricRecord(
        record_id=record_id,
        run_id=view.run_id,
        candidate_id=view.candidate_id,
        method_id=view.method_id,
        scientific_config_digest=view.scientific_config_digest,
        checkpoint_id=view.checkpoint_id,
        epoch=view.epoch,
        seed=view.seed,
        value=value,
        sample_count=sample_count,
        metric_kind="final_test",
        seed_stage=SeedStage.FINAL,
        split_name=cast(Literal["test_ood"], view.descriptor.name),
        metric_name="accuracy",
    )
