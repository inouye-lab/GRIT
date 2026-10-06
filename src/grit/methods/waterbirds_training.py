"""Minimal frozen-feature trainer and restoration path for Waterbirds ERM/GRIT."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TypeAlias

from pydantic import Field, StrictStr

from grit.config import LinearProbeTrainingConfig
from grit.data.waterbirds import WaterbirdsAdjustedWeightSpec
from grit.features.waterbirds import (
    WaterbirdsEvaluationFeatureTable,
    WaterbirdsTrainingFeatureTable,
)
from grit.methods.checkpoints import CheckpointStore
from grit.methods.training import (
    InMemoryLinearCheckpointStore,
    LinearProbeAlgorithm,
    LinearProbeState,
    LinearProbeTrainingMethod,
    train_linear_probe_epochs,
)
from grit.methods.types import MethodId
from grit.schemas import SeedStage, StrictBoundaryModel
from grit.selection.cmnist import CheckpointIdentity
from grit.selection.waterbirds import (
    FrozenWaterbirdsCheckpoint,
    WaterbirdsMetricRecord,
    WaterbirdsSelector,
    WaterbirdsSelectorRecord,
    WaterbirdsValidationMetricRecord,
    compute_waterbirds_validation_metric,
)
from grit.tracking import RunTracker

WaterbirdsMethod: TypeAlias = MethodId


class WaterbirdsRestorationReceipt(StrictBoundaryModel):
    receipt_id: StrictStr = Field(min_length=1)
    candidate_selection_id: StrictStr = Field(min_length=1)
    store_id: StrictStr = Field(min_length=1)
    checkpoint: CheckpointIdentity


@dataclass(frozen=True, slots=True)
class TrainedWaterbirdsRun:
    run_id: str
    candidate_id: str
    method_id: WaterbirdsMethod
    seed_stage: SeedStage
    seed: int
    projection_rank: int | None
    validation_metrics: tuple[WaterbirdsValidationMetricRecord, ...]
    store: InMemoryLinearCheckpointStore
    algorithm: LinearProbeAlgorithm
    epoch_losses: tuple[float, ...]

    def selector_records(
        self, selector: WaterbirdsSelector
    ) -> tuple[WaterbirdsSelectorRecord, ...]:
        del selector
        return self.validation_metrics


def train_waterbirds_linear_probe(
    training: WaterbirdsTrainingFeatureTable,
    validation: WaterbirdsEvaluationFeatureTable,
    adjusted_weights: WaterbirdsAdjustedWeightSpec,
    config: LinearProbeTrainingConfig,
    *,
    run_id: str,
    candidate_id: str,
    scientific_config_digest: str,
    seed_stage: SeedStage,
    seed: int,
    method: LinearProbeTrainingMethod,
    tracker: RunTracker | None = None,
) -> TrainedWaterbirdsRun:
    """Run trainer-owned epochs and emit one four-group validation record each.

    `tracker` only mirrors measurements that were computed anyway.
    """

    if training.dataset_manifest_digest != validation.dataset_manifest_digest:
        raise ValueError("Waterbirds train and validation datasets do not match")
    if (
        training.feature_cache_manifest_digest
        != validation.feature_cache_manifest_digest
    ):
        raise ValueError("Waterbirds train and validation feature caches do not match")
    if training.normalization != validation.normalization:
        raise ValueError("Waterbirds train and validation normalizations do not match")
    if int(training.features.shape[0]) == 0:
        raise ValueError("Waterbirds training data must not be empty")
    metrics: list[WaterbirdsValidationMetricRecord] = []

    def validate_epoch(
        algorithm: LinearProbeAlgorithm,
        identity: CheckpointIdentity,
        train_objective: float,
    ) -> None:
        predictions = algorithm.predict(validation.features)
        record = compute_waterbirds_validation_metric(
            validation,
            predictions,
            adjusted_weights=adjusted_weights,
            record_id=f"metric:{identity.checkpoint_id}:validation",
            run_id=run_id,
            candidate_id=candidate_id,
            method_id=method.method_id,
            scientific_config_digest=scientific_config_digest,
            checkpoint_id=identity.checkpoint_id,
            epoch=identity.epoch,
            seed_stage=seed_stage,
            seed=seed,
            projection_rank=method.projection_rank,
        )
        metrics.append(record)
        if tracker is None:
            return
        # `train_objective` carries each method's own penalties; it is not a
        # cross-method cross-entropy.
        values = {"train_objective": train_objective}
        values.update(algorithm.training_diagnostics())
        values.update(
            {
                f"validation/{name}": value
                for name, value in waterbirds_metric_values(record).items()
            }
        )
        tracker.log_epoch(identity.epoch, values)

    core = train_linear_probe_epochs(
        training.features,
        training.labels,
        config,
        run_id=run_id,
        candidate_id=candidate_id,
        scientific_config_digest=scientific_config_digest,
        seed=seed,
        method=method,
        validate_epoch=validate_epoch,
    )
    return TrainedWaterbirdsRun(
        run_id=run_id,
        candidate_id=candidate_id,
        method_id=method.method_id,
        seed_stage=seed_stage,
        seed=seed,
        projection_rank=method.projection_rank,
        validation_metrics=tuple(metrics),
        store=core.store,
        algorithm=core.algorithm,
        epoch_losses=core.epoch_losses,
    )


def waterbirds_metric_values(record: WaterbirdsMetricRecord) -> dict[str, float]:
    """The four-group shape every Waterbirds mirror reports."""

    values = {
        "worst_group_accuracy": float(record.worst_group_accuracy),
        "adjusted_average_accuracy": float(record.adjusted_average_accuracy),
        "raw_average_accuracy": float(record.raw_average_accuracy),
    }
    values.update(
        {
            f"group_{group.group_id}_accuracy": float(group.accuracy)
            for group in record.groups
        }
    )
    return values


def restore_waterbirds_checkpoint(
    selection: FrozenWaterbirdsCheckpoint,
    store: CheckpointStore[LinearProbeState],
    algorithm: LinearProbeAlgorithm,
) -> WaterbirdsRestorationReceipt:
    """Restore exactly the selected inference state and issue a typed receipt."""

    frozen = FrozenWaterbirdsCheckpoint.model_validate_json(selection.canonical_json())
    stored = store.load(frozen.checkpoint.checkpoint_id)
    if stored.identity != frozen.checkpoint:
        raise ValueError("Waterbirds checkpoint store returned the wrong identity")
    algorithm.restore_inference_state(stored.state)
    return WaterbirdsRestorationReceipt(
        receipt_id=f"restored:{store.store_id}:{stored.identity.checkpoint_id}",
        candidate_selection_id=frozen.candidate_selection_id,
        store_id=store.store_id,
        checkpoint=stored.identity,
    )
