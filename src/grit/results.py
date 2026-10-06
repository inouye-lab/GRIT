"""Canonical ordinary run-result boundaries."""

from __future__ import annotations

from statistics import fmean
from typing import Annotated, Literal, TypeAlias

from pydantic import Field, StrictStr, TypeAdapter, model_validator

from grit.config import (
    LinearProjectionConfig,
    OrdinaryExperimentConfig,
    algorithm_method_id,
)
from grit.methods.checkpoints import RestorationReceipt
from grit.methods.types import consumes_pairs
from grit.schemas import StrictBoundaryModel
from grit.selection.cmnist import (
    FinalTestMetricRecord,
    FrozenCandidateSelection,
    FrozenCheckpointSelection,
    ValidationMetricRecord,
)

NonEmptyStr: TypeAlias = Annotated[StrictStr, Field(min_length=1)]


class SucceededStatus(StrictBoundaryModel):
    kind: Literal["succeeded"]


class FailedStatus(StrictBoundaryModel):
    kind: Literal["failed"]
    phase: NonEmptyStr
    error_type: NonEmptyStr
    message: NonEmptyStr


class IncompleteStatus(StrictBoundaryModel):
    kind: Literal["incomplete"]
    last_completed_phase: NonEmptyStr
    reason: NonEmptyStr


RunStatus: TypeAlias = Annotated[
    SucceededStatus | FailedStatus | IncompleteStatus,
    Field(discriminator="kind"),
]


class CodeProvenance(StrictBoundaryModel):
    git_revision: NonEmptyStr
    git_dirty: bool


class EnvironmentProvenance(StrictBoundaryModel):
    python_version: NonEmptyStr
    lock_digest: NonEmptyStr
    device: NonEmptyStr


class ArtifactReference(StrictBoundaryModel):
    """Lightweight metadata only; no storage or tensor format is implied."""

    artifact_id: NonEmptyStr
    kind: NonEmptyStr
    relative_uri: NonEmptyStr
    digest: NonEmptyStr


class OrdinaryRunResult(StrictBoundaryModel):
    schema_version: Literal["grit.run-result/v1"]
    result_kind: Literal["ordinary"]
    run_id: NonEmptyStr
    resolved_config: OrdinaryExperimentConfig
    resolved_config_digest: NonEmptyStr
    status: RunStatus
    code: CodeProvenance
    environment: EnvironmentProvenance
    validation_metrics: tuple[ValidationMetricRecord, ...]
    candidate_selection: FrozenCandidateSelection | None
    checkpoint_selection: FrozenCheckpointSelection | None
    restoration: RestorationReceipt | None
    final_test_metrics: tuple[FinalTestMetricRecord, ...] | None
    artifacts: tuple[ArtifactReference, ...]

    @model_validator(mode="after")
    def _validate_result_state(self) -> OrdinaryRunResult:
        if self.resolved_config_digest != self.resolved_config.canonical_digest():
            raise ValueError(
                "result resolved_config_digest does not match resolved_config"
            )
        succeeded = isinstance(self.status, SucceededStatus)
        required = (
            self.candidate_selection,
            self.checkpoint_selection,
            self.restoration,
            self.final_test_metrics,
        )
        if succeeded and any(value is None for value in required):
            raise ValueError(
                "a successful ordinary result requires selection and final data"
            )
        if not succeeded and self.final_test_metrics is not None:
            raise ValueError(
                "failed or incomplete ordinary results cannot claim final metrics"
            )
        if not succeeded:
            return self

        candidate = self.candidate_selection
        checkpoint = self.checkpoint_selection
        restoration = self.restoration
        metrics = self.final_test_metrics
        if (
            candidate is None
            or checkpoint is None
            or restoration is None
            or metrics is None
        ):
            raise AssertionError("successful result requirements were not narrowed")
        if len(metrics) != 1:
            raise ValueError("a successful ordinary result requires one final metric")
        if not self.validation_metrics:
            raise ValueError("a successful ordinary result requires validation metrics")
        if candidate.method_id != algorithm_method_id(self.resolved_config.algorithm):
            raise ValueError("frozen candidate method does not match result algorithm")
        if candidate.selector is not self.resolved_config.selection.selector:
            raise ValueError("frozen candidate selector does not match result config")
        if candidate.seed_sets != self.resolved_config.seed_sets:
            raise ValueError("frozen candidate seed sets do not match result config")
        expected_rank: int | None = None
        if isinstance(self.resolved_config.projection, LinearProjectionConfig):
            projection = self.resolved_config.projection
            expected_rank = projection.requested_rank
        if candidate.decision.projection_rank != expected_rank:
            raise ValueError("candidate rank does not match result projection config")
        if (
            candidate.scientific_config_digest
            != self.resolved_config.scientific_config_digest()
        ):
            raise ValueError("frozen candidate configuration does not match the result")
        if checkpoint.candidate_selection_id != candidate.frozen_selection_id:
            raise ValueError("checkpoint selection does not belong to result candidate")
        if checkpoint.selector is not candidate.selector:
            raise ValueError("checkpoint selector does not match result candidate")
        if checkpoint.method_id != candidate.method_id:
            raise ValueError("checkpoint method does not match result candidate")
        if checkpoint.decision.seed not in candidate.seed_sets.final:
            raise ValueError("checkpoint seed is not configured for final evaluation")
        if checkpoint.decision.projection_rank != expected_rank:
            raise ValueError("checkpoint rank does not match result projection config")
        checkpoint_identity = checkpoint.checkpoint
        expected_checkpoint_identity = (
            candidate.candidate_id,
            self.run_id,
            candidate.scientific_config_digest,
        )
        observed_checkpoint_identity = (
            checkpoint_identity.candidate_id,
            checkpoint_identity.run_id,
            checkpoint_identity.scientific_config_digest,
        )
        if observed_checkpoint_identity != expected_checkpoint_identity:
            raise ValueError(
                "checkpoint identity does not match result candidate and run"
            )
        if restoration.candidate_selection_id != candidate.frozen_selection_id:
            raise ValueError("restoration does not belong to result candidate")
        if restoration.checkpoint != checkpoint.checkpoint:
            raise ValueError("restoration does not match result checkpoint")
        expected_validation_identity = (
            self.run_id,
            candidate.candidate_id,
            candidate.method_id,
            candidate.scientific_config_digest,
            checkpoint.decision.seed_stage,
            checkpoint.decision.seed,
        )
        for metric in self.validation_metrics:
            observed = (
                metric.run_id,
                metric.candidate_id,
                metric.method_id,
                metric.scientific_config_digest,
                metric.seed_stage,
                metric.seed,
            )
            if observed != expected_validation_identity:
                raise ValueError(
                    "validation metric identity does not match selected result state"
                )
            if metric.projection_rank != expected_rank:
                raise ValueError(
                    "validation metric rank does not match result projection config"
                )
        validation_by_id = {
            metric.record_id: metric for metric in self.validation_metrics
        }
        if len(validation_by_id) != len(self.validation_metrics):
            raise ValueError("validation metric record IDs must be unique")
        contributor_ids = checkpoint.decision.contributing_record_ids
        if any(record_id not in validation_by_id for record_id in contributor_ids):
            raise ValueError(
                "checkpoint selection cites unavailable validation metrics"
            )
        contributors = tuple(
            validation_by_id[record_id] for record_id in contributor_ids
        )
        all_splits = tuple(self.resolved_config.dataset.validation_split_names)
        expected_splits = (
            all_splits
            if checkpoint.selector.value == "primary_robust"
            else all_splits[:2]
        )
        contributor_identity = {
            (metric.checkpoint_id, metric.epoch) for metric in contributors
        }
        if contributor_identity != {
            (checkpoint.checkpoint.checkpoint_id, checkpoint.checkpoint.epoch)
        }:
            raise ValueError("validation contributors do not identify the checkpoint")
        if tuple(metric.split_name for metric in contributors) != expected_splits:
            raise ValueError("validation contributors do not cover selector splits")
        contributor_ranks = {metric.projection_rank for metric in contributors}
        if contributor_ranks != {checkpoint.decision.projection_rank}:
            raise ValueError(
                "validation contributor rank does not match checkpoint decision"
            )
        contributor_values = tuple(float(metric.value) for metric in contributors)
        if min(contributor_values) != float(checkpoint.decision.objective_value):
            raise ValueError(
                "checkpoint objective does not match validation contributors"
            )
        if fmean(contributor_values) != float(checkpoint.decision.mean_accuracy):
            raise ValueError("checkpoint mean does not match validation contributors")
        for metric in metrics:
            if metric.split_name != "test_ood":
                raise ValueError("final metric uses the wrong protocol split")
            identity = (
                metric.run_id,
                metric.candidate_id,
                metric.method_id,
                metric.scientific_config_digest,
                metric.checkpoint_id,
                metric.epoch,
                metric.seed,
            )
            expected = (
                self.run_id,
                candidate.candidate_id,
                candidate.method_id,
                candidate.scientific_config_digest,
                checkpoint.checkpoint.checkpoint_id,
                checkpoint.checkpoint.epoch,
                checkpoint.decision.seed,
            )
            if identity != expected:
                raise ValueError(
                    "final metric identity does not match selected result state"
                )
        if self.resolved_config.reportable:
            self._validate_reportable_artifacts()
        return self

    def _validate_reportable_artifacts(self) -> None:
        lineage = self.resolved_config.artifact_lineage
        if lineage is None:
            raise ValueError("reportable result lacks artifact lineage")
        by_kind = {artifact.kind: artifact for artifact in self.artifacts}
        if len(by_kind) != len(self.artifacts):
            raise ValueError("CMNIST result artifact kinds must be unique")
        if len({artifact.artifact_id for artifact in self.artifacts}) != len(
            self.artifacts
        ) or len({artifact.relative_uri for artifact in self.artifacts}) != len(
            self.artifacts
        ):
            raise ValueError("CMNIST result artifact references must be unique")
        required = {
            "dataset_manifest",
            "feature_manifest",
            "selected_linear_checkpoint",
        }
        if consumes_pairs(algorithm_method_id(self.resolved_config.algorithm)):
            required.add("pair_manifest")
        if isinstance(self.resolved_config.projection, LinearProjectionConfig):
            required.add("projection_diagnostics")
        if set(by_kind) != required:
            raise ValueError("CMNIST result required artifact references are missing")
        if (
            by_kind["dataset_manifest"].digest != lineage.dataset_manifest_digest
            or by_kind["feature_manifest"].digest
            != lineage.feature_cache_manifest_digest
        ):
            raise ValueError("CMNIST result artifact lineage is inconsistent")
        if "pair_manifest" in required:
            if by_kind["pair_manifest"].digest != lineage.pair_manifest_digest:
                raise ValueError("CMNIST result pair lineage is inconsistent")


RunResult: TypeAlias = OrdinaryRunResult

_RUN_RESULT_ADAPTER: TypeAdapter[RunResult] = TypeAdapter(RunResult)


def parse_run_result_json(payload: str) -> RunResult:
    """Parse canonical JSON through the strict discriminated result root."""

    return _RUN_RESULT_ADAPTER.validate_json(payload)
