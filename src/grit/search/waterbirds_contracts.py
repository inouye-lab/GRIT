"""Strict resolved-config and canonical result boundaries for Waterbirds smoke runs."""

from __future__ import annotations

import math
from statistics import fmean, stdev
from typing import Annotated, Literal, TypeAlias

from pydantic import (
    Field,
    FiniteFloat,
    StrictBool,
    StrictFloat,
    StrictInt,
    StrictStr,
    model_validator,
)

from grit.config import (
    AlgorithmConfig,
    ComposedAlgorithmConfig,
    FishrAlgorithmConfig,
    GroupDroAlgorithmConfig,
    IrmAlgorithmConfig,
    LinearProbeTrainingConfig,
    LisaAlgorithmConfig,
    RdmAlgorithmConfig,
    RexAlgorithmConfig,
    SeedSets,
    algorithm_method_id,
)
from grit.methods.types import (
    METHOD_LABELS,
    MethodId,
    consumes_pairs,
    pair_intervention,
)
from grit.methods.waterbirds_training import WaterbirdsRestorationReceipt
from grit.results import CodeProvenance, EnvironmentProvenance
from grit.schemas import StrictBoundaryModel, canonical_digest_value
from grit.selection.cmnist import CheckpointIdentity
from grit.selection.waterbirds import (
    FrozenWaterbirdsCandidate,
    FrozenWaterbirdsCheckpoint,
    WaterbirdsFinalTestMetricRecord,
    WaterbirdsSelector,
    WaterbirdsSelectorRecord,
    WaterbirdsValidationMetricRecord,
)

NonEmptyStr: TypeAlias = Annotated[StrictStr, Field(min_length=1)]


class WaterbirdsCandidateConfig(StrictBoundaryModel):
    schema_version: Literal["grit.waterbirds-candidate/v3"]
    protocol_id: Literal["waterbirds_cf/v1"]
    non_reportable: StrictBool
    method_id: MethodId
    selector: WaterbirdsSelector
    dataset_profile: Literal["fixture", "production"]
    dataset_manifest_digest: NonEmptyStr
    feature_cache_manifest_digest: NonEmptyStr
    normalization: Literal["none", "l2"]
    adjusted_weight_spec_digest: NonEmptyStr
    pair_manifest_digest: NonEmptyStr | None
    projection_diagnostics_digest: NonEmptyStr | None
    projection_rank: Annotated[StrictInt, Field(ge=0, le=24)] | None
    relative_singular_value_tolerance: Annotated[StrictFloat, Field(gt=0.0)] | None
    algorithm: AlgorithmConfig
    training: LinearProbeTrainingConfig
    seed_sets: SeedSets

    @model_validator(mode="after")
    def _validate_method(self) -> WaterbirdsCandidateConfig:
        if (self.dataset_profile == "fixture") != self.non_reportable:
            raise ValueError(
                "Waterbirds fixture configs must be non-reportable and production "
                "configs must be reportable"
            )
        if algorithm_method_id(self.algorithm) != self.method_id:
            raise ValueError("Waterbirds candidate algorithm does not match method")
        uses_pairs = consumes_pairs(algorithm_method_id(self.algorithm))
        if uses_pairs != (self.pair_manifest_digest is not None):
            raise ValueError(
                f"Waterbirds {self.method_id} must bind oracle pairs exactly when "
                "its algorithm consumes them"
            )
        projection_fields = (
            self.projection_diagnostics_digest,
            self.projection_rank,
            self.relative_singular_value_tolerance,
        )
        if pair_intervention(self.method_id) == "grit" and any(
            v is None for v in projection_fields
        ):
            raise ValueError("Waterbirds GRIT config requires oracle projection")
        if pair_intervention(self.method_id) != "grit" and any(
            v is not None for v in projection_fields
        ):
            raise ValueError(
                f"Waterbirds {self.method_id} config cannot contain a projection"
            )
        algorithm = (
            self.algorithm.base_objective
            if isinstance(self.algorithm, ComposedAlgorithmConfig)
            else self.algorithm
        )
        # Background is the environment, so LISA's two definitions coincide here.
        if isinstance(
            algorithm, GroupDroAlgorithmConfig
        ) and algorithm.group_definition not in (
            "target_background",
            "training_environment",
        ):
            raise ValueError(
                "Waterbirds GroupDRO uses label/background groups or backgrounds"
            )
        if (
            isinstance(algorithm, LisaAlgorithmConfig)
            and algorithm.group_definition != "target_background"
        ):
            raise ValueError("Waterbirds LISA uses label/background groups")
        if isinstance(
            algorithm,
            RexAlgorithmConfig
            | IrmAlgorithmConfig
            | FishrAlgorithmConfig
            | RdmAlgorithmConfig,
        ) and algorithm.environment_names != ("background_land", "background_water"):
            raise ValueError("Waterbirds invariant methods use background environments")
        return self

    def scientific_config_digest(self) -> str:
        """Identify the trainable candidate independently of selector and seeds."""

        return canonical_digest_value(
            self.model_dump(
                mode="json",
                exclude={"projection_diagnostics_digest", "seed_sets", "selector"},
            )
        )


class _WaterbirdsArtifactReference(StrictBoundaryModel):
    artifact_id: NonEmptyStr
    relative_uri: NonEmptyStr
    digest: NonEmptyStr


class WaterbirdsDatasetArtifactReference(_WaterbirdsArtifactReference):
    kind: Literal["dataset_manifest"]


class WaterbirdsFeatureArtifactReference(_WaterbirdsArtifactReference):
    kind: Literal["feature_manifest"]
    dataset_manifest_digest: NonEmptyStr
    normalization: Literal["none", "l2"]


class WaterbirdsCheckpointArtifactReference(_WaterbirdsArtifactReference):
    kind: Literal["selected_linear_checkpoint"]
    checkpoint: CheckpointIdentity


class WaterbirdsPairArtifactReference(_WaterbirdsArtifactReference):
    kind: Literal["pair_manifest"]
    dataset_manifest_digest: NonEmptyStr


class WaterbirdsProjectionArtifactReference(_WaterbirdsArtifactReference):
    kind: Literal["projection_diagnostics"]
    pair_manifest_digest: NonEmptyStr
    feature_cache_manifest_digest: NonEmptyStr
    normalization: Literal["none", "l2"]
    requested_rank: Annotated[StrictInt, Field(ge=0, le=24)]


WaterbirdsArtifactReference = Annotated[
    WaterbirdsDatasetArtifactReference
    | WaterbirdsFeatureArtifactReference
    | WaterbirdsCheckpointArtifactReference
    | WaterbirdsPairArtifactReference
    | WaterbirdsProjectionArtifactReference,
    Field(discriminator="kind"),
]


WaterbirdsTestMetric: TypeAlias = WaterbirdsFinalTestMetricRecord


def _metric_identity(
    metric: WaterbirdsSelectorRecord | WaterbirdsTestMetric,
) -> tuple[object, ...]:
    return (
        metric.run_id,
        metric.candidate_id,
        metric.method_id,
        metric.scientific_config_digest,
        metric.seed_stage,
        metric.seed,
        metric.projection_rank,
        metric.dataset_manifest_digest,
        metric.feature_cache_manifest_digest,
        metric.normalization,
        metric.adjusted_weight_spec_digest,
    )


class _WaterbirdsRunResultBase(StrictBoundaryModel):
    """Fields and lifecycle checks of a Waterbirds final result."""

    schema_version: Literal["grit.waterbirds-run-result/v3"]
    status: Literal["succeeded"]
    run_id: NonEmptyStr
    final_seed: StrictInt
    resolved_config: WaterbirdsCandidateConfig
    resolved_config_digest: NonEmptyStr
    code: CodeProvenance
    environment: EnvironmentProvenance
    validation_metrics: tuple[WaterbirdsValidationMetricRecord, ...]
    restoration: WaterbirdsRestorationReceipt
    selected_checkpoint_manifest_digest: NonEmptyStr
    artifacts: tuple[WaterbirdsArtifactReference, ...]

    @property
    def selected_candidate(self) -> FrozenWaterbirdsCandidate:
        raise NotImplementedError

    @property
    def selected_checkpoint(self) -> FrozenWaterbirdsCheckpoint:
        raise NotImplementedError

    @property
    def reported_test_metric(self) -> WaterbirdsTestMetric:
        """The four-group test record this result reports for its final seed."""

        raise NotImplementedError

    def _validate_lifecycle(
        self,
        candidate: FrozenWaterbirdsCandidate,
        checkpoint: FrozenWaterbirdsCheckpoint,
        decision_records: tuple[WaterbirdsSelectorRecord, ...],
        reported: WaterbirdsTestMetric,
        selector: WaterbirdsSelector,
    ) -> None:
        config = self.resolved_config
        digest = config.scientific_config_digest()
        if self.resolved_config_digest != config.canonical_digest():
            raise ValueError("Waterbirds result config digest is inconsistent")
        if config.selector != selector or candidate.selector != selector:
            raise ValueError(
                f"Waterbirds {self.__class__.__name__} requires the {selector} selector"
            )
        if (
            candidate.method_id != config.method_id
            or candidate.scientific_config_digest != digest
            or candidate.projection_rank != config.projection_rank
            or candidate.seed_sets != config.seed_sets
            or candidate.dataset_manifest_digest != config.dataset_manifest_digest
            or candidate.feature_cache_manifest_digest
            != config.feature_cache_manifest_digest
            or candidate.normalization != config.normalization
            or candidate.adjusted_weight_spec_digest
            != config.adjusted_weight_spec_digest
        ):
            raise ValueError("Waterbirds frozen candidate does not match result config")
        if (
            checkpoint.candidate_selection_id != candidate.frozen_selection_id
            or checkpoint.method_id != candidate.method_id
            or checkpoint.decision.selector != selector
            or checkpoint.checkpoint.run_id != self.run_id
            or checkpoint.checkpoint.candidate_id != candidate.candidate_id
            or checkpoint.checkpoint.scientific_config_digest != digest
            or checkpoint.decision.seed not in config.seed_sets.final
            or checkpoint.decision.seed != self.final_seed
            or checkpoint.decision.projection_rank != config.projection_rank
            or checkpoint.dataset_manifest_digest != config.dataset_manifest_digest
            or checkpoint.feature_cache_manifest_digest
            != config.feature_cache_manifest_digest
            or checkpoint.normalization != config.normalization
            or checkpoint.adjusted_weight_spec_digest
            != config.adjusted_weight_spec_digest
        ):
            raise ValueError("Waterbirds checkpoint does not match result lifecycle")
        if (
            self.restoration.candidate_selection_id != candidate.frozen_selection_id
            or self.restoration.checkpoint != checkpoint.checkpoint
        ):
            raise ValueError("Waterbirds restoration does not match result checkpoint")
        if not self.validation_metrics:
            raise ValueError("Waterbirds result requires final-run validation history")
        expected = (
            self.run_id,
            candidate.candidate_id,
            config.method_id,
            digest,
            checkpoint.decision.seed_stage,
            checkpoint.decision.seed,
            config.projection_rank,
            config.dataset_manifest_digest,
            config.feature_cache_manifest_digest,
            config.normalization,
            config.adjusted_weight_spec_digest,
        )
        for history in (self.validation_metrics, decision_records):
            ids = {item.record_id for item in history}
            if len(ids) != len(history):
                raise ValueError("Waterbirds metric record IDs must be unique")
            if any(_metric_identity(metric) != expected for metric in history):
                raise ValueError("Waterbirds metric history identity is inconsistent")
        by_id = {item.record_id: item for item in decision_records}
        contributor = by_id.get(checkpoint.decision.contributing_record_id)
        if contributor is None:
            raise ValueError("Waterbirds checkpoint cites an unavailable metric")
        if (
            contributor.checkpoint_id != checkpoint.checkpoint.checkpoint_id
            or contributor.epoch != checkpoint.checkpoint.epoch
            or contributor.worst_group_accuracy
            != checkpoint.decision.worst_group_accuracy
            or contributor.adjusted_average_accuracy
            != checkpoint.decision.adjusted_average_accuracy
        ):
            raise ValueError("Waterbirds checkpoint decision trace is inconsistent")
        if _metric_identity(reported) != expected or (
            reported.checkpoint_id,
            reported.epoch,
        ) != (checkpoint.checkpoint.checkpoint_id, checkpoint.checkpoint.epoch):
            raise ValueError("Waterbirds final metric identity is inconsistent")
        self._validate_artifacts(checkpoint)

    def _validate_artifacts(
        self, checkpoint_selection: FrozenWaterbirdsCheckpoint
    ) -> None:
        config = self.resolved_config
        artifacts_by_kind = {artifact.kind: artifact for artifact in self.artifacts}
        if len(artifacts_by_kind) != len(self.artifacts):
            raise ValueError("Waterbirds result artifact kinds must be unique")
        if len({artifact.artifact_id for artifact in self.artifacts}) != len(
            self.artifacts
        ) or len({artifact.relative_uri for artifact in self.artifacts}) != len(
            self.artifacts
        ):
            raise ValueError("Waterbirds result artifact references must be unique")
        required = {
            "dataset_manifest",
            "feature_manifest",
            "selected_linear_checkpoint",
        }
        if config.pair_manifest_digest is not None:
            required.add("pair_manifest")
        if pair_intervention(config.method_id) == "grit":
            required.add("projection_diagnostics")
        if set(artifacts_by_kind) != required:
            raise ValueError(
                "Waterbirds result required artifact references are missing"
            )
        dataset = artifacts_by_kind["dataset_manifest"]
        feature = artifacts_by_kind["feature_manifest"]
        checkpoint = artifacts_by_kind["selected_linear_checkpoint"]
        if (
            dataset.digest != config.dataset_manifest_digest
            or not isinstance(feature, WaterbirdsFeatureArtifactReference)
            or feature.digest != config.feature_cache_manifest_digest
            or feature.dataset_manifest_digest != config.dataset_manifest_digest
            or feature.normalization != config.normalization
            or not isinstance(checkpoint, WaterbirdsCheckpointArtifactReference)
            or checkpoint.digest != self.selected_checkpoint_manifest_digest
            or checkpoint.checkpoint != checkpoint_selection.checkpoint
        ):
            raise ValueError("Waterbirds result artifact lineage is inconsistent")
        if config.pair_manifest_digest is not None:
            pair = artifacts_by_kind["pair_manifest"]
            if (
                not isinstance(pair, WaterbirdsPairArtifactReference)
                or pair.digest != config.pair_manifest_digest
                or pair.dataset_manifest_digest != config.dataset_manifest_digest
            ):
                raise ValueError(
                    f"Waterbirds {METHOD_LABELS[config.method_id]} artifact lineage "
                    "is inconsistent"
                )
        if pair_intervention(config.method_id) == "grit":
            projection = artifacts_by_kind["projection_diagnostics"]
            if (
                not isinstance(projection, WaterbirdsProjectionArtifactReference)
                or projection.digest != config.projection_diagnostics_digest
                or projection.pair_manifest_digest != config.pair_manifest_digest
                or projection.feature_cache_manifest_digest
                != config.feature_cache_manifest_digest
                or projection.normalization != config.normalization
                or projection.requested_rank != config.projection_rank
            ):
                raise ValueError("Waterbirds GRIT artifact lineage is inconsistent")


class WaterbirdsRunResult(_WaterbirdsRunResultBase):
    """One validation-selected final seed with its gate-authorized test metric."""

    result_kind: Literal["ordinary_waterbirds"]
    candidate_selection: FrozenWaterbirdsCandidate
    checkpoint_selection: FrozenWaterbirdsCheckpoint
    final_test_metric: WaterbirdsFinalTestMetricRecord

    @property
    def selected_candidate(self) -> FrozenWaterbirdsCandidate:
        return self.candidate_selection

    @property
    def selected_checkpoint(self) -> FrozenWaterbirdsCheckpoint:
        return self.checkpoint_selection

    @property
    def reported_test_metric(self) -> WaterbirdsFinalTestMetricRecord:
        return self.final_test_metric

    @model_validator(mode="after")
    def _validate_result(self) -> WaterbirdsRunResult:
        self._validate_lifecycle(
            self.candidate_selection,
            self.checkpoint_selection,
            self.validation_metrics,
            self.final_test_metric,
            "waterbirds_validation_worst_group",
        )
        return self


MetricName: TypeAlias = Literal[
    "worst_group_accuracy",
    "adjusted_average_accuracy",
    "raw_average_accuracy",
    "grit_minus_erm_worst_group_accuracy",
]


class WaterbirdsMetricSummary(StrictBoundaryModel):
    metric_name: MetricName
    seed_count: Literal[10]
    mean: FiniteFloat
    sample_standard_deviation: Annotated[FiniteFloat, Field(ge=0.0)]
    ci95_lower: FiniteFloat
    ci95_upper: FiniteFloat


def make_waterbirds_metric_summary(
    metric_name: MetricName,
    values: tuple[float, ...],
) -> WaterbirdsMetricSummary:
    if len(values) != 10:
        raise ValueError("Waterbirds final metric summary requires ten seeds")
    mean = fmean(values)
    standard_deviation = stdev(values)
    # Two-sided 95% Student-t critical value with nine degrees of freedom.
    half_width = 2.2621571627409915 * standard_deviation / math.sqrt(10)
    return WaterbirdsMetricSummary(
        metric_name=metric_name,
        seed_count=10,
        mean=mean,
        sample_standard_deviation=standard_deviation,
        ci95_lower=mean - half_width,
        ci95_upper=mean + half_width,
    )


class WaterbirdsFinalSeedObservation(StrictBoundaryModel):
    seed: StrictInt
    method_id: MethodId
    result_path: NonEmptyStr
    metric_record_id: NonEmptyStr
    worst_group_accuracy: FiniteFloat
    adjusted_average_accuracy: FiniteFloat
    raw_average_accuracy: FiniteFloat


class WaterbirdsPairedSeedDifference(StrictBoundaryModel):
    seed: StrictInt
    grit_minus_erm_worst_group_accuracy: FiniteFloat


class WaterbirdsMethodSmokeSummary(StrictBoundaryModel):
    method_id: MethodId
    dataset_manifest_digest: NonEmptyStr
    feature_cache_manifest_digest: NonEmptyStr
    normalization: Literal["none", "l2"]
    adjusted_weight_spec_digest: NonEmptyStr
    selected_candidate_id: NonEmptyStr
    finalist_candidate_ids: tuple[NonEmptyStr, NonEmptyStr, NonEmptyStr]
    configured_final_seeds: Annotated[
        tuple[StrictInt, ...], Field(min_length=10, max_length=10)
    ]
    final_observations: Annotated[
        tuple[WaterbirdsFinalSeedObservation, ...], Field(min_length=10, max_length=10)
    ]
    worst_group_summary: WaterbirdsMetricSummary
    adjusted_average_summary: WaterbirdsMetricSummary
    raw_average_summary: WaterbirdsMetricSummary

    @model_validator(mode="after")
    def _validate_summary(self) -> WaterbirdsMethodSmokeSummary:
        if len(set(self.configured_final_seeds)) != 10:
            raise ValueError("Waterbirds configured final seeds must be unique")
        observed_seeds = tuple(item.seed for item in self.final_observations)
        if observed_seeds != self.configured_final_seeds:
            raise ValueError(
                "Waterbirds final observations require configured seeds exactly once"
            )
        if any(item.method_id != self.method_id for item in self.final_observations):
            raise ValueError("Waterbirds final observation method is inconsistent")
        if (
            len({item.result_path for item in self.final_observations}) != 10
            or len({item.metric_record_id for item in self.final_observations}) != 10
        ):
            raise ValueError(
                "Waterbirds final observations must be uniquely attributable"
            )
        series: tuple[
            tuple[MetricName, tuple[float, ...], WaterbirdsMetricSummary], ...
        ] = (
            (
                "worst_group_accuracy",
                self.final_worst_group_accuracies,
                self.worst_group_summary,
            ),
            (
                "adjusted_average_accuracy",
                self.final_adjusted_average_accuracies,
                self.adjusted_average_summary,
            ),
            (
                "raw_average_accuracy",
                self.final_raw_average_accuracies,
                self.raw_average_summary,
            ),
        )
        for metric_name, values, summary in series:
            if len(values) != 10:
                raise ValueError("Waterbirds smoke metric series requires ten seeds")
            if summary != make_waterbirds_metric_summary(metric_name, values):
                raise ValueError("Waterbirds smoke metric summary is inconsistent")
        return self

    @property
    def result_paths(self) -> tuple[str, ...]:
        return tuple(item.result_path for item in self.final_observations)

    @property
    def final_worst_group_accuracies(self) -> tuple[float, ...]:
        return tuple(
            float(item.worst_group_accuracy) for item in self.final_observations
        )

    @property
    def final_adjusted_average_accuracies(self) -> tuple[float, ...]:
        return tuple(
            float(item.adjusted_average_accuracy) for item in self.final_observations
        )

    @property
    def final_raw_average_accuracies(self) -> tuple[float, ...]:
        return tuple(
            float(item.raw_average_accuracy) for item in self.final_observations
        )


class WaterbirdsSmokeSummary(StrictBoundaryModel):
    schema_version: Literal["grit.waterbirds-smoke-result/v2"]
    non_reportable: Literal[True]
    dataset_manifest_digest: NonEmptyStr
    pair_manifest_digest: NonEmptyStr
    feature_cache_manifest_digest: NonEmptyStr
    normalization: Literal["none", "l2"]
    adjusted_weight_spec_digest: NonEmptyStr
    methods: tuple[WaterbirdsMethodSmokeSummary, WaterbirdsMethodSmokeSummary]
    paired_worst_group_differences: tuple[WaterbirdsPairedSeedDifference, ...]
    paired_worst_group_summary: WaterbirdsMetricSummary

    @model_validator(mode="after")
    def _validate_paired_summary(self) -> WaterbirdsSmokeSummary:
        if tuple(item.method_id for item in self.methods) != ("erm", "grit"):
            raise ValueError("Waterbirds smoke methods must be ordered ERM then GRIT")
        for method in self.methods:
            if (
                method.dataset_manifest_digest != self.dataset_manifest_digest
                or method.feature_cache_manifest_digest
                != self.feature_cache_manifest_digest
                or method.normalization != self.normalization
                or method.adjusted_weight_spec_digest
                != self.adjusted_weight_spec_digest
            ):
                raise ValueError("Waterbirds smoke method lineage is inconsistent")
        if self.methods[0].configured_final_seeds != (
            self.methods[1].configured_final_seeds
        ):
            raise ValueError("Waterbirds paired methods require identical final seeds")
        erm_by_seed = {
            item.seed: float(item.worst_group_accuracy)
            for item in self.methods[0].final_observations
        }
        grit_by_seed = {
            item.seed: float(item.worst_group_accuracy)
            for item in self.methods[1].final_observations
        }
        expected = tuple(
            WaterbirdsPairedSeedDifference(
                seed=seed,
                grit_minus_erm_worst_group_accuracy=(
                    grit_by_seed[seed] - erm_by_seed[seed]
                ),
            )
            for seed in self.methods[0].configured_final_seeds
        )
        if self.paired_worst_group_differences != expected:
            raise ValueError("Waterbirds paired differences are inconsistent")
        if self.paired_worst_group_summary != make_waterbirds_metric_summary(
            "grit_minus_erm_worst_group_accuracy",
            tuple(float(item.grit_minus_erm_worst_group_accuracy) for item in expected),
        ):
            raise ValueError("Waterbirds paired summary is inconsistent")
        return self
