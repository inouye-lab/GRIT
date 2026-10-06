"""Four-group metrics and validation-only selection for Waterbirds-CF."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from statistics import fmean
from typing import Annotated, Literal, TypeAlias

import torch
from pydantic import (
    Field,
    FiniteFloat,
    PositiveInt,
    StrictInt,
    StrictStr,
    model_validator,
)

from grit.config import SeedSets
from grit.data.waterbirds import (
    WATERBIRDS_GROUP_ORDER,
    GroupId,
    WaterbirdsAdjustedWeightSpec,
)
from grit.features.waterbirds import (
    Normalization,
    WaterbirdsEvaluationFeatureTable,
    WaterbirdsFinalTestView,
)
from grit.methods.types import MethodId, pair_intervention
from grit.schemas import SeedStage, StrictBoundaryModel
from grit.selection.cmnist import CheckpointIdentity

NonEmptyStr: TypeAlias = Annotated[StrictStr, Field(min_length=1)]
NonNegativeInt: TypeAlias = Annotated[StrictInt, Field(ge=0)]
Accuracy: TypeAlias = Annotated[FiniteFloat, Field(ge=0.0, le=1.0)]

GROUP_ORDER = WATERBIRDS_GROUP_ORDER

# The selector scores validation records.
WaterbirdsSelector: TypeAlias = Literal["waterbirds_validation_worst_group"]
ORDINARY_WATERBIRDS_SELECTOR: WaterbirdsSelector = "waterbirds_validation_worst_group"


class _WaterbirdsArtifactLineage(StrictBoundaryModel):
    dataset_manifest_digest: NonEmptyStr
    feature_cache_manifest_digest: NonEmptyStr
    normalization: Normalization
    adjusted_weight_spec_digest: NonEmptyStr


class WaterbirdsGroupAccuracy(StrictBoundaryModel):
    group_id: GroupId
    count: PositiveInt
    correct: NonNegativeInt
    accuracy: Accuracy

    @model_validator(mode="after")
    def _validate_accuracy(self) -> WaterbirdsGroupAccuracy:
        if self.correct > self.count:
            raise ValueError("Waterbirds group correct count exceeds sample count")
        if float(self.accuracy) != self.correct / self.count:
            raise ValueError("Waterbirds group accuracy does not match its counts")
        return self


class _WaterbirdsMetricIdentity(_WaterbirdsArtifactLineage):
    record_id: NonEmptyStr
    run_id: NonEmptyStr
    candidate_id: NonEmptyStr
    method_id: MethodId
    scientific_config_digest: NonEmptyStr
    checkpoint_id: NonEmptyStr
    epoch: NonNegativeInt
    seed: StrictInt
    projection_rank: NonNegativeInt | None
    groups: tuple[
        WaterbirdsGroupAccuracy,
        WaterbirdsGroupAccuracy,
        WaterbirdsGroupAccuracy,
        WaterbirdsGroupAccuracy,
    ]
    adjusted_weight_spec: WaterbirdsAdjustedWeightSpec
    worst_group_accuracy: Accuracy
    adjusted_average_accuracy: Accuracy
    raw_average_accuracy: Accuracy

    @model_validator(mode="after")
    def _validate_aggregates(self) -> _WaterbirdsMetricIdentity:
        if tuple(group.group_id for group in self.groups) != GROUP_ORDER:
            raise ValueError(
                "Waterbirds metrics require all four groups in fixed order"
            )
        weights = self.adjusted_weight_spec.training_group_counts.as_tuple()
        if (
            self.adjusted_weight_spec.dataset_manifest_digest
            != self.dataset_manifest_digest
            or self.adjusted_weight_spec.canonical_digest()
            != self.adjusted_weight_spec_digest
        ):
            raise ValueError("Waterbirds adjusted-weight lineage is inconsistent")
        accuracies = tuple(float(group.accuracy) for group in self.groups)
        expected_worst = min(accuracies)
        expected_adjusted = sum(
            accuracy * count
            for accuracy, count in zip(accuracies, weights, strict=True)
        ) / sum(weights)
        expected_raw = sum(group.correct for group in self.groups) / sum(
            group.count for group in self.groups
        )
        if float(self.worst_group_accuracy) != expected_worst:
            raise ValueError("Waterbirds worst-group accuracy is inconsistent")
        if float(self.adjusted_average_accuracy) != expected_adjusted:
            raise ValueError("Waterbirds adjusted-average accuracy is inconsistent")
        if float(self.raw_average_accuracy) != expected_raw:
            raise ValueError("Waterbirds raw-average accuracy is inconsistent")
        if (
            pair_intervention(self.method_id) != "grit"
            and self.projection_rank is not None
        ):
            raise ValueError("only Waterbirds GRIT metrics carry a projection rank")
        if pair_intervention(self.method_id) == "grit" and self.projection_rank is None:
            raise ValueError("Waterbirds GRIT metrics require a projection rank")
        return self


class WaterbirdsValidationMetricRecord(_WaterbirdsMetricIdentity):
    metric_kind: Literal["validation"]
    split_name: Literal["validation"]
    seed_stage: SeedStage


class WaterbirdsFinalTestMetricRecord(_WaterbirdsMetricIdentity):
    metric_kind: Literal["final_test"]
    split_name: Literal["test"]
    seed_stage: Literal[SeedStage.FINAL]


WaterbirdsSelectorRecord: TypeAlias = WaterbirdsValidationMetricRecord
# Any four-group record, including the gated final one.
WaterbirdsMetricRecord: TypeAlias = (
    WaterbirdsSelectorRecord | WaterbirdsFinalTestMetricRecord
)


class WaterbirdsCheckpointSelection(_WaterbirdsArtifactLineage):
    decision_id: NonEmptyStr
    selector: WaterbirdsSelector
    method_id: MethodId
    seed_stage: SeedStage
    seed: StrictInt
    checkpoint: CheckpointIdentity
    worst_group_accuracy: Accuracy
    adjusted_average_accuracy: Accuracy
    projection_rank: NonNegativeInt | None
    contributing_record_id: NonEmptyStr


class WaterbirdsCandidateSelection(_WaterbirdsArtifactLineage):
    decision_id: NonEmptyStr
    selector: WaterbirdsSelector
    method_id: MethodId
    candidate_id: NonEmptyStr
    scientific_config_digest: NonEmptyStr
    mean_worst_group_accuracy: Accuracy
    mean_adjusted_average_accuracy: Accuracy
    projection_rank: NonNegativeInt | None
    checkpoint_decisions: tuple[WaterbirdsCheckpointSelection, ...]

    @model_validator(mode="after")
    def _validate_decisions(self) -> WaterbirdsCandidateSelection:
        if not self.checkpoint_decisions:
            raise ValueError("Waterbirds candidate selection requires seed decisions")
        seed_keys = tuple(
            (decision.seed_stage, decision.seed)
            for decision in self.checkpoint_decisions
        )
        if len(seed_keys) != len(set(seed_keys)):
            raise ValueError("Waterbirds candidate selection has duplicate seeds")
        for decision in self.checkpoint_decisions:
            if decision.method_id != self.method_id:
                raise ValueError("Waterbirds candidate method identity is inconsistent")
            if decision.selector != self.selector:
                raise ValueError("Waterbirds candidate mixes checkpoint selectors")
            if decision.checkpoint.candidate_id != self.candidate_id:
                raise ValueError("Waterbirds candidate ID is inconsistent")
            if (
                decision.checkpoint.scientific_config_digest
                != self.scientific_config_digest
            ):
                raise ValueError("Waterbirds candidate config identity is inconsistent")
            if decision.projection_rank != self.projection_rank:
                raise ValueError("Waterbirds candidate projection rank is inconsistent")
            if _lineage(decision) != _lineage(self):
                raise ValueError(
                    "Waterbirds candidate artifact lineage is inconsistent"
                )
            if decision.seed_stage is SeedStage.FINAL:
                raise ValueError("final-stage records cannot select hyperparameters")
        if fmean(
            float(decision.worst_group_accuracy)
            for decision in self.checkpoint_decisions
        ) != float(self.mean_worst_group_accuracy):
            raise ValueError("Waterbirds candidate worst-group mean is inconsistent")
        if fmean(
            float(decision.adjusted_average_accuracy)
            for decision in self.checkpoint_decisions
        ) != float(self.mean_adjusted_average_accuracy):
            raise ValueError("Waterbirds candidate adjusted mean is inconsistent")
        return self


class WaterbirdsTuningFinalists(_WaterbirdsArtifactLineage):
    schema_version: Literal["grit.waterbirds-tuning-finalists/v1"] = (
        "grit.waterbirds-tuning-finalists/v1"
    )
    artifact_id: NonEmptyStr
    selector: WaterbirdsSelector
    method_id: MethodId
    tuning_seeds: Annotated[tuple[StrictInt, ...], Field(min_length=3, max_length=3)]
    ordered_candidates: Annotated[
        tuple[WaterbirdsCandidateSelection, ...], Field(min_length=3, max_length=3)
    ]

    @model_validator(mode="after")
    def _validate_finalists(self) -> WaterbirdsTuningFinalists:
        if len(set(self.tuning_seeds)) != 3:
            raise ValueError("Waterbirds tuning seeds must be unique")
        candidate_ids = tuple(item.candidate_id for item in self.ordered_candidates)
        if len(set(candidate_ids)) != 3:
            raise ValueError("Waterbirds finalists must be three unique candidates")
        expected = {(SeedStage.TUNING, seed) for seed in self.tuning_seeds}
        for decision in self.ordered_candidates:
            if decision.method_id != self.method_id:
                raise ValueError("Waterbirds finalist method is inconsistent")
            if decision.selector != self.selector:
                raise ValueError("Waterbirds finalist selector is inconsistent")
            if _lineage(decision) != _lineage(self):
                raise ValueError("Waterbirds finalist artifact lineage is inconsistent")
            observed = {
                (item.seed_stage, item.seed) for item in decision.checkpoint_decisions
            }
            if observed != expected or len(decision.checkpoint_decisions) != 3:
                raise ValueError("Waterbirds finalist requires exact tuning decisions")
        if self.ordered_candidates != tuple(
            sorted(self.ordered_candidates, key=_candidate_key)
        ):
            raise ValueError("Waterbirds finalists must retain deterministic order")
        return self


class FrozenWaterbirdsCandidate(_WaterbirdsArtifactLineage):
    schema_version: Literal["grit.waterbirds-frozen-candidate/v1"] = (
        "grit.waterbirds-frozen-candidate/v1"
    )
    frozen_selection_id: NonEmptyStr
    selector: WaterbirdsSelector
    method_id: MethodId
    candidate_id: NonEmptyStr
    scientific_config_digest: NonEmptyStr
    projection_rank: NonNegativeInt | None
    seed_sets: SeedSets
    finalists: WaterbirdsTuningFinalists
    decision: WaterbirdsCandidateSelection

    @model_validator(mode="after")
    def _validate_freeze(self) -> FrozenWaterbirdsCandidate:
        finalists = {
            item.candidate_id: item for item in self.finalists.ordered_candidates
        }
        tuning = finalists.get(self.candidate_id)
        if tuning is None:
            raise ValueError("frozen Waterbirds candidate is outside its finalists")
        identity = (
            self.method_id,
            self.candidate_id,
            self.scientific_config_digest,
            self.projection_rank,
        )
        if identity != (
            self.decision.method_id,
            self.decision.candidate_id,
            self.decision.scientific_config_digest,
            self.decision.projection_rank,
        ) or identity != (
            tuning.method_id,
            tuning.candidate_id,
            tuning.scientific_config_digest,
            tuning.projection_rank,
        ):
            raise ValueError("frozen Waterbirds candidate identity is inconsistent")
        if self.selector != self.finalists.selector or (
            self.selector != self.decision.selector
        ):
            raise ValueError("frozen Waterbirds candidate selector is inconsistent")
        if _lineage(self) != _lineage(self.finalists) or _lineage(self) != _lineage(
            self.decision
        ):
            raise ValueError("frozen Waterbirds artifact lineage is inconsistent")
        if self.finalists.tuning_seeds != self.seed_sets.tuning:
            raise ValueError("Waterbirds finalist seeds do not match configuration")
        observed = {
            (item.seed_stage, item.seed) for item in self.decision.checkpoint_decisions
        }
        expected = {
            *((SeedStage.TUNING, seed) for seed in self.seed_sets.tuning),
            *((SeedStage.CONFIRMATION, seed) for seed in self.seed_sets.confirmation),
        }
        if observed != expected or len(self.decision.checkpoint_decisions) != 5:
            raise ValueError("Waterbirds freeze requires exact five-seed decisions")
        return self


class FrozenWaterbirdsCheckpoint(_WaterbirdsArtifactLineage):
    frozen_checkpoint_id: NonEmptyStr
    candidate_selection_id: NonEmptyStr
    method_id: MethodId
    checkpoint: CheckpointIdentity
    decision: WaterbirdsCheckpointSelection

    @model_validator(mode="after")
    def _validate_freeze(self) -> FrozenWaterbirdsCheckpoint:
        if self.decision.seed_stage is not SeedStage.FINAL:
            raise ValueError("Waterbirds final checkpoint requires a final seed")
        if self.decision.method_id != self.method_id:
            raise ValueError("Waterbirds frozen checkpoint method is inconsistent")
        if self.decision.checkpoint != self.checkpoint:
            raise ValueError("Waterbirds frozen checkpoint identity is inconsistent")
        if _lineage(self.decision) != _lineage(self):
            raise ValueError("Waterbirds frozen checkpoint lineage is inconsistent")
        return self


def compute_waterbirds_validation_metric(
    table: WaterbirdsEvaluationFeatureTable,
    predictions: torch.Tensor,
    *,
    adjusted_weights: WaterbirdsAdjustedWeightSpec,
    record_id: str,
    run_id: str,
    candidate_id: str,
    method_id: MethodId,
    scientific_config_digest: str,
    checkpoint_id: str,
    epoch: int,
    seed_stage: SeedStage,
    seed: int,
    projection_rank: int | None,
) -> WaterbirdsValidationMetricRecord:
    if table.split_role != "validation":
        raise TypeError("validation metrics require a Waterbirds validation table")
    weights = WaterbirdsAdjustedWeightSpec.model_validate_json(
        adjusted_weights.canonical_json()
    )
    if weights.dataset_manifest_digest != table.dataset_manifest_digest:
        raise ValueError("Waterbirds adjusted weights belong to another dataset")
    values = _group_values(table, predictions)
    aggregate = _aggregate_fields(values, weights)
    return WaterbirdsValidationMetricRecord(
        record_id=record_id,
        run_id=run_id,
        candidate_id=candidate_id,
        method_id=method_id,
        scientific_config_digest=scientific_config_digest,
        dataset_manifest_digest=table.dataset_manifest_digest,
        feature_cache_manifest_digest=table.feature_cache_manifest_digest,
        normalization=table.normalization,
        adjusted_weight_spec_digest=weights.canonical_digest(),
        checkpoint_id=checkpoint_id,
        epoch=epoch,
        seed=seed,
        projection_rank=projection_rank,
        groups=values,
        adjusted_weight_spec=weights,
        worst_group_accuracy=aggregate[0],
        adjusted_average_accuracy=aggregate[1],
        raw_average_accuracy=aggregate[2],
        metric_kind="validation",
        split_name="validation",
        seed_stage=seed_stage,
    )


def compute_waterbirds_final_metric(
    view: WaterbirdsFinalTestView,
    predictions: torch.Tensor,
    *,
    adjusted_weights: WaterbirdsAdjustedWeightSpec,
    record_id: str,
) -> WaterbirdsFinalTestMetricRecord:
    if (
        type(view) is not WaterbirdsFinalTestView
        or view.table.split_role != "final_test"
    ):
        raise TypeError("final metrics require a gate-authorized Waterbirds final view")
    weights = WaterbirdsAdjustedWeightSpec.model_validate_json(
        adjusted_weights.canonical_json()
    )
    if weights.dataset_manifest_digest != view.table.dataset_manifest_digest:
        raise ValueError("Waterbirds adjusted weights belong to another dataset")
    values = _group_values(view.table, predictions)
    aggregate = _aggregate_fields(values, weights)
    return WaterbirdsFinalTestMetricRecord(
        record_id=record_id,
        run_id=view.run_id,
        candidate_id=view.candidate_id,
        method_id=view.method_id,
        scientific_config_digest=view.scientific_config_digest,
        dataset_manifest_digest=view.table.dataset_manifest_digest,
        feature_cache_manifest_digest=view.feature_cache_manifest_digest,
        normalization=view.table.normalization,
        adjusted_weight_spec_digest=weights.canonical_digest(),
        checkpoint_id=view.checkpoint_id,
        epoch=view.epoch,
        seed=view.seed,
        projection_rank=view.projection_rank,
        groups=values,
        adjusted_weight_spec=weights,
        worst_group_accuracy=aggregate[0],
        adjusted_average_accuracy=aggregate[1],
        raw_average_accuracy=aggregate[2],
        metric_kind="final_test",
        split_name="test",
        seed_stage=SeedStage.FINAL,
    )


def select_waterbirds_checkpoint(
    records: Sequence[WaterbirdsSelectorRecord],
    selector: WaterbirdsSelector = ORDINARY_WATERBIRDS_SELECTOR,
) -> WaterbirdsCheckpointSelection:
    validated = _selector_records(records, selector)
    run_keys = {
        (
            item.run_id,
            item.candidate_id,
            item.method_id,
            item.scientific_config_digest,
            item.seed_stage,
            item.seed,
            item.projection_rank,
            item.dataset_manifest_digest,
            item.feature_cache_manifest_digest,
            item.normalization,
            item.adjusted_weight_spec_digest,
        )
        for item in validated
    }
    if len(run_keys) != 1:
        raise ValueError("Waterbirds checkpoint selection requires one candidate run")
    checkpoint_ids = [item.checkpoint_id for item in validated]
    if len(checkpoint_ids) != len(set(checkpoint_ids)):
        raise ValueError("Waterbirds checkpoint metric records must be unique")
    selected = min(
        validated,
        key=lambda item: (
            -float(item.worst_group_accuracy),
            -float(item.adjusted_average_accuracy),
            item.epoch,
            item.checkpoint_id,
        ),
    )
    checkpoint = CheckpointIdentity(
        checkpoint_id=selected.checkpoint_id,
        candidate_id=selected.candidate_id,
        run_id=selected.run_id,
        scientific_config_digest=selected.scientific_config_digest,
        epoch=selected.epoch,
    )
    return WaterbirdsCheckpointSelection(
        decision_id=f"waterbirds-checkpoint:{selected.checkpoint_id}",
        selector=selector,
        method_id=selected.method_id,
        seed_stage=selected.seed_stage,
        seed=selected.seed,
        checkpoint=checkpoint,
        worst_group_accuracy=selected.worst_group_accuracy,
        adjusted_average_accuracy=selected.adjusted_average_accuracy,
        projection_rank=selected.projection_rank,
        dataset_manifest_digest=selected.dataset_manifest_digest,
        feature_cache_manifest_digest=selected.feature_cache_manifest_digest,
        normalization=selected.normalization,
        adjusted_weight_spec_digest=selected.adjusted_weight_spec_digest,
        contributing_record_id=selected.record_id,
    )


def rank_waterbirds_tuning_candidates(
    records: Sequence[WaterbirdsSelectorRecord],
    seed_sets: SeedSets,
    selector: WaterbirdsSelector = ORDINARY_WATERBIRDS_SELECTOR,
) -> tuple[WaterbirdsCandidateSelection, ...]:
    validated_seeds = _seed_sets(seed_sets)
    validated = _selector_records(records, selector)
    if any(item.seed_stage is not SeedStage.TUNING for item in validated):
        raise ValueError("Waterbirds tuning ranking accepts tuning records only")
    expected = {(SeedStage.TUNING, seed) for seed in validated_seeds.tuning}
    return _rank_candidates(validated, expected, selector)


def make_waterbirds_tuning_finalists(
    records: Sequence[WaterbirdsSelectorRecord],
    seed_sets: SeedSets,
    selector: WaterbirdsSelector = ORDINARY_WATERBIRDS_SELECTOR,
) -> WaterbirdsTuningFinalists:
    validated_seeds = _seed_sets(seed_sets)
    ranked = rank_waterbirds_tuning_candidates(records, validated_seeds, selector)
    if len(ranked) < 3:
        raise ValueError("Waterbirds tuning requires at least three candidates")
    top = ranked[:3]
    return WaterbirdsTuningFinalists(
        artifact_id=f"waterbirds-finalists:{selector}:{top[0].method_id}",
        selector=selector,
        method_id=top[0].method_id,
        dataset_manifest_digest=top[0].dataset_manifest_digest,
        feature_cache_manifest_digest=top[0].feature_cache_manifest_digest,
        normalization=top[0].normalization,
        adjusted_weight_spec_digest=top[0].adjusted_weight_spec_digest,
        tuning_seeds=validated_seeds.tuning,
        ordered_candidates=top,
    )


def select_confirmed_waterbirds_candidate(
    confirmation_records: Sequence[WaterbirdsSelectorRecord],
    finalists: WaterbirdsTuningFinalists,
    seed_sets: SeedSets,
) -> WaterbirdsCandidateSelection:
    validated_seeds = _seed_sets(seed_sets)
    validated_finalists = WaterbirdsTuningFinalists.model_validate_json(
        finalists.canonical_json()
    )
    selector = validated_finalists.selector
    if validated_finalists.tuning_seeds != validated_seeds.tuning:
        raise ValueError("Waterbirds finalist tuning seeds do not match configuration")
    records = _selector_records(confirmation_records, selector)
    if any(item.seed_stage is not SeedStage.CONFIRMATION for item in records):
        raise ValueError("Waterbirds confirmation accepts confirmation records only")
    finalist_by_id = {
        item.candidate_id: item for item in validated_finalists.ordered_candidates
    }
    if {item.candidate_id for item in records} != set(finalist_by_id):
        raise ValueError("Waterbirds confirmation requires exactly its finalists")
    if any(item.method_id != validated_finalists.method_id for item in records):
        raise ValueError("Waterbirds confirmation method does not match finalists")
    expected = {(SeedStage.CONFIRMATION, seed) for seed in validated_seeds.confirmation}
    confirmations = _rank_candidates(records, expected, selector)
    combined: list[WaterbirdsCandidateSelection] = []
    for confirmation in confirmations:
        tuning = finalist_by_id[confirmation.candidate_id]
        if (
            confirmation.scientific_config_digest != tuning.scientific_config_digest
            or confirmation.projection_rank != tuning.projection_rank
            or _lineage(confirmation) != _lineage(tuning)
        ):
            raise ValueError("Waterbirds confirmation candidate identity changed")
        decisions = (*tuning.checkpoint_decisions, *confirmation.checkpoint_decisions)
        combined.append(
            WaterbirdsCandidateSelection(
                decision_id=f"waterbirds-candidate:{confirmation.candidate_id}:confirmed",
                selector=selector,
                method_id=confirmation.method_id,
                candidate_id=confirmation.candidate_id,
                scientific_config_digest=confirmation.scientific_config_digest,
                mean_worst_group_accuracy=fmean(
                    float(item.worst_group_accuracy) for item in decisions
                ),
                mean_adjusted_average_accuracy=fmean(
                    float(item.adjusted_average_accuracy) for item in decisions
                ),
                projection_rank=confirmation.projection_rank,
                dataset_manifest_digest=confirmation.dataset_manifest_digest,
                feature_cache_manifest_digest=(
                    confirmation.feature_cache_manifest_digest
                ),
                normalization=confirmation.normalization,
                adjusted_weight_spec_digest=(confirmation.adjusted_weight_spec_digest),
                checkpoint_decisions=decisions,
            )
        )
    return min(combined, key=_candidate_key)


def freeze_waterbirds_candidate(
    decision: WaterbirdsCandidateSelection,
    finalists: WaterbirdsTuningFinalists,
    seed_sets: SeedSets,
) -> FrozenWaterbirdsCandidate:
    validated = WaterbirdsCandidateSelection.model_validate_json(
        decision.canonical_json()
    )
    artifact = WaterbirdsTuningFinalists.model_validate_json(finalists.canonical_json())
    if artifact.selector != validated.selector:
        raise ValueError("Waterbirds freeze mixes finalist and decision selectors")
    frozen = FrozenWaterbirdsCandidate(
        frozen_selection_id=f"frozen:{validated.decision_id}",
        selector=validated.selector,
        method_id=validated.method_id,
        candidate_id=validated.candidate_id,
        scientific_config_digest=validated.scientific_config_digest,
        projection_rank=validated.projection_rank,
        dataset_manifest_digest=validated.dataset_manifest_digest,
        feature_cache_manifest_digest=validated.feature_cache_manifest_digest,
        normalization=validated.normalization,
        adjusted_weight_spec_digest=validated.adjusted_weight_spec_digest,
        seed_sets=_seed_sets(seed_sets),
        finalists=artifact,
        decision=validated,
    )
    return frozen


def freeze_waterbirds_final_checkpoint(
    decision: WaterbirdsCheckpointSelection,
    candidate: FrozenWaterbirdsCandidate,
) -> FrozenWaterbirdsCheckpoint:
    selected = WaterbirdsCheckpointSelection.model_validate_json(
        decision.canonical_json()
    )
    frozen_candidate = FrozenWaterbirdsCandidate.model_validate_json(
        candidate.canonical_json()
    )
    if selected.seed_stage is not SeedStage.FINAL:
        raise ValueError("Waterbirds final checkpoint requires final-stage validation")
    if selected.seed not in frozen_candidate.seed_sets.final:
        raise ValueError("Waterbirds final checkpoint seed is not configured")
    if selected.method_id != frozen_candidate.method_id:
        raise ValueError("Waterbirds final checkpoint method changed")
    if selected.selector != frozen_candidate.selector:
        raise ValueError("Waterbirds final checkpoint selector changed")
    if selected.checkpoint.candidate_id != frozen_candidate.candidate_id:
        raise ValueError("Waterbirds final checkpoint candidate changed")
    if (
        selected.checkpoint.scientific_config_digest
        != frozen_candidate.scientific_config_digest
        or selected.projection_rank != frozen_candidate.projection_rank
        or _lineage(selected) != _lineage(frozen_candidate)
    ):
        raise ValueError("Waterbirds final checkpoint hyperparameters changed")
    return FrozenWaterbirdsCheckpoint(
        frozen_checkpoint_id=f"frozen:{selected.decision_id}",
        candidate_selection_id=frozen_candidate.frozen_selection_id,
        method_id=selected.method_id,
        dataset_manifest_digest=selected.dataset_manifest_digest,
        feature_cache_manifest_digest=selected.feature_cache_manifest_digest,
        normalization=selected.normalization,
        adjusted_weight_spec_digest=selected.adjusted_weight_spec_digest,
        checkpoint=selected.checkpoint,
        decision=selected,
    )


def _rank_candidates(
    records: tuple[WaterbirdsSelectorRecord, ...],
    expected_seed_keys: set[tuple[SeedStage, int]],
    selector: WaterbirdsSelector,
) -> tuple[WaterbirdsCandidateSelection, ...]:
    methods = {item.method_id for item in records}
    if len(methods) != 1:
        raise ValueError("Waterbirds methods must be selected independently")
    lineages = {_lineage(item) for item in records}
    if len(lineages) != 1:
        raise ValueError("Waterbirds selection cannot mix artifact lineage")
    by_run: dict[tuple[str, str], list[WaterbirdsSelectorRecord]] = defaultdict(list)
    for record in records:
        by_run[(record.candidate_id, record.run_id)].append(record)
    by_candidate: dict[str, list[WaterbirdsCheckpointSelection]] = defaultdict(list)
    for (candidate_id, _), run_records in by_run.items():
        by_candidate[candidate_id].append(
            select_waterbirds_checkpoint(run_records, selector)
        )
    ranked: list[WaterbirdsCandidateSelection] = []
    for candidate_id, decisions in by_candidate.items():
        seed_keys = [(item.seed_stage, item.seed) for item in decisions]
        if (
            len(seed_keys) != len(set(seed_keys))
            or set(seed_keys) != expected_seed_keys
        ):
            raise ValueError(
                f"Waterbirds candidate {candidate_id!r} lacks exact stage/seed runs"
            )
        decisions.sort(key=lambda item: (item.seed_stage.value, item.seed))
        configs = {item.checkpoint.scientific_config_digest for item in decisions}
        ranks = {item.projection_rank for item in decisions}
        if len(configs) != 1 or len(ranks) != 1:
            raise ValueError("Waterbirds candidate identity changes across seeds")
        ranked.append(
            WaterbirdsCandidateSelection(
                decision_id=f"waterbirds-candidate:{candidate_id}",
                selector=selector,
                method_id=decisions[0].method_id,
                candidate_id=candidate_id,
                scientific_config_digest=configs.pop(),
                mean_worst_group_accuracy=fmean(
                    float(item.worst_group_accuracy) for item in decisions
                ),
                mean_adjusted_average_accuracy=fmean(
                    float(item.adjusted_average_accuracy) for item in decisions
                ),
                projection_rank=ranks.pop(),
                dataset_manifest_digest=decisions[0].dataset_manifest_digest,
                feature_cache_manifest_digest=(
                    decisions[0].feature_cache_manifest_digest
                ),
                normalization=decisions[0].normalization,
                adjusted_weight_spec_digest=(decisions[0].adjusted_weight_spec_digest),
                checkpoint_decisions=tuple(decisions),
            )
        )
    return tuple(sorted(ranked, key=_candidate_key))


def _candidate_key(
    decision: WaterbirdsCandidateSelection,
) -> tuple[float, float, int, str]:
    rank = decision.projection_rank if decision.projection_rank is not None else 0
    return (
        -float(decision.mean_worst_group_accuracy),
        -float(decision.mean_adjusted_average_accuracy),
        rank,
        decision.candidate_id,
    )


def _selector_records(
    records: Sequence[WaterbirdsSelectorRecord],
    selector: WaterbirdsSelector,
) -> tuple[WaterbirdsSelectorRecord, ...]:
    """Revalidate records and refuse the wrong split for the selector.

    The selector accepts validation records only, so a test record can never steer it.
    """

    del selector
    materialized = tuple(records)
    if not materialized:
        raise ValueError("Waterbirds selection requires metrics")
    if any(type(item) is not WaterbirdsValidationMetricRecord for item in materialized):
        raise TypeError("Waterbirds ordinary selection accepts validation metrics only")
    return tuple(
        WaterbirdsValidationMetricRecord.model_validate_json(item.canonical_json())
        for item in materialized
    )


def _seed_sets(seed_sets: SeedSets) -> SeedSets:
    return SeedSets.model_validate_json(seed_sets.canonical_json())


def _group_values(
    table: WaterbirdsEvaluationFeatureTable,
    predictions: torch.Tensor,
) -> tuple[
    WaterbirdsGroupAccuracy,
    WaterbirdsGroupAccuracy,
    WaterbirdsGroupAccuracy,
    WaterbirdsGroupAccuracy,
]:
    if predictions.ndim != 1 or int(predictions.shape[0]) != len(table.record_ids):
        raise ValueError("Waterbirds predictions must align with evaluation rows")
    predicted = predictions.detach().cpu().to(torch.int64)
    if bool(((predicted != 0) & (predicted != 1)).any()):
        raise ValueError("Waterbirds predictions must be binary")
    values: list[WaterbirdsGroupAccuracy] = []
    for group in GROUP_ORDER:
        indices = torch.tensor(
            [index for index, value in enumerate(table.group_ids) if value == group],
            dtype=torch.int64,
        )
        if int(indices.numel()) == 0:
            raise ValueError(f"Waterbirds evaluation is missing group {group!r}")
        correct = int((predicted[indices] == table.labels[indices]).sum().item())
        count = int(indices.numel())
        values.append(
            WaterbirdsGroupAccuracy(
                group_id=group,
                count=count,
                correct=correct,
                accuracy=correct / count,
            )
        )
    return (values[0], values[1], values[2], values[3])


def _aggregate_fields(
    groups: tuple[
        WaterbirdsGroupAccuracy,
        WaterbirdsGroupAccuracy,
        WaterbirdsGroupAccuracy,
        WaterbirdsGroupAccuracy,
    ],
    adjusted_weights: WaterbirdsAdjustedWeightSpec,
) -> tuple[float, float, float]:
    accuracies = tuple(float(group.accuracy) for group in groups)
    weights = adjusted_weights.training_group_counts.as_tuple()
    if min(weights) <= 0:
        raise ValueError("Waterbirds training weights require every group")
    adjusted = sum(
        accuracy * count for accuracy, count in zip(accuracies, weights, strict=True)
    ) / sum(weights)
    raw = sum(group.correct for group in groups) / sum(group.count for group in groups)
    return min(accuracies), adjusted, raw


def _lineage(
    value: _WaterbirdsArtifactLineage,
) -> tuple[str, str, Normalization, str]:
    return (
        value.dataset_manifest_digest,
        value.feature_cache_manifest_digest,
        value.normalization,
        value.adjusted_weight_spec_digest,
    )
