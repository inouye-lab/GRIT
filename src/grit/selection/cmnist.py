"""Validation-only selection contracts for repeated-view linear-probe studies."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from statistics import fmean
from typing import Annotated, Literal, TypeAlias

from pydantic import (
    Field,
    FiniteFloat,
    PositiveInt,
    StrictInt,
    StrictStr,
    model_validator,
)

from grit.config import SeedSets
from grit.schemas import (
    HELD_OUT_VALIDATION_NAMES,
    IN_DOMAIN_VALIDATION_NAMES,
    CmnistSelector,
    SeedStage,
    StrictBoundaryModel,
)

NonEmptyStr: TypeAlias = Annotated[StrictStr, Field(min_length=1)]
NonNegativeInt: TypeAlias = Annotated[StrictInt, Field(ge=0)]
ValidationSplitName: TypeAlias = Literal[
    "val_e01",
    "val_e02",
    "val_e03",
    "val_e04",
    "val_e05",
    "val_e06",
    "val_e07",
]
# The primary selector cites three environments; the secondary cites two.
SELECTOR_SPLIT_COUNTS: dict[CmnistSelector, frozenset[int]] = {
    CmnistSelector.PRIMARY_ROBUST: frozenset({3}),
    CmnistSelector.SECONDARY_SOURCE: frozenset({2}),
}


class _MetricIdentity(StrictBoundaryModel):
    record_id: NonEmptyStr
    run_id: NonEmptyStr
    candidate_id: NonEmptyStr
    method_id: NonEmptyStr
    scientific_config_digest: NonEmptyStr
    checkpoint_id: NonEmptyStr
    epoch: NonNegativeInt
    seed: StrictInt
    value: FiniteFloat
    sample_count: PositiveInt


class ValidationMetricRecord(_MetricIdentity):
    """The only metric record accepted by ordinary selectors."""

    metric_kind: Literal["validation"]
    seed_stage: SeedStage
    split_name: ValidationSplitName
    metric_name: Literal["accuracy"]
    projection_rank: NonNegativeInt | None


class FinalTestMetricRecord(_MetricIdentity):
    metric_kind: Literal["final_test"]
    seed_stage: Literal[SeedStage.FINAL]
    split_name: Literal["test_ood"]
    metric_name: Literal["accuracy"]


SelectorRecord: TypeAlias = ValidationMetricRecord


class CheckpointIdentity(StrictBoundaryModel):
    checkpoint_id: NonEmptyStr
    candidate_id: NonEmptyStr
    run_id: NonEmptyStr
    scientific_config_digest: NonEmptyStr
    epoch: NonNegativeInt


class CheckpointSelectionDecision(StrictBoundaryModel):
    decision_id: NonEmptyStr
    selector: CmnistSelector
    method_id: NonEmptyStr
    seed_stage: SeedStage
    seed: StrictInt
    checkpoint: CheckpointIdentity
    objective_value: FiniteFloat
    mean_accuracy: FiniteFloat
    projection_rank: NonNegativeInt | None
    contributing_record_ids: tuple[NonEmptyStr, ...]

    @model_validator(mode="after")
    def _validate_contributors(self) -> CheckpointSelectionDecision:
        allowed = SELECTOR_SPLIT_COUNTS[self.selector]
        if len(self.contributing_record_ids) not in allowed:
            raise ValueError(
                "checkpoint selection must cite every selector split exactly once"
            )
        if len(set(self.contributing_record_ids)) != len(self.contributing_record_ids):
            raise ValueError("checkpoint selection record IDs must be unique")
        return self


class CandidateSelectionDecision(StrictBoundaryModel):
    decision_id: NonEmptyStr
    selector: CmnistSelector
    method_id: NonEmptyStr
    candidate_id: NonEmptyStr
    scientific_config_digest: NonEmptyStr
    objective_value: FiniteFloat
    mean_accuracy: FiniteFloat
    projection_rank: NonNegativeInt | None
    contributing_checkpoint_decisions: tuple[CheckpointSelectionDecision, ...]

    @model_validator(mode="after")
    def _validate_contributors(self) -> CandidateSelectionDecision:
        if not self.contributing_checkpoint_decisions:
            raise ValueError("candidate selection requires checkpoint decisions")
        seed_keys = [
            (decision.seed_stage, decision.seed)
            for decision in self.contributing_checkpoint_decisions
        ]
        if len(seed_keys) != len(set(seed_keys)):
            raise ValueError("candidate selection cannot contain duplicate seed runs")
        for decision in self.contributing_checkpoint_decisions:
            if decision.selector is not self.selector:
                raise ValueError(
                    "checkpoint selector does not match candidate selector"
                )
            if decision.method_id != self.method_id:
                raise ValueError("checkpoint method does not match candidate method")
            if decision.checkpoint.candidate_id != self.candidate_id:
                raise ValueError(
                    "checkpoint candidate does not match candidate decision"
                )
            if (
                decision.checkpoint.scientific_config_digest
                != self.scientific_config_digest
            ):
                raise ValueError("checkpoint config does not match candidate decision")
            if decision.seed_stage is SeedStage.FINAL:
                raise ValueError("final-stage validation cannot select a candidate")
            if decision.projection_rank != self.projection_rank:
                raise ValueError("checkpoint rank does not match candidate decision")
        objectives = tuple(
            float(decision.objective_value)
            for decision in self.contributing_checkpoint_decisions
        )
        means = tuple(
            float(decision.mean_accuracy)
            for decision in self.contributing_checkpoint_decisions
        )
        if fmean(objectives) != float(self.objective_value):
            raise ValueError("candidate objective does not match checkpoint decisions")
        if fmean(means) != float(self.mean_accuracy):
            raise ValueError("candidate mean does not match checkpoint decisions")
        return self


class TuningFinalistsArtifact(StrictBoundaryModel):
    """Durable ordered top three for one method and ordinary selector."""

    schema_version: Literal["grit.cmnist-tuning-finalists/v1"] = (
        "grit.cmnist-tuning-finalists/v1"
    )
    artifact_id: NonEmptyStr
    selector: CmnistSelector
    method_id: NonEmptyStr
    tuning_seeds: Annotated[tuple[StrictInt, ...], Field(min_length=3, max_length=3)]
    ordered_candidates: Annotated[
        tuple[CandidateSelectionDecision, ...], Field(min_length=3, max_length=3)
    ]

    @model_validator(mode="after")
    def _validate_finalists(self) -> TuningFinalistsArtifact:
        if len(set(self.tuning_seeds)) != 3:
            raise ValueError("tuning finalist seeds must be unique")
        candidate_ids = tuple(
            decision.candidate_id for decision in self.ordered_candidates
        )
        if len(set(candidate_ids)) != 3:
            raise ValueError("tuning finalists must contain three unique candidates")
        expected_seed_keys = {(SeedStage.TUNING, seed) for seed in self.tuning_seeds}
        for decision in self.ordered_candidates:
            if decision.selector is not self.selector:
                raise ValueError("finalist selector does not match artifact selector")
            if decision.method_id != self.method_id:
                raise ValueError("finalist method does not match artifact method")
            observed_seed_keys = {
                (checkpoint.seed_stage, checkpoint.seed)
                for checkpoint in decision.contributing_checkpoint_decisions
            }
            if observed_seed_keys != expected_seed_keys:
                raise ValueError(
                    "each tuning finalist requires exactly the artifact tuning seeds"
                )
        if self.ordered_candidates != tuple(
            sorted(self.ordered_candidates, key=_candidate_sort_key)
        ):
            raise ValueError("tuning finalists must preserve deterministic rank order")
        return self


class FrozenCandidateSelection(StrictBoundaryModel):
    schema_version: Literal["grit.cmnist-frozen-candidate/v1"] = (
        "grit.cmnist-frozen-candidate/v1"
    )
    frozen_selection_id: NonEmptyStr
    selector: CmnistSelector
    method_id: NonEmptyStr
    candidate_id: NonEmptyStr
    scientific_config_digest: NonEmptyStr
    seed_sets: SeedSets
    finalists: TuningFinalistsArtifact
    decision: CandidateSelectionDecision

    @model_validator(mode="after")
    def _validate_decision(self) -> FrozenCandidateSelection:
        expected = (
            self.selector,
            self.method_id,
            self.candidate_id,
            self.scientific_config_digest,
        )
        observed = (
            self.decision.selector,
            self.decision.method_id,
            self.decision.candidate_id,
            self.decision.scientific_config_digest,
        )
        if observed != expected:
            raise ValueError("frozen candidate fields must match its decision")
        if self.finalists.selector is not self.selector:
            raise ValueError("frozen candidate selector does not match finalists")
        if self.finalists.method_id != self.method_id:
            raise ValueError("frozen candidate method does not match finalists")
        if self.finalists.tuning_seeds != self.seed_sets.tuning:
            raise ValueError("frozen candidate seed sets do not match finalists")
        finalist_by_id = {
            finalist.candidate_id: finalist
            for finalist in self.finalists.ordered_candidates
        }
        tuning_decision = finalist_by_id.get(self.candidate_id)
        if tuning_decision is None:
            raise ValueError("frozen candidate is outside the tuning finalists")
        if (
            tuning_decision.scientific_config_digest != self.scientific_config_digest
            or tuning_decision.projection_rank != self.decision.projection_rank
        ):
            raise ValueError("frozen candidate identity does not match its finalist")
        observed_seed_keys = {
            (checkpoint.seed_stage, checkpoint.seed)
            for checkpoint in self.decision.contributing_checkpoint_decisions
        }
        expected_seed_keys = {
            *((SeedStage.TUNING, seed) for seed in self.seed_sets.tuning),
            *((SeedStage.CONFIRMATION, seed) for seed in self.seed_sets.confirmation),
        }
        if observed_seed_keys != expected_seed_keys:
            raise ValueError(
                "frozen candidate requires exactly its configured tuning and "
                "confirmation seeds"
            )
        return self


class FrozenCheckpointSelection(StrictBoundaryModel):
    frozen_checkpoint_id: NonEmptyStr
    candidate_selection_id: NonEmptyStr
    selector: CmnistSelector
    method_id: NonEmptyStr
    checkpoint: CheckpointIdentity
    decision: CheckpointSelectionDecision

    @model_validator(mode="after")
    def _validate_decision(self) -> FrozenCheckpointSelection:
        if self.decision.seed_stage is not SeedStage.FINAL:
            raise ValueError(
                "only final-stage validation may freeze a final checkpoint"
            )
        if self.decision.selector is not self.selector:
            raise ValueError("frozen checkpoint selector mismatch")
        if self.decision.method_id != self.method_id:
            raise ValueError("frozen checkpoint method mismatch")
        if self.decision.checkpoint != self.checkpoint:
            raise ValueError("frozen checkpoint identity must match its decision")
        return self


class FinalistUnion(StrictBoundaryModel):
    """Primary/secondary top sets confirmed once without merging their winners."""

    schema_version: Literal["grit.cmnist-finalist-union/v1"] = (
        "grit.cmnist-finalist-union/v1"
    )
    method_id: NonEmptyStr
    primary: TuningFinalistsArtifact
    secondary: TuningFinalistsArtifact
    confirmation_candidate_ids: tuple[NonEmptyStr, ...]

    @model_validator(mode="after")
    def _validate_union(self) -> FinalistUnion:
        if self.primary.selector is not CmnistSelector.PRIMARY_ROBUST:
            raise ValueError("primary finalist artifact has the wrong selector")
        if self.secondary.selector is not CmnistSelector.SECONDARY_SOURCE:
            raise ValueError("secondary finalist artifact has the wrong selector")
        if (
            self.primary.method_id != self.method_id
            or self.secondary.method_id != self.method_id
        ):
            raise ValueError("finalist union methods must match")
        if self.primary.tuning_seeds != self.secondary.tuning_seeds:
            raise ValueError("finalist union tuning seeds must match")
        primary_ids = tuple(
            decision.candidate_id for decision in self.primary.ordered_candidates
        )
        secondary_ids = tuple(
            decision.candidate_id for decision in self.secondary.ordered_candidates
        )
        primary_by_id = {
            decision.candidate_id: (
                decision.scientific_config_digest,
                decision.projection_rank,
            )
            for decision in self.primary.ordered_candidates
        }
        secondary_by_id = {
            decision.candidate_id: (
                decision.scientific_config_digest,
                decision.projection_rank,
            )
            for decision in self.secondary.ordered_candidates
        }
        for candidate_id in set(primary_by_id) & set(secondary_by_id):
            if primary_by_id[candidate_id] != secondary_by_id[candidate_id]:
                raise ValueError(
                    "shared primary/secondary finalist identity is inconsistent"
                )
        expected = tuple(dict.fromkeys(primary_ids + secondary_ids))
        if self.confirmation_candidate_ids != expected:
            raise ValueError(
                "confirmation candidates must be the ordered finalist union"
            )
        return self


def _candidate_sort_key(
    decision: CandidateSelectionDecision,
) -> tuple[float, float, int, str]:
    rank_key = decision.projection_rank if decision.projection_rank is not None else 0
    return (
        -float(decision.objective_value),
        -float(decision.mean_accuracy),
        rank_key,
        decision.candidate_id,
    )


def _revalidate_seed_sets(seed_sets: SeedSets) -> SeedSets:
    return SeedSets.model_validate(seed_sets.model_dump(mode="python"))


def _require_validation_records(
    records: Sequence[ValidationMetricRecord],
) -> tuple[ValidationMetricRecord, ...]:
    materialized = tuple(records)
    if not materialized:
        raise ValueError("selection requires validation metric records")
    if any(type(record) is not ValidationMetricRecord for record in materialized):
        raise TypeError("ordinary selectors accept ValidationMetricRecord values only")
    return tuple(
        ValidationMetricRecord.model_validate(record.model_dump(mode="python"))
        for record in materialized
    )


def _require_selector_records(
    records: Sequence[SelectorRecord],
    selector: CmnistSelector,
) -> tuple[SelectorRecord, ...]:
    """Selectors see validation records only."""

    del selector
    return _require_validation_records(records)


def _selector_splits(
    selector: CmnistSelector, observed_splits: set[str]
) -> tuple[str, ...]:
    in_domain = set(IN_DOMAIN_VALIDATION_NAMES)
    held_out = set(HELD_OUT_VALIDATION_NAMES)
    if observed_splits <= in_domain | held_out:
        if selector is CmnistSelector.SECONDARY_SOURCE:
            return IN_DOMAIN_VALIDATION_NAMES
        observed_held_out = sorted(observed_splits & held_out)
        if len(observed_held_out) != 1:
            raise ValueError(
                "the primary selector requires exactly one held-out validation "
                f"rendering, found {observed_held_out!r}"
            )
        return (*IN_DOMAIN_VALIDATION_NAMES, observed_held_out[0])
    raise ValueError("validation records mix or use unsupported split namespaces")


def _checkpoint_sort_key(
    item: tuple[
        CheckpointIdentity,
        float,
        float,
        int | None,
        tuple[str, ...],
    ],
) -> tuple[float, float, int, int, str]:
    checkpoint, objective, mean_accuracy, projection_rank, _ = item
    rank_key = projection_rank if projection_rank is not None else 0
    return (
        -objective,
        -mean_accuracy,
        rank_key,
        checkpoint.epoch,
        checkpoint.checkpoint_id,
    )


def select_checkpoint(
    records: Sequence[SelectorRecord],
    selector: CmnistSelector,
) -> CheckpointSelectionDecision:
    """Select one persisted checkpoint within one candidate/run/seed."""

    validation_records = _require_selector_records(records, selector)
    run_keys = {
        (
            record.run_id,
            record.candidate_id,
            record.method_id,
            record.scientific_config_digest,
            record.seed_stage,
            record.seed,
        )
        for record in validation_records
    }
    if len(run_keys) != 1:
        raise ValueError(
            "checkpoint selection must stay within one candidate run and seed"
        )
    run_projection_ranks = {record.projection_rank for record in validation_records}
    if len(run_projection_ranks) != 1:
        raise ValueError("checkpoint selection cannot change projection rank")

    grouped: dict[str, list[SelectorRecord]] = defaultdict(list)
    for record in validation_records:
        grouped[record.checkpoint_id].append(record)

    required_splits = _selector_splits(
        selector, {record.split_name for record in validation_records}
    )
    candidates: list[
        tuple[CheckpointIdentity, float, float, int | None, tuple[str, ...]]
    ] = []
    for checkpoint_id, checkpoint_records in grouped.items():
        by_split = {record.split_name: record for record in checkpoint_records}
        if len(by_split) != len(checkpoint_records):
            raise ValueError("a checkpoint has duplicate validation split records")
        missing = set(required_splits) - set(by_split)
        if missing:
            missing_names = sorted(missing)
            raise ValueError(
                f"checkpoint {checkpoint_id!r} is missing selector splits "
                f"{missing_names!r}"
            )
        identities = {
            (
                record.candidate_id,
                record.run_id,
                record.scientific_config_digest,
                record.epoch,
            )
            for record in checkpoint_records
        }
        if len(identities) != 1:
            raise ValueError(
                "checkpoint validation records have inconsistent identities"
            )
        projection_ranks = {record.projection_rank for record in checkpoint_records}
        if len(projection_ranks) != 1:
            raise ValueError("checkpoint validation records have inconsistent ranks")
        selected_records = tuple(by_split[name] for name in required_splits)
        values = tuple(float(record.value) for record in selected_records)
        first = selected_records[0]
        checkpoint = CheckpointIdentity(
            checkpoint_id=checkpoint_id,
            candidate_id=first.candidate_id,
            run_id=first.run_id,
            scientific_config_digest=first.scientific_config_digest,
            epoch=first.epoch,
        )
        candidates.append(
            (
                checkpoint,
                min(values),
                fmean(values),
                projection_ranks.pop(),
                tuple(record.record_id for record in selected_records),
            )
        )

    selected = min(candidates, key=_checkpoint_sort_key)
    checkpoint, objective, mean_accuracy, projection_rank, record_ids = selected
    first_record = validation_records[0]
    decision_id = f"checkpoint-selection:{selector.value}:{checkpoint.checkpoint_id}"
    return CheckpointSelectionDecision(
        decision_id=decision_id,
        selector=selector,
        method_id=first_record.method_id,
        seed_stage=first_record.seed_stage,
        seed=first_record.seed,
        checkpoint=checkpoint,
        objective_value=objective,
        mean_accuracy=mean_accuracy,
        projection_rank=projection_rank,
        contributing_record_ids=record_ids,
    )


def _rank_candidates_for_seed_keys(
    records: Sequence[SelectorRecord],
    selector: CmnistSelector,
    expected_seed_keys: set[tuple[SeedStage, int]],
) -> tuple[CandidateSelectionDecision, ...]:
    """Rank candidates after enforcing one exact configured stage/seed table."""

    validation_records = _require_selector_records(records, selector)
    methods = {record.method_id for record in validation_records}
    if len(methods) != 1:
        raise ValueError("methods must be selected independently")

    by_run: dict[tuple[str, str], list[SelectorRecord]] = defaultdict(list)
    for record in validation_records:
        by_run[(record.candidate_id, record.run_id)].append(record)

    by_candidate: dict[str, list[CheckpointSelectionDecision]] = defaultdict(list)
    for (candidate_id, _), run_records in by_run.items():
        decision = select_checkpoint(run_records, selector)
        by_candidate[candidate_id].append(decision)

    for candidate_id, decisions in by_candidate.items():
        seed_keys = [(decision.seed_stage, decision.seed) for decision in decisions]
        if len(seed_keys) != len(set(seed_keys)):
            raise ValueError("a candidate has duplicate runs for one seed stage")
        if set(seed_keys) != expected_seed_keys:
            raise ValueError(
                f"candidate {candidate_id!r} does not have exactly the required "
                "stage/seed records"
            )

    ranked: list[CandidateSelectionDecision] = []
    for candidate_id, decisions in by_candidate.items():
        decisions.sort(key=lambda decision: (decision.seed_stage.value, decision.seed))
        config_digests = {
            decision.checkpoint.scientific_config_digest for decision in decisions
        }
        projection_ranks = {decision.projection_rank for decision in decisions}
        if len(config_digests) != 1 or len(projection_ranks) != 1:
            raise ValueError(
                "candidate runs must share configuration and projection rank"
            )
        objective = fmean(float(decision.objective_value) for decision in decisions)
        mean_accuracy = fmean(float(decision.mean_accuracy) for decision in decisions)
        method_id = decisions[0].method_id
        ranked.append(
            CandidateSelectionDecision(
                decision_id=f"candidate-selection:{selector.value}:{candidate_id}",
                selector=selector,
                method_id=method_id,
                candidate_id=candidate_id,
                scientific_config_digest=config_digests.pop(),
                objective_value=objective,
                mean_accuracy=mean_accuracy,
                projection_rank=projection_ranks.pop(),
                contributing_checkpoint_decisions=tuple(decisions),
            )
        )

    return tuple(sorted(ranked, key=_candidate_sort_key))


def rank_tuning_candidates(
    records: Sequence[SelectorRecord],
    selector: CmnistSelector,
    seed_sets: SeedSets,
) -> tuple[CandidateSelectionDecision, ...]:
    """Rank one method using exactly the configured three tuning seeds."""

    validated_seed_sets = _revalidate_seed_sets(seed_sets)
    validation_records = _require_selector_records(records, selector)
    if any(record.seed_stage is not SeedStage.TUNING for record in validation_records):
        raise ValueError("tuning ranking accepts tuning-stage records only")
    expected = {(SeedStage.TUNING, seed) for seed in validated_seed_sets.tuning}
    return _rank_candidates_for_seed_keys(validation_records, selector, expected)


def make_tuning_finalists(
    records: Sequence[SelectorRecord],
    selector: CmnistSelector,
    seed_sets: SeedSets,
) -> TuningFinalistsArtifact:
    """Persist one selector's ordered top three after tuning only."""

    validated_seed_sets = _revalidate_seed_sets(seed_sets)
    ranked = rank_tuning_candidates(records, selector, validated_seed_sets)
    if len(ranked) < 3:
        raise ValueError("tuning requires at least three candidate configurations")
    top_three = ranked[:3]
    method_id = top_three[0].method_id
    return TuningFinalistsArtifact(
        artifact_id=f"tuning-finalists:{method_id}:{selector.value}",
        selector=selector,
        method_id=method_id,
        tuning_seeds=validated_seed_sets.tuning,
        ordered_candidates=top_three,
    )


def select_confirmed_candidate(
    confirmation_records: Sequence[SelectorRecord],
    finalists: TuningFinalistsArtifact,
    seed_sets: SeedSets,
) -> CandidateSelectionDecision:
    """Combine fresh confirmation records with the artifact's tuning decisions."""

    validated_seed_sets = _revalidate_seed_sets(seed_sets)
    validated_finalists = TuningFinalistsArtifact.model_validate(
        finalists.model_dump(mode="python")
    )
    if validated_finalists.tuning_seeds != validated_seed_sets.tuning:
        raise ValueError("finalist artifact tuning seeds do not match configuration")
    validation_records = _require_selector_records(
        confirmation_records, validated_finalists.selector
    )
    if any(
        record.seed_stage is not SeedStage.CONFIRMATION for record in validation_records
    ):
        raise ValueError(
            "confirmation comparison accepts confirmation-stage records only"
        )
    finalist_by_id = {
        finalist.candidate_id: finalist
        for finalist in validated_finalists.ordered_candidates
    }
    observed_candidate_ids = {record.candidate_id for record in validation_records}
    if observed_candidate_ids != set(finalist_by_id):
        raise ValueError(
            "confirmation comparison must contain exactly its selector finalists"
        )
    if any(
        record.method_id != validated_finalists.method_id
        for record in validation_records
    ):
        raise ValueError("confirmation records do not match finalist method")
    expected_confirmation = {
        (SeedStage.CONFIRMATION, seed) for seed in validated_seed_sets.confirmation
    }
    confirmation_decisions = _rank_candidates_for_seed_keys(
        validation_records,
        validated_finalists.selector,
        expected_confirmation,
    )
    combined_decisions: list[CandidateSelectionDecision] = []
    for confirmation_decision in confirmation_decisions:
        tuning_decision = finalist_by_id[confirmation_decision.candidate_id]
        if (
            confirmation_decision.scientific_config_digest
            != tuning_decision.scientific_config_digest
            or confirmation_decision.projection_rank != tuning_decision.projection_rank
        ):
            raise ValueError(
                "confirmation candidate identity does not match tuning finalist"
            )
        contributing_decisions = (
            *tuning_decision.contributing_checkpoint_decisions,
            *confirmation_decision.contributing_checkpoint_decisions,
        )
        combined_decisions.append(
            CandidateSelectionDecision(
                decision_id=(
                    "candidate-selection:"
                    f"{validated_finalists.selector.value}:"
                    f"{confirmation_decision.candidate_id}:confirmed"
                ),
                selector=validated_finalists.selector,
                method_id=validated_finalists.method_id,
                candidate_id=confirmation_decision.candidate_id,
                scientific_config_digest=tuning_decision.scientific_config_digest,
                objective_value=fmean(
                    float(decision.objective_value)
                    for decision in contributing_decisions
                ),
                mean_accuracy=fmean(
                    float(decision.mean_accuracy) for decision in contributing_decisions
                ),
                projection_rank=tuning_decision.projection_rank,
                contributing_checkpoint_decisions=contributing_decisions,
            )
        )
    return min(combined_decisions, key=_candidate_sort_key)


def make_finalist_union(
    primary: TuningFinalistsArtifact,
    secondary: TuningFinalistsArtifact,
) -> FinalistUnion:
    """Return one ordered confirmation union without merging selector artifacts."""

    validated_primary = TuningFinalistsArtifact.model_validate(
        primary.model_dump(mode="python")
    )
    validated_secondary = TuningFinalistsArtifact.model_validate(
        secondary.model_dump(mode="python")
    )
    primary_ids = tuple(
        item.candidate_id for item in validated_primary.ordered_candidates
    )
    secondary_ids = tuple(
        item.candidate_id for item in validated_secondary.ordered_candidates
    )
    union_ids = tuple(dict.fromkeys(primary_ids + secondary_ids))
    return FinalistUnion(
        method_id=validated_primary.method_id,
        primary=validated_primary,
        secondary=validated_secondary,
        confirmation_candidate_ids=union_ids,
    )


def freeze_candidate(
    decision: CandidateSelectionDecision,
    finalists: TuningFinalistsArtifact,
    seed_sets: SeedSets,
) -> FrozenCandidateSelection:
    """Freeze one confirmed winner through its selector-specific finalists."""

    validated_seed_sets = _revalidate_seed_sets(seed_sets)
    validated_decision = CandidateSelectionDecision.model_validate(
        decision.model_dump(mode="python")
    )
    validated_finalists = TuningFinalistsArtifact.model_validate(
        finalists.model_dump(mode="python")
    )
    observed_seed_keys = {
        (checkpoint.seed_stage, checkpoint.seed)
        for checkpoint in validated_decision.contributing_checkpoint_decisions
    }
    expected_seed_keys = {
        *((SeedStage.TUNING, seed) for seed in validated_seed_sets.tuning),
        *((SeedStage.CONFIRMATION, seed) for seed in validated_seed_sets.confirmation),
    }
    if observed_seed_keys != expected_seed_keys:
        raise ValueError(
            "candidate freeze requires exactly the configured tuning and "
            "confirmation seeds"
        )
    finalist_by_id = {
        finalist.candidate_id: finalist
        for finalist in validated_finalists.ordered_candidates
    }
    tuning_decision = finalist_by_id.get(validated_decision.candidate_id)
    if tuning_decision is None:
        raise ValueError("candidate cannot be frozen outside its finalist artifact")
    if validated_decision.selector is not validated_finalists.selector:
        raise ValueError("candidate selector does not match finalist artifact")
    if validated_decision.method_id != validated_finalists.method_id:
        raise ValueError("candidate method does not match finalist artifact")
    if (
        validated_decision.scientific_config_digest
        != tuning_decision.scientific_config_digest
        or validated_decision.projection_rank != tuning_decision.projection_rank
    ):
        raise ValueError("candidate identity does not match tuning finalist")
    return FrozenCandidateSelection(
        frozen_selection_id=f"frozen:{validated_decision.decision_id}",
        selector=validated_decision.selector,
        method_id=validated_decision.method_id,
        candidate_id=validated_decision.candidate_id,
        scientific_config_digest=validated_decision.scientific_config_digest,
        seed_sets=validated_seed_sets,
        finalists=validated_finalists,
        decision=validated_decision,
    )


def freeze_final_checkpoint(
    decision: CheckpointSelectionDecision,
    candidate: FrozenCandidateSelection,
) -> FrozenCheckpointSelection:
    """Freeze a final run's checkpoint without allowing candidate mutation."""

    validated_decision = CheckpointSelectionDecision.model_validate(
        decision.model_dump(mode="python")
    )
    validated_candidate = FrozenCandidateSelection.model_validate(
        candidate.model_dump(mode="python")
    )
    if validated_decision.seed_stage is not SeedStage.FINAL:
        raise ValueError("a frozen final checkpoint requires final-stage validation")
    if validated_decision.selector is not validated_candidate.selector:
        raise ValueError("checkpoint selector does not match the frozen candidate")
    if validated_decision.method_id != validated_candidate.method_id:
        raise ValueError("checkpoint method does not match the frozen candidate")
    if validated_decision.seed not in validated_candidate.seed_sets.final:
        raise ValueError("final checkpoint seed is not configured for final evaluation")
    if validated_decision.checkpoint.candidate_id != validated_candidate.candidate_id:
        raise ValueError("final-stage validation cannot change the frozen candidate")
    if (
        validated_decision.checkpoint.scientific_config_digest
        != validated_candidate.scientific_config_digest
    ):
        raise ValueError("final-stage validation cannot change frozen hyperparameters")
    return FrozenCheckpointSelection(
        frozen_checkpoint_id=f"frozen:{validated_decision.decision_id}",
        candidate_selection_id=validated_candidate.frozen_selection_id,
        selector=validated_candidate.selector,
        method_id=validated_candidate.method_id,
        checkpoint=validated_decision.checkpoint,
        decision=validated_decision,
    )
