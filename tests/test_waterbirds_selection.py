"""Regression tests for Waterbirds four-group validation selection."""

from __future__ import annotations

import json
from collections.abc import Sequence
from typing import Literal, cast

import pytest
import torch
from pydantic import ValidationError

from grit.config import SeedSets
from grit.data.waterbirds import (
    WATERBIRDS_GROUP_ORDER,
    WaterbirdsAdjustedWeightSpec,
    WaterbirdsGroupCounts,
)
from grit.features.waterbirds import WaterbirdsEvaluationFeatureTable
from grit.schemas import SeedStage
from grit.selection.waterbirds import (
    GROUP_ORDER,
    WaterbirdsFinalTestMetricRecord,
    WaterbirdsGroupAccuracy,
    WaterbirdsValidationMetricRecord,
    compute_waterbirds_validation_metric,
    freeze_waterbirds_candidate,
    freeze_waterbirds_final_checkpoint,
    make_waterbirds_tuning_finalists,
    rank_waterbirds_tuning_candidates,
    select_confirmed_waterbirds_candidate,
    select_waterbirds_checkpoint,
)


def _seed_sets() -> SeedSets:
    return SeedSets(
        tuning=(101, 102, 103),
        confirmation=(201, 202),
        final=(301, 302, 303, 304, 305, 306, 307, 308, 309, 310),
    )


def _training_counts() -> WaterbirdsGroupCounts:
    return WaterbirdsGroupCounts(
        landbird_land=4,
        landbird_water=2,
        waterbird_land=1,
        waterbird_water=3,
    )


def _weight_spec(
    dataset_manifest_digest: str = "dataset:fixture",
) -> WaterbirdsAdjustedWeightSpec:
    return WaterbirdsAdjustedWeightSpec(
        schema_version="grit.waterbirds-adjusted-weights/v1",
        group_order=WATERBIRDS_GROUP_ORDER,
        training_group_counts=_training_counts(),
        dataset_manifest_digest=dataset_manifest_digest,
    )


def _record(
    *,
    candidate: str,
    seed_stage: SeedStage,
    seed: int,
    epoch: int = 1,
    correct: tuple[int, int, int, int] = (8, 8, 8, 8),
    method: Literal["erm", "grit"] = "erm",
    rank: int | None = None,
    dataset_manifest_digest: str = "dataset:fixture",
    feature_cache_manifest_digest: str = "features:fixture",
    normalization: Literal["none", "l2"] = "none",
    weight_spec: WaterbirdsAdjustedWeightSpec | None = None,
) -> WaterbirdsValidationMetricRecord:
    groups = tuple(
        WaterbirdsGroupAccuracy(
            group_id=group,
            count=10,
            correct=value,
            accuracy=value / 10,
        )
        for group, value in zip(GROUP_ORDER, correct, strict=True)
    )
    typed_groups = (groups[0], groups[1], groups[2], groups[3])
    adjusted_weights = weight_spec or _weight_spec(dataset_manifest_digest)
    weights = adjusted_weights.training_group_counts.as_tuple()
    accuracies = tuple(value / 10 for value in correct)
    adjusted = sum(
        accuracy * count for accuracy, count in zip(accuracies, weights, strict=True)
    ) / sum(weights)
    raw = sum(correct) / 40
    checkpoint = f"checkpoint:{candidate}:{seed_stage.value}:{seed}:{epoch}"
    run = f"run:{candidate}:{seed_stage.value}:{seed}"
    return WaterbirdsValidationMetricRecord(
        record_id=f"metric:{checkpoint}",
        run_id=run,
        candidate_id=candidate,
        method_id=method,
        scientific_config_digest=f"config:{candidate}",
        dataset_manifest_digest=dataset_manifest_digest,
        feature_cache_manifest_digest=feature_cache_manifest_digest,
        normalization=normalization,
        adjusted_weight_spec_digest=adjusted_weights.canonical_digest(),
        checkpoint_id=checkpoint,
        epoch=epoch,
        seed=seed,
        projection_rank=rank,
        groups=typed_groups,
        adjusted_weight_spec=adjusted_weights,
        worst_group_accuracy=min(accuracies),
        adjusted_average_accuracy=adjusted,
        raw_average_accuracy=raw,
        metric_kind="validation",
        split_name="validation",
        seed_stage=seed_stage,
    )


def _records_for_candidates(
    candidates: Sequence[tuple[str, tuple[int, int, int, int], int | None]],
    *,
    stage: SeedStage,
    seeds: tuple[int, ...],
    method: Literal["erm", "grit"] = "erm",
) -> tuple[WaterbirdsValidationMetricRecord, ...]:
    return tuple(
        _record(
            candidate=candidate,
            seed_stage=stage,
            seed=seed,
            correct=correct,
            method=method,
            rank=rank,
        )
        for candidate, correct, rank in candidates
        for seed in seeds
    )


def test_group_metric_uses_training_proportions_not_validation_counts() -> None:
    table = WaterbirdsEvaluationFeatureTable(
        dataset_manifest_digest="dataset:fixture",
        feature_cache_manifest_digest="features:fixture",
        normalization="none",
        split_role="validation",
        record_ids=("a", "b", "c", "d"),
        features=torch.zeros((4, 512)),
        labels=torch.tensor([0, 0, 1, 1]),
        backgrounds=torch.tensor([0, 1, 0, 1]),
        group_ids=GROUP_ORDER,
    )
    metric = compute_waterbirds_validation_metric(
        table,
        torch.tensor([0, 0, 0, 1]),
        adjusted_weights=_weight_spec(),
        record_id="metric:one",
        run_id="run:one",
        candidate_id="candidate:one",
        method_id="erm",
        scientific_config_digest="config:one",
        checkpoint_id="checkpoint:one",
        epoch=1,
        seed_stage=SeedStage.TUNING,
        seed=101,
        projection_rank=None,
    )
    assert metric.worst_group_accuracy == 0.0
    assert metric.raw_average_accuracy == 0.75
    assert metric.adjusted_average_accuracy == 0.9
    assert metric.adjusted_weight_spec.training_group_counts == _training_counts()


def test_group_metric_rejects_missing_group() -> None:
    table = WaterbirdsEvaluationFeatureTable(
        dataset_manifest_digest="dataset:fixture",
        feature_cache_manifest_digest="features:fixture",
        normalization="none",
        split_role="validation",
        record_ids=("a", "b", "c"),
        features=torch.zeros((3, 512)),
        labels=torch.tensor([0, 0, 1]),
        backgrounds=torch.tensor([0, 1, 0]),
        group_ids=GROUP_ORDER[:3],
    )
    with pytest.raises(ValueError, match="missing group"):
        compute_waterbirds_validation_metric(
            table,
            torch.tensor([0, 0, 1]),
            adjusted_weights=_weight_spec(),
            record_id="metric:missing",
            run_id="run:missing",
            candidate_id="candidate:missing",
            method_id="erm",
            scientific_config_digest="config:missing",
            checkpoint_id="checkpoint:missing",
            epoch=1,
            seed_stage=SeedStage.TUNING,
            seed=101,
            projection_rank=None,
        )


def test_group_metric_rejects_adjusted_weights_from_another_dataset() -> None:
    table = WaterbirdsEvaluationFeatureTable(
        dataset_manifest_digest="dataset:fixture",
        feature_cache_manifest_digest="features:fixture",
        normalization="none",
        split_role="validation",
        record_ids=("a", "b", "c", "d"),
        features=torch.zeros((4, 512)),
        labels=torch.tensor([0, 0, 1, 1]),
        backgrounds=torch.tensor([0, 1, 0, 1]),
        group_ids=GROUP_ORDER,
    )
    with pytest.raises(ValueError, match="another dataset"):
        compute_waterbirds_validation_metric(
            table,
            torch.tensor([0, 0, 1, 1]),
            adjusted_weights=_weight_spec("dataset:other"),
            record_id="metric:cross-dataset-weights",
            run_id="run:one",
            candidate_id="candidate:one",
            method_id="erm",
            scientific_config_digest="config:one",
            checkpoint_id="checkpoint:one",
            epoch=1,
            seed_stage=SeedStage.TUNING,
            seed=101,
            projection_rank=None,
        )


def test_checkpoint_ties_use_adjusted_average_then_earlier_epoch() -> None:
    adjusted_low = _record(
        candidate="a",
        seed_stage=SeedStage.TUNING,
        seed=101,
        epoch=1,
        correct=(7, 7, 7, 7),
    )
    adjusted_high_late = _record(
        candidate="a",
        seed_stage=SeedStage.TUNING,
        seed=101,
        epoch=3,
        correct=(9, 8, 7, 8),
    )
    adjusted_high_early = _record(
        candidate="a",
        seed_stage=SeedStage.TUNING,
        seed=101,
        epoch=2,
        correct=(9, 8, 7, 8),
    )
    selected = select_waterbirds_checkpoint(
        (adjusted_low, adjusted_high_late, adjusted_high_early)
    )
    assert selected.checkpoint.epoch == 2
    assert selected.contributing_record_id == adjusted_high_early.record_id


def test_tuning_and_confirmation_enforce_exact_stages_and_finalists() -> None:
    candidates = (
        ("a", (9, 9, 9, 9), None),
        ("b", (8, 8, 8, 8), None),
        ("c", (7, 7, 7, 7), None),
        ("d", (6, 6, 6, 6), None),
    )
    tuning = _records_for_candidates(
        candidates,
        stage=SeedStage.TUNING,
        seeds=_seed_sets().tuning,
    )
    finalists = make_waterbirds_tuning_finalists(tuning, _seed_sets())
    assert tuple(item.candidate_id for item in finalists.ordered_candidates) == (
        "a",
        "b",
        "c",
    )
    confirmation = _records_for_candidates(
        candidates[:3],
        stage=SeedStage.CONFIRMATION,
        seeds=_seed_sets().confirmation,
    )
    decision = select_confirmed_waterbirds_candidate(
        confirmation, finalists, _seed_sets()
    )
    assert decision.candidate_id == "a"
    assert decision.checkpoint_decisions[:3] == (
        finalists.ordered_candidates[0].checkpoint_decisions
    )
    frozen = freeze_waterbirds_candidate(decision, finalists, _seed_sets())
    assert frozen.candidate_id == "a"

    with pytest.raises(ValueError, match="exact stage/seed"):
        rank_waterbirds_tuning_candidates(tuning[:-1], _seed_sets())
    premature = _records_for_candidates(
        candidates,
        stage=SeedStage.CONFIRMATION,
        seeds=_seed_sets().tuning,
    )
    with pytest.raises(ValueError, match="tuning records only"):
        rank_waterbirds_tuning_candidates(premature, _seed_sets())
    with pytest.raises(ValueError, match="exactly its finalists"):
        select_confirmed_waterbirds_candidate(
            (
                *confirmation,
                _record(
                    candidate="d",
                    seed_stage=SeedStage.CONFIRMATION,
                    seed=201,
                ),
            ),
            finalists,
            _seed_sets(),
        )
    with pytest.raises(ValueError, match="exact stage/seed"):
        select_confirmed_waterbirds_candidate(
            confirmation[:-1], finalists, _seed_sets()
        )
    extra_seed = tuple(
        _record(
            candidate=candidate,
            seed_stage=SeedStage.CONFIRMATION,
            seed=999,
            correct=correct,
        )
        for candidate, correct, _ in candidates[:3]
    )
    with pytest.raises(ValueError, match="exact stage/seed"):
        select_confirmed_waterbirds_candidate(
            (*confirmation, *extra_seed), finalists, _seed_sets()
        )

    final_records = (
        _record(
            candidate="a",
            seed_stage=SeedStage.FINAL,
            seed=301,
            epoch=1,
            correct=(7, 7, 7, 7),
        ),
        _record(
            candidate="a",
            seed_stage=SeedStage.FINAL,
            seed=301,
            epoch=2,
            correct=(9, 9, 8, 9),
        ),
    )
    checkpoint = select_waterbirds_checkpoint(final_records)
    frozen_checkpoint = freeze_waterbirds_final_checkpoint(checkpoint, frozen)
    assert frozen_checkpoint.checkpoint.epoch == 2


def test_candidate_tie_uses_lower_projection_rank_and_methods_are_independent() -> None:
    candidates = (
        ("rank-two", (8, 8, 8, 8), 2),
        ("rank-one", (8, 8, 8, 8), 1),
        ("rank-three", (7, 7, 7, 7), 3),
    )
    records = _records_for_candidates(
        candidates,
        stage=SeedStage.TUNING,
        seeds=_seed_sets().tuning,
        method="grit",
    )
    ranked = rank_waterbirds_tuning_candidates(records, _seed_sets())
    assert ranked[0].candidate_id == "rank-one"

    mixed = (
        *records,
        _record(
            candidate="erm",
            seed_stage=SeedStage.TUNING,
            seed=101,
        ),
    )
    with pytest.raises(ValueError, match="independently"):
        rank_waterbirds_tuning_candidates(mixed, _seed_sets())


def test_ordinary_selector_rejects_final_test_metric_type() -> None:
    validation = _record(
        candidate="a",
        seed_stage=SeedStage.FINAL,
        seed=301,
    )
    payload = validation.model_dump(mode="python")
    payload["metric_kind"] = "final_test"
    payload["split_name"] = "test"
    final = WaterbirdsFinalTestMetricRecord.model_validate(payload)
    with pytest.raises(TypeError, match="validation metrics only"):
        select_waterbirds_checkpoint((cast(WaterbirdsValidationMetricRecord, final),))


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("dataset_manifest_digest", "dataset:other"),
        ("feature_cache_manifest_digest", "features:other"),
        ("normalization", "l2"),
    ),
)
def test_checkpoint_selection_rejects_cross_artifact_metric_mixing(
    field: str,
    value: str,
) -> None:
    base = _record(candidate="a", seed_stage=SeedStage.TUNING, seed=101, epoch=1)
    if field == "dataset_manifest_digest":
        changed = _record(
            candidate="a",
            seed_stage=SeedStage.TUNING,
            seed=101,
            epoch=2,
            dataset_manifest_digest=value,
            weight_spec=_weight_spec(value),
        )
    elif field == "feature_cache_manifest_digest":
        changed = _record(
            candidate="a",
            seed_stage=SeedStage.TUNING,
            seed=101,
            epoch=2,
            feature_cache_manifest_digest=value,
        )
    else:
        changed = _record(
            candidate="a",
            seed_stage=SeedStage.TUNING,
            seed=101,
            epoch=2,
            normalization=cast(Literal["none", "l2"], value),
        )
    with pytest.raises(ValueError, match="one candidate run"):
        select_waterbirds_checkpoint((base, changed))


def test_checkpoint_selection_rejects_cross_weight_metric_mixing() -> None:
    base = _record(candidate="a", seed_stage=SeedStage.TUNING, seed=101, epoch=1)
    changed_weights = WaterbirdsAdjustedWeightSpec(
        schema_version="grit.waterbirds-adjusted-weights/v1",
        group_order=WATERBIRDS_GROUP_ORDER,
        training_group_counts=WaterbirdsGroupCounts(
            landbird_land=3,
            landbird_water=2,
            waterbird_land=1,
            waterbird_water=4,
        ),
        dataset_manifest_digest="dataset:fixture",
    )
    changed = _record(
        candidate="a",
        seed_stage=SeedStage.TUNING,
        seed=101,
        epoch=2,
        weight_spec=changed_weights,
    )
    with pytest.raises(ValueError, match="one candidate run"):
        select_waterbirds_checkpoint((base, changed))


def test_tuning_and_confirmation_reject_lineage_changes() -> None:
    candidates = (
        ("a", (9, 9, 9, 9), None),
        ("b", (8, 8, 8, 8), None),
        ("c", (7, 7, 7, 7), None),
    )
    tuning = _records_for_candidates(
        candidates, stage=SeedStage.TUNING, seeds=_seed_sets().tuning
    )
    with pytest.raises(ValueError, match="artifact lineage"):
        make_waterbirds_tuning_finalists(
            (
                *tuning[:-1],
                _record(
                    candidate="c",
                    seed_stage=SeedStage.TUNING,
                    seed=103,
                    correct=(7, 7, 7, 7),
                    feature_cache_manifest_digest="features:other",
                ),
            ),
            _seed_sets(),
        )

    finalists = make_waterbirds_tuning_finalists(tuning, _seed_sets())
    serialized = json.loads(finalists.canonical_json())
    serialized["feature_cache_manifest_digest"] = "features:other"
    with pytest.raises(ValidationError, match="finalist artifact lineage"):
        type(finalists).model_validate_json(json.dumps(serialized))
    changed_confirmation = tuple(
        _record(
            candidate=candidate,
            seed_stage=SeedStage.CONFIRMATION,
            seed=seed,
            correct=correct,
            feature_cache_manifest_digest="features:other",
        )
        for candidate, correct, _ in candidates
        for seed in _seed_sets().confirmation
    )
    with pytest.raises(ValueError, match="identity changed"):
        select_confirmed_waterbirds_candidate(
            changed_confirmation, finalists, _seed_sets()
        )
