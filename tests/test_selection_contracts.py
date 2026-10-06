"""Validation-only CMNIST selection and deterministic tie tests."""

from __future__ import annotations

from collections.abc import Sequence
from typing import cast

import pytest
from pydantic import ValidationError

from grit.schemas import CmnistSelector, SeedStage
from grit.selection.cmnist import (
    FinalTestMetricRecord,
    TuningFinalistsArtifact,
    ValidationMetricRecord,
    freeze_candidate,
    freeze_final_checkpoint,
    make_finalist_union,
    make_tuning_finalists,
    rank_tuning_candidates,
    select_checkpoint,
    select_confirmed_candidate,
)
from tests.contract_fixtures import ordinary_grit_config, seed_sets, validation_records


def _candidate_records(
    candidate_id: str,
    scores: tuple[float, float, float],
    *,
    rank: int,
    scientific_config_digest: str,
    seeds: tuple[int, ...] = (101, 102, 103),
    seed_stage: SeedStage = SeedStage.TUNING,
) -> tuple[ValidationMetricRecord, ...]:
    records: list[ValidationMetricRecord] = []
    for seed in seeds:
        run_id = f"run:{candidate_id}:{seed}"
        records.extend(
            validation_records(
                candidate_id=candidate_id,
                scientific_config_digest=scientific_config_digest,
                run_id=run_id,
                seed_stage=seed_stage,
                seed=seed,
                checkpoint_id=f"checkpoint:{candidate_id}:{seed}",
                epoch=3,
                scores=scores,
                projection_rank=rank,
                method_id="grit",
            )
        )
    return tuple(records)


def _stage_records(
    candidates: tuple[tuple[str, tuple[float, float, float]], ...],
    *,
    seed_stage: SeedStage,
    seeds: tuple[int, ...],
) -> tuple[ValidationMetricRecord, ...]:
    return tuple(
        record
        for candidate_id, scores in candidates
        for record in _candidate_records(
            candidate_id,
            scores,
            rank=2,
            scientific_config_digest=f"sha256:{candidate_id}",
            seeds=seeds,
            seed_stage=seed_stage,
        )
    )


def test_validation_selectors_reject_final_and_diagnostic_metric_types() -> None:
    final_metric = FinalTestMetricRecord(
        record_id="metric:final",
        run_id="run:final",
        candidate_id="candidate:a",
        method_id="erm",
        scientific_config_digest="sha256:config",
        checkpoint_id="checkpoint:a",
        epoch=1,
        seed=301,
        value=0.8,
        sample_count=10,
        metric_kind="final_test",
        seed_stage=SeedStage.FINAL,
        split_name="test_ood",
        metric_name="accuracy",
    )
    invalid_final = cast(
        Sequence[ValidationMetricRecord], cast(object, (final_metric,))
    )
    with pytest.raises(TypeError, match="ValidationMetricRecord"):
        select_checkpoint(invalid_final, CmnistSelector.PRIMARY_ROBUST)
    with pytest.raises(TypeError, match="ValidationMetricRecord"):
        rank_tuning_candidates(
            invalid_final,
            CmnistSelector.PRIMARY_ROBUST,
            seed_sets(),
        )


def test_public_selector_revalidates_model_copy_inputs() -> None:
    records = validation_records(
        candidate_id="candidate:a",
        scientific_config_digest="sha256:a",
        run_id="run:a",
        seed_stage=SeedStage.TUNING,
        seed=101,
        checkpoint_id="checkpoint:a",
        epoch=1,
        scores=(0.7, 0.7, 0.7),
        projection_rank=1,
        method_id="grit",
    )
    malformed = records[0].model_copy(update={"split_name": "test_ood"})
    malformed_records = cast(
        Sequence[ValidationMetricRecord],
        cast(object, (malformed, *records[1:])),
    )
    with pytest.raises(ValidationError, match="split_name"):
        select_checkpoint(malformed_records, CmnistSelector.PRIMARY_ROBUST)


def test_primary_and_secondary_selectors_apply_distinct_prespecified_splits() -> None:
    records = (
        *_candidate_records(
            "candidate:source-strong",
            (0.9, 0.9, 0.4),
            rank=2,
            scientific_config_digest="sha256:source-strong",
        ),
        *_candidate_records(
            "candidate:robust",
            (0.7, 0.7, 0.7),
            rank=2,
            scientific_config_digest="sha256:robust",
        ),
    )
    primary = rank_tuning_candidates(
        records, CmnistSelector.PRIMARY_ROBUST, seed_sets()
    )[0]
    secondary = rank_tuning_candidates(
        records, CmnistSelector.SECONDARY_SOURCE, seed_sets()
    )[0]
    assert primary.candidate_id == "candidate:robust"
    assert secondary.candidate_id == "candidate:source-strong"


def test_candidate_ties_use_mean_rank_then_stable_identity_deterministically() -> None:
    mean_tie_records = (
        *_candidate_records(
            "candidate:high-mean",
            (0.5, 0.9, 0.7),
            rank=3,
            scientific_config_digest="sha256:high-mean",
        ),
        *_candidate_records(
            "candidate:low-mean",
            (0.5, 0.5, 0.5),
            rank=1,
            scientific_config_digest="sha256:low-mean",
        ),
    )
    assert (
        rank_tuning_candidates(
            mean_tie_records, CmnistSelector.PRIMARY_ROBUST, seed_sets()
        )[0].candidate_id
        == "candidate:high-mean"
    )

    rank_tie_records = (
        *_candidate_records(
            "candidate:rank-two",
            (0.6, 0.6, 0.6),
            rank=2,
            scientific_config_digest="sha256:rank-two",
        ),
        *_candidate_records(
            "candidate:rank-one",
            (0.6, 0.6, 0.6),
            rank=1,
            scientific_config_digest="sha256:rank-one",
        ),
    )
    assert (
        rank_tuning_candidates(
            rank_tie_records, CmnistSelector.PRIMARY_ROBUST, seed_sets()
        )[0].candidate_id
        == "candidate:rank-one"
    )

    identity_tie_records = (
        *_candidate_records(
            "candidate:b",
            (0.6, 0.6, 0.6),
            rank=1,
            scientific_config_digest="sha256:b",
        ),
        *_candidate_records(
            "candidate:a",
            (0.6, 0.6, 0.6),
            rank=1,
            scientific_config_digest="sha256:a",
        ),
    )
    forward = rank_tuning_candidates(
        identity_tie_records, CmnistSelector.PRIMARY_ROBUST, seed_sets()
    )[0]
    reverse = rank_tuning_candidates(
        tuple(reversed(identity_tie_records)),
        CmnistSelector.PRIMARY_ROBUST,
        seed_sets(),
    )[0]
    assert forward.candidate_id == reverse.candidate_id == "candidate:a"


def test_checkpoint_ties_prefer_earlier_epoch_then_stable_identity() -> None:
    scientific_config_digest = ordinary_grit_config().scientific_config_digest()
    records = (
        *validation_records(
            candidate_id="candidate:a",
            scientific_config_digest=scientific_config_digest,
            run_id="run:final",
            seed_stage=SeedStage.FINAL,
            seed=301,
            checkpoint_id="checkpoint:later",
            epoch=2,
            scores=(0.7, 0.7, 0.7),
            projection_rank=2,
            method_id="grit",
        ),
        *validation_records(
            candidate_id="candidate:a",
            scientific_config_digest=scientific_config_digest,
            run_id="run:final",
            seed_stage=SeedStage.FINAL,
            seed=301,
            checkpoint_id="checkpoint:earlier",
            epoch=1,
            scores=(0.7, 0.7, 0.7),
            projection_rank=2,
            method_id="grit",
        ),
    )
    decision = select_checkpoint(records, CmnistSelector.PRIMARY_ROBUST)
    assert decision.checkpoint.checkpoint_id == "checkpoint:earlier"

    same_epoch = tuple(
        record.model_copy(
            update={
                "checkpoint_id": "checkpoint:a"
                if record.checkpoint_id == "checkpoint:earlier"
                else "checkpoint:b",
                "epoch": 1,
                "record_id": record.record_id.replace(
                    record.checkpoint_id,
                    "checkpoint:a"
                    if record.checkpoint_id == "checkpoint:earlier"
                    else "checkpoint:b",
                ),
            }
        )
        for record in records
    )
    stable = select_checkpoint(same_epoch, CmnistSelector.PRIMARY_ROBUST)
    assert stable.checkpoint.checkpoint_id == "checkpoint:a"


def test_checkpoint_selection_cannot_change_candidate_rank() -> None:
    scientific_config_digest = ordinary_grit_config().scientific_config_digest()
    records = (
        *validation_records(
            candidate_id="candidate:a",
            scientific_config_digest=scientific_config_digest,
            run_id="run:final",
            seed_stage=SeedStage.FINAL,
            seed=301,
            checkpoint_id="checkpoint:rank-one",
            epoch=1,
            scores=(0.7, 0.7, 0.7),
            projection_rank=1,
            method_id="grit",
        ),
        *validation_records(
            candidate_id="candidate:a",
            scientific_config_digest=scientific_config_digest,
            run_id="run:final",
            seed_stage=SeedStage.FINAL,
            seed=301,
            checkpoint_id="checkpoint:rank-two",
            epoch=2,
            scores=(0.8, 0.8, 0.8),
            projection_rank=2,
            method_id="grit",
        ),
    )
    with pytest.raises(ValueError, match="cannot change projection rank"):
        select_checkpoint(records, CmnistSelector.PRIMARY_ROBUST)


def test_candidate_selection_rejects_duplicate_seed_runs() -> None:
    records = (
        *_candidate_records(
            "candidate:a",
            (0.7, 0.7, 0.7),
            rank=1,
            scientific_config_digest="sha256:a",
        ),
        *tuple(
            record.model_copy(
                update={
                    "run_id": f"{record.run_id}:duplicate",
                    "record_id": f"{record.record_id}:duplicate",
                }
            )
            for record in _candidate_records(
                "candidate:a",
                (0.9, 0.9, 0.9),
                rank=1,
                scientific_config_digest="sha256:a",
            )
        ),
    )
    with pytest.raises(ValueError, match="duplicate runs"):
        rank_tuning_candidates(
            records,
            CmnistSelector.PRIMARY_ROBUST,
            seed_sets(),
        )


def test_tuning_requires_exact_configured_seeds_and_rejects_other_stages() -> None:
    one_seed = _candidate_records(
        "candidate:a",
        (0.7, 0.7, 0.7),
        rank=2,
        scientific_config_digest="sha256:a",
        seeds=(101,),
    )
    with pytest.raises(ValueError, match="exactly the required"):
        rank_tuning_candidates(
            one_seed,
            CmnistSelector.PRIMARY_ROBUST,
            seed_sets(),
        )

    premature_confirmation = _candidate_records(
        "candidate:a",
        (0.7, 0.7, 0.7),
        rank=2,
        scientific_config_digest="sha256:a",
        seeds=(201,),
        seed_stage=SeedStage.CONFIRMATION,
    )
    with pytest.raises(ValueError, match="tuning-stage"):
        rank_tuning_candidates(
            (
                *_candidate_records(
                    "candidate:a",
                    (0.7, 0.7, 0.7),
                    rank=2,
                    scientific_config_digest="sha256:a",
                ),
                *premature_confirmation,
            ),
            CmnistSelector.PRIMARY_ROBUST,
            seed_sets(),
        )


def test_connected_finalists_confirmation_and_freeze_protocol() -> None:
    candidates = (
        ("candidate:a", (0.9, 0.9, 0.4)),
        ("candidate:b", (0.8, 0.8, 0.8)),
        ("candidate:c", (0.7, 0.7, 0.7)),
        ("candidate:d", (0.6, 0.6, 0.6)),
    )
    seeds = seed_sets()
    tuning = _stage_records(
        candidates,
        seed_stage=SeedStage.TUNING,
        seeds=seeds.tuning,
    )
    primary = make_tuning_finalists(tuning, CmnistSelector.PRIMARY_ROBUST, seeds)
    primary_ids = tuple(
        decision.candidate_id for decision in primary.ordered_candidates
    )
    assert primary_ids == ("candidate:b", "candidate:c", "candidate:d")
    tuning_for_primary = tuple(
        record for record in tuning if record.candidate_id in primary_ids
    )
    confirmation = _stage_records(
        tuple(item for item in candidates if item[0] in primary_ids),
        seed_stage=SeedStage.CONFIRMATION,
        seeds=seeds.confirmation,
    )
    decision = select_confirmed_candidate(
        confirmation,
        primary,
        seeds,
    )
    stored_tuning = next(
        finalist
        for finalist in primary.ordered_candidates
        if finalist.candidate_id == decision.candidate_id
    )
    assert decision.contributing_checkpoint_decisions[:3] == (
        stored_tuning.contributing_checkpoint_decisions
    )
    candidate = freeze_candidate(decision, primary, seeds)
    assert candidate.candidate_id in primary_ids

    with pytest.raises(ValueError, match="confirmation-stage"):
        select_confirmed_candidate(
            (*tuning_for_primary, *confirmation),
            primary,
            seeds,
        )
    with pytest.raises(ValueError, match="exactly the required"):
        select_confirmed_candidate(
            confirmation[:-3],
            primary,
            seeds,
        )
    extra_confirmation = _candidate_records(
        primary_ids[0],
        (0.8, 0.8, 0.8),
        rank=2,
        scientific_config_digest=f"sha256:{primary_ids[0]}",
        seeds=(999,),
        seed_stage=SeedStage.CONFIRMATION,
    )
    with pytest.raises(ValueError, match="exactly the required"):
        select_confirmed_candidate(
            (*confirmation, *extra_confirmation),
            primary,
            seeds,
        )
    duplicate_confirmation = tuple(
        record.model_copy(
            update={
                "record_id": f"{record.record_id}:duplicate",
                "run_id": f"{record.run_id}:duplicate",
            }
        )
        for record in confirmation[:3]
    )
    with pytest.raises(ValueError, match="duplicate runs"):
        select_confirmed_candidate(
            (*confirmation, *duplicate_confirmation),
            primary,
            seeds,
        )

    nonfinalist_candidates = (
        ("candidate:x", (0.95, 0.95, 0.95)),
        ("candidate:y", (0.85, 0.85, 0.85)),
        ("candidate:z", (0.75, 0.75, 0.75)),
    )
    other_tuning = _stage_records(
        nonfinalist_candidates,
        seed_stage=SeedStage.TUNING,
        seeds=seeds.tuning,
    )
    other_artifact = make_tuning_finalists(
        other_tuning, CmnistSelector.PRIMARY_ROBUST, seeds
    )
    other_confirmation = _stage_records(
        nonfinalist_candidates,
        seed_stage=SeedStage.CONFIRMATION,
        seeds=seeds.confirmation,
    )
    outside_decision = select_confirmed_candidate(
        other_confirmation, other_artifact, seeds
    )
    with pytest.raises(ValueError, match="outside its finalist artifact"):
        freeze_candidate(outside_decision, primary, seeds)

    scientific_config_digest = candidate.scientific_config_digest
    final_a = validation_records(
        candidate_id=candidate.candidate_id,
        scientific_config_digest=scientific_config_digest,
        run_id="run:final-a",
        seed_stage=SeedStage.FINAL,
        seed=301,
        checkpoint_id="checkpoint:final-a",
        epoch=2,
        scores=(0.8, 0.8, 0.8),
        projection_rank=2,
        method_id="grit",
    )
    checkpoint = freeze_final_checkpoint(
        select_checkpoint(final_a, CmnistSelector.PRIMARY_ROBUST), candidate
    )
    assert checkpoint.candidate_selection_id == candidate.frozen_selection_id
    assert checkpoint.frozen_checkpoint_id != candidate.frozen_selection_id

    final_b = validation_records(
        candidate_id="candidate:outside",
        scientific_config_digest="sha256:different-config",
        run_id="run:final-b",
        seed_stage=SeedStage.FINAL,
        seed=301,
        checkpoint_id="checkpoint:final-b",
        epoch=2,
        scores=(0.99, 0.99, 0.99),
        projection_rank=1,
        method_id="grit",
    )
    with pytest.raises(ValueError, match="cannot change the frozen candidate"):
        freeze_final_checkpoint(
            select_checkpoint(final_b, CmnistSelector.PRIMARY_ROBUST), candidate
        )
    with pytest.raises(ValueError, match="tuning-stage"):
        rank_tuning_candidates(
            final_a,
            CmnistSelector.PRIMARY_ROBUST,
            seeds,
        )


def test_primary_secondary_finalists_form_one_union_and_keep_separate_winners() -> None:
    candidates = (
        ("candidate:a", (0.9, 0.9, 0.4)),
        ("candidate:b", (0.8, 0.8, 0.8)),
        ("candidate:c", (0.7, 0.7, 0.7)),
        ("candidate:d", (0.6, 0.6, 0.6)),
    )
    records = _stage_records(
        candidates,
        seed_stage=SeedStage.TUNING,
        seeds=seed_sets().tuning,
    )
    primary = make_tuning_finalists(records, CmnistSelector.PRIMARY_ROBUST, seed_sets())
    secondary = make_tuning_finalists(
        records, CmnistSelector.SECONDARY_SOURCE, seed_sets()
    )
    finalists = make_finalist_union(primary, secondary)
    assert (
        TuningFinalistsArtifact.model_validate_json(primary.canonical_json()) == primary
    )
    assert primary.ordered_candidates[0].candidate_id == "candidate:b"
    assert secondary.ordered_candidates[0].candidate_id == "candidate:a"
    assert finalists.confirmation_candidate_ids == (
        "candidate:b",
        "candidate:c",
        "candidate:d",
        "candidate:a",
    )

    assert len(finalists.confirmation_candidate_ids) == len(
        set(finalists.confirmation_candidate_ids)
    )

    primary_ids = {decision.candidate_id for decision in primary.ordered_candidates}
    primary_confirmation = _stage_records(
        tuple(item for item in candidates if item[0] in primary_ids),
        seed_stage=SeedStage.CONFIRMATION,
        seeds=seed_sets().confirmation,
    )
    with pytest.raises(ValueError, match="exactly its selector finalists"):
        select_confirmed_candidate(
            primary_confirmation,
            secondary,
            seed_sets(),
        )

    mixed_method = secondary.model_copy(update={"method_id": "erm"})
    with pytest.raises(ValidationError, match="method"):
        make_finalist_union(primary, mixed_method)


def test_metric_records_reject_nan_and_infinity() -> None:
    with pytest.raises(ValidationError, match="finite"):
        validation_records(
            candidate_id="candidate:a",
            scientific_config_digest="sha256:a",
            run_id="run:a",
            seed_stage=SeedStage.TUNING,
            seed=101,
            checkpoint_id="checkpoint:a",
            epoch=1,
            scores=(float("nan"), 0.5, 0.5),
            projection_rank=1,
            method_id="grit",
        )
    with pytest.raises(ValidationError, match="finite"):
        validation_records(
            candidate_id="candidate:a",
            scientific_config_digest="sha256:a",
            run_id="run:a",
            seed_stage=SeedStage.TUNING,
            seed=101,
            checkpoint_id="checkpoint:a",
            epoch=1,
            scores=(float("inf"), 0.5, 0.5),
            projection_rank=1,
            method_id="grit",
        )
