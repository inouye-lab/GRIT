"""End-to-end synthetic lifecycle, result, and null-tracking tests."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import pytest
from pydantic import ValidationError

from grit.data.views import (
    ExampleIdentity,
    FinalTestHandle,
    FinalTestSplitDescriptor,
    TrainingSplitDescriptor,
    TrainingView,
    ValidationSplitDescriptor,
    ValidationView,
    issue_final_test_handle,
    validate_cmnist_repeated_validation_views,
)
from grit.lifecycle import open_final_test, record_final_accuracy
from grit.methods.checkpoints import (
    CheckpointStore,
    RestorationReceipt,
    StoredCheckpoint,
    restore_checkpoint,
)
from grit.results import (
    ArtifactReference,
    CodeProvenance,
    EnvironmentProvenance,
    FailedStatus,
    IncompleteStatus,
    OrdinaryRunResult,
    SucceededStatus,
    parse_run_result_json,
)
from grit.schemas import CmnistSelector, SeedStage
from grit.selection.cmnist import (
    CheckpointIdentity,
    FinalTestMetricRecord,
    FrozenCandidateSelection,
    FrozenCheckpointSelection,
    ValidationMetricRecord,
    freeze_candidate,
    freeze_final_checkpoint,
    make_tuning_finalists,
    select_checkpoint,
    select_confirmed_candidate,
)
from grit.tracking import LifecycleEvent, NullEventSink
from tests.contract_fixtures import ordinary_erm_config


@dataclass
class FakeAlgorithm:
    """Owns its bounded state update and inference-state restoration."""

    state: str = "untrained"

    def update(self, example: ExampleIdentity) -> None:
        self.state = f"updated:{example.example_id}"

    def restore_inference_state(self, state: str) -> None:
        self.state = state


class FakeTrainer:
    """Owns iteration while delegating bounded mutations to the algorithm."""

    def run(self, view: TrainingView, algorithm: FakeAlgorithm) -> None:
        for example in view.examples:
            algorithm.update(example)


class FakeValidationEvaluator:
    """Consumes validation-scoped views and emits validation-only records."""

    def evaluate(
        self,
        views: tuple[ValidationView, ValidationView, ValidationView],
        *,
        candidate_id: str,
        scientific_config_digest: str,
        run_id: str,
        seed_stage: SeedStage,
        seed: int,
        checkpoint_id: str,
        epoch: int,
        scores: tuple[float, float, float],
        projection_rank: int | None,
        method_id: str = "erm",
    ) -> tuple[ValidationMetricRecord, ...]:
        validate_cmnist_repeated_validation_views(views)
        split_names: tuple[
            Literal["val_e01"], Literal["val_e02"], Literal["val_e05"]
        ] = ("val_e01", "val_e02", "val_e05")
        return tuple(
            ValidationMetricRecord(
                record_id=f"metric:{run_id}:{checkpoint_id}:{view.descriptor.name}",
                run_id=run_id,
                candidate_id=candidate_id,
                method_id=method_id,
                scientific_config_digest=scientific_config_digest,
                checkpoint_id=checkpoint_id,
                epoch=epoch,
                seed=seed,
                value=score,
                sample_count=len(view.examples),
                metric_kind="validation",
                seed_stage=seed_stage,
                split_name=split_name,
                metric_name="accuracy",
                projection_rank=projection_rank,
            )
            for view, split_name, score in zip(views, split_names, scores, strict=True)
        )


class InMemoryCheckpointStore(CheckpointStore[str]):
    """Test-only store with no persistence or resumability semantics."""

    def __init__(self) -> None:
        self._items: dict[str, StoredCheckpoint[str]] = {}

    @property
    def store_id(self) -> str:
        return "store:in-memory"

    def save(self, identity: CheckpointIdentity, state: str) -> None:
        self._items[identity.checkpoint_id] = StoredCheckpoint(identity, state)

    def load(self, checkpoint_id: str) -> StoredCheckpoint[str]:
        return self._items[checkpoint_id]


@dataclass(frozen=True)
class CompletedLifecycle:
    scientific_config_digest: str
    validation_metrics: tuple[ValidationMetricRecord, ...]
    candidate: FrozenCandidateSelection
    checkpoint: FrozenCheckpointSelection
    restoration: RestorationReceipt
    handle: FinalTestHandle
    final_metric: FinalTestMetricRecord
    algorithm_state: str


def _examples(view_id: str) -> tuple[ExampleIdentity, ...]:
    return tuple(
        ExampleIdentity(
            example_id=f"{view_id}:example:{index}",
            source_id=f"source:{index}",
            view_id=view_id,
        )
        for index in range(2)
    )


def _provenance() -> tuple[CodeProvenance, EnvironmentProvenance]:
    return (
        CodeProvenance(git_revision="synthetic-revision", git_dirty=False),
        EnvironmentProvenance(
            python_version="3.10.20",
            lock_digest="sha256:synthetic-lock",
            device="cpu",
        ),
    )


def _final_descriptor() -> FinalTestSplitDescriptor:
    return FinalTestSplitDescriptor(
        dataset_id="cmnist",
        manifest_id="manifest:synthetic",
        name="test_ood",
        role="final_test",
        source_partition_id="test_sources",
        view_id="test_ood",
    )


def _validation_views() -> tuple[ValidationView, ValidationView, ValidationView]:
    def build(
        name: Literal["val_e01", "val_e02", "val_e05"],
    ) -> ValidationView:
        return ValidationView(
            descriptor=ValidationSplitDescriptor(
                dataset_id="cmnist",
                manifest_id="manifest:synthetic",
                name=name,
                role="validation",
                source_partition_id="validation_sources",
                view_id=name,
            ),
            examples=_examples(name),
        )

    return (build("val_e01"), build("val_e02"), build("val_e05"))


def _run_completed_lifecycle() -> CompletedLifecycle:
    config = ordinary_erm_config()
    scientific_config_digest = config.scientific_config_digest()
    training = TrainingView(
        descriptor=TrainingSplitDescriptor(
            dataset_id="cmnist",
            manifest_id="manifest:synthetic",
            name="train_e01",
            role="training",
            source_partition_id="train_e01_sources",
            view_id="train_e01",
        ),
        examples=_examples("train_e01"),
    )
    algorithm = FakeAlgorithm()
    FakeTrainer().run(training, algorithm)
    assert algorithm.state == "updated:train_e01:example:1"
    validation_views = _validation_views()
    evaluator = FakeValidationEvaluator()

    tuning_records: list[ValidationMetricRecord] = []
    confirmation_records: list[ValidationMetricRecord] = []
    for candidate_id, digest, score in (
        ("candidate:selected", scientific_config_digest, 0.8),
        ("candidate:other", "sha256:other-config", 0.7),
        ("candidate:third", "sha256:third-config", 0.6),
    ):
        for seed_stage, seeds in (
            (SeedStage.TUNING, (101, 102, 103)),
            (SeedStage.CONFIRMATION, (201, 202)),
        ):
            for seed in seeds:
                target = (
                    tuning_records
                    if seed_stage is SeedStage.TUNING
                    else confirmation_records
                )
                target.extend(
                    evaluator.evaluate(
                        validation_views,
                        candidate_id=candidate_id,
                        scientific_config_digest=digest,
                        run_id=f"run:{candidate_id}:{seed}",
                        seed_stage=seed_stage,
                        seed=seed,
                        checkpoint_id=f"checkpoint:{candidate_id}:{seed}",
                        epoch=1,
                        scores=(score, score, score),
                        projection_rank=None,
                    )
                )
    finalists = make_tuning_finalists(
        tuning_records,
        CmnistSelector.PRIMARY_ROBUST,
        config.seed_sets,
    )
    combined_decision = select_confirmed_candidate(
        confirmation_records,
        finalists,
        config.seed_sets,
    )
    candidate = freeze_candidate(
        combined_decision,
        finalists,
        config.seed_sets,
    )
    assert candidate.candidate_id == "candidate:selected"

    final_validation = (
        *evaluator.evaluate(
            validation_views,
            candidate_id=candidate.candidate_id,
            scientific_config_digest=scientific_config_digest,
            run_id="run:final",
            seed_stage=SeedStage.FINAL,
            seed=301,
            checkpoint_id="checkpoint:selected-epoch",
            epoch=1,
            scores=(0.85, 0.85, 0.85),
            projection_rank=None,
        ),
        *evaluator.evaluate(
            validation_views,
            candidate_id=candidate.candidate_id,
            scientific_config_digest=scientific_config_digest,
            run_id="run:final",
            seed_stage=SeedStage.FINAL,
            seed=301,
            checkpoint_id="checkpoint:last-epoch",
            epoch=2,
            scores=(0.6, 0.6, 0.6),
            projection_rank=None,
        ),
    )
    checkpoint = freeze_final_checkpoint(
        select_checkpoint(final_validation, CmnistSelector.PRIMARY_ROBUST),
        candidate,
    )

    selected_identity = checkpoint.checkpoint
    last_identity = CheckpointIdentity(
        checkpoint_id="checkpoint:last-epoch",
        candidate_id=candidate.candidate_id,
        run_id="run:final",
        scientific_config_digest=scientific_config_digest,
        epoch=2,
    )
    store = InMemoryCheckpointStore()
    store.save(selected_identity, "state:selected-epoch")
    store.save(last_identity, "state:last-epoch")
    algorithm.state = "state:last-epoch"
    restoration = restore_checkpoint(
        checkpoint,
        store,
        algorithm.restore_inference_state,
    )
    assert algorithm.state == "state:selected-epoch"

    handle = issue_final_test_handle(
        handle_id="handle:final",
        run_id="run:final",
        candidate_id=candidate.candidate_id,
        scientific_config_digest=scientific_config_digest,
        descriptor=_final_descriptor(),
        examples=_examples("test_ood"),
    )
    final_view = open_final_test(handle, candidate, checkpoint, restoration)
    final_value = 0.91 if algorithm.state == "state:selected-epoch" else 0.09
    final_metric = record_final_accuracy(
        final_view,
        record_id="metric:final",
        value=final_value,
        sample_count=len(final_view.examples),
    )
    return CompletedLifecycle(
        scientific_config_digest=scientific_config_digest,
        validation_metrics=final_validation,
        candidate=candidate,
        checkpoint=checkpoint,
        restoration=restoration,
        handle=handle,
        final_metric=final_metric,
        algorithm_state=algorithm.state,
    )


def _ordinary_result(lifecycle: CompletedLifecycle) -> OrdinaryRunResult:
    code, environment = _provenance()
    return OrdinaryRunResult(
        schema_version="grit.run-result/v1",
        result_kind="ordinary",
        run_id="run:final",
        resolved_config=ordinary_erm_config(),
        resolved_config_digest=ordinary_erm_config().canonical_digest(),
        status=SucceededStatus(kind="succeeded"),
        code=code,
        environment=environment,
        validation_metrics=lifecycle.validation_metrics,
        candidate_selection=lifecycle.candidate,
        checkpoint_selection=lifecycle.checkpoint,
        restoration=lifecycle.restoration,
        final_test_metrics=(lifecycle.final_metric,),
        artifacts=(
            ArtifactReference(
                artifact_id="artifact:checkpoint",
                kind="checkpoint_reference",
                relative_uri="checkpoints/selected.fake",
                digest="sha256:synthetic-checkpoint",
            ),
        ),
    )


def test_in_memory_lifecycle_restores_before_final_and_round_trips_result() -> None:
    lifecycle = _run_completed_lifecycle()
    assert lifecycle.checkpoint.checkpoint.epoch == 1
    assert lifecycle.algorithm_state == "state:selected-epoch"
    assert lifecycle.final_metric.value == 0.91

    result = _ordinary_result(lifecycle)
    payload = result.canonical_json()
    reparsed = parse_run_result_json(payload)
    assert reparsed == result
    assert reparsed.canonical_json() == payload
    assert reparsed.canonical_digest() == result.canonical_digest()

    sink = NullEventSink()
    before = result.canonical_json()
    sink.emit(
        LifecycleEvent(
            event_id="event:completed",
            run_id=result.run_id,
            name="result_completed",
        )
    )
    assert result.canonical_json() == before


def test_final_gate_rejects_candidate_run_config_and_checkpoint_mismatches() -> None:
    lifecycle = _run_completed_lifecycle()
    bad_handles = (
        issue_final_test_handle(
            handle_id="handle:bad-candidate",
            run_id="run:final",
            candidate_id="candidate:other",
            scientific_config_digest=lifecycle.scientific_config_digest,
            descriptor=_final_descriptor(),
            examples=_examples("test_ood"),
        ),
        issue_final_test_handle(
            handle_id="handle:bad-run",
            run_id="run:other",
            candidate_id=lifecycle.candidate.candidate_id,
            scientific_config_digest=lifecycle.scientific_config_digest,
            descriptor=_final_descriptor(),
            examples=_examples("test_ood"),
        ),
        issue_final_test_handle(
            handle_id="handle:bad-config",
            run_id="run:final",
            candidate_id=lifecycle.candidate.candidate_id,
            scientific_config_digest="sha256:other-config",
            descriptor=_final_descriptor(),
            examples=_examples("test_ood"),
        ),
    )
    for handle in bad_handles:
        with pytest.raises(ValueError):
            open_final_test(
                handle,
                lifecycle.candidate,
                lifecycle.checkpoint,
                lifecycle.restoration,
            )

    wrong_identity = lifecycle.checkpoint.checkpoint.model_copy(
        update={"checkpoint_id": "checkpoint:not-restored"}
    )
    wrong_receipt = RestorationReceipt(
        receipt_id="restored:wrong",
        candidate_selection_id=lifecycle.candidate.frozen_selection_id,
        store_id="store:in-memory",
        checkpoint=wrong_identity,
    )
    with pytest.raises(ValueError, match="restoration"):
        open_final_test(
            lifecycle.handle,
            lifecycle.candidate,
            lifecycle.checkpoint,
            wrong_receipt,
        )


def test_ordinary_results_reject_foreign_selection_fields() -> None:
    ordinary = _ordinary_result(_run_completed_lifecycle())
    mixed_ordinary = ordinary.canonical_json().replace(
        '{"artifacts"', '{"diagnostic_selection":{},"artifacts"', 1
    )
    with pytest.raises(ValidationError, match="diagnostic_selection"):
        parse_run_result_json(mixed_ordinary)


def test_failed_and_incomplete_result_states_round_trip_without_final_claims() -> None:
    config = ordinary_erm_config()
    code, environment = _provenance()
    failed = OrdinaryRunResult(
        schema_version="grit.run-result/v1",
        result_kind="ordinary",
        run_id="run:failed",
        resolved_config=config,
        resolved_config_digest=config.canonical_digest(),
        status=FailedStatus(
            kind="failed",
            phase="training",
            error_type="SyntheticError",
            message="synthetic failure",
        ),
        code=code,
        environment=environment,
        validation_metrics=(),
        candidate_selection=None,
        checkpoint_selection=None,
        restoration=None,
        final_test_metrics=None,
        artifacts=(),
    )
    assert parse_run_result_json(failed.canonical_json()) == failed

    incomplete = OrdinaryRunResult(
        schema_version="grit.run-result/v1",
        result_kind="ordinary",
        run_id="run:incomplete",
        resolved_config=config,
        resolved_config_digest=config.canonical_digest(),
        status=IncompleteStatus(
            kind="incomplete",
            last_completed_phase="configured",
            reason="run was not executed",
        ),
        code=code,
        environment=environment,
        validation_metrics=(),
        candidate_selection=None,
        checkpoint_selection=None,
        restoration=None,
        final_test_metrics=None,
        artifacts=(),
    )
    assert parse_run_result_json(incomplete.canonical_json()) == incomplete


def test_checkpoint_store_identity_mismatch_is_rejected_before_state_restore() -> None:
    lifecycle = _run_completed_lifecycle()
    selected = lifecycle.checkpoint.checkpoint
    store = InMemoryCheckpointStore()
    store.save(
        selected.model_copy(update={"run_id": "run:wrong"}),
        "state:wrong",
    )
    restored: list[str] = []
    with pytest.raises(ValueError, match="identity"):
        restore_checkpoint(lifecycle.checkpoint, store, restored.append)
    assert restored == []


def test_successful_result_requires_bound_nonempty_final_metrics() -> None:
    lifecycle = _run_completed_lifecycle()
    result = _ordinary_result(lifecycle)

    with pytest.raises(ValidationError, match="requires one final metric"):
        OrdinaryRunResult.model_validate(
            result.model_dump(mode="python") | {"final_test_metrics": ()}
        )

    with pytest.raises(ValidationError, match="requires one final metric"):
        OrdinaryRunResult.model_validate(
            result.model_dump(mode="python")
            | {"final_test_metrics": (lifecycle.final_metric,) * 2}
        )

    with pytest.raises(ValidationError, match="requires validation metrics"):
        OrdinaryRunResult.model_validate(
            result.model_dump(mode="python") | {"validation_metrics": ()}
        )

    missing_contributor = tuple(
        metric
        for metric in result.validation_metrics
        if metric.record_id != lifecycle.checkpoint.decision.contributing_record_ids[0]
    )
    with pytest.raises(ValidationError, match="unavailable validation metrics"):
        OrdinaryRunResult.model_validate(
            result.model_dump(mode="python")
            | {"validation_metrics": missing_contributor}
        )

    wrong_method = lifecycle.final_metric.model_copy(update={"method_id": "grit"})
    with pytest.raises(ValidationError, match="final metric identity"):
        OrdinaryRunResult.model_validate(
            result.model_dump(mode="python") | {"final_test_metrics": (wrong_method,)}
        )

    wrong_seed = lifecycle.final_metric.model_copy(update={"seed": 999})
    with pytest.raises(ValidationError, match="final metric identity"):
        OrdinaryRunResult.model_validate(
            result.model_dump(mode="python") | {"final_test_metrics": (wrong_seed,)}
        )

    secondary_config = ordinary_erm_config(CmnistSelector.SECONDARY_SOURCE)
    with pytest.raises(ValidationError, match="selector does not match"):
        OrdinaryRunResult.model_validate(
            result.model_dump(mode="python")
            | {
                "resolved_config": secondary_config,
                "resolved_config_digest": secondary_config.canonical_digest(),
            }
        )

    contributor_decisions = (
        lifecycle.candidate.decision.contributing_checkpoint_decisions
    )
    candidate_decision = lifecycle.candidate.decision.model_copy(
        update={
            "projection_rank": 1,
            "contributing_checkpoint_decisions": tuple(
                decision.model_copy(update={"projection_rank": 1})
                for decision in contributor_decisions
            ),
        }
    )
    candidate_with_rank = lifecycle.candidate.model_copy(
        update={"decision": candidate_decision}
    )
    checkpoint_decision = lifecycle.checkpoint.decision.model_copy(
        update={"projection_rank": 1}
    )
    checkpoint_with_rank = lifecycle.checkpoint.model_copy(
        update={"decision": checkpoint_decision}
    )
    validation_with_rank = tuple(
        metric.model_copy(update={"projection_rank": 1})
        for metric in result.validation_metrics
    )
    with pytest.raises(ValidationError, match="does not match"):
        OrdinaryRunResult.model_validate(
            result.model_dump(mode="python")
            | {
                "candidate_selection": candidate_with_rank,
                "checkpoint_selection": checkpoint_with_rank,
                "validation_metrics": validation_with_rank,
            }
        )
