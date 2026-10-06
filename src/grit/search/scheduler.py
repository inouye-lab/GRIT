"""Narrow local run scheduler with canonical run-level continuation."""

from __future__ import annotations

from collections.abc import Callable, Hashable, Sequence
from functools import lru_cache
from pathlib import Path
from typing import Annotated, Literal, TypeAlias, cast

from pydantic import Field, StrictInt, StrictStr, TypeAdapter, model_validator

from grit.methods.training import (
    PersistedLinearCheckpointManifest,
    PersistedLinearCheckpointStore,
)
from grit.methods.types import MethodId
from grit.results import (
    CodeProvenance,
    EnvironmentProvenance,
    OrdinaryRunResult,
)
from grit.schemas import (
    ORDINARY_CMNIST_SELECTORS,
    CmnistSelector,
    SeedStage,
    StrictBoundaryModel,
)
from grit.search.execution import record_execution
from grit.search.plan import SearchCandidate, SearchLineage, SearchPlan
from grit.search.waterbirds_contracts import WaterbirdsRunResult
from grit.selection.cmnist import (
    CheckpointSelectionDecision,
    FrozenCandidateSelection,
    ValidationMetricRecord,
    select_checkpoint,
)
from grit.selection.waterbirds import (
    ORDINARY_WATERBIRDS_SELECTOR,
    FrozenWaterbirdsCandidate,
    WaterbirdsCheckpointSelection,
    WaterbirdsSelector,
    WaterbirdsSelectorRecord,
    WaterbirdsValidationMetricRecord,
    select_waterbirds_checkpoint,
)

NonEmptyStr: TypeAlias = Annotated[StrictStr, Field(min_length=1)]
RunStage: TypeAlias = Literal[
    SeedStage.TUNING,
    SeedStage.CONFIRMATION,
    SeedStage.FINAL,
]


class FrozenWinnerReference(StrictBoundaryModel):
    """Shared scheduling reference to a dataset-specific validated winner."""

    dataset: Literal["cmnist", "waterbirds_cf"]
    selector: NonEmptyStr
    method_id: MethodId
    candidate_id: NonEmptyStr
    scientific_config_digest: NonEmptyStr
    frozen_selection_id: NonEmptyStr
    frozen_artifact_digest: NonEmptyStr


class SearchRunTask(StrictBoundaryModel):
    """One deterministic candidate/stage/seed unit of resumable work."""

    task_id: NonEmptyStr
    plan_digest: NonEmptyStr
    dataset: Literal["cmnist", "waterbirds_cf"]
    lineage: SearchLineage
    candidate: SearchCandidate
    stage: RunStage
    seed: StrictInt
    selector: NonEmptyStr | None
    frozen_winner: FrozenWinnerReference | None
    relative_directory: NonEmptyStr

    @model_validator(mode="after")
    def _validate_identity(self) -> SearchRunTask:
        expected_payload = {
            "plan_digest": self.plan_digest,
            "dataset": self.dataset,
            "lineage_digest": self.lineage.canonical_digest(),
            "candidate_id": self.candidate.candidate_id,
            "scientific_config_digest": self.candidate.scientific_config_digest,
            "method_id": self.candidate.method_id,
            "stage": self.stage.value,
            "seed": self.seed,
            "selector": self.selector,
            "frozen_winner_digest": (
                self.frozen_winner.frozen_artifact_digest
                if self.frozen_winner is not None
                else None
            ),
        }
        expected_id = f"task:{self.canonical_identity_digest(expected_payload)}"
        if self.task_id != expected_id:
            raise ValueError("search task ID is inconsistent")
        expected_directory = _task_directory_parts(self)
        if self.relative_directory != "/".join(expected_directory):
            raise ValueError("search task directory is inconsistent")
        if self.stage is SeedStage.FINAL:
            if self.selector is None or self.frozen_winner is None:
                raise ValueError("final search tasks require a frozen winner")
            frozen_identity = (
                self.frozen_winner.dataset,
                self.frozen_winner.selector,
                self.frozen_winner.method_id,
                self.frozen_winner.candidate_id,
                self.frozen_winner.scientific_config_digest,
            )
            task_identity = (
                self.dataset,
                self.selector,
                self.candidate.method_id,
                self.candidate.candidate_id,
                self.candidate.scientific_config_digest,
            )
            if frozen_identity != task_identity:
                raise ValueError("final search task does not match its frozen winner")
        elif self.selector is not None or self.frozen_winner is not None:
            raise ValueError("pre-final search tasks cannot contain a frozen winner")
        return self

    @staticmethod
    def canonical_identity_digest(payload: object) -> str:
        from grit.schemas import canonical_digest_value

        return canonical_digest_value(payload).removeprefix("sha256:")[:32]


class CmnistCompletedStageRun(StrictBoundaryModel):
    schema_version: Literal["grit.cmnist-search-stage-run/v1"]
    dataset: Literal["cmnist"]
    status: Literal["complete"]
    task: SearchRunTask
    lineage: SearchLineage
    code: CodeProvenance | None = None
    environment: EnvironmentProvenance | None = None
    validation_metrics: tuple[ValidationMetricRecord, ...]
    # Always null; kept so stage records retain their recorded schema.
    diagnostic_metrics: None = None
    checkpoint_decisions: tuple[CheckpointSelectionDecision, ...]
    final_result_relative_path: Literal["final-result.json"] | None = None
    final_result_digest: NonEmptyStr | None = None
    final_result: OrdinaryRunResult | None = None

    @model_validator(mode="after")
    def _validate_run(self) -> CmnistCompletedStageRun:
        if self.task.dataset != self.dataset:
            raise ValueError("linear-probe stage result has another dataset task")
        if self.lineage != self.task.lineage:
            raise ValueError("CMNIST stage result lineage does not match its task")
        if not self.validation_metrics or not self.checkpoint_decisions:
            raise ValueError("CMNIST completed stage requires validation evidence")
        expected_identity = (
            self.task.candidate.candidate_id,
            self.task.candidate.method_id,
            self.task.candidate.scientific_config_digest,
            self.task.stage,
            self.task.seed,
            self.task.candidate.requested_rank,
        )
        if any(
            (
                metric.candidate_id,
                metric.method_id,
                metric.scientific_config_digest,
                metric.seed_stage,
                metric.seed,
                metric.projection_rank,
            )
            != expected_identity
            for metric in self.validation_metrics
        ):
            raise ValueError("CMNIST stage metric identity is inconsistent")
        if self.task.stage is SeedStage.FINAL:
            expected_selectors = (CmnistSelector(self.task.selector),)
        else:
            expected_selectors = ORDINARY_CMNIST_SELECTORS
        if tuple(item.selector for item in self.checkpoint_decisions) != (
            expected_selectors
        ):
            raise ValueError("CMNIST stage checkpoint selectors are inconsistent")
        recomputed = tuple(
            select_checkpoint(self.validation_metrics, selector)
            for selector in expected_selectors
        )
        if self.checkpoint_decisions != recomputed:
            raise ValueError("CMNIST checkpoint decision trace is inconsistent")
        final_values = (
            self.final_result_relative_path,
            self.final_result_digest,
            self.final_result,
        )
        if self.task.stage is SeedStage.FINAL:
            if any(value is None for value in final_values):
                raise ValueError("CMNIST final task requires its ordinary result")
            result = self.final_result
            if result is None:
                raise AssertionError("validated CMNIST final result disappeared")
            frozen_winner = self.task.frozen_winner
            if frozen_winner is None:
                raise AssertionError("validated CMNIST frozen winner disappeared")
            selected_candidate = result.candidate_selection
            if selected_candidate is None:
                raise ValueError("CMNIST final result lacks its frozen candidate")
            if (
                self.final_result_digest != result.canonical_digest()
                or result.run_id != self.checkpoint_decisions[0].checkpoint.run_id
                or selected_candidate.frozen_selection_id
                != frozen_winner.frozen_selection_id
                or result.resolved_config.scientific_config_digest()
                != self.task.candidate.scientific_config_digest
            ):
                raise ValueError("CMNIST final result identity is inconsistent")
        elif any(value is not None for value in final_values):
            raise ValueError("pre-final CMNIST tasks cannot contain test results")
        return self


class WaterbirdsCompletedStageRun(StrictBoundaryModel):
    schema_version: Literal["grit.waterbirds-search-stage-run/v1"]
    dataset: Literal["waterbirds_cf"]
    status: Literal["complete"]
    task: SearchRunTask
    lineage: SearchLineage
    code: CodeProvenance | None = None
    environment: EnvironmentProvenance | None = None
    validation_metrics: tuple[WaterbirdsValidationMetricRecord, ...]
    # Always null; kept so stage records retain their recorded schema.
    diagnostic_metrics: None = None
    checkpoint_decision: WaterbirdsCheckpointSelection
    final_result_relative_path: Literal["final-result.json"] | None = None
    final_result_digest: NonEmptyStr | None = None
    final_result: WaterbirdsRunResult | None = None

    @property
    def selector(self) -> WaterbirdsSelector:
        return ORDINARY_WATERBIRDS_SELECTOR

    def selector_records(
        self, selector: WaterbirdsSelector
    ) -> tuple[WaterbirdsSelectorRecord, ...]:
        """The validation records the selector reads."""

        del selector
        return self.validation_metrics

    @model_validator(mode="after")
    def _validate_run(self) -> WaterbirdsCompletedStageRun:
        if self.task.dataset != "waterbirds_cf":
            raise ValueError("Waterbirds stage result has another dataset task")
        if self.lineage != self.task.lineage:
            raise ValueError("Waterbirds stage result lineage does not match its task")
        if not self.validation_metrics:
            raise ValueError("Waterbirds completed stage requires validation evidence")
        expected_identity = (
            self.task.candidate.candidate_id,
            self.task.candidate.method_id,
            self.task.candidate.scientific_config_digest,
            self.task.stage,
            self.task.seed,
            self.task.candidate.requested_rank,
            self.lineage.dataset_manifest_digest,
            self.lineage.feature_cache_manifest_digest,
            self.lineage.normalization,
            self.lineage.adjusted_weight_spec_digest,
        )
        if any(
            (
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
            != expected_identity
            for metric in self.validation_metrics
        ):
            raise ValueError("Waterbirds stage metric identity is inconsistent")
        selector = self.selector
        if self.task.stage is SeedStage.FINAL and self.task.selector != selector:
            raise ValueError(
                "Waterbirds final stage evidence does not match its selector track"
            )
        if self.checkpoint_decision.selector != selector:
            raise ValueError("Waterbirds stage checkpoint selector is inconsistent")
        recomputed = select_waterbirds_checkpoint(
            self.selector_records(selector), selector
        )
        if self.checkpoint_decision != recomputed:
            raise ValueError("Waterbirds checkpoint decision trace is inconsistent")
        final_values = (
            self.final_result_relative_path,
            self.final_result_digest,
            self.final_result,
        )
        if self.task.stage is SeedStage.FINAL:
            if any(value is None for value in final_values):
                raise ValueError("Waterbirds final task requires its result")
            result = self.final_result
            if result is None:
                raise AssertionError("validated Waterbirds final result disappeared")
            frozen_winner = self.task.frozen_winner
            if frozen_winner is None:
                raise AssertionError("validated Waterbirds frozen winner disappeared")
            if (
                self.final_result_digest != result.canonical_digest()
                or result.run_id != self.checkpoint_decision.checkpoint.run_id
                or result.selected_candidate.frozen_selection_id
                != frozen_winner.frozen_selection_id
                or result.resolved_config.scientific_config_digest()
                != self.task.candidate.scientific_config_digest
            ):
                raise ValueError("Waterbirds final result identity is inconsistent")
        elif any(value is not None for value in final_values):
            raise ValueError("pre-final Waterbirds tasks cannot contain test results")
        return self


CompletedStageRun: TypeAlias = Annotated[
    CmnistCompletedStageRun | WaterbirdsCompletedStageRun,
    Field(discriminator="dataset"),
]
_STAGE_RUN_ADAPTER: TypeAdapter[CompletedStageRun] = TypeAdapter(CompletedStageRun)


class IncompleteRunMarker(StrictBoundaryModel):
    schema_version: Literal["grit.search-incomplete/v1"]
    status: Literal["incomplete"]
    task: SearchRunTask


class SearchStatus(StrictBoundaryModel):
    schema_version: Literal["grit.search-status/v1"]
    plan_digest: NonEmptyStr
    complete_task_ids: tuple[NonEmptyStr, ...]
    missing_task_ids: tuple[NonEmptyStr, ...]
    interrupted_task_ids: tuple[NonEmptyStr, ...]


StageExecutor: TypeAlias = Callable[[SearchRunTask, Path], CompletedStageRun]


def make_search_task(
    plan: SearchPlan,
    candidate: SearchCandidate,
    stage: Literal[SeedStage.TUNING, SeedStage.CONFIRMATION],
    seed: int,
) -> SearchRunTask:
    """Derive one tuning or confirmation task from the canonical plan."""

    return _make_task(plan, candidate, stage, seed, selector=None, frozen=None)


def make_final_search_task(
    plan: SearchPlan,
    candidate: SearchCandidate,
    seed: int,
    frozen_candidate: FrozenCandidateSelection | FrozenWaterbirdsCandidate,
) -> SearchRunTask:
    """Derive a final task only from a revalidated dataset-specific freeze."""

    if isinstance(frozen_candidate, FrozenCandidateSelection):
        frozen = FrozenCandidateSelection.model_validate_json(
            frozen_candidate.canonical_json()
        )
        if frozen.method_id not in plan.methods:
            raise ValueError("CMNIST frozen winner has an unsupported method")
        reference = FrozenWinnerReference(
            dataset=plan.dataset,
            selector=frozen.selector.value,
            method_id=frozen.method_id,
            candidate_id=frozen.candidate_id,
            scientific_config_digest=frozen.scientific_config_digest,
            frozen_selection_id=frozen.frozen_selection_id,
            frozen_artifact_digest=frozen.canonical_digest(),
        )
    else:
        frozen = FrozenWaterbirdsCandidate.model_validate_json(
            frozen_candidate.canonical_json()
        )
        reference = FrozenWinnerReference(
            dataset="waterbirds_cf",
            selector=frozen.selector,
            method_id=frozen.method_id,
            candidate_id=frozen.candidate_id,
            scientific_config_digest=frozen.scientific_config_digest,
            frozen_selection_id=frozen.frozen_selection_id,
            frozen_artifact_digest=frozen.canonical_digest(),
        )
    return _make_task(
        plan,
        candidate,
        SeedStage.FINAL,
        seed,
        selector=reference.selector,
        frozen=reference,
    )


def _make_task(
    plan: SearchPlan,
    candidate: SearchCandidate,
    stage: RunStage,
    seed: int,
    *,
    selector: str | None,
    frozen: FrozenWinnerReference | None,
) -> SearchRunTask:

    validated_plan, plan_digest = _validated_plan_context(cast(Hashable, plan))
    if candidate not in validated_plan.candidates:
        raise ValueError("search task candidate is absent from the canonical plan")
    configured = {
        SeedStage.TUNING: validated_plan.seeds.stages.tuning,
        SeedStage.CONFIRMATION: validated_plan.seeds.stages.confirmation,
        SeedStage.FINAL: validated_plan.seeds.stages.final,
    }
    if seed not in configured[stage]:
        raise ValueError("search task seed is not configured for its stage")
    payload = {
        "plan_digest": plan_digest,
        "dataset": validated_plan.dataset,
        "lineage_digest": validated_plan.resolved_config.lineage.canonical_digest(),
        "candidate_id": candidate.candidate_id,
        "scientific_config_digest": candidate.scientific_config_digest,
        "method_id": candidate.method_id,
        "stage": stage.value,
        "seed": seed,
        "selector": selector,
        "frozen_winner_digest": (
            frozen.frozen_artifact_digest if frozen is not None else None
        ),
    }
    task_id = f"task:{SearchRunTask.canonical_identity_digest(payload)}"
    selector_parts = (selector,) if selector is not None else ()
    directory = "/".join(
        (
            "runs",
            stage.value,
            candidate.method_id,
            *selector_parts,
            candidate.candidate_id,
            str(seed),
        )
    )
    return SearchRunTask(
        task_id=task_id,
        plan_digest=plan_digest,
        dataset=validated_plan.dataset,
        lineage=validated_plan.resolved_config.lineage,
        candidate=candidate,
        stage=stage,
        seed=seed,
        selector=selector,
        frozen_winner=frozen,
        relative_directory=directory,
    )


@lru_cache(maxsize=8)
def _validated_plan_context(plan_key: Hashable) -> tuple[SearchPlan, str]:
    plan = cast(SearchPlan, plan_key)
    validated = SearchPlan.model_validate_json(plan.canonical_json())
    return validated, validated.canonical_digest()


class LocalRunScheduler:
    """Execute independent run tasks and reuse only fully validated results."""

    def __init__(self, output_root: Path, plan: SearchPlan) -> None:
        self._root = output_root
        self._plan = SearchPlan.model_validate_json(plan.canonical_json())
        self._plan_digest = self._plan.canonical_digest()

    def run_tasks(
        self,
        tasks: Sequence[SearchRunTask],
        execute: StageExecutor,
    ) -> tuple[CompletedStageRun, ...]:
        """Run missing tasks in order and validate every reused result."""

        results: list[CompletedStageRun] = []
        for task in tasks:
            validated_task = self._validate_task(task)
            completed = self._load_completed(validated_task)
            if completed is None:
                completed = self._execute_one(validated_task, execute)
            results.append(completed)
        return tuple(results)

    def run_tasks_bounded(
        self,
        tasks: Sequence[SearchRunTask],
        execute: StageExecutor,
        *,
        max_new_runs: int,
    ) -> tuple[tuple[CompletedStageRun, ...], int]:
        """Reuse completed tasks and execute at most ``max_new_runs`` missing tasks.

        The bound is operational: task identities and stored results are identical to an
        unrestricted run. A zero bound is useful internally after a shared invocation
        budget has been exhausted; the public CLI rejects nonpositive user values.
        """

        if max_new_runs < 0:
            raise ValueError("max_new_runs must not be negative")
        inspected = tuple(
            (validated, self._load_completed(validated))
            for task in tasks
            for validated in (self._validate_task(task),)
        )
        results: list[CompletedStageRun] = []
        newly_executed = 0
        for validated_task, completed in inspected:
            if completed is None:
                if newly_executed >= max_new_runs:
                    continue
                completed = self._execute_one(validated_task, execute)
                newly_executed += 1
            results.append(completed)
        return tuple(results), newly_executed

    def status(self, tasks: Sequence[SearchRunTask]) -> SearchStatus:
        """Inspect canonical run state without invoking training or final access."""

        complete: list[str] = []
        missing: list[str] = []
        interrupted: list[str] = []
        for task in tasks:
            validated = self._validate_task(task)
            if self._load_completed(validated) is not None:
                complete.append(validated.task_id)
            elif self._staging_path(validated).exists():
                self._load_incomplete_marker(validated)
                interrupted.append(validated.task_id)
            else:
                missing.append(validated.task_id)
        return SearchStatus(
            schema_version="grit.search-status/v1",
            plan_digest=self._plan_digest,
            complete_task_ids=tuple(complete),
            missing_task_ids=tuple(missing),
            interrupted_task_ids=tuple(interrupted),
        )

    def completed_results(
        self, tasks: Sequence[SearchRunTask]
    ) -> tuple[CompletedStageRun, ...]:
        """Load a complete canonical stage without executing missing work."""

        results: list[CompletedStageRun] = []
        for task in tasks:
            validated = self._validate_task(task)
            completed = self._load_completed(validated)
            if completed is None:
                raise ValueError(
                    "cannot load canonical stage results while planned tasks "
                    "are missing"
                )
            results.append(completed)
        return tuple(results)

    def _validate_task(self, task: SearchRunTask) -> SearchRunTask:
        validated = SearchRunTask.model_validate_json(task.canonical_json())
        if validated.plan_digest != self._plan_digest:
            raise ValueError("search task belongs to another plan")
        if (
            validated.dataset != self._plan.dataset
            or validated.lineage != self._plan.resolved_config.lineage
        ):
            raise ValueError("search task input lineage does not match the plan")
        if validated.candidate not in self._plan.candidates:
            raise ValueError("search task candidate is not present in the plan")
        return validated

    def _load_completed(self, task: SearchRunTask) -> CompletedStageRun | None:
        directory = self._root / task.relative_directory
        if not directory.exists():
            return None
        if not directory.is_dir():
            raise ValueError(f"search run path is not a directory: {directory}")
        return self._load_result_from_directory(task, directory)

    def _load_result_from_directory(
        self,
        task: SearchRunTask,
        directory: Path,
    ) -> CompletedStageRun:
        result_path = directory / "result.json"
        if not result_path.is_file():
            raise ValueError(f"existing search run is not complete: {directory}")
        try:
            parsed = _STAGE_RUN_ADAPTER.validate_json(
                result_path.read_text(encoding="utf-8")
            )
        except (OSError, ValueError) as error:
            raise ValueError(
                f"existing search result is corrupted: {result_path}"
            ) from error
        if parsed.task != task:
            raise ValueError("existing search result is incompatible with its task")
        if parsed.lineage != self._plan.resolved_config.lineage:
            raise ValueError("existing search result has incompatible input lineage")
        self._validate_final_result_file(directory, parsed)
        return parsed

    def _execute_one(
        self,
        task: SearchRunTask,
        execute: StageExecutor,
    ) -> CompletedStageRun:
        final_path = self._root / task.relative_directory
        staging = self._staging_path(task)
        if staging.exists():
            self._load_incomplete_marker(task)
            if (staging / "result.json").is_file():
                recovered = self._load_result_from_directory(task, staging)
                if final_path.exists():
                    raise FileExistsError(
                        f"refusing to overwrite search run: {final_path}"
                    )
                final_path.parent.mkdir(parents=True, exist_ok=True)
                (staging / "run-state.json").unlink()
                staging.replace(final_path)
                return recovered
            interrupted = _next_interrupted_path(staging)
            staging.replace(interrupted)
        staging.mkdir(parents=True, exist_ok=False)
        marker = IncompleteRunMarker(
            schema_version="grit.search-incomplete/v1",
            status="incomplete",
            task=task,
        )
        _atomic_write(staging / "run-state.json", marker.canonical_json() + "\n")
        record_execution(staging)
        result = execute(task, staging)
        validated = _STAGE_RUN_ADAPTER.validate_json(result.canonical_json())
        if validated.task != task:
            raise ValueError("run executor returned a result for another task")
        if validated.lineage != self._plan.resolved_config.lineage:
            raise ValueError("run executor returned incompatible input lineage")
        self._validate_final_result_file(staging, validated)
        _atomic_write(staging / "result.json", validated.canonical_json() + "\n")
        (staging / "run-state.json").unlink()
        if final_path.exists():
            raise FileExistsError(f"refusing to overwrite search run: {final_path}")
        final_path.parent.mkdir(parents=True, exist_ok=True)
        staging.replace(final_path)
        return validated

    def _staging_path(self, task: SearchRunTask) -> Path:
        final = self._root / task.relative_directory
        return final.with_name(f".{final.name}.incomplete")

    def _load_incomplete_marker(self, task: SearchRunTask) -> IncompleteRunMarker:
        marker_path = self._staging_path(task) / "run-state.json"
        if not marker_path.is_file():
            raise ValueError("interrupted search directory lacks a valid marker")
        try:
            marker = IncompleteRunMarker.model_validate_json(
                marker_path.read_text(encoding="utf-8")
            )
        except (OSError, ValueError) as error:
            raise ValueError("interrupted search marker is corrupted") from error
        if marker.task != task:
            raise ValueError("interrupted search marker belongs to another task")
        return marker

    @staticmethod
    def _validate_final_result_file(
        directory: Path,
        result: CompletedStageRun,
    ) -> None:
        if result.task.stage is not SeedStage.FINAL:
            return
        relative = result.final_result_relative_path
        embedded = result.final_result
        if relative is None or embedded is None:
            raise ValueError("final search run lacks its result artifact")
        path = directory / relative
        if not path.is_file():
            raise ValueError("final search result artifact is missing")
        if path.read_text(encoding="utf-8").strip() != embedded.canonical_json():
            raise ValueError("final search result artifact is corrupted")
        checkpoint_root = directory / "selected-checkpoint"
        manifest_path = checkpoint_root / "manifest.json"
        if not manifest_path.is_file():
            raise ValueError("selected checkpoint manifest is missing")
        try:
            manifest = PersistedLinearCheckpointManifest.model_validate_json(
                manifest_path.read_text(encoding="utf-8")
            )
            stored = PersistedLinearCheckpointStore(checkpoint_root).load(
                manifest.checkpoint.checkpoint_id
            )
        except (OSError, ValueError, KeyError) as error:
            raise ValueError("selected checkpoint artifact is corrupted") from error
        if isinstance(result, CmnistCompletedStageRun):
            cmnist_result = result.final_result
            if cmnist_result is None:
                raise ValueError("CMNIST final result lacks checkpoint selection")
            checkpoint_selection = cmnist_result.checkpoint_selection
            if checkpoint_selection is None:
                raise ValueError("CMNIST final result lacks checkpoint selection")
            checkpoint_identity = checkpoint_selection.checkpoint
            checkpoint_references = tuple(
                item
                for item in cmnist_result.artifacts
                if item.kind == "selected_linear_checkpoint"
            )
            if len(checkpoint_references) != 1:
                raise ValueError("CMNIST final result lacks checkpoint provenance")
            expected_manifest_digest = checkpoint_references[0].digest
        else:
            waterbirds_result = result.final_result
            if waterbirds_result is None:
                raise ValueError("Waterbirds final result lacks checkpoint selection")
            checkpoint_identity = waterbirds_result.selected_checkpoint.checkpoint
            expected_manifest_digest = (
                waterbirds_result.selected_checkpoint_manifest_digest
            )
        if manifest.checkpoint != checkpoint_identity:
            raise ValueError("selected checkpoint identity is inconsistent")
        if stored.identity != manifest.checkpoint:
            raise ValueError("selected checkpoint contents are inconsistent")
        if manifest.canonical_digest() != expected_manifest_digest:
            raise ValueError("selected checkpoint manifest digest is inconsistent")


def _task_directory_parts(task: SearchRunTask) -> tuple[str, ...]:
    selector = (task.selector,) if task.selector is not None else ()
    return (
        "runs",
        task.stage.value,
        task.candidate.method_id,
        *selector,
        task.candidate.candidate_id,
        str(task.seed),
    )


def _next_interrupted_path(staging: Path) -> Path:
    for attempt in range(1, 10_000):
        candidate = staging.with_name(f"{staging.name}.interrupted-{attempt}")
        if not candidate.exists():
            return candidate
    raise RuntimeError("too many interrupted attempts for one search run")


def _atomic_write(path: Path, payload: str) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(payload, encoding="utf-8")
    temporary.replace(path)
