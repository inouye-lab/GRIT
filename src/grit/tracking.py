"""Tracking boundary: a null event sink and the optional Weights & Biases mirror.

Tracking is operational, never scientific. It is configured only by environment
variables (`WANDB_PROJECT` enables it; `WANDB_ENTITY` and `WANDB_MODE` are the usual
W&B settings), so it stays out of the YAML configs and their digests, and its files
live under `$PROJECT_SCRATCH/wandb` so the canonical result directories and the
experiment index never see them.

Every mirror call is best effort: a tracking failure warns and returns, and cannot
change a candidate, a checkpoint, a metric, or a result. Trackers are created only by
code that is about to execute a task, so reused (skipped) tasks, dry runs, and status
checks create no runs.
"""

from __future__ import annotations

import importlib
import os
import uuid
import warnings
from collections.abc import Generator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Protocol, TypeAlias, cast

from pydantic import Field, StrictStr

from grit.config import (
    ComposedAlgorithmConfig,
    FishrAlgorithmConfig,
    GroupDroAlgorithmConfig,
    IrmAlgorithmConfig,
    LisaAlgorithmConfig,
    RdmAlgorithmConfig,
    RexAlgorithmConfig,
)
from grit.methods.types import base_objective, pair_intervention
from grit.paths import scratch_root
from grit.schemas import StrictBoundaryModel
from grit.search.execution import execution_metadata

if TYPE_CHECKING:
    from grit.search.plan import SearchCandidate, SearchPlan
    from grit.search.scheduler import SearchRunTask


class LifecycleEvent(StrictBoundaryModel):
    event_id: StrictStr = Field(min_length=1)
    run_id: StrictStr = Field(min_length=1)
    name: StrictStr = Field(min_length=1)


class EventSink(Protocol):
    def emit(self, event: LifecycleEvent) -> None: ...


class NullEventSink:
    """Discard events without retaining state or influencing control flow."""

    def emit(self, event: LifecycleEvent) -> None:
        del event


TrackingValue: TypeAlias = float | int | str | bool | None


class RunTracker(Protocol):
    """What training and orchestration may tell a tracker about a live task."""

    def log_epoch(self, epoch: int, values: Mapping[str, float]) -> None: ...

    def record_selection(self, values: Mapping[str, TrackingValue]) -> None: ...

    def record_table(
        self,
        name: str,
        columns: Sequence[str],
        rows: Sequence[Sequence[TrackingValue]],
    ) -> None: ...

    def finish(self, *, failed: bool = False) -> None: ...


class NullRunTracker:
    """Accept every tracking call and retain nothing."""

    def log_epoch(self, epoch: int, values: Mapping[str, float]) -> None:
        del epoch, values

    def record_selection(self, values: Mapping[str, TrackingValue]) -> None:
        del values

    def record_table(
        self,
        name: str,
        columns: Sequence[str],
        rows: Sequence[Sequence[TrackingValue]],
    ) -> None:
        del name, columns, rows

    def finish(self, *, failed: bool = False) -> None:
        del failed


@dataclass(frozen=True, slots=True)
class TrackingSettings:
    """Resolved mirror settings; `project` is what turns tracking on."""

    project: str | None
    entity: str | None
    mode: str
    directory: Path

    @property
    def enabled(self) -> bool:
        return bool(self.project) and self.mode != "disabled"


def tracking_settings_from_environment() -> TrackingSettings:
    """Read the mirror's settings from the environment, never from a config."""

    return TrackingSettings(
        project=os.environ.get("WANDB_PROJECT", "").strip() or None,
        entity=os.environ.get("WANDB_ENTITY", "").strip() or None,
        mode=os.environ.get("WANDB_MODE", "").strip() or "online",
        directory=scratch_root() / "wandb",
    )


class TrackedRun(Protocol):
    """One live backend run, as narrow as this repository's use of it."""

    def log(self, values: Mapping[str, object]) -> None: ...

    def update_summary(self, values: Mapping[str, object]) -> None: ...

    def log_table(
        self,
        name: str,
        columns: Sequence[str],
        rows: Sequence[Sequence[TrackingValue]],
    ) -> None: ...

    def finish(self, exit_code: int | None = None) -> None: ...


class TrackingBackend(Protocol):
    def start(
        self,
        *,
        project: str,
        entity: str | None,
        mode: str,
        directory: Path,
        group: str,
        name: str,
        job_type: str,
        config: Mapping[str, TrackingValue],
    ) -> TrackedRun: ...


class _WandbSummary(Protocol):
    def update(self, values: Mapping[str, object]) -> None: ...


class _WandbRun(Protocol):
    @property
    def summary(self) -> _WandbSummary: ...

    def log(self, data: Mapping[str, object]) -> None: ...

    def finish(self, exit_code: int | None = None) -> None: ...


class _WandbModule(Protocol):
    def init(
        self,
        *,
        project: str,
        entity: str | None,
        mode: str,
        dir: str,  # the W&B keyword argument is named `dir`
        group: str,
        name: str,
        job_type: str,
        config: Mapping[str, TrackingValue],
        reinit: bool,
    ) -> _WandbRun: ...

    def Table(  # the W&B constructor is named `Table`
        self,
        *,
        columns: Sequence[str],
        data: Sequence[Sequence[TrackingValue]],
    ) -> object: ...


@dataclass(frozen=True, slots=True)
class _WandbTrackedRun:
    module: _WandbModule
    run: _WandbRun

    def log(self, values: Mapping[str, object]) -> None:
        self.run.log(dict(values))

    def update_summary(self, values: Mapping[str, object]) -> None:
        self.run.summary.update(dict(values))

    def log_table(
        self,
        name: str,
        columns: Sequence[str],
        rows: Sequence[Sequence[TrackingValue]],
    ) -> None:
        table = self.module.Table(
            columns=list(columns), data=[list(row) for row in rows]
        )
        self.run.log({name: table})

    def finish(self, exit_code: int | None = None) -> None:
        self.run.finish(exit_code)


class WandbBackend:
    """Import and drive W&B lazily so a disabled mirror costs nothing."""

    def start(
        self,
        *,
        project: str,
        entity: str | None,
        mode: str,
        directory: Path,
        group: str,
        name: str,
        job_type: str,
        config: Mapping[str, TrackingValue],
    ) -> TrackedRun:
        module = cast(_WandbModule, importlib.import_module("wandb"))
        directory.mkdir(parents=True, exist_ok=True)
        run = module.init(
            project=project,
            entity=entity,
            mode=mode,
            dir=directory.as_posix(),
            group=group,
            name=name,
            job_type=job_type,
            config=dict(config),
            reinit=True,
        )
        return _WandbTrackedRun(module, run)


@dataclass(frozen=True, slots=True)
class MirrorRunTracker:
    """Forward measurements to one backend run, swallowing tracking failures."""

    run: TrackedRun

    def log_epoch(self, epoch: int, values: Mapping[str, float]) -> None:
        payload: dict[str, object] = {"epoch": epoch}
        payload.update({key: float(value) for key, value in values.items()})
        self._guard("log an epoch", lambda: self.run.log(payload))

    def record_selection(self, values: Mapping[str, TrackingValue]) -> None:
        payload = dict(values)
        self._guard("record a selection", lambda: self.run.update_summary(payload))

    def record_table(
        self,
        name: str,
        columns: Sequence[str],
        rows: Sequence[Sequence[TrackingValue]],
    ) -> None:
        self._guard("record a table", lambda: self.run.log_table(name, columns, rows))

    def finish(self, *, failed: bool = False) -> None:
        # A nonzero exit code is how W&B distinguishes a crashed run from a
        # finished one, so a failed task must never look complete in the dashboard.
        exit_code = 1 if failed else 0
        self._guard("finish a run", lambda: self.run.finish(exit_code))

    @staticmethod
    def _guard(action: str, call: _TrackingCall) -> None:
        try:
            call()
        except Exception as error:  # tracking must never reach the experiment
            _warn(action, error)


class _TrackingCall(Protocol):
    def __call__(self) -> None: ...


def start_task_tracker(
    plan: SearchPlan,
    task: SearchRunTask,
    *,
    algorithm: StrictBoundaryModel | None = None,
    settings: TrackingSettings | None = None,
    backend: TrackingBackend | None = None,
) -> RunTracker:
    """Open one mirror run for a task attempt that is about to be executed.

    ``algorithm`` is the candidate's resolved algorithm configuration, so the mirror
    shows the method settings a candidate holds fixed as well as the searched ones.
    """

    resolved = settings or tracking_settings_from_environment()
    project = resolved.project
    if not resolved.enabled or project is None:
        return NullRunTracker()
    candidate = task.candidate
    selector = task.selector
    config: dict[str, TrackingValue] = {
        **_plan_config(plan),
        **_candidate_config(candidate),
        **_algorithm_config(algorithm),
        "base_objective": base_objective(candidate.method_id),
        "pair_intervention": pair_intervention(candidate.method_id),
        "training_metadata": training_metadata(plan.dataset, algorithm),
        "task_id": task.task_id,
        "attempt_id": uuid.uuid4().hex[:12],
        "relative_directory": task.relative_directory,
        "stage": task.stage.value,
        "seed": task.seed,
        # Pre-final tasks carry no selector, so the track comes from the plan.
        "selector": selector,
        "track": plan_track(plan),
    }
    name = "/".join(
        (
            candidate.method_id,
            _short(candidate.candidate_id),
            *((selector,) if selector is not None else ()),
            task.stage.value,
            f"seed-{task.seed}",
        )
    )
    return _start(
        resolved,
        project,
        group=_group(plan),
        name=name,
        job_type=task.stage.value,
        config=config,
        backend=backend,
    )


def start_named_tracker(
    *,
    group: str,
    name: str,
    job_type: str,
    config: Mapping[str, TrackingValue],
) -> RunTracker:
    """The same optional mirror for compact dataset adapters with their own plans."""
    settings = tracking_settings_from_environment()
    if not settings.enabled or settings.project is None:
        return NullRunTracker()
    return _start(
        settings,
        settings.project,
        group=group,
        name=name,
        job_type=job_type,
        config=config,
    )


@contextmanager
def task_tracker(
    plan: SearchPlan,
    task: SearchRunTask,
    *,
    algorithm: StrictBoundaryModel | None = None,
    settings: TrackingSettings | None = None,
    backend: TrackingBackend | None = None,
) -> Generator[RunTracker]:
    """Track one task attempt, marking the run failed if the attempt raises."""

    tracker = start_task_tracker(
        plan, task, algorithm=algorithm, settings=settings, backend=backend
    )
    try:
        yield tracker
    except BaseException:
        tracker.finish(failed=True)
        raise
    tracker.finish()


def start_summary_tracker(
    plan: SearchPlan,
    *,
    track: str,
    settings: TrackingSettings | None = None,
    backend: TrackingBackend | None = None,
) -> RunTracker:
    """Open one mirror run for a completed search's aggregate summary."""

    resolved = settings or tracking_settings_from_environment()
    project = resolved.project
    if not resolved.enabled or project is None:
        return NullRunTracker()
    return _start(
        resolved,
        project,
        group=_group(plan),
        name=f"summary/{track}",
        job_type="summary",
        config={**_plan_config(plan), "track": track},
        backend=backend,
    )


def log_summary_table(
    plan: SearchPlan,
    *,
    track: str,
    table_name: str,
    columns: Sequence[str],
    rows: Sequence[Sequence[TrackingValue]],
    settings: TrackingSettings | None = None,
    backend: TrackingBackend | None = None,
) -> None:
    """Mirror one completed search's canonical ten-seed summary as a table."""

    tracker = start_summary_tracker(
        plan, track=track, settings=settings, backend=backend
    )
    try:
        tracker.record_table(table_name, columns, rows)
    except BaseException:
        tracker.finish(failed=True)
        raise
    tracker.finish()


def selection_values(
    selector: str,
    *,
    epoch: int,
    checkpoint_id: str,
    metrics: Mapping[str, float],
) -> dict[str, TrackingValue]:
    """The shared shape for one selector's frozen epoch and its scores."""

    values: dict[str, TrackingValue] = {
        f"selected/{selector}/epoch": epoch,
        f"selected/{selector}/checkpoint_id": checkpoint_id,
    }
    values.update(
        {f"selected/{selector}/{name}": float(value) for name, value in metrics.items()}
    )
    return values


def final_values(
    selector: str,
    metrics: Mapping[str, float],
) -> dict[str, TrackingValue]:
    """Final metrics under the `final_test` prefix."""

    return {
        f"final_test/{selector}/{name}": float(value) for name, value in metrics.items()
    }


def _start(
    settings: TrackingSettings,
    project: str,
    *,
    group: str,
    name: str,
    job_type: str,
    config: Mapping[str, TrackingValue],
    backend: TrackingBackend | None = None,
) -> RunTracker:
    chosen = backend or WandbBackend()
    try:
        run = chosen.start(
            project=project,
            entity=settings.entity,
            mode=settings.mode,
            directory=settings.directory,
            group=group,
            name=name,
            job_type=job_type,
            config={
                **config,
                **{f"execution/{k}": v for k, v in execution_metadata().items()},
            },
        )
    except Exception as error:  # tracking must never reach the experiment
        _warn("start a run", error)
        return NullRunTracker()
    return MirrorRunTracker(run)


def _group(plan: SearchPlan) -> str:
    return f"{plan.experiment_name}:{_short(plan.canonical_digest())}"


def plan_track(plan: SearchPlan) -> str:
    """Every production search selects on validation data."""

    del plan
    return "ordinary"


def _plan_config(plan: SearchPlan) -> dict[str, TrackingValue]:
    config = plan.resolved_config.config
    lineage = plan.resolved_config.lineage
    # A resumed search executes new tasks with whatever code is checked out now, which
    # is not necessarily the revision that wrote the plan.
    executing = _executing_code()
    return {
        "dataset": plan.dataset,
        "protocol_id": plan.protocol_id,
        "experiment_name": plan.experiment_name,
        "experiment_variant": plan.experiment_variant,
        "normalization": plan.normalization,
        "plan_digest": plan.canonical_digest(),
        "batch_size": config.batch_size,
        "max_epochs": config.max_epochs,
        "pair_count": config.pair_count,
        "git_revision": executing[0],
        "git_dirty": executing[1],
        "plan_git_revision": plan.code.git_revision,
        "plan_git_dirty": plan.code.git_dirty,
        "dataset_manifest_digest": lineage.dataset_manifest_digest,
        "feature_cache_manifest_digest": lineage.feature_cache_manifest_digest,
        "pair_manifest_digest": lineage.pair_manifest_digest,
        "adjusted_weight_spec_digest": lineage.adjusted_weight_spec_digest,
        "held_out_validation_split": lineage.held_out_validation_split,
    }


def _executing_code() -> tuple[str, bool]:
    # Imported here: the search layer must be free to import this module.
    from grit.search.plan import current_code_provenance

    try:
        provenance = current_code_provenance()
    except Exception as error:  # tracking must never reach the experiment
        _warn("read the executing revision", error)
        return "unrecorded", False
    return provenance.git_revision, provenance.git_dirty


def training_metadata(dataset: str, algorithm: StrictBoundaryModel | None) -> str:
    """The per-example training annotation a base objective uses; pairs are separate.

    One flat field, so standalone and composed runs filter the same way.
    """

    base = (
        algorithm.base_objective
        if isinstance(algorithm, ComposedAlgorithmConfig)
        else algorithm
    )
    environment = "background" if dataset == "waterbirds_cf" else "source_environment"
    if isinstance(base, GroupDroAlgorithmConfig | LisaAlgorithmConfig):
        return {
            "target_color": "label_x_color",
            "target_background": "label_x_background",
            "training_environment": environment,
            "target_environment": f"label_x_{environment}",
        }[base.group_definition]
    if isinstance(
        base,
        RexAlgorithmConfig
        | IrmAlgorithmConfig
        | FishrAlgorithmConfig
        | RdmAlgorithmConfig,
    ):
        return environment
    return "none"


def _algorithm_config(
    algorithm: StrictBoundaryModel | None,
) -> dict[str, TrackingValue]:
    """Flatten one resolved algorithm configuration, fixed settings included."""

    if algorithm is None:
        return {}
    dumped = cast(dict[str, object], algorithm.model_dump(mode="json"))
    values: dict[str, TrackingValue] = {}
    for name, value in dumped.items():
        if isinstance(value, dict):
            for nested_name, nested_value in cast(dict[str, object], value).items():
                values[f"algorithm/{name}/{nested_name}"] = (
                    ",".join(str(item) for item in cast(list[object], nested_value))
                    if isinstance(nested_value, list)
                    else cast(TrackingValue, nested_value)
                )
        elif isinstance(value, list | tuple):
            items = cast("list[object] | tuple[object, ...]", value)
            values[f"algorithm/{name}"] = ",".join(str(item) for item in items)
        else:
            values[f"algorithm/{name}"] = cast(TrackingValue, value)
    return values


def _candidate_config(candidate: SearchCandidate) -> dict[str, TrackingValue]:
    dumped = cast(dict[str, object], candidate.model_dump(mode="json"))
    return {key: cast(TrackingValue, value) for key, value in dumped.items()}


def _short(value: str) -> str:
    return value.removeprefix("sha256:").split(":")[-1][:12]


def _warn(action: str, error: Exception) -> None:
    warnings.warn(
        f"experiment tracking could not {action}: {error!r}",
        RuntimeWarning,
        stacklevel=3,
    )
