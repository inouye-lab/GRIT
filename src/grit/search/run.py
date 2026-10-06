"""Dataset-neutral search entry points: plan, run, status, pilot, and outputs.

Dataset-specific orchestration lives in `grit.search.cmnist` and
`grit.search.waterbirds`; this module dispatches to them and owns everything the
datasets share.
"""

from __future__ import annotations

import hashlib
import json
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, TypeAlias, cast

from pydantic import Field, StrictInt, StrictStr, model_validator

from grit.methods.types import MethodId, pair_intervention
from grit.schemas import StrictBoundaryModel
from grit.search.execution import (
    SEQUENTIAL_EXECUTION,
    ExecutionOptions,
    current_execution_options,
    output_lock,
)
from grit.search.outputs import (
    CmnistPairedSelectorSummary,
    CmnistProductionSummary,
    ExperimentIndex,
    IndexedArtifact,
    WaterbirdsPairedSummaryArtifact,
    WaterbirdsProductionSummary,
    verify_experiment_index,
)
from grit.search.plan import (
    CmnistProductionSearchConfig,
    ResolvedProductionSearchConfig,
    SearchCandidate,
    SearchPlan,
    load_production_search_config,
    load_search_yaml,
    write_search_plan,
)

NonEmptyStr: TypeAlias = str


def is_dataset_search(config_path: Path, dataset: str) -> bool:
    """Dispatch compact dataset adapters through the shared script entry points."""
    if not config_path.is_file():
        # Leave missing paths to the ordinary loader's existing error handling.
        return False
    value = load_search_yaml(config_path)
    return (
        isinstance(value, dict)
        and cast(dict[str, object], value).get("dataset") == dataset
    )


class ProductionSearchStatus(StrictBoundaryModel):
    schema_version: Literal["grit.production-search-status/v1"]
    dataset: Literal["cmnist", "waterbirds_cf"]
    plan_digest: StrictStr = Field(min_length=1)
    phase: Literal[
        "tuning",
        "confirmation",
        "final",
        "complete",
    ]
    tuning_expected: StrictInt = Field(ge=0)
    tuning_complete: StrictInt = Field(ge=0)
    confirmation_expected: StrictInt = Field(ge=0)
    confirmation_complete: StrictInt = Field(ge=0)
    frozen_winner_count: StrictInt = Field(ge=0)
    final_expected: StrictInt = Field(ge=0)
    final_complete: StrictInt = Field(ge=0)

    @model_validator(mode="after")
    def _validate_counts(self) -> ProductionSearchStatus:
        if (
            self.tuning_complete > self.tuning_expected
            or self.confirmation_complete > self.confirmation_expected
            or self.final_complete > self.final_expected
        ):
            raise ValueError("production search status exceeds its planned work")
        return self


@dataclass(frozen=True, slots=True)
class ProductionExecutionLimits:
    """Operational task limits that never participate in scientific identity."""

    stop_after: Literal["tuning"] | None = None
    method: MethodId | Literal["all"] = "all"
    candidate_ids: tuple[str, ...] = ()
    tuning_seed: int | None = None
    max_new_runs: int | None = None


class PilotCandidateSelection(StrictBoundaryModel):
    """Read-only deterministic presentation of the two recommended pilot tasks."""

    schema_version: Literal["grit.production-pilot-candidates/v1"]
    dataset: Literal["cmnist", "waterbirds_cf"]
    plan_digest: StrictStr = Field(min_length=1)
    tuning_seed: StrictInt
    erm: SearchCandidate
    grit_nonzero_rank: SearchCandidate


def plan_production_search(config_path: Path) -> SearchPlan:
    """Validate all production inputs and atomically write or reuse the plan."""

    config = load_production_search_config(config_path)
    return write_search_plan(config, config_path=config_path)


def run_production_search(
    config_path: Path,
    limits: ProductionExecutionLimits | None = None,
    *,
    execution: ExecutionOptions = SEQUENTIAL_EXECUTION,
) -> CmnistProductionSummary | WaterbirdsProductionSummary | ProductionSearchStatus:
    """Run or continue the dataset-specific search selected by strict YAML."""

    plan = plan_production_search(config_path)
    checked_limits = validate_execution_limits(plan, limits)
    if isinstance(plan.resolved_config.config, CmnistProductionSearchConfig):
        from grit.search.cmnist import run_cmnist_search

        return run_cmnist_search(plan, checked_limits, execution=execution)
    from grit.search.waterbirds import run_waterbirds_production_search

    return run_waterbirds_production_search(plan, checked_limits, execution=execution)


def production_pilot_candidates(config_path: Path) -> PilotCandidateSelection:
    """Present canonical ERM/nonzero-GRIT pilot candidates from an existing plan."""

    plan = _load_existing_search_plan(config_path)
    by_method = {candidate.method_id: candidate for candidate in pilot_candidates(plan)}
    try:
        erm = by_method["erm"]
        grit = by_method["grit"]
    except KeyError as error:
        raise ValueError("legacy pilot view requires both ERM and GRIT") from error
    return PilotCandidateSelection(
        schema_version="grit.production-pilot-candidates/v1",
        dataset=plan.dataset,
        plan_digest=plan.canonical_digest(),
        tuning_seed=plan.seeds.stages.tuning[0],
        erm=erm,
        grit_nonzero_rank=grit,
    )


def pilot_candidates(plan: SearchPlan) -> tuple[SearchCandidate, ...]:
    """Choose one deterministic tuning candidate for every planned method."""

    selected: list[SearchCandidate] = []
    for method in plan.methods:
        selected.append(
            next(
                candidate
                for candidate in plan.candidates
                if candidate.method_id == method
                and (
                    pair_intervention(method) != "grit"
                    or (
                        candidate.requested_rank is not None
                        and candidate.requested_rank > 0
                    )
                )
            )
        )
    return tuple(selected)


def production_search_status(config_path: Path) -> ProductionSearchStatus:
    """Report canonical completed work without invoking a trainer."""

    plan = _load_existing_search_plan(config_path)
    if isinstance(plan.resolved_config.config, CmnistProductionSearchConfig):
        from grit.search.cmnist import cmnist_status_from_plan

        return cmnist_status_from_plan(plan)
    from grit.search.waterbirds import waterbirds_status_from_plan

    return waterbirds_status_from_plan(plan)


def _load_existing_search_plan(config_path: Path) -> SearchPlan:
    """Load one compatible planning triplet without resolving inputs or writing."""

    supplied_path = config_path.resolve()
    supplied = load_production_search_config(supplied_path)
    output_value = Path(supplied.output_root)
    output_root = (
        output_value.resolve()
        if output_value.is_absolute()
        else (supplied_path.parent / output_value).resolve()
    )
    authored_path = output_root / "authored-config.yaml"
    resolved_path = output_root / "resolved-config.json"
    plan_path = output_root / "search-plan.json"
    planning_paths = (authored_path, resolved_path, plan_path)
    if not all(path.is_file() for path in planning_paths):
        raise ValueError(
            "no complete production search plan exists; run scripts/run_search.py first"
        )
    stored_authored = load_production_search_config(authored_path)
    stored_resolved = ResolvedProductionSearchConfig.model_validate_json(
        resolved_path.read_text(encoding="utf-8")
    )
    stored_plan = SearchPlan.model_validate_json(plan_path.read_text(encoding="utf-8"))
    if stored_authored != supplied:
        raise ValueError("saved authored config does not match the supplied YAML")
    if (
        stored_resolved.config != supplied
        or Path(stored_resolved.output_root).resolve() != output_root
        or stored_plan.resolved_config != stored_resolved
    ):
        raise ValueError(
            "saved resolved config or search plan does not match the supplied YAML"
        )
    return stored_plan


def validate_execution_limits(
    plan: SearchPlan,
    limits: ProductionExecutionLimits | None,
) -> ProductionExecutionLimits | None:
    if limits is None or limits == ProductionExecutionLimits():
        return None
    if limits.stop_after not in {None, "tuning"}:
        raise ValueError("stop_after supports only the tuning boundary")
    if limits.method != "all" and limits.method not in plan.methods:
        choices = ", ".join((*plan.methods, "all"))
        raise ValueError(f"method must be one of: {choices}")
    max_new_runs = _validate_optional_exact_integer(limits.max_new_runs, "max_new_runs")
    tuning_seed = _validate_optional_exact_integer(limits.tuning_seed, "tuning_seed")
    candidate_ids = _validate_candidate_ids(limits.candidate_ids)
    if max_new_runs is not None and max_new_runs <= 0:
        raise ValueError("max_new_runs must be positive")
    if len(candidate_ids) != len(set(candidate_ids)):
        raise ValueError("candidate_id filters must be unique")
    uses_tuning_filter = (
        limits.method != "all" or bool(candidate_ids) or tuning_seed is not None
    )
    if uses_tuning_filter and limits.stop_after != "tuning":
        raise ValueError(
            "method, candidate, and tuning-seed filters require stop_after=tuning"
        )
    by_id = {candidate.candidate_id: candidate for candidate in plan.candidates}
    unknown = tuple(
        candidate_id for candidate_id in candidate_ids if candidate_id not in by_id
    )
    if unknown:
        raise ValueError(f"candidate IDs are not present in the plan: {unknown}")
    if limits.method != "all":
        wrong_method = tuple(
            candidate_id
            for candidate_id in candidate_ids
            if by_id[candidate_id].method_id != limits.method
        )
        if wrong_method:
            raise ValueError(
                f"candidate IDs do not agree with the method filter: {wrong_method}"
            )
    if tuning_seed is not None and tuning_seed not in plan.seeds.stages.tuning:
        raise ValueError("tuning_seed is not one of the configured tuning seeds")
    return limits


def _validate_optional_exact_integer(value: object, field_name: str) -> int | None:
    if value is None:
        return None
    if type(value) is not int:
        raise ValueError(f"{field_name} must be an integer")
    return value


def _validate_candidate_ids(values: tuple[object, ...]) -> tuple[str, ...]:
    validated: list[str] = []
    for value in values:
        if not isinstance(value, str) or not value:
            raise ValueError("candidate IDs must be nonempty strings")
        validated.append(value)
    return tuple(validated)


def limited_tuning_candidates(
    plan: SearchPlan,
    limits: ProductionExecutionLimits | None,
) -> tuple[SearchCandidate, ...]:
    if limits is None or limits.stop_after is None:
        return plan.candidates
    selected_ids = set(limits.candidate_ids)
    candidates = tuple(
        candidate
        for candidate in plan.candidates
        if (limits.method == "all" or candidate.method_id == limits.method)
        and (not selected_ids or candidate.candidate_id in selected_ids)
    )
    if not candidates:
        raise ValueError("tuning filters select no canonical plan candidates")
    return candidates


def planned_candidate(
    candidates: dict[str, SearchCandidate], candidate_id: str
) -> SearchCandidate:
    candidate = candidates.get(candidate_id)
    if candidate is None:
        raise ValueError(
            f"selection artifact references an out-of-plan candidate: {candidate_id}"
        )
    return candidate


def complete_outputs_valid(plan: SearchPlan) -> bool:
    root = Path(plan.resolved_config.output_root)
    index_path = root / "experiment-index.json"
    if plan.dataset == "cmnist":
        summary_path = root / "summaries" / "cmnist-summary.json"
        paired_paths = (
            tuple(
                root / "summaries" / f"cmnist-{selector}-paired-differences.json"
                for selector in plan.selectors
            )
            if "erm" in plan.methods and "grit" in plan.methods
            else ()
        )
        required = (summary_path, *paired_paths, index_path)
        if not all(path.is_file() for path in required):
            return False
        summary = CmnistProductionSummary.model_validate_json(
            summary_path.read_text(encoding="utf-8")
        )
        paired = tuple(
            CmnistPairedSelectorSummary.model_validate_json(
                path.read_text(encoding="utf-8")
            )
            for path in paired_paths
        )
        if (
            summary.plan_digest != plan.canonical_digest()
            or summary.lineage != plan.resolved_config.lineage
            or tuple(item.value for item in summary.selectors) != plan.selectors
            or summary.paired_selectors != paired
        ):
            raise ValueError("CMNIST completion summary is inconsistent")
    else:
        summary_path = root / "summaries" / "waterbirds-summary.json"
        paired_path = root / "summaries" / "waterbirds-paired-differences.json"
        pairable = "erm" in plan.methods and "grit" in plan.methods
        required = (summary_path, index_path, *((paired_path,) if pairable else ()))
        if not all(path.is_file() for path in required):
            return False
        summary = WaterbirdsProductionSummary.model_validate_json(
            summary_path.read_text(encoding="utf-8")
        )
        if (
            summary.plan_digest != plan.canonical_digest()
            or summary.lineage != plan.resolved_config.lineage
            or (summary.selector,) != plan.selectors
            or summary.paired != pairable
        ):
            raise ValueError("Waterbirds completion summary is inconsistent")
        if pairable:
            paired = WaterbirdsPairedSummaryArtifact.model_validate_json(
                paired_path.read_text(encoding="utf-8")
            )
            if (
                paired.configured_final_seeds != plan.seeds.stages.final
                or paired.paired_worst_group_by_seed
                != summary.paired_worst_group_by_seed
                or paired.paired_worst_group_summary
                != summary.paired_worst_group_summary
            ):
                raise ValueError("Waterbirds completion summary is inconsistent")
    index = ExperimentIndex.model_validate_json(index_path.read_text(encoding="utf-8"))
    verify_experiment_index(root, index)
    return True


def persist_canonical_artifact(path: Path, model: StrictBoundaryModel) -> None:
    # Workers can discover the same projection rank concurrently. Serialize only
    # publication, not fitting, and retain the existing immutable-content check.
    lock = (
        output_lock(path.parent / ".artifacts.lock", blocking=True)
        if current_execution_options().workers > 1
        else nullcontext()
    )
    with lock:
        _persist_canonical_artifact(path, model)


def _persist_canonical_artifact(path: Path, model: StrictBoundaryModel) -> None:
    payload = model.canonical_json() + "\n"
    if path.exists():
        if not path.is_file() or path.read_text(encoding="utf-8") != payload:
            raise ValueError(f"refusing to overwrite incompatible artifact: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(payload, encoding="utf-8")
    temporary.replace(path)


def write_experiment_index(plan: SearchPlan, output_root: Path) -> ExperimentIndex:
    index_path = output_root / "experiment-index.json"
    artifacts: list[IndexedArtifact] = []
    paths = tuple(output_root.rglob("*.json")) + tuple(output_root.rglob("*.yaml"))
    for path in sorted(paths):
        if path == index_path or any(
            ".incomplete" in part or ".interrupted-" in part for part in path.parts
        ):
            continue
        relative = path.relative_to(output_root).as_posix()
        schema_version: str | None = None
        try:
            loaded = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                mapping = cast(dict[str, object], loaded)
                value = mapping.get("schema_version")
                if isinstance(value, str):
                    schema_version = value
        except (OSError, ValueError):
            schema_version = None
        artifacts.append(
            IndexedArtifact(
                kind=_artifact_kind(relative),
                relative_path=relative,
                sha256=_file_sha256(path),
                schema_version=schema_version,
            )
        )
    index = ExperimentIndex(
        schema_version="grit.experiment-index/v1",
        dataset=plan.dataset,
        reportable=True,
        plan_digest=plan.canonical_digest(),
        resolved_config_digest=plan.resolved_config.canonical_digest(),
        artifacts=tuple(artifacts),
    )
    persist_canonical_artifact(index_path, index)
    return verify_experiment_index(output_root, index)


def _artifact_kind(relative: str) -> str:
    name = Path(relative).name
    if name == "result.json":
        return "stage_run"
    if name == "final-result.json":
        return "final_result"
    if "checkpoint" in name or "selected-checkpoint" in relative:
        return "selected_checkpoint_manifest"
    if "summary" in name:
        return "dataset_summary"
    if "winner" in name:
        return "frozen_winner"
    if "finalist" in name or "union" in name:
        return "selection_artifact"
    if "projection" in relative:
        return "projection_diagnostics"
    if name == "search-plan.json":
        return "search_plan"
    if name == "resolved-config.json":
        return "resolved_config"
    return "canonical_json"


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return f"sha256:{digest.hexdigest()}"


def completed_task_revisions(output_root: Path) -> dict[str, int]:
    """Count completed task results by the code revision that produced them."""

    counts: dict[str, int] = {}
    for result_path in sorted(output_root.glob("runs/**/result.json")):
        try:
            payload = json.loads(result_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        code = cast(dict[str, object], payload.get("code") or {})
        revision = str(code.get("git_revision", "unrecorded"))[:12]
        if code.get("git_dirty"):
            revision += " (dirty)"
        counts[revision] = counts.get(revision, 0) + 1
    return counts
