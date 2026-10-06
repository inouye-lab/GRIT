"""Waterbirds-CF adapter for the shared production local-search lifecycle.

Candidates and checkpoints are selected by validation worst-group accuracy. Every
implemented method binds to the same trainer through `_waterbirds_method`; the dataset
supplies only the environment, group, and pair bindings the method definitions require.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, cast

from grit.config import (
    ComposedAlgorithmConfig,
    FishrAlgorithmConfig,
    GroupDroAlgorithmConfig,
    IrmAlgorithmConfig,
    LinearProbeTrainingConfig,
    LisaAlgorithmConfig,
    RdmAlgorithmConfig,
    RexAlgorithmConfig,
    SdAlgorithmConfig,
)
from grit.data.waterbirds import (
    WaterbirdsAdjustedWeightSpec,
    WaterbirdsDatasetManifest,
    mint_waterbirds_adjusted_weight_spec,
)
from grit.data.waterbirds_pairs import (
    WaterbirdsOraclePairManifest,
    WaterbirdsOraclePairSet,
)
from grit.features.waterbirds import (
    WaterbirdsFeatureCache,
    WaterbirdsTuningFeatureCache,
    fit_waterbirds_oracle_projection,
    load_waterbirds_feature_cache,
    load_waterbirds_tuning_feature_cache,
)
from grit.methods.baselines import (
    LisaLinearProbeMethod,
    SpectralDecouplingLinearProbeMethod,
    bind_risk_matching_method,
)
from grit.methods.interventions import ComposedLinearProbeMethod
from grit.methods.pair_inputs import PairInputs
from grit.methods.projection import FittedLinearProjection
from grit.methods.training import (
    GroupDroLinearProbeMethod,
    IrmLinearProbeMethod,
    LinearProbeTrainingMethod,
    OrdinaryLinearProbeMethod,
    PersistedLinearCheckpointStore,
    RexLinearProbeMethod,
    persist_selected_linear_checkpoint,
    persist_training_diagnostics,
)
from grit.methods.types import (
    base_objective,
    consumes_pairs,
    is_composed,
    pair_intervention,
)
from grit.methods.waterbirds_training import (
    TrainedWaterbirdsRun,
    WaterbirdsMethod,
    restore_waterbirds_checkpoint,
    train_waterbirds_linear_probe,
    waterbirds_metric_values,
)
from grit.schemas import SeedStage
from grit.search.execution import (
    SEQUENTIAL_EXECUTION,
    ExecutionOptions,
    execution_scope,
)
from grit.search.lifecycle import (
    ProductionLifecycleHooks,
    ProductionStatusHooks,
    production_lifecycle_status,
    run_production_lifecycle,
)
from grit.search.outputs import (
    WaterbirdsPairedSummaryArtifact,
    WaterbirdsProductionMethodSummary,
    WaterbirdsProductionSummary,
)
from grit.search.pair_inputs import waterbirds_pair_inputs
from grit.search.plan import (
    SearchCandidate,
    SearchPlan,
    WaterbirdsProductionSearchConfig,
    current_code_provenance,
    current_environment_provenance,
    training_optimizer,
    waterbirds_candidate_algorithm,
)
from grit.search.run import (
    ProductionExecutionLimits,
    ProductionSearchStatus,
    persist_canonical_artifact,
)
from grit.search.scheduler import (
    CompletedStageRun,
    SearchRunTask,
    StageExecutor,
    WaterbirdsCompletedStageRun,
    make_final_search_task,
    make_search_task,
)
from grit.search.waterbirds_contracts import (
    WaterbirdsArtifactReference,
    WaterbirdsCandidateConfig,
    WaterbirdsCheckpointArtifactReference,
    WaterbirdsDatasetArtifactReference,
    WaterbirdsFeatureArtifactReference,
    WaterbirdsPairArtifactReference,
    WaterbirdsProjectionArtifactReference,
    WaterbirdsRunResult,
    make_waterbirds_metric_summary,
)
from grit.selection.cmnist import CheckpointIdentity
from grit.selection.waterbirds import (
    ORDINARY_WATERBIRDS_SELECTOR,
    FrozenWaterbirdsCandidate,
    WaterbirdsCheckpointSelection,
    WaterbirdsSelector,
    WaterbirdsSelectorRecord,
    WaterbirdsTuningFinalists,
    compute_waterbirds_final_metric,
    freeze_waterbirds_candidate,
    freeze_waterbirds_final_checkpoint,
    make_waterbirds_tuning_finalists,
    select_confirmed_waterbirds_candidate,
    select_waterbirds_checkpoint,
)
from grit.tracking import (
    RunTracker,
    TrackingValue,
    final_values,
    log_summary_table,
    selection_values,
    task_tracker,
)

WATERBIRDS_GROUP_COUNT = 4
WATERBIRDS_ENVIRONMENT_COUNT = 2

WaterbirdsFinalists = dict[WaterbirdsMethod, WaterbirdsTuningFinalists]
WaterbirdsWinners = dict[WaterbirdsMethod, FrozenWaterbirdsCandidate]


@dataclass(frozen=True, slots=True)
class _RuntimeCandidate:
    planned: SearchCandidate
    config: WaterbirdsCandidateConfig
    projection: FittedLinearProjection | None
    pairs: PairInputs | None


def waterbirds_selector(config: WaterbirdsProductionSearchConfig) -> WaterbirdsSelector:
    return config.selectors[0]


FINALISTS_FILENAME = "tuning-finalists.json"
WINNER_FILENAME = "winner.json"


def run_waterbirds_production_search(
    plan: SearchPlan,
    limits: ProductionExecutionLimits | None = None,
    *,
    execution: ExecutionOptions = SEQUENTIAL_EXECUTION,
) -> WaterbirdsProductionSummary | ProductionSearchStatus:
    with execution_scope(Path(plan.resolved_config.output_root), execution):
        return run_production_lifecycle(
            plan,
            limits,
            build_waterbirds_hooks(plan, limits),
            hooks_factory=build_waterbirds_hooks,
            execution=execution,
        )


def build_waterbirds_hooks(
    plan: SearchPlan,
    limits: ProductionExecutionLimits | None = None,
) -> ProductionLifecycleHooks[
    WaterbirdsCompletedStageRun,
    WaterbirdsFinalists,
    WaterbirdsWinners,
    WaterbirdsProductionSummary,
]:
    """Bind Waterbirds caches and executors to the shared lifecycle callbacks."""

    config = _search_config(plan)
    selector = waterbirds_selector(config)
    output_root = Path(plan.resolved_config.output_root)
    dataset = _dataset_manifest(plan)
    pairs = (
        WaterbirdsOraclePairSet(manifest=_pair_manifest(plan))
        if any(consumes_pairs(method) for method in plan.methods)
        else None
    )
    # Tuning never reads the test split, so it runs from the redacted cache.
    tuning_only = limits is not None and limits.stop_after == "tuning"
    cache = _load_cache(plan, tuning_only=tuning_only)
    weights = mint_waterbirds_adjusted_weight_spec(dataset)
    if weights.canonical_digest() != (
        plan.resolved_config.lineage.adjusted_weight_spec_digest
    ):
        raise ValueError("Waterbirds adjusted weights changed after planning")
    projections: dict[int, FittedLinearProjection] = {}
    pair_bank: dict[str, PairInputs] = {}

    def runtime_candidate(candidate: SearchCandidate) -> _RuntimeCandidate:
        projection: FittedLinearProjection | None = None
        pair_inputs: PairInputs | None = None
        if consumes_pairs(candidate.method_id) and pairs is None:
            raise AssertionError("pair-consuming candidate requires training pairs")
        if pair_intervention(candidate.method_id) == "grit":
            assert pairs is not None
            rank = candidate.requested_rank
            if rank is None:
                raise AssertionError("planned Waterbirds GRIT candidate lacks a rank")
            projection = projections.get(rank)
            if projection is None:
                projection = fit_waterbirds_oracle_projection(
                    cache,
                    pairs,
                    requested_rank=rank,
                    relative_singular_value_tolerance=(
                        config.relative_singular_value_tolerance
                    ),
                )
                persist_canonical_artifact(
                    output_root / "projections" / f"grit-rank-{rank}.json",
                    projection.diagnostics,
                )
                projections[rank] = projection
        elif consumes_pairs(candidate.method_id):
            assert pairs is not None
            pair_inputs = pair_bank.get("land_minus_water")
            if pair_inputs is None:
                pair_inputs = waterbirds_pair_inputs(cache, pairs)
                pair_bank["land_minus_water"] = pair_inputs
        resolved = materialize_waterbirds_candidate_config(
            plan, candidate, projection, selector
        )
        return _RuntimeCandidate(candidate, resolved, projection, pair_inputs)

    def execute_pre_final(task: SearchRunTask, run_root: Path) -> CompletedStageRun:
        runtime = runtime_candidate(task.candidate)
        with task_tracker(plan, task, algorithm=runtime.config.algorithm) as tracker:
            trained = _train_task(cache, weights, runtime, task, tracker)
            persist_training_diagnostics(trained.algorithm, run_root)
            decision = select_waterbirds_checkpoint(
                trained.selector_records(selector), selector
            )
            tracker.record_selection(_waterbirds_selection_values(decision))
            return WaterbirdsCompletedStageRun(
                schema_version="grit.waterbirds-search-stage-run/v1",
                dataset="waterbirds_cf",
                status="complete",
                task=task,
                lineage=plan.resolved_config.lineage,
                code=current_code_provenance(),
                environment=current_environment_provenance(),
                validation_metrics=trained.validation_metrics,
                checkpoint_decision=decision,
            )

    candidates_by_id = {item.candidate_id: item for item in plan.candidates}

    def coerce_runs(
        runs: tuple[CompletedStageRun, ...],
    ) -> tuple[WaterbirdsCompletedStageRun, ...]:
        if any(not isinstance(run, WaterbirdsCompletedStageRun) for run in runs):
            raise ValueError("Waterbirds lifecycle received another dataset result")
        return cast(tuple[WaterbirdsCompletedStageRun, ...], runs)

    def confirmation_tasks(
        finalists: WaterbirdsFinalists,
    ) -> tuple[SearchRunTask, ...]:
        return tuple(
            make_search_task(
                plan,
                candidates_by_id[item.candidate_id],
                SeedStage.CONFIRMATION,
                seed,
            )
            for method in plan.methods
            for item in finalists[method].ordered_candidates
            for seed in config.seeds.stages.confirmation
        )

    def make_winners(
        finalists: WaterbirdsFinalists,
        runs: tuple[WaterbirdsCompletedStageRun, ...],
        root: Path,
    ) -> WaterbirdsWinners:
        return _freeze_winners(plan, finalists, runs, root)

    def final_tasks(winners: WaterbirdsWinners) -> tuple[SearchRunTask, ...]:
        return tuple(
            make_final_search_task(
                plan,
                candidates_by_id[winners[method].candidate_id],
                seed,
                winners[method],
            )
            for method in plan.methods
            for seed in config.seeds.stages.final
        )

    def final_executor(winners: WaterbirdsWinners) -> StageExecutor:
        if isinstance(cache, WaterbirdsTuningFeatureCache):
            raise AssertionError("tuning-only Waterbirds execution reached final stage")
        full_cache = cache

        def execute_final(task: SearchRunTask, run_root: Path) -> CompletedStageRun:
            runtime = runtime_candidate(task.candidate)
            with task_tracker(
                plan, task, algorithm=runtime.config.algorithm
            ) as tracker:
                return execute_final_tracked(task, run_root, tracker, runtime)

        def execute_final_tracked(
            task: SearchRunTask,
            run_root: Path,
            tracker: RunTracker,
            runtime: _RuntimeCandidate,
        ) -> CompletedStageRun:
            frozen = winners[task.candidate.method_id]
            trained = _train_task(full_cache, weights, runtime, task, tracker)
            persist_training_diagnostics(trained.algorithm, run_root)
            decision = select_waterbirds_checkpoint(
                trained.selector_records(selector), selector
            )
            tracker.record_selection(_waterbirds_selection_values(decision))
            frozen_checkpoint = freeze_waterbirds_final_checkpoint(decision, frozen)
            selected = trained.store.load(frozen_checkpoint.checkpoint.checkpoint_id)
            checkpoint_root = run_root / "selected-checkpoint"
            checkpoint_manifest = persist_selected_linear_checkpoint(
                selected, checkpoint_root
            )
            persisted = PersistedLinearCheckpointStore(checkpoint_root)
            restoration = restore_waterbirds_checkpoint(
                frozen_checkpoint, persisted, trained.algorithm
            )
            artifacts = _result_artifacts(
                plan,
                task,
                checkpoint_manifest.canonical_digest(),
                runtime,
                frozen_checkpoint.checkpoint,
            )
            handle = full_cache.issue_final_handle(
                run_id=trained.run_id,
                candidate_id=task.candidate.candidate_id,
                method_id=task.candidate.method_id,
                scientific_config_digest=task.candidate.scientific_config_digest,
                seed=task.seed,
                projection_rank=task.candidate.requested_rank,
            )
            final_view = handle.open(frozen, frozen_checkpoint, restoration)
            final_table = full_cache.verify_final_view(final_view)
            predictions = trained.algorithm.predict(final_table.features)
            final_metric = compute_waterbirds_final_metric(
                final_view,
                predictions,
                adjusted_weights=weights,
                record_id=f"metric:{trained.run_id}:test",
            )
            result = WaterbirdsRunResult(
                schema_version="grit.waterbirds-run-result/v3",
                result_kind="ordinary_waterbirds",
                status="succeeded",
                run_id=trained.run_id,
                final_seed=task.seed,
                resolved_config=runtime.config,
                resolved_config_digest=runtime.config.canonical_digest(),
                code=current_code_provenance(),
                environment=current_environment_provenance(),
                validation_metrics=trained.validation_metrics,
                candidate_selection=frozen,
                checkpoint_selection=frozen_checkpoint,
                restoration=restoration,
                final_test_metric=final_metric,
                selected_checkpoint_manifest_digest=(
                    checkpoint_manifest.canonical_digest()
                ),
                artifacts=artifacts,
            )
            tracker.record_selection(
                final_values(selector, waterbirds_metric_values(final_metric))
            )
            persist_canonical_artifact(run_root / "final-result.json", result)
            completed = WaterbirdsCompletedStageRun(
                schema_version="grit.waterbirds-search-stage-run/v1",
                dataset="waterbirds_cf",
                status="complete",
                task=task,
                lineage=plan.resolved_config.lineage,
                code=current_code_provenance(),
                environment=current_environment_provenance(),
                validation_metrics=trained.validation_metrics,
                checkpoint_decision=decision,
                final_result_relative_path="final-result.json",
                final_result_digest=result.canonical_digest(),
                final_result=result,
            )
            persist_canonical_artifact(run_root / "result.json", completed)
            return completed

        return execute_final

    def persist_summary(summary: WaterbirdsProductionSummary, root: Path) -> None:
        persist_canonical_artifact(
            root / "summaries" / "waterbirds-summary.json", summary
        )
        if (
            summary.paired_worst_group_by_seed is not None
            and summary.paired_worst_group_summary is not None
        ):
            persist_canonical_artifact(
                root / "summaries" / "waterbirds-paired-differences.json",
                WaterbirdsPairedSummaryArtifact(
                    schema_version="grit.waterbirds-paired-summary/v1",
                    configured_final_seeds=plan.seeds.stages.final,
                    paired_worst_group_by_seed=summary.paired_worst_group_by_seed,
                    paired_worst_group_summary=summary.paired_worst_group_summary,
                ),
            )
        _mirror_waterbirds_summary(plan, summary)

    hooks = ProductionLifecycleHooks(
        coerce_runs=coerce_runs,
        execute_pre_final=execute_pre_final,
        make_finalists=lambda runs, root: _finalists(plan, runs, root),
        confirmation_tasks=confirmation_tasks,
        make_winners=make_winners,
        final_tasks=final_tasks,
        final_executor=final_executor,
        make_summary=lambda finalists, winners, runs: _summary(
            plan, finalists, winners, runs
        ),
        persist_summary=persist_summary,
        status=lambda: waterbirds_status_from_plan(plan),
    )
    return hooks


def _finalists(
    plan: SearchPlan,
    runs: tuple[WaterbirdsCompletedStageRun, ...],
    output_root: Path,
) -> WaterbirdsFinalists:
    artifacts = compute_waterbirds_finalists(plan, runs)
    for method, finalists in artifacts.items():
        persist_canonical_artifact(
            output_root / "selection" / method / FINALISTS_FILENAME,
            finalists,
        )
    return artifacts


def _selector_records(
    runs: tuple[WaterbirdsCompletedStageRun, ...],
    method: WaterbirdsMethod,
    selector: WaterbirdsSelector,
) -> tuple[WaterbirdsSelectorRecord, ...]:
    return tuple(
        record
        for run in runs
        if run.task.candidate.method_id == method
        for record in run.selector_records(selector)
    )


def compute_waterbirds_finalists(
    plan: SearchPlan,
    runs: tuple[WaterbirdsCompletedStageRun, ...],
) -> WaterbirdsFinalists:
    selector = waterbirds_selector(_search_config(plan))
    artifacts: WaterbirdsFinalists = {}
    for method in plan.methods:
        artifacts[method] = make_waterbirds_tuning_finalists(
            _selector_records(runs, method, selector), plan.seeds.stages, selector
        )
    return artifacts


def _freeze_winners(
    plan: SearchPlan,
    finalists: WaterbirdsFinalists,
    runs: tuple[WaterbirdsCompletedStageRun, ...],
    output_root: Path,
) -> WaterbirdsWinners:
    winners = compute_waterbirds_winners(plan, finalists, runs)
    for method, frozen in winners.items():
        persist_canonical_artifact(
            output_root / "selection" / method / WINNER_FILENAME, frozen
        )
    return winners


def compute_waterbirds_winners(
    plan: SearchPlan,
    finalists: WaterbirdsFinalists,
    runs: tuple[WaterbirdsCompletedStageRun, ...],
) -> WaterbirdsWinners:
    selector = waterbirds_selector(_search_config(plan))
    winners: WaterbirdsWinners = {}
    for method in plan.methods:
        artifact = finalists[method]
        finalist_ids = {item.candidate_id for item in artifact.ordered_candidates}
        records = tuple(
            item
            for item in _selector_records(runs, method, selector)
            if item.candidate_id in finalist_ids
        )
        decision = select_confirmed_waterbirds_candidate(
            records, artifact, plan.seeds.stages
        )
        winners[method] = freeze_waterbirds_candidate(
            decision, artifact, plan.seeds.stages
        )
    return winners


def materialize_waterbirds_candidate_config(
    plan: SearchPlan,
    candidate: SearchCandidate,
    projection: FittedLinearProjection | None,
    selector: WaterbirdsSelector = ORDINARY_WATERBIRDS_SELECTOR,
) -> WaterbirdsCandidateConfig:
    config = _search_config(plan)
    lineage = plan.resolved_config.lineage
    weights = lineage.adjusted_weight_spec_digest
    if weights is None:
        raise ValueError("Waterbirds plan lacks adjusted-weight lineage")
    pair_digest = (
        lineage.pair_manifest_digest if consumes_pairs(candidate.method_id) else None
    )
    projection_digest = None
    rank = None
    tolerance = None
    if pair_intervention(candidate.method_id) == "grit":
        if projection is None or candidate.requested_rank is None:
            raise ValueError("Waterbirds GRIT candidate lacks fitted projection")
        projection_digest = projection.diagnostics.canonical_digest()
        rank = candidate.requested_rank
        tolerance = float(config.relative_singular_value_tolerance)
    elif projection is not None:
        raise ValueError("only Waterbirds GRIT candidates carry a projection")
    resolved = WaterbirdsCandidateConfig(
        schema_version="grit.waterbirds-candidate/v3",
        protocol_id="waterbirds_cf/v1",
        non_reportable=False,
        method_id=candidate.method_id,
        selector=selector,
        dataset_profile="production",
        dataset_manifest_digest=lineage.dataset_manifest_digest,
        feature_cache_manifest_digest=lineage.feature_cache_manifest_digest,
        normalization=config.normalization,
        adjusted_weight_spec_digest=weights,
        pair_manifest_digest=pair_digest,
        projection_diagnostics_digest=projection_digest,
        projection_rank=rank,
        relative_singular_value_tolerance=tolerance,
        algorithm=waterbirds_candidate_algorithm(
            config, candidate.method_id, candidate.settings()
        ),
        training=LinearProbeTrainingConfig(
            optimizer=training_optimizer(candidate.method_id),
            batch_size=config.batch_size,
            learning_rate=float(candidate.learning_rate),
            weight_decay=float(candidate.weight_decay),
            max_epochs=config.max_epochs,
        ),
        seed_sets=config.seeds.stages,
    )
    if resolved.scientific_config_digest() != candidate.scientific_config_digest:
        raise ValueError("planned Waterbirds candidate digest cannot be materialized")
    return resolved


def _train_task(
    cache: WaterbirdsFeatureCache | WaterbirdsTuningFeatureCache,
    weights: WaterbirdsAdjustedWeightSpec,
    runtime: _RuntimeCandidate,
    task: SearchRunTask,
    tracker: RunTracker | None = None,
) -> TrainedWaterbirdsRun:
    run_id = f"run:{task.task_id}"
    return train_waterbirds_linear_probe(
        cache.training_table(),
        cache.validation_table(),
        weights,
        runtime.config.training,
        run_id=run_id,
        candidate_id=task.candidate.candidate_id,
        scientific_config_digest=task.candidate.scientific_config_digest,
        seed_stage=task.stage,
        seed=task.seed,
        method=_waterbirds_method(cache, runtime, task),
        tracker=tracker,
    )


def _waterbirds_method(
    cache: WaterbirdsFeatureCache | WaterbirdsTuningFeatureCache,
    runtime: _RuntimeCandidate,
    task: SearchRunTask,
) -> LinearProbeTrainingMethod:
    algorithm = runtime.config.algorithm
    if not isinstance(algorithm, ComposedAlgorithmConfig):
        return _waterbirds_base_method(cache, runtime, task)
    base_runtime = _RuntimeCandidate(
        runtime.planned,
        runtime.config.model_copy(update={"algorithm": algorithm.base_objective}),
        None,
        None,
    )
    return ComposedLinearProbeMethod(
        base=_waterbirds_base_method(cache, base_runtime, task),
        method_id=task.candidate.method_id,
        projection=runtime.projection,
        projection_rank=task.candidate.requested_rank,
        pair_differences=None if runtime.pairs is None else runtime.pairs.differences,
        consistency_weight=algorithm.consistency_weight or 0.0,
    )


def _waterbirds_base_method(
    cache: WaterbirdsFeatureCache | WaterbirdsTuningFeatureCache,
    runtime: _RuntimeCandidate,
    task: SearchRunTask,
) -> LinearProbeTrainingMethod:
    """Bind one planned candidate's algorithm config to the shared trainer.

    Backgrounds and groups are read through the cache's method-definition accessors;
    ERM, GRIT, and SD never touch them.
    """

    algorithm = runtime.config.algorithm
    if isinstance(algorithm, SdAlgorithmConfig):
        return SpectralDecouplingLinearProbeMethod(
            penalty_weight=float(algorithm.penalty_weight)
        )
    if isinstance(
        algorithm,
        RexAlgorithmConfig
        | IrmAlgorithmConfig
        | FishrAlgorithmConfig
        | RdmAlgorithmConfig,
    ):
        environment_ids = cache.training_environment_ids()
        if isinstance(algorithm, FishrAlgorithmConfig | RdmAlgorithmConfig):
            # The shared binder also applies the warm-up guard, which needs this
            # candidate's actual epoch length.
            return bind_risk_matching_method(
                algorithm,
                environment_ids=environment_ids,
                environment_count=WATERBIRDS_ENVIRONMENT_COUNT,
                training=runtime.config.training,
            )
        if isinstance(algorithm, RexAlgorithmConfig):
            return RexLinearProbeMethod(
                environment_ids=environment_ids,
                environment_count=WATERBIRDS_ENVIRONMENT_COUNT,
                penalty_weight=float(algorithm.penalty_weight),
                penalty_anneal_updates=int(algorithm.penalty_anneal_updates),
            )
        return IrmLinearProbeMethod(
            environment_ids=environment_ids,
            environment_count=WATERBIRDS_ENVIRONMENT_COUNT,
            penalty_weight=float(algorithm.penalty_weight),
            penalty_anneal_updates=int(algorithm.penalty_anneal_updates),
        )
    if isinstance(algorithm, GroupDroAlgorithmConfig | LisaAlgorithmConfig):
        if (
            isinstance(algorithm, GroupDroAlgorithmConfig)
            and algorithm.group_definition == "training_environment"
        ):
            # Background groups alone, as the environment methods receive them.
            return GroupDroLinearProbeMethod(
                group_ids=cache.training_environment_ids(),
                group_count=WATERBIRDS_ENVIRONMENT_COUNT,
                step_size=float(algorithm.adversarial_step_size),
            )
        if algorithm.group_definition != "target_background":
            raise ValueError(
                f"Waterbirds {algorithm.kind} cannot use {algorithm.group_definition}"
            )
        group_ids = cache.training_group_ids()
        if isinstance(algorithm, GroupDroAlgorithmConfig):
            return GroupDroLinearProbeMethod(
                group_ids=group_ids,
                group_count=WATERBIRDS_GROUP_COUNT,
                step_size=float(algorithm.adversarial_step_size),
            )
        return LisaLinearProbeMethod(
            group_ids=group_ids,
            group_count=WATERBIRDS_GROUP_COUNT,
            selection_prob=float(algorithm.selection_prob),
        )
    return OrdinaryLinearProbeMethod(
        method_id=(
            base_objective(task.candidate.method_id)
            if is_composed(task.candidate.method_id)
            else task.candidate.method_id
        ),
        projection=runtime.projection,
        projection_rank=(
            task.candidate.requested_rank if runtime.projection is not None else None
        ),
    )


def _result_artifacts(
    plan: SearchPlan,
    task: SearchRunTask,
    checkpoint_digest: str,
    runtime: _RuntimeCandidate,
    checkpoint: CheckpointIdentity,
) -> tuple[WaterbirdsArtifactReference, ...]:
    root = Path(plan.resolved_config.output_root)
    inputs = {item.kind: item for item in plan.resolved_config.input_artifacts}
    config = runtime.config
    values: list[WaterbirdsArtifactReference] = [
        WaterbirdsDatasetArtifactReference(
            artifact_id="waterbirds-cf-dataset",
            kind="dataset_manifest",
            relative_uri=os.path.relpath(inputs["dataset_manifest"].path, root),
            digest=inputs["dataset_manifest"].digest,
        ),
        WaterbirdsFeatureArtifactReference(
            artifact_id="waterbirds-feature-cache",
            kind="feature_manifest",
            relative_uri=os.path.relpath(inputs["feature_manifest"].path, root),
            digest=inputs["feature_manifest"].digest,
            dataset_manifest_digest=config.dataset_manifest_digest,
            normalization=config.normalization,
        ),
        WaterbirdsCheckpointArtifactReference(
            artifact_id=f"checkpoint:{task.task_id}",
            kind="selected_linear_checkpoint",
            relative_uri=(
                f"{task.relative_directory}/selected-checkpoint/manifest.json"
            ),
            digest=checkpoint_digest,
            checkpoint=checkpoint,
        ),
    ]
    if config.pair_manifest_digest is not None:
        values.append(
            WaterbirdsPairArtifactReference(
                artifact_id="waterbirds-oracle-pairs",
                kind="pair_manifest",
                relative_uri=os.path.relpath(inputs["pair_manifest"].path, root),
                digest=config.pair_manifest_digest,
                dataset_manifest_digest=config.dataset_manifest_digest,
            )
        )
    if pair_intervention(task.candidate.method_id) == "grit":
        projection = runtime.projection
        if projection is None or config.pair_manifest_digest is None:
            raise AssertionError("Waterbirds GRIT artifact lineage disappeared")
        values.append(
            WaterbirdsProjectionArtifactReference(
                artifact_id=f"projection:grit:{task.candidate.requested_rank}",
                kind="projection_diagnostics",
                relative_uri=(
                    f"projections/grit-rank-{task.candidate.requested_rank}.json"
                ),
                digest=projection.diagnostics.canonical_digest(),
                pair_manifest_digest=config.pair_manifest_digest,
                feature_cache_manifest_digest=config.feature_cache_manifest_digest,
                normalization=config.normalization,
                requested_rank=projection.diagnostics.requested_rank,
            )
        )
    return tuple(values)


def _summary(
    plan: SearchPlan,
    finalists: WaterbirdsFinalists,
    winners: WaterbirdsWinners,
    runs: tuple[WaterbirdsCompletedStageRun, ...],
) -> WaterbirdsProductionSummary:
    from grit.methods.types import consumes_pairs
    from grit.search.outputs import make_waterbirds_intervention_comparisons

    matrix = any(method.endswith(("_grit", "_consistency")) for method in plan.methods)

    selector = waterbirds_selector(_search_config(plan))
    by_identity = {(run.task.candidate.method_id, run.task.seed): run for run in runs}
    methods: list[WaterbirdsProductionMethodSummary] = []
    for method in plan.methods:
        paths: list[tuple[int, str]] = []
        worst: list[tuple[int, float]] = []
        adjusted: list[tuple[int, float]] = []
        raw: list[tuple[int, float]] = []
        for seed in plan.seeds.stages.final:
            run = by_identity[(method, seed)]
            result = run.final_result
            if result is None:
                raise AssertionError("completed Waterbirds final run lacks result")
            metric = result.reported_test_metric
            paths.append((seed, f"{run.task.relative_directory}/final-result.json"))
            worst.append((seed, float(metric.worst_group_accuracy)))
            adjusted.append((seed, float(metric.adjusted_average_accuracy)))
            raw.append((seed, float(metric.raw_average_accuracy)))
        top = finalists[method].ordered_candidates
        methods.append(
            WaterbirdsProductionMethodSummary(
                method_id=method,
                selector=selector,
                lineage=plan.resolved_config.lineage,
                pair_count=(
                    _search_config(plan).pair_count
                    if matrix and consumes_pairs(method)
                    else None
                ),
                selected_candidate_id=winners[method].candidate_id,
                finalist_candidate_ids=(
                    top[0].candidate_id,
                    top[1].candidate_id,
                    top[2].candidate_id,
                ),
                configured_final_seeds=plan.seeds.stages.final,
                result_paths_by_seed=tuple(paths),
                worst_group_by_seed=tuple(worst),
                adjusted_average_by_seed=tuple(adjusted),
                raw_average_by_seed=tuple(raw),
                worst_group_summary=make_waterbirds_metric_summary(
                    "worst_group_accuracy", tuple(value for _, value in worst)
                ),
                adjusted_average_summary=make_waterbirds_metric_summary(
                    "adjusted_average_accuracy",
                    tuple(value for _, value in adjusted),
                ),
                raw_average_summary=make_waterbirds_metric_summary(
                    "raw_average_accuracy", tuple(value for _, value in raw)
                ),
            )
        )
    by_method = {item.method_id: item for item in methods}
    paired: tuple[tuple[int, float], ...] | None = None
    paired_summary = None
    if "erm" in by_method and "grit" in by_method:
        erm_values = dict(by_method["erm"].worst_group_by_seed)
        grit_values = dict(by_method["grit"].worst_group_by_seed)
        paired = tuple(
            (seed, float(grit_values[seed]) - float(erm_values[seed]))
            for seed in plan.seeds.stages.final
        )
        paired_summary = make_waterbirds_metric_summary(
            "grit_minus_erm_worst_group_accuracy",
            tuple(value for _, value in paired),
        )
    return WaterbirdsProductionSummary(
        schema_version="grit.waterbirds-production-summary/v2",
        reportable=True,
        plan_digest=plan.canonical_digest(),
        lineage=plan.resolved_config.lineage,
        selector=selector,
        methods=tuple(methods),
        intervention_comparisons=(
            make_waterbirds_intervention_comparisons(tuple(methods)) if matrix else ()
        ),
        paired_worst_group_by_seed=paired,
        paired_worst_group_summary=paired_summary,
    )


def _waterbirds_selection_values(
    decision: WaterbirdsCheckpointSelection,
) -> dict[str, TrackingValue]:
    return selection_values(
        decision.selector,
        epoch=decision.checkpoint.epoch,
        checkpoint_id=decision.checkpoint.checkpoint_id,
        metrics={
            "worst_group_accuracy": float(decision.worst_group_accuracy),
            "adjusted_average_accuracy": float(decision.adjusted_average_accuracy),
        },
    )


def _mirror_waterbirds_summary(
    plan: SearchPlan, summary: WaterbirdsProductionSummary
) -> None:
    """Mirror the canonical ten-seed summary."""

    rows: list[list[TrackingValue]] = []
    for item in summary.methods:
        for metric in (
            item.worst_group_summary,
            item.adjusted_average_summary,
            item.raw_average_summary,
        ):
            rows.append(
                [
                    item.method_id,
                    item.selector,
                    item.selected_candidate_id,
                    metric.metric_name,
                    metric.seed_count,
                    float(metric.mean),
                    float(metric.sample_standard_deviation),
                    float(metric.ci95_lower),
                    float(metric.ci95_upper),
                ]
            )
    paired = summary.paired_worst_group_summary
    if paired is not None:
        rows.append(
            [
                "grit_minus_erm",
                summary.selector,
                None,
                paired.metric_name,
                paired.seed_count,
                float(paired.mean),
                float(paired.sample_standard_deviation),
                float(paired.ci95_lower),
                float(paired.ci95_upper),
            ]
        )
    existing_contrasts = {"grit_minus_erm"}
    for comparison in summary.intervention_comparisons:
        contrast = f"{comparison.minuend}_minus_{comparison.subtrahend}"
        if (
            contrast in existing_contrasts
            and comparison.metric_name == "worst_group_accuracy"
        ):
            continue
        rows.append(
            [
                contrast,
                comparison.selector,
                None,
                comparison.metric_name,
                len(comparison.configured_final_seeds),
                float(comparison.mean),
                float(comparison.sample_standard_deviation),
                float(comparison.ci95_lower),
                float(comparison.ci95_upper),
            ]
        )
    log_summary_table(
        plan,
        track="ordinary",
        table_name="waterbirds_final_summary",
        columns=(
            "method",
            "selector",
            "candidate_id",
            "metric",
            "seed_count",
            "mean",
            "sd",
            "ci95_lower",
            "ci95_upper",
        ),
        rows=rows,
    )


def _dataset_manifest(plan: SearchPlan) -> WaterbirdsDatasetManifest:
    path = _input_path(plan, "dataset_manifest")
    manifest = WaterbirdsDatasetManifest.model_validate_json(
        path.read_text(encoding="utf-8")
    )
    if (
        manifest.canonical_digest()
        != plan.resolved_config.lineage.dataset_manifest_digest
    ):
        raise ValueError("Waterbirds dataset manifest changed after planning")
    return manifest


def _pair_manifest(plan: SearchPlan) -> WaterbirdsOraclePairManifest:
    path = _input_path(plan, "pair_manifest")
    manifest = WaterbirdsOraclePairManifest.model_validate_json(
        path.read_text(encoding="utf-8")
    )
    if manifest.canonical_digest() != plan.resolved_config.lineage.pair_manifest_digest:
        raise ValueError("Waterbirds pair manifest changed after planning")
    return manifest


def _load_cache(
    plan: SearchPlan,
    *,
    tuning_only: bool = False,
) -> WaterbirdsFeatureCache | WaterbirdsTuningFeatureCache:
    loader = (
        load_waterbirds_tuning_feature_cache
        if tuning_only
        else load_waterbirds_feature_cache
    )
    cache = loader(
        _input_path(plan, "feature_manifest").parent,
        expected_dataset_manifest_digest=(
            plan.resolved_config.lineage.dataset_manifest_digest
        ),
        expected_normalization=plan.normalization,
    )
    if (
        cache.manifest.canonical_digest()
        != plan.resolved_config.lineage.feature_cache_manifest_digest
    ):
        raise ValueError("Waterbirds feature manifest changed after planning")
    return cache


def _input_path(
    plan: SearchPlan,
    kind: Literal["dataset_manifest", "feature_manifest", "pair_manifest"],
) -> Path:
    return next(
        Path(item.path)
        for item in plan.resolved_config.input_artifacts
        if item.kind == kind
    )


def _search_config(plan: SearchPlan) -> WaterbirdsProductionSearchConfig:
    config = plan.resolved_config.config
    if not isinstance(config, WaterbirdsProductionSearchConfig):
        raise TypeError("Waterbirds production runner received another dataset")
    return config


def waterbirds_status_from_plan(plan: SearchPlan) -> ProductionSearchStatus:
    config = _search_config(plan)
    by_id = {item.candidate_id: item for item in plan.candidates}

    def coerce_runs(
        runs: tuple[CompletedStageRun, ...],
    ) -> tuple[WaterbirdsCompletedStageRun, ...]:
        if any(not isinstance(run, WaterbirdsCompletedStageRun) for run in runs):
            raise ValueError("Waterbirds status received another dataset result")
        return cast(tuple[WaterbirdsCompletedStageRun, ...], runs)

    def finalists_complete(finalists: WaterbirdsFinalists, root: Path) -> bool:
        count = 0
        for method, expected in finalists.items():
            path = root / "selection" / method / FINALISTS_FILENAME
            if not path.exists():
                continue
            count += 1
            observed = WaterbirdsTuningFinalists.model_validate_json(
                path.read_text(encoding="utf-8")
            )
            if observed != expected:
                raise ValueError(
                    "Waterbirds finalist artifact does not match canonical tuning "
                    f"results: {path}"
                )
        return count == len(plan.methods)

    def confirmation_tasks(
        finalists: WaterbirdsFinalists,
    ) -> tuple[SearchRunTask, ...]:
        return tuple(
            make_search_task(
                plan,
                by_id[item.candidate_id],
                SeedStage.CONFIRMATION,
                seed,
            )
            for method in plan.methods
            for item in finalists[method].ordered_candidates
            for seed in config.seeds.stages.confirmation
        )

    def winner_count(winners: WaterbirdsWinners, root: Path) -> int:
        count = 0
        for method, expected in winners.items():
            path = root / "selection" / method / WINNER_FILENAME
            if not path.exists():
                continue
            count += 1
            observed = FrozenWaterbirdsCandidate.model_validate_json(
                path.read_text(encoding="utf-8")
            )
            if observed != expected:
                raise ValueError(
                    "Waterbirds frozen winner does not match canonical confirmation "
                    f"results: {path}"
                )
        return count

    def final_tasks(winners: WaterbirdsWinners) -> tuple[SearchRunTask, ...]:
        return tuple(
            make_final_search_task(plan, by_id[winner.candidate_id], seed, winner)
            for winner in winners.values()
            for seed in config.seeds.stages.final
        )

    hooks = ProductionStatusHooks(
        coerce_runs=coerce_runs,
        compute_finalists=lambda runs: compute_waterbirds_finalists(plan, runs),
        finalists_complete=finalists_complete,
        confirmation_tasks=confirmation_tasks,
        compute_winners=lambda finalists, runs: compute_waterbirds_winners(
            plan, finalists, runs
        ),
        winner_count=winner_count,
        expected_winner_count=len(plan.methods),
        final_tasks=final_tasks,
    )
    return production_lifecycle_status(plan, hooks)
