"""CMNIST-specific search orchestration, selection, and summaries."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import torch

from grit.config import (
    OPENAI_CLIP_PREPROCESSING_ID,
    OPENAI_CLIP_REVISION,
    OPENAI_CLIP_WEIGHTS_IDENTITY,
    CmnistArtifactLineageConfig,
    CmnistDatasetConfig,
    CmnistSourceCounts,
    ComposedAlgorithmConfig,
    CpuRuntimeConfig,
    FishrAlgorithmConfig,
    FrozenFeatureConfig,
    GroupDroAlgorithmConfig,
    IrmAlgorithmConfig,
    LinearProbeTrainingConfig,
    LisaAlgorithmConfig,
    OraclePairsConfig,
    OrdinaryExperimentConfig,
    OrdinarySelectionConfig,
    RdmAlgorithmConfig,
    RexAlgorithmConfig,
    SdAlgorithmConfig,
    algorithm_method_id,
)
from grit.data.cmnist import CmnistOraclePairManifest
from grit.features.cmnist import (
    CmnistFeatureCache,
    CmnistTuningFeatureCache,
    load_cmnist_feature_cache,
    load_cmnist_tuning_feature_cache,
)
from grit.lifecycle import open_final_test, record_final_accuracy
from grit.methods.baselines import (
    LisaLinearProbeMethod,
    SpectralDecouplingLinearProbeMethod,
    bind_risk_matching_method,
)
from grit.methods.checkpoints import restore_checkpoint
from grit.methods.groupdro import CMNIST_GROUP_COUNT, cmnist_group_ids
from grit.methods.interventions import ComposedLinearProbeMethod
from grit.methods.pair_inputs import PairInputs
from grit.methods.projection import FittedLinearProjection, fit_linear_projection
from grit.methods.training import (
    GroupDroLinearProbeMethod,
    IrmLinearProbeMethod,
    LinearProbeTrainingMethod,
    MethodId,
    OrdinaryLinearProbeMethod,
    PersistedLinearCheckpointStore,
    RexLinearProbeMethod,
    TrainedLinearProbeRun,
    evaluate_accuracy,
    persist_selected_linear_checkpoint,
    persist_training_diagnostics,
    train_linear_probe,
)
from grit.methods.types import (
    base_objective,
    consumes_pairs,
    is_composed,
    pair_intervention,
)
from grit.results import (
    ArtifactReference,
    OrdinaryRunResult,
    SucceededStatus,
)
from grit.schemas import ORDINARY_CMNIST_SELECTORS, CmnistSelector, SeedStage
from grit.search.execution import (
    SEQUENTIAL_EXECUTION,
    ExecutionOptions,
    execution_scope,
)
from grit.search.lifecycle import (
    ProductionLifecycleHooks,
    ProductionStatusHooks,
    checkpoint_decision_values,
    production_lifecycle_status,
    run_production_lifecycle,
)
from grit.search.outputs import (
    CmnistFinalSeedObservation,
    CmnistMethodSelectorSummary,
    CmnistPairedSeedDifference,
    CmnistPairedSelectorSummary,
    CmnistProductionSummary,
    make_cmnist_accuracy_summary,
)
from grit.search.pair_inputs import cmnist_pair_inputs
from grit.search.plan import (
    CmnistProductionSearchConfig,
    SearchCandidate,
    SearchPlan,
    cmnist_candidate_components,
    current_code_provenance,
    current_environment_provenance,
    training_optimizer,
)
from grit.search.run import (
    ProductionExecutionLimits,
    ProductionSearchStatus,
    persist_canonical_artifact,
)
from grit.search.scheduler import (
    CmnistCompletedStageRun,
    CompletedStageRun,
    SearchRunTask,
    StageExecutor,
    make_final_search_task,
    make_search_task,
)
from grit.selection.cmnist import (
    FinalistUnion,
    FrozenCandidateSelection,
    SelectorRecord,
    TuningFinalistsArtifact,
    freeze_candidate,
    freeze_final_checkpoint,
    make_finalist_union,
    make_tuning_finalists,
    select_checkpoint,
    select_confirmed_candidate,
)
from grit.tracking import (
    RunTracker,
    TrackingValue,
    final_values,
    log_summary_table,
    task_tracker,
)

CmnistCandidateConfig = OrdinaryExperimentConfig
CmnistFinalists = dict[tuple[MethodId, CmnistSelector], TuningFinalistsArtifact]
CmnistUnions = dict[MethodId, FinalistUnion | None]
CmnistWinners = dict[tuple[MethodId, CmnistSelector], FrozenCandidateSelection]


@dataclass(frozen=True, slots=True)
class _CmnistRuntimeCandidate:
    planned: SearchCandidate
    config: CmnistCandidateConfig
    projection: FittedLinearProjection | None


@dataclass(frozen=True, slots=True)
class CmnistTrainedTask:
    run: TrainedLinearProbeRun

    def records(self, selector: CmnistSelector) -> tuple[SelectorRecord, ...]:
        del selector
        return self.run.validation_metrics


def cmnist_selectors(
    config: CmnistProductionSearchConfig,
) -> tuple[CmnistSelector, ...]:
    return tuple(CmnistSelector(value) for value in config.selectors)


def finalists_filename(selector: CmnistSelector) -> str:
    return {
        CmnistSelector.PRIMARY_ROBUST: "primary-tuning-finalists.json",
        CmnistSelector.SECONDARY_SOURCE: "secondary-tuning-finalists.json",
    }[selector]


def confirmation_candidate_ids(
    finalists: CmnistFinalists,
    unions: CmnistUnions,
    method: MethodId,
    selectors: tuple[CmnistSelector, ...],
) -> tuple[str, ...]:
    union = unions[method]
    if union is not None:
        return union.confirmation_candidate_ids
    return tuple(
        dict.fromkeys(
            decision.candidate_id
            for selector in selectors
            for decision in finalists[(method, selector)].ordered_candidates
        )
    )


def run_cmnist_search(
    plan: SearchPlan,
    limits: ProductionExecutionLimits | None = None,
    *,
    execution: ExecutionOptions = SEQUENTIAL_EXECUTION,
) -> CmnistProductionSummary | ProductionSearchStatus:
    with execution_scope(Path(plan.resolved_config.output_root), execution):
        return run_production_lifecycle(
            plan,
            limits,
            build_cmnist_hooks(plan, limits),
            hooks_factory=build_cmnist_hooks,
            execution=execution,
        )


def build_cmnist_hooks(
    plan: SearchPlan,
    limits: ProductionExecutionLimits | None = None,
) -> ProductionLifecycleHooks[
    CmnistCompletedStageRun,
    tuple[CmnistFinalists, CmnistUnions],
    CmnistWinners,
    CmnistProductionSummary,
]:
    config = _cmnist_search_config(plan)
    selectors = cmnist_selectors(config)
    output_root = Path(plan.resolved_config.output_root)
    # Test features are never loaded before the final stage.
    cache = _load_cmnist_cache(
        plan,
        tuning_only=limits is not None and limits.stop_after == "tuning",
    )
    pair_manifest = (
        _cmnist_pair_manifest(plan)
        if any(consumes_pairs(method) for method in plan.methods)
        else None
    )
    projections: dict[int, FittedLinearProjection] = {}

    def runtime_candidate(candidate: SearchCandidate) -> _CmnistRuntimeCandidate:
        projection: FittedLinearProjection | None = None
        if pair_intervention(candidate.method_id) == "grit":
            rank = candidate.requested_rank
            if rank is None:
                raise AssertionError("planned GRIT candidate lacks a rank")
            if pair_manifest is None:
                raise AssertionError("planned GRIT search lacks its pair manifest")
            projection = projections.get(rank)
            if projection is None:
                red, green = cache.pair_tables()
                # Pairs are stored in seeded hash order, so the first N form the
                # N-pair bank for the same seed.
                projection = fit_linear_projection(
                    red.features[: config.pair_count],
                    green.features[: config.pair_count],
                    requested_rank=rank,
                    pair_manifest_digest=pair_manifest.canonical_digest(),
                    feature_cache_manifest_digest=cache.manifest.canonical_digest(),
                    relative_singular_value_tolerance=(
                        config.relative_singular_value_tolerance
                    ),
                )
                persist_canonical_artifact(
                    output_root / "projections" / f"grit-rank-{rank}.json",
                    projection.diagnostics,
                )
                projections[rank] = projection
        resolved = materialize_cmnist_candidate_config(plan, candidate, selectors[0])
        return _CmnistRuntimeCandidate(candidate, resolved, projection)

    def execute_pre_final(task: SearchRunTask, run_root: Path) -> CompletedStageRun:
        runtime = runtime_candidate(task.candidate)
        with task_tracker(plan, task, algorithm=runtime.config.algorithm) as tracker:
            trained = _train_cmnist_task(cache, runtime, task, tracker)
            persist_training_diagnostics(trained.run.algorithm, run_root)
            decisions = tuple(
                select_checkpoint(trained.records(selector), selector)
                for selector in selectors
            )
            for decision in decisions:
                tracker.record_selection(checkpoint_decision_values(decision))
            return CmnistCompletedStageRun(
                schema_version="grit.cmnist-search-stage-run/v1",
                dataset="cmnist",
                status="complete",
                task=task,
                lineage=plan.resolved_config.lineage,
                code=current_code_provenance(),
                environment=current_environment_provenance(),
                validation_metrics=trained.run.validation_metrics,
                checkpoint_decisions=decisions,
            )

    candidates_by_id = {item.candidate_id: item for item in plan.candidates}

    def coerce_runs(
        runs: tuple[CompletedStageRun, ...],
    ) -> tuple[CmnistCompletedStageRun, ...]:
        if any(not isinstance(run, CmnistCompletedStageRun) for run in runs):
            raise ValueError("CMNIST lifecycle received another dataset result")
        return cast(tuple[CmnistCompletedStageRun, ...], runs)

    def confirmation_tasks(
        state: tuple[CmnistFinalists, CmnistUnions],
    ) -> tuple[SearchRunTask, ...]:
        finalists, unions = state
        return tuple(
            make_search_task(
                plan,
                candidates_by_id[candidate_id],
                SeedStage.CONFIRMATION,
                seed,
            )
            for method in plan.methods
            for candidate_id in confirmation_candidate_ids(
                finalists, unions, method, selectors
            )
            for seed in config.seeds.stages.confirmation
        )

    def make_winners(
        state: tuple[CmnistFinalists, CmnistUnions],
        runs: tuple[CmnistCompletedStageRun, ...],
        root: Path,
    ) -> CmnistWinners:
        finalists, _ = state
        return _freeze_cmnist_winners(plan, finalists, runs, root)

    def final_tasks(winners: CmnistWinners) -> tuple[SearchRunTask, ...]:
        return tuple(
            make_final_search_task(
                plan,
                candidates_by_id[frozen.candidate_id],
                seed,
                frozen,
            )
            for method in plan.methods
            for selector in selectors
            for frozen in (winners[(method, selector)],)
            for seed in config.seeds.stages.final
        )

    def final_executor(winners: CmnistWinners) -> StageExecutor:
        if isinstance(cache, CmnistTuningFeatureCache):
            raise AssertionError("tuning-only CMNIST execution reached final stage")
        full_cache = cache

        def execute_final(task: SearchRunTask, run_root: Path) -> CompletedStageRun:
            selector = CmnistSelector(task.selector)
            initial = runtime_candidate(task.candidate)
            runtime = _CmnistRuntimeCandidate(
                initial.planned,
                materialize_cmnist_candidate_config(plan, task.candidate, selector),
                initial.projection,
            )
            with task_tracker(
                plan, task, algorithm=runtime.config.algorithm
            ) as tracker:
                return _execute_cmnist_final(task, run_root, tracker, selector, runtime)

        def _execute_cmnist_final(
            task: SearchRunTask,
            run_root: Path,
            tracker: RunTracker,
            selector: CmnistSelector,
            runtime: _CmnistRuntimeCandidate,
        ) -> CompletedStageRun:
            frozen = winners[(task.candidate.method_id, selector)]
            trained = _train_cmnist_task(full_cache, runtime, task, tracker)
            persist_training_diagnostics(trained.run.algorithm, run_root)
            decision = select_checkpoint(trained.records(selector), selector)
            tracker.record_selection(checkpoint_decision_values(decision))
            frozen_checkpoint = freeze_final_checkpoint(decision, frozen)
            run = trained.run
            selected = run.store.load(frozen_checkpoint.checkpoint.checkpoint_id)
            checkpoint_root = run_root / "selected-checkpoint"
            checkpoint_manifest = persist_selected_linear_checkpoint(
                selected, checkpoint_root
            )
            persisted = PersistedLinearCheckpointStore(checkpoint_root)
            restoration = restore_checkpoint(
                frozen_checkpoint,
                persisted,
                run.algorithm.restore_inference_state,
            )
            artifacts = _cmnist_result_artifacts(
                plan, task, checkpoint_manifest.canonical_digest(), runtime
            )
            handle = full_cache.issue_final_handle(
                run_id=run.run_id,
                candidate_id=task.candidate.candidate_id,
                scientific_config_digest=task.candidate.scientific_config_digest,
            )
            final_view = open_final_test(handle, frozen, frozen_checkpoint, restoration)
            final_table = full_cache.open_final_table(final_view)
            final_metric = record_final_accuracy(
                final_view,
                record_id=f"metric:{run.run_id}:test_ood",
                value=evaluate_accuracy(run.algorithm, final_table),
                sample_count=len(final_table.source_ids),
            )
            result = OrdinaryRunResult(
                schema_version="grit.run-result/v1",
                result_kind="ordinary",
                run_id=run.run_id,
                resolved_config=runtime.config,
                resolved_config_digest=runtime.config.canonical_digest(),
                status=SucceededStatus(kind="succeeded"),
                code=current_code_provenance(),
                environment=current_environment_provenance(),
                validation_metrics=run.validation_metrics,
                candidate_selection=frozen,
                checkpoint_selection=frozen_checkpoint,
                restoration=restoration,
                final_test_metrics=(final_metric,),
                artifacts=artifacts,
            )
            tracker.record_selection(
                final_values(
                    selector.value,
                    {"test_ood_accuracy": float(final_metric.value)},
                )
            )
            persist_canonical_artifact(run_root / "final-result.json", result)
            completed = CmnistCompletedStageRun(
                schema_version="grit.cmnist-search-stage-run/v1",
                dataset="cmnist",
                status="complete",
                task=task,
                lineage=plan.resolved_config.lineage,
                code=current_code_provenance(),
                environment=current_environment_provenance(),
                validation_metrics=run.validation_metrics,
                checkpoint_decisions=(decision,),
                final_result_relative_path="final-result.json",
                final_result_digest=result.canonical_digest(),
                final_result=result,
            )
            persist_canonical_artifact(run_root / "result.json", completed)
            return completed

        return execute_final

    def make_summary(
        state: tuple[CmnistFinalists, CmnistUnions],
        winners: CmnistWinners,
        runs: tuple[CmnistCompletedStageRun, ...],
    ) -> CmnistProductionSummary:
        finalists, _ = state
        return _cmnist_summary(plan, finalists, winners, runs)

    def persist_summary(summary: CmnistProductionSummary, root: Path) -> None:
        persist_canonical_artifact(root / "summaries" / "cmnist-summary.json", summary)
        for paired in summary.paired_selectors:
            persist_canonical_artifact(
                root
                / "summaries"
                / f"cmnist-{paired.selector.value}-paired-differences.json",
                paired,
            )
        _mirror_cmnist_summary(plan, summary)

    hooks = ProductionLifecycleHooks(
        coerce_runs=coerce_runs,
        execute_pre_final=execute_pre_final,
        make_finalists=lambda runs, root: _cmnist_finalists(plan, runs, root),
        confirmation_tasks=confirmation_tasks,
        make_winners=make_winners,
        final_tasks=final_tasks,
        final_executor=final_executor,
        make_summary=make_summary,
        persist_summary=persist_summary,
        status=lambda: cmnist_status_from_plan(plan),
    )
    return hooks


def _cmnist_finalists(
    plan: SearchPlan,
    runs: tuple[CmnistCompletedStageRun, ...],
    output_root: Path,
) -> tuple[CmnistFinalists, CmnistUnions]:
    artifacts, unions = compute_cmnist_finalists(plan, runs)
    selectors = cmnist_selectors(_cmnist_search_config(plan))
    for method in plan.methods:
        root = output_root / "selection" / method
        for selector in selectors:
            persist_canonical_artifact(
                root / finalists_filename(selector), artifacts[(method, selector)]
            )
        union = unions[method]
        if union is not None:
            persist_canonical_artifact(root / "confirmation-union.json", union)
    return artifacts, unions


def _selector_records(
    runs: tuple[CmnistCompletedStageRun, ...],
    method: MethodId,
    selector: CmnistSelector,
) -> tuple[SelectorRecord, ...]:
    """Both selectors read validation records."""

    del selector
    records: list[SelectorRecord] = []
    for run in runs:
        if run.task.candidate.method_id != method:
            continue
        records.extend(run.validation_metrics)
    return tuple(records)


def compute_cmnist_finalists(
    plan: SearchPlan,
    runs: tuple[CmnistCompletedStageRun, ...],
) -> tuple[CmnistFinalists, CmnistUnions]:
    selectors = cmnist_selectors(_cmnist_search_config(plan))
    artifacts: CmnistFinalists = {}
    unions: CmnistUnions = {}
    for method in plan.methods:
        for selector in selectors:
            artifacts[(method, selector)] = make_tuning_finalists(
                _selector_records(runs, method, selector),
                selector,
                plan.seeds.stages,
            )
        unions[method] = (
            make_finalist_union(
                artifacts[(method, CmnistSelector.PRIMARY_ROBUST)],
                artifacts[(method, CmnistSelector.SECONDARY_SOURCE)],
            )
            if selectors == ORDINARY_CMNIST_SELECTORS
            else None
        )
    return artifacts, unions


def _freeze_cmnist_winners(
    plan: SearchPlan,
    finalists: CmnistFinalists,
    runs: tuple[CmnistCompletedStageRun, ...],
    output_root: Path,
) -> CmnistWinners:
    winners = compute_cmnist_winners(plan, finalists, runs)
    for (method, selector), frozen in winners.items():
        persist_canonical_artifact(
            output_root / "selection" / method / f"{selector.value}-winner.json",
            frozen,
        )
    return winners


def compute_cmnist_winners(
    plan: SearchPlan,
    finalists: CmnistFinalists,
    runs: tuple[CmnistCompletedStageRun, ...],
) -> CmnistWinners:
    winners: CmnistWinners = {}
    for method in plan.methods:
        for selector in cmnist_selectors(_cmnist_search_config(plan)):
            artifact = finalists[(method, selector)]
            finalist_ids = {item.candidate_id for item in artifact.ordered_candidates}
            records = tuple(
                item
                for item in _selector_records(runs, method, selector)
                if item.candidate_id in finalist_ids
            )
            decision = select_confirmed_candidate(records, artifact, plan.seeds.stages)
            frozen = freeze_candidate(decision, artifact, plan.seeds.stages)
            winners[(method, selector)] = frozen
    return winners


def materialize_cmnist_candidate_config(
    plan: SearchPlan,
    candidate: SearchCandidate,
    selector: CmnistSelector,
) -> CmnistCandidateConfig:
    config = _cmnist_search_config(plan)
    lineage = plan.resolved_config.lineage
    training = LinearProbeTrainingConfig(
        optimizer=training_optimizer(candidate.method_id),
        batch_size=config.batch_size,
        learning_rate=float(candidate.learning_rate),
        weight_decay=float(candidate.weight_decay),
        max_epochs=config.max_epochs,
    )
    components = cmnist_candidate_components(
        config, lineage, candidate.method_id, candidate.settings()
    )
    common: dict[str, object] = dict(
        schema_version="grit.experiment/v1",
        experiment_name=config.experiment_name,
        protocol_id="cmnist/v1",
        reportable=True,
        dataset=CmnistDatasetConfig(
            dataset_id="cmnist",
            construction_method_id="cmnist-stratified-hash-v1",
            construction_seed=config.seeds.construction,
            label_flip_prob=0.25,
            source_counts=CmnistSourceCounts(
                train_e01=25_000,
                train_e02=25_000,
                validation=10_000,
                test=10_000,
            ),
            training_split_names=("train_e01", "train_e02"),
            validation_split_names=(
                "val_e01",
                "val_e02",
                lineage.held_out_validation_split,
            ),
            final_test_split_name="test_ood",
        ),
        representation=FrozenFeatureConfig(
            kind="frozen_features",
            encoder_id="openai-clip-vit-b32",
            encoder_revision=OPENAI_CLIP_REVISION,
            weights_identity=OPENAI_CLIP_WEIGHTS_IDENTITY,
            preprocessing_identity=OPENAI_CLIP_PREPROCESSING_ID,
            feature_dimension=512,
            normalization=config.normalization,
        ),
        pairs=components.pairs,
        projection=components.projection,
        algorithm=components.algorithm,
        training=training,
        runtime=CpuRuntimeConfig(device="cpu", deterministic_algorithms=True),
        seed_sets=config.seeds.stages,
        artifact_lineage=CmnistArtifactLineageConfig(
            dataset_manifest_digest=lineage.dataset_manifest_digest,
            feature_cache_manifest_digest=lineage.feature_cache_manifest_digest,
            pair_manifest_digest=components.pair_manifest_digest,
        ),
    )
    resolved = OrdinaryExperimentConfig(
        run_kind="ordinary",
        selection=OrdinarySelectionConfig(selector=selector),
        **common,  # pyright: ignore[reportArgumentType]
    )
    if resolved.scientific_config_digest() != candidate.scientific_config_digest:
        raise ValueError("planned CMNIST candidate digest cannot be materialized")
    return resolved


def _train_cmnist_task(
    cache: CmnistFeatureCache | CmnistTuningFeatureCache,
    runtime: _CmnistRuntimeCandidate,
    task: SearchRunTask,
    tracker: RunTracker | None = None,
) -> CmnistTrainedTask:
    training_tables = cache.training_tables()
    method = _cmnist_method(cache, runtime, task)
    run_id = f"run:{task.task_id}"
    run = train_linear_probe(
        training_tables,
        cache.validation_tables(),
        runtime.config.training,
        run_id=run_id,
        candidate_id=task.candidate.candidate_id,
        scientific_config_digest=task.candidate.scientific_config_digest,
        seed_stage=task.stage,
        seed=task.seed,
        method=method,
        tracker=tracker,
    )
    return CmnistTrainedTask(run)


def _cmnist_method(
    cache: CmnistFeatureCache | CmnistTuningFeatureCache,
    runtime: _CmnistRuntimeCandidate,
    task: SearchRunTask,
) -> LinearProbeTrainingMethod:
    algorithm = runtime.config.algorithm
    if not isinstance(algorithm, ComposedAlgorithmConfig):
        return _cmnist_base_method(cache, runtime, task)
    base_runtime = _CmnistRuntimeCandidate(
        runtime.planned,
        runtime.config.model_copy(update={"algorithm": algorithm.base_objective}),
        None,
    )
    pairs = None
    if algorithm.pair_intervention != "grit":
        pairs = _cmnist_pair_inputs(cache, runtime)
    return ComposedLinearProbeMethod(
        base=_cmnist_base_method(cache, base_runtime, task),
        method_id=task.candidate.method_id,
        projection=runtime.projection,
        projection_rank=task.candidate.requested_rank,
        pair_differences=None if pairs is None else pairs.differences,
        consistency_weight=algorithm.consistency_weight or 0.0,
    )


def _cmnist_pair_inputs(
    cache: CmnistFeatureCache | CmnistTuningFeatureCache,
    runtime: _CmnistRuntimeCandidate,
) -> PairInputs:
    # Pairs are stored in seeded hash order; the first N form the N-pair bank.
    pairs = runtime.config.pairs
    if not isinstance(pairs, OraclePairsConfig):
        raise ValueError("pair methods require training oracle pairs")
    red, green = cache.pair_tables()
    return cmnist_pair_inputs(red, green, pairs.pair_count)


def _cmnist_base_method(
    cache: CmnistFeatureCache | CmnistTuningFeatureCache,
    runtime: _CmnistRuntimeCandidate,
    task: SearchRunTask,
) -> LinearProbeTrainingMethod:
    """Bind one planned candidate's algorithm config to the shared trainer."""

    training_tables = cache.training_tables()
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
        names = tuple(table.name for table in training_tables)
        if names != algorithm.environment_names:
            raise ValueError(
                "invariant training tables do not match configured environments"
            )
        environment_ids = torch.cat(
            [
                torch.full((len(table.source_ids),), index, dtype=torch.int64)
                for index, table in enumerate(training_tables)
            ]
        )
        if isinstance(algorithm, RexAlgorithmConfig):
            return RexLinearProbeMethod(
                environment_ids=environment_ids,
                environment_count=len(training_tables),
                penalty_weight=float(algorithm.penalty_weight),
                penalty_anneal_updates=int(algorithm.penalty_anneal_updates),
            )
        if isinstance(algorithm, IrmAlgorithmConfig):
            return IrmLinearProbeMethod(
                environment_ids=environment_ids,
                environment_count=len(training_tables),
                penalty_weight=float(algorithm.penalty_weight),
                penalty_anneal_updates=int(algorithm.penalty_anneal_updates),
            )
        # Fishr and RDM are all that remain; the shared binder also applies the
        # warm-up guard, which needs this candidate's epoch length.
        return bind_risk_matching_method(
            algorithm,
            environment_ids=environment_ids,
            environment_count=len(training_tables),
            training=runtime.config.training,
        )
    if isinstance(algorithm, GroupDroAlgorithmConfig | LisaAlgorithmConfig):
        targets = torch.cat([table.targets for table in training_tables], dim=0)
        # Source-environment groups never read color.
        environments = torch.cat(
            [
                torch.full((len(table.source_ids),), index, dtype=torch.int64)
                for index, table in enumerate(training_tables)
            ]
        )
        group_count = CMNIST_GROUP_COUNT
        if algorithm.group_definition == "target_color":
            group_ids = cmnist_group_ids(
                targets,
                torch.cat([table.colors for table in training_tables], dim=0),
            )
        elif algorithm.group_definition == "training_environment" and isinstance(
            algorithm, GroupDroAlgorithmConfig
        ):
            group_ids, group_count = environments, len(training_tables)
        elif algorithm.group_definition == "target_environment":
            # Same (target, domain) layout as target-color groups: 2 * y + e.
            group_ids = cmnist_group_ids(targets, environments)
        else:
            raise ValueError(
                f"CMNIST {algorithm.kind} cannot use {algorithm.group_definition}"
            )
        if isinstance(algorithm, GroupDroAlgorithmConfig):
            return GroupDroLinearProbeMethod(
                group_ids=group_ids,
                group_count=group_count,
                step_size=float(algorithm.adversarial_step_size),
            )
        return LisaLinearProbeMethod(
            group_ids=group_ids,
            group_count=group_count,
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


def _cmnist_result_artifacts(
    plan: SearchPlan,
    task: SearchRunTask,
    checkpoint_manifest_digest: str,
    runtime: _CmnistRuntimeCandidate,
) -> tuple[ArtifactReference, ...]:
    root = Path(plan.resolved_config.output_root)
    inputs = {item.kind: item for item in plan.resolved_config.input_artifacts}
    values = [
        ArtifactReference(
            artifact_id="cmnist-dataset-manifest",
            kind="dataset_manifest",
            relative_uri=os.path.relpath(inputs["dataset_manifest"].path, root),
            digest=inputs["dataset_manifest"].digest,
        ),
        ArtifactReference(
            artifact_id="cmnist-feature-manifest",
            kind="feature_manifest",
            relative_uri=os.path.relpath(inputs["feature_manifest"].path, root),
            digest=inputs["feature_manifest"].digest,
        ),
        ArtifactReference(
            artifact_id=f"checkpoint:{task.task_id}",
            kind="selected_linear_checkpoint",
            relative_uri=(
                f"{task.relative_directory}/selected-checkpoint/manifest.json"
            ),
            digest=checkpoint_manifest_digest,
        ),
    ]
    if consumes_pairs(algorithm_method_id(runtime.config.algorithm)):
        values.append(
            ArtifactReference(
                artifact_id="cmnist-oracle-pair-manifest",
                kind="pair_manifest",
                relative_uri=os.path.relpath(inputs["pair_manifest"].path, root),
                digest=inputs["pair_manifest"].digest,
            )
        )
    if pair_intervention(task.candidate.method_id) == "grit":
        projection = runtime.projection
        if projection is None:
            raise AssertionError("GRIT result lacks projection")
        rank = task.candidate.requested_rank
        values.append(
            ArtifactReference(
                artifact_id=f"projection:grit:{rank}",
                kind="projection_diagnostics",
                relative_uri=f"projections/grit-rank-{rank}.json",
                digest=projection.diagnostics.canonical_digest(),
            )
        )
    return tuple(values)


def _cmnist_summary(
    plan: SearchPlan,
    finalists: CmnistFinalists,
    winners: CmnistWinners,
    runs: tuple[CmnistCompletedStageRun, ...],
) -> CmnistProductionSummary:
    from grit.methods.types import consumes_pairs
    from grit.search.outputs import make_cmnist_intervention_comparisons

    matrix = any(method.endswith(("_grit", "_consistency")) for method in plan.methods)

    selectors = cmnist_selectors(_cmnist_search_config(plan))
    by_identity = {
        (
            run.task.candidate.method_id,
            CmnistSelector(run.task.selector),
            run.task.seed,
        ): run
        for run in runs
    }
    methods: list[CmnistMethodSelectorSummary] = []
    for method in plan.methods:
        for selector in selectors:
            observations: list[CmnistFinalSeedObservation] = []
            for seed in plan.seeds.stages.final:
                run = by_identity[(method, selector, seed)]
                record_id, value = _final_test_observation(run)
                observations.append(
                    CmnistFinalSeedObservation(
                        seed=seed,
                        method_id=method,
                        selector=selector,
                        result_path=(
                            f"{run.task.relative_directory}/final-result.json"
                        ),
                        metric_record_id=record_id,
                        test_ood_accuracy=value,
                    )
                )
            values = tuple(float(item.test_ood_accuracy) for item in observations)
            top = finalists[(method, selector)].ordered_candidates
            methods.append(
                CmnistMethodSelectorSummary(
                    method_id=method,
                    selector=selector,
                    lineage=plan.resolved_config.lineage,
                    pair_count=(
                        _cmnist_search_config(plan).pair_count
                        if matrix and consumes_pairs(method)
                        else None
                    ),
                    selected_candidate_id=winners[(method, selector)].candidate_id,
                    finalist_candidate_ids=(
                        top[0].candidate_id,
                        top[1].candidate_id,
                        top[2].candidate_id,
                    ),
                    configured_final_seeds=plan.seeds.stages.final,
                    final_observations=tuple(observations),
                    accuracy_summary=make_cmnist_accuracy_summary(
                        "test_ood_accuracy", values
                    ),
                )
            )
    paired: list[CmnistPairedSelectorSummary] = []
    method_map = {(item.method_id, item.selector): item for item in methods}
    if "erm" in plan.methods and "grit" in plan.methods:
        for selector in selectors:
            erm = method_map[("erm", selector)]
            grit = method_map[("grit", selector)]
            erm_values = {
                item.seed: float(item.test_ood_accuracy)
                for item in erm.final_observations
            }
            grit_values = {
                item.seed: float(item.test_ood_accuracy)
                for item in grit.final_observations
            }
            differences = tuple(
                CmnistPairedSeedDifference(
                    seed=seed,
                    selector=selector,
                    grit_minus_erm_test_ood_accuracy=(
                        grit_values[seed] - erm_values[seed]
                    ),
                )
                for seed in plan.seeds.stages.final
            )
            paired.append(
                CmnistPairedSelectorSummary(
                    selector=selector,
                    configured_final_seeds=plan.seeds.stages.final,
                    paired_differences=differences,
                    difference_summary=make_cmnist_accuracy_summary(
                        "grit_minus_erm_test_ood_accuracy",
                        tuple(
                            float(item.grit_minus_erm_test_ood_accuracy)
                            for item in differences
                        ),
                    ),
                )
            )
    return CmnistProductionSummary(
        schema_version="grit.cmnist-production-summary/v1",
        reportable=True,
        plan_digest=plan.canonical_digest(),
        lineage=plan.resolved_config.lineage,
        selectors=selectors,
        methods=tuple(methods),
        intervention_comparisons=(
            make_cmnist_intervention_comparisons(tuple(methods)) if matrix else ()
        ),
        paired_selectors=tuple(paired),
    )


def _mirror_cmnist_summary(plan: SearchPlan, summary: CmnistProductionSummary) -> None:
    """Mirror the canonical ten-seed summary."""

    rows: list[list[TrackingValue]] = [
        [
            item.method_id,
            item.selector.value,
            item.selected_candidate_id,
            item.accuracy_summary.metric_name,
            item.accuracy_summary.seed_count,
            float(item.accuracy_summary.mean),
            float(item.accuracy_summary.sample_standard_deviation),
            float(item.accuracy_summary.ci95_lower),
            float(item.accuracy_summary.ci95_upper),
        ]
        for item in summary.methods
    ]
    rows.extend(
        [
            "grit_minus_erm",
            paired.selector.value,
            None,
            paired.difference_summary.metric_name,
            paired.difference_summary.seed_count,
            float(paired.difference_summary.mean),
            float(paired.difference_summary.sample_standard_deviation),
            float(paired.difference_summary.ci95_lower),
            float(paired.difference_summary.ci95_upper),
        ]
        for paired in summary.paired_selectors
    )
    existing_contrasts = {"grit_minus_erm"}
    for comparison in summary.intervention_comparisons:
        contrast = f"{comparison.minuend}_minus_{comparison.subtrahend}"
        if (
            contrast in existing_contrasts
            and comparison.metric_name == "test_ood_accuracy"
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
        table_name="cmnist_final_summary",
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


def _final_test_observation(run: CmnistCompletedStageRun) -> tuple[str, float]:
    """Return the reported test_ood record for a final run."""

    result = run.final_result
    if isinstance(result, OrdinaryRunResult):
        if result.final_test_metrics is None:
            raise AssertionError("completed final run lacks CMNIST metrics")
        metric = result.final_test_metrics[0]
        return metric.record_id, float(metric.value)
    raise AssertionError("completed final run lacks its result")


def _load_cmnist_cache(
    plan: SearchPlan,
    *,
    tuning_only: bool = False,
) -> CmnistFeatureCache | CmnistTuningFeatureCache:
    paths = {
        item.kind: Path(item.path) for item in plan.resolved_config.input_artifacts
    }
    loader = (
        load_cmnist_tuning_feature_cache if tuning_only else load_cmnist_feature_cache
    )
    cache = loader(
        paths["feature_manifest"].parent,
        expected_source_manifest_digest=(
            plan.resolved_config.lineage.dataset_manifest_digest
        ),
        expected_pair_manifest_digest=plan.resolved_config.lineage.pair_manifest_digest,
        include_pairs=any(consumes_pairs(method) for method in plan.methods),
        expected_normalization=plan.normalization,
    )
    if (
        cache.manifest.canonical_digest()
        != plan.resolved_config.lineage.feature_cache_manifest_digest
    ):
        raise ValueError("CMNIST feature manifest changed after planning")
    return cache


def _cmnist_pair_manifest(plan: SearchPlan) -> CmnistOraclePairManifest:
    path = next(
        Path(item.path)
        for item in plan.resolved_config.input_artifacts
        if item.kind == "pair_manifest"
    )
    manifest = CmnistOraclePairManifest.model_validate_json(
        path.read_text(encoding="utf-8")
    )
    if manifest.canonical_digest() != plan.resolved_config.lineage.pair_manifest_digest:
        raise ValueError("CMNIST pair manifest changed after planning")
    return manifest


def _cmnist_search_config(plan: SearchPlan) -> CmnistProductionSearchConfig:
    config = plan.resolved_config.config
    if not isinstance(config, CmnistProductionSearchConfig):
        raise TypeError("CMNIST production runner received another dataset")
    return config


def cmnist_status_from_plan(plan: SearchPlan) -> ProductionSearchStatus:
    config = _cmnist_search_config(plan)
    selectors = cmnist_selectors(config)
    by_id = {item.candidate_id: item for item in plan.candidates}

    def coerce_runs(
        runs: tuple[CompletedStageRun, ...],
    ) -> tuple[CmnistCompletedStageRun, ...]:
        if any(not isinstance(run, CmnistCompletedStageRun) for run in runs):
            raise ValueError("CMNIST status received another dataset result")
        return cast(tuple[CmnistCompletedStageRun, ...], runs)

    def finalists_complete(
        state: tuple[CmnistFinalists, CmnistUnions],
        root: Path,
    ) -> bool:
        expected_finalists, expected_unions = state
        count = 0
        expected_count = 0
        for method in plan.methods:
            selection_root = root / "selection" / method
            artifacts: list[
                tuple[Path, type[TuningFinalistsArtifact] | type[FinalistUnion], object]
            ] = [
                (
                    selection_root / finalists_filename(selector),
                    TuningFinalistsArtifact,
                    expected_finalists[(method, selector)],
                )
                for selector in selectors
            ]
            union = expected_unions[method]
            if union is not None:
                artifacts.append(
                    (selection_root / "confirmation-union.json", FinalistUnion, union)
                )
            expected_count += len(artifacts)
            for path, artifact_type, expected in artifacts:
                if not path.exists():
                    continue
                count += 1
                observed = artifact_type.model_validate_json(
                    path.read_text(encoding="utf-8")
                )
                if observed != expected:
                    raise ValueError(
                        "CMNIST selection artifact does not match canonical tuning "
                        f"results: {path}"
                    )
        return count == expected_count

    def confirmation_tasks(
        state: tuple[CmnistFinalists, CmnistUnions],
    ) -> tuple[SearchRunTask, ...]:
        finalists, unions = state
        return tuple(
            make_search_task(
                plan,
                by_id[candidate_id],
                SeedStage.CONFIRMATION,
                seed,
            )
            for method in plan.methods
            for candidate_id in confirmation_candidate_ids(
                finalists, unions, method, selectors
            )
            for seed in config.seeds.stages.confirmation
        )

    def winner_count(winners: CmnistWinners, root: Path) -> int:
        count = 0
        for identity, expected in winners.items():
            method, selector = identity
            path = root / "selection" / method / f"{selector.value}-winner.json"
            if not path.exists():
                continue
            count += 1
            observed = FrozenCandidateSelection.model_validate_json(
                path.read_text(encoding="utf-8")
            )
            if observed != expected:
                raise ValueError(
                    "CMNIST frozen winner does not match canonical confirmation "
                    f"results: {path}"
                )
        return count

    def final_tasks(winners: CmnistWinners) -> tuple[SearchRunTask, ...]:
        return tuple(
            make_final_search_task(
                plan,
                by_id[winner.candidate_id],
                seed,
                winner,
            )
            for winner in winners.values()
            for seed in config.seeds.stages.final
        )

    hooks = ProductionStatusHooks(
        coerce_runs=coerce_runs,
        compute_finalists=lambda runs: compute_cmnist_finalists(plan, runs),
        finalists_complete=finalists_complete,
        confirmation_tasks=confirmation_tasks,
        compute_winners=lambda state, runs: compute_cmnist_winners(
            plan, state[0], runs
        ),
        winner_count=winner_count,
        expected_winner_count=len(plan.methods) * len(config.selectors),
        final_tasks=final_tasks,
    )
    return production_lifecycle_status(plan, hooks)
