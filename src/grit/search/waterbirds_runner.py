"""Server preparation and non-reportable Waterbirds ERM/GRIT smoke runner."""

from __future__ import annotations

import hashlib
import importlib
import json
import platform
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Literal, Protocol, TypeAlias, cast

import torch
from pydantic import (
    Field,
    PositiveInt,
    StrictFloat,
    StrictInt,
    StrictStr,
    model_validator,
)

from grit.config import (
    ErmAlgorithmConfig,
    GritAlgorithmConfig,
    LinearProbeTrainingConfig,
    SeedSets,
)
from grit.data.waterbirds import (
    BASE_ARTIFACT_NAME,
    ProductionWaterbirdsProfile,
    WaterbirdsConstruction,
    construct_waterbirds_cf,
    load_waterbirds_assets,
    mint_waterbirds_adjusted_weight_spec,
    waterbirds_oracle_relation_view,
)
from grit.data.waterbirds_pairs import (
    WaterbirdsOraclePairSet,
    build_waterbirds_oracle_pairs,
)
from grit.data.waterbirds_smoke_assets import make_waterbirds_smoke_assets
from grit.features.cmnist import OfficialOpenAiClipEncoder
from grit.features.waterbirds import (
    DeterministicFakeWaterbirdsEncoder,
    Normalization,
    WaterbirdsFeatureCache,
    fit_waterbirds_oracle_projection,
    load_waterbirds_feature_cache,
    prepare_waterbirds_feature_cache,
)
from grit.methods.projection import FittedLinearProjection
from grit.methods.training import (
    OrdinaryLinearProbeMethod,
    PersistedLinearCheckpointStore,
    persist_selected_linear_checkpoint,
)
from grit.methods.waterbirds_training import (
    TrainedWaterbirdsRun,
    WaterbirdsMethod,
    restore_waterbirds_checkpoint,
    train_waterbirds_linear_probe,
)
from grit.paths import REPO_ROOT
from grit.results import CodeProvenance, EnvironmentProvenance
from grit.schemas import SeedStage, StrictBoundaryModel
from grit.search.waterbirds_contracts import (
    WaterbirdsArtifactReference,
    WaterbirdsCandidateConfig,
    WaterbirdsCheckpointArtifactReference,
    WaterbirdsDatasetArtifactReference,
    WaterbirdsFeatureArtifactReference,
    WaterbirdsFinalSeedObservation,
    WaterbirdsMethodSmokeSummary,
    WaterbirdsPairArtifactReference,
    WaterbirdsPairedSeedDifference,
    WaterbirdsProjectionArtifactReference,
    WaterbirdsRunResult,
    WaterbirdsSmokeSummary,
    make_waterbirds_metric_summary,
)
from grit.selection.waterbirds import (
    FrozenWaterbirdsCandidate,
    WaterbirdsValidationMetricRecord,
    compute_waterbirds_final_metric,
    freeze_waterbirds_candidate,
    freeze_waterbirds_final_checkpoint,
    make_waterbirds_tuning_finalists,
    select_confirmed_waterbirds_candidate,
    select_waterbirds_checkpoint,
)
from grit.tracking import EventSink, LifecycleEvent, NullEventSink

NonEmptyStr: TypeAlias = Annotated[StrictStr, Field(min_length=1)]
WATERBIRDS_CONSTRUCTION_RELATIVE_ROOT = Path("construction")
WATERBIRDS_PAIR_MANIFEST_RELATIVE_PATH = Path("pair-manifest.json")
WATERBIRDS_FEATURE_CACHE_RELATIVE_ROOT = Path("feature-cache")


class WaterbirdsSmokeRunConfig(StrictBoundaryModel):
    schema_version: Literal["grit.waterbirds-smoke/v1"]
    non_reportable: Literal[True]
    output_root: NonEmptyStr
    construction_seed: StrictInt
    fake_encoder_seed: StrictInt
    normalization: Literal["none"]
    projection_rank: Annotated[StrictInt, Field(ge=0, le=3)]
    relative_singular_value_tolerance: Annotated[StrictFloat, Field(gt=0.0)]
    learning_rates: Annotated[
        tuple[Annotated[StrictFloat, Field(gt=0.0)], ...],
        Field(min_length=3, max_length=3),
    ]
    weight_decay: Annotated[StrictFloat, Field(ge=0.0)]
    batch_size: PositiveInt
    max_epochs: PositiveInt
    seed_sets: SeedSets

    @model_validator(mode="after")
    def _validate_candidates(self) -> WaterbirdsSmokeRunConfig:
        if len(set(self.learning_rates)) != 3:
            raise ValueError("Waterbirds smoke learning rates must be unique")
        return self


class _YamlModule(Protocol):
    def safe_load(self, payload: str) -> object: ...


_yaml = cast(_YamlModule, importlib.import_module("yaml"))


@dataclass(frozen=True, slots=True)
class _Candidate:
    candidate_id: str
    config: WaterbirdsCandidateConfig
    projection: FittedLinearProjection | None


def load_waterbirds_smoke_config(path: Path) -> WaterbirdsSmokeRunConfig:
    authored = _yaml.safe_load(path.read_text(encoding="utf-8"))
    return WaterbirdsSmokeRunConfig.model_validate_json(
        json.dumps(authored, allow_nan=False)
    )


def prepare_server_waterbirds(
    *,
    released_root: Path,
    cub_root: Path,
    masks_root: Path,
    places_root: Path,
    clip_weights_root: Path,
    output_root: Path,
    construction_seed: int,
    normalization: Normalization,
    allow_clip_download: bool,
    feature_device: str,
    clip_batch_size: int,
) -> None:
    """Prepare production assets from supplied local data; never acquire datasets."""

    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError(f"Waterbirds output is not empty: {output_root}")
    encoder = OfficialOpenAiClipEncoder(
        weights_root=clip_weights_root,
        allow_download=allow_clip_download,
        device=feature_device,
        batch_size=clip_batch_size,
    )
    encoder.preflight()
    assets = load_waterbirds_assets(
        released_root=released_root,
        cub_root=cub_root,
        masks_root=masks_root,
        places_root=places_root,
        artifact_name=BASE_ARTIFACT_NAME,
    )
    construction = construct_waterbirds_cf(
        assets,
        ProductionWaterbirdsProfile(
            kind="production",
            base_artifact_name=BASE_ARTIFACT_NAME,
            construction_seed=construction_seed,
        ),
        output_root / WATERBIRDS_CONSTRUCTION_RELATIVE_ROOT,
    )
    pairs = build_waterbirds_oracle_pairs(waterbirds_oracle_relation_view(construction))
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / WATERBIRDS_PAIR_MANIFEST_RELATIVE_PATH).write_text(
        pairs.manifest.canonical_json() + "\n", encoding="utf-8"
    )
    _ = prepare_waterbirds_feature_cache(
        construction,
        encoder,
        output_root / WATERBIRDS_FEATURE_CACHE_RELATIVE_ROOT,
        normalization=normalization,
        encode_batch_size=clip_batch_size,
    )


def run_waterbirds_smoke(
    config: WaterbirdsSmokeRunConfig,
    *,
    event_sink: EventSink | None = None,
) -> WaterbirdsSmokeSummary:
    """Run construction through restored final evaluation on offline fixture assets."""

    smoke = WaterbirdsSmokeRunConfig.model_validate_json(config.canonical_json())
    output_root = Path(smoke.output_root)
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError(f"Waterbirds smoke output is not empty: {output_root}")
    output_root.mkdir(parents=True, exist_ok=True)
    sink = event_sink if event_sink is not None else NullEventSink()
    sink.emit(
        LifecycleEvent(
            event_id="waterbirds-smoke:start",
            run_id="waterbirds-smoke",
            name="start",
        )
    )
    fixture = make_waterbirds_smoke_assets(
        output_root / "source-assets",
        construction_seed=smoke.construction_seed,
    )
    construction = construct_waterbirds_cf(
        fixture.assets,
        fixture.profile,
        output_root / "construction",
    )
    pairs = build_waterbirds_oracle_pairs(waterbirds_oracle_relation_view(construction))
    (output_root / "pair-manifest.json").write_text(
        pairs.manifest.canonical_json() + "\n", encoding="utf-8"
    )
    feature_root = output_root / "feature-cache"
    feature_manifest = prepare_waterbirds_feature_cache(
        construction,
        DeterministicFakeWaterbirdsEncoder(seed=smoke.fake_encoder_seed),
        feature_root,
        normalization=smoke.normalization,
        encode_batch_size=4,
    )
    cache = load_waterbirds_feature_cache(
        feature_root,
        expected_dataset_manifest_digest=construction.manifest.canonical_digest(),
        expected_normalization=smoke.normalization,
    )
    methods = (
        _run_method(smoke, construction, pairs, cache, "erm", output_root),
        _run_method(smoke, construction, pairs, cache, "grit", output_root),
    )
    erm_by_seed = {
        item.seed: float(item.worst_group_accuracy)
        for item in methods[0].final_observations
    }
    grit_by_seed = {
        item.seed: float(item.worst_group_accuracy)
        for item in methods[1].final_observations
    }
    paired_worst = tuple(
        WaterbirdsPairedSeedDifference(
            seed=seed,
            grit_minus_erm_worst_group_accuracy=(
                grit_by_seed[seed] - erm_by_seed[seed]
            ),
        )
        for seed in smoke.seed_sets.final
    )
    summary = WaterbirdsSmokeSummary(
        schema_version="grit.waterbirds-smoke-result/v2",
        non_reportable=True,
        dataset_manifest_digest=construction.manifest.canonical_digest(),
        pair_manifest_digest=pairs.manifest.canonical_digest(),
        feature_cache_manifest_digest=feature_manifest.canonical_digest(),
        normalization=feature_manifest.normalization,
        adjusted_weight_spec_digest=mint_waterbirds_adjusted_weight_spec(
            construction.manifest
        ).canonical_digest(),
        methods=methods,
        paired_worst_group_differences=paired_worst,
        paired_worst_group_summary=make_waterbirds_metric_summary(
            "grit_minus_erm_worst_group_accuracy",
            tuple(
                float(item.grit_minus_erm_worst_group_accuracy) for item in paired_worst
            ),
        ),
    )
    (output_root / "smoke-result.json").write_text(
        summary.canonical_json() + "\n", encoding="utf-8"
    )
    sink.emit(
        LifecycleEvent(
            event_id="waterbirds-smoke:complete",
            run_id="waterbirds-smoke",
            name="complete",
        )
    )
    return summary


def _run_method(
    smoke: WaterbirdsSmokeRunConfig,
    construction: WaterbirdsConstruction,
    pairs: WaterbirdsOraclePairSet,
    cache: WaterbirdsFeatureCache,
    method: WaterbirdsMethod,
    output_root: Path,
) -> WaterbirdsMethodSmokeSummary:
    candidates = _candidates(smoke, construction, pairs, cache, method, output_root)
    tuning = _run_stage(
        candidates,
        cache,
        construction,
        smoke.seed_sets.tuning,
        SeedStage.TUNING,
    )
    finalists = make_waterbirds_tuning_finalists(tuning, smoke.seed_sets)
    method_root = output_root / "runs" / method
    method_root.mkdir(parents=True, exist_ok=True)
    (method_root / "tuning-finalists.json").write_text(
        finalists.canonical_json() + "\n", encoding="utf-8"
    )
    by_id = {candidate.candidate_id: candidate for candidate in candidates}
    confirmation_candidates = tuple(
        by_id[item.candidate_id] for item in finalists.ordered_candidates
    )
    confirmation = _run_stage(
        confirmation_candidates,
        cache,
        construction,
        smoke.seed_sets.confirmation,
        SeedStage.CONFIRMATION,
    )
    winner = select_confirmed_waterbirds_candidate(
        confirmation, finalists, smoke.seed_sets
    )
    frozen = freeze_waterbirds_candidate(winner, finalists, smoke.seed_sets)
    (method_root / "winner.json").write_text(
        frozen.canonical_json() + "\n", encoding="utf-8"
    )
    selected = by_id[frozen.candidate_id]
    observations: list[WaterbirdsFinalSeedObservation] = []
    for seed in smoke.seed_sets.final:
        result, relative = _run_final_seed(
            construction,
            pairs,
            cache,
            selected,
            frozen,
            seed,
            output_root,
        )
        observations.append(
            WaterbirdsFinalSeedObservation(
                seed=seed,
                method_id=method,
                result_path=relative,
                metric_record_id=result.final_test_metric.record_id,
                worst_group_accuracy=result.final_test_metric.worst_group_accuracy,
                adjusted_average_accuracy=(
                    result.final_test_metric.adjusted_average_accuracy
                ),
                raw_average_accuracy=result.final_test_metric.raw_average_accuracy,
            )
        )
    typed_observations = tuple(observations)
    worst_values = tuple(float(item.worst_group_accuracy) for item in observations)
    adjusted_values = tuple(
        float(item.adjusted_average_accuracy) for item in observations
    )
    raw_values = tuple(float(item.raw_average_accuracy) for item in observations)
    return WaterbirdsMethodSmokeSummary(
        method_id=method,
        dataset_manifest_digest=construction.manifest.canonical_digest(),
        feature_cache_manifest_digest=cache.manifest.canonical_digest(),
        normalization=cache.manifest.normalization,
        adjusted_weight_spec_digest=mint_waterbirds_adjusted_weight_spec(
            construction.manifest
        ).canonical_digest(),
        selected_candidate_id=frozen.candidate_id,
        finalist_candidate_ids=(
            finalists.ordered_candidates[0].candidate_id,
            finalists.ordered_candidates[1].candidate_id,
            finalists.ordered_candidates[2].candidate_id,
        ),
        configured_final_seeds=smoke.seed_sets.final,
        final_observations=typed_observations,
        worst_group_summary=make_waterbirds_metric_summary(
            "worst_group_accuracy", worst_values
        ),
        adjusted_average_summary=make_waterbirds_metric_summary(
            "adjusted_average_accuracy", adjusted_values
        ),
        raw_average_summary=make_waterbirds_metric_summary(
            "raw_average_accuracy", raw_values
        ),
    )


def _candidates(
    smoke: WaterbirdsSmokeRunConfig,
    construction: WaterbirdsConstruction,
    pairs: WaterbirdsOraclePairSet,
    cache: WaterbirdsFeatureCache,
    method: WaterbirdsMethod,
    output_root: Path,
) -> tuple[_Candidate, ...]:
    projection: FittedLinearProjection | None = None
    if method == "grit":
        projection = fit_waterbirds_oracle_projection(
            cache,
            pairs,
            requested_rank=smoke.projection_rank,
            relative_singular_value_tolerance=(smoke.relative_singular_value_tolerance),
        )
        projection_root = output_root / "runs" / "grit"
        projection_root.mkdir(parents=True, exist_ok=True)
        (projection_root / "projection.json").write_text(
            projection.diagnostics.canonical_json() + "\n", encoding="utf-8"
        )
    values: list[_Candidate] = []
    for learning_rate in smoke.learning_rates:
        config = WaterbirdsCandidateConfig(
            schema_version="grit.waterbirds-candidate/v3",
            protocol_id="waterbirds_cf/v1",
            non_reportable=True,
            method_id=method,
            selector="waterbirds_validation_worst_group",
            dataset_profile="fixture",
            dataset_manifest_digest=construction.manifest.canonical_digest(),
            feature_cache_manifest_digest=cache.manifest.canonical_digest(),
            normalization=smoke.normalization,
            adjusted_weight_spec_digest=mint_waterbirds_adjusted_weight_spec(
                construction.manifest
            ).canonical_digest(),
            pair_manifest_digest=(
                pairs.manifest.canonical_digest() if method == "grit" else None
            ),
            projection_diagnostics_digest=(
                projection.diagnostics.canonical_digest()
                if projection is not None
                else None
            ),
            projection_rank=smoke.projection_rank if method == "grit" else None,
            relative_singular_value_tolerance=(
                smoke.relative_singular_value_tolerance if method == "grit" else None
            ),
            algorithm=(
                GritAlgorithmConfig(kind="grit")
                if method == "grit"
                else ErmAlgorithmConfig(kind="erm")
            ),
            training=LinearProbeTrainingConfig(
                optimizer="adam",
                batch_size=smoke.batch_size,
                learning_rate=float(learning_rate),
                weight_decay=float(smoke.weight_decay),
                max_epochs=smoke.max_epochs,
            ),
            seed_sets=smoke.seed_sets,
        )
        digest = config.scientific_config_digest().removeprefix("sha256:")
        values.append(
            _Candidate(
                candidate_id=f"candidate:{method}:{digest[:16]}",
                config=config,
                projection=projection,
            )
        )
    return tuple(values)


def _run_stage(
    candidates: tuple[_Candidate, ...],
    cache: WaterbirdsFeatureCache,
    construction: WaterbirdsConstruction,
    seeds: tuple[int, ...],
    stage: Literal[SeedStage.TUNING, SeedStage.CONFIRMATION],
) -> tuple[WaterbirdsValidationMetricRecord, ...]:
    records: list[WaterbirdsValidationMetricRecord] = []
    for candidate in candidates:
        for seed in seeds:
            run_id = (
                f"run:{candidate.config.method_id}:{candidate.candidate_id}:"
                f"{stage.value}:{seed}"
            )
            trained = _train(candidate, cache, construction, run_id, stage, seed)
            records.extend(trained.validation_metrics)
    return tuple(records)


def _train(
    candidate: _Candidate,
    cache: WaterbirdsFeatureCache,
    construction: WaterbirdsConstruction,
    run_id: str,
    stage: SeedStage,
    seed: int,
) -> TrainedWaterbirdsRun:
    return train_waterbirds_linear_probe(
        cache.training_table(),
        cache.validation_table(),
        mint_waterbirds_adjusted_weight_spec(construction.manifest),
        candidate.config.training,
        run_id=run_id,
        candidate_id=candidate.candidate_id,
        scientific_config_digest=candidate.config.scientific_config_digest(),
        seed_stage=stage,
        seed=seed,
        method=OrdinaryLinearProbeMethod(
            method_id=candidate.config.method_id,
            projection=candidate.projection,
            projection_rank=candidate.config.projection_rank,
        ),
    )


def _run_final_seed(
    construction: WaterbirdsConstruction,
    pairs: WaterbirdsOraclePairSet,
    cache: WaterbirdsFeatureCache,
    candidate: _Candidate,
    frozen_candidate: FrozenWaterbirdsCandidate,
    seed: int,
    output_root: Path,
) -> tuple[WaterbirdsRunResult, str]:
    method = candidate.config.method_id
    run_id = f"run:{method}:{candidate.candidate_id}:final:{seed}"
    trained = _train(candidate, cache, construction, run_id, SeedStage.FINAL, seed)
    decision = select_waterbirds_checkpoint(trained.validation_metrics)
    frozen_checkpoint = freeze_waterbirds_final_checkpoint(decision, frozen_candidate)
    selected = trained.store.load(frozen_checkpoint.checkpoint.checkpoint_id)
    run_root = output_root / "runs" / method / "final" / str(seed)
    checkpoint_root = run_root / "selected-checkpoint"
    checkpoint_manifest = persist_selected_linear_checkpoint(selected, checkpoint_root)
    persisted = PersistedLinearCheckpointStore(checkpoint_root)
    restoration = restore_waterbirds_checkpoint(
        frozen_checkpoint, persisted, trained.algorithm
    )
    handle = cache.issue_final_handle(
        run_id=run_id,
        candidate_id=candidate.candidate_id,
        method_id=method,
        scientific_config_digest=candidate.config.scientific_config_digest(),
        seed=seed,
        projection_rank=candidate.config.projection_rank,
    )
    final_view = handle.open(frozen_candidate, frozen_checkpoint, restoration)
    final_table = cache.verify_final_view(final_view)
    predictions = trained.algorithm.predict(final_table.features)
    final_metric = compute_waterbirds_final_metric(
        final_view,
        predictions,
        adjusted_weights=mint_waterbirds_adjusted_weight_spec(construction.manifest),
        record_id=f"metric:{run_id}:test",
    )
    checkpoint_relative = checkpoint_root.relative_to(output_root).as_posix()
    artifacts: list[WaterbirdsArtifactReference] = [
        WaterbirdsDatasetArtifactReference(
            artifact_id="waterbirds-cf-dataset",
            kind="dataset_manifest",
            relative_uri="construction/dataset-manifest.json",
            digest=construction.manifest.canonical_digest(),
        ),
        WaterbirdsFeatureArtifactReference(
            artifact_id="waterbirds-feature-cache",
            kind="feature_manifest",
            relative_uri="feature-cache/manifest.json",
            digest=cache.manifest.canonical_digest(),
            dataset_manifest_digest=construction.manifest.canonical_digest(),
            normalization=cache.manifest.normalization,
        ),
        WaterbirdsCheckpointArtifactReference(
            artifact_id=checkpoint_manifest.store_id,
            kind="selected_linear_checkpoint",
            relative_uri=f"{checkpoint_relative}/manifest.json",
            digest=checkpoint_manifest.canonical_digest(),
            checkpoint=frozen_checkpoint.checkpoint,
        ),
    ]
    if method == "grit":
        artifacts.extend(
            (
                WaterbirdsPairArtifactReference(
                    artifact_id="waterbirds-oracle-pairs",
                    kind="pair_manifest",
                    relative_uri="pair-manifest.json",
                    digest=pairs.manifest.canonical_digest(),
                    dataset_manifest_digest=construction.manifest.canonical_digest(),
                ),
                WaterbirdsProjectionArtifactReference(
                    artifact_id="waterbirds-linear-projection",
                    kind="projection_diagnostics",
                    relative_uri="runs/grit/projection.json",
                    digest=_projection(candidate).diagnostics.canonical_digest(),
                    pair_manifest_digest=pairs.manifest.canonical_digest(),
                    feature_cache_manifest_digest=cache.manifest.canonical_digest(),
                    normalization=cache.manifest.normalization,
                    requested_rank=_projection(candidate).diagnostics.requested_rank,
                ),
            )
        )
    result = WaterbirdsRunResult(
        schema_version="grit.waterbirds-run-result/v3",
        result_kind="ordinary_waterbirds",
        status="succeeded",
        run_id=run_id,
        final_seed=seed,
        resolved_config=candidate.config,
        resolved_config_digest=candidate.config.canonical_digest(),
        code=_code_provenance(),
        environment=_environment_provenance(),
        validation_metrics=trained.validation_metrics,
        candidate_selection=frozen_candidate,
        checkpoint_selection=frozen_checkpoint,
        restoration=restoration,
        final_test_metric=final_metric,
        selected_checkpoint_manifest_digest=checkpoint_manifest.canonical_digest(),
        artifacts=tuple(artifacts),
    )
    run_root.mkdir(parents=True, exist_ok=True)
    result_path = run_root / "result.json"
    result_path.write_text(result.canonical_json() + "\n", encoding="utf-8")
    if (
        WaterbirdsRunResult.model_validate_json(result_path.read_text(encoding="utf-8"))
        != result
    ):
        raise AssertionError("Waterbirds result failed its canonical round trip")
    return result, result_path.relative_to(output_root).as_posix()


def _projection(candidate: _Candidate) -> FittedLinearProjection:
    if candidate.projection is None:
        raise AssertionError("Waterbirds GRIT candidate lacks its projection")
    return candidate.projection


def _code_provenance() -> CodeProvenance:
    root = REPO_ROOT
    if not (root / ".git").exists():
        return CodeProvenance(git_revision="no-git-checkout", git_dirty=False)
    revision = subprocess.run(
        ("git", "rev-parse", "HEAD"),
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    dirty = bool(
        subprocess.run(
            ("git", "status", "--short"),
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    )
    return CodeProvenance(git_revision=revision, git_dirty=dirty)


def _environment_provenance() -> EnvironmentProvenance:
    root = REPO_ROOT
    lock_digest = (
        f"sha256:{hashlib.sha256((root / 'uv.lock').read_bytes()).hexdigest()}"
    )
    return EnvironmentProvenance(
        python_version=platform.python_version(),
        lock_digest=lock_digest,
        device=f"cpu;torch={torch.__version__}",
    )
