"""Concrete local CMNIST ERM/oracle-GRIT vertical-slice orchestration."""

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
    CmnistDatasetConfig,
    CmnistSourceCounts,
    CpuRuntimeConfig,
    DisabledPairsConfig,
    DisabledProjectionConfig,
    ErmAlgorithmConfig,
    FrozenFeatureConfig,
    GritAlgorithmConfig,
    LinearProbeTrainingConfig,
    LinearProjectionConfig,
    OraclePairsConfig,
    OrdinaryExperimentConfig,
    OrdinarySelectionConfig,
    SeedSets,
    algorithm_method_id,
)
from grit.data.cmnist import (
    PRODUCTION_PARTITION_TARGETS,
    CmnistConstruction,
    CmnistOraclePairSet,
    CmnistPartitionTargets,
    MnistPool,
    build_clean_oracle_pairs,
    construct_cmnist,
    pair_source_view,
)
from grit.features.cmnist import (
    CmnistFeatureCache,
    DeterministicFakeEncoder,
    Normalization,
    OfficialOpenAiClipEncoder,
    load_cmnist_feature_cache,
    load_torchvision_mnist_pools,
    prepare_cmnist_feature_cache,
)
from grit.lifecycle import open_final_test, record_final_accuracy
from grit.methods.checkpoints import restore_checkpoint
from grit.methods.projection import FittedLinearProjection, fit_linear_projection
from grit.methods.training import (
    MethodId,
    OrdinaryLinearProbeMethod,
    PersistedLinearCheckpointStore,
    TrainedLinearProbeRun,
    evaluate_accuracy,
    persist_selected_linear_checkpoint,
    train_linear_probe,
)
from grit.paths import REPO_ROOT
from grit.results import (
    ArtifactReference,
    CodeProvenance,
    EnvironmentProvenance,
    OrdinaryRunResult,
    SucceededStatus,
)
from grit.schemas import CmnistSelector, SeedStage, StrictBoundaryModel
from grit.selection.cmnist import (
    FinalistUnion,
    FrozenCandidateSelection,
    TuningFinalistsArtifact,
    ValidationMetricRecord,
    freeze_candidate,
    freeze_final_checkpoint,
    make_finalist_union,
    make_tuning_finalists,
    select_checkpoint,
    select_confirmed_candidate,
)

NonEmptyStr: TypeAlias = Annotated[StrictStr, Field(min_length=1)]
CMNIST_DATASET_MANIFEST_RELATIVE_PATH = Path("dataset-manifest.json")
CMNIST_PAIR_MANIFEST_RELATIVE_PATH = Path("pair-manifest.json")
CMNIST_FEATURE_CACHE_RELATIVE_ROOT = Path("feature-cache")


class CmnistSmokeRunConfig(StrictBoundaryModel):
    """Tiny validated profile that can never be mistaken for a reportable run."""

    schema_version: Literal["grit.cmnist-smoke/v1"]
    non_reportable: Literal[True]
    output_root: NonEmptyStr
    construction_seed: StrictInt
    pair_seed: StrictInt
    fake_encoder_seed: StrictInt
    image_size: PositiveInt
    source_counts: CmnistSourceCounts
    label_flip_prob: Annotated[StrictFloat, Field(ge=0.25, le=0.25)]
    normalization: Normalization
    pair_count: Literal[256]
    projection_rank: Annotated[StrictInt, Field(ge=0, le=24)]
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
    def _validate_smoke_profile(self) -> CmnistSmokeRunConfig:
        counts = self.source_counts
        if counts.train_e01 + counts.train_e02 < self.pair_count:
            raise ValueError("smoke training support must contain all 256 pair sources")
        observed = (
            counts.train_e01,
            counts.train_e02,
            counts.validation,
            counts.test,
        )
        if observed == (25_000, 25_000, 10_000, 10_000):
            raise ValueError("the smoke profile must remain explicitly non-production")
        if len(set(self.learning_rates)) != 3:
            raise ValueError("smoke learning-rate candidates must be unique")
        return self


class MethodSmokeSummary(StrictBoundaryModel):
    method_id: MethodId
    primary_candidate_id: NonEmptyStr
    secondary_candidate_id: NonEmptyStr
    finalist_union_ids: tuple[NonEmptyStr, ...]
    result_paths: tuple[NonEmptyStr, ...]
    final_accuracies: tuple[float, ...]


class CmnistSmokeSummary(StrictBoundaryModel):
    schema_version: Literal["grit.cmnist-smoke-result/v1"]
    non_reportable: Literal[True]
    dataset_manifest_digest: NonEmptyStr
    feature_manifest_digest: NonEmptyStr
    pair_manifest_digest: NonEmptyStr
    methods: tuple[MethodSmokeSummary, MethodSmokeSummary]


class _YamlModule(Protocol):
    def safe_load(self, payload: str) -> object: ...


_yaml = cast(_YamlModule, importlib.import_module("yaml"))


@dataclass(frozen=True, slots=True)
class _Candidate:
    candidate_id: str
    config: OrdinaryExperimentConfig
    projection: FittedLinearProjection | None


def load_cmnist_smoke_config(path: Path) -> CmnistSmokeRunConfig:
    """Parse strict YAML through JSON semantics and the Pydantic trust boundary."""

    authored = _yaml.safe_load(path.read_text(encoding="utf-8"))
    payload = json.dumps(authored, allow_nan=False)
    return CmnistSmokeRunConfig.model_validate_json(payload)


def run_cmnist_smoke(config: CmnistSmokeRunConfig) -> CmnistSmokeSummary:
    """Execute the hermetic construction-to-result ERM/oracle-GRIT smoke slice."""

    validated = CmnistSmokeRunConfig.model_validate(
        config.model_dump(mode="python")
    )
    output_root = Path(validated.output_root)
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError(f"smoke output directory is not empty: {output_root}")
    output_root.mkdir(parents=True, exist_ok=True)
    train_pool, test_pool = _synthetic_mnist_pools(validated)
    targets = _partition_targets(validated.source_counts)
    construction = construct_cmnist(
        train_pool,
        test_pool,
        construction_seed=validated.construction_seed,
        label_flip_prob=validated.label_flip_prob,
        targets=targets,
    )
    pairs = build_clean_oracle_pairs(
        pair_source_view(construction, train_pool),
        pair_seed=validated.pair_seed,
        pair_count=validated.pair_count,
    )
    _write_construction_manifests(output_root, construction, pairs)
    cache_root = output_root / "feature-cache"
    feature_manifest = prepare_cmnist_feature_cache(
        construction,
        pairs,
        DeterministicFakeEncoder(seed=validated.fake_encoder_seed),
        cache_root,
        normalization=validated.normalization,
    )
    cache = load_cmnist_feature_cache(
        cache_root,
        expected_source_manifest_digest=construction.manifest.canonical_digest(),
        expected_pair_manifest_digest=pairs.manifest.canonical_digest(),
        expected_normalization=validated.normalization,
    )
    summaries = (
        _run_method(validated, construction, pairs, cache, "erm", output_root),
        _run_method(validated, construction, pairs, cache, "grit", output_root),
    )
    summary = CmnistSmokeSummary(
        schema_version="grit.cmnist-smoke-result/v1",
        non_reportable=True,
        dataset_manifest_digest=construction.manifest.canonical_digest(),
        feature_manifest_digest=feature_manifest.canonical_digest(),
        pair_manifest_digest=pairs.manifest.canonical_digest(),
        methods=summaries,
    )
    (output_root / "smoke-result.json").write_text(
        summary.canonical_json() + "\n", encoding="utf-8"
    )
    return summary


def prepare_official_cmnist(
    *,
    data_root: Path,
    clip_weights_root: Path,
    output_root: Path,
    construction_seed: int,
    pair_seed: int,
    normalization: Normalization,
    allow_download: bool,
    feature_device: str,
    clip_batch_size: int,
    pair_count: int = 256,
    held_out_flip_prob: float = 0.5,
) -> None:
    """Explicit real-data/official-CLIP preparation command implementation."""

    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError(
            f"preparation output directory is not empty: {output_root}"
        )
    encoder = OfficialOpenAiClipEncoder(
        weights_root=clip_weights_root,
        allow_download=allow_download,
        device=feature_device,
        batch_size=clip_batch_size,
    )
    encoder.preflight()
    train_pool, test_pool = load_torchvision_mnist_pools(
        data_root,
        allow_download=allow_download,
    )
    construction = construct_cmnist(
        train_pool,
        test_pool,
        construction_seed=construction_seed,
        targets=PRODUCTION_PARTITION_TARGETS,
        held_out_flip_prob=held_out_flip_prob,
    )
    pairs = build_clean_oracle_pairs(
        pair_source_view(construction, train_pool),
        pair_seed=pair_seed,
        pair_count=pair_count,
    )
    output_root.mkdir(parents=True, exist_ok=True)
    _write_construction_manifests(output_root, construction, pairs)
    _ = prepare_cmnist_feature_cache(
        construction,
        pairs,
        encoder,
        output_root / CMNIST_FEATURE_CACHE_RELATIVE_ROOT,
        normalization=normalization,
    )


def _run_method(
    smoke: CmnistSmokeRunConfig,
    construction: CmnistConstruction,
    pairs: CmnistOraclePairSet,
    cache: CmnistFeatureCache,
    method_id: MethodId,
    output_root: Path,
) -> MethodSmokeSummary:
    candidates = _method_candidates(smoke, pairs, cache, method_id)
    tuning_metrics = _run_stage(
        candidates,
        cache,
        smoke.seed_sets.tuning,
        SeedStage.TUNING,
    )
    primary = make_tuning_finalists(
        tuning_metrics, CmnistSelector.PRIMARY_ROBUST, smoke.seed_sets
    )
    secondary = make_tuning_finalists(
        tuning_metrics, CmnistSelector.SECONDARY_SOURCE, smoke.seed_sets
    )
    union = make_finalist_union(primary, secondary)
    method_root = output_root / "runs" / method_id
    method_root.mkdir(parents=True, exist_ok=True)
    _write_selection_artifact(method_root / "primary-finalists.json", primary)
    _write_selection_artifact(method_root / "secondary-finalists.json", secondary)
    _write_selection_artifact(method_root / "finalist-union.json", union)

    by_id = {candidate.candidate_id: candidate for candidate in candidates}
    confirmation_candidates = tuple(
        by_id[candidate_id] for candidate_id in union.confirmation_candidate_ids
    )
    confirmation_metrics = _run_stage(
        confirmation_candidates,
        cache,
        smoke.seed_sets.confirmation,
        SeedStage.CONFIRMATION,
    )
    primary_ids = {item.candidate_id for item in primary.ordered_candidates}
    secondary_ids = {item.candidate_id for item in secondary.ordered_candidates}
    primary_decision = select_confirmed_candidate(
        tuple(
            metric
            for metric in confirmation_metrics
            if metric.candidate_id in primary_ids
        ),
        primary,
        smoke.seed_sets,
    )
    secondary_decision = select_confirmed_candidate(
        tuple(
            metric
            for metric in confirmation_metrics
            if metric.candidate_id in secondary_ids
        ),
        secondary,
        smoke.seed_sets,
    )
    frozen_primary = freeze_candidate(primary_decision, primary, smoke.seed_sets)
    frozen_secondary = freeze_candidate(
        secondary_decision, secondary, smoke.seed_sets
    )
    _write_selection_artifact(method_root / "primary-winner.json", frozen_primary)
    _write_selection_artifact(method_root / "secondary-winner.json", frozen_secondary)

    result_paths: list[str] = []
    final_accuracies: list[float] = []
    for selector, frozen in (
        (CmnistSelector.PRIMARY_ROBUST, frozen_primary),
        (CmnistSelector.SECONDARY_SOURCE, frozen_secondary),
    ):
        candidate = by_id[frozen.candidate_id]
        resolved = _config_for_selector(candidate.config, selector)
        for seed in smoke.seed_sets.final:
            result, relative_path = _run_final_seed(
                smoke,
                construction,
                pairs,
                cache,
                candidate,
                resolved,
                frozen,
                selector,
                seed,
                output_root,
            )
            result_paths.append(relative_path)
            final_metrics = result.final_test_metrics
            if final_metrics is None:
                raise AssertionError("successful final result must contain one metric")
            final_accuracies.append(float(final_metrics[0].value))
    return MethodSmokeSummary(
        method_id=method_id,
        primary_candidate_id=frozen_primary.candidate_id,
        secondary_candidate_id=frozen_secondary.candidate_id,
        finalist_union_ids=union.confirmation_candidate_ids,
        result_paths=tuple(result_paths),
        final_accuracies=tuple(final_accuracies),
    )


def _run_final_seed(
    smoke: CmnistSmokeRunConfig,
    construction: CmnistConstruction,
    pairs: CmnistOraclePairSet,
    cache: CmnistFeatureCache,
    candidate: _Candidate,
    resolved_config: OrdinaryExperimentConfig,
    frozen_candidate: FrozenCandidateSelection,
    selector: CmnistSelector,
    seed: int,
    output_root: Path,
) -> tuple[OrdinaryRunResult, str]:
    run_id = f"run:{candidate.config.algorithm.kind}:{selector.value}:final:{seed}"
    trained = _train_candidate(candidate, cache, run_id, SeedStage.FINAL, seed)
    checkpoint_decision = select_checkpoint(trained.validation_metrics, selector)
    frozen_checkpoint = freeze_final_checkpoint(
        checkpoint_decision,
        frozen_candidate,
    )
    selected = trained.store.load(frozen_checkpoint.checkpoint.checkpoint_id)
    run_root = (
        output_root
        / "runs"
        / candidate.config.algorithm.kind
        / selector.value
        / str(seed)
    )
    checkpoint_root = run_root / "selected-checkpoint"
    checkpoint_manifest = persist_selected_linear_checkpoint(selected, checkpoint_root)
    persisted_store = PersistedLinearCheckpointStore(checkpoint_root)
    restoration = restore_checkpoint(
        frozen_checkpoint,
        persisted_store,
        trained.algorithm.restore_inference_state,
    )
    handle = cache.issue_final_handle(
        run_id=run_id,
        candidate_id=candidate.candidate_id,
        scientific_config_digest=candidate.config.scientific_config_digest(),
    )
    final_view = open_final_test(
        handle,
        frozen_candidate,
        frozen_checkpoint,
        restoration,
    )
    final_table = cache.open_final_table(final_view)
    final_accuracy = evaluate_accuracy(trained.algorithm, final_table)
    final_metric = record_final_accuracy(
        final_view,
        record_id=f"metric:{run_id}:test_ood",
        value=final_accuracy,
        sample_count=len(final_table.source_ids),
    )
    checkpoint_relative = checkpoint_root.relative_to(output_root).as_posix()
    artifacts = [
        ArtifactReference(
            artifact_id="cmnist-dataset-manifest",
            kind="dataset_manifest",
            relative_uri="dataset-manifest.json",
            digest=construction.manifest.canonical_digest(),
        ),
        ArtifactReference(
            artifact_id="cmnist-feature-manifest",
            kind="feature_manifest",
            relative_uri="feature-cache/manifest.json",
            digest=cache.manifest.canonical_digest(),
        ),
        ArtifactReference(
            artifact_id=checkpoint_manifest.store_id,
            kind="selected_linear_checkpoint",
            relative_uri=f"{checkpoint_relative}/manifest.json",
            digest=checkpoint_manifest.canonical_digest(),
        ),
    ]
    if candidate.config.algorithm.kind == "grit":
        artifacts.extend(
            (
                ArtifactReference(
                    artifact_id="cmnist-oracle-pair-manifest",
                    kind="pair_manifest",
                    relative_uri="pair-manifest.json",
                    digest=pairs.manifest.canonical_digest(),
                ),
                ArtifactReference(
                    artifact_id=f"projection:{candidate.candidate_id}",
                    kind="projection_diagnostics",
                    relative_uri=(
                        f"runs/grit/projection-{smoke.projection_rank}.json"
                    ),
                    digest=_required_projection(candidate).diagnostics.canonical_digest(),
                ),
            )
        )
    result = OrdinaryRunResult(
        schema_version="grit.run-result/v1",
        result_kind="ordinary",
        run_id=run_id,
        resolved_config=resolved_config,
        resolved_config_digest=resolved_config.canonical_digest(),
        status=SucceededStatus(kind="succeeded"),
        code=_code_provenance(),
        environment=_environment_provenance(output_root),
        validation_metrics=trained.validation_metrics,
        candidate_selection=frozen_candidate,
        checkpoint_selection=frozen_checkpoint,
        restoration=restoration,
        final_test_metrics=(final_metric,),
        artifacts=tuple(artifacts),
    )
    run_root.mkdir(parents=True, exist_ok=True)
    result_path = run_root / "result.json"
    result_path.write_text(result.canonical_json() + "\n", encoding="utf-8")
    reparsed = OrdinaryRunResult.model_validate_json(
        result_path.read_text(encoding="utf-8")
    )
    if reparsed != result:
        raise AssertionError("canonical ordinary result did not round-trip")
    return result, result_path.relative_to(output_root).as_posix()


def _method_candidates(
    smoke: CmnistSmokeRunConfig,
    pairs: CmnistOraclePairSet,
    cache: CmnistFeatureCache,
    method_id: MethodId,
) -> tuple[_Candidate, ...]:
    projection: FittedLinearProjection | None = None
    if method_id == "grit":
        red, green = cache.pair_tables()
        projection = fit_linear_projection(
            red.features,
            green.features,
            requested_rank=smoke.projection_rank,
            pair_manifest_digest=pairs.manifest.canonical_digest(),
            feature_cache_manifest_digest=cache.manifest.canonical_digest(),
            relative_singular_value_tolerance=(
                smoke.relative_singular_value_tolerance
            ),
        )
        projection_path = Path(smoke.output_root) / "runs" / "grit"
        projection_path.mkdir(parents=True, exist_ok=True)
        (projection_path / f"projection-{smoke.projection_rank}.json").write_text(
            projection.diagnostics.canonical_json() + "\n",
            encoding="utf-8",
        )
    candidates: list[_Candidate] = []
    for learning_rate in smoke.learning_rates:
        config = _resolved_candidate_config(
            smoke,
            cache,
            method_id,
            learning_rate=float(learning_rate),
            selector=CmnistSelector.PRIMARY_ROBUST,
        )
        digest = config.scientific_config_digest()
        candidate_id = f"candidate:{method_id}:{digest.removeprefix('sha256:')[:16]}"
        candidates.append(
            _Candidate(
                candidate_id=candidate_id,
                config=config,
                projection=projection,
            )
        )
    return tuple(candidates)


def _resolved_candidate_config(
    smoke: CmnistSmokeRunConfig,
    cache: CmnistFeatureCache,
    method_id: MethodId,
    *,
    learning_rate: float,
    selector: CmnistSelector,
) -> OrdinaryExperimentConfig:
    counts = smoke.source_counts
    dataset = CmnistDatasetConfig(
        dataset_id="cmnist",
        construction_method_id="cmnist-stratified-hash-v1",
        construction_seed=smoke.construction_seed,
        label_flip_prob=smoke.label_flip_prob,
        source_counts=counts,
        training_split_names=("train_e01", "train_e02"),
        validation_split_names=("val_e01", "val_e02", "val_e05"),
        final_test_split_name="test_ood",
    )
    identity = cache.manifest.encoder
    representation = FrozenFeatureConfig(
        kind="frozen_features",
        encoder_id="synthetic-fake-512",
        encoder_revision=identity.implementation_revision,
        weights_identity=identity.weights_identity,
        preprocessing_identity=identity.preprocessing_identity,
        feature_dimension=512,
        normalization=smoke.normalization,
    )
    if method_id == "erm":
        pairs = DisabledPairsConfig(kind="disabled")
        projection = DisabledProjectionConfig(kind="disabled")
        algorithm = ErmAlgorithmConfig(kind="erm")
    else:
        pairs = OraclePairsConfig(
            kind="oracle",
            construction_id="cmnist-clean-oracle-pairs-v1",
            source_partition_ids=("train_e01_sources", "train_e02_sources"),
            pair_count=256,
            pair_seed=smoke.pair_seed,
            orientation="red_minus_green",
        )
        projection = LinearProjectionConfig(
            kind="linear_pair_difference",
            requested_rank=smoke.projection_rank,
            center_differences=False,
            relative_singular_value_tolerance=(
                smoke.relative_singular_value_tolerance
            ),
        )
        algorithm = GritAlgorithmConfig(kind="grit")
    return OrdinaryExperimentConfig(
        schema_version="grit.experiment/v1",
        run_kind="ordinary",
        experiment_name="non-reportable-cmnist-smoke",
        protocol_id="cmnist/v1",
        reportable=False,
        dataset=dataset,
        representation=representation,
        pairs=pairs,
        projection=projection,
        algorithm=algorithm,
        training=LinearProbeTrainingConfig(
            optimizer="adam",
            batch_size=smoke.batch_size,
            learning_rate=learning_rate,
            weight_decay=float(smoke.weight_decay),
            max_epochs=smoke.max_epochs,
        ),
        runtime=CpuRuntimeConfig(device="cpu", deterministic_algorithms=True),
        seed_sets=smoke.seed_sets,
        selection=OrdinarySelectionConfig(selector=selector),
    )


def _config_for_selector(
    config: OrdinaryExperimentConfig,
    selector: CmnistSelector,
) -> OrdinaryExperimentConfig:
    payload = config.model_dump(mode="python")
    payload["selection"] = {"selector": selector}
    return OrdinaryExperimentConfig.model_validate(payload)


def _run_stage(
    candidates: tuple[_Candidate, ...],
    cache: CmnistFeatureCache,
    seeds: tuple[int, ...],
    stage: Literal[SeedStage.TUNING, SeedStage.CONFIRMATION],
) -> tuple[ValidationMetricRecord, ...]:
    metrics: list[ValidationMetricRecord] = []
    for candidate in candidates:
        for seed in seeds:
            run_id = (
                f"run:{candidate.config.algorithm.kind}:{candidate.candidate_id}:"
                f"{stage.value}:{seed}"
            )
            trained = _train_candidate(candidate, cache, run_id, stage, seed)
            metrics.extend(trained.validation_metrics)
    return tuple(metrics)


def _train_candidate(
    candidate: _Candidate,
    cache: CmnistFeatureCache,
    run_id: str,
    stage: SeedStage,
    seed: int,
) -> TrainedLinearProbeRun:
    method_id = algorithm_method_id(candidate.config.algorithm)
    rank = (
        candidate.config.projection.requested_rank
        if isinstance(candidate.config.projection, LinearProjectionConfig)
        else None
    )
    return train_linear_probe(
        cache.training_tables(),
        cache.validation_tables(),
        candidate.config.training,
        run_id=run_id,
        candidate_id=candidate.candidate_id,
        scientific_config_digest=candidate.config.scientific_config_digest(),
        seed_stage=stage,
        seed=seed,
        method=OrdinaryLinearProbeMethod(
            method_id=method_id,
            projection=candidate.projection,
            projection_rank=rank,
        ),
    )


def _required_projection(candidate: _Candidate) -> FittedLinearProjection:
    if candidate.projection is None:
        raise AssertionError("GRIT candidate is missing its projection")
    return candidate.projection


def _synthetic_mnist_pools(
    smoke: CmnistSmokeRunConfig,
) -> tuple[MnistPool, MnistPool]:
    train_count = (
        smoke.source_counts.train_e01
        + smoke.source_counts.train_e02
        + smoke.source_counts.validation
    )
    test_count = smoke.source_counts.test
    return (
        _synthetic_pool("train", train_count, smoke.image_size),
        _synthetic_pool("test", test_count, smoke.image_size),
    )


def _synthetic_pool(
    split: Literal["train", "test"],
    count: int,
    image_size: int,
) -> MnistPool:
    indices = torch.arange(count, dtype=torch.int64)
    digits = indices.remainder(10)
    images = torch.zeros((count, image_size, image_size), dtype=torch.float32)
    for row in range(count):
        digit = int(digits[row].item())
        images[row, digit % image_size, :] = (digit + 1) / 10
        images[row, :, (row // 10) % image_size] += 0.25
    return MnistPool(
        official_split=split,
        source_indices=indices,
        images=images.clamp(0.0, 1.0),
        digits=digits,
    )


def _partition_targets(counts: CmnistSourceCounts) -> CmnistPartitionTargets:
    return CmnistPartitionTargets(
        train_e01=counts.train_e01,
        train_e02=counts.train_e02,
        validation=counts.validation,
        test=counts.test,
    )


def _write_construction_manifests(
    root: Path,
    construction: CmnistConstruction,
    pairs: CmnistOraclePairSet,
) -> None:
    values = (
        ("partition-manifest.json", construction.partitions.manifest.canonical_json()),
        (
            CMNIST_DATASET_MANIFEST_RELATIVE_PATH.as_posix(),
            construction.manifest.canonical_json(),
        ),
        (
            CMNIST_PAIR_MANIFEST_RELATIVE_PATH.as_posix(),
            pairs.manifest.canonical_json(),
        ),
    )
    for name, payload in values:
        (root / name).write_text(payload + "\n", encoding="utf-8")


def _write_selection_artifact(
    path: Path,
    artifact: TuningFinalistsArtifact | FinalistUnion | FrozenCandidateSelection,
) -> None:
    path.write_text(artifact.canonical_json() + "\n", encoding="utf-8")


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


def _environment_provenance(output_root: Path) -> EnvironmentProvenance:
    del output_root
    root = REPO_ROOT
    lock_sha256 = hashlib.sha256((root / "uv.lock").read_bytes()).hexdigest()
    lock_digest = f"sha256:{lock_sha256}"
    return EnvironmentProvenance(
        python_version=platform.python_version(),
        lock_digest=lock_digest,
        device=f"cpu;torch={torch.__version__}",
    )
