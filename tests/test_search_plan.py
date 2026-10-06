"""Canonical production-search configuration and planning tests."""

from __future__ import annotations

import hashlib
import importlib.util
import json
from collections.abc import Callable, Sequence
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Literal, cast

import pytest
import torch
from pydantic import ValidationError

from grit.config import SeedSets
from grit.data.cmnist import (
    CmnistDatasetManifest,
    CmnistOraclePairManifest,
    CmnistPartitionManifest,
    CmnistPartitionTargets,
    DigitCount,
    EnvironmentManifest,
    OraclePairRecord,
    SourcePartitionRecord,
    cmnist_environment_specs,
)
from grit.data.waterbirds import (
    WATERBIRDS_GROUP_ORDER,
    WaterbirdsAdjustedWeightSpec,
    WaterbirdsGroupCounts,
)
from grit.features.cmnist import (
    ArrayFileManifest,
    CmnistFeatureCacheManifest,
    EncoderIdentity,
    FeatureExtractionRuntime,
    FeatureTable,
    FeatureTableManifest,
)
from grit.methods.projection import FittedLinearProjection, fit_linear_projection
from grit.methods.training import TrainedLinearProbeRun
from grit.results import CodeProvenance, EnvironmentProvenance
from grit.schemas import (
    CmnistSelector,
    SeedStage,
    canonical_digest_value,
    held_out_validation_name,
)
from grit.search.cmnist import (
    CmnistTrainedTask,
    compute_cmnist_finalists,
    compute_cmnist_winners,
    materialize_cmnist_candidate_config,
)
from grit.search.cmnist_runner import (
    CMNIST_DATASET_MANIFEST_RELATIVE_PATH,
    CMNIST_FEATURE_CACHE_RELATIVE_ROOT,
    CMNIST_PAIR_MANIFEST_RELATIVE_PATH,
)
from grit.search.execution import ExecutionOptions
from grit.search.plan import (
    APPROVED_GROUPDRO_STEP_SIZES,
    APPROVED_IRM_PENALTY_WEIGHTS,
    APPROVED_LEARNING_RATES,
    APPROVED_RANKS,
    APPROVED_REX_PENALTY_WEIGHTS,
    APPROVED_WEIGHT_DECAYS,
    IRM_PENALTY_ANNEAL_UPDATES,
    REX_PENALTY_ANNEAL_UPDATES,
    CmnistProductionSearchConfig,
    ResolvedProductionSearchConfig,
    SearchArtifactPaths,
    SearchCandidate,
    SearchLineage,
    SearchPlan,
    SearchRuntimeConfig,
    SearchSeedConfig,
    SearchSpaceConfig,
    VerifiedInputArtifact,
    WaterbirdsProductionSearchConfig,
    build_search_plan,
    load_production_search_config,
    resolve_production_search_config,
)
from grit.search.run import (
    ProductionExecutionLimits,
    ProductionSearchStatus,
    limited_tuning_candidates,
    persist_canonical_artifact,
    pilot_candidates,
    plan_production_search,
    production_pilot_candidates,
    production_search_status,
    run_production_search,
    validate_execution_limits,
)
from grit.search.scheduler import (
    CmnistCompletedStageRun,
    CompletedStageRun,
    LocalRunScheduler,
    SearchRunTask,
    SearchStatus,
    WaterbirdsCompletedStageRun,
    make_search_task,
)
from grit.search.waterbirds import (
    compute_waterbirds_finalists,
    compute_waterbirds_winners,
    materialize_waterbirds_candidate_config,
)
from grit.search.waterbirds_runner import (
    WATERBIRDS_CONSTRUCTION_RELATIVE_ROOT,
    WATERBIRDS_FEATURE_CACHE_RELATIVE_ROOT,
    WATERBIRDS_PAIR_MANIFEST_RELATIVE_PATH,
)
from grit.selection.cmnist import (
    ValidationMetricRecord,
    make_tuning_finalists,
    select_checkpoint,
)
from grit.selection.waterbirds import (
    WaterbirdsGroupAccuracy,
    WaterbirdsValidationMetricRecord,
    select_waterbirds_checkpoint,
)
from tests.production_artifact_fixtures import (
    write_manifest_only_waterbirds_production,
)

_CLEAN_CODE_PROVENANCE = CodeProvenance(
    git_revision="1" * 40,
    git_dirty=False,
)


def _yaml_compatible_json(payload: object) -> str:
    return (
        json.dumps(payload)
        .replace("1e-05", "0.00001")
        .replace("1e-12", "0.000000000001")
    )


@pytest.fixture(autouse=True)
def _explicit_clean_search_provenance(  # pyright: ignore[reportUnusedFunction]
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "grit.search.plan._code_provenance", lambda: _CLEAN_CODE_PROVENANCE
    )


def _seeds() -> SearchSeedConfig:
    return SearchSeedConfig(
        construction=17,
        pairs=23,
        stages=SeedSets(
            tuning=(101, 102, 103),
            confirmation=(201, 202),
            final=(301, 302, 303, 304, 305, 306, 307, 308, 309, 310),
        ),
    )


def test_checked_production_examples_match_preparation_layout_and_seeds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from grit.paths import DEFAULT_CONSTRUCTION_SEED, DEFAULT_PAIR_SEED

    repository = Path(__file__).resolve().parents[1]
    cmnist_path = (
        repository
        / "configs/cmnist/held-out-e03/erm-representation-interventions-search.yaml"
    )
    monkeypatch.setenv("PROJECT_SCRATCH", "/scratch")
    cmnist = load_production_search_config(cmnist_path)
    waterbirds = load_production_search_config(
        repository / "configs/waterbirds/erm-representation-interventions-search.yaml"
    )
    assert isinstance(cmnist, CmnistProductionSearchConfig)
    assert isinstance(waterbirds, WaterbirdsProductionSearchConfig)
    # Preparation defaults and the checked configs must agree on seeds and layout.
    assert cmnist.seeds.construction == DEFAULT_CONSTRUCTION_SEED
    assert cmnist.seeds.pairs == DEFAULT_PAIR_SEED
    cmnist_root = "/scratch/artifacts/cmnist-none-e03/"
    assert cmnist.artifacts.dataset_manifest == (
        cmnist_root + CMNIST_DATASET_MANIFEST_RELATIVE_PATH.as_posix()
    )
    assert cmnist.artifacts.feature_cache_manifest == (
        cmnist_root + (CMNIST_FEATURE_CACHE_RELATIVE_ROOT / "manifest.json").as_posix()
    )
    assert cmnist.artifacts.oracle_pair_manifest == (
        cmnist_root + CMNIST_PAIR_MANIFEST_RELATIVE_PATH.as_posix()
    )
    assert cmnist.search_space.projection_ranks == (2, 8, 16, 24)
    assert cmnist.output_root == (
        "/scratch/outputs/cmnist-erm-interventions-representation-v1-e03"
    )

    assert waterbirds.seeds.construction == DEFAULT_CONSTRUCTION_SEED
    waterbirds_root = "/scratch/artifacts/waterbirds-none/"
    assert waterbirds.artifacts.dataset_manifest == (
        waterbirds_root
        + (WATERBIRDS_CONSTRUCTION_RELATIVE_ROOT / "dataset-manifest.json").as_posix()
    )
    assert waterbirds.artifacts.feature_cache_manifest == (
        waterbirds_root
        + (WATERBIRDS_FEATURE_CACHE_RELATIVE_ROOT / "manifest.json").as_posix()
    )
    assert waterbirds.artifacts.oracle_pair_manifest == (
        waterbirds_root + WATERBIRDS_PAIR_MANIFEST_RELATIVE_PATH.as_posix()
    )
    assert waterbirds.search_space.projection_ranks == (2, 8, 16, 24)
    assert waterbirds.output_root == (
        "/scratch/outputs/waterbirds-erm-interventions-representation-v1"
    )

    monkeypatch.delenv("PROJECT_SCRATCH")
    monkeypatch.setenv("GRIT_SCRATCH", "/alt")
    fallback = load_production_search_config(cmnist_path)
    assert fallback.output_root == (
        "/alt/outputs/cmnist-erm-interventions-representation-v1-e03"
    )
    monkeypatch.delenv("GRIT_SCRATCH")
    local = load_production_search_config(cmnist_path)
    expected = (
        repository / "scratch/outputs/cmnist-erm-interventions-representation-v1-e03"
    )
    assert local.output_root == expected.as_posix()


def _space() -> SearchSpaceConfig:
    return SearchSpaceConfig(
        methods=("erm", "grit"),
        learning_rates=APPROVED_LEARNING_RATES,
        weight_decays=APPROVED_WEIGHT_DECAYS,
        projection_ranks=APPROVED_RANKS,
    )


def _config(dataset: str = "cmnist") -> (
    CmnistProductionSearchConfig | WaterbirdsProductionSearchConfig
):
    common: dict[str, object] = {
        "schema_version": "grit.production-search/v1",
        "experiment_name": f"{dataset}-primary",
        "experiment_variant": "primary_unnormalized",
        "normalization": "none",
        "artifacts": SearchArtifactPaths(
            dataset_manifest="dataset.json",
            feature_cache_manifest="features.json",
            oracle_pair_manifest="pairs.json",
        ),
        "seeds": _seeds(),
        "search_space": _space(),
        "output_root": "output",
        "runtime": SearchRuntimeConfig(
            device="cpu", deterministic_algorithms=True, workers=1
        ),
        "relative_singular_value_tolerance": 1e-12,
    }
    if dataset == "cmnist":
        return CmnistProductionSearchConfig.model_validate(
            {
                **common,
                "dataset": "cmnist",
                "protocol_id": "cmnist/v1",
                "pair_count": 256,
                "selectors": ("primary_robust", "secondary_source"),
                "batch_size": 256,
                "max_epochs": 40,
            }
        )
    return WaterbirdsProductionSearchConfig.model_validate(
        {
            **common,
            "dataset": "waterbirds_cf",
            "protocol_id": "waterbirds_cf/v1",
            "pair_count": 240,
            "selectors": ("waterbirds_validation_worst_group",),
            "batch_size": 256,
            "max_epochs": 100,
        }
    )


def resolved_search_fixture(
    dataset: str = "cmnist",
) -> ResolvedProductionSearchConfig:
    config = _config(dataset)
    schema_versions = (
        (
            "grit.cmnist-dataset/v1",
            "grit.cmnist-features/v2",
            "grit.cmnist-oracle-pairs/v2",
        )
        if dataset == "cmnist"
        else (
            "grit.waterbirds-cf-dataset/v2",
            "grit.waterbirds-features/v2",
            "grit.waterbirds-oracle-pairs/v2",
        )
    )
    inputs = (
        VerifiedInputArtifact(
            kind="dataset_manifest",
            path="/prepared/dataset_manifest.json",
            digest="sha256:dataset",
            schema_version=schema_versions[0],
        ),
        VerifiedInputArtifact(
            kind="feature_manifest",
            path="/prepared/feature_manifest.json",
            digest="sha256:features",
            schema_version=schema_versions[1],
        ),
        VerifiedInputArtifact(
            kind="pair_manifest",
            path="/prepared/pair_manifest.json",
            digest="sha256:pairs",
            schema_version=schema_versions[2],
        ),
    )
    return ResolvedProductionSearchConfig(
        schema_version="grit.resolved-production-search/v1",
        authored_config_digest=config.canonical_digest(),
        authored_config_path=f"/configs/{dataset}.yaml",
        output_root=f"/outputs/{dataset}",
        config=config,
        lineage=SearchLineage(
            dataset_manifest_digest="sha256:dataset",
            feature_cache_manifest_digest="sha256:features",
            pair_manifest_digest="sha256:pairs",
            normalization="none",
            adjusted_weight_spec_digest=(
                "sha256:weights" if dataset == "waterbirds_cf" else None
            ),
        ),
        input_artifacts=inputs,
    )


def _write_stored_plan_fixture(
    root: Path,
    dataset: Literal["cmnist", "waterbirds_cf"] = "cmnist",
    *,
    adjusted_weight_spec_digest: str | None = None,
) -> tuple[Path, SearchPlan]:
    output_root = root / "output"
    config_payload = _config(dataset).model_dump(mode="json")
    config_payload["output_root"] = output_root.as_posix()
    config_type = (
        CmnistProductionSearchConfig
        if dataset == "cmnist"
        else WaterbirdsProductionSearchConfig
    )
    config = config_type.model_validate_json(json.dumps(config_payload))
    config_path = root / "production.yaml"
    authored_payload = _yaml_compatible_json(config.model_dump(mode="json")) + "\n"
    config_path.write_text(authored_payload, encoding="utf-8")

    resolved_payload = resolved_search_fixture(dataset).model_dump(mode="json")
    resolved_payload["authored_config_digest"] = config.canonical_digest()
    resolved_payload["authored_config_path"] = config_path.resolve().as_posix()
    resolved_payload["output_root"] = output_root.resolve().as_posix()
    resolved_payload["config"] = config.model_dump(mode="json")
    if adjusted_weight_spec_digest is not None:
        resolved_payload["lineage"]["adjusted_weight_spec_digest"] = (
            adjusted_weight_spec_digest
        )
    resolved = ResolvedProductionSearchConfig.model_validate_json(
        json.dumps(resolved_payload)
    )
    plan = build_search_plan(resolved)
    output_root.mkdir(parents=True)
    (output_root / "authored-config.yaml").write_text(
        authored_payload, encoding="utf-8"
    )
    (output_root / "resolved-config.json").write_text(
        resolved.canonical_json() + "\n", encoding="utf-8"
    )
    (output_root / "search-plan.json").write_text(
        plan.canonical_json() + "\n", encoding="utf-8"
    )
    return config_path, plan


def _filesystem_snapshot(root: Path) -> tuple[tuple[object, ...], ...]:
    entries: list[tuple[object, ...]] = []
    for path in sorted((root, *root.rglob("*"))):
        stat = path.stat()
        relative = "." if path == root else path.relative_to(root).as_posix()
        entries.append(
            (
                relative,
                "directory" if path.is_dir() else "file",
                stat.st_mode,
                stat.st_size,
                stat.st_mtime_ns,
                hashlib.sha256(path.read_bytes()).hexdigest()
                if path.is_file()
                else None,
            )
        )
    return tuple(entries)


def _digit_counts(indices: tuple[int, ...]) -> tuple[DigitCount, ...]:
    return tuple(
        DigitCount(
            digit=digit,
            count=sum(index % 10 == digit for index in indices),
        )
        for digit in range(10)
    )


def _partition_record(
    name: str,
    official_split: str,
    indices: tuple[int, ...],
) -> SourcePartitionRecord:
    return SourcePartitionRecord.model_validate(
        {
            "name": name,
            "official_split": official_split,
            "source_indices": indices,
            "digit_counts": _digit_counts(indices),
            "membership_digest": canonical_digest_value(
                {
                    "name": name,
                    "official_split": official_split,
                    "source_indices": indices,
                }
            ),
        }
    )


def _write_manifest_only_cmnist_production(
    root: Path,
    *,
    encoder: EncoderIdentity | None = None,
    invalid_official_train_universe: bool = False,
    held_out_flip_prob: float = 0.5,
) -> tuple[Path, Path, Path]:
    train_e01_indices = tuple(range(25_000))
    if invalid_official_train_universe:
        train_e01_indices = (*train_e01_indices[:-1], 60_000)
    partitions = (
        _partition_record("train_e01_sources", "train", train_e01_indices),
        _partition_record(
            "train_e02_sources", "train", tuple(range(25_000, 50_000))
        ),
        _partition_record(
            "validation_sources", "train", tuple(range(50_000, 60_000))
        ),
        _partition_record("test_sources", "test", tuple(range(10_000))),
    )
    partition_manifest = CmnistPartitionManifest(
        schema_version="grit.cmnist-partitions/v1",
        dataset_id="cmnist",
        construction_method_id="cmnist-stratified-hash-v1",
        construction_seed=17,
        targets=CmnistPartitionTargets(
            train_e01=25_000,
            train_e02=25_000,
            validation=10_000,
            test=10_000,
        ),
        partitions=partitions,
    )
    source_ids = {
        "train_e01": tuple(f"mnist:train:{index}" for index in range(25_000)),
        "train_e02": tuple(
            f"mnist:train:{index}" for index in range(25_000, 50_000)
        ),
        "validation": tuple(
            f"mnist:train:{index}" for index in range(50_000, 60_000)
        ),
        "test": tuple(f"mnist:test:{index}" for index in range(10_000)),
    }
    environments = tuple(
        EnvironmentManifest(
            name=spec.name,
            role=spec.role,
            source_partition_id=spec.source_partition_id,
            color_flip_prob=spec.color_flip_prob,
            count=(
                25_000
                if spec.name in {"train_e01", "train_e02"}
                else 10_000
            ),
            source_membership_digest=canonical_digest_value(
                source_ids[
                    "validation"
                    if spec.name.startswith("val_")
                    else "test"
                    if spec.name == "test_ood"
                    else spec.name
                ]
            ),
            record_digest=f"sha256:records:{spec.name}",
        )
        for spec in cmnist_environment_specs(held_out_flip_prob)
    )
    dataset = CmnistDatasetManifest(
        schema_version="grit.cmnist-dataset/v1",
        dataset_id="cmnist",
        partition_manifest=partition_manifest,
        partition_manifest_digest=partition_manifest.canonical_digest(),
        official_train_pool_digest="sha256:official-train-pool",
        official_test_pool_digest="sha256:official-test-pool",
        rendering_method_id="cmnist-rgb-render-v1",
        construction_seed=17,
        label_flip_prob=0.25,
        environments=environments,
    )
    dataset_digest = dataset.canonical_digest()
    pair_records = tuple(
        OraclePairRecord(
            pair_id=canonical_digest_value(
                {
                    "dataset_manifest_digest": dataset_digest,
                    "method": "cmnist-clean-oracle-pairs-v1",
                    "pair_seed": 23,
                    "source_id": f"mnist:train:{index}",
                    "orientation": "red_minus_green",
                }
            ),
            source_id=f"mnist:train:{index}",
            official_source_index=index,
            digit=index % 10,
            clean_label=int(index % 10 >= 5),
            noisy_target=int(index % 10 >= 5),
            left_color="red",
            right_color="green",
        )
        for index in range(256)
    )
    pairs = CmnistOraclePairManifest(
        schema_version="grit.cmnist-oracle-pairs/v2",
        dataset_manifest_digest=dataset_digest,
        construction_method_id="cmnist-clean-oracle-pairs-v1",
        pair_seed=23,
        requested_count=256,
        realized_count=256,
        source_partition_ids=("train_e01_sources", "train_e02_sources"),
        orientation="red_minus_green",
        records=pair_records,
        membership_digest=canonical_digest_value(
            tuple(record.source_id for record in pair_records)
        ),
    )
    table_specs: tuple[
        tuple[
            str,
            Literal["training", "validation", "final_test", "pair_projection"],
            tuple[str, ...],
        ],
        ...,
    ] = (
        ("train_e01", "training", source_ids["train_e01"]),
        ("train_e02", "training", source_ids["train_e02"]),
        ("val_e01", "validation", source_ids["validation"]),
        ("val_e02", "validation", source_ids["validation"]),
        (
            held_out_validation_name(held_out_flip_prob),
            "validation",
            source_ids["validation"],
        ),
        ("test_ood", "final_test", source_ids["test"]),
        (
            "oracle_pair_red",
            "pair_projection",
            tuple(record.source_id for record in pair_records),
        ),
        (
            "oracle_pair_green",
            "pair_projection",
            tuple(record.source_id for record in pair_records),
        ),
    )
    feature_root = root / "feature-cache"
    tables: list[FeatureTableManifest] = []
    for table_name, role, table_sources in table_specs:
        files: list[ArrayFileManifest] = []
        for field_name, shape, dtype in (
            ("features", (len(table_sources), 512), "float32"),
            ("digits", (len(table_sources),), "int64"),
            ("clean_labels", (len(table_sources),), "int64"),
            ("targets", (len(table_sources),), "int64"),
            ("colors", (len(table_sources),), "int64"),
        ):
            path = feature_root / table_name / f"{field_name}.npy"
            path.parent.mkdir(parents=True, exist_ok=True)
            payload = f"manifest-only:{table_name}:{field_name}".encode()
            path.write_bytes(payload)
            files.append(
                ArrayFileManifest.model_validate(
                    {
                        "field_name": field_name,
                        "relative_path": path.relative_to(feature_root).as_posix(),
                        "sha256": f"sha256:{hashlib.sha256(payload).hexdigest()}",
                        "shape": shape,
                        "dtype": dtype,
                    }
                )
            )
        tables.append(
            FeatureTableManifest(
                table_name=table_name,
                role=role,
                source_ids=table_sources,
                source_ids_digest=canonical_digest_value(table_sources),
                row_count=len(table_sources),
                feature_dimension=512,
                feature_dtype="float32",
                files=tuple(files),
            )
        )
    feature = CmnistFeatureCacheManifest(
        schema_version="grit.cmnist-features/v2",
        dataset_id="cmnist",
        source_manifest_digest=dataset_digest,
        pair_manifest_digest=pairs.canonical_digest(),
        encoder=encoder
        or EncoderIdentity(
            implementation="openai/CLIP",
            implementation_revision=(
                "d05afc436d78f1c48dc0dbf8e5980a9d471f35f6"
            ),
            model_name="ViT-B/32",
            weights_identity=(
                "sha256:40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af"
            ),
            preprocessing_identity=(
                "openai-clip-vit-b32-preprocess@"
                "d05afc436d78f1c48dc0dbf8e5980a9d471f35f6"
            ),
            raw_output_dimension=512,
        ),
        extraction_runtime=FeatureExtractionRuntime(
            requested_device="cpu",
            resolved_device="cpu",
            computation_dtype="torch.float32",
            deterministic_algorithms=True,
            tf32_enabled=False,
            mixed_precision=False,
            batch_size=256,
            torch_version=str(torch.__version__),
            cuda_runtime_version=None,
            device_name="cpu",
            compute_capability=None,
        ),
        normalization="none",
        feature_dimension=512,
        feature_dtype="float32",
        tables=tuple(tables),
    )
    dataset_path = root / "dataset-manifest.json"
    pair_path = root / "pair-manifest.json"
    feature_path = feature_root / "manifest.json"
    dataset_path.write_text(dataset.canonical_json(), encoding="utf-8")
    pair_path.write_text(pairs.canonical_json(), encoding="utf-8")
    feature_path.write_text(feature.canonical_json(), encoding="utf-8")
    return dataset_path, feature_path, pair_path


def write_cmnist_production_config(
    root: Path,
    *,
    output_root: Path,
    overrides: dict[str, object] | None = None,
    held_out_flip_prob: float = 0.5,
) -> tuple[Path, tuple[Path, Path, Path]]:
    artifact_paths = _write_manifest_only_cmnist_production(
        root / "prepared", held_out_flip_prob=held_out_flip_prob
    )
    payload = _config().model_dump(mode="json")
    payload.update(overrides or {})
    payload["artifacts"] = {
        "dataset_manifest": artifact_paths[0].as_posix(),
        "feature_cache_manifest": artifact_paths[1].as_posix(),
        "oracle_pair_manifest": artifact_paths[2].as_posix(),
    }
    payload["output_root"] = output_root.as_posix()
    config_path = root / "production.yaml"
    config_path.write_text(
        _yaml_compatible_json(payload),
        encoding="utf-8",
    )
    return config_path, artifact_paths


@pytest.mark.parametrize("dataset", ["cmnist", "waterbirds_cf"])
def test_production_plan_has_exact_deterministic_candidate_grid(dataset: str) -> None:
    first = build_search_plan(resolved_search_fixture(dataset))
    second = build_search_plan(resolved_search_fixture(dataset))
    assert first.candidates == second.candidates
    assert sum(item.method_id == "erm" for item in first.candidates) == 16
    assert sum(item.method_id == "grit" for item in first.candidates) == 368
    assert first.expected_run_counts.tuning == 1_152
    assert tuple(item.requested_rank for item in first.candidates[:16]) == (None,) * 16
    assert tuple(item.method_id for item in first.candidates) == (
        *("erm" for _ in range(16)),
        *("grit" for _ in range(368)),
    )


def test_yaml_mapping_order_does_not_change_config_or_candidates(
    tmp_path: Path,
) -> None:
    payload = _config().model_dump(mode="json")
    first_path = tmp_path / "first.yaml"
    second_path = tmp_path / "second.yaml"
    first_path.write_text(
        json.dumps(payload)
        .replace("1e-05", "0.00001")
        .replace("1e-12", "0.000000000001"),
        encoding="utf-8",
    )
    second_path.write_text(
        json.dumps(dict(reversed(tuple(payload.items())))).replace(
            "1e-05", "0.00001"
        ).replace("1e-12", "0.000000000001"),
        encoding="utf-8",
    )
    first = load_production_search_config(first_path)
    second = load_production_search_config(second_path)
    assert first == second
    assert build_search_plan(resolved_search_fixture()).candidates == build_search_plan(
        resolved_search_fixture()
    ).candidates


def test_search_plan_rejects_duplicate_config_and_candidate_id_collision() -> None:
    plan = build_search_plan(resolved_search_fixture())
    payload = plan.model_dump(mode="json")
    candidates = payload["candidates"]
    duplicate_config = dict(candidates[1])
    duplicate_config["scientific_config_digest"] = candidates[0][
        "scientific_config_digest"
    ]
    candidates[1] = duplicate_config
    with pytest.raises(ValidationError, match="candidate grid"):
        SearchPlan.model_validate_json(json.dumps(payload))

    payload = plan.model_dump(mode="json")
    candidates = payload["candidates"]
    collision = dict(candidates[1])
    collision["candidate_id"] = candidates[0]["candidate_id"]
    candidates[1] = collision
    with pytest.raises(ValidationError, match="candidate grid"):
        SearchPlan.model_validate_json(json.dumps(payload))


def test_production_config_requires_exact_disjoint_seed_sets() -> None:
    payload = _config().model_dump(mode="json")
    payload["seeds"]["stages"]["confirmation"] = [101, 202]
    with pytest.raises(ValidationError, match="disjoint"):
        CmnistProductionSearchConfig.model_validate_json(json.dumps(payload))
    payload = _config().model_dump(mode="json")
    payload["seeds"]["stages"]["tuning"] = [101, 102]
    with pytest.raises(ValidationError, match="contain 3"):
        CmnistProductionSearchConfig.model_validate_json(json.dumps(payload))


def test_plan_round_trip_revalidates_canonical_boundary() -> None:
    plan = build_search_plan(resolved_search_fixture("waterbirds_cf"))
    reparsed = SearchPlan.model_validate_json(plan.canonical_json())
    assert reparsed == plan
    assert reparsed.canonical_digest() == plan.canonical_digest()
    assert reparsed.code == CodeProvenance.model_validate(plan.code)
    assert reparsed.environment == EnvironmentProvenance.model_validate(
        plan.environment
    )
    payload = plan.model_dump(mode="json")
    payload["expected_run_counts"]["tuning"] = 1
    with pytest.raises(ValidationError, match="run counts"):
        SearchPlan.model_validate_json(json.dumps(payload))
    payload = plan.model_dump(mode="json")
    payload["output_schemas"][0]["schema_version"] = "forged/v1"
    with pytest.raises(ValidationError, match="schema inventory"):
        SearchPlan.model_validate_json(json.dumps(payload))


def test_planned_candidate_digests_materialize_into_dataset_configs() -> None:
    cmnist = build_search_plan(resolved_search_fixture("cmnist"))
    for candidate in cmnist.candidates:
        resolved = materialize_cmnist_candidate_config(
            cmnist, candidate, CmnistSelector.PRIMARY_ROBUST
        )
        assert resolved.scientific_config_digest() == (
            candidate.scientific_config_digest
        )

    waterbirds = build_search_plan(resolved_search_fixture("waterbirds_cf"))
    erm = waterbirds.candidates[0]
    assert materialize_waterbirds_candidate_config(
        waterbirds, erm, None
    ).scientific_config_digest() == erm.scientific_config_digest
    grit = next(
        item
        for item in waterbirds.candidates
        if item.method_id == "grit" and item.requested_rank == 24
    )
    left = torch.zeros((25, 512), dtype=torch.float32)
    left[:, :25] = torch.eye(25, dtype=torch.float32)
    pair_digest = waterbirds.resolved_config.lineage.pair_manifest_digest
    if not pair_digest:
        raise AssertionError("Waterbirds test plan lacks pair lineage")
    projection = fit_linear_projection(
        left,
        torch.zeros_like(left),
        requested_rank=24,
        pair_manifest_digest=pair_digest,
        feature_cache_manifest_digest=(
            waterbirds.resolved_config.lineage.feature_cache_manifest_digest
        ),
    )
    assert materialize_waterbirds_candidate_config(
        waterbirds, grit, projection
    ).scientific_config_digest() == grit.scientific_config_digest


def test_waterbirds_scientific_candidate_identity_excludes_stage_seeds() -> None:
    plan = build_search_plan(resolved_search_fixture("waterbirds_cf"))
    resolved = materialize_waterbirds_candidate_config(
        plan, plan.candidates[0], None
    )
    payload = resolved.model_dump(mode="json")
    payload["seed_sets"] = {
        "tuning": (901, 902, 903),
        "confirmation": (904, 905),
        "final": (906, 907, 908, 909, 910, 911, 912, 913, 914, 915),
    }
    changed = type(resolved).model_validate(payload)
    assert changed.scientific_config_digest() == resolved.scientific_config_digest()


def test_plan_only_verifies_official_manifest_lineage_without_training(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def forbidden(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("plan-only operation crossed a runtime/final boundary")

    monkeypatch.setattr("grit.search.cmnist._load_cmnist_cache", forbidden)
    monkeypatch.setattr("grit.search.cmnist.train_linear_probe", forbidden)
    monkeypatch.setattr("grit.search.cmnist.open_final_test", forbidden)
    dataset_path, feature_path, pair_path = (
        _write_manifest_only_cmnist_production(tmp_path / "prepared")
    )
    cuda_feature_payload = cast(
        dict[str, object], json.loads(feature_path.read_text(encoding="utf-8"))
    )
    cuda_feature_payload["extraction_runtime"] = {
        "requested_device": "cuda",
        "resolved_device": "cuda:0",
        "computation_dtype": "torch.float32",
        "deterministic_algorithms": True,
        "tf32_enabled": False,
        "mixed_precision": False,
        "batch_size": 128,
        "torch_version": "2.11.0+cu128",
        "cuda_runtime_version": "12.8",
        "device_name": "NVIDIA RTX A5000",
        "compute_capability": [8, 6],
    }
    feature_path.write_text(json.dumps(cuda_feature_payload), encoding="utf-8")
    config = _config()
    payload = config.model_dump(mode="json")
    payload["artifacts"] = {
        "dataset_manifest": dataset_path.as_posix(),
        "feature_cache_manifest": feature_path.as_posix(),
        "oracle_pair_manifest": pair_path.as_posix(),
    }
    payload["output_root"] = (tmp_path / "planned").as_posix()
    config_path = tmp_path / "production.yaml"
    config_path.write_text(
        json.dumps(payload)
        .replace("1e-05", "0.00001")
        .replace("1e-12", "0.000000000001"),
        encoding="utf-8",
    )
    plan = plan_production_search(config_path)
    assert len(plan.candidates) == 384
    assert plan.resolved_config.lineage.feature_cache_manifest_digest
    assert plan_production_search(config_path) == plan
    assert not (tmp_path / "planned" / "runs").exists()
    assert not tuple((tmp_path / "planned").rglob("*checkpoint*"))

    feature_payload = cast(
        dict[str, object],
        json.loads(feature_path.read_text(encoding="utf-8")),
    )
    encoder_payload = cast(dict[str, object], feature_payload["encoder"])
    encoder_payload["implementation"] = "grit.synthetic"
    feature_path.write_text(json.dumps(feature_payload), encoding="utf-8")
    rejected = dict(payload)
    rejected["output_root"] = (tmp_path / "rejected").as_posix()
    rejected_path = tmp_path / "fake-encoder.yaml"
    rejected_path.write_text(
        json.dumps(rejected)
        .replace("1e-05", "0.00001")
        .replace("1e-12", "0.000000000001"),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="official OpenAI CLIP"):
        plan_production_search(rejected_path)
    assert not (tmp_path / "rejected").exists()


def test_cmnist_plan_rejects_nonofficial_source_universe(tmp_path: Path) -> None:
    dataset_path, feature_path, pair_path = _write_manifest_only_cmnist_production(
        tmp_path / "prepared", invalid_official_train_universe=True
    )
    payload = _config().model_dump(mode="json")
    payload["artifacts"] = {
        "dataset_manifest": dataset_path.as_posix(),
        "feature_cache_manifest": feature_path.as_posix(),
        "oracle_pair_manifest": pair_path.as_posix(),
    }
    payload["output_root"] = (tmp_path / "planned").as_posix()
    config_path = tmp_path / "production.yaml"
    config_path.write_text(
        json.dumps(payload)
        .replace("1e-05", "0.00001")
        .replace("1e-12", "0.000000000001"),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="official source universes"):
        plan_production_search(config_path)


@pytest.mark.skipif(
    not (Path(__file__).resolve().parents[1] / ".git").exists(),
    reason="Git-ignore check needs a Git checkout; a source archive has none",
)
def test_plan_rejects_nonignored_output_inside_repository(tmp_path: Path) -> None:
    dataset_path, feature_path, pair_path = _write_manifest_only_cmnist_production(
        tmp_path / "prepared"
    )
    payload = _config().model_dump(mode="json")
    payload["artifacts"] = {
        "dataset_manifest": dataset_path.as_posix(),
        "feature_cache_manifest": feature_path.as_posix(),
        "oracle_pair_manifest": pair_path.as_posix(),
    }
    payload["output_root"] = (
        Path.cwd() / "unignored-test-output"
    ).as_posix()
    config_path = tmp_path / "unsafe-output.yaml"
    config_path.write_text(
        json.dumps(payload)
        .replace("1e-05", "0.00001")
        .replace("1e-12", "0.000000000001"),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="must be Git-ignored"):
        plan_production_search(config_path)
    assert not (Path.cwd() / "unignored-test-output").exists()


def test_waterbirds_plan_only_accepts_verified_production_manifests(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def forbidden(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("plan-only operation crossed a runtime/final boundary")

    monkeypatch.setattr(
        "grit.search.waterbirds._load_cache", forbidden
    )
    monkeypatch.setattr(
        "grit.search.waterbirds.train_waterbirds_linear_probe",
        forbidden,
    )
    dataset_path, feature_path, pair_path = (
        write_manifest_only_waterbirds_production(tmp_path / "prepared-waterbirds")
    )
    config = _config("waterbirds_cf")
    payload = config.model_dump(mode="json")
    payload["artifacts"] = {
        "dataset_manifest": dataset_path.as_posix(),
        "feature_cache_manifest": feature_path.as_posix(),
        "oracle_pair_manifest": pair_path.as_posix(),
    }
    payload["output_root"] = (tmp_path / "waterbirds-plan").as_posix()
    config_path = tmp_path / "waterbirds-production.yaml"
    config_path.write_text(
        json.dumps(payload)
        .replace("1e-05", "0.00001")
        .replace("1e-12", "0.000000000001"),
        encoding="utf-8",
    )
    plan = plan_production_search(config_path)
    assert plan.dataset == "waterbirds_cf"
    assert len(plan.candidates) == 384
    assert plan.resolved_config.lineage.adjusted_weight_spec_digest is not None
    assert not (tmp_path / "waterbirds-plan" / "runs").exists()

    original_feature = feature_path.read_text(encoding="utf-8")
    feature_payload = cast(dict[str, object], json.loads(original_feature))
    feature_payload["normalization"] = "l2"
    feature_path.write_text(json.dumps(feature_payload), encoding="utf-8")
    mixed_payload = dict(payload)
    mixed_payload["output_root"] = (tmp_path / "mixed-normalization").as_posix()
    mixed_path = tmp_path / "mixed-normalization.yaml"
    mixed_path.write_text(
        json.dumps(mixed_payload)
        .replace("1e-05", "0.00001")
        .replace("1e-12", "0.000000000001"),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="normalization"):
        plan_production_search(mixed_path)
    feature_path.write_text(original_feature, encoding="utf-8")

    feature_payload = cast(dict[str, object], json.loads(original_feature))
    feature_payload["dataset_manifest_digest"] = "sha256:another-dataset"
    feature_path.write_text(json.dumps(feature_payload), encoding="utf-8")
    mixed_payload["output_root"] = (tmp_path / "mixed-dataset").as_posix()
    mixed_path.write_text(
        json.dumps(mixed_payload)
        .replace("1e-05", "0.00001")
        .replace("1e-12", "0.000000000001"),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="lineage"):
        plan_production_search(mixed_path)
    feature_path.write_text(original_feature, encoding="utf-8")

    feature_payload = cast(dict[str, object], json.loads(original_feature))
    encoder_payload = cast(dict[str, object], feature_payload["encoder"])
    encoder_payload["implementation"] = "grit.synthetic"
    feature_path.write_text(json.dumps(feature_payload), encoding="utf-8")
    mixed_payload["output_root"] = (tmp_path / "fake-encoder").as_posix()
    mixed_path.write_text(
        json.dumps(mixed_payload)
        .replace("1e-05", "0.00001")
        .replace("1e-12", "0.000000000001"),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="official OpenAI CLIP"):
        plan_production_search(mixed_path)
    feature_path.write_text(original_feature, encoding="utf-8")

    original_pair = pair_path.read_text(encoding="utf-8")
    pair_payload = cast(dict[str, object], json.loads(original_pair))
    pair_payload["dataset_manifest_digest"] = "sha256:another-dataset"
    pair_path.write_text(json.dumps(pair_payload), encoding="utf-8")
    mixed_payload["output_root"] = (tmp_path / "mixed-pairs").as_posix()
    mixed_path.write_text(
        json.dumps(mixed_payload)
        .replace("1e-05", "0.00001")
        .replace("1e-12", "0.000000000001"),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="lineage"):
        plan_production_search(mixed_path)
    pair_path.write_text(original_pair, encoding="utf-8")

    dataset_payload = cast(
        dict[str, object], json.loads(dataset_path.read_text(encoding="utf-8"))
    )
    dataset_payload["non_reportable"] = True
    dataset_path.write_text(json.dumps(dataset_payload), encoding="utf-8")
    mixed_payload["output_root"] = (tmp_path / "non-reportable").as_posix()
    mixed_path.write_text(
        json.dumps(mixed_payload)
        .replace("1e-05", "0.00001")
        .replace("1e-12", "0.000000000001"),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="production Waterbirds manifest"):
        plan_production_search(mixed_path)


def test_dirty_worktree_is_recorded_not_rejected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output_root = tmp_path / "output"
    config_path, _ = write_cmnist_production_config(
        tmp_path, output_root=output_root
    )
    dirty = CodeProvenance(git_revision="2" * 40, git_dirty=True)
    monkeypatch.setattr("grit.search.plan._code_provenance", lambda: dirty)
    plan = plan_production_search(config_path)
    assert plan.code == dirty
    # A later commit continues the same plan instead of rejecting it.
    clean = CodeProvenance(git_revision="3" * 40, git_dirty=False)
    monkeypatch.setattr("grit.search.plan._code_provenance", lambda: clean)
    assert plan_production_search(config_path).code == dirty


def test_ignored_repository_output_remains_an_allowed_isolated_root(
    tmp_path: Path,
) -> None:
    ignored_output = Path.cwd() / "outputs" / f"production-safety-{tmp_path.name}"
    config_path, _ = write_cmnist_production_config(
        tmp_path,
        output_root=ignored_output,
    )
    config = load_production_search_config(config_path)
    resolved = resolve_production_search_config(config, config_path=config_path)
    assert Path(resolved.output_root) == ignored_output
    assert not Path(resolved.output_root).exists()


@pytest.mark.parametrize("unsafe_kind", ("filesystem", "repository"))
def test_plan_rejects_broad_output_roots(
    tmp_path: Path,
    unsafe_kind: Literal["filesystem", "repository"],
) -> None:
    output_root = Path("/") if unsafe_kind == "filesystem" else Path.cwd()
    config_path, _ = write_cmnist_production_config(
        tmp_path, output_root=output_root
    )
    with pytest.raises(ValueError, match="filesystem root|repository root"):
        plan_production_search(config_path)


@pytest.mark.parametrize("placement", ("prepared", "nested", "contains"))
def test_plan_rejects_output_that_overlaps_prepared_artifacts(
    tmp_path: Path,
    placement: Literal["prepared", "nested", "contains"],
) -> None:
    container = tmp_path / "container"
    prepared = container / "prepared"
    if placement == "prepared":
        output_root = prepared
    elif placement == "nested":
        output_root = prepared / "output"
    else:
        output_root = container
    artifact_paths = _write_manifest_only_cmnist_production(prepared)
    payload = _config().model_dump(mode="json")
    payload["artifacts"] = {
        "dataset_manifest": artifact_paths[0].as_posix(),
        "feature_cache_manifest": artifact_paths[1].as_posix(),
        "oracle_pair_manifest": artifact_paths[2].as_posix(),
    }
    payload["output_root"] = output_root.as_posix()
    config_path = tmp_path / "overlap.yaml"
    config_path.write_text(_yaml_compatible_json(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="prepared-artifact|contain prepared"):
        plan_production_search(config_path)


def test_first_plan_refuses_nonempty_unrelated_output_directory(
    tmp_path: Path,
) -> None:
    output_root = tmp_path / "output"
    output_root.mkdir()
    marker = output_root / "unrelated.txt"
    marker.write_text("preserve me", encoding="utf-8")
    config_path, _ = write_cmnist_production_config(
        tmp_path, output_root=output_root
    )
    with pytest.raises(ValueError, match="empty or contain a complete compatible plan"):
        plan_production_search(config_path)
    assert marker.read_text(encoding="utf-8") == "preserve me"
    assert not (output_root / "search-plan.json").exists()


def test_first_plan_accepts_an_empty_dedicated_output_directory(
    tmp_path: Path,
) -> None:
    output_root = tmp_path / "output"
    output_root.mkdir()
    config_path, _ = write_cmnist_production_config(
        tmp_path, output_root=output_root
    )
    plan = plan_production_search(config_path)
    assert Path(plan.resolved_config.output_root) == output_root
    assert (output_root / "search-plan.json").is_file()


def test_status_requires_existing_matching_triplet_and_never_creates_it(
    tmp_path: Path,
) -> None:
    output_root = tmp_path / "missing-output"
    config_payload = _config().model_dump(mode="json")
    config_payload["output_root"] = output_root.as_posix()
    config_path = tmp_path / "production.yaml"
    config_path.write_text(
        _yaml_compatible_json(config_payload), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="run scripts/run_search.py first"):
        production_search_status(config_path)
    assert not output_root.exists()


def test_status_is_read_only_and_uses_saved_clean_provenance_from_dirty_tree(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_path, plan = _write_stored_plan_fixture(tmp_path)
    output_root = Path(plan.resolved_config.output_root)
    before = _filesystem_snapshot(output_root)

    def forbidden(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("status crossed a planning, writing, or runtime boundary")

    monkeypatch.setattr("grit.search.run.plan_production_search", forbidden)
    monkeypatch.setattr("grit.search.run.write_search_plan", forbidden)
    monkeypatch.setattr("grit.search.cmnist._load_cmnist_cache", forbidden)
    monkeypatch.setattr("grit.search.plan._code_provenance", forbidden)
    status = production_search_status(config_path)
    after = _filesystem_snapshot(output_root)
    assert status.phase == "tuning"
    assert status.tuning_complete == 0
    assert before == after

    config_path.write_text(
        config_path.read_text(encoding="utf-8").replace(
            "cmnist-primary", "cmnist-renamed"
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="does not match the supplied YAML"):
        production_search_status(config_path)


def _cmnist_executor(
    plan: SearchPlan,
    calls: list[str],
    *,
    fail_once: list[bool] | None = None,
):
    def execute(task: SearchRunTask, _run_root: Path) -> CompletedStageRun:
        calls.append(task.task_id)
        if fail_once is not None and fail_once:
            fail_once.pop()
            raise RuntimeError("synthetic interruption")
        records: list[ValidationMetricRecord] = []
        for epoch, base in ((0, 0.60), (1, 0.70)):
            checkpoint_id = f"checkpoint:{task.task_id}:{epoch}"
            for split_name, offset in zip(
                ("val_e01", "val_e02", "val_e05"),
                (0.02, 0.01, 0.0),
                strict=True,
            ):
                records.append(
                    ValidationMetricRecord(
                        record_id=f"metric:{checkpoint_id}:{split_name}",
                        run_id=f"run:{task.task_id}",
                        candidate_id=task.candidate.candidate_id,
                        method_id=task.candidate.method_id,
                        scientific_config_digest=(
                            task.candidate.scientific_config_digest
                        ),
                        checkpoint_id=checkpoint_id,
                        epoch=epoch,
                        seed=task.seed,
                        value=base + offset,
                        sample_count=10,
                        metric_kind="validation",
                        seed_stage=task.stage,
                        split_name=split_name,
                        metric_name="accuracy",
                        projection_rank=task.candidate.requested_rank,
                    )
                )
        materialized = tuple(records)
        selectors = (
            CmnistSelector.PRIMARY_ROBUST,
            CmnistSelector.SECONDARY_SOURCE,
        )
        return CmnistCompletedStageRun(
            schema_version="grit.cmnist-search-stage-run/v1",
            dataset="cmnist",
            status="complete",
            task=task,
            lineage=plan.resolved_config.lineage,
            validation_metrics=materialized,
            checkpoint_decisions=tuple(
                select_checkpoint(materialized, selector) for selector in selectors
            ),
        )

    return execute


def _cmnist_stage_runs(
    plan: SearchPlan,
    candidates: Sequence[SearchCandidate],
    stage: Literal[SeedStage.TUNING, SeedStage.CONFIRMATION],
) -> tuple[CmnistCompletedStageRun, ...]:
    seeds = (
        plan.seeds.stages.tuning
        if stage is SeedStage.TUNING
        else plan.seeds.stages.confirmation
    )
    execute = _cmnist_executor(plan, [])
    return tuple(
        cast(
            CmnistCompletedStageRun,
            execute(make_search_task(plan, candidate, stage, seed), Path(".")),
        )
        for candidate in candidates
        for seed in seeds
    )


def _waterbirds_weight_spec() -> WaterbirdsAdjustedWeightSpec:
    return WaterbirdsAdjustedWeightSpec(
        schema_version="grit.waterbirds-adjusted-weights/v1",
        group_order=WATERBIRDS_GROUP_ORDER,
        training_group_counts=WaterbirdsGroupCounts(
            landbird_land=3_498,
            landbird_water=184,
            waterbird_land=56,
            waterbird_water=1_057,
        ),
        dataset_manifest_digest="sha256:dataset",
    )


def _waterbirds_stage_run(
    task: SearchRunTask,
    weights: WaterbirdsAdjustedWeightSpec,
) -> WaterbirdsCompletedStageRun:
    correct = 7
    groups = tuple(
        WaterbirdsGroupAccuracy(
            group_id=group_id,
            count=10,
            correct=correct,
            accuracy=correct / 10,
        )
        for group_id in WATERBIRDS_GROUP_ORDER
    )
    typed_groups = (groups[0], groups[1], groups[2], groups[3])
    checkpoint_id = f"checkpoint:{task.task_id}:0"
    metric = WaterbirdsValidationMetricRecord(
        record_id=f"metric:{checkpoint_id}",
        run_id=f"run:{task.task_id}",
        candidate_id=task.candidate.candidate_id,
        method_id=task.candidate.method_id,
        scientific_config_digest=task.candidate.scientific_config_digest,
        dataset_manifest_digest=task.lineage.dataset_manifest_digest,
        feature_cache_manifest_digest=task.lineage.feature_cache_manifest_digest,
        normalization=task.lineage.normalization,
        adjusted_weight_spec_digest=weights.canonical_digest(),
        checkpoint_id=checkpoint_id,
        epoch=0,
        seed=task.seed,
        projection_rank=task.candidate.requested_rank,
        groups=typed_groups,
        adjusted_weight_spec=weights,
        worst_group_accuracy=0.7,
        adjusted_average_accuracy=0.7,
        raw_average_accuracy=0.7,
        metric_kind="validation",
        split_name="validation",
        seed_stage=task.stage,
    )
    return WaterbirdsCompletedStageRun(
        schema_version="grit.waterbirds-search-stage-run/v1",
        dataset="waterbirds_cf",
        status="complete",
        task=task,
        lineage=task.lineage,
        validation_metrics=(metric,),
        checkpoint_decision=select_waterbirds_checkpoint((metric,)),
    )


def _waterbirds_stage_runs(
    plan: SearchPlan,
    candidates: Sequence[SearchCandidate],
    stage: Literal[SeedStage.TUNING, SeedStage.CONFIRMATION],
    weights: WaterbirdsAdjustedWeightSpec,
) -> tuple[WaterbirdsCompletedStageRun, ...]:
    seeds = (
        plan.seeds.stages.tuning
        if stage is SeedStage.TUNING
        else plan.seeds.stages.confirmation
    )
    return tuple(
        _waterbirds_stage_run(
            make_search_task(plan, candidate, stage, seed), weights
        )
        for candidate in candidates
        for seed in seeds
    )


def _mock_complete_transition_stages(
    monkeypatch: pytest.MonkeyPatch,
    tuning_runs: tuple[CompletedStageRun, ...],
    confirmation_runs: tuple[CompletedStageRun, ...],
) -> None:
    def status(
        _scheduler: LocalRunScheduler,
        tasks: Sequence[SearchRunTask],
    ) -> SearchStatus:
        task_ids = tuple(task.task_id for task in tasks)
        if tasks and tasks[0].stage is SeedStage.FINAL:
            complete_ids: tuple[str, ...] = ()
            missing_ids = task_ids
        else:
            complete_ids = task_ids
            missing_ids = ()
        return SearchStatus(
            schema_version="grit.search-status/v1",
            plan_digest=tasks[0].plan_digest,
            complete_task_ids=complete_ids,
            missing_task_ids=missing_ids,
            interrupted_task_ids=(),
        )

    def completed_results(
        _scheduler: LocalRunScheduler,
        tasks: Sequence[SearchRunTask],
    ) -> tuple[CompletedStageRun, ...]:
        if not tasks:
            raise AssertionError("status requested an empty completed stage")
        return (
            tuning_runs
            if tasks[0].stage is SeedStage.TUNING
            else confirmation_runs
        )

    monkeypatch.setattr(LocalRunScheduler, "status", status)
    monkeypatch.setattr(
        LocalRunScheduler, "completed_results", completed_results
    )


def test_status_recomputes_cmnist_finalists_unions_and_winners_read_only(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_path, plan = _write_stored_plan_fixture(tmp_path)
    selected_candidates = tuple(
        candidate
        for method in ("erm", "grit")
        for candidate in tuple(
            item for item in plan.candidates if item.method_id == method
        )[:3]
    )
    tuning_runs = _cmnist_stage_runs(
        plan, selected_candidates, SeedStage.TUNING
    )
    finalists, unions = compute_cmnist_finalists(plan, tuning_runs)
    root = Path(plan.resolved_config.output_root)
    for method in ("erm", "grit"):
        selection_root = root / "selection" / method
        persist_canonical_artifact(
            selection_root / "primary-tuning-finalists.json",
            finalists[(method, CmnistSelector.PRIMARY_ROBUST)],
        )
        persist_canonical_artifact(
            selection_root / "secondary-tuning-finalists.json",
            finalists[(method, CmnistSelector.SECONDARY_SOURCE)],
        )
        union = unions[method]
        assert union is not None
        persist_canonical_artifact(
            selection_root / "confirmation-union.json", union
        )
    by_id = {candidate.candidate_id: candidate for candidate in plan.candidates}
    confirmation_candidates = tuple(
        by_id[candidate_id]
        for method in ("erm", "grit")
        for union in (unions[method],)
        if union is not None
        for candidate_id in union.confirmation_candidate_ids
    )
    confirmation_runs = _cmnist_stage_runs(
        plan, confirmation_candidates, SeedStage.CONFIRMATION
    )
    winners = compute_cmnist_winners(plan, finalists, confirmation_runs)
    for (method, selector), winner in winners.items():
        persist_canonical_artifact(
            root / "selection" / method / f"{selector.value}-winner.json",
            winner,
        )
    _mock_complete_transition_stages(
        monkeypatch,
        cast(tuple[CompletedStageRun, ...], tuning_runs),
        cast(tuple[CompletedStageRun, ...], confirmation_runs),
    )
    before = _filesystem_snapshot(root)
    status = production_search_status(config_path)
    assert status.phase == "final"
    assert status.frozen_winner_count == 4
    assert _filesystem_snapshot(root) == before

    primary_path = root / "selection/erm/primary_robust-winner.json"
    primary_payload = primary_path.read_text(encoding="utf-8")
    primary_path.write_text(
        winners[("erm", CmnistSelector.SECONDARY_SOURCE)].canonical_json(),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="canonical confirmation"):
        production_search_status(config_path)
    primary_path.write_text(primary_payload, encoding="utf-8")

    finalist_path = root / "selection/erm/primary-tuning-finalists.json"
    original_finalist_payload = finalists[
        ("erm", CmnistSelector.PRIMARY_ROBUST)
    ].canonical_json()
    wrong_seed = cast(dict[str, object], json.loads(original_finalist_payload))
    wrong_seed["tuning_seeds"] = [999, 102, 103]
    finalist_path.write_text(json.dumps(wrong_seed), encoding="utf-8")
    with pytest.raises(ValueError, match="tuning"):
        production_search_status(config_path)

    out_of_plan = cast(dict[str, object], json.loads(original_finalist_payload))
    ordered = cast(list[dict[str, object]], out_of_plan["ordered_candidates"])
    ordered[0]["candidate_id"] = "candidate:000-forged"
    decisions = cast(
        list[dict[str, object]],
        ordered[0]["contributing_checkpoint_decisions"],
    )
    for decision in decisions:
        checkpoint = cast(dict[str, object], decision["checkpoint"])
        checkpoint["candidate_id"] = "candidate:000-forged"
    finalist_path.write_text(json.dumps(out_of_plan), encoding="utf-8")
    with pytest.raises(ValueError, match="canonical tuning"):
        production_search_status(config_path)

    wrong_method_payload = finalists[
        ("grit", CmnistSelector.PRIMARY_ROBUST)
    ].canonical_json()
    finalist_path.write_text(wrong_method_payload, encoding="utf-8")
    with pytest.raises(ValueError, match="canonical tuning"):
        production_search_status(config_path)


def test_status_recomputes_waterbirds_finalists_and_winners(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    weights = _waterbirds_weight_spec()
    config_path, plan = _write_stored_plan_fixture(
        tmp_path,
        "waterbirds_cf",
        adjusted_weight_spec_digest=weights.canonical_digest(),
    )
    selected_candidates = tuple(
        candidate
        for method in ("erm", "grit")
        for candidate in tuple(
            item for item in plan.candidates if item.method_id == method
        )[:3]
    )
    tuning_runs = _waterbirds_stage_runs(
        plan, selected_candidates, SeedStage.TUNING, weights
    )
    finalists = compute_waterbirds_finalists(plan, tuning_runs)
    root = Path(plan.resolved_config.output_root)
    for method, finalist in finalists.items():
        persist_canonical_artifact(
            root / "selection" / method / "tuning-finalists.json", finalist
        )
    by_id = {candidate.candidate_id: candidate for candidate in plan.candidates}
    confirmation_candidates = tuple(
        by_id[item.candidate_id]
        for method in ("erm", "grit")
        for item in finalists[method].ordered_candidates
    )
    confirmation_runs = _waterbirds_stage_runs(
        plan, confirmation_candidates, SeedStage.CONFIRMATION, weights
    )
    winners = compute_waterbirds_winners(plan, finalists, confirmation_runs)
    for method, winner in winners.items():
        persist_canonical_artifact(
            root / "selection" / method / "winner.json", winner
        )
    _mock_complete_transition_stages(
        monkeypatch,
        cast(tuple[CompletedStageRun, ...], tuning_runs),
        cast(tuple[CompletedStageRun, ...], confirmation_runs),
    )
    status = production_search_status(config_path)
    assert status.phase == "final"
    assert status.frozen_winner_count == 2

    finalist_path = root / "selection/erm/tuning-finalists.json"
    original = cast(
        dict[str, object],
        json.loads(finalists["erm"].canonical_json()),
    )
    original["dataset_manifest_digest"] = "sha256:forged-dataset"
    original["feature_cache_manifest_digest"] = "sha256:forged-cache"
    original["adjusted_weight_spec_digest"] = "sha256:forged-weights"
    for candidate in cast(
        list[dict[str, object]], original["ordered_candidates"]
    ):
        candidate["dataset_manifest_digest"] = "sha256:forged-dataset"
        candidate["feature_cache_manifest_digest"] = "sha256:forged-cache"
        candidate["adjusted_weight_spec_digest"] = "sha256:forged-weights"
        for decision in cast(
            list[dict[str, object]], candidate["checkpoint_decisions"]
        ):
            decision["dataset_manifest_digest"] = "sha256:forged-dataset"
            decision["feature_cache_manifest_digest"] = "sha256:forged-cache"
            decision["adjusted_weight_spec_digest"] = "sha256:forged-weights"
    finalist_path.write_text(json.dumps(original), encoding="utf-8")
    with pytest.raises(ValueError, match="canonical tuning"):
        production_search_status(config_path)

    finalist_path.write_text(
        finalists["grit"].canonical_json(), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="canonical tuning"):
        production_search_status(config_path)


def test_completed_run_reuse_and_missing_run_continuation(tmp_path: Path) -> None:
    plan = build_search_plan(resolved_search_fixture())
    tasks = tuple(
        make_search_task(plan, plan.candidates[0], SeedStage.TUNING, seed)
        for seed in plan.seeds.stages.tuning[:2]
    )
    scheduler = LocalRunScheduler(tmp_path, plan)
    calls: list[str] = []
    first = scheduler.run_tasks(tasks[:1], _cmnist_executor(plan, calls))
    resumed = scheduler.run_tasks(tasks, _cmnist_executor(plan, calls))
    assert resumed[0] == first[0]
    assert calls == [tasks[0].task_id, tasks[1].task_id]
    status = scheduler.status(tasks)
    assert status.complete_task_ids == tuple(task.task_id for task in tasks)
    assert not status.missing_task_ids
    assert scheduler.completed_results(tasks) == resumed


def test_corrupt_or_incompatible_prior_result_is_rejected(tmp_path: Path) -> None:
    plan = build_search_plan(resolved_search_fixture())
    task = make_search_task(
        plan, plan.candidates[0], SeedStage.TUNING, plan.seeds.stages.tuning[0]
    )
    scheduler = LocalRunScheduler(tmp_path, plan)
    _ = scheduler.run_tasks((task,), _cmnist_executor(plan, []))
    result_path = tmp_path / task.relative_directory / "result.json"
    result_path.write_text("{not-json", encoding="utf-8")
    with pytest.raises(ValueError, match="corrupted"):
        scheduler.run_tasks((task,), _cmnist_executor(plan, []))

    other_plan = build_search_plan(resolved_search_fixture("waterbirds_cf"))
    with pytest.raises(ValueError, match="another plan"):
        LocalRunScheduler(tmp_path / "other", other_plan).run_tasks(
            (task,), _cmnist_executor(plan, [])
        )

    forged_payload = task.model_dump(mode="json")
    forged_payload["lineage"]["dataset_manifest_digest"] = "sha256:forged"
    forged_lineage = SearchLineage.model_validate(forged_payload["lineage"])
    identity_payload = {
        "plan_digest": task.plan_digest,
        "dataset": task.dataset,
        "lineage_digest": forged_lineage.canonical_digest(),
        "candidate_id": task.candidate.candidate_id,
        "scientific_config_digest": task.candidate.scientific_config_digest,
        "method_id": task.candidate.method_id,
        "stage": task.stage.value,
        "seed": task.seed,
        "selector": task.selector,
        "frozen_winner_digest": None,
    }
    forged_payload["task_id"] = (
        "task:" + SearchRunTask.canonical_identity_digest(identity_payload)
    )
    forged = SearchRunTask.model_validate(forged_payload)
    calls: list[str] = []
    with pytest.raises(ValueError, match="input lineage"):
        LocalRunScheduler(tmp_path / "forged", plan).run_tasks(
            (forged,), _cmnist_executor(plan, calls)
        )
    assert not calls


def test_interrupted_run_is_archived_then_reproduced(tmp_path: Path) -> None:
    plan = build_search_plan(resolved_search_fixture())
    task = make_search_task(
        plan, plan.candidates[0], SeedStage.TUNING, plan.seeds.stages.tuning[0]
    )
    resumed_scheduler = LocalRunScheduler(tmp_path / "resumed", plan)
    with pytest.raises(RuntimeError, match="synthetic interruption"):
        resumed_scheduler.run_tasks(
            (task,), _cmnist_executor(plan, [], fail_once=[True])
        )
    interrupted = resumed_scheduler.status((task,))
    assert interrupted.interrupted_task_ids == (task.task_id,)
    resumed = resumed_scheduler.run_tasks((task,), _cmnist_executor(plan, []))

    uninterrupted = LocalRunScheduler(tmp_path / "uninterrupted", plan).run_tasks(
        (task,), _cmnist_executor(plan, [])
    )
    assert resumed == uninterrupted
    assert not (
        tmp_path / "resumed" / task.relative_directory / "run-state.json"
    ).exists()
    staging_parent = (tmp_path / "resumed" / task.relative_directory).parent
    assert any("interrupted-1" in path.name for path in staging_parent.iterdir())


def test_interrupted_continuation_preserves_finalist_selection_summary(
    tmp_path: Path,
) -> None:
    plan = build_search_plan(resolved_search_fixture())
    candidates = tuple(
        candidate for candidate in plan.candidates if candidate.method_id == "erm"
    )[:3]
    tasks = tuple(
        make_search_task(plan, candidate, SeedStage.TUNING, seed)
        for candidate in candidates
        for seed in plan.seeds.stages.tuning
    )
    resumed_scheduler = LocalRunScheduler(tmp_path / "resumed-selection", plan)
    _ = resumed_scheduler.run_tasks(
        tasks[:3], _cmnist_executor(plan, [])
    )
    with pytest.raises(RuntimeError, match="synthetic interruption"):
        resumed_scheduler.run_tasks(
            tasks[3:5], _cmnist_executor(plan, [], fail_once=[True])
        )
    resumed = resumed_scheduler.run_tasks(tasks, _cmnist_executor(plan, []))
    uninterrupted = LocalRunScheduler(
        tmp_path / "uninterrupted-selection", plan
    ).run_tasks(tasks, _cmnist_executor(plan, []))

    def selection_summary(
        runs: tuple[CompletedStageRun, ...],
    ) -> tuple[tuple[str, ...], tuple[str, ...]]:
        records = tuple(
            metric
            for run in runs
            if isinstance(run, CmnistCompletedStageRun)
            for metric in run.validation_metrics
        )
        primary = make_tuning_finalists(
            records, CmnistSelector.PRIMARY_ROBUST, plan.seeds.stages
        )
        secondary = make_tuning_finalists(
            records, CmnistSelector.SECONDARY_SOURCE, plan.seeds.stages
        )
        return (
            tuple(item.candidate_id for item in primary.ordered_candidates),
            tuple(item.candidate_id for item in secondary.ordered_candidates),
        )

    assert resumed == uninterrupted
    assert selection_summary(resumed) == selection_summary(uninterrupted)


def test_completed_staging_result_is_promoted_without_reexecution(
    tmp_path: Path,
) -> None:
    plan = build_search_plan(resolved_search_fixture())
    task = make_search_task(
        plan, plan.candidates[0], SeedStage.TUNING, plan.seeds.stages.tuning[0]
    )
    scheduler = LocalRunScheduler(tmp_path, plan)
    base_executor = _cmnist_executor(plan, [])

    def finish_then_interrupt(
        staged_task: SearchRunTask,
        run_root: Path,
    ) -> CompletedStageRun:
        completed = base_executor(staged_task, run_root)
        (run_root / "result.json").write_text(
            completed.canonical_json() + "\n", encoding="utf-8"
        )
        raise RuntimeError("interrupted after durable completion")

    with pytest.raises(RuntimeError, match="durable completion"):
        scheduler.run_tasks((task,), finish_then_interrupt)

    calls: list[str] = []
    recovered = scheduler.run_tasks((task,), _cmnist_executor(plan, calls))
    assert len(recovered) == 1
    assert not calls


def test_bounded_scheduler_counts_only_new_runs_and_reuses_exact_results(
    tmp_path: Path,
) -> None:
    plan = build_search_plan(resolved_search_fixture())
    tasks = tuple(
        make_search_task(
            plan,
            plan.candidates[index],
            SeedStage.TUNING,
            plan.seeds.stages.tuning[0],
        )
        for index in range(3)
    )
    scheduler = LocalRunScheduler(tmp_path, plan)
    calls: list[str] = []
    first = scheduler.run_tasks((tasks[0],), _cmnist_executor(plan, calls))
    first_payload = (
        tmp_path / tasks[0].relative_directory / "result.json"
    ).read_bytes()

    available, new_count = scheduler.run_tasks_bounded(
        tasks[:2], _cmnist_executor(plan, calls), max_new_runs=1
    )
    assert available[0] == first[0]
    assert len(available) == 2
    assert new_count == 1
    assert calls == [tasks[0].task_id, tasks[1].task_id]
    repeated, repeated_new = scheduler.run_tasks_bounded(
        tasks[:2], _cmnist_executor(plan, calls), max_new_runs=1
    )
    assert repeated == available
    assert repeated_new == 0
    assert calls == [tasks[0].task_id, tasks[1].task_id]

    continued, continued_new = scheduler.run_tasks_bounded(
        tasks, _cmnist_executor(plan, calls), max_new_runs=2
    )
    assert len(continued) == 3
    assert continued_new == 1
    assert scheduler.run_tasks(tasks, _cmnist_executor(plan, calls)) == continued
    assert (
        tmp_path / tasks[0].relative_directory / "result.json"
    ).read_bytes() == first_payload


def test_bounded_scheduler_rejects_bad_prior_state_before_new_training(
    tmp_path: Path,
) -> None:
    plan = build_search_plan(resolved_search_fixture())
    tasks = tuple(
        make_search_task(
            plan,
            plan.candidates[index],
            SeedStage.TUNING,
            plan.seeds.stages.tuning[0],
        )
        for index in range(2)
    )
    scheduler = LocalRunScheduler(tmp_path, plan)
    _ = scheduler.run_tasks((tasks[1],), _cmnist_executor(plan, []))
    (tmp_path / tasks[1].relative_directory / "result.json").write_text(
        "{corrupt", encoding="utf-8"
    )
    calls: list[str] = []
    with pytest.raises(ValueError, match="corrupted"):
        scheduler.run_tasks_bounded(
            tasks, _cmnist_executor(plan, calls), max_new_runs=1
        )
    assert not calls


@pytest.mark.parametrize("dataset", ("cmnist", "waterbirds_cf"))
def test_tuning_filters_select_only_canonical_tasks_for_each_dataset(
    dataset: Literal["cmnist", "waterbirds_cf"],
) -> None:
    plan = build_search_plan(resolved_search_fixture(dataset))
    erm = next(item for item in plan.candidates if item.method_id == "erm")
    grit = next(
        item
        for item in plan.candidates
        if item.method_id == "grit" and item.requested_rank == 2
    )
    for candidate in (erm, grit):
        limits = validate_execution_limits(
            plan,
            ProductionExecutionLimits(
                stop_after="tuning",
                method=candidate.method_id,
                candidate_ids=(candidate.candidate_id,),
                tuning_seed=plan.seeds.stages.tuning[0],
                max_new_runs=1,
            ),
        )
        assert limits is not None
        assert limited_tuning_candidates(plan, limits) == (candidate,)


@pytest.mark.parametrize(
    ("limits", "message"),
    (
        (ProductionExecutionLimits(max_new_runs=0), "positive"),
        (ProductionExecutionLimits(max_new_runs=-1), "positive"),
        (
            ProductionExecutionLimits(max_new_runs=cast(int, 1.5)),
            "integer",
        ),
        (
            ProductionExecutionLimits(max_new_runs=cast(int, True)),
            "integer",
        ),
        (
            ProductionExecutionLimits(
                stop_after="tuning", tuning_seed=cast(int, 101.0)
            ),
            "integer",
        ),
        (
            ProductionExecutionLimits(
                stop_after="tuning", tuning_seed=cast(int, False)
            ),
            "integer",
        ),
        (
            ProductionExecutionLimits(
                stop_after="tuning", candidate_ids=("",)
            ),
            "nonempty strings",
        ),
        (
            ProductionExecutionLimits(
                stop_after="tuning", candidate_ids=(cast(str, 7),)
            ),
            "nonempty strings",
        ),
        (
            ProductionExecutionLimits(
                stop_after="tuning",
                candidate_ids=("candidate:duplicate", "candidate:duplicate"),
            ),
            "unique",
        ),
        (
            ProductionExecutionLimits(method="erm"),
            "require stop_after=tuning",
        ),
        (
            ProductionExecutionLimits(
                stop_after="tuning", candidate_ids=("candidate:unknown",)
            ),
            "not present",
        ),
        (
            ProductionExecutionLimits(stop_after="tuning", tuning_seed=999),
            "configured tuning seeds",
        ),
    ),
)
def test_invalid_execution_limits_fail_before_any_executor(
    limits: ProductionExecutionLimits,
    message: str,
) -> None:
    plan = build_search_plan(resolved_search_fixture())
    calls: list[str] = []
    with pytest.raises(ValueError, match=message):
        _ = validate_execution_limits(plan, limits)
    assert not calls


def test_execution_limits_accept_exact_integers_and_nonempty_candidate_ids() -> None:
    plan = build_search_plan(resolved_search_fixture())
    candidate = plan.candidates[0]
    limits = ProductionExecutionLimits(
        stop_after="tuning",
        candidate_ids=(candidate.candidate_id,),
        tuning_seed=plan.seeds.stages.tuning[0],
        max_new_runs=1,
    )
    assert validate_execution_limits(plan, limits) is limits


@pytest.mark.parametrize(
    "limits",
    (
        ProductionExecutionLimits(max_new_runs=cast(int, 1.0)),
        ProductionExecutionLimits(max_new_runs=cast(int, True)),
        ProductionExecutionLimits(
            stop_after="tuning", tuning_seed=cast(int, 101.0)
        ),
        ProductionExecutionLimits(
            stop_after="tuning", candidate_ids=(cast(str, object()),)
        ),
    ),
)
def test_programmatic_invalid_limits_fail_before_dataset_dispatch(
    monkeypatch: pytest.MonkeyPatch,
    limits: ProductionExecutionLimits,
) -> None:
    plan = build_search_plan(resolved_search_fixture())
    dispatch_calls: list[str] = []

    def fake_plan(_path: Path) -> SearchPlan:
        return plan

    def forbidden_dispatch(
        _plan: SearchPlan,
        _limits: ProductionExecutionLimits | None = None,
    ) -> None:
        dispatch_calls.append("dispatch")

    monkeypatch.setattr("grit.search.run.plan_production_search", fake_plan)
    monkeypatch.setattr("grit.search.cmnist.run_cmnist_search", forbidden_dispatch)
    with pytest.raises(ValueError, match="integer|nonempty strings"):
        _ = run_production_search(Path("unused.yaml"), limits)
    assert not dispatch_calls


def test_cmnist_real_plan_pilot_runs_two_canonical_tuning_tasks_only(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_path, plan = _write_stored_plan_fixture(tmp_path)
    output_root = Path(plan.resolved_config.output_root)
    before_candidate_inspection = _filesystem_snapshot(output_root)
    pilot = production_pilot_candidates(config_path)
    assert _filesystem_snapshot(output_root) == before_candidate_inspection
    assert pilot.tuning_seed == plan.seeds.stages.tuning[0]
    assert pilot.erm.method_id == "erm"
    assert pilot.grit_nonzero_rank.method_id == "grit"
    assert pilot.grit_nonzero_rank.requested_rank is not None
    assert pilot.grit_nonzero_rank.requested_rank > 0

    class _ManifestIdentity:
        def canonical_digest(self) -> str:
            return plan.resolved_config.lineage.feature_cache_manifest_digest

    features = torch.arange(4 * 512, dtype=torch.float32).reshape(4, 512)
    labels = torch.tensor((0, 1, 0, 1), dtype=torch.int64)

    def table(name: str) -> FeatureTable:
        return FeatureTable(
            name=name,
            role="pair_projection",
            source_ids=tuple(f"source:{index}" for index in range(4)),
            features=features + (1.0 if name.endswith("green") else 0.0),
            digits=labels,
            clean_labels=labels,
            targets=labels,
            colors=labels,
        )

    class _TuningOnlyCache:
        manifest = _ManifestIdentity()

        def pair_tables(self) -> tuple[FeatureTable, FeatureTable]:
            return table("oracle_pair_red"), table("oracle_pair_green")

    class _PairIdentity:
        def canonical_digest(self) -> str:
            return plan.resolved_config.lineage.pair_manifest_digest

    cache_loads: list[bool] = []

    def fake_cache_loader(
        _plan: SearchPlan, *, tuning_only: bool = False
    ) -> _TuningOnlyCache:
        cache_loads.append(tuning_only)
        return _TuningOnlyCache()

    train_calls: list[str] = []

    def fake_train(
        _cache: object,
        _runtime: object,
        task: SearchRunTask,
        _tracker: object = None,
    ) -> CmnistTrainedTask:
        train_calls.append(task.task_id)
        completed = _cmnist_executor(plan, [])(task, tmp_path)
        assert isinstance(completed, CmnistCompletedStageRun)
        run = cast(
            TrainedLinearProbeRun,
            SimpleNamespace(
                validation_metrics=completed.validation_metrics,
                algorithm=SimpleNamespace(diagnostic_history={}, method_diagnostics={}),
            ),
        )
        return CmnistTrainedTask(run)

    def fake_plan(_path: Path) -> SearchPlan:
        return plan

    def fake_pair_manifest(_plan: SearchPlan) -> _PairIdentity:
        return _PairIdentity()

    monkeypatch.setattr("grit.search.run.plan_production_search", fake_plan)
    monkeypatch.setattr("grit.search.cmnist._load_cmnist_cache", fake_cache_loader)
    monkeypatch.setattr(
        "grit.search.cmnist._cmnist_pair_manifest", fake_pair_manifest
    )
    monkeypatch.setattr("grit.search.cmnist._train_cmnist_task", fake_train)

    for candidate in (pilot.erm, pilot.grit_nonzero_rank):
        result = run_production_search(
            config_path,
            ProductionExecutionLimits(
                stop_after="tuning",
                method=candidate.method_id,
                candidate_ids=(candidate.candidate_id,),
                tuning_seed=pilot.tuning_seed,
                max_new_runs=1,
            ),
        )
        assert isinstance(result, ProductionSearchStatus)
        assert result.phase == "tuning"

    assert len(train_calls) == 2
    assert cache_loads == [True, True]
    projection_path = (
        Path(plan.resolved_config.output_root)
        / "projections"
        / f"grit-rank-{pilot.grit_nonzero_rank.requested_rank}.json"
    )
    assert projection_path.is_file()

    repeated = run_production_search(
        config_path,
        ProductionExecutionLimits(
            stop_after="tuning",
            method="grit",
            candidate_ids=(pilot.grit_nonzero_rank.candidate_id,),
            tuning_seed=pilot.tuning_seed,
            max_new_runs=1,
        ),
    )
    assert isinstance(repeated, ProductionSearchStatus)
    assert len(train_calls) == 2
    assert not (output_root / "selection").exists()
    assert not (output_root / "summaries").exists()
    assert not (output_root / "experiment-index.json").exists()
    assert not tuple(output_root.rglob("final-result.json"))


def test_waterbirds_real_plan_pilot_uses_same_bounded_lifecycle(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    weights = _waterbirds_weight_spec()
    config_path, plan = _write_stored_plan_fixture(
        tmp_path,
        "waterbirds_cf",
        adjusted_weight_spec_digest=weights.canonical_digest(),
    )
    pilot = production_pilot_candidates(config_path)

    class _ManifestIdentity:
        def canonical_digest(self) -> str:
            return plan.resolved_config.lineage.feature_cache_manifest_digest

    class _TuningOnlyCache:
        manifest = _ManifestIdentity()

    class _PairIdentity:
        def canonical_digest(self) -> str:
            return plan.resolved_config.lineage.pair_manifest_digest

    pair_identity = _PairIdentity()
    projection_calls: list[int] = []
    train_calls: list[str] = []

    def fake_plan(_path: Path) -> SearchPlan:
        return plan

    def fake_cache(
        _plan: SearchPlan, *, tuning_only: bool = False
    ) -> _TuningOnlyCache:
        assert tuning_only is True
        return _TuningOnlyCache()

    def fake_projection(
        _cache: object,
        _pairs: object,
        *,
        requested_rank: int,
        relative_singular_value_tolerance: float,
    ) -> FittedLinearProjection:
        projection_calls.append(requested_rank)
        return fit_linear_projection(
            torch.eye(4, 512, dtype=torch.float64),
            torch.zeros((4, 512), dtype=torch.float64),
            requested_rank=requested_rank,
            pair_manifest_digest=plan.resolved_config.lineage.pair_manifest_digest,
            feature_cache_manifest_digest=(
                plan.resolved_config.lineage.feature_cache_manifest_digest
            ),
            relative_singular_value_tolerance=(
                relative_singular_value_tolerance
            ),
        )

    def fake_train(
        _cache: object,
        _weights: object,
        _runtime: object,
        task: SearchRunTask,
        _tracker: object = None,
    ) -> SimpleNamespace:
        train_calls.append(task.task_id)
        completed = _waterbirds_stage_run(task, weights)
        return SimpleNamespace(
            validation_metrics=completed.validation_metrics,
            algorithm=SimpleNamespace(diagnostic_history={}, method_diagnostics={}),
            diagnostic_metrics=None,
            selector_records=lambda _selector: completed.validation_metrics,  # pyright: ignore[reportUnknownLambdaType]
        )

    def fake_dataset(_plan: SearchPlan) -> object:
        return object()

    def fake_pair(_plan: SearchPlan) -> _PairIdentity:
        return pair_identity

    def fake_weights(_dataset: object) -> WaterbirdsAdjustedWeightSpec:
        return weights

    monkeypatch.setattr("grit.search.run.plan_production_search", fake_plan)
    monkeypatch.setattr(
        "grit.search.waterbirds._dataset_manifest", fake_dataset
    )
    monkeypatch.setattr(
        "grit.search.waterbirds._pair_manifest", fake_pair
    )
    monkeypatch.setattr("grit.search.waterbirds._load_cache", fake_cache)
    monkeypatch.setattr(
        "grit.search.waterbirds.mint_waterbirds_adjusted_weight_spec",
        fake_weights,
    )
    monkeypatch.setattr(
        "grit.search.waterbirds.fit_waterbirds_oracle_projection",
        fake_projection,
    )
    monkeypatch.setattr("grit.search.waterbirds._train_task", fake_train)

    for candidate in (pilot.erm, pilot.grit_nonzero_rank):
        result = run_production_search(
            config_path,
            ProductionExecutionLimits(
                stop_after="tuning",
                method=candidate.method_id,
                candidate_ids=(candidate.candidate_id,),
                tuning_seed=pilot.tuning_seed,
                max_new_runs=1,
            ),
        )
        assert isinstance(result, ProductionSearchStatus)
        assert result.phase == "tuning"

    assert len(train_calls) == 2
    assert projection_calls == [pilot.grit_nonzero_rank.requested_rank]
    output_root = Path(plan.resolved_config.output_root)
    assert not (output_root / "selection").exists()
    assert not (output_root / "summaries").exists()
    assert not (output_root / "experiment-index.json").exists()


def test_cli_run_passes_no_limits_for_unrestricted_run(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    script = _run_search_script()
    main = cast(Callable[[Sequence[str] | None], int], script.main)

    observed: list[ProductionExecutionLimits | None] = []

    class _Summary:
        schema_version = "summary/v1"
        plan_digest = "sha256:plan"

    def fake_run(
        _path: Path,
        limits: ProductionExecutionLimits | None = None,
    ) -> _Summary:
        observed.append(limits)
        return _Summary()

    monkeypatch.setattr(script, "plan_production_search", _fake_plan)
    monkeypatch.setattr(script, "run_production_search", fake_run)
    assert main(("config.yaml",)) == 0
    assert observed == [None]



@pytest.mark.parametrize("urbancars", [False, True])
def test_cli_forwards_execution_options(
    monkeypatch: pytest.MonkeyPatch, urbancars: bool
) -> None:
    script = _run_search_script()
    main = cast(Callable[[Sequence[str] | None], int], script.main)
    observed: list[object] = []

    class Summary:
        schema_version = "test"
        plan_digest = "test"

        def canonical_json(self) -> str:
            return "{}"

    def is_dataset(_path: Path, dataset: str) -> bool:
        return urbancars and dataset == "urbancars"

    def run(_path: Path, *_args: object, **kwargs: object) -> Summary:
        observed.append(kwargs["execution"])
        return Summary()

    def run_urbancars(_path: Path, **kwargs: object) -> dict[str, object]:
        observed.append(kwargs["execution"])
        return {}

    monkeypatch.setattr(script, "is_dataset_search", is_dataset)
    monkeypatch.setattr(script, "plan_production_search", _fake_plan)
    monkeypatch.setattr(script, "run_production_search", run)
    monkeypatch.setattr("grit.search.urbancars.run_urbancars", run_urbancars)
    assert main(("config.yaml", "--workers", "3", "--threads-per-worker", "2")) == 0
    assert observed == [ExecutionOptions(3, 2)]


@pytest.mark.parametrize("flag", ["--workers", "--threads-per-worker"])
def test_cli_rejects_bad_execution_before_planning(flag: str) -> None:
    main = cast(Callable[[Sequence[str] | None], int], _run_search_script().main)
    with pytest.raises(SystemExit) as error:
        main(("does-not-exist.yaml", flag, "0"))
    assert error.value.code == 2


def _run_search_script() -> ModuleType:
    """Load scripts/run_search.py; scripts are entry points, not a package."""

    path = Path(__file__).resolve().parents[1] / "scripts" / "run_search.py"
    spec = importlib.util.spec_from_file_location("run_search", path)
    if spec is None or spec.loader is None:
        raise AssertionError("could not load scripts/run_search.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _FakePlan:
    dataset = "cmnist"
    candidates = ()

    class resolved_config:
        output_root = "/tmp/out"


def _fake_plan(_path: Path) -> _FakePlan:
    return _FakePlan()


def test_cli_run_constructs_bounded_tuning_controls(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    script = _run_search_script()
    main = cast(Callable[[Sequence[str] | None], int], script.main)

    observed: list[ProductionExecutionLimits | None] = []

    def fake_run(
        _path: Path,
        limits: ProductionExecutionLimits | None = None,
    ) -> ProductionSearchStatus:
        observed.append(limits)
        return ProductionSearchStatus(
            schema_version="grit.production-search-status/v1",
            dataset="cmnist",
            plan_digest="sha256:plan",
            phase="tuning",
            tuning_expected=1152,
            tuning_complete=2,
            confirmation_expected=0,
            confirmation_complete=0,
            frozen_winner_count=0,
            final_expected=0,
            final_complete=0,
        )

    monkeypatch.setattr(script, "plan_production_search", _fake_plan)
    monkeypatch.setattr(script, "run_production_search", fake_run)
    assert (
        main(
            (
                "config.yaml",
                "--candidate-id",
                "candidate:erm",
                "--candidate-id",
                "candidate:grit",
                "--seed",
                "101",
                "--limit",
                "2",
            )
        )
        == 0
    )
    assert observed == [
        ProductionExecutionLimits(
            stop_after="tuning",
            method="all",
            candidate_ids=("candidate:erm", "candidate:grit"),
            tuning_seed=101,
            max_new_runs=2,
        )
    ]


def test_cli_pilot_selects_erm_and_grit_at_first_seed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    script = _run_search_script()
    main = cast(Callable[[Sequence[str] | None], int], script.main)

    config_path, _ = write_cmnist_production_config(
        tmp_path, output_root=tmp_path / "output"
    )
    monkeypatch.setattr(
        "grit.search.plan._code_provenance", lambda: _CLEAN_CODE_PROVENANCE
    )
    plan = plan_production_search(config_path)
    observed: list[ProductionExecutionLimits | None] = []

    def fake_run(
        _path: Path,
        limits: ProductionExecutionLimits | None = None,
    ) -> ProductionSearchStatus:
        observed.append(limits)
        return _cmnist_status_stub(plan)

    monkeypatch.setattr(script, "run_production_search", fake_run)
    assert main((str(config_path), "--pilot")) == 0
    (limits,) = observed
    assert limits is not None
    assert limits.stop_after == "tuning"
    assert limits.max_new_runs == 2
    assert limits.tuning_seed == plan.seeds.stages.tuning[0]
    by_id = {candidate.candidate_id: candidate for candidate in plan.candidates}
    erm, grit = (by_id[candidate_id] for candidate_id in limits.candidate_ids)
    assert erm.method_id == "erm"
    assert grit.method_id == "grit" and (grit.requested_rank or 0) > 0


def test_cli_dry_run_plans_and_reports_without_training(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    script = _run_search_script()
    main = cast(Callable[[Sequence[str] | None], int], script.main)

    output_root = tmp_path / "output"
    config_path, _ = write_cmnist_production_config(tmp_path, output_root=output_root)
    monkeypatch.setattr(
        "grit.search.plan._code_provenance", lambda: _CLEAN_CODE_PROVENANCE
    )

    def forbidden(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("dry run reached training")

    monkeypatch.setattr(script, "run_production_search", forbidden)
    assert main((str(config_path), "--dry-run")) == 0
    assert (output_root / "search-plan.json").is_file()
    assert not (output_root / "runs").exists()


def _cmnist_status_stub(plan: SearchPlan) -> ProductionSearchStatus:
    return ProductionSearchStatus(
        schema_version="grit.production-search-status/v1",
        dataset="cmnist",
        plan_digest=plan.canonical_digest(),
        phase="tuning",
        tuning_expected=plan.expected_run_counts.tuning,
        tuning_complete=0,
        confirmation_expected=0,
        confirmation_complete=0,
        frozen_winner_count=0,
        final_expected=0,
        final_complete=0,
    )


def test_custom_grid_epochs_and_pair_count_plan_from_yaml(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The YAML is the grid; nothing in the code pins the approved values."""

    monkeypatch.setattr(
        "grit.search.plan._code_provenance", lambda: _CLEAN_CODE_PROVENANCE
    )
    config_path, _ = write_cmnist_production_config(
        tmp_path,
        output_root=tmp_path / "output",
        overrides={
            "search_space": {
                "methods": ["erm", "grit"],
                "learning_rates": [0.001, 0.01, 0.1],
                "weight_decays": [0.0],
                "projection_ranks": [0, 2, 4],
            },
            "max_epochs": 5,
            "pair_count": 64,
            "relative_singular_value_tolerance": 1e-10,
        },
    )
    plan = plan_production_search(config_path)
    assert len(plan.candidates) == 3 + 3 * 3
    assert {c.requested_rank for c in plan.candidates if c.method_id == "grit"} == {
        0,
        2,
        4,
    }
    assert plan.resolved_config.config.max_epochs == 5
    assert plan.resolved_config.config.pair_count == 64
    assert plan.expected_run_counts.tuning == 12 * 3


def test_cmnist_groupdro_only_grid_plans_and_materializes(tmp_path: Path) -> None:
    config_path, _ = write_cmnist_production_config(
        tmp_path,
        output_root=tmp_path / "groupdro-output",
        overrides={
            "experiment_name": "cmnist-groupdro",
            "search_space": {
                "methods": ["groupdro"],
                "learning_rates": [0.0001, 0.0003, 0.001, 0.003],
                "weight_decays": [0.0, 0.00001, 0.0001, 0.001],
                "groupdro_step_sizes": list(APPROVED_GROUPDRO_STEP_SIZES),
            },
        },
    )
    plan = plan_production_search(config_path)
    assert plan.methods == ("groupdro",)
    assert len(plan.candidates) == 48
    assert plan.expected_run_counts.tuning == 144
    assert plan.expected_run_counts.final == 20
    assert "paired_summary" not in {
        item.artifact_kind for item in plan.output_schemas
    }
    assert "projection_diagnostics" not in {
        item.artifact_kind for item in plan.output_schemas
    }
    assert {item.groupdro_step_size for item in plan.candidates} == set(
        APPROVED_GROUPDRO_STEP_SIZES
    )
    assert {item.requested_rank for item in plan.candidates} == {None}
    resolved = materialize_cmnist_candidate_config(
        plan, plan.candidates[0], CmnistSelector.PRIMARY_ROBUST
    )
    assert resolved.algorithm.kind == "groupdro"
    assert resolved.artifact_lineage is not None
    assert resolved.artifact_lineage.pair_manifest_digest is None
    assert resolved.scientific_config_digest() == (
        plan.candidates[0].scientific_config_digest
    )
    assert pilot_candidates(plan) == (plan.candidates[0],)


def test_cmnist_rex_only_grid_plans_materializes_and_selects_pilot(
    tmp_path: Path,
) -> None:
    config_path, _ = write_cmnist_production_config(
        tmp_path,
        output_root=tmp_path / "rex-output",
        overrides={
            "experiment_name": "cmnist-rex",
            "search_space": {
                "methods": ["rex"],
                "learning_rates": [0.0001, 0.0003, 0.001, 0.003],
                "weight_decays": [0.0, 0.00001, 0.0001, 0.001],
                "rex_penalty_weights": list(APPROVED_REX_PENALTY_WEIGHTS),
                "rex_penalty_anneal_updates": REX_PENALTY_ANNEAL_UPDATES,
            },
        },
    )
    plan = plan_production_search(config_path)
    assert plan.methods == ("rex",)
    assert len(plan.candidates) == 64
    assert plan.expected_run_counts.tuning == 192
    assert plan.expected_run_counts.final == 20
    assert {item.penalty_weight for item in plan.candidates} == set(
        APPROVED_REX_PENALTY_WEIGHTS
    )
    assert {item.requested_rank for item in plan.candidates} == {None}
    resolved = materialize_cmnist_candidate_config(
        plan, plan.candidates[0], CmnistSelector.PRIMARY_ROBUST
    )
    assert resolved.algorithm.kind == "rex"
    assert (
        resolved.algorithm.penalty_anneal_updates
        == REX_PENALTY_ANNEAL_UPDATES
    )
    assert resolved.artifact_lineage is not None
    assert resolved.artifact_lineage.pair_manifest_digest is None
    assert resolved.scientific_config_digest() == (
        plan.candidates[0].scientific_config_digest
    )
    assert pilot_candidates(plan) == (plan.candidates[0],)


def test_cmnist_irm_only_grid_plans_materializes_and_selects_pilot(
    tmp_path: Path,
) -> None:
    config_path, _ = write_cmnist_production_config(
        tmp_path,
        output_root=tmp_path / "irm-output",
        overrides={
            "experiment_name": "cmnist-irm",
            "search_space": {
                "methods": ["irm"],
                "learning_rates": [0.0001, 0.0003, 0.001, 0.003],
                "weight_decays": [0.0, 0.00001, 0.0001, 0.001],
                "irm_penalty_weights": list(APPROVED_IRM_PENALTY_WEIGHTS),
                "irm_penalty_anneal_updates": IRM_PENALTY_ANNEAL_UPDATES,
            },
        },
    )
    plan = plan_production_search(config_path)
    assert plan.methods == ("irm",)
    assert len(plan.candidates) == 64
    assert plan.expected_run_counts.tuning == 192
    assert plan.expected_run_counts.final == 20
    assert {item.penalty_weight for item in plan.candidates} == set(
        APPROVED_IRM_PENALTY_WEIGHTS
    )
    assert {item.requested_rank for item in plan.candidates} == {None}
    resolved = materialize_cmnist_candidate_config(
        plan, plan.candidates[0], CmnistSelector.PRIMARY_ROBUST
    )
    assert resolved.algorithm.kind == "irm"
    assert (
        resolved.algorithm.penalty_anneal_updates
        == IRM_PENALTY_ANNEAL_UPDATES
    )
    assert resolved.artifact_lineage is not None
    assert resolved.artifact_lineage.pair_manifest_digest is None
    assert resolved.scientific_config_digest() == (
        plan.candidates[0].scientific_config_digest
    )
    assert pilot_candidates(plan) == (plan.candidates[0],)


@pytest.mark.parametrize(
    ("method", "method_space"),
    (
        (
            "rex",
            {
                "rex_penalty_weights": list(APPROVED_REX_PENALTY_WEIGHTS),
                "rex_penalty_anneal_updates": REX_PENALTY_ANNEAL_UPDATES,
            },
        ),
        (
            "irm",
            {
                "irm_penalty_weights": list(APPROVED_IRM_PENALTY_WEIGHTS),
                "irm_penalty_anneal_updates": IRM_PENALTY_ANNEAL_UPDATES,
            },
        ),
    ),
)
def test_cmnist_invariant_method_pilot_runs_one_real_training_task(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    method: Literal["rex", "irm"],
    method_space: dict[str, object],
) -> None:
    config_path, _ = write_cmnist_production_config(
        tmp_path / method,
        output_root=tmp_path / f"{method}-output",
        overrides={
            "experiment_name": f"cmnist-{method}-pilot",
            "search_space": {
                "methods": [method],
                "learning_rates": [0.001],
                "weight_decays": [0.0, 0.0001, 0.001],
                **method_space,
            },
            "max_epochs": 1,
        },
    )
    plan = plan_production_search(config_path)
    candidate = pilot_candidates(plan)[0]

    def table(
        name: str,
        role: Literal["training", "validation"],
        seed: int,
    ) -> FeatureTable:
        features = torch.randn(
            (8, 512), generator=torch.Generator(device="cpu").manual_seed(seed)
        )
        targets = (features[:, 0] > 0).to(torch.int64)
        return FeatureTable(
            name=name,
            role=role,
            source_ids=tuple(f"source:{name}:{index}" for index in range(8)),
            features=features,
            digits=targets * 5,
            clean_labels=targets,
            targets=targets,
            colors=targets,
        )

    class _PilotCache:
        def training_tables(self) -> tuple[FeatureTable, FeatureTable]:
            return (
                table("train_e01", "training", 1),
                table("train_e02", "training", 2),
            )

        def validation_tables(
            self,
        ) -> tuple[FeatureTable, FeatureTable, FeatureTable]:
            return (
                table("val_e01", "validation", 3),
                table("val_e02", "validation", 4),
                table("val_e05", "validation", 5),
            )

    def fake_cache_loader(
        _plan: SearchPlan, *, tuning_only: bool = False
    ) -> _PilotCache:
        return _PilotCache()

    monkeypatch.setattr("grit.search.cmnist._load_cmnist_cache", fake_cache_loader)
    result = run_production_search(
        config_path,
        ProductionExecutionLimits(
            stop_after="tuning",
            method=method,
            candidate_ids=(candidate.candidate_id,),
            tuning_seed=plan.seeds.stages.tuning[0],
            max_new_runs=1,
        ),
    )
    assert isinstance(result, ProductionSearchStatus)
    assert result.tuning_complete == 1


def test_pair_count_above_prepared_bank_is_rejected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "grit.search.plan._code_provenance", lambda: _CLEAN_CODE_PROVENANCE
    )
    config_path, _ = write_cmnist_production_config(
        tmp_path, output_root=tmp_path / "output", overrides={"pair_count": 512}
    )
    with pytest.raises(ValueError, match="prepared bank holds 256"):
        plan_production_search(config_path)


@pytest.mark.parametrize(
    ("overrides", "message"),
    (
        (
            {
                "search_space": {
                    "methods": ["erm", "grit"],
                    "learning_rates": [0.001, 0.01],
                    "weight_decays": [0.0, 0.001],
                    "projection_ranks": [0, 300],
                },
            },
            "exceed min\\(pair_count, feature_dim\\) = 256",
        ),
        (
            {
                "search_space": {
                    "methods": ["erm", "grit"],
                    "learning_rates": [0.001, 0.01],
                    "weight_decays": [0.0],
                    "projection_ranks": [0, 2, 4],
                },
            },
            "grid yields 2 ERM",
        ),
    ),
)
def test_unrunnable_grids_fail_at_plan_time(
    tmp_path: Path,
    overrides: dict[str, object],
    message: str,
) -> None:
    config_path, _ = write_cmnist_production_config(
        tmp_path, output_root=tmp_path / "output", overrides=overrides
    )
    with pytest.raises(ValueError, match=message):
        plan_production_search(config_path)
    assert not (tmp_path / "output").exists()
