"""Small typed constructors shared by the contract-spine tests."""

from __future__ import annotations

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
)
from grit.schemas import CmnistSelector, SeedStage
from grit.selection.cmnist import ValidationMetricRecord


def seed_sets() -> SeedSets:
    return SeedSets(
        tuning=(101, 102, 103),
        confirmation=(201, 202),
        final=(301, 302, 303, 304, 305, 306, 307, 308, 309, 310),
    )


def dataset_config() -> CmnistDatasetConfig:
    return CmnistDatasetConfig(
        dataset_id="cmnist",
        construction_method_id="cmnist-stratified-hash-v1",
        construction_seed=0,
        label_flip_prob=0.25,
        source_counts=CmnistSourceCounts(
            train_e01=25_000,
            train_e02=25_000,
            validation=10_000,
            test=10_000,
        ),
        training_split_names=("train_e01", "train_e02"),
        validation_split_names=("val_e01", "val_e02", "val_e05"),
        final_test_split_name="test_ood",
    )


def feature_config() -> FrozenFeatureConfig:
    return FrozenFeatureConfig(
        kind="frozen_features",
        encoder_id="openai-clip-vit-b32",
        encoder_revision="d05afc436d78f1c48dc0dbf8e5980a9d471f35f6",
        weights_identity=(
            "sha256:40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af"
        ),
        preprocessing_identity=(
            "openai-clip-vit-b32-preprocess@"
            "d05afc436d78f1c48dc0dbf8e5980a9d471f35f6"
        ),
        feature_dimension=512,
        normalization="none",
    )


def training_config() -> LinearProbeTrainingConfig:
    return LinearProbeTrainingConfig(
        optimizer="adam",
        batch_size=256,
        learning_rate=0.001,
        weight_decay=0.0,
        max_epochs=40,
    )


def runtime_config() -> CpuRuntimeConfig:
    return CpuRuntimeConfig(device="cpu", deterministic_algorithms=True)


def ordinary_erm_config(
    selector: CmnistSelector = CmnistSelector.PRIMARY_ROBUST,
) -> OrdinaryExperimentConfig:
    return OrdinaryExperimentConfig(
        schema_version="grit.experiment/v1",
        run_kind="ordinary",
        experiment_name="synthetic-cmnist-erm",
        protocol_id="cmnist/v1",
        reportable=False,
        dataset=dataset_config(),
        representation=feature_config(),
        pairs=DisabledPairsConfig(kind="disabled"),
        projection=DisabledProjectionConfig(kind="disabled"),
        algorithm=ErmAlgorithmConfig(kind="erm"),
        training=training_config(),
        runtime=runtime_config(),
        seed_sets=seed_sets(),
        selection=OrdinarySelectionConfig(selector=selector),
    )


def ordinary_grit_config(
    selector: CmnistSelector = CmnistSelector.PRIMARY_ROBUST,
) -> OrdinaryExperimentConfig:
    return OrdinaryExperimentConfig(
        schema_version="grit.experiment/v1",
        run_kind="ordinary",
        experiment_name="synthetic-cmnist-oracle-grit",
        protocol_id="cmnist/v1",
        reportable=False,
        dataset=dataset_config(),
        representation=feature_config(),
        pairs=OraclePairsConfig(
            kind="oracle",
            construction_id="cmnist-clean-oracle-pairs-v1",
            source_partition_ids=("train_e01_sources", "train_e02_sources"),
            pair_count=256,
            pair_seed=0,
            orientation="red_minus_green",
        ),
        projection=LinearProjectionConfig(
            kind="linear_pair_difference",
            requested_rank=2,
            center_differences=False,
            relative_singular_value_tolerance=1e-12,
        ),
        algorithm=GritAlgorithmConfig(kind="grit"),
        training=training_config(),
        runtime=runtime_config(),
        seed_sets=seed_sets(),
        selection=OrdinarySelectionConfig(selector=selector),
    )


def validation_records(
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
    split_names = ("val_e01", "val_e02", "val_e05")
    return tuple(
        ValidationMetricRecord(
            record_id=f"metric:{run_id}:{checkpoint_id}:{split_name}",
            run_id=run_id,
            candidate_id=candidate_id,
            method_id=method_id,
            scientific_config_digest=scientific_config_digest,
            checkpoint_id=checkpoint_id,
            epoch=epoch,
            seed=seed,
            value=value,
            sample_count=10,
            metric_kind="validation",
            seed_stage=seed_stage,
            split_name=split_name,
            metric_name="accuracy",
            projection_rank=projection_rank,
        )
        for split_name, value in zip(split_names, scores, strict=True)
    )
