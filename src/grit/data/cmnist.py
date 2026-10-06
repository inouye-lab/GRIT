"""Deterministic ColoredMNIST construction for the first rewrite vertical slice."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Annotated, Literal, Protocol, TypeAlias, cast

import numpy as np
import torch
from numpy.typing import NDArray
from pydantic import Field, FiniteFloat, StrictInt, StrictStr, model_validator

from grit.data.views import (
    ExampleIdentity,
    TrainingSplitDescriptor,
    TrainingView,
    ValidationSplitDescriptor,
    ValidationView,
    validate_cmnist_repeated_validation_views,
)
from grit.schemas import (
    HELD_OUT_VALIDATION_NAMES,
    StrictBoundaryModel,
    canonical_digest_value,
    held_out_validation_flip_prob,
    held_out_validation_name,
)

PARTITION_METHOD_ID = "cmnist-stratified-hash-v1"
RENDERING_METHOD_ID = "cmnist-rgb-render-v1"
ORACLE_PAIR_METHOD_ID = "cmnist-clean-oracle-pairs-v1"

NonEmptyStr: TypeAlias = Annotated[StrictStr, Field(min_length=1)]
NonNegativeInt: TypeAlias = Annotated[StrictInt, Field(ge=0)]
Probability: TypeAlias = Annotated[FiniteFloat, Field(ge=0.0, le=1.0)]
BinaryInt: TypeAlias = Annotated[StrictInt, Field(ge=0, le=1)]
SourcePartitionName: TypeAlias = Literal[
    "train_e01_sources",
    "train_e02_sources",
    "validation_sources",
    "test_sources",
]
EnvironmentName: TypeAlias = Literal[
    "train_e01",
    "train_e02",
    "val_e01",
    "val_e02",
    "val_e03",
    "val_e04",
    "val_e05",
    "val_e06",
    "val_e07",
    "test_ood",
]


class CmnistPartitionTargets(StrictBoundaryModel):
    """Explicit source counts for production or non-reportable smoke construction."""

    train_e01: NonNegativeInt
    train_e02: NonNegativeInt
    validation: NonNegativeInt
    test: NonNegativeInt

    @model_validator(mode="after")
    def _require_nonempty_partitions(self) -> CmnistPartitionTargets:
        if min(self.train_e01, self.train_e02, self.validation, self.test) <= 0:
            raise ValueError("every CMNIST source partition must be non-empty")
        return self

    @property
    def official_train_count(self) -> int:
        return self.train_e01 + self.train_e02 + self.validation


PRODUCTION_PARTITION_TARGETS = CmnistPartitionTargets(
    train_e01=25_000,
    train_e02=25_000,
    validation=10_000,
    test=10_000,
)


class DigitCount(StrictBoundaryModel):
    digit: Annotated[StrictInt, Field(ge=0, le=9)]
    count: NonNegativeInt


class SourcePartitionRecord(StrictBoundaryModel):
    name: SourcePartitionName
    official_split: Literal["train", "test"]
    source_indices: tuple[NonNegativeInt, ...]
    digit_counts: tuple[DigitCount, ...]
    membership_digest: NonEmptyStr

    @model_validator(mode="after")
    def _validate_membership(self) -> SourcePartitionRecord:
        if len(self.source_indices) != len(set(self.source_indices)):
            raise ValueError("source partition indices must be unique")
        if tuple(item.digit for item in self.digit_counts) != tuple(range(10)):
            raise ValueError(
                "source partition digit counts must be ordered 0 through 9"
            )
        if sum(item.count for item in self.digit_counts) != len(self.source_indices):
            raise ValueError("digit counts must sum to source partition size")
        expected = canonical_digest_value(
            {
                "name": self.name,
                "official_split": self.official_split,
                "source_indices": self.source_indices,
            }
        )
        if self.membership_digest != expected:
            raise ValueError("source partition membership digest is inconsistent")
        return self


class CmnistPartitionManifest(StrictBoundaryModel):
    schema_version: Literal["grit.cmnist-partitions/v1"]
    dataset_id: Literal["cmnist"]
    construction_method_id: Literal["cmnist-stratified-hash-v1"]
    construction_seed: StrictInt
    targets: CmnistPartitionTargets
    partitions: tuple[SourcePartitionRecord, ...]

    @model_validator(mode="after")
    def _validate_partitions(self) -> CmnistPartitionManifest:
        expected_names = (
            "train_e01_sources",
            "train_e02_sources",
            "validation_sources",
            "test_sources",
        )
        if tuple(partition.name for partition in self.partitions) != expected_names:
            raise ValueError("CMNIST partition records must use canonical ordering")
        train_sets = [
            set(partition.source_indices) for partition in self.partitions[:3]
        ]
        if any(
            train_sets[left] & train_sets[right]
            for left in range(3)
            for right in range(left + 1, 3)
        ):
            raise ValueError("official-train CMNIST source partitions must be disjoint")
        expected_counts = (
            self.targets.train_e01,
            self.targets.train_e02,
            self.targets.validation,
            self.targets.test,
        )
        observed_counts = tuple(
            len(partition.source_indices) for partition in self.partitions
        )
        if observed_counts != expected_counts:
            raise ValueError("CMNIST source partition counts do not match targets")
        if any(
            partition.official_split != "train" for partition in self.partitions[:3]
        ) or self.partitions[3].official_split != "test":
            raise ValueError("CMNIST partitions use an incorrect official MNIST split")
        return self


@dataclass(frozen=True, slots=True)
class MnistPool:
    """Injected MNIST-like pool; torchvision is kept behind a separate adapter."""

    official_split: Literal["train", "test"]
    source_indices: torch.Tensor
    images: torch.Tensor
    digits: torch.Tensor


@dataclass(frozen=True, slots=True)
class CmnistPartitions:
    train_e01_source_indices: tuple[int, ...]
    train_e02_source_indices: tuple[int, ...]
    validation_source_indices: tuple[int, ...]
    test_source_indices: tuple[int, ...]
    manifest: CmnistPartitionManifest


class CmnistEnvironmentSpec(StrictBoundaryModel):
    name: EnvironmentName
    role: Literal["training", "validation", "final_test"]
    source_partition_id: SourcePartitionName
    color_flip_prob: Probability


def cmnist_environment_specs(
    held_out_flip_prob: float = 0.5,
) -> tuple[CmnistEnvironmentSpec, ...]:
    """The protocol environments; only the held-out validation rate is a choice."""

    held_out = held_out_validation_name(held_out_flip_prob)
    return (
        CmnistEnvironmentSpec(
            name="train_e01",
            role="training",
            source_partition_id="train_e01_sources",
            color_flip_prob=0.1,
        ),
        CmnistEnvironmentSpec(
            name="train_e02",
            role="training",
            source_partition_id="train_e02_sources",
            color_flip_prob=0.2,
        ),
        CmnistEnvironmentSpec(
            name="val_e01",
            role="validation",
            source_partition_id="validation_sources",
            color_flip_prob=0.1,
        ),
        CmnistEnvironmentSpec(
            name="val_e02",
            role="validation",
            source_partition_id="validation_sources",
            color_flip_prob=0.2,
        ),
        CmnistEnvironmentSpec(
            name=held_out,
            role="validation",
            source_partition_id="validation_sources",
            color_flip_prob=held_out_validation_flip_prob(held_out),
        ),
        CmnistEnvironmentSpec(
            name="test_ood",
            role="final_test",
            source_partition_id="test_sources",
            color_flip_prob=0.9,
        ),
    )


CMNIST_ENVIRONMENT_SPECS: tuple[CmnistEnvironmentSpec, ...] = (
    cmnist_environment_specs(0.5)
)
HELD_OUT_ENVIRONMENT_INDEX = 4


class EnvironmentManifest(StrictBoundaryModel):
    name: EnvironmentName
    role: Literal["training", "validation", "final_test"]
    source_partition_id: SourcePartitionName
    color_flip_prob: Probability
    count: NonNegativeInt
    source_membership_digest: NonEmptyStr
    record_digest: NonEmptyStr


class CmnistDatasetManifest(StrictBoundaryModel):
    schema_version: Literal["grit.cmnist-dataset/v1"]
    dataset_id: Literal["cmnist"]
    partition_manifest: CmnistPartitionManifest
    partition_manifest_digest: NonEmptyStr
    official_train_pool_digest: NonEmptyStr
    official_test_pool_digest: NonEmptyStr
    rendering_method_id: Literal["cmnist-rgb-render-v1"]
    construction_seed: StrictInt
    label_flip_prob: Probability
    environments: tuple[EnvironmentManifest, ...]

    @model_validator(mode="after")
    def _validate_identity(self) -> CmnistDatasetManifest:
        if self.partition_manifest_digest != self.partition_manifest.canonical_digest():
            raise ValueError("partition manifest digest is inconsistent")
        if self.construction_seed != self.partition_manifest.construction_seed:
            raise ValueError("dataset and partition construction seeds must match")
        names = tuple(item.name for item in self.environments)
        if (
            len(names) != len(CMNIST_ENVIRONMENT_SPECS)
            or names[HELD_OUT_ENVIRONMENT_INDEX] not in HELD_OUT_VALIDATION_NAMES
        ):
            raise ValueError(
                "dataset environments must use canonical protocol ordering"
            )
        expected_names = tuple(
            spec.name
            for spec in cmnist_environment_specs(
                held_out_validation_flip_prob(names[HELD_OUT_ENVIRONMENT_INDEX])
            )
        )
        if names != expected_names:
            raise ValueError(
                "dataset environments must use canonical protocol ordering"
            )
        return self

    @property
    def held_out_validation_name(self) -> str:
        return self.environments[HELD_OUT_ENVIRONMENT_INDEX].name

    def environment_specs(self) -> tuple[CmnistEnvironmentSpec, ...]:
        return cmnist_environment_specs(
            float(self.environments[HELD_OUT_ENVIRONMENT_INDEX].color_flip_prob)
        )


@dataclass(frozen=True, slots=True)
class RenderedCmnistTable:
    name: EnvironmentName
    role: Literal["training", "validation", "final_test"]
    source_partition_id: SourcePartitionName
    source_ids: tuple[str, ...]
    example_ids: tuple[str, ...]
    images: torch.Tensor
    digits: torch.Tensor
    clean_labels: torch.Tensor
    targets: torch.Tensor
    colors: torch.Tensor
    color_flip_prob: float

    def identities(self) -> tuple[ExampleIdentity, ...]:
        return tuple(
            ExampleIdentity(
                example_id=example_id,
                source_id=source_id,
                view_id=self.name,
            )
            for example_id, source_id in zip(
                self.example_ids, self.source_ids, strict=True
            )
        )


@dataclass(frozen=True, slots=True)
class CmnistConstruction:
    """Fixed-role construction; test rows are intentionally not a generic mapping."""

    train_e01: RenderedCmnistTable
    train_e02: RenderedCmnistTable
    val_e01: RenderedCmnistTable
    val_e02: RenderedCmnistTable
    val_held_out: RenderedCmnistTable
    _test_ood: RenderedCmnistTable
    partitions: CmnistPartitions
    manifest: CmnistDatasetManifest

    def training_views(self) -> tuple[TrainingView, TrainingView]:
        manifest_id = self.manifest.canonical_digest()
        views = tuple(
            TrainingView(
                descriptor=TrainingSplitDescriptor(
                    dataset_id="cmnist",
                    manifest_id=manifest_id,
                    name=table.name,
                    role="training",
                    source_partition_id=table.source_partition_id,
                    view_id=table.name,
                ),
                examples=table.identities(),
            )
            for table in (self.train_e01, self.train_e02)
        )
        return (views[0], views[1])

    def validation_views(self) -> tuple[ValidationView, ValidationView, ValidationView]:
        manifest_id = self.manifest.canonical_digest()
        views = tuple(
            ValidationView(
                descriptor=ValidationSplitDescriptor(
                    dataset_id="cmnist",
                    manifest_id=manifest_id,
                    name=table.name,
                    role="validation",
                    source_partition_id="validation_sources",
                    view_id=table.name,
                ),
                examples=table.identities(),
            )
            for table in (self.val_e01, self.val_e02, self.val_held_out)
        )
        typed_views = (views[0], views[1], views[2])
        validate_cmnist_repeated_validation_views(typed_views)
        return typed_views

    def preparation_tables(self) -> tuple[RenderedCmnistTable, ...]:
        """Expose fixed tables only to the explicit feature-preparation boundary."""

        return (
            self.train_e01,
            self.train_e02,
            self.val_e01,
            self.val_e02,
            self.val_held_out,
            self._test_ood,
        )


@dataclass(frozen=True, slots=True)
class CmnistPairSourceView:
    """Training-source capability accepted by the clean-oracle builder."""

    train_pool: MnistPool
    partitions: CmnistPartitions
    dataset_manifest: CmnistDatasetManifest

    @property
    def dataset_manifest_digest(self) -> str:
        return self.dataset_manifest.canonical_digest()


class OraclePairRecord(StrictBoundaryModel):
    pair_id: NonEmptyStr
    source_id: NonEmptyStr
    official_source_index: NonNegativeInt
    digit: Annotated[StrictInt, Field(ge=0, le=9)]
    clean_label: BinaryInt
    noisy_target: BinaryInt
    left_color: Literal["red"]
    right_color: Literal["green"]


class CmnistOraclePairManifest(StrictBoundaryModel):
    schema_version: Literal["grit.cmnist-oracle-pairs/v2"]
    dataset_manifest_digest: NonEmptyStr
    construction_method_id: Literal["cmnist-clean-oracle-pairs-v1"]
    pair_seed: StrictInt
    requested_count: NonNegativeInt
    realized_count: NonNegativeInt
    source_partition_ids: tuple[
        Literal["train_e01_sources", "train_e02_sources"], ...
    ]
    orientation: Literal["red_minus_green"]
    records: tuple[OraclePairRecord, ...]
    membership_digest: NonEmptyStr

    @model_validator(mode="after")
    def _validate_pairs(self) -> CmnistOraclePairManifest:
        if self.source_partition_ids != (
            "train_e01_sources",
            "train_e02_sources",
        ):
            raise ValueError("oracle pair sources must be the two training partitions")
        if self.requested_count != self.realized_count:
            raise ValueError(
                "oracle pair construction must realize its requested count"
            )
        if self.realized_count != len(self.records):
            raise ValueError("oracle pair record count is inconsistent")
        source_ids = tuple(record.source_id for record in self.records)
        if len(source_ids) != len(set(source_ids)):
            raise ValueError("oracle pair sources must be unique")
        if self.membership_digest != canonical_digest_value(source_ids):
            raise ValueError("oracle pair membership digest is inconsistent")
        for record in self.records:
            expected_pair_id = canonical_digest_value(
                {
                    "dataset_manifest_digest": self.dataset_manifest_digest,
                    "method": self.construction_method_id,
                    "pair_seed": self.pair_seed,
                    "source_id": record.source_id,
                    "orientation": self.orientation,
                }
            )
            if record.pair_id != expected_pair_id:
                raise ValueError("oracle pair identity is inconsistent")
        return self


@dataclass(frozen=True, slots=True)
class CmnistOraclePairSet:
    left_red: torch.Tensor
    right_green: torch.Tensor
    records: tuple[OraclePairRecord, ...]
    manifest: CmnistOraclePairManifest


def partition_cmnist_sources(
    train_pool: MnistPool,
    test_pool: MnistPool,
    *,
    construction_seed: int,
    targets: CmnistPartitionTargets = PRODUCTION_PARTITION_TARGETS,
) -> CmnistPartitions:
    """Apply ``cmnist-stratified-hash-v1`` independently of input row order."""

    train = _validated_pool(train_pool, "train", targets.official_train_count)
    test = _validated_pool(test_pool, "test", targets.test)
    if targets == PRODUCTION_PARTITION_TARGETS:
        if set(_tensor_ints(train.source_indices)) != set(range(60_000)):
            raise ValueError(
                "official MNIST train indices must be exactly 0 through 59999"
            )
        if set(_tensor_ints(test.source_indices)) != set(range(10_000)):
            raise ValueError(
                "official MNIST test indices must be exactly 0 through 9999"
            )

    train_indices = _tensor_ints(train.source_indices)
    train_digits = _tensor_ints(train.digits)
    by_digit: dict[int, list[int]] = {digit: [] for digit in range(10)}
    digit_by_index: dict[int, int] = {}
    for source_index, digit in zip(train_indices, train_digits, strict=True):
        digit_by_index[source_index] = digit
        by_digit[digit].append(source_index)
    for digit in range(10):
        by_digit[digit].sort(
            key=lambda source_index: (
                _membership_hash(construction_seed, "train", source_index),
                source_index,
            )
        )

    original_counts = {digit: len(by_digit[digit]) for digit in range(10)}
    validation_counts = _largest_remainder(original_counts, targets.validation)
    remaining_counts = {
        digit: original_counts[digit] - validation_counts[digit]
        for digit in range(10)
    }
    train_e01_counts = _largest_remainder(remaining_counts, targets.train_e01)

    validation_indices: list[int] = []
    train_e01_indices: list[int] = []
    train_e02_indices: list[int] = []
    for digit in range(10):
        ordered = by_digit[digit]
        validation_stop = validation_counts[digit]
        train_e01_stop = validation_stop + train_e01_counts[digit]
        validation_indices.extend(ordered[:validation_stop])
        train_e01_indices.extend(ordered[validation_stop:train_e01_stop])
        train_e02_indices.extend(ordered[train_e01_stop:])

    memberships = (
        tuple(sorted(train_e01_indices)),
        tuple(sorted(train_e02_indices)),
        tuple(sorted(validation_indices)),
        tuple(sorted(_tensor_ints(test.source_indices))),
    )
    names: tuple[SourcePartitionName, ...] = (
        "train_e01_sources",
        "train_e02_sources",
        "validation_sources",
        "test_sources",
    )
    digit_maps = (
        digit_by_index,
        digit_by_index,
        digit_by_index,
        dict(
            zip(
                _tensor_ints(test.source_indices),
                _tensor_ints(test.digits),
                strict=True,
            )
        ),
    )
    split_names: tuple[Literal["train", "test"], ...] = (
        "train",
        "train",
        "train",
        "test",
    )
    records = tuple(
        _partition_record(name, split, membership, digit_map)
        for name, split, membership, digit_map in zip(
            names, split_names, memberships, digit_maps, strict=True
        )
    )
    manifest = CmnistPartitionManifest(
        schema_version="grit.cmnist-partitions/v1",
        dataset_id="cmnist",
        construction_method_id=PARTITION_METHOD_ID,
        construction_seed=construction_seed,
        targets=targets,
        partitions=records,
    )
    return CmnistPartitions(
        train_e01_source_indices=memberships[0],
        train_e02_source_indices=memberships[1],
        validation_source_indices=memberships[2],
        test_source_indices=memberships[3],
        manifest=manifest,
    )


def construct_cmnist(
    train_pool: MnistPool,
    test_pool: MnistPool,
    *,
    construction_seed: int,
    label_flip_prob: float = 0.25,
    targets: CmnistPartitionTargets = PRODUCTION_PARTITION_TARGETS,
    held_out_flip_prob: float = 0.5,
    environment_specs: tuple[CmnistEnvironmentSpec, ...] | None = None,
) -> CmnistConstruction:
    """Partition sources, sample source-stable labels, and render fixed environments.

    ``held_out_flip_prob`` chooses the third validation rendering (``val_e03`` to
    ``val_e07``); everything else is fixed by the protocol.
    """

    if not 0.0 <= label_flip_prob <= 1.0:
        raise ValueError("label_flip_prob must lie in [0, 1]")
    protocol_specs = cmnist_environment_specs(held_out_flip_prob)
    if environment_specs is None:
        environment_specs = protocol_specs
    train = _validated_pool(train_pool, "train", targets.official_train_count)
    test = _validated_pool(test_pool, "test", targets.test)
    partitions = partition_cmnist_sources(
        train,
        test,
        construction_seed=construction_seed,
        targets=targets,
    )
    canonical_specs = {spec.name: spec for spec in protocol_specs}
    supplied_specs = {spec.name: spec for spec in environment_specs}
    if supplied_specs != canonical_specs or len(environment_specs) != len(
        canonical_specs
    ):
        raise ValueError("CMNIST environment specifications must match the protocol")

    pool_by_split = {"train": train, "test": test}
    memberships: dict[SourcePartitionName, tuple[int, ...]] = {
        "train_e01_sources": partitions.train_e01_source_indices,
        "train_e02_sources": partitions.train_e02_source_indices,
        "validation_sources": partitions.validation_source_indices,
        "test_sources": partitions.test_source_indices,
    }
    split_by_partition: dict[SourcePartitionName, Literal["train", "test"]] = {
        "train_e01_sources": "train",
        "train_e02_sources": "train",
        "validation_sources": "train",
        "test_sources": "test",
    }
    tables: dict[str, RenderedCmnistTable] = {}
    for name in tuple(spec.name for spec in protocol_specs):
        spec = supplied_specs[name]
        pool = pool_by_split[split_by_partition[spec.source_partition_id]]
        tables[name] = _render_environment(
            pool,
            memberships[spec.source_partition_id],
            spec,
            construction_seed=construction_seed,
            label_flip_prob=label_flip_prob,
        )

    environment_manifests = tuple(
        _environment_manifest(tables[spec.name], spec) for spec in protocol_specs
    )
    manifest = CmnistDatasetManifest(
        schema_version="grit.cmnist-dataset/v1",
        dataset_id="cmnist",
        partition_manifest=partitions.manifest,
        partition_manifest_digest=partitions.manifest.canonical_digest(),
        official_train_pool_digest=_pool_digest(train),
        official_test_pool_digest=_pool_digest(test),
        rendering_method_id=RENDERING_METHOD_ID,
        construction_seed=construction_seed,
        label_flip_prob=label_flip_prob,
        environments=environment_manifests,
    )
    return CmnistConstruction(
        train_e01=tables["train_e01"],
        train_e02=tables["train_e02"],
        val_e01=tables["val_e01"],
        val_e02=tables["val_e02"],
        val_held_out=tables[protocol_specs[HELD_OUT_ENVIRONMENT_INDEX].name],
        _test_ood=tables["test_ood"],
        partitions=partitions,
        manifest=manifest,
    )


def pair_source_view(
    construction: CmnistConstruction,
    train_pool: MnistPool,
) -> CmnistPairSourceView:
    """Mint the sole pair capability from the two training source partitions."""

    train = _validated_pool(
        train_pool,
        "train",
        construction.partitions.manifest.targets.official_train_count,
    )
    capability = CmnistPairSourceView(
        train_pool=train,
        partitions=construction.partitions,
        dataset_manifest=construction.manifest,
    )
    _ = _validated_pair_source_manifest(capability)
    return capability


def _validated_pair_source_manifest(
    sources: CmnistPairSourceView,
) -> CmnistDatasetManifest:
    manifest = CmnistDatasetManifest.model_validate(
        sources.dataset_manifest.model_dump(mode="python")
    )
    partition_manifest = CmnistPartitionManifest.model_validate(
        sources.partitions.manifest.model_dump(mode="python")
    )
    if partition_manifest != manifest.partition_manifest:
        raise ValueError(
            "pair-source partitions do not match the construction manifest"
        )
    observed_memberships = (
        sources.partitions.train_e01_source_indices,
        sources.partitions.train_e02_source_indices,
        sources.partitions.validation_source_indices,
        sources.partitions.test_source_indices,
    )
    expected_memberships = tuple(
        record.source_indices for record in partition_manifest.partitions
    )
    if observed_memberships != expected_memberships:
        raise ValueError(
            "pair-source partition membership does not match its manifest"
        )
    train = _validated_pool(
        sources.train_pool,
        "train",
        partition_manifest.targets.official_train_count,
    )
    if _pool_digest(train) != manifest.official_train_pool_digest:
        raise ValueError(
            "pair-source training pool digest does not match the construction manifest"
        )
    return manifest


def build_clean_oracle_pairs(
    sources: CmnistPairSourceView,
    *,
    pair_seed: int,
    pair_count: int = 256,
) -> CmnistOraclePairSet:
    """Sample unique training sources and orient every pair red minus green."""

    if type(sources) is not CmnistPairSourceView:
        raise TypeError("clean oracle pairs require CmnistPairSourceView")
    dataset_manifest = _validated_pair_source_manifest(sources)
    dataset_manifest_digest = dataset_manifest.canonical_digest()
    support = (
        sources.partitions.train_e01_source_indices
        + sources.partitions.train_e02_source_indices
    )
    if pair_count <= 0:
        raise ValueError("pair_count must be positive")
    if pair_count > len(support):
        raise ValueError("pair_count exceeds the training-source support")
    ordered = sorted(
        support,
        key=lambda index: (
            _oracle_pair_hash(pair_seed, _source_id("train", index)),
            index,
        ),
    )
    selected = tuple(ordered[:pair_count])
    pool = sources.train_pool
    positions = _pool_positions(pool)
    row_indices = torch.tensor(
        [positions[index] for index in selected], dtype=torch.int64
    )
    gray = _normalized_images(pool)[row_indices]
    digits = pool.digits.to(torch.int64)[row_indices]
    clean = (digits >= 5).to(torch.int64)
    noisy = torch.tensor(
        [
            int(clean_item)
            ^ _keyed_bernoulli(
                float(dataset_manifest.label_flip_prob),
                "label",
                dataset_manifest.construction_seed,
                _source_id("train", source_index),
            )
            for source_index, clean_item in zip(
                selected, _tensor_ints(clean), strict=True
            )
        ],
        dtype=torch.int64,
    )
    red = _render_rgb(gray, torch.zeros(pair_count, dtype=torch.int64))
    green = _render_rgb(gray, torch.ones(pair_count, dtype=torch.int64))
    records = tuple(
        OraclePairRecord(
            pair_id=canonical_digest_value(
                {
                    "dataset_manifest_digest": dataset_manifest_digest,
                    "method": ORACLE_PAIR_METHOD_ID,
                    "pair_seed": pair_seed,
                    "source_id": _source_id("train", source_index),
                    "orientation": "red_minus_green",
                }
            ),
            source_id=_source_id("train", source_index),
            official_source_index=source_index,
            digit=digit,
            clean_label=clean_label,
            noisy_target=target,
            left_color="red",
            right_color="green",
        )
        for source_index, digit, clean_label, target in zip(
            selected,
            _tensor_ints(digits),
            _tensor_ints(clean),
            _tensor_ints(noisy),
            strict=True,
        )
    )
    source_ids = tuple(record.source_id for record in records)
    manifest = CmnistOraclePairManifest(
        schema_version="grit.cmnist-oracle-pairs/v2",
        dataset_manifest_digest=dataset_manifest_digest,
        construction_method_id=ORACLE_PAIR_METHOD_ID,
        pair_seed=pair_seed,
        requested_count=pair_count,
        realized_count=len(records),
        source_partition_ids=("train_e01_sources", "train_e02_sources"),
        orientation="red_minus_green",
        records=records,
        membership_digest=canonical_digest_value(source_ids),
    )
    return CmnistOraclePairSet(
        left_red=red,
        right_green=green,
        records=records,
        manifest=manifest,
    )


def _validated_pool(
    pool: MnistPool,
    expected_split: Literal["train", "test"],
    expected_count: int,
) -> MnistPool:
    if pool.official_split != expected_split:
        raise ValueError(f"expected official MNIST {expected_split!r} pool")
    if pool.source_indices.ndim != 1 or pool.digits.ndim != 1:
        raise ValueError("MNIST source indices and digits must be one-dimensional")
    if pool.images.ndim != 3:
        raise ValueError("MNIST images must have shape [N, H, W]")
    count = int(pool.source_indices.shape[0])
    if count != expected_count:
        raise ValueError(
            f"official MNIST {expected_split} pool must contain {expected_count} rows"
        )
    if int(pool.images.shape[0]) != count or int(pool.digits.shape[0]) != count:
        raise ValueError("MNIST pool arrays must have aligned row counts")
    indices = _tensor_ints(pool.source_indices)
    if len(indices) != len(set(indices)):
        raise ValueError("official MNIST source indices must be unique")
    digits = _tensor_ints(pool.digits)
    if any(digit < 0 or digit > 9 for digit in digits):
        raise ValueError("MNIST digit labels must lie in 0 through 9")
    _ = _normalized_images(pool)
    return pool


def _normalized_images(pool: MnistPool) -> torch.Tensor:
    images = pool.images.detach().cpu()
    if images.dtype == torch.uint8:
        return images.to(torch.float32).div(255.0)
    normalized = images.to(torch.float32)
    if not bool(torch.isfinite(normalized).all()):
        raise ValueError("MNIST images must be finite")
    if float(normalized.min()) < 0.0 or float(normalized.max()) > 1.0:
        raise ValueError("floating-point MNIST images must lie in [0, 1]")
    return normalized


def _largest_remainder(counts: dict[int, int], requested: int) -> dict[int, int]:
    total = sum(counts.values())
    if requested < 0 or requested > total:
        raise ValueError("apportionment request exceeds available sources")
    allocated = {
        digit: (counts[digit] * requested) // total for digit in range(10)
    }
    remainders = {
        digit: (counts[digit] * requested) % total for digit in range(10)
    }
    remaining = requested - sum(allocated.values())
    for digit in sorted(range(10), key=lambda item: (-remainders[item], item))[
        :remaining
    ]:
        allocated[digit] += 1
    return allocated


def _membership_hash(seed: int, split: str, source_index: int) -> bytes:
    payload = f"{PARTITION_METHOD_ID}\0{seed}\0{split}\0{source_index}".encode()
    return hashlib.sha256(payload).digest()


def _oracle_pair_hash(seed: int, source_id: str) -> bytes:
    payload = f"{ORACLE_PAIR_METHOD_ID}\0{seed}\0{source_id}".encode()
    return hashlib.sha256(payload).digest()


def _keyed_bernoulli(probability: float, *identity: object) -> int:
    if probability <= 0.0:
        return 0
    if probability >= 1.0:
        return 1
    payload = "\0".join(str(item) for item in identity).encode()
    draw = int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")
    threshold = int(probability * (1 << 64))
    return int(draw < threshold)


def _source_id(split: str, source_index: int) -> str:
    return f"mnist:{split}:{source_index}"


def _pool_positions(pool: MnistPool) -> dict[int, int]:
    return {
        source_index: row
        for row, source_index in enumerate(_tensor_ints(pool.source_indices))
    }


def _render_environment(
    pool: MnistPool,
    source_indices: tuple[int, ...],
    spec: CmnistEnvironmentSpec,
    *,
    construction_seed: int,
    label_flip_prob: float,
) -> RenderedCmnistTable:
    positions = _pool_positions(pool)
    rows = torch.tensor(
        [positions[source_index] for source_index in source_indices], dtype=torch.int64
    )
    gray = _normalized_images(pool)[rows]
    digits = pool.digits.detach().cpu().to(torch.int64)[rows]
    clean = (digits >= 5).to(torch.int64)
    source_ids = tuple(
        _source_id(pool.official_split, source_index)
        for source_index in source_indices
    )
    targets = torch.tensor(
        [
            clean_label
            ^ _keyed_bernoulli(
                label_flip_prob,
                "label",
                construction_seed,
                source_id,
            )
            for clean_label, source_id in zip(
                _tensor_ints(clean), source_ids, strict=True
            )
        ],
        dtype=torch.int64,
    )
    colors = torch.tensor(
        [
            target
            ^ _keyed_bernoulli(
                float(spec.color_flip_prob),
                "color",
                construction_seed,
                source_id,
                spec.name,
            )
            for target, source_id in zip(
                _tensor_ints(targets), source_ids, strict=True
            )
        ],
        dtype=torch.int64,
    )
    example_ids = tuple(f"{source_id}:view:{spec.name}" for source_id in source_ids)
    return RenderedCmnistTable(
        name=spec.name,
        role=spec.role,
        source_partition_id=spec.source_partition_id,
        source_ids=source_ids,
        example_ids=example_ids,
        images=_render_rgb(gray, colors),
        digits=digits,
        clean_labels=clean,
        targets=targets,
        colors=colors,
        color_flip_prob=float(spec.color_flip_prob),
    )


def _render_rgb(gray: torch.Tensor, colors: torch.Tensor) -> torch.Tensor:
    count, height, width = (int(value) for value in gray.shape)
    rgb = torch.zeros((count, 3, height, width), dtype=torch.float32)
    red = colors == 0
    green = colors == 1
    rgb[red, 0] = gray[red]
    rgb[green, 1] = gray[green]
    return rgb


def _partition_record(
    name: SourcePartitionName,
    split: Literal["train", "test"],
    membership: tuple[int, ...],
    digit_by_index: dict[int, int],
) -> SourcePartitionRecord:
    counts = tuple(
        DigitCount(
            digit=digit,
            count=sum(digit_by_index[index] == digit for index in membership),
        )
        for digit in range(10)
    )
    return SourcePartitionRecord(
        name=name,
        official_split=split,
        source_indices=membership,
        digit_counts=counts,
        membership_digest=canonical_digest_value(
            {
                "name": name,
                "official_split": split,
                "source_indices": membership,
            }
        ),
    )


def _environment_manifest(
    table: RenderedCmnistTable,
    spec: CmnistEnvironmentSpec,
) -> EnvironmentManifest:
    return EnvironmentManifest(
        name=spec.name,
        role=spec.role,
        source_partition_id=spec.source_partition_id,
        color_flip_prob=spec.color_flip_prob,
        count=len(table.source_ids),
        source_membership_digest=canonical_digest_value(table.source_ids),
        record_digest=canonical_digest_value(
            {
                "source_ids": table.source_ids,
                "digits": _tensor_ints(table.digits),
                "clean_labels": _tensor_ints(table.clean_labels),
                "targets": _tensor_ints(table.targets),
                "colors": _tensor_ints(table.colors),
            }
        ),
    )


def _tensor_ints(values: torch.Tensor) -> tuple[int, ...]:
    flattened = torch.unbind(values.detach().cpu().reshape(-1))
    return tuple(int(value.item()) for value in flattened)


class _TensorToNumpy(Protocol):
    def __call__(self, *, force: bool = False) -> NDArray[np.generic]: ...


def _pool_digest(pool: MnistPool) -> str:
    positions = _pool_positions(pool)
    ordered_indices = tuple(sorted(positions))
    rows = torch.tensor(
        [positions[source_index] for source_index in ordered_indices],
        dtype=torch.int64,
    )
    arrays = (
        pool.source_indices.detach().cpu()[rows],
        pool.digits.detach().cpu()[rows],
        pool.images.detach().cpu()[rows],
    )
    digest = hashlib.sha256()
    digest.update(pool.official_split.encode("utf-8"))
    for tensor in arrays:
        contiguous = tensor.contiguous()
        to_numpy = cast(_TensorToNumpy, contiguous.numpy)
        array = to_numpy()
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(str(tuple(int(value) for value in array.shape)).encode("ascii"))
        digest.update(array.tobytes(order="C"))
    return f"sha256:{digest.hexdigest()}"
