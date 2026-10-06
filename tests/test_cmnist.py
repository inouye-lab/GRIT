"""Deterministic CMNIST partition, rendering, and oracle-pair tests."""

from __future__ import annotations

import inspect
from dataclasses import replace
from typing import Literal, cast

import pytest
import torch

from grit.data.cmnist import (
    CMNIST_ENVIRONMENT_SPECS,
    PRODUCTION_PARTITION_TARGETS,
    CmnistDatasetManifest,
    CmnistOraclePairManifest,
    CmnistPairSourceView,
    CmnistPartitionManifest,
    CmnistPartitionTargets,
    MnistPool,
    build_clean_oracle_pairs,
    construct_cmnist,
    pair_source_view,
    partition_cmnist_sources,
)
from grit.data.views import ValidationView
from grit.schemas import canonical_digest_value

_TRAIN_DIGIT_COUNTS = (5923, 6742, 5958, 6131, 5842, 5421, 5918, 6265, 5851, 5949)
_TEST_DIGIT_COUNTS = (980, 1135, 1032, 1010, 982, 892, 958, 1028, 974, 1009)


def _pool(
    split: Literal["train", "test"],
    digit_counts: tuple[int, ...],
    *,
    reverse: bool = False,
) -> MnistPool:
    digits = torch.cat(
        [
            torch.full((count,), digit, dtype=torch.int64)
            for digit, count in enumerate(digit_counts)
        ]
    )
    source_indices = torch.arange(len(digits), dtype=torch.int64)
    images = (source_indices % 251).to(torch.uint8).reshape(-1, 1, 1)
    if reverse:
        order = torch.arange(len(digits) - 1, -1, -1)
        digits = digits[order]
        source_indices = source_indices[order]
        images = images[order]
    return MnistPool(
        official_split=split,
        source_indices=source_indices,
        images=images,
        digits=digits,
    )


def _small_pools(*, reverse: bool = False) -> tuple[MnistPool, MnistPool]:
    return (
        _pool("train", (5,) * 10, reverse=reverse),
        _pool("test", (1,) * 10, reverse=reverse),
    )


def _small_targets() -> CmnistPartitionTargets:
    return CmnistPartitionTargets(
        train_e01=20,
        train_e02=20,
        validation=10,
        test=10,
    )


def test_production_partition_counts_disjointness_and_apportionment() -> None:
    construction = construct_cmnist(
        _pool("train", _TRAIN_DIGIT_COUNTS),
        _pool("test", _TEST_DIGIT_COUNTS),
        construction_seed=17,
    )
    partitions = construction.partitions
    assert partitions.manifest.targets == PRODUCTION_PARTITION_TARGETS
    observed_counts = tuple(
        len(record.source_indices) for record in partitions.manifest.partitions
    )
    assert observed_counts == (25_000, 25_000, 10_000, 10_000)
    training_a = set(partitions.train_e01_source_indices)
    training_b = set(partitions.train_e02_source_indices)
    validation = set(partitions.validation_source_indices)
    assert not training_a & training_b
    assert not training_a & validation
    assert not training_b & validation
    assert training_a | training_b | validation == set(range(60_000))

    validation_counts = tuple(
        item.count for item in partitions.manifest.partitions[2].digit_counts
    )
    assert validation_counts == (
        987,
        1124,
        993,
        1022,
        974,
        904,
        986,
        1044,
        975,
        991,
    )
    assert sum(validation_counts) == 10_000
    remaining_counts = tuple(
        source - held_out
        for source, held_out in zip(
            _TRAIN_DIGIT_COUNTS, validation_counts, strict=True
        )
    )
    train_e01_counts = tuple(
        item.count for item in partitions.manifest.partitions[0].digit_counts
    )
    assert all(
        abs(allocated - remaining / 2) <= 0.5
        for allocated, remaining in zip(
            train_e01_counts, remaining_counts, strict=True
        )
    )
    for table in construction.preparation_tables():
        realized_agreement = float((table.colors == table.targets).float().mean())
        assert abs(realized_agreement - (1.0 - table.color_flip_prob)) <= 0.015


def test_partition_is_input_order_invariant_and_seeded() -> None:
    forward = partition_cmnist_sources(
        *_small_pools(), construction_seed=5, targets=_small_targets()
    )
    reversed_inputs = partition_cmnist_sources(
        *_small_pools(reverse=True), construction_seed=5, targets=_small_targets()
    )
    changed_seed = partition_cmnist_sources(
        *_small_pools(), construction_seed=6, targets=_small_targets()
    )
    assert forward.manifest == reversed_inputs.manifest
    assert forward.manifest != changed_seed.manifest
    assert (
        CmnistPartitionManifest.model_validate_json(forward.manifest.canonical_json())
        == forward.manifest
    )


def test_renderings_share_validation_sources_labels_and_content() -> None:
    train, test = _small_pools()
    construction = construct_cmnist(
        train,
        test,
        construction_seed=11,
        targets=_small_targets(),
    )
    reordered = construct_cmnist(
        train,
        test,
        construction_seed=11,
        targets=_small_targets(),
        environment_specs=tuple(reversed(CMNIST_ENVIRONMENT_SPECS)),
    )
    reordered_inputs = construct_cmnist(
        *_small_pools(reverse=True),
        construction_seed=11,
        targets=_small_targets(),
    )
    construction.validation_views()
    assert construction.manifest == reordered.manifest
    assert construction.manifest == reordered_inputs.manifest
    validation_tables = (
        construction.val_e01,
        construction.val_e02,
        construction.val_held_out,
    )
    assert len({table.source_ids for table in validation_tables}) == 1
    assert all(
        torch.equal(table.digits, construction.val_e01.digits)
        and torch.equal(table.clean_labels, construction.val_e01.clean_labels)
        and torch.equal(table.targets, construction.val_e01.targets)
        for table in validation_tables
    )
    for table in construction.preparation_tables():
        assert torch.equal(table.images[:, 2], torch.zeros_like(table.images[:, 2]))
        recovered_gray = table.images[:, 0] + table.images[:, 1]
        assert bool((recovered_gray >= 0).all())
    assert (
        CmnistDatasetManifest.model_validate_json(
            construction.manifest.canonical_json()
        )
        == construction.manifest
    )


def test_clean_oracle_pairs_are_unique_training_only_recolors() -> None:
    train, test = _small_pools()
    construction = construct_cmnist(
        train,
        test,
        construction_seed=3,
        targets=_small_targets(),
    )
    pairs = build_clean_oracle_pairs(
        pair_source_view(construction, train),
        pair_seed=19,
        pair_count=16,
    )
    repeated = build_clean_oracle_pairs(
        pair_source_view(construction, train),
        pair_seed=19,
        pair_count=16,
    )
    changed_seed = build_clean_oracle_pairs(
        pair_source_view(construction, train),
        pair_seed=20,
        pair_count=16,
    )
    source_ids = tuple(record.source_id for record in pairs.records)
    training_ids = {
        *construction.train_e01.source_ids,
        *construction.train_e02.source_ids,
    }
    assert len(source_ids) == len(set(source_ids)) == 16
    assert pairs.manifest.dataset_manifest_digest == (
        construction.manifest.canonical_digest()
    )
    first = pairs.records[0]
    assert first.pair_id == canonical_digest_value(
        {
            "dataset_manifest_digest": construction.manifest.canonical_digest(),
            "method": "cmnist-clean-oracle-pairs-v1",
            "pair_seed": 19,
            "source_id": first.source_id,
            "orientation": "red_minus_green",
        }
    )
    assert repeated.manifest == pairs.manifest
    assert changed_seed.manifest.membership_digest != pairs.manifest.membership_digest
    assert set(source_ids) <= training_ids
    assert not set(source_ids) & set(construction.val_e01.source_ids)
    assert torch.equal(pairs.left_red[:, 0], pairs.right_green[:, 1])
    assert torch.equal(pairs.left_red[:, 1], torch.zeros_like(pairs.left_red[:, 1]))
    assert torch.equal(
        pairs.right_green[:, 0], torch.zeros_like(pairs.right_green[:, 0])
    )
    assert all(
        record.left_color == "red" and record.right_color == "green"
        for record in pairs.records
    )
    assert (
        CmnistOraclePairManifest.model_validate_json(pairs.manifest.canonical_json())
        == pairs.manifest
    )


@pytest.mark.parametrize("changed_field", ("pixels", "digits"))
def test_pair_source_rejects_same_indices_with_changed_pool_content(
    changed_field: str,
) -> None:
    train, test = _small_pools()
    construction = construct_cmnist(
        train,
        test,
        construction_seed=3,
        targets=_small_targets(),
    )
    images = train.images.clone()
    digits = train.digits.clone()
    if changed_field == "pixels":
        images[0, 0, 0] = 1
    else:
        digits[0] = 1
    changed_pool = MnistPool(
        official_split="train",
        source_indices=train.source_indices.clone(),
        images=images,
        digits=digits,
    )
    with pytest.raises(ValueError, match="pool digest"):
        pair_source_view(construction, changed_pool)


def test_pair_dataset_digest_is_derived_from_the_capability() -> None:
    assert "dataset_manifest_digest" not in inspect.signature(
        build_clean_oracle_pairs
    ).parameters
    train, test = _small_pools()
    construction = construct_cmnist(
        train,
        test,
        construction_seed=3,
        targets=_small_targets(),
    )
    capability = pair_source_view(construction, train)
    pairs = build_clean_oracle_pairs(capability, pair_seed=1, pair_count=2)
    assert capability.dataset_manifest_digest == (
        construction.manifest.canonical_digest()
    )
    assert pairs.manifest.dataset_manifest_digest == capability.dataset_manifest_digest
    spoofed = replace(
        capability,
        dataset_manifest=capability.dataset_manifest.model_copy(
            update={"official_train_pool_digest": "sha256:spoofed-pool"}
        ),
    )
    with pytest.raises(ValueError, match="pool digest"):
        build_clean_oracle_pairs(spoofed, pair_seed=1, pair_count=2)


def test_oracle_builder_rejects_a_validation_view() -> None:
    train, test = _small_pools()
    construction = construct_cmnist(
        train,
        test,
        construction_seed=3,
        targets=_small_targets(),
    )
    validation = construction.validation_views()[0]
    invalid = cast(CmnistPairSourceView, cast(object, validation))
    with pytest.raises(TypeError, match="CmnistPairSourceView"):
        build_clean_oracle_pairs(
            invalid,
            pair_seed=1,
            pair_count=2,
        )


def test_validation_view_type_is_not_a_pair_source_capability() -> None:
    assert ValidationView is not CmnistPairSourceView
