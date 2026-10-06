"""Hermetic Waterbirds feature-cache and oracle-projection tests."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
from PIL import Image
from pydantic import ValidationError

from grit.data.waterbirds import (
    WaterbirdsConstruction,
    construct_waterbirds_cf,
    waterbirds_oracle_relation_view,
)
from grit.data.waterbirds_pairs import (
    WaterbirdsOraclePairSet,
    build_waterbirds_oracle_pairs,
)
from grit.data.waterbirds_smoke_assets import (
    make_waterbirds_smoke_assets as make_waterbirds_fixture,
)
from grit.features.cmnist import EncoderIdentity, FeatureExtractionRuntime
from grit.features.waterbirds import (
    DeterministicFakeWaterbirdsEncoder,
    Normalization,
    WaterbirdsEvaluationFeatureTable,
    WaterbirdsFeatureCache,
    WaterbirdsFeatureCacheError,
    WaterbirdsFeatureCacheManifest,
    WaterbirdsFinalTestView,
    fit_waterbirds_oracle_projection,
    load_waterbirds_feature_cache,
    load_waterbirds_tuning_feature_cache,
    prepare_waterbirds_feature_cache,
    waterbirds_oracle_pair_features,
)
from grit.methods.projection import ProjectionDiagnostics


def _prepared(
    tmp_path: Path,
    *,
    name: str = "one",
    normalization: Normalization = "none",
) -> tuple[WaterbirdsConstruction, WaterbirdsOraclePairSet, WaterbirdsFeatureCache]:
    fixture = make_waterbirds_fixture(tmp_path / f"assets-{name}")
    construction = construct_waterbirds_cf(
        fixture.assets, fixture.profile, tmp_path / f"construction-{name}"
    )
    pairs = build_waterbirds_oracle_pairs(waterbirds_oracle_relation_view(construction))
    cache_root = tmp_path / f"features-{name}"
    prepare_waterbirds_feature_cache(
        construction,
        DeterministicFakeWaterbirdsEncoder(seed=5),
        cache_root,
        normalization=normalization,
        encode_batch_size=3,
    )
    cache = load_waterbirds_feature_cache(
        cache_root,
        expected_dataset_manifest_digest=construction.manifest.canonical_digest(),
        expected_normalization=normalization,
    )
    return construction, pairs, cache


def test_feature_cache_is_canonical_role_scoped_and_non_reportable(
    tmp_path: Path,
) -> None:
    fixture = make_waterbirds_fixture(tmp_path / "assets")
    construction = construct_waterbirds_cf(
        fixture.assets, fixture.profile, tmp_path / "construction"
    )
    first_root = tmp_path / "first"
    second_root = tmp_path / "second"
    first = prepare_waterbirds_feature_cache(
        construction,
        DeterministicFakeWaterbirdsEncoder(seed=5),
        first_root,
        normalization="none",
        encode_batch_size=2,
    )
    second = prepare_waterbirds_feature_cache(
        construction,
        DeterministicFakeWaterbirdsEncoder(seed=5),
        second_root,
        normalization="none",
        encode_batch_size=7,
    )

    assert first == second
    assert first.non_reportable is True
    assert first.canonical_digest() == second.canonical_digest()
    assert (first_root / "features.npy").read_bytes() == (
        second_root / "features.npy"
    ).read_bytes()
    cache = load_waterbirds_feature_cache(first_root)
    training = cache.training_table()
    validation = cache.validation_table()
    assert training.features.shape == (10, 512)
    assert not hasattr(training, "backgrounds")
    assert validation.features.shape == (4, 512)
    assert set(validation.group_ids) == {
        "landbird_land",
        "landbird_water",
        "waterbird_land",
        "waterbird_water",
    }


def test_tuning_cache_exposes_no_waterbirds_final_capability(tmp_path: Path) -> None:
    construction, pairs, cache = _prepared(tmp_path)
    tuning = load_waterbirds_tuning_feature_cache(
        cache.root,
        expected_dataset_manifest_digest=construction.manifest.canonical_digest(),
        expected_normalization="none",
    )
    assert tuning.training_table().record_ids == cache.training_table().record_ids
    assert tuning.validation_table().record_ids == cache.validation_table().record_ids
    left, right = waterbirds_oracle_pair_features(tuning, pairs)
    assert left.shape == right.shape == (len(pairs.manifest.records), 512)
    assert not hasattr(tuning, "issue_final_handle")
    assert not hasattr(tuning, "verify_final_view")


def test_cache_rejects_digest_tampering_and_normalization_mixing(
    tmp_path: Path,
) -> None:
    construction, _, cache = _prepared(tmp_path)
    with pytest.raises(WaterbirdsFeatureCacheError, match="normalization"):
        load_waterbirds_feature_cache(cache.root, expected_normalization="l2")
    with pytest.raises(WaterbirdsFeatureCacheError, match="dataset"):
        load_waterbirds_feature_cache(
            cache.root,
            expected_dataset_manifest_digest="sha256:not-the-dataset",
        )

    feature_path = cache.root / "features.npy"
    feature_path.write_bytes(feature_path.read_bytes() + b"changed")
    with pytest.raises(WaterbirdsFeatureCacheError, match="digest"):
        load_waterbirds_feature_cache(
            cache.root,
            expected_dataset_manifest_digest=(construction.manifest.canonical_digest()),
        )


def test_feature_preparation_writes_nothing_after_encoder_failure(
    tmp_path: Path,
) -> None:
    fixture = make_waterbirds_fixture(tmp_path / "assets")
    construction = construct_waterbirds_cf(
        fixture.assets, fixture.profile, tmp_path / "construction"
    )

    class InvalidEncoder:
        @property
        def identity(self) -> EncoderIdentity:
            return DeterministicFakeWaterbirdsEncoder().identity

        @property
        def extraction_runtime(self) -> FeatureExtractionRuntime:
            return DeterministicFakeWaterbirdsEncoder().extraction_runtime

        def encode_pil(self, images: tuple[Image.Image, ...]) -> torch.Tensor:
            return torch.zeros((len(images), 2), dtype=torch.float32)

    output = tmp_path / "invalid"
    with pytest.raises(ValueError, match="return shape"):
        prepare_waterbirds_feature_cache(
            construction,
            InvalidEncoder(),
            output,
            normalization="none",
        )
    assert not output.exists()


def test_pair_features_reject_another_construction(tmp_path: Path) -> None:
    first_fixture = make_waterbirds_fixture(tmp_path / "first-assets")
    first_construction = construct_waterbirds_cf(
        first_fixture.assets,
        first_fixture.profile,
        tmp_path / "first-construction",
    )
    first_pairs = build_waterbirds_oracle_pairs(
        waterbirds_oracle_relation_view(first_construction)
    )

    second_fixture = make_waterbirds_fixture(tmp_path / "second-assets")
    second_profile = second_fixture.profile.model_copy(update={"construction_seed": 99})
    second_construction = construct_waterbirds_cf(
        second_fixture.assets,
        second_profile,
        tmp_path / "second-construction",
    )
    cache_root = tmp_path / "second-features"
    prepare_waterbirds_feature_cache(
        second_construction,
        DeterministicFakeWaterbirdsEncoder(),
        cache_root,
        normalization="none",
    )
    second_cache = load_waterbirds_feature_cache(cache_root)

    with pytest.raises(WaterbirdsFeatureCacheError, match="different datasets"):
        waterbirds_oracle_pair_features(second_cache, first_pairs)


def test_projection_binds_exact_pair_and_feature_manifests(tmp_path: Path) -> None:
    _, pairs_value, cache_value = _prepared(tmp_path)
    pairs = pairs_value
    cache = cache_value
    projection = fit_waterbirds_oracle_projection(
        cache,
        pairs,
        requested_rank=2,
    )

    assert projection.diagnostics.pair_manifest_digest == (
        pairs.manifest.canonical_digest()
    )
    assert projection.diagnostics.feature_cache_manifest_digest == (
        cache.manifest.canonical_digest()
    )
    assert projection.diagnostics.operation == "uncentered_left_minus_right"
    restored = ProjectionDiagnostics.model_validate_json(
        projection.diagnostics.canonical_json()
    )
    assert restored == projection.diagnostics
    rank_zero = fit_waterbirds_oracle_projection(
        cache,
        pairs,
        requested_rank=0,
    )
    values = torch.randn((4, 512), generator=torch.Generator().manual_seed(4))
    assert torch.equal(rank_zero.transform(values), values)


def test_feature_manifest_round_trip_rejects_record_reordering(tmp_path: Path) -> None:
    _, _, cache_value = _prepared(tmp_path)
    cache = cache_value
    restored = WaterbirdsFeatureCacheManifest.model_validate_json(
        cache.manifest.canonical_json()
    )
    assert restored == cache.manifest

    payload = json.loads(cache.manifest.canonical_json())
    payload["records"][0]["row_index"] = 1
    with pytest.raises(ValidationError, match="contiguous"):
        WaterbirdsFeatureCacheManifest.model_validate_json(json.dumps(payload))


def test_final_view_is_bound_to_exact_waterbirds_feature_cache(
    tmp_path: Path,
) -> None:
    fixture = make_waterbirds_fixture(tmp_path / "assets")
    construction = construct_waterbirds_cf(
        fixture.assets, fixture.profile, tmp_path / "construction"
    )
    caches: list[WaterbirdsFeatureCache] = []
    for seed in (1, 2):
        root = tmp_path / f"cache-{seed}"
        prepare_waterbirds_feature_cache(
            construction,
            DeterministicFakeWaterbirdsEncoder(seed=seed),
            root,
            normalization="none",
        )
        caches.append(load_waterbirds_feature_cache(root))
    first, second = caches
    final_rows = tuple(
        record for record in first.manifest.records if record.split_role == "final_test"
    )
    indices = torch.tensor(
        [record.row_index for record in final_rows], dtype=torch.int64
    )
    table = WaterbirdsEvaluationFeatureTable(
        dataset_manifest_digest=first.manifest.dataset_manifest_digest,
        feature_cache_manifest_digest=first.manifest.canonical_digest(),
        normalization=first.manifest.normalization,
        split_role="final_test",
        record_ids=tuple(record.record_id for record in final_rows),
        features=first.features[indices],
        labels=torch.tensor([record.bird_label for record in final_rows]),
        backgrounds=torch.tensor([record.background for record in final_rows]),
        group_ids=tuple(record.group_id for record in final_rows),
    )
    view = WaterbirdsFinalTestView(
        authorization_id="final:fixture",
        run_id="run:fixture",
        candidate_id="candidate:fixture",
        method_id="erm",
        scientific_config_digest="config:fixture",
        checkpoint_id="checkpoint:fixture",
        epoch=1,
        seed=301,
        projection_rank=None,
        feature_cache_manifest_digest=first.manifest.canonical_digest(),
        table=table,
    )

    assert first.verify_final_view(view) == table
    with pytest.raises(WaterbirdsFeatureCacheError, match="another"):
        second.verify_final_view(view)
