"""Hermetic tests for deterministic Waterbirds-CF construction."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import cast

import numpy as np
import pytest
from PIL import Image
from pydantic import ValidationError

from grit.data.waterbirds import (
    BASE_ARTIFACT_NAME,
    ParsedWaterbirdsAssets,
    ProductionWaterbirdsProfile,
    WaterbirdsDatasetManifest,
    WaterbirdsOracleRelationView,
    composite_groupdro,
    construct_waterbirds_cf,
    mint_waterbirds_adjusted_weight_spec,
    parse_places_backgrounds,
    waterbirds_oracle_relation_view,
    waterbirds_training_view,
    waterbirds_validation_view,
)
from grit.data.waterbirds_pairs import (
    WaterbirdsOraclePairManifest,
    build_waterbirds_oracle_pairs,
)
from grit.data.waterbirds_smoke_assets import (
    make_waterbirds_smoke_assets as make_waterbirds_fixture,
)


def test_fixture_construction_replaces_minority_and_preserves_counts(
    tmp_path: Path,
) -> None:
    fixture = make_waterbirds_fixture(tmp_path / "assets")
    construction = construct_waterbirds_cf(
        fixture.assets, fixture.profile, tmp_path / "output"
    )

    assert construction.manifest.non_reportable is True
    assert construction.manifest.counts.supervised_training == 10
    assert construction.manifest.counts.unpaired_majority == 4
    assert construction.manifest.counts.retained_majority_endpoints == 3
    assert construction.manifest.counts.generated_minority_endpoints == 3
    assert construction.manifest.counts.oracle_relationships == 3
    assert construction.manifest.counts.training_groups.as_tuple() == (4, 2, 1, 3)
    assert set(construction.manifest.replaced_released_record_ids) == {
        "waterbirds:released:5",
        "waterbirds:released:6",
        "waterbirds:released:7",
    }
    training_ids = {record.record_id for record in construction.training_records()}
    assert not training_ids.intersection(
        construction.manifest.replaced_released_record_ids
    )
    assert len(construction.manifest.relationships) == 3
    assert {
        relationship.orientation for relationship in construction.manifest.relationships
    } == {"land_minus_water"}
    records = {record.record_id: record for record in construction.manifest.records}
    for relationship in construction.manifest.relationships:
        majority = records[relationship.majority_record_id]
        assert majority.selection_position == relationship.source_selection_position
        assert (
            majority.canonical_masked_source_foreground_digest
            == relationship.canonical_masked_source_foreground_digest
        )

    weights = mint_waterbirds_adjusted_weight_spec(construction.manifest)
    assert weights.group_order == (
        "landbird_land",
        "landbird_water",
        "waterbird_land",
        "waterbird_water",
    )
    assert weights.training_group_counts == construction.manifest.counts.training_groups
    assert weights.dataset_manifest_digest == construction.manifest.canonical_digest()
    assert weights.canonical_digest().startswith("sha256:")


def test_construction_is_order_independent_and_seeded(tmp_path: Path) -> None:
    fixture = make_waterbirds_fixture(tmp_path / "assets")
    first = construct_waterbirds_cf(fixture.assets, fixture.profile, tmp_path / "first")
    reordered_assets = replace(
        fixture.assets,
        released=fixture.assets.released.model_copy(
            update={"records": tuple(reversed(fixture.assets.released.records))}
        ),
        cub=fixture.assets.cub.model_copy(
            update={"records": tuple(reversed(fixture.assets.cub.records))}
        ),
        places=fixture.assets.places.model_copy(
            update={"records": tuple(reversed(fixture.assets.places.records))}
        ),
    )
    second = construct_waterbirds_cf(
        reordered_assets, fixture.profile, tmp_path / "second"
    )
    different_seed = construct_waterbirds_cf(
        fixture.assets,
        fixture.profile.model_copy(update={"construction_seed": 23}),
        tmp_path / "different-seed",
    )

    assert first.manifest.canonical_digest() == second.manifest.canonical_digest()
    assert (
        first.manifest.canonical_digest() != different_seed.manifest.canonical_digest()
    )
    assert tuple(
        item.majority_record_id for item in first.manifest.relationships
    ) != tuple(
        item.majority_record_id for item in different_seed.manifest.relationships
    )


def test_released_validation_and_test_bytes_are_unchanged(tmp_path: Path) -> None:
    fixture = make_waterbirds_fixture(tmp_path / "assets")
    construction = construct_waterbirds_cf(
        fixture.assets, fixture.profile, tmp_path / "output"
    )
    released_by_id = {
        record.record_id: record for record in fixture.assets.released.records
    }

    for record in (
        *construction.validation_records(),
        *construction.final_test_records(),
    ):
        source = fixture.assets.released_root / record.image_relative_path
        assert (
            construction.path_for(record.record_id).read_bytes() == source.read_bytes()
        )
        assert record.image_sha256 == released_by_id[record.record_id].image_sha256


def test_groupdro_compositor_preserves_foreground_and_recolors_background() -> None:
    source = np.zeros((4, 6, 3), dtype=np.uint8)
    source[:, :, :] = (200, 10, 20)
    mask = np.zeros((4, 6, 3), dtype=np.uint8)
    mask[:, :3, :] = 255
    background = np.zeros((8, 12, 3), dtype=np.uint8)
    background[:, :, :] = (1, 120, 240)

    result, geometry, foreground_digest = composite_groupdro(
        Image.fromarray(source),
        Image.fromarray(mask),
        Image.fromarray(background),
    )
    pixels = np.asarray(result)

    assert result.size == (6, 4)
    assert np.array_equal(pixels[:, :3, :], source[:, :3, :])
    assert np.array_equal(pixels[:, 3:, :], background[:4, :3, :])
    assert geometry.width == 6
    assert geometry.height == 4
    assert foreground_digest.startswith("sha256:")


def test_manifest_round_trip_revalidates_relationship_identity(tmp_path: Path) -> None:
    fixture = make_waterbirds_fixture(tmp_path / "assets")
    construction = construct_waterbirds_cf(
        fixture.assets, fixture.profile, tmp_path / "output"
    )
    restored = WaterbirdsDatasetManifest.model_validate_json(
        construction.manifest.canonical_json()
    )
    assert restored == construction.manifest

    payload = json.loads(construction.manifest.canonical_json())
    first_relationship = payload["relationships"][0]
    first_relationship["land_record_id"] = "waterbirds:released:missing"
    with pytest.raises(ValidationError, match="outside training"):
        WaterbirdsDatasetManifest.model_validate_json(json.dumps(payload))

    spoofed = json.loads(construction.manifest.canonical_json())
    relation = spoofed["relationships"][0]
    endpoint_ids = {relation["land_record_id"], relation["water_record_id"]}
    relation["pair_id"] = "sha256:caller-controlled"
    for record in spoofed["records"]:
        if record["record_id"] in endpoint_ids:
            record["pair_id"] = "sha256:caller-controlled"
    with pytest.raises(ValidationError, match="pair ID"):
        WaterbirdsDatasetManifest.model_validate_json(json.dumps(spoofed))

    production_claim = json.loads(construction.manifest.canonical_json())
    production_claim["profile_kind"] = "production"
    production_claim["non_reportable"] = False
    production_claim["base_artifact_name"] = BASE_ARTIFACT_NAME
    with pytest.raises(ValidationError, match="canonical inventory"):
        WaterbirdsDatasetManifest.model_validate_json(json.dumps(production_claim))


def test_manifest_rejects_missing_majority_position_and_invalid_replacements(
    tmp_path: Path,
) -> None:
    fixture = make_waterbirds_fixture(tmp_path / "assets")
    construction = construct_waterbirds_cf(
        fixture.assets, fixture.profile, tmp_path / "output"
    )

    missing_position = json.loads(construction.manifest.canonical_json())
    majority_id = missing_position["relationships"][0]["majority_record_id"]
    for record in missing_position["records"]:
        if record["record_id"] == majority_id:
            record["selection_position"] = None
            break
    with pytest.raises(ValidationError, match="selection position"):
        WaterbirdsDatasetManifest.model_validate_json(json.dumps(missing_position))

    duplicate = json.loads(construction.manifest.canonical_json())
    duplicate["replaced_released_record_ids"][1] = (
        duplicate["replaced_released_record_ids"][0]
    )
    with pytest.raises(ValidationError, match="must be unique"):
        WaterbirdsDatasetManifest.model_validate_json(json.dumps(duplicate))

    reintroduced = json.loads(construction.manifest.canonical_json())
    reintroduced["replaced_released_record_ids"][0] = reintroduced["records"][0][
        "record_id"
    ]
    with pytest.raises(ValidationError, match="absent from current records"):
        WaterbirdsDatasetManifest.model_validate_json(json.dumps(reintroduced))


def test_production_profile_rejects_fixture_inventory(tmp_path: Path) -> None:
    fixture = make_waterbirds_fixture(tmp_path / "assets")
    released = fixture.assets.released.model_copy(
        update={"artifact_name": BASE_ARTIFACT_NAME}
    )
    production_assets = ParsedWaterbirdsAssets(
        released_root=fixture.assets.released_root,
        cub_root=fixture.assets.cub_root,
        masks_root=fixture.assets.masks_root,
        places_root=fixture.assets.places_root,
        released=released,
        cub=fixture.assets.cub,
        places=fixture.assets.places,
    )
    with pytest.raises(ValueError, match="group counts"):
        construct_waterbirds_cf(
            production_assets,
            ProductionWaterbirdsProfile(
                kind="production",
                base_artifact_name=BASE_ARTIFACT_NAME,
                construction_seed=0,
            ),
            tmp_path / "production",
        )


def test_places_parser_rejects_missing_approved_category(tmp_path: Path) -> None:
    fixture = make_waterbirds_fixture(tmp_path / "assets")
    ocean = fixture.assets.places_root / "data_large" / "o" / "ocean"
    for path in ocean.iterdir():
        path.unlink()
    ocean.rmdir()

    with pytest.raises(FileNotFoundError, match="ocean"):
        parse_places_backgrounds(fixture.assets.places_root)


def test_training_view_redacts_backgrounds_and_oracle_relations(tmp_path: Path) -> None:
    fixture = make_waterbirds_fixture(tmp_path / "assets")
    construction = construct_waterbirds_cf(
        fixture.assets, fixture.profile, tmp_path / "output"
    )
    erm_view = waterbirds_training_view(construction)
    grit_view = waterbirds_training_view(construction)
    validation = waterbirds_validation_view(construction)

    assert tuple(record.record_id for record in erm_view.records) == tuple(
        record.record_id for record in grit_view.records
    )
    assert not hasattr(erm_view.records[0], "pair_id")
    assert not hasattr(erm_view.records[0], "background")
    assert {record.group_id for record in validation.records} == {
        "landbird_land",
        "landbird_water",
        "waterbird_land",
        "waterbird_water",
    }
    assert all(
        record.split_role == "validation"
        for record in construction.validation_records()
    )


def test_oracle_pairs_require_separate_capability_and_bind_dataset(
    tmp_path: Path,
) -> None:
    fixture = make_waterbirds_fixture(tmp_path / "assets")
    construction = construct_waterbirds_cf(
        fixture.assets, fixture.profile, tmp_path / "output"
    )
    training = waterbirds_training_view(construction)
    relations = waterbirds_oracle_relation_view(construction)
    pair_set = build_waterbirds_oracle_pairs(relations)

    assert pair_set.manifest.dataset_manifest_digest == (
        construction.manifest.canonical_digest()
    )
    assert pair_set.manifest.pair_count == 3
    assert pair_set.manifest.landbird_pair_count == 2
    assert pair_set.manifest.waterbird_pair_count == 1
    assert pair_set.manifest.orientation == "land_minus_water"
    with pytest.raises(TypeError, match="OracleRelationView"):
        build_waterbirds_oracle_pairs(cast(WaterbirdsOracleRelationView, training))


def test_oracle_pair_manifest_round_trip_rejects_tampering(tmp_path: Path) -> None:
    fixture = make_waterbirds_fixture(tmp_path / "assets")
    construction = construct_waterbirds_cf(
        fixture.assets, fixture.profile, tmp_path / "output"
    )
    pair_set = build_waterbirds_oracle_pairs(
        waterbirds_oracle_relation_view(construction)
    )
    restored = WaterbirdsOraclePairManifest.model_validate_json(
        pair_set.manifest.canonical_json()
    )
    assert restored == pair_set.manifest

    payload = json.loads(pair_set.manifest.canonical_json())
    payload["records"][0]["orientation"] = "water_minus_land"
    with pytest.raises(ValidationError):
        WaterbirdsOraclePairManifest.model_validate_json(json.dumps(payload))


def test_view_issuance_rejects_changed_image_bytes(tmp_path: Path) -> None:
    fixture = make_waterbirds_fixture(tmp_path / "assets")
    construction = construct_waterbirds_cf(
        fixture.assets, fixture.profile, tmp_path / "output"
    )
    path = construction.path_for(construction.training_records()[0].record_id)
    path.write_bytes(b"changed")

    with pytest.raises(ValueError, match="image identity"):
        waterbirds_training_view(construction)
