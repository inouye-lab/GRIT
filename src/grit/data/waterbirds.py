"""Deterministic Waterbirds-CF construction from explicit local assets."""

from __future__ import annotations

import csv
import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Literal, TypeAlias, cast

import numpy as np
from numpy.typing import NDArray
from PIL import Image
from pydantic import (
    Field,
    FiniteFloat,
    PositiveInt,
    StrictBool,
    StrictInt,
    StrictStr,
    model_validator,
)

from grit.schemas import StrictBoundaryModel, canonical_digest_value

BASE_ARTIFACT_NAME = "waterbird_complete95_forest2water2"
CONSTRUCTION_METHOD_ID = "waterbirds-cf-sha256-v1"
COMPOSITOR_ID = "groupdro-center-crop-lanczos-v1"
GENERATED_ENCODING_ID = "png-rgb-v1"

NonEmptyStr: TypeAlias = Annotated[StrictStr, Field(min_length=1)]
NonNegativeInt: TypeAlias = Annotated[StrictInt, Field(ge=0)]
BinaryInt: TypeAlias = Annotated[StrictInt, Field(ge=0, le=1)]
NonNegativeFloat: TypeAlias = Annotated[FiniteFloat, Field(ge=0.0)]

BirdLabel: TypeAlias = Literal[0, 1]
Background: TypeAlias = Literal[0, 1]
SplitRole: TypeAlias = Literal["training", "validation", "final_test"]
GroupId: TypeAlias = Literal[
    "landbird_land",
    "landbird_water",
    "waterbird_land",
    "waterbird_water",
]
WATERBIRDS_GROUP_ORDER: tuple[GroupId, GroupId, GroupId, GroupId] = (
    "landbird_land",
    "landbird_water",
    "waterbird_land",
    "waterbird_water",
)
Component: TypeAlias = Literal[
    "unpaired_majority",
    "majority_endpoint",
    "generated_minority",
    "released_validation",
    "released_test",
]
EndpointRole: TypeAlias = Literal["land", "water"]
PlaceCategory: TypeAlias = Literal[
    "bamboo_forest",
    "forest/broadleaf",
    "lake/natural",
    "ocean",
]

APPROVED_PLACE_CATEGORIES: tuple[PlaceCategory, ...] = (
    "bamboo_forest",
    "forest/broadleaf",
    "lake/natural",
    "ocean",
)


class WaterbirdsGroupCounts(StrictBoundaryModel):
    landbird_land: NonNegativeInt
    landbird_water: NonNegativeInt
    waterbird_land: NonNegativeInt
    waterbird_water: NonNegativeInt

    @property
    def total(self) -> int:
        return (
            self.landbird_land
            + self.landbird_water
            + self.waterbird_land
            + self.waterbird_water
        )

    def as_tuple(self) -> tuple[int, int, int, int]:
        return (
            self.landbird_land,
            self.landbird_water,
            self.waterbird_land,
            self.waterbird_water,
        )


class WaterbirdsAdjustedWeightSpec(StrictBoundaryModel):
    """Dataset-bound adjusted-average weights issued from a validated manifest."""

    schema_version: Literal["grit.waterbirds-adjusted-weights/v1"]
    group_order: tuple[GroupId, GroupId, GroupId, GroupId]
    training_group_counts: WaterbirdsGroupCounts
    dataset_manifest_digest: NonEmptyStr

    @model_validator(mode="after")
    def _validate_weights(self) -> WaterbirdsAdjustedWeightSpec:
        if self.group_order != WATERBIRDS_GROUP_ORDER:
            raise ValueError(
                "Waterbirds adjusted weights require canonical group order"
            )
        if min(self.training_group_counts.as_tuple()) <= 0:
            raise ValueError("Waterbirds adjusted weights require every training group")
        return self


PRODUCTION_TRAIN_GROUP_COUNTS = WaterbirdsGroupCounts(
    landbird_land=3_498,
    landbird_water=184,
    waterbird_land=56,
    waterbird_water=1_057,
)


class ProductionWaterbirdsProfile(StrictBoundaryModel):
    kind: Literal["production"]
    base_artifact_name: Literal["waterbird_complete95_forest2water2"]
    construction_seed: StrictInt


class FixtureWaterbirdsProfile(StrictBoundaryModel):
    kind: Literal["fixture"]
    non_reportable: Literal[True]
    base_artifact_name: NonEmptyStr
    construction_seed: StrictInt
    train_group_counts: WaterbirdsGroupCounts
    landbird_pair_count: PositiveInt
    waterbird_pair_count: PositiveInt
    validation_count: PositiveInt
    test_count: PositiveInt

    @model_validator(mode="after")
    def _validate_replacement_counts(self) -> FixtureWaterbirdsProfile:
        if self.landbird_pair_count != self.train_group_counts.landbird_water:
            raise ValueError(
                "fixture landbird pair count must replace every landbird/water record"
            )
        if self.waterbird_pair_count != self.train_group_counts.waterbird_land:
            raise ValueError(
                "fixture waterbird pair count must replace every waterbird/land record"
            )
        if self.landbird_pair_count > self.train_group_counts.landbird_land:
            raise ValueError("fixture lacks landbird majority endpoints")
        if self.waterbird_pair_count > self.train_group_counts.waterbird_water:
            raise ValueError("fixture lacks waterbird majority endpoints")
        return self


WaterbirdsConstructionProfile: TypeAlias = (
    ProductionWaterbirdsProfile | FixtureWaterbirdsProfile
)


class ReleasedWaterbirdsRecord(StrictBoundaryModel):
    record_id: NonEmptyStr
    cub_image_id: PositiveInt
    image_relative_path: NonEmptyStr
    bird_label: BinaryInt
    background: BinaryInt
    split_role: SplitRole
    place_relative_path: NonEmptyStr | None
    image_sha256: NonEmptyStr


class ReleasedWaterbirdsManifest(StrictBoundaryModel):
    schema_version: Literal["grit.waterbirds-released/v1"]
    artifact_name: NonEmptyStr
    metadata_file_sha256: NonEmptyStr
    metadata_rows_digest: NonEmptyStr
    records: tuple[ReleasedWaterbirdsRecord, ...]

    @model_validator(mode="after")
    def _validate_records(self) -> ReleasedWaterbirdsManifest:
        if not self.records:
            raise ValueError("released Waterbirds metadata must not be empty")
        record_ids = tuple(record.record_id for record in self.records)
        cub_ids = tuple(record.cub_image_id for record in self.records)
        if len(set(record_ids)) != len(record_ids):
            raise ValueError("released Waterbirds record IDs must be unique")
        if len(set(cub_ids)) != len(cub_ids):
            raise ValueError("released Waterbirds CUB image IDs must be unique")
        expected = canonical_digest_value(
            tuple(
                record.model_dump(mode="json")
                for record in sorted(self.records, key=lambda item: item.record_id)
            )
        )
        if self.metadata_rows_digest != expected:
            raise ValueError("released Waterbirds metadata row digest is inconsistent")
        return self


class CubSourceRecord(StrictBoundaryModel):
    cub_image_id: PositiveInt
    image_relative_path: NonEmptyStr
    mask_relative_path: NonEmptyStr
    class_id: PositiveInt
    species: NonEmptyStr
    bounding_box_xywh: tuple[
        NonNegativeFloat,
        NonNegativeFloat,
        NonNegativeFloat,
        NonNegativeFloat,
    ]
    image_sha256: NonEmptyStr
    mask_sha256: NonEmptyStr

    @model_validator(mode="after")
    def _validate_box(self) -> CubSourceRecord:
        if self.bounding_box_xywh[2] <= 0.0 or self.bounding_box_xywh[3] <= 0.0:
            raise ValueError("CUB bounding-box width and height must be positive")
        return self


class CubSourceManifest(StrictBoundaryModel):
    schema_version: Literal["grit.cub-sources/v1"]
    images_file_sha256: NonEmptyStr
    labels_file_sha256: NonEmptyStr
    classes_file_sha256: NonEmptyStr
    bounding_boxes_file_sha256: NonEmptyStr
    records_digest: NonEmptyStr
    records: tuple[CubSourceRecord, ...]

    @model_validator(mode="after")
    def _validate_records(self) -> CubSourceManifest:
        ids = tuple(record.cub_image_id for record in self.records)
        if not ids or len(ids) != len(set(ids)):
            raise ValueError("CUB source IDs must be nonempty and unique")
        expected = canonical_digest_value(
            tuple(
                record.model_dump(mode="json")
                for record in sorted(self.records, key=lambda item: item.cub_image_id)
            )
        )
        if self.records_digest != expected:
            raise ValueError("CUB source record digest is inconsistent")
        return self


class PlacesBackgroundRecord(StrictBoundaryModel):
    asset_id: NonEmptyStr
    category: PlaceCategory
    background: BinaryInt
    relative_path: NonEmptyStr
    image_sha256: NonEmptyStr


class PlacesSourceManifest(StrictBoundaryModel):
    schema_version: Literal["grit.places-backgrounds/v1"]
    categories: tuple[PlaceCategory, ...]
    records_digest: NonEmptyStr
    records: tuple[PlacesBackgroundRecord, ...]

    @model_validator(mode="after")
    def _validate_records(self) -> PlacesSourceManifest:
        if self.categories != APPROVED_PLACE_CATEGORIES:
            raise ValueError(
                "Places manifest must contain the four approved categories"
            )
        asset_ids = tuple(record.asset_id for record in self.records)
        paths = tuple(record.relative_path for record in self.records)
        if not asset_ids or len(asset_ids) != len(set(asset_ids)):
            raise ValueError("Places background IDs must be nonempty and unique")
        if len(paths) != len(set(paths)):
            raise ValueError("Places background paths must be unique")
        if {record.category for record in self.records} != set(self.categories):
            raise ValueError("every approved Places category must contain an image")
        expected = canonical_digest_value(
            tuple(
                record.model_dump(mode="json")
                for record in sorted(self.records, key=lambda item: item.asset_id)
            )
        )
        if self.records_digest != expected:
            raise ValueError("Places source record digest is inconsistent")
        return self


@dataclass(frozen=True, slots=True)
class ParsedWaterbirdsAssets:
    released_root: Path
    cub_root: Path
    masks_root: Path
    places_root: Path
    released: ReleasedWaterbirdsManifest
    cub: CubSourceManifest
    places: PlacesSourceManifest

    def source_bundle_digest(self) -> str:
        return canonical_digest_value(
            {
                "released": {
                    "schema_version": self.released.schema_version,
                    "artifact_name": self.released.artifact_name,
                    "metadata_file_sha256": self.released.metadata_file_sha256,
                    "metadata_rows_digest": self.released.metadata_rows_digest,
                    "records": tuple(
                        record.model_dump(mode="json")
                        for record in sorted(
                            self.released.records,
                            key=lambda item: item.record_id,
                        )
                    ),
                },
                "cub": {
                    "schema_version": self.cub.schema_version,
                    "images_file_sha256": self.cub.images_file_sha256,
                    "labels_file_sha256": self.cub.labels_file_sha256,
                    "classes_file_sha256": self.cub.classes_file_sha256,
                    "bounding_boxes_file_sha256": (self.cub.bounding_boxes_file_sha256),
                    "records_digest": self.cub.records_digest,
                    "records": tuple(
                        record.model_dump(mode="json")
                        for record in sorted(
                            self.cub.records,
                            key=lambda item: item.cub_image_id,
                        )
                    ),
                },
                "places": {
                    "schema_version": self.places.schema_version,
                    "categories": self.places.categories,
                    "records_digest": self.places.records_digest,
                    "records": tuple(
                        record.model_dump(mode="json")
                        for record in sorted(
                            self.places.records,
                            key=lambda item: item.asset_id,
                        )
                    ),
                },
            }
        )


class GroupDroGeometry(StrictBoundaryModel):
    compositor_id: Literal["groupdro-center-crop-lanczos-v1"]
    generated_encoding_id: Literal["png-rgb-v1"]
    width: PositiveInt
    height: PositiveInt
    resampling: Literal["pillow_lanczos"]


class WaterbirdsRecord(StrictBoundaryModel):
    record_id: NonEmptyStr
    split_role: SplitRole
    component: Component
    bird_label: BinaryInt
    background: BinaryInt
    group_id: GroupId
    source_cub_image_id: PositiveInt
    species: NonEmptyStr
    image_location: Literal["released", "generated"]
    image_relative_path: NonEmptyStr
    image_sha256: NonEmptyStr
    source_image_sha256: NonEmptyStr
    mask_sha256: NonEmptyStr
    canonical_masked_source_foreground_digest: NonEmptyStr
    bounding_box_xywh: tuple[
        NonNegativeFloat,
        NonNegativeFloat,
        NonNegativeFloat,
        NonNegativeFloat,
    ]
    pair_id: NonEmptyStr | None
    endpoint_role: EndpointRole | None
    background_asset_id: NonEmptyStr | None
    background_asset_sha256: NonEmptyStr | None
    construction_seed: StrictInt | None
    selection_position: NonNegativeInt | None
    geometry: GroupDroGeometry

    @model_validator(mode="after")
    def _validate_component(self) -> WaterbirdsRecord:
        paired = self.component in {"majority_endpoint", "generated_minority"}
        has_pair_fields = self.pair_id is not None and self.endpoint_role is not None
        has_partial_pair_fields = (self.pair_id is None) != (self.endpoint_role is None)
        if has_partial_pair_fields or paired != has_pair_fields:
            raise ValueError(
                "paired Waterbirds records require pair and endpoint roles"
            )
        generated = self.component == "generated_minority"
        if generated != (self.image_location == "generated"):
            raise ValueError("only generated-minority records use generated images")
        if generated and (
            self.background_asset_id is None
            or self.background_asset_sha256 is None
            or self.construction_seed is None
            or self.selection_position is None
        ):
            raise ValueError(
                "generated endpoints require background and seed provenance"
            )
        if self.group_id != waterbirds_group_id(
            int(self.bird_label), int(self.background)
        ):
            raise ValueError("Waterbirds record group ID is inconsistent")
        return self


class WaterbirdsOracleRelationship(StrictBoundaryModel):
    pair_id: NonEmptyStr
    construction_input_digest: NonEmptyStr
    source_cub_image_id: PositiveInt
    bird_label: BinaryInt
    land_record_id: NonEmptyStr
    water_record_id: NonEmptyStr
    majority_record_id: NonEmptyStr
    generated_record_id: NonEmptyStr
    land_image_sha256: NonEmptyStr
    water_image_sha256: NonEmptyStr
    source_image_sha256: NonEmptyStr
    mask_sha256: NonEmptyStr
    canonical_masked_source_foreground_digest: NonEmptyStr
    land_background_asset_id: NonEmptyStr
    water_background_asset_id: NonEmptyStr
    geometry: GroupDroGeometry
    construction_seed: StrictInt
    source_selection_position: NonNegativeInt
    background_selection_position: NonNegativeInt
    orientation: Literal["land_minus_water"]


class WaterbirdsConstructionCounts(StrictBoundaryModel):
    supervised_training: PositiveInt
    unpaired_majority: NonNegativeInt
    retained_majority_endpoints: PositiveInt
    generated_minority_endpoints: PositiveInt
    oracle_relationships: PositiveInt
    landbird_relationships: PositiveInt
    waterbird_relationships: PositiveInt
    validation: PositiveInt
    test: PositiveInt
    training_groups: WaterbirdsGroupCounts

    @model_validator(mode="after")
    def _validate_equations(self) -> WaterbirdsConstructionCounts:
        if self.retained_majority_endpoints != self.oracle_relationships:
            raise ValueError("every oracle relationship needs one majority endpoint")
        if self.generated_minority_endpoints != self.oracle_relationships:
            raise ValueError("every oracle relationship needs one generated endpoint")
        if (
            self.landbird_relationships + self.waterbird_relationships
            != self.oracle_relationships
        ):
            raise ValueError("relationship label strata must sum to pair count")
        if (
            self.unpaired_majority
            + self.retained_majority_endpoints
            + self.generated_minority_endpoints
            != self.supervised_training
        ):
            raise ValueError("Waterbirds-CF training components do not sum to total")
        if self.training_groups.total != self.supervised_training:
            raise ValueError("Waterbirds-CF training groups do not sum to total")
        return self


class WaterbirdsDatasetManifest(StrictBoundaryModel):
    schema_version: Literal["grit.waterbirds-cf-dataset/v2"]
    dataset_id: Literal["waterbirds_cf"]
    profile_kind: Literal["production", "fixture"]
    non_reportable: StrictBool
    base_artifact_name: NonEmptyStr
    construction_method_id: Literal["waterbirds-cf-sha256-v1"]
    compositor_id: Literal["groupdro-center-crop-lanczos-v1"]
    generated_encoding_id: Literal["png-rgb-v1"]
    construction_seed: StrictInt
    source_bundle_digest: NonEmptyStr
    released_metadata_file_sha256: NonEmptyStr
    released_metadata_rows_digest: NonEmptyStr
    counts: WaterbirdsConstructionCounts
    records: tuple[WaterbirdsRecord, ...]
    relationships: tuple[WaterbirdsOracleRelationship, ...]
    replaced_released_record_ids: tuple[NonEmptyStr, ...]
    training_membership_digest: NonEmptyStr
    validation_membership_digest: NonEmptyStr
    test_membership_digest: NonEmptyStr
    released_validation_bytes_digest: NonEmptyStr
    released_test_bytes_digest: NonEmptyStr

    @model_validator(mode="after")
    def _validate_manifest(self) -> WaterbirdsDatasetManifest:
        if self.profile_kind == "production":
            expected_production = WaterbirdsConstructionCounts(
                supervised_training=4_795,
                unpaired_majority=4_315,
                retained_majority_endpoints=240,
                generated_minority_endpoints=240,
                oracle_relationships=240,
                landbird_relationships=184,
                waterbird_relationships=56,
                validation=1_199,
                test=5_794,
                training_groups=PRODUCTION_TRAIN_GROUP_COUNTS,
            )
            if (
                self.non_reportable
                or self.base_artifact_name != BASE_ARTIFACT_NAME
                or self.counts != expected_production
            ):
                raise ValueError(
                    "production Waterbirds manifest requires the canonical inventory"
                )
        elif not self.non_reportable:
            raise ValueError("fixture Waterbirds manifests must be non-reportable")
        record_ids = tuple(record.record_id for record in self.records)
        if len(record_ids) != len(set(record_ids)):
            raise ValueError("Waterbirds-CF record IDs must be unique")
        training = tuple(
            record for record in self.records if record.split_role == "training"
        )
        validation = tuple(
            record for record in self.records if record.split_role == "validation"
        )
        test = tuple(
            record for record in self.records if record.split_role == "final_test"
        )
        training_sources = {record.source_cub_image_id for record in training}
        validation_sources = {record.source_cub_image_id for record in validation}
        test_sources = {record.source_cub_image_id for record in test}
        if (
            training_sources & validation_sources
            or training_sources & test_sources
            or validation_sources & test_sources
        ):
            raise ValueError("Waterbirds split source identities must be disjoint")
        if (len(training), len(validation), len(test)) != (
            self.counts.supervised_training,
            self.counts.validation,
            self.counts.test,
        ):
            raise ValueError("Waterbirds-CF split counts are inconsistent")
        observed_groups = waterbirds_group_counts(training)
        if observed_groups != self.counts.training_groups:
            raise ValueError("Waterbirds-CF training group counts are inconsistent")
        component_counts = {
            component: sum(record.component == component for record in training)
            for component in (
                "unpaired_majority",
                "majority_endpoint",
                "generated_minority",
            )
        }
        if component_counts != {
            "unpaired_majority": self.counts.unpaired_majority,
            "majority_endpoint": self.counts.retained_majority_endpoints,
            "generated_minority": self.counts.generated_minority_endpoints,
        }:
            raise ValueError("Waterbirds-CF component counts are inconsistent")
        if len(self.relationships) != self.counts.oracle_relationships:
            raise ValueError("Waterbirds-CF oracle relationship count is inconsistent")
        if len(self.replaced_released_record_ids) != self.counts.oracle_relationships:
            raise ValueError("Waterbirds-CF replacement count is inconsistent")
        if len(set(self.replaced_released_record_ids)) != len(
            self.replaced_released_record_ids
        ):
            raise ValueError("replaced released Waterbirds record IDs must be unique")
        if set(self.replaced_released_record_ids) & set(record_ids):
            raise ValueError(
                "replaced released Waterbirds record IDs must be absent from current "
                "records"
            )
        training_by_id = {record.record_id: record for record in training}
        majority_ids: set[str] = set()
        generated_ids: set[str] = set()
        source_positions: set[int] = set()
        background_positions: set[int] = set()
        label_counts = {0: 0, 1: 0}
        for relationship in self.relationships:
            land = training_by_id.get(relationship.land_record_id)
            water = training_by_id.get(relationship.water_record_id)
            majority = training_by_id.get(relationship.majority_record_id)
            generated = training_by_id.get(relationship.generated_record_id)
            if any(item is None for item in (land, water, majority, generated)):
                raise ValueError("oracle relationship endpoint is outside training")
            if land is None or water is None or majority is None or generated is None:
                raise AssertionError("relationship endpoint narrowing failed")
            if land.background != 0 or water.background != 1:
                raise ValueError("oracle relationships must be land minus water")
            expected_pair_id = canonical_digest_value(
                {
                    "method": self.construction_method_id,
                    "source_bundle_digest": self.source_bundle_digest,
                    "construction_seed": self.construction_seed,
                    "majority_record_id": relationship.majority_record_id,
                    "generated_record_id": relationship.generated_record_id,
                    "orientation": "land_minus_water",
                }
            )
            if relationship.pair_id != expected_pair_id:
                raise ValueError("oracle relationship pair ID is inconsistent")
            if (
                land.record_id != relationship.land_record_id
                or water.record_id != relationship.water_record_id
                or majority.record_id != relationship.majority_record_id
                or generated.record_id != relationship.generated_record_id
                or land.pair_id != relationship.pair_id
                or water.pair_id != relationship.pair_id
                or land.endpoint_role != "land"
                or water.endpoint_role != "water"
            ):
                raise ValueError(
                    "oracle relationship endpoint identity is inconsistent"
                )
            if {
                land.source_cub_image_id,
                water.source_cub_image_id,
                majority.source_cub_image_id,
                generated.source_cub_image_id,
            } != {relationship.source_cub_image_id}:
                raise ValueError("oracle relationship source identity is inconsistent")
            if {land.bird_label, water.bird_label} != {relationship.bird_label}:
                raise ValueError("oracle relationship label is inconsistent")
            if (
                land.image_sha256 != relationship.land_image_sha256
                or water.image_sha256 != relationship.water_image_sha256
                or land.source_image_sha256 != relationship.source_image_sha256
                or water.source_image_sha256 != relationship.source_image_sha256
                or land.mask_sha256 != relationship.mask_sha256
                or water.mask_sha256 != relationship.mask_sha256
                or land.canonical_masked_source_foreground_digest
                != relationship.canonical_masked_source_foreground_digest
                or water.canonical_masked_source_foreground_digest
                != relationship.canonical_masked_source_foreground_digest
                or land.geometry != relationship.geometry
                or water.geometry != relationship.geometry
                or land.bounding_box_xywh != water.bounding_box_xywh
            ):
                raise ValueError(
                    "oracle relationship foreground geometry is inconsistent"
                )
            if (
                land.background_asset_id != relationship.land_background_asset_id
                or water.background_asset_id != relationship.water_background_asset_id
                or relationship.construction_input_digest != self.source_bundle_digest
                or relationship.construction_seed != self.construction_seed
            ):
                raise ValueError(
                    "oracle relationship construction provenance is inconsistent"
                )
            if majority.component != "majority_endpoint":
                raise ValueError("oracle majority endpoint has the wrong component")
            if generated.component != "generated_minority":
                raise ValueError("oracle generated endpoint has the wrong component")
            expected_generated_id = "waterbirds:generated:" + canonical_digest_value(
                {
                    "method": self.construction_method_id,
                    "source_bundle_digest": self.source_bundle_digest,
                    "construction_seed": self.construction_seed,
                    "source_cub_image_id": relationship.source_cub_image_id,
                    "background_asset_id": generated.background_asset_id,
                    "target_background": int(generated.background),
                    "selection_position": generated.selection_position,
                }
            ).removeprefix("sha256:")
            if generated.record_id != expected_generated_id:
                raise ValueError("generated Waterbirds record ID is inconsistent")
            if (
                majority.selection_position
                != relationship.source_selection_position
                or generated.selection_position
                != relationship.source_selection_position
                or relationship.source_selection_position
                != relationship.background_selection_position
            ):
                raise ValueError(
                    "oracle relationship selection position is inconsistent"
                )
            majority_ids.add(majority.record_id)
            generated_ids.add(generated.record_id)
            source_positions.add(relationship.source_selection_position)
            background_positions.add(relationship.background_selection_position)
            label_counts[int(relationship.bird_label)] += 1
        if len(majority_ids) != len(self.relationships) or len(generated_ids) != len(
            self.relationships
        ):
            raise ValueError("oracle relationship endpoints must be unique")
        expected_positions = set(range(len(self.relationships)))
        if (
            source_positions != expected_positions
            or background_positions != expected_positions
        ):
            raise ValueError(
                "oracle relationship positions must be contiguous and unique"
            )
        if label_counts != {
            0: self.counts.landbird_relationships,
            1: self.counts.waterbird_relationships,
        }:
            raise ValueError("oracle relationship label counts are inconsistent")
        memberships = (
            (training, self.training_membership_digest),
            (validation, self.validation_membership_digest),
            (test, self.test_membership_digest),
        )
        for records, digest in memberships:
            expected = canonical_digest_value(
                tuple(record.record_id for record in records)
            )
            if digest != expected:
                raise ValueError(
                    "Waterbirds-CF split membership digest is inconsistent"
                )
        if self.released_validation_bytes_digest != canonical_digest_value(
            tuple((record.record_id, record.image_sha256) for record in validation)
        ):
            raise ValueError("Waterbirds validation byte digest is inconsistent")
        if self.released_test_bytes_digest != canonical_digest_value(
            tuple((record.record_id, record.image_sha256) for record in test)
        ):
            raise ValueError("Waterbirds test byte digest is inconsistent")
        return self


@dataclass(frozen=True, slots=True)
class WaterbirdsConstruction:
    manifest: WaterbirdsDatasetManifest
    image_paths: dict[str, Path]

    def training_records(self) -> tuple[WaterbirdsRecord, ...]:
        return tuple(
            record
            for record in self.manifest.records
            if record.split_role == "training"
        )

    def validation_records(self) -> tuple[WaterbirdsRecord, ...]:
        return tuple(
            record
            for record in self.manifest.records
            if record.split_role == "validation"
        )

    def final_test_records(self) -> tuple[WaterbirdsRecord, ...]:
        return tuple(
            record
            for record in self.manifest.records
            if record.split_role == "final_test"
        )

    def path_for(self, record_id: str) -> Path:
        try:
            return self.image_paths[record_id]
        except KeyError as error:
            raise KeyError(
                f"Waterbirds image path is unavailable: {record_id}"
            ) from error


class WaterbirdsSupervisedRecord(StrictBoundaryModel):
    """Training metadata deliberately stripped of background and pair identity."""

    record_id: NonEmptyStr
    bird_label: BinaryInt
    image_sha256: NonEmptyStr


class WaterbirdsEvaluationRecord(StrictBoundaryModel):
    """Group-bearing metadata available only to validation/final evaluators."""

    record_id: NonEmptyStr
    bird_label: BinaryInt
    background: BinaryInt
    group_id: GroupId
    image_sha256: NonEmptyStr


class WaterbirdsTrainingView:
    """Role-scoped supervised training capability with oracle metadata redacted."""

    __slots__ = ("__dataset_manifest_digest", "__paths", "__records")

    def __init__(
        self,
        *,
        dataset_manifest_digest: str,
        records: tuple[WaterbirdsSupervisedRecord, ...],
        paths: dict[str, Path],
    ) -> None:
        self.__dataset_manifest_digest = dataset_manifest_digest
        self.__records = records
        self.__paths = dict(paths)

    @property
    def dataset_manifest_digest(self) -> str:
        return self.__dataset_manifest_digest

    @property
    def records(self) -> tuple[WaterbirdsSupervisedRecord, ...]:
        return self.__records

    def path_for(self, record_id: str) -> Path:
        try:
            return self.__paths[record_id]
        except KeyError as error:
            raise KeyError(
                f"record is outside the Waterbirds training view: {record_id}"
            ) from error


class WaterbirdsValidationView:
    """Role-scoped validation capability with approved four-group metadata."""

    __slots__ = ("__dataset_manifest_digest", "__paths", "__records")

    def __init__(
        self,
        *,
        dataset_manifest_digest: str,
        records: tuple[WaterbirdsEvaluationRecord, ...],
        paths: dict[str, Path],
    ) -> None:
        self.__dataset_manifest_digest = dataset_manifest_digest
        self.__records = records
        self.__paths = dict(paths)

    @property
    def dataset_manifest_digest(self) -> str:
        return self.__dataset_manifest_digest

    @property
    def records(self) -> tuple[WaterbirdsEvaluationRecord, ...]:
        return self.__records

    def path_for(self, record_id: str) -> Path:
        try:
            return self.__paths[record_id]
        except KeyError as error:
            raise KeyError(
                f"record is outside the Waterbirds validation view: {record_id}"
            ) from error


class WaterbirdsOracleRelationView:
    """Training-only oracle relation capability, issued separately from endpoints."""

    __slots__ = ("__dataset_manifest_json",)

    def __init__(self, *, dataset_manifest_json: str) -> None:
        self.__dataset_manifest_json = dataset_manifest_json

    def validated_manifest(self) -> WaterbirdsDatasetManifest:
        return WaterbirdsDatasetManifest.model_validate_json(
            self.__dataset_manifest_json
        )

    @property
    def dataset_manifest_digest(self) -> str:
        return self.validated_manifest().canonical_digest()

    @property
    def relationships(self) -> tuple[WaterbirdsOracleRelationship, ...]:
        return self.validated_manifest().relationships


def waterbirds_training_view(
    construction: WaterbirdsConstruction,
) -> WaterbirdsTrainingView:
    """Issue common ERM/GRIT supervision without backgrounds or relations."""

    manifest = validate_waterbirds_construction(construction)
    records = tuple(
        WaterbirdsSupervisedRecord(
            record_id=record.record_id,
            bird_label=record.bird_label,
            image_sha256=record.image_sha256,
        )
        for record in manifest.records
        if record.split_role == "training"
    )
    return WaterbirdsTrainingView(
        dataset_manifest_digest=manifest.canonical_digest(),
        records=records,
        paths={
            record.record_id: construction.path_for(record.record_id)
            for record in records
        },
    )


def waterbirds_validation_view(
    construction: WaterbirdsConstruction,
) -> WaterbirdsValidationView:
    """Issue validation-only group-bearing records."""

    manifest = validate_waterbirds_construction(construction)
    records = tuple(
        WaterbirdsEvaluationRecord(
            record_id=record.record_id,
            bird_label=record.bird_label,
            background=record.background,
            group_id=record.group_id,
            image_sha256=record.image_sha256,
        )
        for record in manifest.records
        if record.split_role == "validation"
    )
    return WaterbirdsValidationView(
        dataset_manifest_digest=manifest.canonical_digest(),
        records=records,
        paths={
            record.record_id: construction.path_for(record.record_id)
            for record in records
        },
    )


def waterbirds_oracle_relation_view(
    construction: WaterbirdsConstruction,
) -> WaterbirdsOracleRelationView:
    """Issue exact pair identities only to the explicit oracle path."""

    manifest = validate_waterbirds_construction(construction)
    return WaterbirdsOracleRelationView(dataset_manifest_json=manifest.canonical_json())


def validate_waterbirds_construction(
    construction: WaterbirdsConstruction,
) -> WaterbirdsDatasetManifest:
    if type(construction) is not WaterbirdsConstruction:
        raise TypeError("Waterbirds views require a WaterbirdsConstruction")
    manifest = WaterbirdsDatasetManifest.model_validate_json(
        construction.manifest.canonical_json()
    )
    expected_ids = {record.record_id for record in manifest.records}
    if set(construction.image_paths) != expected_ids:
        raise ValueError("Waterbirds construction paths do not match manifest records")
    for record in manifest.records:
        path = construction.path_for(record.record_id)
        if not path.is_file() or _file_sha256(path) != record.image_sha256:
            raise ValueError(
                f"Waterbirds construction image identity is invalid: {record.record_id}"
            )
    return manifest


def mint_waterbirds_adjusted_weight_spec(
    manifest: WaterbirdsDatasetManifest,
) -> WaterbirdsAdjustedWeightSpec:
    """Mint adjusted weights only from a fully revalidated dataset manifest."""

    validated = WaterbirdsDatasetManifest.model_validate_json(manifest.canonical_json())
    return WaterbirdsAdjustedWeightSpec(
        schema_version="grit.waterbirds-adjusted-weights/v1",
        group_order=WATERBIRDS_GROUP_ORDER,
        training_group_counts=validated.counts.training_groups,
        dataset_manifest_digest=validated.canonical_digest(),
    )


def waterbirds_group_id(label: int, background: int) -> GroupId:
    mapping: dict[tuple[int, int], GroupId] = {
        (0, 0): "landbird_land",
        (0, 1): "landbird_water",
        (1, 0): "waterbird_land",
        (1, 1): "waterbird_water",
    }
    try:
        return mapping[(label, background)]
    except KeyError as error:
        raise ValueError("Waterbirds label/background values must be binary") from error


def waterbirds_group_counts(
    records: tuple[WaterbirdsRecord, ...],
) -> WaterbirdsGroupCounts:
    counts: dict[GroupId, int] = {
        "landbird_land": 0,
        "landbird_water": 0,
        "waterbird_land": 0,
        "waterbird_water": 0,
    }
    for record in records:
        counts[record.group_id] += 1
    return WaterbirdsGroupCounts(
        landbird_land=counts["landbird_land"],
        landbird_water=counts["landbird_water"],
        waterbird_land=counts["waterbird_land"],
        waterbird_water=counts["waterbird_water"],
    )


def parse_released_waterbirds(
    root: Path,
    *,
    artifact_name: str,
) -> ReleasedWaterbirdsManifest:
    metadata_path = root / "metadata.csv"
    if not metadata_path.is_file():
        raise FileNotFoundError(
            f"released Waterbirds metadata is missing: {metadata_path}"
        )
    records: list[ReleasedWaterbirdsRecord] = []
    with metadata_path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        required = {"img_id", "img_filename", "y", "place", "split"}
        if reader.fieldnames is None or not required <= set(reader.fieldnames):
            raise ValueError(
                "released Waterbirds metadata requires img_id, img_filename, y, "
                "place, split"
            )
        for row in reader:
            image_id = _strict_csv_int(row, "img_id")
            relative_path = _strict_csv_text(row, "img_filename")
            label = _binary_csv_int(row, "y")
            background = _binary_csv_int(row, "place")
            split_value = _strict_csv_int(row, "split")
            split_roles: dict[int, SplitRole] = {
                0: "training",
                1: "validation",
                2: "final_test",
            }
            if split_value not in split_roles:
                raise ValueError("released Waterbirds split must be 0, 1, or 2")
            image_path = root / relative_path
            if not image_path.is_file():
                raise FileNotFoundError(
                    f"released Waterbirds image is missing: {image_path}"
                )
            place_path = row.get("place_filename")
            place_relative_path = place_path.strip() if place_path else None
            records.append(
                ReleasedWaterbirdsRecord(
                    record_id=f"waterbirds:released:{image_id}",
                    cub_image_id=image_id,
                    image_relative_path=relative_path,
                    bird_label=label,
                    background=background,
                    split_role=split_roles[split_value],
                    place_relative_path=place_relative_path,
                    image_sha256=_file_sha256(image_path),
                )
            )
    ordered = tuple(sorted(records, key=lambda record: record.record_id))
    rows_digest = canonical_digest_value(
        tuple(record.model_dump(mode="json") for record in ordered)
    )
    return ReleasedWaterbirdsManifest(
        schema_version="grit.waterbirds-released/v1",
        artifact_name=artifact_name,
        metadata_file_sha256=_file_sha256(metadata_path),
        metadata_rows_digest=rows_digest,
        records=ordered,
    )


def parse_cub_sources(
    cub_root: Path,
    masks_root: Path,
) -> CubSourceManifest:
    images_path = cub_root / "images.txt"
    labels_path = cub_root / "image_class_labels.txt"
    classes_path = cub_root / "classes.txt"
    boxes_path = cub_root / "bounding_boxes.txt"
    for path in (images_path, labels_path, classes_path, boxes_path):
        if not path.is_file():
            raise FileNotFoundError(f"required CUB annotation is missing: {path}")
    image_names = _read_indexed_text(images_path)
    class_ids = {
        key: int(value) for key, value in _read_indexed_text(labels_path).items()
    }
    species = _read_indexed_text(classes_path)
    boxes = _read_bounding_boxes(boxes_path)
    if set(image_names) != set(class_ids) or set(image_names) != set(boxes):
        raise ValueError("CUB image, class-label, and bounding-box IDs must align")
    records: list[CubSourceRecord] = []
    for image_id in sorted(image_names):
        relative_path = image_names[image_id]
        image_path = cub_root / "images" / relative_path
        mask_relative = str(Path(relative_path).with_suffix(".png"))
        mask_path = masks_root / mask_relative
        if not image_path.is_file():
            raise FileNotFoundError(f"CUB image is missing: {image_path}")
        if not mask_path.is_file():
            raise FileNotFoundError(f"CUB segmentation mask is missing: {mask_path}")
        class_id = class_ids[image_id]
        if class_id not in species:
            raise ValueError(f"CUB class ID {class_id} is missing from classes.txt")
        records.append(
            CubSourceRecord(
                cub_image_id=image_id,
                image_relative_path=relative_path,
                mask_relative_path=mask_relative,
                class_id=class_id,
                species=species[class_id],
                bounding_box_xywh=boxes[image_id],
                image_sha256=_file_sha256(image_path),
                mask_sha256=_file_sha256(mask_path),
            )
        )
    ordered = tuple(records)
    digest = canonical_digest_value(
        tuple(record.model_dump(mode="json") for record in ordered)
    )
    return CubSourceManifest(
        schema_version="grit.cub-sources/v1",
        images_file_sha256=_file_sha256(images_path),
        labels_file_sha256=_file_sha256(labels_path),
        classes_file_sha256=_file_sha256(classes_path),
        bounding_boxes_file_sha256=_file_sha256(boxes_path),
        records_digest=digest,
        records=ordered,
    )


def parse_places_backgrounds(root: Path) -> PlacesSourceManifest:
    records: list[PlacesBackgroundRecord] = []
    for category in APPROVED_PLACE_CATEGORIES:
        background: Background = (
            0
            if category
            in {
                "bamboo_forest",
                "forest/broadleaf",
            }
            else 1
        )
        directory = places_category_directory(root, category)
        files = tuple(
            sorted(
                path
                for path in directory.iterdir()
                if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png"}
            )
        )
        if not files:
            raise FileNotFoundError(
                f"approved Places category contains no images: {directory}"
            )
        for path in files:
            relative_path = path.relative_to(root).as_posix()
            image_sha256 = _file_sha256(path)
            records.append(
                PlacesBackgroundRecord(
                    asset_id=canonical_digest_value(
                        {
                            "category": category,
                            "relative_path": relative_path,
                            "image_sha256": image_sha256,
                        }
                    ),
                    category=category,
                    background=background,
                    relative_path=relative_path,
                    image_sha256=image_sha256,
                )
            )
    ordered = tuple(sorted(records, key=lambda record: record.asset_id))
    digest = canonical_digest_value(
        tuple(record.model_dump(mode="json") for record in ordered)
    )
    return PlacesSourceManifest(
        schema_version="grit.places-backgrounds/v1",
        categories=APPROVED_PLACE_CATEGORIES,
        records_digest=digest,
        records=ordered,
    )


def load_waterbirds_assets(
    *,
    released_root: Path,
    cub_root: Path,
    masks_root: Path,
    places_root: Path,
    artifact_name: str,
) -> ParsedWaterbirdsAssets:
    return ParsedWaterbirdsAssets(
        released_root=released_root,
        cub_root=cub_root,
        masks_root=masks_root,
        places_root=places_root,
        released=parse_released_waterbirds(released_root, artifact_name=artifact_name),
        cub=parse_cub_sources(cub_root, masks_root),
        places=parse_places_backgrounds(places_root),
    )


def crop_and_resize_groupdro(
    source: Image.Image,
    target_size: tuple[int, int],
) -> Image.Image:
    """Reproduce GroupDRO's recursive center-crop and ANTIALIAS resize geometry."""

    source_rgb = source.convert("RGB")
    source_width, source_height = source_rgb.size
    target_width, target_height = target_size
    if min(source_width, source_height, target_width, target_height) <= 0:
        raise ValueError("source and target image dimensions must be positive")
    if source_width < target_width or source_height < target_height:
        width_resize = (
            target_width,
            int((target_width / source_width) * source_height),
        )
        if width_resize[1] >= target_height:
            resized = source_rgb.resize(width_resize, Image.Resampling.LANCZOS)
        else:
            height_resize = (
                int((target_height / source_height) * source_width),
                target_height,
            )
            if height_resize[0] < target_width:
                raise AssertionError("GroupDRO resize failed to cover target")
            resized = source_rgb.resize(height_resize, Image.Resampling.LANCZOS)
        return crop_and_resize_groupdro(resized, target_size)
    source_aspect = source_width / source_height
    target_aspect = target_width / target_height
    if source_aspect > target_aspect:
        new_width = int(target_aspect * source_height)
        offset = (source_width - new_width) // 2
        crop_box = (offset, 0, source_width - offset, source_height)
    else:
        new_height = int(source_width / target_aspect)
        offset = (source_height - new_height) // 2
        crop_box = (0, offset, source_width, source_height - offset)
    return source_rgb.crop(crop_box).resize(
        target_size,
        Image.Resampling.LANCZOS,
    )


def composite_groupdro(
    foreground: Image.Image,
    mask: Image.Image,
    background: Image.Image,
) -> tuple[Image.Image, GroupDroGeometry, str]:
    """Apply the official GroupDRO mask/composite operation with recorded geometry."""

    foreground_rgb = foreground.convert("RGB")
    mask_rgb = mask.convert("RGB")
    if mask_rgb.size != foreground_rgb.size:
        raise ValueError("CUB foreground and segmentation mask sizes must match")
    foreground_array = np.asarray(foreground_rgb, dtype=np.uint8)
    mask_array = np.asarray(mask_rgb, dtype=np.float64) / 255.0
    resized_background = crop_and_resize_groupdro(background, foreground_rgb.size)
    background_array = np.asarray(resized_background, dtype=np.uint8)
    foreground_masked = np.rint(foreground_array * mask_array).astype(np.uint8)
    background_masked = np.rint(background_array * (1.0 - mask_array)).astype(np.uint8)
    combined = foreground_masked + background_masked
    geometry = GroupDroGeometry(
        compositor_id=COMPOSITOR_ID,
        generated_encoding_id=GENERATED_ENCODING_ID,
        width=foreground_rgb.size[0],
        height=foreground_rgb.size[1],
        resampling="pillow_lanczos",
    )
    foreground_digest = _array_sha256(foreground_masked)
    return Image.fromarray(combined, mode="RGB"), geometry, foreground_digest


def construct_waterbirds_cf(
    assets: ParsedWaterbirdsAssets,
    profile: WaterbirdsConstructionProfile,
    output_root: Path,
) -> WaterbirdsConstruction:
    """Construct a manifest overlay and only the controlled generated endpoints."""

    requirements = _requirements(profile)
    if assets.released.artifact_name != requirements.base_artifact_name:
        raise ValueError("released Waterbirds artifact name does not match the profile")
    source_bundle_digest = assets.source_bundle_digest()
    released = tuple(
        sorted(assets.released.records, key=lambda record: record.record_id)
    )
    cub_by_id = {record.cub_image_id: record for record in assets.cub.records}
    for record in released:
        cub = cub_by_id.get(record.cub_image_id)
        if cub is None:
            raise ValueError("released Waterbirds record lacks its CUB source")
        if cub.image_relative_path != record.image_relative_path:
            raise ValueError("released Waterbirds and CUB image paths disagree")
    training_released = tuple(
        record for record in released if record.split_role == "training"
    )
    observed_groups = _released_group_counts(training_released)
    if observed_groups != requirements.training_groups:
        raise ValueError(
            "released Waterbirds training group counts do not match profile"
        )
    validation_released = tuple(
        record for record in released if record.split_role == "validation"
    )
    test_released = tuple(
        record for record in released if record.split_role == "final_test"
    )
    if len(validation_released) != requirements.validation_count:
        raise ValueError("released Waterbirds validation count does not match profile")
    if len(test_released) != requirements.test_count:
        raise ValueError("released Waterbirds test count does not match profile")

    land_majority = tuple(
        record
        for record in training_released
        if record.bird_label == 0 and record.background == 0
    )
    water_majority = tuple(
        record
        for record in training_released
        if record.bird_label == 1 and record.background == 1
    )
    minority = tuple(
        record for record in training_released if record.bird_label != record.background
    )
    selected_land = _select_records(
        land_majority,
        count=requirements.landbird_pair_count,
        seed=requirements.construction_seed,
        namespace="landbird-majority",
    )
    selected_water = _select_records(
        water_majority,
        count=requirements.waterbird_pair_count,
        seed=requirements.construction_seed,
        namespace="waterbird-majority",
    )
    selected_majority = (*selected_land, *selected_water)
    land_backgrounds = _select_backgrounds(
        assets.places.records,
        target_background=0,
        count=requirements.waterbird_pair_count,
        seed=requirements.construction_seed,
    )
    water_backgrounds = _select_backgrounds(
        assets.places.records,
        target_background=1,
        count=requirements.landbird_pair_count,
        seed=requirements.construction_seed,
    )

    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError(f"Waterbirds-CF output is not empty: {output_root}")
    generated_root = output_root / "generated"
    generated_root.mkdir(parents=True, exist_ok=True)

    selected_backgrounds = (*water_backgrounds, *land_backgrounds)
    relationships: list[WaterbirdsOracleRelationship] = []
    generated_records: list[WaterbirdsRecord] = []
    pair_by_majority: dict[str, tuple[str, EndpointRole, int]] = {}
    generated_paths: dict[str, Path] = {}
    for position, (majority, background) in enumerate(
        zip(selected_majority, selected_backgrounds, strict=True)
    ):
        cub = cub_by_id[majority.cub_image_id]
        source_path = assets.cub_root / "images" / cub.image_relative_path
        mask_path = assets.masks_root / cub.mask_relative_path
        background_path = assets.places_root / background.relative_path
        with (
            Image.open(source_path) as source_image,
            Image.open(mask_path) as mask_image,
            Image.open(background_path) as background_image,
        ):
            generated_image, geometry, foreground_digest = composite_groupdro(
                source_image,
                mask_image,
                background_image,
            )
        generated_id = "waterbirds:generated:" + canonical_digest_value(
            {
                "method": CONSTRUCTION_METHOD_ID,
                "source_bundle_digest": source_bundle_digest,
                "construction_seed": requirements.construction_seed,
                "source_cub_image_id": majority.cub_image_id,
                "background_asset_id": background.asset_id,
                "target_background": int(background.background),
                "selection_position": position,
            }
        ).removeprefix("sha256:")
        pair_id = canonical_digest_value(
            {
                "method": CONSTRUCTION_METHOD_ID,
                "source_bundle_digest": source_bundle_digest,
                "construction_seed": requirements.construction_seed,
                "majority_record_id": majority.record_id,
                "generated_record_id": generated_id,
                "orientation": "land_minus_water",
            }
        )
        relative_generated = (
            f"generated/{generated_id.removeprefix('waterbirds:generated:')}.png"
        )
        generated_path = output_root / relative_generated
        generated_image.save(generated_path, format="PNG")
        generated_hash = _file_sha256(generated_path)
        target_background = cast(Background, 1 - int(majority.background))
        generated_endpoint: EndpointRole = "land" if target_background == 0 else "water"
        majority_endpoint: EndpointRole = (
            "land" if majority.background == 0 else "water"
        )
        generated_record = WaterbirdsRecord(
            record_id=generated_id,
            split_role="training",
            component="generated_minority",
            bird_label=majority.bird_label,
            background=target_background,
            group_id=waterbirds_group_id(
                int(majority.bird_label), int(target_background)
            ),
            source_cub_image_id=majority.cub_image_id,
            species=cub.species,
            image_location="generated",
            image_relative_path=relative_generated,
            image_sha256=generated_hash,
            source_image_sha256=cub.image_sha256,
            mask_sha256=cub.mask_sha256,
            canonical_masked_source_foreground_digest=foreground_digest,
            bounding_box_xywh=cub.bounding_box_xywh,
            pair_id=pair_id,
            endpoint_role=generated_endpoint,
            background_asset_id=background.asset_id,
            background_asset_sha256=background.image_sha256,
            construction_seed=requirements.construction_seed,
            selection_position=position,
            geometry=geometry,
        )
        generated_records.append(generated_record)
        generated_paths[generated_id] = generated_path
        pair_by_majority[majority.record_id] = (
            pair_id,
            majority_endpoint,
            position,
        )

        majority_background_id = (
            f"released-place:{majority.place_relative_path}"
            if majority.place_relative_path
            else f"released-background:{majority.record_id}"
        )
        land_record_id = (
            majority.record_id if majority.background == 0 else generated_id
        )
        water_record_id = (
            majority.record_id if majority.background == 1 else generated_id
        )
        land_hash = (
            majority.image_sha256 if majority.background == 0 else generated_hash
        )
        water_hash = (
            majority.image_sha256 if majority.background == 1 else generated_hash
        )
        land_background_id = (
            majority_background_id if majority.background == 0 else background.asset_id
        )
        water_background_id = (
            majority_background_id if majority.background == 1 else background.asset_id
        )
        relationships.append(
            WaterbirdsOracleRelationship(
                pair_id=pair_id,
                construction_input_digest=source_bundle_digest,
                source_cub_image_id=majority.cub_image_id,
                bird_label=majority.bird_label,
                land_record_id=land_record_id,
                water_record_id=water_record_id,
                majority_record_id=majority.record_id,
                generated_record_id=generated_id,
                land_image_sha256=land_hash,
                water_image_sha256=water_hash,
                source_image_sha256=cub.image_sha256,
                mask_sha256=cub.mask_sha256,
                canonical_masked_source_foreground_digest=foreground_digest,
                land_background_asset_id=land_background_id,
                water_background_asset_id=water_background_id,
                geometry=geometry,
                construction_seed=requirements.construction_seed,
                source_selection_position=position,
                background_selection_position=position,
                orientation="land_minus_water",
            )
        )

    selected_majority_ids = set(pair_by_majority)
    minority_ids = {record.record_id for record in minority}
    records: list[WaterbirdsRecord] = []
    paths: dict[str, Path] = {}
    for released_record in released:
        if released_record.record_id in minority_ids:
            continue
        cub = cub_by_id[released_record.cub_image_id]
        image_path = assets.released_root / released_record.image_relative_path
        with (
            Image.open(assets.cub_root / "images" / cub.image_relative_path) as source,
            Image.open(assets.masks_root / cub.mask_relative_path) as mask,
        ):
            source_rgb = source.convert("RGB")
            mask_rgb = mask.convert("RGB")
            if source_rgb.size != mask_rgb.size:
                raise ValueError("CUB source and mask dimensions must match")
            source_array = np.asarray(source_rgb, dtype=np.uint8)
            mask_array = np.asarray(mask_rgb, dtype=np.float64) / 255.0
            foreground_digest = _array_sha256(
                np.rint(source_array * mask_array).astype(np.uint8)
            )
            geometry = GroupDroGeometry(
                compositor_id=COMPOSITOR_ID,
                generated_encoding_id=GENERATED_ENCODING_ID,
                width=source_rgb.size[0],
                height=source_rgb.size[1],
                resampling="pillow_lanczos",
            )
        pair_data: tuple[str, EndpointRole, int] | None = None
        if released_record.split_role == "training":
            pair_data = pair_by_majority.get(released_record.record_id)
            component: Component = (
                "majority_endpoint" if pair_data is not None else "unpaired_majority"
            )
            pair_id = pair_data[0] if pair_data else None
            endpoint_role = pair_data[1] if pair_data else None
        elif released_record.split_role == "validation":
            component = "released_validation"
            pair_id = None
            endpoint_role = None
        else:
            component = "released_test"
            pair_id = None
            endpoint_role = None
        records.append(
            WaterbirdsRecord(
                record_id=released_record.record_id,
                split_role=released_record.split_role,
                component=component,
                bird_label=released_record.bird_label,
                background=released_record.background,
                group_id=waterbirds_group_id(
                    int(released_record.bird_label),
                    int(released_record.background),
                ),
                source_cub_image_id=released_record.cub_image_id,
                species=cub.species,
                image_location="released",
                image_relative_path=released_record.image_relative_path,
                image_sha256=released_record.image_sha256,
                source_image_sha256=cub.image_sha256,
                mask_sha256=cub.mask_sha256,
                canonical_masked_source_foreground_digest=foreground_digest,
                bounding_box_xywh=cub.bounding_box_xywh,
                pair_id=pair_id,
                endpoint_role=endpoint_role,
                background_asset_id=(
                    f"released-place:{released_record.place_relative_path}"
                    if released_record.place_relative_path
                    else f"released-background:{released_record.record_id}"
                ),
                background_asset_sha256=None,
                construction_seed=(
                    requirements.construction_seed if pair_data else None
                )
                if released_record.split_role == "training"
                else None,
                selection_position=pair_data[2] if pair_data is not None else None,
                geometry=geometry,
            )
        )
        paths[released_record.record_id] = image_path
    records.extend(generated_records)
    paths.update(generated_paths)
    ordered_records = tuple(
        sorted(
            records,
            key=lambda record: (
                {"training": 0, "validation": 1, "final_test": 2}[record.split_role],
                record.record_id,
            ),
        )
    )
    training_records = tuple(
        record for record in ordered_records if record.split_role == "training"
    )
    validation_records = tuple(
        record for record in ordered_records if record.split_role == "validation"
    )
    test_records = tuple(
        record for record in ordered_records if record.split_role == "final_test"
    )
    counts = WaterbirdsConstructionCounts(
        supervised_training=len(training_records),
        unpaired_majority=sum(
            record.component == "unpaired_majority" for record in training_records
        ),
        retained_majority_endpoints=len(selected_majority_ids),
        generated_minority_endpoints=len(generated_records),
        oracle_relationships=len(relationships),
        landbird_relationships=len(selected_land),
        waterbird_relationships=len(selected_water),
        validation=len(validation_records),
        test=len(test_records),
        training_groups=waterbirds_group_counts(training_records),
    )
    if counts != requirements.expected_counts:
        raise ValueError("constructed Waterbirds-CF counts do not match the profile")
    manifest = WaterbirdsDatasetManifest(
        schema_version="grit.waterbirds-cf-dataset/v2",
        dataset_id="waterbirds_cf",
        profile_kind=requirements.profile_kind,
        non_reportable=requirements.non_reportable,
        base_artifact_name=requirements.base_artifact_name,
        construction_method_id=CONSTRUCTION_METHOD_ID,
        compositor_id=COMPOSITOR_ID,
        generated_encoding_id=GENERATED_ENCODING_ID,
        construction_seed=requirements.construction_seed,
        source_bundle_digest=source_bundle_digest,
        released_metadata_file_sha256=assets.released.metadata_file_sha256,
        released_metadata_rows_digest=assets.released.metadata_rows_digest,
        counts=counts,
        records=ordered_records,
        relationships=tuple(sorted(relationships, key=lambda item: item.pair_id)),
        replaced_released_record_ids=tuple(sorted(minority_ids)),
        training_membership_digest=canonical_digest_value(
            tuple(record.record_id for record in training_records)
        ),
        validation_membership_digest=canonical_digest_value(
            tuple(record.record_id for record in validation_records)
        ),
        test_membership_digest=canonical_digest_value(
            tuple(record.record_id for record in test_records)
        ),
        released_validation_bytes_digest=canonical_digest_value(
            tuple(
                (record.record_id, record.image_sha256) for record in validation_records
            )
        ),
        released_test_bytes_digest=canonical_digest_value(
            tuple((record.record_id, record.image_sha256) for record in test_records)
        ),
    )
    (output_root / "dataset-manifest.json").write_text(
        manifest.canonical_json() + "\n",
        encoding="utf-8",
    )
    return WaterbirdsConstruction(manifest=manifest, image_paths=paths)


@dataclass(frozen=True, slots=True)
class _ConstructionRequirements:
    profile_kind: Literal["production", "fixture"]
    non_reportable: bool
    base_artifact_name: str
    construction_seed: int
    training_groups: WaterbirdsGroupCounts
    landbird_pair_count: int
    waterbird_pair_count: int
    validation_count: int
    test_count: int
    expected_counts: WaterbirdsConstructionCounts


def _requirements(
    profile: WaterbirdsConstructionProfile,
) -> _ConstructionRequirements:
    if isinstance(profile, ProductionWaterbirdsProfile):
        training_groups = PRODUCTION_TRAIN_GROUP_COUNTS
        landbird_pairs = 184
        waterbird_pairs = 56
        validation_count = 1_199
        test_count = 5_794
        non_reportable = False
        profile_kind: Literal["production", "fixture"] = "production"
    else:
        training_groups = profile.train_group_counts
        landbird_pairs = profile.landbird_pair_count
        waterbird_pairs = profile.waterbird_pair_count
        validation_count = profile.validation_count
        test_count = profile.test_count
        non_reportable = True
        profile_kind = "fixture"
    relationships = landbird_pairs + waterbird_pairs
    expected_counts = WaterbirdsConstructionCounts(
        supervised_training=training_groups.total,
        unpaired_majority=(
            training_groups.landbird_land
            + training_groups.waterbird_water
            - relationships
        ),
        retained_majority_endpoints=relationships,
        generated_minority_endpoints=relationships,
        oracle_relationships=relationships,
        landbird_relationships=landbird_pairs,
        waterbird_relationships=waterbird_pairs,
        validation=validation_count,
        test=test_count,
        training_groups=training_groups,
    )
    return _ConstructionRequirements(
        profile_kind=profile_kind,
        non_reportable=non_reportable,
        base_artifact_name=profile.base_artifact_name,
        construction_seed=profile.construction_seed,
        training_groups=training_groups,
        landbird_pair_count=landbird_pairs,
        waterbird_pair_count=waterbird_pairs,
        validation_count=validation_count,
        test_count=test_count,
        expected_counts=expected_counts,
    )


def _select_records(
    records: tuple[ReleasedWaterbirdsRecord, ...],
    *,
    count: int,
    seed: int,
    namespace: str,
) -> tuple[ReleasedWaterbirdsRecord, ...]:
    if count > len(records):
        raise ValueError(f"not enough records for deterministic {namespace} selection")
    return tuple(
        sorted(
            records,
            key=lambda record: (
                _selection_hash(seed, namespace, record.record_id),
                record.record_id,
            ),
        )[:count]
    )


def _select_backgrounds(
    records: tuple[PlacesBackgroundRecord, ...],
    *,
    target_background: Background,
    count: int,
    seed: int,
) -> tuple[PlacesBackgroundRecord, ...]:
    eligible = tuple(
        record for record in records if record.background == target_background
    )
    if count > len(eligible):
        kind = "land" if target_background == 0 else "water"
        raise ValueError(f"not enough approved {kind} Places backgrounds")
    namespace = f"background:{target_background}"
    return tuple(
        sorted(
            eligible,
            key=lambda record: (
                _selection_hash(seed, namespace, record.asset_id),
                record.asset_id,
            ),
        )[:count]
    )


def _selection_hash(seed: int, namespace: str, identity: str) -> bytes:
    payload = f"{CONSTRUCTION_METHOD_ID}\0{seed}\0{namespace}\0{identity}".encode()
    return hashlib.sha256(payload).digest()


def _released_group_counts(
    records: tuple[ReleasedWaterbirdsRecord, ...],
) -> WaterbirdsGroupCounts:
    counts = {
        "landbird_land": 0,
        "landbird_water": 0,
        "waterbird_land": 0,
        "waterbird_water": 0,
    }
    for record in records:
        counts[waterbirds_group_id(int(record.bird_label), int(record.background))] += 1
    return WaterbirdsGroupCounts(**counts)


def _strict_csv_text(row: dict[str, str | None], name: str) -> str:
    value = row.get(name)
    if value is None or not value.strip():
        raise ValueError(f"released Waterbirds metadata field {name!r} is empty")
    return value.strip()


def _strict_csv_int(row: dict[str, str | None], name: str) -> int:
    value = _strict_csv_text(row, name)
    try:
        return int(value)
    except ValueError as error:
        raise ValueError(
            f"released Waterbirds metadata field {name!r} must be an integer"
        ) from error


def _binary_csv_int(row: dict[str, str | None], name: str) -> BinaryInt:
    value = _strict_csv_int(row, name)
    if value not in {0, 1}:
        raise ValueError(f"released Waterbirds metadata field {name!r} must be binary")
    return value


def _read_indexed_text(path: Path) -> dict[int, str]:
    result: dict[int, str] = {}
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), 1
    ):
        fields = line.strip().split(maxsplit=1)
        if len(fields) != 2:
            raise ValueError(f"invalid indexed annotation at {path}:{line_number}")
        key = int(fields[0])
        if key in result:
            raise ValueError(f"duplicate annotation ID {key} in {path}")
        result[key] = fields[1]
    if not result:
        raise ValueError(f"annotation file is empty: {path}")
    return result


def _read_bounding_boxes(
    path: Path,
) -> dict[int, tuple[float, float, float, float]]:
    result: dict[int, tuple[float, float, float, float]] = {}
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), 1
    ):
        fields = line.strip().split()
        if len(fields) != 5:
            raise ValueError(f"invalid CUB bounding box at {path}:{line_number}")
        image_id = int(fields[0])
        if image_id in result:
            raise ValueError(f"duplicate CUB bounding-box ID {image_id}")
        result[image_id] = (
            float(fields[1]),
            float(fields[2]),
            float(fields[3]),
            float(fields[4]),
        )
    return result


def places_category_directory(root: Path, category: PlaceCategory) -> Path:
    first = category[0]
    candidates = (
        root / "data_large" / first / category,
        root / first / category,
        root / category,
    )
    for candidate in candidates:
        if candidate.is_dir():
            return candidate
    raise FileNotFoundError(
        f"approved Places category is missing under {root}: {category}"
    )


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return f"sha256:{digest.hexdigest()}"


def _array_sha256(array: NDArray[np.uint8]) -> str:
    contiguous = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(str(contiguous.dtype).encode("ascii"))
    digest.update(str(tuple(int(value) for value in contiguous.shape)).encode("ascii"))
    digest.update(contiguous.tobytes(order="C"))
    return f"sha256:{digest.hexdigest()}"
