"""Explicit Waterbirds-CF oracle relationships, separate from supervision."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Annotated, Literal, TypeAlias

from pydantic import Field, StrictBool, StrictInt, StrictStr, model_validator

from grit.data.waterbirds import (
    WaterbirdsOracleRelationship,
    WaterbirdsOracleRelationView,
)
from grit.schemas import StrictBoundaryModel, canonical_digest_value

NonEmptyStr: TypeAlias = Annotated[StrictStr, Field(min_length=1)]
PositiveInt: TypeAlias = Annotated[StrictInt, Field(gt=0)]


class WaterbirdsOraclePairManifest(StrictBoundaryModel):
    schema_version: Literal["grit.waterbirds-oracle-pairs/v2"]
    dataset_manifest_digest: NonEmptyStr
    construction_method_id: Literal["waterbirds-clean-oracle-pairs-v1"]
    orientation: Literal["land_minus_water"]
    profile_kind: Literal["production", "fixture"]
    non_reportable: StrictBool
    pair_count: PositiveInt
    landbird_pair_count: PositiveInt
    waterbird_pair_count: PositiveInt
    records: tuple[WaterbirdsOracleRelationship, ...]
    membership_digest: NonEmptyStr

    @model_validator(mode="after")
    def _validate_records(self) -> WaterbirdsOraclePairManifest:
        if self.pair_count != len(self.records):
            raise ValueError("Waterbirds oracle pair count is inconsistent")
        pair_ids = tuple(record.pair_id for record in self.records)
        if len(pair_ids) != len(set(pair_ids)):
            raise ValueError("Waterbirds oracle pair IDs must be unique")
        if self.membership_digest != canonical_digest_value(pair_ids):
            raise ValueError("Waterbirds oracle pair membership digest is inconsistent")
        landbirds = sum(record.bird_label == 0 for record in self.records)
        waterbirds = sum(record.bird_label == 1 for record in self.records)
        if (landbirds, waterbirds) != (
            self.landbird_pair_count,
            self.waterbird_pair_count,
        ):
            raise ValueError("Waterbirds oracle label strata are inconsistent")
        if any(record.orientation != self.orientation for record in self.records):
            raise ValueError("Waterbirds oracle pair orientation is inconsistent")
        if self.profile_kind == "production" and (
            self.non_reportable
            or self.pair_count != 240
            or self.landbird_pair_count != 184
            or self.waterbird_pair_count != 56
        ):
            raise ValueError("production Waterbirds oracle pairs must use exact 184/56")
        if self.profile_kind == "fixture" and not self.non_reportable:
            raise ValueError("fixture Waterbirds oracle pairs must be non-reportable")
        return self


@dataclass(frozen=True, slots=True)
class WaterbirdsOraclePairSet:
    manifest: WaterbirdsOraclePairManifest


def build_waterbirds_oracle_pairs(
    relations: WaterbirdsOracleRelationView,
) -> WaterbirdsOraclePairSet:
    """Materialize the approved relation manifest from an oracle-only capability."""

    if type(relations) is not WaterbirdsOracleRelationView:
        raise TypeError("Waterbirds oracle pairs require WaterbirdsOracleRelationView")
    dataset = relations.validated_manifest()
    records = tuple(sorted(dataset.relationships, key=lambda record: record.pair_id))
    if any(
        record.construction_input_digest != dataset.source_bundle_digest
        for record in records
    ):
        raise ValueError("Waterbirds oracle relation lineage is inconsistent")
    pair_ids = tuple(record.pair_id for record in records)
    manifest = WaterbirdsOraclePairManifest(
        schema_version="grit.waterbirds-oracle-pairs/v2",
        dataset_manifest_digest=dataset.canonical_digest(),
        construction_method_id="waterbirds-clean-oracle-pairs-v1",
        orientation="land_minus_water",
        profile_kind=dataset.profile_kind,
        non_reportable=dataset.non_reportable,
        pair_count=len(records),
        landbird_pair_count=dataset.counts.landbird_relationships,
        waterbird_pair_count=dataset.counts.waterbird_relationships,
        records=records,
        membership_digest=canonical_digest_value(pair_ids),
    )
    return WaterbirdsOraclePairSet(manifest=manifest)
