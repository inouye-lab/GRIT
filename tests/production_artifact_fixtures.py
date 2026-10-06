"""Manifest-only production fixtures for plan validation; never used for training."""

from __future__ import annotations

import hashlib
from pathlib import Path

import torch

from grit.config import (
    OPENAI_CLIP_PREPROCESSING_ID,
    OPENAI_CLIP_REVISION,
    OPENAI_CLIP_WEIGHTS_IDENTITY,
)
from grit.data.waterbirds import (
    BASE_ARTIFACT_NAME,
    PRODUCTION_TRAIN_GROUP_COUNTS,
    GroupDroGeometry,
    WaterbirdsConstructionCounts,
    WaterbirdsDatasetManifest,
    WaterbirdsOracleRelationship,
    WaterbirdsRecord,
)
from grit.data.waterbirds_pairs import WaterbirdsOraclePairManifest
from grit.features.cmnist import EncoderIdentity, FeatureExtractionRuntime
from grit.features.waterbirds import (
    WaterbirdsFeatureCacheManifest,
    WaterbirdsFeatureFile,
    WaterbirdsFeatureRecord,
)
from grit.schemas import canonical_digest_value


def write_manifest_only_waterbirds_production(
    root: Path,
) -> tuple[Path, Path, Path]:
    """Write internally valid production manifests with deliberately tiny fake bytes."""

    construction_seed = 17
    source_bundle_digest = "sha256:manifest-only-waterbirds-source-bundle"
    geometry = GroupDroGeometry(
        compositor_id="groupdro-center-crop-lanczos-v1",
        generated_encoding_id="png-rgb-v1",
        width=224,
        height=224,
        resampling="pillow_lanczos",
    )
    relationships: list[WaterbirdsOracleRelationship] = []
    paired_records: list[WaterbirdsRecord] = []
    for position in range(240):
        label = 0 if position < 184 else 1
        majority_background = label
        generated_background = 1 - label
        source_id = position + 1
        majority_id = f"waterbirds:released:majority:{position}"
        generated_background_id = (
            "water-background" if generated_background == 1 else "land-background"
        )
        generated_id = "waterbirds:generated:" + canonical_digest_value(
            {
                "method": "waterbirds-cf-sha256-v1",
                "source_bundle_digest": source_bundle_digest,
                "construction_seed": construction_seed,
                "source_cub_image_id": source_id,
                "background_asset_id": generated_background_id,
                "target_background": generated_background,
                "selection_position": position,
            }
        ).removeprefix("sha256:")
        pair_id = canonical_digest_value(
            {
                "method": "waterbirds-cf-sha256-v1",
                "source_bundle_digest": source_bundle_digest,
                "construction_seed": construction_seed,
                "majority_record_id": majority_id,
                "generated_record_id": generated_id,
                "orientation": "land_minus_water",
            }
        )
        source_sha = f"sha256:source:{source_id}"
        mask_sha = f"sha256:mask:{source_id}"
        foreground = f"sha256:foreground:{source_id}"
        majority_sha = f"sha256:majority:{position}"
        generated_sha = f"sha256:generated:{position}"
        majority = _record(
            record_id=majority_id,
            split_role="training",
            component="majority_endpoint",
            label=label,
            background=majority_background,
            source_id=source_id,
            image_location="released",
            image_sha256=majority_sha,
            source_sha=source_sha,
            mask_sha=mask_sha,
            foreground_digest=foreground,
            pair_id=pair_id,
            endpoint_role="land" if majority_background == 0 else "water",
            background_asset_id=(
                "land-background"
                if majority_background == 0
                else "water-background"
            ),
            construction_seed=construction_seed,
            selection_position=position,
            geometry=geometry,
        )
        generated = _record(
            record_id=generated_id,
            split_role="training",
            component="generated_minority",
            label=label,
            background=generated_background,
            source_id=source_id,
            image_location="generated",
            image_sha256=generated_sha,
            source_sha=source_sha,
            mask_sha=mask_sha,
            foreground_digest=foreground,
            pair_id=pair_id,
            endpoint_role="land" if generated_background == 0 else "water",
            background_asset_id=generated_background_id,
            construction_seed=construction_seed,
            selection_position=position,
            geometry=geometry,
        )
        land = majority if majority_background == 0 else generated
        water = majority if majority_background == 1 else generated
        relationships.append(
            WaterbirdsOracleRelationship(
                pair_id=pair_id,
                construction_input_digest=source_bundle_digest,
                source_cub_image_id=source_id,
                bird_label=label,
                land_record_id=land.record_id,
                water_record_id=water.record_id,
                majority_record_id=majority.record_id,
                generated_record_id=generated.record_id,
                land_image_sha256=land.image_sha256,
                water_image_sha256=water.image_sha256,
                source_image_sha256=source_sha,
                mask_sha256=mask_sha,
                canonical_masked_source_foreground_digest=foreground,
                land_background_asset_id="land-background",
                water_background_asset_id="water-background",
                geometry=geometry,
                construction_seed=construction_seed,
                source_selection_position=position,
                background_selection_position=position,
                orientation="land_minus_water",
            )
        )
        paired_records.extend((majority, generated))

    unpaired: list[WaterbirdsRecord] = []
    next_source = 241
    for label, background, count in ((0, 0, 3_314), (1, 1, 1_001)):
        for _ in range(count):
            source_id = next_source
            next_source += 1
            unpaired.append(
                _record(
                    record_id=f"waterbirds:released:unpaired:{source_id}",
                    split_role="training",
                    component="unpaired_majority",
                    label=label,
                    background=background,
                    source_id=source_id,
                    image_location="released",
                    image_sha256=f"sha256:unpaired:{source_id}",
                    source_sha=f"sha256:source:{source_id}",
                    mask_sha=f"sha256:mask:{source_id}",
                    foreground_digest=f"sha256:foreground:{source_id}",
                    pair_id=None,
                    endpoint_role=None,
                    background_asset_id=None,
                    construction_seed=None,
                    selection_position=None,
                    geometry=geometry,
                )
            )
    training = tuple((*unpaired, *paired_records))
    validation = tuple(
        _released_evaluation_record(
            source_id=next_source + index,
            split_role="validation",
            component="released_validation",
            geometry=geometry,
        )
        for index in range(1_199)
    )
    next_source += 1_199
    test = tuple(
        _released_evaluation_record(
            source_id=next_source + index,
            split_role="final_test",
            component="released_test",
            geometry=geometry,
        )
        for index in range(5_794)
    )
    records = (*training, *validation, *test)
    ordered_relationships = tuple(
        sorted(relationships, key=lambda item: item.pair_id)
    )
    manifest = WaterbirdsDatasetManifest(
        schema_version="grit.waterbirds-cf-dataset/v2",
        dataset_id="waterbirds_cf",
        profile_kind="production",
        non_reportable=False,
        base_artifact_name=BASE_ARTIFACT_NAME,
        construction_method_id="waterbirds-cf-sha256-v1",
        compositor_id="groupdro-center-crop-lanczos-v1",
        generated_encoding_id="png-rgb-v1",
        construction_seed=construction_seed,
        source_bundle_digest=source_bundle_digest,
        released_metadata_file_sha256="sha256:released-metadata",
        released_metadata_rows_digest="sha256:released-rows",
        counts=WaterbirdsConstructionCounts(
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
        ),
        records=records,
        relationships=ordered_relationships,
        replaced_released_record_ids=tuple(
            f"waterbirds:replaced:{index}" for index in range(240)
        ),
        training_membership_digest=canonical_digest_value(
            tuple(item.record_id for item in training)
        ),
        validation_membership_digest=canonical_digest_value(
            tuple(item.record_id for item in validation)
        ),
        test_membership_digest=canonical_digest_value(
            tuple(item.record_id for item in test)
        ),
        released_validation_bytes_digest=canonical_digest_value(
            tuple((item.record_id, item.image_sha256) for item in validation)
        ),
        released_test_bytes_digest=canonical_digest_value(
            tuple((item.record_id, item.image_sha256) for item in test)
        ),
    )
    dataset_digest = manifest.canonical_digest()
    pairs = WaterbirdsOraclePairManifest(
        schema_version="grit.waterbirds-oracle-pairs/v2",
        dataset_manifest_digest=dataset_digest,
        construction_method_id="waterbirds-clean-oracle-pairs-v1",
        orientation="land_minus_water",
        profile_kind="production",
        non_reportable=False,
        pair_count=240,
        landbird_pair_count=184,
        waterbird_pair_count=56,
        records=ordered_relationships,
        membership_digest=canonical_digest_value(
            tuple(item.pair_id for item in ordered_relationships)
        ),
    )
    feature_records = tuple(
        WaterbirdsFeatureRecord(
            record_id=record.record_id,
            row_index=index,
            split_role=record.split_role,
            bird_label=record.bird_label,
            background=record.background,
            group_id=record.group_id,
            image_sha256=record.image_sha256,
        )
        for index, record in enumerate(records)
    )
    feature_root = root / "feature-cache"
    feature_root.mkdir(parents=True, exist_ok=True)
    feature_bytes = b"manifest-only-waterbirds-features"
    (feature_root / "features.npy").write_bytes(feature_bytes)
    features = WaterbirdsFeatureCacheManifest(
        schema_version="grit.waterbirds-features/v2",
        dataset_id="waterbirds_cf",
        dataset_manifest_digest=dataset_digest,
        non_reportable=False,
        encoder=EncoderIdentity(
            implementation="openai/CLIP",
            implementation_revision=OPENAI_CLIP_REVISION,
            model_name="ViT-B/32",
            weights_identity=OPENAI_CLIP_WEIGHTS_IDENTITY,
            preprocessing_identity=OPENAI_CLIP_PREPROCESSING_ID,
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
        records=feature_records,
        membership_digest=canonical_digest_value(
            tuple(item.record_id for item in feature_records)
        ),
        feature_file=WaterbirdsFeatureFile(
            relative_path="features.npy",
            sha256=f"sha256:{hashlib.sha256(feature_bytes).hexdigest()}",
            shape=(len(records), 512),
            dtype="float32",
        ),
    )
    dataset_path = root / "dataset-manifest.json"
    pair_path = root / "pair-manifest.json"
    feature_path = feature_root / "manifest.json"
    root.mkdir(parents=True, exist_ok=True)
    dataset_path.write_text(manifest.canonical_json(), encoding="utf-8")
    pair_path.write_text(pairs.canonical_json(), encoding="utf-8")
    feature_path.write_text(features.canonical_json(), encoding="utf-8")
    return dataset_path, feature_path, pair_path


def _record(
    *,
    record_id: str,
    split_role: str,
    component: str,
    label: int,
    background: int,
    source_id: int,
    image_location: str,
    image_sha256: str,
    source_sha: str,
    mask_sha: str,
    foreground_digest: str,
    pair_id: str | None,
    endpoint_role: str | None,
    background_asset_id: str | None,
    construction_seed: int | None,
    selection_position: int | None,
    geometry: GroupDroGeometry,
) -> WaterbirdsRecord:
    return WaterbirdsRecord.model_validate(
        {
            "record_id": record_id,
            "split_role": split_role,
            "component": component,
            "bird_label": label,
            "background": background,
            "group_id": _group_id(label, background),
            "source_cub_image_id": source_id,
            "species": "manifest-only-bird",
            "image_location": image_location,
            "image_relative_path": f"images/{record_id}.png",
            "image_sha256": image_sha256,
            "source_image_sha256": source_sha,
            "mask_sha256": mask_sha,
            "canonical_masked_source_foreground_digest": foreground_digest,
            "bounding_box_xywh": (0.0, 0.0, 1.0, 1.0),
            "pair_id": pair_id,
            "endpoint_role": endpoint_role,
            "background_asset_id": background_asset_id,
            "background_asset_sha256": (
                f"sha256:{background_asset_id}"
                if background_asset_id is not None
                else None
            ),
            "construction_seed": construction_seed,
            "selection_position": selection_position,
            "geometry": geometry,
        }
    )


def _released_evaluation_record(
    *,
    source_id: int,
    split_role: str,
    component: str,
    geometry: GroupDroGeometry,
) -> WaterbirdsRecord:
    label = source_id % 2
    background = (source_id // 2) % 2
    return _record(
        record_id=f"waterbirds:released:evaluation:{source_id}",
        split_role=split_role,
        component=component,
        label=label,
        background=background,
        source_id=source_id,
        image_location="released",
        image_sha256=f"sha256:evaluation:{source_id}",
        source_sha=f"sha256:source:{source_id}",
        mask_sha=f"sha256:mask:{source_id}",
        foreground_digest=f"sha256:foreground:{source_id}",
        pair_id=None,
        endpoint_role=None,
        background_asset_id=None,
        construction_seed=None,
        selection_position=None,
        geometry=geometry,
    )


def _group_id(label: int, background: int) -> str:
    names = (
        ("landbird_land", "landbird_water"),
        ("waterbird_land", "waterbird_water"),
    )
    return names[label][background]
