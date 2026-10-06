"""Pinned frozen-feature cache for the Waterbirds-CF vertical slice."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Literal, Protocol, TypeAlias, cast

import numpy as np
import torch
from numpy.typing import NDArray
from PIL import Image
from pydantic import Field, StrictBool, StrictInt, StrictStr, model_validator

from grit.data.waterbirds import (
    GroupId,
    SplitRole,
    WaterbirdsConstruction,
    validate_waterbirds_construction,
)
from grit.data.waterbirds_pairs import (
    WaterbirdsOraclePairManifest,
    WaterbirdsOraclePairSet,
)
from grit.features.cmnist import (
    EncoderIdentity,
    FeatureExtractionRuntime,
    PilImageEncoder,
)
from grit.methods.projection import FittedLinearProjection, fit_linear_projection
from grit.methods.types import MethodId
from grit.schemas import StrictBoundaryModel, canonical_digest_value

if TYPE_CHECKING:
    from grit.methods.waterbirds_training import WaterbirdsRestorationReceipt
    from grit.selection.waterbirds import (
        FrozenWaterbirdsCandidate,
        FrozenWaterbirdsCheckpoint,
    )

FEATURE_DIMENSION = 512
NonEmptyStr: TypeAlias = Annotated[StrictStr, Field(min_length=1)]
NonNegativeInt: TypeAlias = Annotated[StrictInt, Field(ge=0)]
BinaryInt: TypeAlias = Annotated[StrictInt, Field(ge=0, le=1)]
Normalization: TypeAlias = Literal["none", "l2"]


class _TensorToNumpy(Protocol):
    def __call__(self, *, force: bool = False) -> NDArray[np.generic]: ...


class _TorchFromNumpy(Protocol):
    def __call__(self, array: NDArray[np.generic]) -> torch.Tensor: ...


_torch_from_numpy = cast(_TorchFromNumpy, torch.from_numpy)


class WaterbirdsFeatureCacheError(ValueError):
    """A Waterbirds feature cache is absent or has inconsistent lineage."""


@dataclass(frozen=True, slots=True)
class DeterministicFakeWaterbirdsEncoder:
    """Offline-only pixel projection for deterministic, non-reportable tests."""

    seed: int = 0

    @property
    def identity(self) -> EncoderIdentity:
        return EncoderIdentity(
            implementation="grit.synthetic",
            implementation_revision="waterbirds-fake-encoder-v1",
            model_name="fake-512",
            weights_identity=f"seed:{self.seed}",
            preprocessing_identity="resize-rgb-8x8-bilinear",
            raw_output_dimension=FEATURE_DIMENSION,
        )

    @property
    def extraction_runtime(self) -> FeatureExtractionRuntime:
        return FeatureExtractionRuntime(
            requested_device="cpu",
            resolved_device="cpu",
            computation_dtype="torch.float32",
            deterministic_algorithms=True,
            tf32_enabled=False,
            mixed_precision=False,
            batch_size=None,
            torch_version=str(torch.__version__),
            cuda_runtime_version=None,
            device_name="cpu",
            compute_capability=None,
        )

    def encode_pil(self, images: tuple[Image.Image, ...]) -> torch.Tensor:
        if not images:
            raise ValueError("fake Waterbirds encoder input must not be empty")
        rows = np.stack(
            [
                np.asarray(
                    image.convert("RGB").resize((8, 8), Image.Resampling.BILINEAR),
                    dtype=np.float32,
                ).transpose(2, 0, 1)
                / 255.0
                for image in images
            ]
        )
        inputs = _torch_from_numpy(rows).flatten(start_dim=1)
        generator = torch.Generator(device="cpu").manual_seed(self.seed)
        weights = torch.randn(
            (int(inputs.shape[1]), FEATURE_DIMENSION),
            generator=generator,
            dtype=torch.float32,
        )
        features = torch.stack([row @ weights for row in inputs])
        features = features / int(inputs.shape[1]) ** 0.5
        features[:, 0] += 1.0
        _validate_features(features, len(images))
        return features


class WaterbirdsFeatureRecord(StrictBoundaryModel):
    record_id: NonEmptyStr
    row_index: NonNegativeInt
    split_role: SplitRole
    bird_label: BinaryInt
    background: BinaryInt
    group_id: GroupId
    image_sha256: NonEmptyStr


class WaterbirdsFeatureFile(StrictBoundaryModel):
    relative_path: Literal["features.npy"]
    sha256: NonEmptyStr
    shape: tuple[NonNegativeInt, Literal[512]]
    dtype: Literal["float32"]


class WaterbirdsFeatureCacheManifest(StrictBoundaryModel):
    schema_version: Literal["grit.waterbirds-features/v2"]
    dataset_id: Literal["waterbirds_cf"]
    dataset_manifest_digest: NonEmptyStr
    non_reportable: StrictBool
    encoder: EncoderIdentity
    extraction_runtime: FeatureExtractionRuntime
    normalization: Normalization
    feature_dimension: Literal[512]
    feature_dtype: Literal["float32"]
    records: tuple[WaterbirdsFeatureRecord, ...]
    membership_digest: NonEmptyStr
    feature_file: WaterbirdsFeatureFile

    @model_validator(mode="after")
    def _validate_records(self) -> WaterbirdsFeatureCacheManifest:
        if not self.records:
            raise ValueError("Waterbirds feature records must not be empty")
        expected_rows = tuple(range(len(self.records)))
        if tuple(record.row_index for record in self.records) != expected_rows:
            raise ValueError("Waterbirds feature rows must be contiguous and ordered")
        record_ids = tuple(record.record_id for record in self.records)
        if len(record_ids) != len(set(record_ids)):
            raise ValueError("Waterbirds feature record IDs must be unique")
        if self.membership_digest != canonical_digest_value(record_ids):
            raise ValueError("Waterbirds feature membership digest is inconsistent")
        if self.feature_file.shape != (len(self.records), FEATURE_DIMENSION):
            raise ValueError("Waterbirds feature array shape is inconsistent")
        roles = {record.split_role for record in self.records}
        if roles != {"training", "validation", "final_test"}:
            raise ValueError("Waterbirds feature cache must contain all split roles")
        if (
            self.encoder.implementation == "openai/CLIP"
            and self.extraction_runtime.batch_size is None
        ):
            raise ValueError("official CLIP feature caches require a batch size")
        return self


@dataclass(frozen=True, slots=True)
class WaterbirdsTrainingFeatureTable:
    dataset_manifest_digest: str
    feature_cache_manifest_digest: str
    normalization: Normalization
    record_ids: tuple[str, ...]
    features: torch.Tensor
    labels: torch.Tensor


@dataclass(frozen=True, slots=True)
class WaterbirdsEvaluationFeatureTable:
    dataset_manifest_digest: str
    feature_cache_manifest_digest: str
    normalization: Normalization
    split_role: Literal["validation", "final_test"]
    record_ids: tuple[str, ...]
    features: torch.Tensor
    labels: torch.Tensor
    backgrounds: torch.Tensor
    group_ids: tuple[GroupId, ...]


@dataclass(frozen=True, slots=True)
class WaterbirdsFinalTestView:
    """Final features materialized only by the post-restoration gate."""

    authorization_id: str
    run_id: str
    candidate_id: str
    method_id: MethodId
    scientific_config_digest: str
    checkpoint_id: str
    epoch: int
    seed: int
    projection_rank: int | None
    feature_cache_manifest_digest: str
    table: WaterbirdsEvaluationFeatureTable


class WaterbirdsFinalTestHandle:
    """Opaque final table bound to one cache and intended final run."""

    __slots__ = (
        "__candidate_id",
        "__feature_cache_manifest_digest",
        "__method_id",
        "__projection_rank",
        "__run_id",
        "__scientific_config_digest",
        "__seed",
        "__table",
    )

    def __init__(
        self,
        *,
        run_id: str,
        candidate_id: str,
        method_id: MethodId,
        scientific_config_digest: str,
        seed: int,
        projection_rank: int | None,
        feature_cache_manifest_digest: str,
        table: WaterbirdsEvaluationFeatureTable,
    ) -> None:
        self.__run_id = run_id
        self.__candidate_id = candidate_id
        self.__method_id: MethodId = method_id
        self.__scientific_config_digest = scientific_config_digest
        self.__seed = seed
        self.__projection_rank = projection_rank
        self.__feature_cache_manifest_digest = feature_cache_manifest_digest
        self.__table = table

    def open(
        self,
        candidate: FrozenWaterbirdsCandidate,
        checkpoint: FrozenWaterbirdsCheckpoint,
        restoration: WaterbirdsRestorationReceipt,
    ) -> WaterbirdsFinalTestView:
        """Open only after candidate/checkpoint selection and matching restoration."""

        if checkpoint.candidate_selection_id != candidate.frozen_selection_id:
            raise ValueError("final checkpoint does not belong to Waterbirds candidate")
        if checkpoint.method_id != candidate.method_id:
            raise ValueError(
                "final checkpoint method does not match Waterbirds candidate"
            )
        identity = checkpoint.checkpoint
        expected = (
            self.__run_id,
            self.__candidate_id,
            self.__scientific_config_digest,
            self.__method_id,
            self.__seed,
            self.__projection_rank,
        )
        observed = (
            identity.run_id,
            identity.candidate_id,
            identity.scientific_config_digest,
            checkpoint.method_id,
            checkpoint.decision.seed,
            checkpoint.decision.projection_rank,
        )
        if observed != expected:
            raise ValueError(
                "final handle identity does not match Waterbirds selection"
            )
        if restoration.candidate_selection_id != candidate.frozen_selection_id:
            raise ValueError("Waterbirds restoration belongs to another candidate")
        if restoration.checkpoint != identity:
            raise ValueError("Waterbirds restoration does not match final checkpoint")
        if (
            self.__table.feature_cache_manifest_digest
            != self.__feature_cache_manifest_digest
        ):
            raise ValueError("Waterbirds final handle feature cache is inconsistent")
        return WaterbirdsFinalTestView(
            authorization_id=f"final:{self.__run_id}:{restoration.receipt_id}",
            run_id=self.__run_id,
            candidate_id=self.__candidate_id,
            method_id=self.__method_id,
            scientific_config_digest=self.__scientific_config_digest,
            checkpoint_id=identity.checkpoint_id,
            epoch=identity.epoch,
            seed=self.__seed,
            projection_rank=self.__projection_rank,
            feature_cache_manifest_digest=self.__feature_cache_manifest_digest,
            table=self.__table,
        )


def _training_rows(
    manifest: WaterbirdsFeatureCacheManifest,
) -> tuple[WaterbirdsFeatureRecord, ...]:
    return tuple(
        record for record in manifest.records if record.split_role == "training"
    )


def _training_table(
    manifest: WaterbirdsFeatureCacheManifest, features: torch.Tensor
) -> WaterbirdsTrainingFeatureTable:
    rows = _training_rows(manifest)
    indices = torch.tensor([record.row_index for record in rows], dtype=torch.int64)
    return WaterbirdsTrainingFeatureTable(
        dataset_manifest_digest=manifest.dataset_manifest_digest,
        feature_cache_manifest_digest=manifest.canonical_digest(),
        normalization=manifest.normalization,
        record_ids=tuple(record.record_id for record in rows),
        features=features[indices].clone(),
        labels=torch.tensor([record.bird_label for record in rows], dtype=torch.int64),
    )


def _training_environment_ids(manifest: WaterbirdsFeatureCacheManifest) -> torch.Tensor:
    """Training backgrounds (0 land, 1 water) aligned with `training_table()` rows.

    Method-definition access for V-REx, IRMv1, Fishr, and RDM; the ordinary
    training table used by ERM and SD stays redacted.
    """

    return torch.tensor(
        [record.background for record in _training_rows(manifest)], dtype=torch.int64
    )


def _training_group_ids(manifest: WaterbirdsFeatureCacheManifest) -> torch.Tensor:
    """Canonical `2 * label + background` groups aligned with `training_table()` rows.

    Method-definition access for GroupDRO and LISA only.
    """

    return torch.tensor(
        [
            2 * record.bird_label + record.background
            for record in _training_rows(manifest)
        ],
        dtype=torch.int64,
    )


@dataclass(frozen=True, slots=True)
class WaterbirdsFeatureCache:
    manifest: WaterbirdsFeatureCacheManifest
    features: torch.Tensor
    root: Path

    def training_table(self) -> WaterbirdsTrainingFeatureTable:
        return _training_table(self.manifest, self.features)

    def training_environment_ids(self) -> torch.Tensor:
        return _training_environment_ids(self.manifest)

    def training_group_ids(self) -> torch.Tensor:
        return _training_group_ids(self.manifest)

    def validation_table(self) -> WaterbirdsEvaluationFeatureTable:
        return self._evaluation_table("validation")

    def issue_final_handle(
        self,
        *,
        run_id: str,
        candidate_id: str,
        method_id: MethodId,
        scientific_config_digest: str,
        seed: int,
        projection_rank: int | None,
    ) -> WaterbirdsFinalTestHandle:
        return WaterbirdsFinalTestHandle(
            run_id=run_id,
            candidate_id=candidate_id,
            method_id=method_id,
            scientific_config_digest=scientific_config_digest,
            seed=seed,
            projection_rank=projection_rank,
            feature_cache_manifest_digest=self.manifest.canonical_digest(),
            table=self._evaluation_table("final_test"),
        )

    def verify_final_view(
        self,
        view: WaterbirdsFinalTestView,
    ) -> WaterbirdsEvaluationFeatureTable:
        if view.feature_cache_manifest_digest != self.manifest.canonical_digest():
            raise WaterbirdsFeatureCacheError(
                "authorized final view belongs to another Waterbirds feature cache"
            )
        expected = self._evaluation_table("final_test")
        if (
            view.table.record_ids != expected.record_ids
            or view.table.feature_cache_manifest_digest
            != expected.feature_cache_manifest_digest
        ):
            raise WaterbirdsFeatureCacheError(
                "authorized final view does not match cached final records"
            )
        return view.table

    def _evaluation_table(
        self,
        role: Literal["validation", "final_test"],
    ) -> WaterbirdsEvaluationFeatureTable:
        rows = tuple(
            record for record in self.manifest.records if record.split_role == role
        )
        indices = torch.tensor([record.row_index for record in rows], dtype=torch.int64)
        return WaterbirdsEvaluationFeatureTable(
            dataset_manifest_digest=self.manifest.dataset_manifest_digest,
            feature_cache_manifest_digest=self.manifest.canonical_digest(),
            normalization=self.manifest.normalization,
            split_role=role,
            record_ids=tuple(record.record_id for record in rows),
            features=self.features[indices].clone(),
            labels=torch.tensor(
                [record.bird_label for record in rows], dtype=torch.int64
            ),
            backgrounds=torch.tensor(
                [record.background for record in rows], dtype=torch.int64
            ),
            group_ids=tuple(record.group_id for record in rows),
        )


@dataclass(frozen=True, slots=True)
class WaterbirdsTuningFeatureCache:
    """Training/validation cache whose public surface has no final-test access."""

    manifest: WaterbirdsFeatureCacheManifest
    features: torch.Tensor
    root: Path

    def training_table(self) -> WaterbirdsTrainingFeatureTable:
        return _training_table(self.manifest, self.features)

    def training_environment_ids(self) -> torch.Tensor:
        return _training_environment_ids(self.manifest)

    def training_group_ids(self) -> torch.Tensor:
        return _training_group_ids(self.manifest)

    def validation_table(self) -> WaterbirdsEvaluationFeatureTable:
        rows = tuple(
            record
            for record in self.manifest.records
            if record.split_role == "validation"
        )
        indices = torch.tensor([record.row_index for record in rows], dtype=torch.int64)
        return WaterbirdsEvaluationFeatureTable(
            dataset_manifest_digest=self.manifest.dataset_manifest_digest,
            feature_cache_manifest_digest=self.manifest.canonical_digest(),
            normalization=self.manifest.normalization,
            split_role="validation",
            record_ids=tuple(record.record_id for record in rows),
            features=self.features[indices].clone(),
            labels=torch.tensor(
                [record.bird_label for record in rows], dtype=torch.int64
            ),
            backgrounds=torch.tensor(
                [record.background for record in rows], dtype=torch.int64
            ),
            group_ids=tuple(record.group_id for record in rows),
        )


def prepare_waterbirds_feature_cache(
    construction: WaterbirdsConstruction,
    encoder: PilImageEncoder,
    output_dir: Path,
    *,
    normalization: Normalization,
    encode_batch_size: int = 64,
) -> WaterbirdsFeatureCacheManifest:
    """Validate, encode, and write one canonical Waterbirds-CF feature array."""

    dataset = validate_waterbirds_construction(construction)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Waterbirds feature cache is not empty: {output_dir}")
    if encode_batch_size <= 0:
        raise ValueError("encode_batch_size must be positive")
    identity = EncoderIdentity.model_validate_json(encoder.identity.canonical_json())
    if identity.raw_output_dimension != FEATURE_DIMENSION:
        raise ValueError("Waterbirds encoder must emit 512-dimensional features")
    ordered = dataset.records
    batches: list[torch.Tensor] = []
    for start in range(0, len(ordered), encode_batch_size):
        records = ordered[start : start + encode_batch_size]
        images: list[Image.Image] = []
        try:
            for record in records:
                with Image.open(construction.path_for(record.record_id)) as image:
                    images.append(image.convert("RGB"))
            batch = encoder.encode_pil(tuple(images))
        finally:
            for image in images:
                image.close()
        _validate_features(batch, len(records))
        batches.append(batch.detach().cpu().to(torch.float32))
    features = _normalize_features(torch.cat(batches, dim=0), normalization)

    output_dir.mkdir(parents=True, exist_ok=True)
    feature_path = output_dir / "features.npy"
    to_numpy = cast(_TensorToNumpy, features.numpy)
    array = cast(NDArray[np.float32], to_numpy(force=True))
    np.save(feature_path, array, allow_pickle=False)
    records = tuple(
        WaterbirdsFeatureRecord(
            record_id=record.record_id,
            row_index=row,
            split_role=record.split_role,
            bird_label=record.bird_label,
            background=record.background,
            group_id=record.group_id,
            image_sha256=record.image_sha256,
        )
        for row, record in enumerate(ordered)
    )
    manifest = WaterbirdsFeatureCacheManifest(
        schema_version="grit.waterbirds-features/v2",
        dataset_id="waterbirds_cf",
        dataset_manifest_digest=dataset.canonical_digest(),
        non_reportable=(
            dataset.non_reportable or identity.implementation == "grit.synthetic"
        ),
        encoder=identity,
        extraction_runtime=encoder.extraction_runtime,
        normalization=normalization,
        feature_dimension=FEATURE_DIMENSION,
        feature_dtype="float32",
        records=records,
        membership_digest=canonical_digest_value(
            tuple(record.record_id for record in records)
        ),
        feature_file=WaterbirdsFeatureFile(
            relative_path="features.npy",
            sha256=_file_sha256(feature_path),
            shape=(len(records), FEATURE_DIMENSION),
            dtype="float32",
        ),
    )
    (output_dir / "manifest.json").write_text(
        manifest.canonical_json() + "\n", encoding="utf-8"
    )
    return manifest


def load_waterbirds_feature_cache(
    root: Path,
    *,
    expected_dataset_manifest_digest: str | None = None,
    expected_normalization: Normalization | None = None,
) -> WaterbirdsFeatureCache:
    manifest, feature_path = _load_waterbirds_feature_manifest(
        root,
        expected_dataset_manifest_digest=expected_dataset_manifest_digest,
        expected_normalization=expected_normalization,
    )
    array = _load_waterbirds_feature_array(feature_path, manifest)
    copied = array.copy()
    features = _torch_from_numpy(copied).to(torch.float32)
    _validate_features(features, len(manifest.records))
    return WaterbirdsFeatureCache(manifest=manifest, features=features, root=root)


def load_waterbirds_tuning_feature_cache(
    root: Path,
    *,
    expected_dataset_manifest_digest: str | None = None,
    expected_normalization: Normalization | None = None,
) -> WaterbirdsTuningFeatureCache:
    """Load only training/validation rows for bounded tuning execution."""

    manifest, feature_path = _load_waterbirds_feature_manifest(
        root,
        expected_dataset_manifest_digest=expected_dataset_manifest_digest,
        expected_normalization=expected_normalization,
    )
    array = _load_waterbirds_feature_array(feature_path, manifest, memory_mapped=True)
    selected_rows = tuple(
        record.row_index
        for record in manifest.records
        if record.split_role != "final_test"
    )
    selected = np.array(array[list(selected_rows)], copy=True)
    selected_tensor = _torch_from_numpy(selected).to(torch.float32)
    _validate_features(selected_tensor, len(selected_rows))
    features = torch.zeros(
        (len(manifest.records), FEATURE_DIMENSION), dtype=torch.float32
    )
    features[torch.tensor(selected_rows, dtype=torch.int64)] = selected_tensor
    return WaterbirdsTuningFeatureCache(manifest=manifest, features=features, root=root)


def _load_waterbirds_feature_manifest(
    root: Path,
    *,
    expected_dataset_manifest_digest: str | None,
    expected_normalization: Normalization | None,
) -> tuple[WaterbirdsFeatureCacheManifest, Path]:
    manifest_path = root / "manifest.json"
    if not manifest_path.is_file():
        raise WaterbirdsFeatureCacheError(
            f"Waterbirds feature manifest is missing: {manifest_path}"
        )
    try:
        manifest = WaterbirdsFeatureCacheManifest.model_validate_json(
            manifest_path.read_text(encoding="utf-8")
        )
    except (OSError, ValueError) as error:
        raise WaterbirdsFeatureCacheError(
            f"Waterbirds feature manifest is invalid: {manifest_path}"
        ) from error
    if (
        expected_dataset_manifest_digest is not None
        and manifest.dataset_manifest_digest != expected_dataset_manifest_digest
    ):
        raise WaterbirdsFeatureCacheError(
            "feature cache dataset manifest does not match"
        )
    if (
        expected_normalization is not None
        and manifest.normalization != expected_normalization
    ):
        raise WaterbirdsFeatureCacheError("feature cache normalization does not match")
    feature_path = root / manifest.feature_file.relative_path
    if not feature_path.is_file() or _file_sha256(feature_path) != (
        manifest.feature_file.sha256
    ):
        raise WaterbirdsFeatureCacheError("Waterbirds feature array digest is invalid")
    return manifest, feature_path


def _load_waterbirds_feature_array(
    path: Path,
    manifest: WaterbirdsFeatureCacheManifest,
    *,
    memory_mapped: bool = False,
) -> NDArray[np.generic]:
    array = np.load(
        path,
        allow_pickle=False,
        mmap_mode="r" if memory_mapped else None,
    )
    if tuple(int(value) for value in array.shape) != manifest.feature_file.shape:
        raise WaterbirdsFeatureCacheError("Waterbirds feature array shape is invalid")
    if str(array.dtype) != manifest.feature_file.dtype:
        raise WaterbirdsFeatureCacheError("Waterbirds feature array dtype is invalid")
    return cast(NDArray[np.generic], array)


def waterbirds_oracle_pair_features(
    cache: WaterbirdsFeatureCache | WaterbirdsTuningFeatureCache,
    pairs: WaterbirdsOraclePairSet,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Resolve canonical land-minus-water rows from validated training endpoints."""

    cache_manifest = WaterbirdsFeatureCacheManifest.model_validate_json(
        cache.manifest.canonical_json()
    )
    pair_manifest = WaterbirdsOraclePairManifest.model_validate_json(
        pairs.manifest.canonical_json()
    )
    if pair_manifest.dataset_manifest_digest != cache_manifest.dataset_manifest_digest:
        raise WaterbirdsFeatureCacheError(
            "oracle pairs and Waterbirds feature cache use different datasets"
        )
    by_id = {record.record_id: record for record in cache_manifest.records}
    left_rows: list[torch.Tensor] = []
    right_rows: list[torch.Tensor] = []
    for relation in pair_manifest.records:
        land = by_id.get(relation.land_record_id)
        water = by_id.get(relation.water_record_id)
        if land is None or water is None:
            raise WaterbirdsFeatureCacheError(
                "oracle endpoint is absent from feature cache"
            )
        if land.split_role != "training" or water.split_role != "training":
            raise WaterbirdsFeatureCacheError("oracle endpoint is outside training")
        if (
            land.image_sha256 != relation.land_image_sha256
            or water.image_sha256 != relation.water_image_sha256
            or land.bird_label != relation.bird_label
            or water.bird_label != relation.bird_label
            or land.background != 0
            or water.background != 1
        ):
            raise WaterbirdsFeatureCacheError(
                "oracle endpoint metadata is inconsistent"
            )
        left_rows.append(cache.features[land.row_index])
        right_rows.append(cache.features[water.row_index])
    return torch.stack(left_rows), torch.stack(right_rows)


def fit_waterbirds_oracle_projection(
    cache: WaterbirdsFeatureCache | WaterbirdsTuningFeatureCache,
    pairs: WaterbirdsOraclePairSet,
    *,
    requested_rank: int,
    relative_singular_value_tolerance: float = 1e-12,
) -> FittedLinearProjection:
    left, right = waterbirds_oracle_pair_features(cache, pairs)
    return fit_linear_projection(
        left,
        right,
        requested_rank=requested_rank,
        pair_manifest_digest=pairs.manifest.canonical_digest(),
        feature_cache_manifest_digest=cache.manifest.canonical_digest(),
        relative_singular_value_tolerance=relative_singular_value_tolerance,
    )


def _normalize_features(
    features: torch.Tensor,
    normalization: Normalization,
) -> torch.Tensor:
    _validate_features(features, int(features.shape[0]))
    result = features.detach().cpu().to(torch.float32)
    if normalization == "l2":
        norms = torch.sqrt(result.square().sum(dim=1, keepdim=True))
        if bool((norms == 0).any()):
            raise ValueError("cannot L2-normalize a zero Waterbirds feature")
        result = result / norms
    return result.contiguous()


def _validate_features(features: torch.Tensor, expected_rows: int) -> None:
    if features.ndim != 2 or tuple(features.shape) != (
        expected_rows,
        FEATURE_DIMENSION,
    ):
        raise ValueError(f"Waterbirds encoder must return shape ({expected_rows}, 512)")
    if not features.is_floating_point() or not bool(torch.isfinite(features).all()):
        raise ValueError("Waterbirds encoder features must be finite floating point")


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return f"sha256:{digest.hexdigest()}"
