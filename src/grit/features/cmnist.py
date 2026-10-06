"""Narrow frozen-feature cache and official OpenAI CLIP adapter for CMNIST."""

from __future__ import annotations

import hashlib
import importlib
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Literal, Protocol, TypeAlias, cast

import numpy as np
import torch
from numpy.typing import NDArray
from PIL import Image
from pydantic import Field, StrictInt, StrictStr, model_validator

from grit.config import (
    OPENAI_CLIP_PREPROCESSING_ID,
    OPENAI_CLIP_REVISION,
    OPENAI_CLIP_WEIGHTS_IDENTITY,
)
from grit.data.cmnist import (
    CmnistConstruction,
    CmnistDatasetManifest,
    CmnistOraclePairManifest,
    CmnistOraclePairSet,
    MnistPool,
    OraclePairRecord,
    RenderedCmnistTable,
)
from grit.data.views import (
    ExampleIdentity,
    FinalTestHandle,
    FinalTestSplitDescriptor,
    FinalTestView,
    issue_final_test_handle,
)
from grit.schemas import (
    HELD_OUT_VALIDATION_NAMES,
    StrictBoundaryModel,
    canonical_digest_value,
)

OPENAI_CLIP_MODEL = "ViT-B/32"
OPENAI_CLIP_WEIGHTS_SHA256 = OPENAI_CLIP_WEIGHTS_IDENTITY.removeprefix("sha256:")
FEATURE_DIMENSION = 512

NonEmptyStr: TypeAlias = Annotated[StrictStr, Field(min_length=1)]
NonNegativeInt: TypeAlias = Annotated[StrictInt, Field(ge=0)]
PositiveStrictInt: TypeAlias = Annotated[StrictInt, Field(gt=0)]
Normalization: TypeAlias = Literal["none", "l2"]
FeatureDevice: TypeAlias = Literal["cpu", "cuda"]
TableRole: TypeAlias = Literal[
    "training", "validation", "final_test", "pair_projection"
]
ArrayField: TypeAlias = Literal[
    "features", "digits", "clean_labels", "targets", "colors"
]


class FeatureCacheValidationError(ValueError):
    """A CMNIST feature cache is missing, damaged, or scientifically mismatched."""


class EncoderIdentity(StrictBoundaryModel):
    implementation: NonEmptyStr
    implementation_revision: NonEmptyStr
    model_name: NonEmptyStr
    weights_identity: NonEmptyStr
    preprocessing_identity: NonEmptyStr
    raw_output_dimension: NonNegativeInt


class FeatureExtractionRuntime(StrictBoundaryModel):
    """Resolved execution provenance for one frozen-feature cache."""

    requested_device: FeatureDevice
    resolved_device: NonEmptyStr
    computation_dtype: Literal["torch.float32"]
    deterministic_algorithms: Literal[True]
    tf32_enabled: Literal[False]
    mixed_precision: Literal[False]
    batch_size: PositiveStrictInt | None
    torch_version: NonEmptyStr
    cuda_runtime_version: NonEmptyStr | None
    device_name: NonEmptyStr
    compute_capability: tuple[NonNegativeInt, NonNegativeInt] | None

    @model_validator(mode="after")
    def _validate_device_details(self) -> FeatureExtractionRuntime:
        if self.requested_device == "cpu":
            if (
                self.resolved_device != "cpu"
                or self.cuda_runtime_version is not None
                or self.compute_capability is not None
                or self.device_name != "cpu"
            ):
                raise ValueError("CPU feature runtime contains CUDA details")
        elif (
            not self.resolved_device.removeprefix("cuda:").isdigit()
            or self.cuda_runtime_version is None
            or self.compute_capability is None
            or self.device_name == "cpu"
        ):
            raise ValueError("CUDA feature runtime is missing resolved CUDA details")
        return self


class ImageEncoder(Protocol):
    @property
    def identity(self) -> EncoderIdentity: ...

    @property
    def extraction_runtime(self) -> FeatureExtractionRuntime: ...

    def encode(self, images: torch.Tensor) -> torch.Tensor: ...


class PilImageEncoder(Protocol):
    @property
    def identity(self) -> EncoderIdentity: ...

    @property
    def extraction_runtime(self) -> FeatureExtractionRuntime: ...

    def encode_pil(self, images: tuple[Image.Image, ...]) -> torch.Tensor: ...


class _ClipModel(Protocol):
    def eval(self) -> object: ...

    def float(self) -> object: ...

    def encode_image(self, images: torch.Tensor) -> torch.Tensor: ...


class _ClipPreprocess(Protocol):
    def __call__(self, image: Image.Image) -> torch.Tensor: ...


class _ClipModule(Protocol):
    def load(
        self,
        name: str,
        device: str,
        jit: bool,
        download_root: str,
    ) -> tuple[_ClipModel, _ClipPreprocess]: ...


class _ToPilImage(Protocol):
    def __call__(self, image: torch.Tensor, mode: str | None = None) -> Image.Image: ...


class _MnistDataset(Protocol):
    data: torch.Tensor
    targets: torch.Tensor

    def __len__(self) -> int: ...


class _MnistFactory(Protocol):
    def __call__(
        self,
        root: str,
        train: bool,
        download: bool,
    ) -> _MnistDataset: ...


class _TensorToNumpy(Protocol):
    def __call__(self, *, force: bool = False) -> NDArray[np.generic]: ...


class _TorchFromNumpy(Protocol):
    def __call__(self, array: NDArray[np.generic]) -> torch.Tensor: ...


_to_pil_image = cast(
    _ToPilImage,
    importlib.import_module("torchvision.transforms.functional").to_pil_image,
)
_mnist_factory = cast(
    _MnistFactory,
    importlib.import_module("torchvision.datasets").MNIST,
)
_torch_from_numpy = cast(_TorchFromNumpy, torch.from_numpy)


@dataclass(slots=True)
class OfficialOpenAiClipEncoder:
    """Pinned official OpenAI CLIP ViT-B/32 with unnormalized raw outputs."""

    weights_root: Path
    allow_download: bool
    device: str = "cpu"
    batch_size: int = 256
    _model: _ClipModel | None = None
    _preprocess: _ClipPreprocess | None = None
    _runtime: FeatureExtractionRuntime | None = None

    @property
    def identity(self) -> EncoderIdentity:
        return EncoderIdentity(
            implementation="openai/CLIP",
            implementation_revision=OPENAI_CLIP_REVISION,
            model_name=OPENAI_CLIP_MODEL,
            weights_identity=OPENAI_CLIP_WEIGHTS_IDENTITY,
            preprocessing_identity=OPENAI_CLIP_PREPROCESSING_ID,
            raw_output_dimension=FEATURE_DIMENSION,
        )

    @property
    def extraction_runtime(self) -> FeatureExtractionRuntime:
        _ = self._resolved_model()
        if self._runtime is None:
            raise AssertionError("resolved CLIP runtime disappeared")
        return FeatureExtractionRuntime.model_validate_json(
            self._runtime.canonical_json()
        )

    def preflight(self) -> None:
        """Resolve the backend and weights before dataset output is written."""

        _ = self._resolved_model()

    def encode(self, images: torch.Tensor) -> torch.Tensor:
        if images.ndim != 4 or int(images.shape[1]) != 3:
            raise ValueError("CLIP input must have shape [N, 3, H, W]")
        if self.batch_size <= 0:
            raise ValueError("CLIP batch_size must be positive")
        if not images.is_floating_point() or not bool(torch.isfinite(images).all()):
            raise ValueError("CLIP input images must be finite floating-point tensors")
        if float(images.min()) < 0.0 or float(images.max()) > 1.0:
            raise ValueError("CLIP input images must lie in [0, 1]")
        pil_images = tuple(_to_pil_image(image) for image in images)
        return self.encode_pil(pil_images)

    def encode_pil(self, images: tuple[Image.Image, ...]) -> torch.Tensor:
        """Encode variable-sized RGB images with the pinned CLIP preprocessing."""

        if not images:
            raise ValueError("CLIP input must contain at least one image")
        if self.batch_size <= 0:
            raise ValueError("CLIP batch_size must be positive")
        model, preprocess = self._resolved_model()
        runtime = self._runtime
        if runtime is None:
            raise AssertionError("resolved CLIP runtime disappeared")
        batches: list[torch.Tensor] = []
        with torch.inference_mode():
            for start in range(0, len(images), self.batch_size):
                stop = min(start + self.batch_size, len(images))
                inputs = torch.stack(
                    [preprocess(image.convert("RGB")) for image in images[start:stop]]
                ).to(device=runtime.resolved_device, dtype=torch.float32)
                batches.append(
                    model.encode_image(inputs).detach().cpu().to(torch.float32)
                )
        features = torch.cat(batches, dim=0)
        _validate_feature_matrix(features, expected_rows=len(images))
        return features

    def model_inputs(self, images: tuple[Image.Image, ...]) -> torch.Tensor:
        """The preprocessed tensors CLIP receives, on CPU, for input checks."""

        _, preprocess = self._resolved_model()
        return torch.stack([preprocess(image.convert("RGB")) for image in images])

    def _resolved_model(self) -> tuple[_ClipModel, _ClipPreprocess]:
        if self._model is not None and self._preprocess is not None:
            return self._model, self._preprocess
        resolved_device, runtime = self._resolve_runtime()
        self.weights_root.mkdir(parents=True, exist_ok=True)
        weights_path = self.weights_root / "ViT-B-32.pt"
        if not self.allow_download and not weights_path.is_file():
            raise FileNotFoundError(
                "official OpenAI CLIP weights are absent; rerun preparation with "
                "--allow-download"
            )
        module = cast(_ClipModule, importlib.import_module("clip"))
        model, preprocess = module.load(
            OPENAI_CLIP_MODEL,
            device=resolved_device,
            jit=False,
            download_root=str(self.weights_root),
        )
        if _file_sha256(weights_path) != f"sha256:{OPENAI_CLIP_WEIGHTS_SHA256}":
            raise FeatureCacheValidationError(
                "official OpenAI CLIP ViT-B/32 weight digest does not match"
            )
        _ = model.float()
        _ = model.eval()
        self._model = model
        self._preprocess = preprocess
        self._runtime = runtime
        return model, preprocess

    def _resolve_runtime(self) -> tuple[str, FeatureExtractionRuntime]:
        if self.batch_size <= 0:
            raise ValueError("CLIP batch_size must be positive")
        torch.use_deterministic_algorithms(True)
        torch.set_float32_matmul_precision("highest")
        if self.device == "cpu":
            return "cpu", _cpu_feature_runtime(batch_size=self.batch_size)
        if not (self.device == "cuda" or self.device.startswith("cuda:")):
            raise ValueError(f"device must be cpu, cuda, or cuda:N, got {self.device}")

        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA feature preparation was requested but CUDA is unavailable"
            )
        device_index = cuda_device_index(self.device)
        if device_index >= torch.cuda.device_count():
            raise RuntimeError(
                f"requested {self.device} but only "
                f"{torch.cuda.device_count()} CUDA device(s) are visible"
            )
        torch.cuda.set_device(device_index)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        cuda_runtime = cast(str | None, getattr(torch.version, "cuda", None))
        if not cuda_runtime:
            raise RuntimeError("CUDA PyTorch build does not report a CUDA runtime")
        capability = torch.cuda.get_device_capability(device_index)
        resolved = f"cuda:{device_index}"
        return resolved, FeatureExtractionRuntime(
            requested_device="cuda",
            resolved_device=resolved,
            computation_dtype="torch.float32",
            deterministic_algorithms=True,
            tf32_enabled=False,
            mixed_precision=False,
            batch_size=self.batch_size,
            torch_version=str(torch.__version__),
            cuda_runtime_version=cuda_runtime,
            device_name=torch.cuda.get_device_name(device_index),
            compute_capability=(int(capability[0]), int(capability[1])),
        )


def cuda_device_index(device: str) -> int:
    """`cuda` means `cuda:0`; `cuda:N` selects that visible device."""

    suffix = device.removeprefix("cuda")
    if suffix == "":
        return 0
    if not suffix.startswith(":") or not suffix[1:].isdigit():
        raise ValueError(f"device must be cpu, cuda, or cuda:N, got {device}")
    return int(suffix[1:])


@dataclass(frozen=True, slots=True)
class DeterministicFakeEncoder:
    """Explicitly non-reportable 512-D encoder used only by smoke/tests."""

    seed: int = 0

    @property
    def identity(self) -> EncoderIdentity:
        return EncoderIdentity(
            implementation="grit.synthetic",
            implementation_revision="deterministic-fake-encoder-v1",
            model_name="fake-512",
            weights_identity=f"seed:{self.seed}",
            preprocessing_identity="identity-rgb",
            raw_output_dimension=FEATURE_DIMENSION,
        )

    @property
    def extraction_runtime(self) -> FeatureExtractionRuntime:
        return _cpu_feature_runtime(batch_size=None)

    def encode(self, images: torch.Tensor) -> torch.Tensor:
        if images.ndim != 4 or int(images.shape[1]) != 3:
            raise ValueError("fake encoder input must have shape [N, 3, H, W]")
        flattened = images.detach().cpu().to(torch.float32).flatten(start_dim=1)
        generator = torch.Generator(device="cpu").manual_seed(self.seed)
        weights = torch.randn(
            (int(flattened.shape[1]), FEATURE_DIMENSION),
            generator=generator,
            dtype=torch.float32,
        )
        features = flattened @ weights / max(int(flattened.shape[1]), 1) ** 0.5
        features[:, 0] += 1.0
        _validate_feature_matrix(features, expected_rows=int(images.shape[0]))
        return features


class ArrayFileManifest(StrictBoundaryModel):
    field_name: Literal["features", "digits", "clean_labels", "targets", "colors"]
    relative_path: NonEmptyStr
    sha256: NonEmptyStr
    shape: tuple[NonNegativeInt, ...]
    dtype: NonEmptyStr


class FeatureTableManifest(StrictBoundaryModel):
    table_name: NonEmptyStr
    role: TableRole
    source_ids: tuple[NonEmptyStr, ...]
    source_ids_digest: NonEmptyStr
    row_count: NonNegativeInt
    feature_dimension: Literal[512]
    feature_dtype: Literal["float32"]
    files: tuple[ArrayFileManifest, ...]

    @model_validator(mode="after")
    def _validate_table(self) -> FeatureTableManifest:
        if len(self.source_ids) != self.row_count:
            raise ValueError("feature table source count is inconsistent")
        if len(set(self.source_ids)) != len(self.source_ids):
            raise ValueError("feature table source IDs must be unique")
        if self.source_ids_digest != canonical_digest_value(self.source_ids):
            raise ValueError("feature table source digest is inconsistent")
        expected_fields = ("features", "digits", "clean_labels", "targets", "colors")
        if tuple(file.field_name for file in self.files) != expected_fields:
            raise ValueError("feature table files must use canonical field ordering")
        expected_shapes = ((self.row_count, 512),) + ((self.row_count,),) * 4
        expected_dtypes = ("float32", "int64", "int64", "int64", "int64")
        if tuple(file.shape for file in self.files) != expected_shapes:
            raise ValueError("feature table array shapes are inconsistent")
        if tuple(file.dtype for file in self.files) != expected_dtypes:
            raise ValueError("feature table array dtypes are inconsistent")
        relative_paths = tuple(file.relative_path for file in self.files)
        if len(set(relative_paths)) != len(relative_paths):
            raise ValueError("feature table array paths must be unique")
        return self


class CmnistFeatureCacheManifest(StrictBoundaryModel):
    schema_version: Literal["grit.cmnist-features/v2"]
    dataset_id: Literal["cmnist"]
    source_manifest_digest: NonEmptyStr
    pair_manifest_digest: NonEmptyStr
    encoder: EncoderIdentity
    extraction_runtime: FeatureExtractionRuntime
    normalization: Normalization
    feature_dimension: Literal[512]
    feature_dtype: Literal["float32"]
    tables: tuple[FeatureTableManifest, ...]

    @model_validator(mode="after")
    def _validate_tables(self) -> CmnistFeatureCacheManifest:
        names = tuple(table.table_name for table in self.tables)
        held_out = names[4] if len(names) > 4 else None
        expected = (
            "train_e01",
            "train_e02",
            "val_e01",
            "val_e02",
            held_out,
            "test_ood",
            "oracle_pair_red",
            "oracle_pair_green",
        )
        if held_out not in HELD_OUT_VALIDATION_NAMES or names != expected:
            raise ValueError("CMNIST feature cache must contain the fixed table set")
        expected_roles: tuple[TableRole, ...] = (
            "training",
            "training",
            "validation",
            "validation",
            "validation",
            "final_test",
            "pair_projection",
            "pair_projection",
        )
        if tuple(table.role for table in self.tables) != expected_roles:
            raise ValueError("CMNIST feature table roles are inconsistent")
        validation_sources = {table.source_ids for table in self.tables[2:5]}
        if len(validation_sources) != 1:
            raise ValueError("CMNIST cached validation tables must share source order")
        if self.tables[6].source_ids != self.tables[7].source_ids:
            raise ValueError(
                "CMNIST cached oracle pair endpoints must share source order"
            )
        training_sources = set(self.tables[0].source_ids) | set(
            self.tables[1].source_ids
        )
        if not set(self.tables[6].source_ids) <= training_sources:
            raise ValueError("CMNIST cached oracle pairs must use training sources")
        if (
            self.encoder.implementation == "openai/CLIP"
            and self.extraction_runtime.batch_size is None
        ):
            raise ValueError("official CLIP feature caches require a batch size")
        return self


@dataclass(frozen=True, slots=True)
class FeatureTable:
    name: str
    role: TableRole
    source_ids: tuple[str, ...]
    features: torch.Tensor
    digits: torch.Tensor
    clean_labels: torch.Tensor
    targets: torch.Tensor
    colors: torch.Tensor


@dataclass(frozen=True, slots=True)
class CmnistFeatureCache:
    train_e01: FeatureTable
    train_e02: FeatureTable
    val_e01: FeatureTable
    val_e02: FeatureTable
    val_held_out: FeatureTable
    _test_ood: FeatureTable
    oracle_pair_red: FeatureTable | None
    oracle_pair_green: FeatureTable | None
    manifest: CmnistFeatureCacheManifest
    root: Path

    def training_tables(self) -> tuple[FeatureTable, FeatureTable]:
        return self.train_e01, self.train_e02

    def validation_tables(self) -> tuple[FeatureTable, FeatureTable, FeatureTable]:
        return self.val_e01, self.val_e02, self.val_held_out

    def pair_tables(self) -> tuple[FeatureTable, FeatureTable]:
        if self.oracle_pair_red is None or self.oracle_pair_green is None:
            raise ValueError("this cache was opened without pair access")
        return self.oracle_pair_red, self.oracle_pair_green

    def issue_final_handle(
        self,
        *,
        run_id: str,
        candidate_id: str,
        scientific_config_digest: str,
    ) -> FinalTestHandle:
        """Issue identities only; cached test features remain unopened."""

        descriptor, examples = self._test_descriptor_and_examples()
        return issue_final_test_handle(
            handle_id=f"final-handle:{run_id}",
            run_id=run_id,
            candidate_id=candidate_id,
            scientific_config_digest=scientific_config_digest,
            descriptor=descriptor,
            examples=examples,
        )

    def open_final_table(self, view: FinalTestView) -> FeatureTable:
        if view.descriptor.name != "test_ood":
            raise ValueError(
                "final feature access requires an authorized test_ood view"
            )
        return self._authorized_test_table(view.descriptor, view.examples)

    def _test_descriptor_and_examples(
        self,
    ) -> tuple[FinalTestSplitDescriptor, tuple[ExampleIdentity, ...]]:
        descriptor = FinalTestSplitDescriptor(
            dataset_id="cmnist",
            manifest_id=self.manifest.canonical_digest(),
            name="test_ood",
            role="final_test",
            source_partition_id="test_sources",
            view_id="test_ood",
        )
        examples = tuple(
            ExampleIdentity(
                example_id=f"{source_id}:view:test_ood",
                source_id=source_id,
                view_id="test_ood",
            )
            for source_id in self._test_ood.source_ids
        )
        return descriptor, examples

    def _authorized_test_table(
        self,
        descriptor: FinalTestSplitDescriptor,
        examples: tuple[ExampleIdentity, ...],
    ) -> FeatureTable:
        if descriptor.manifest_id != self.manifest.canonical_digest():
            raise FeatureCacheValidationError(
                "authorized test view does not match the feature cache manifest"
            )
        expected = tuple(example.source_id for example in examples)
        if expected != self._test_ood.source_ids:
            raise FeatureCacheValidationError(
                "authorized test view does not match cached test source identities"
            )
        return self._test_ood


@dataclass(frozen=True, slots=True)
class CmnistTuningFeatureCache:
    """Verified training, validation, and pair tables with no final-test capability."""

    train_e01: FeatureTable
    train_e02: FeatureTable
    val_e01: FeatureTable
    val_e02: FeatureTable
    val_held_out: FeatureTable
    oracle_pair_red: FeatureTable | None
    oracle_pair_green: FeatureTable | None
    manifest: CmnistFeatureCacheManifest
    root: Path

    def training_tables(self) -> tuple[FeatureTable, FeatureTable]:
        return self.train_e01, self.train_e02

    def validation_tables(self) -> tuple[FeatureTable, FeatureTable, FeatureTable]:
        return self.val_e01, self.val_e02, self.val_held_out

    def pair_tables(self) -> tuple[FeatureTable, FeatureTable]:
        if self.oracle_pair_red is None or self.oracle_pair_green is None:
            raise ValueError("this cache was opened without pair access")
        return self.oracle_pair_red, self.oracle_pair_green


def prepare_cmnist_feature_cache(
    construction: CmnistConstruction,
    pairs: CmnistOraclePairSet,
    encoder: ImageEncoder,
    output_dir: Path,
    *,
    normalization: Normalization,
    overwrite: bool = False,
) -> CmnistFeatureCacheManifest:
    """Encode and write the fixed CMNIST environment/pair feature tables."""

    _validate_pairs_for_feature_preparation(construction, pairs)
    if output_dir.exists() and any(output_dir.iterdir()) and not overwrite:
        raise FileExistsError(
            f"feature cache directory is not empty: {output_dir}; pass overwrite=True"
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    table_inputs = tuple(
        _from_rendered_table(table) for table in construction.preparation_tables()
    ) + (
        _from_pair_endpoint(pairs, "oracle_pair_red", "red"),
        _from_pair_endpoint(pairs, "oracle_pair_green", "green"),
    )
    manifests: list[FeatureTableManifest] = []
    for table in table_inputs:
        raw_features = encoder.encode(table.images)
        features = _normalized_features(raw_features, normalization)
        manifests.append(_write_feature_table(output_dir, table, features))
    manifest = CmnistFeatureCacheManifest(
        schema_version="grit.cmnist-features/v2",
        dataset_id="cmnist",
        source_manifest_digest=construction.manifest.canonical_digest(),
        pair_manifest_digest=pairs.manifest.canonical_digest(),
        encoder=encoder.identity,
        extraction_runtime=encoder.extraction_runtime,
        normalization=normalization,
        feature_dimension=FEATURE_DIMENSION,
        feature_dtype="float32",
        tables=tuple(manifests),
    )
    (output_dir / "manifest.json").write_text(
        manifest.canonical_json() + "\n", encoding="utf-8"
    )
    return manifest


def _cpu_feature_runtime(*, batch_size: int | None) -> FeatureExtractionRuntime:
    return FeatureExtractionRuntime(
        requested_device="cpu",
        resolved_device="cpu",
        computation_dtype="torch.float32",
        deterministic_algorithms=True,
        tf32_enabled=False,
        mixed_precision=False,
        batch_size=batch_size,
        torch_version=str(torch.__version__),
        cuda_runtime_version=None,
        device_name="cpu",
        compute_capability=None,
    )


def _validate_pairs_for_feature_preparation(
    construction: CmnistConstruction,
    pairs: CmnistOraclePairSet,
) -> None:
    """Reject mixed or malformed pair inputs before the cache writes any files."""

    try:
        dataset_manifest = CmnistDatasetManifest.model_validate(
            construction.manifest.model_dump(mode="python")
        )
        pair_manifest = CmnistOraclePairManifest.model_validate(
            pairs.manifest.model_dump(mode="python")
        )
        pair_records = tuple(
            OraclePairRecord.model_validate(record.model_dump(mode="python"))
            for record in pairs.records
        )
    except ValueError as error:
        raise FeatureCacheValidationError(
            "CMNIST construction or oracle-pair metadata is invalid"
        ) from error

    dataset_manifest_digest = dataset_manifest.canonical_digest()
    if pair_manifest.dataset_manifest_digest != dataset_manifest_digest:
        raise FeatureCacheValidationError(
            "oracle pair dataset manifest does not match the CMNIST construction"
        )
    if pair_records != pair_manifest.records:
        raise FeatureCacheValidationError(
            "oracle pair-set records do not match the pair manifest"
        )
    pair_count = len(pair_records)
    if pair_count != pair_manifest.realized_count:
        raise FeatureCacheValidationError(
            "oracle pair-set record count does not match the pair manifest"
        )
    if (
        pairs.left_red.ndim != 4
        or pairs.right_green.ndim != 4
        or pairs.left_red.shape != pairs.right_green.shape
        or int(pairs.left_red.shape[0]) != pair_count
    ):
        raise FeatureCacheValidationError(
            "oracle pair endpoint rows must align with the pair records"
        )
    if not bool(torch.isfinite(pairs.left_red).all()) or not bool(
        torch.isfinite(pairs.right_green).all()
    ):
        raise FeatureCacheValidationError("oracle pair endpoints must be finite")

    source_indices: dict[str, int] = {}
    digits: dict[str, int] = {}
    clean_labels: dict[str, int] = {}
    noisy_targets: dict[str, int] = {}
    grayscale: dict[str, torch.Tensor] = {}
    training_inputs = (
        (
            construction.train_e01,
            construction.partitions.train_e01_source_indices,
            dataset_manifest.partition_manifest.partitions[0].source_indices,
        ),
        (
            construction.train_e02,
            construction.partitions.train_e02_source_indices,
            dataset_manifest.partition_manifest.partitions[1].source_indices,
        ),
    )
    for table, indices, manifested_indices in training_inputs:
        row_count = len(table.source_ids)
        if (
            indices != manifested_indices
            or len(indices) != row_count
            or table.images.ndim != 4
            or int(table.images.shape[1]) != 3
            or int(table.images.shape[0]) != row_count
            or int(table.digits.shape[0]) != row_count
            or int(table.clean_labels.shape[0]) != row_count
            or int(table.targets.shape[0]) != row_count
        ):
            raise FeatureCacheValidationError(
                "CMNIST construction training rows are internally inconsistent"
            )
        recovered = table.images[:, 0] + table.images[:, 1]
        for row, (source_id, source_index) in enumerate(
            zip(table.source_ids, indices, strict=True)
        ):
            if source_id != f"mnist:train:{source_index}":
                raise FeatureCacheValidationError(
                    "CMNIST training source identity does not match its source index"
                )
            if source_id in source_indices:
                raise FeatureCacheValidationError(
                    "CMNIST training source identities must be unique"
                )
            source_indices[source_id] = source_index
            digits[source_id] = int(table.digits[row].item())
            clean_labels[source_id] = int(table.clean_labels[row].item())
            noisy_targets[source_id] = int(table.targets[row].item())
            grayscale[source_id] = recovered[row]

    for row, record in enumerate(pair_records):
        source_id = str(record.source_id)
        if source_id not in source_indices:
            raise FeatureCacheValidationError(
                "oracle pair record does not name a selected training source"
            )
        observed_metadata = (
            int(record.official_source_index),
            int(record.digit),
            int(record.clean_label),
            int(record.noisy_target),
        )
        expected_metadata = (
            source_indices[source_id],
            digits[source_id],
            clean_labels[source_id],
            noisy_targets[source_id],
        )
        if observed_metadata != expected_metadata:
            raise FeatureCacheValidationError(
                "oracle pair metadata does not match its CMNIST training source"
            )
        gray = grayscale[source_id]
        red = pairs.left_red[row]
        green = pairs.right_green[row]
        zeros = torch.zeros_like(gray)
        if (
            red.shape != green.shape
            or red.ndim != 3
            or int(red.shape[0]) != 3
            or red.shape[1:] != gray.shape
            or not torch.equal(red[0], gray)
            or not torch.equal(green[1], gray)
            or not torch.equal(red[1], zeros)
            or not torch.equal(red[2], zeros)
            or not torch.equal(green[0], zeros)
            or not torch.equal(green[2], zeros)
        ):
            raise FeatureCacheValidationError(
                "oracle pair endpoints violate the clean red/green recoloring invariant"
            )


def load_cmnist_feature_cache(
    root: Path,
    *,
    expected_source_manifest_digest: str | None = None,
    expected_pair_manifest_digest: str | None = None,
    expected_normalization: Normalization | None = None,
    include_pairs: bool = True,
) -> CmnistFeatureCache:
    """Load a verified CMNIST cache and fail helpfully on absent/mixed artifacts."""

    manifest = _load_cmnist_feature_manifest(
        root,
        expected_source_manifest_digest=expected_source_manifest_digest,
        expected_pair_manifest_digest=expected_pair_manifest_digest,
        expected_normalization=expected_normalization,
    )
    loaded = tuple(
        _load_feature_table(root, table)
        for table in manifest.tables
        if include_pairs or table.role != "pair_projection"
    )
    return CmnistFeatureCache(
        train_e01=loaded[0],
        train_e02=loaded[1],
        val_e01=loaded[2],
        val_e02=loaded[3],
        val_held_out=loaded[4],
        _test_ood=loaded[5],
        oracle_pair_red=loaded[6] if include_pairs else None,
        oracle_pair_green=loaded[7] if include_pairs else None,
        manifest=manifest,
        root=root,
    )


def load_cmnist_tuning_feature_cache(
    root: Path,
    *,
    expected_source_manifest_digest: str | None = None,
    expected_pair_manifest_digest: str | None = None,
    expected_normalization: Normalization | None = None,
    include_pairs: bool = True,
) -> CmnistTuningFeatureCache:
    """Load only training, validation, and pair tables for bounded tuning."""

    manifest = _load_cmnist_feature_manifest(
        root,
        expected_source_manifest_digest=expected_source_manifest_digest,
        expected_pair_manifest_digest=expected_pair_manifest_digest,
        expected_normalization=expected_normalization,
    )
    selected = tuple(
        _load_feature_table(root, table)
        for table in manifest.tables
        if table.role != "final_test"
        and (include_pairs or table.role != "pair_projection")
    )
    return CmnistTuningFeatureCache(
        train_e01=selected[0],
        train_e02=selected[1],
        val_e01=selected[2],
        val_e02=selected[3],
        val_held_out=selected[4],
        oracle_pair_red=selected[5] if include_pairs else None,
        oracle_pair_green=selected[6] if include_pairs else None,
        manifest=manifest,
        root=root,
    )


def _load_cmnist_feature_manifest(
    root: Path,
    *,
    expected_source_manifest_digest: str | None,
    expected_pair_manifest_digest: str | None,
    expected_normalization: Normalization | None,
) -> CmnistFeatureCacheManifest:
    manifest_path = root / "manifest.json"
    if not manifest_path.is_file():
        raise FeatureCacheValidationError(
            f"CMNIST feature cache manifest is missing: {manifest_path}"
        )
    try:
        manifest = CmnistFeatureCacheManifest.model_validate_json(
            manifest_path.read_text(encoding="utf-8")
        )
    except (OSError, ValueError) as error:
        raise FeatureCacheValidationError(
            f"CMNIST feature cache manifest is invalid: {manifest_path}"
        ) from error
    if (
        expected_source_manifest_digest is not None
        and manifest.source_manifest_digest != expected_source_manifest_digest
    ):
        raise FeatureCacheValidationError(
            "feature cache source manifest does not match"
        )
    if (
        expected_pair_manifest_digest is not None
        and manifest.pair_manifest_digest != expected_pair_manifest_digest
    ):
        raise FeatureCacheValidationError("feature cache pair manifest does not match")
    if (
        expected_normalization is not None
        and manifest.normalization != expected_normalization
    ):
        raise FeatureCacheValidationError("feature cache normalization does not match")
    return manifest


def load_torchvision_mnist_pools(
    data_root: Path,
    *,
    allow_download: bool,
) -> tuple[MnistPool, MnistPool]:
    """Explicit disk/download adapter; core construction accepts injected pools."""

    try:
        train = _mnist_factory(root=str(data_root), train=True, download=allow_download)
        test = _mnist_factory(root=str(data_root), train=False, download=allow_download)
    except RuntimeError as error:
        raise FileNotFoundError(
            "MNIST artifacts are absent; rerun preparation with --allow-download"
        ) from error
    return (
        MnistPool(
            official_split="train",
            source_indices=torch.arange(len(train), dtype=torch.int64),
            images=train.data,
            digits=train.targets,
        ),
        MnistPool(
            official_split="test",
            source_indices=torch.arange(len(test), dtype=torch.int64),
            images=test.data,
            digits=test.targets,
        ),
    )


@dataclass(frozen=True, slots=True)
class _TableInput:
    name: str
    role: TableRole
    source_ids: tuple[str, ...]
    images: torch.Tensor
    digits: torch.Tensor
    clean_labels: torch.Tensor
    targets: torch.Tensor
    colors: torch.Tensor


def _from_rendered_table(table: RenderedCmnistTable) -> _TableInput:
    return _TableInput(
        name=table.name,
        role=table.role,
        source_ids=table.source_ids,
        images=table.images,
        digits=table.digits,
        clean_labels=table.clean_labels,
        targets=table.targets,
        colors=table.colors,
    )


def _from_pair_endpoint(
    pairs: CmnistOraclePairSet,
    name: Literal["oracle_pair_red", "oracle_pair_green"],
    color: Literal["red", "green"],
) -> _TableInput:
    is_red = color == "red"
    source_ids = tuple(record.source_id for record in pairs.records)
    return _TableInput(
        name=name,
        role="pair_projection",
        source_ids=source_ids,
        images=pairs.left_red if is_red else pairs.right_green,
        digits=torch.tensor([record.digit for record in pairs.records]),
        clean_labels=torch.tensor([record.clean_label for record in pairs.records]),
        targets=torch.tensor([record.noisy_target for record in pairs.records]),
        colors=torch.full((len(pairs.records),), 0 if is_red else 1),
    )


def _normalized_features(
    features: torch.Tensor,
    normalization: Normalization,
) -> torch.Tensor:
    _validate_feature_matrix(features, expected_rows=int(features.shape[0]))
    result = features.detach().cpu().to(torch.float32)
    if normalization == "l2":
        norms = torch.sqrt(result.square().sum(dim=1, keepdim=True))
        if bool((norms == 0).any()):
            raise ValueError("cannot L2-normalize a zero feature vector")
        result = result / norms
    return result.contiguous()


def _validate_feature_matrix(features: torch.Tensor, *, expected_rows: int) -> None:
    if features.ndim != 2 or tuple(features.shape) != (
        expected_rows,
        FEATURE_DIMENSION,
    ):
        raise ValueError(
            f"encoder must return shape ({expected_rows}, {FEATURE_DIMENSION})"
        )
    if not features.is_floating_point() or not bool(torch.isfinite(features).all()):
        raise ValueError("encoder features must be finite floating-point values")


def _write_feature_table(
    root: Path,
    table: _TableInput,
    features: torch.Tensor,
) -> FeatureTableManifest:
    table_dir = root / table.name
    table_dir.mkdir(parents=True, exist_ok=True)
    arrays: tuple[tuple[ArrayField, torch.Tensor], ...] = (
        ("features", features),
        ("digits", table.digits.to(torch.int64)),
        ("clean_labels", table.clean_labels.to(torch.int64)),
        ("targets", table.targets.to(torch.int64)),
        ("colors", table.colors.to(torch.int64)),
    )
    files: list[ArrayFileManifest] = []
    for field_name, tensor in arrays:
        path = table_dir / f"{field_name}.npy"
        to_numpy = cast(_TensorToNumpy, tensor.detach().cpu().numpy)
        array = to_numpy()
        np.save(path, array, allow_pickle=False)
        files.append(
            ArrayFileManifest(
                field_name=field_name,
                relative_path=path.relative_to(root).as_posix(),
                sha256=_file_sha256(path),
                shape=tuple(int(value) for value in tensor.shape),
                dtype=str(array.dtype),
            )
        )
    return FeatureTableManifest(
        table_name=table.name,
        role=table.role,
        source_ids=table.source_ids,
        source_ids_digest=canonical_digest_value(table.source_ids),
        row_count=len(table.source_ids),
        feature_dimension=FEATURE_DIMENSION,
        feature_dtype="float32",
        files=tuple(files),
    )


def _load_feature_table(root: Path, manifest: FeatureTableManifest) -> FeatureTable:
    loaded: dict[str, torch.Tensor] = {}
    for file in manifest.files:
        path = root / file.relative_path
        if not path.is_file():
            raise FeatureCacheValidationError(f"feature cache file is missing: {path}")
        if _file_sha256(path) != file.sha256:
            raise FeatureCacheValidationError(
                f"feature cache file digest mismatch: {path}"
            )
        array = np.load(path, allow_pickle=False)
        if tuple(int(value) for value in array.shape) != file.shape:
            raise FeatureCacheValidationError(
                f"feature cache file shape mismatch: {path}"
            )
        if str(array.dtype) != file.dtype:
            raise FeatureCacheValidationError(
                f"feature cache file dtype mismatch: {path}"
            )
        copied = cast(NDArray[np.generic], np.array(array, copy=True))
        loaded[file.field_name] = _torch_from_numpy(copied)
    features = loaded["features"].to(torch.float32)
    _validate_feature_matrix(features, expected_rows=manifest.row_count)
    digits = loaded["digits"].to(torch.int64)
    clean_labels = loaded["clean_labels"].to(torch.int64)
    targets = loaded["targets"].to(torch.int64)
    colors = loaded["colors"].to(torch.int64)
    if bool(((digits < 0) | (digits > 9)).any()):
        raise FeatureCacheValidationError("feature cache digit labels are invalid")
    for name, values in (
        ("clean_labels", clean_labels),
        ("targets", targets),
        ("colors", colors),
    ):
        if bool(((values < 0) | (values > 1)).any()):
            raise FeatureCacheValidationError(
                f"feature cache {name} must contain binary values"
            )
    return FeatureTable(
        name=manifest.table_name,
        role=manifest.role,
        source_ids=tuple(manifest.source_ids),
        features=features,
        digits=digits,
        clean_labels=clean_labels,
        targets=targets,
        colors=colors,
    )


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return f"sha256:{digest.hexdigest()}"
