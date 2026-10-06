"""Offline feature-cache and encoder-boundary tests for CMNIST."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest
import torch
from PIL import Image
from pydantic import ValidationError

import grit.features.cmnist as feature_module
from grit.data.cmnist import (
    CmnistConstruction,
    CmnistOraclePairSet,
    CmnistPartitionTargets,
    MnistPool,
    build_clean_oracle_pairs,
    construct_cmnist,
    pair_source_view,
)
from grit.data.views import ExampleIdentity, FinalTestSplitDescriptor, FinalTestView
from grit.features.cmnist import (
    CmnistFeatureCacheManifest,
    DeterministicFakeEncoder,
    FeatureCacheValidationError,
    FeatureExtractionRuntime,
    OfficialOpenAiClipEncoder,
    load_cmnist_feature_cache,
    load_cmnist_tuning_feature_cache,
    prepare_cmnist_feature_cache,
)


def _construction_and_pairs(
    *,
    construction_seed: int = 4,
) -> tuple[CmnistConstruction, CmnistOraclePairSet]:
    train_count = 50
    test_count = 10
    train_indices = torch.arange(train_count, dtype=torch.int64)
    test_indices = torch.arange(test_count, dtype=torch.int64)
    train = MnistPool(
        official_split="train",
        source_indices=train_indices,
        images=(train_indices % 17).to(torch.float32).reshape(-1, 1, 1).div(16),
        digits=train_indices.remainder(10),
    )
    test = MnistPool(
        official_split="test",
        source_indices=test_indices,
        images=(test_indices % 9).to(torch.float32).reshape(-1, 1, 1).div(8),
        digits=test_indices.remainder(10),
    )
    targets = CmnistPartitionTargets(
        train_e01=20,
        train_e02=20,
        validation=10,
        test=10,
    )
    construction = construct_cmnist(
        train,
        test,
        construction_seed=construction_seed,
        targets=targets,
    )
    pairs = build_clean_oracle_pairs(
        pair_source_view(construction, train),
        pair_seed=8,
        pair_count=16,
    )
    return construction, pairs


def test_feature_preparation_rejects_pairs_from_another_construction(
    tmp_path: Path,
) -> None:
    construction, _ = _construction_and_pairs(construction_seed=4)
    _, other_pairs = _construction_and_pairs(construction_seed=5)
    output = tmp_path / "mixed"
    with pytest.raises(FeatureCacheValidationError, match="dataset manifest"):
        prepare_cmnist_feature_cache(
            construction,
            other_pairs,
            DeterministicFakeEncoder(),
            output,
            normalization="none",
        )
    assert not output.exists()


def test_feature_preparation_rejects_malformed_pairs_before_writing(
    tmp_path: Path,
) -> None:
    construction, pairs = _construction_and_pairs()
    first = pairs.records[0]
    changed_record = first.model_copy(update={"digit": (int(first.digit) + 1) % 10})
    malformed_records = replace(
        pairs,
        records=(changed_record, *pairs.records[1:]),
    )
    records_output = tmp_path / "malformed-records"
    with pytest.raises(FeatureCacheValidationError, match="records do not match"):
        prepare_cmnist_feature_cache(
            construction,
            malformed_records,
            DeterministicFakeEncoder(),
            records_output,
            normalization="none",
        )
    assert not records_output.exists()

    malformed_metadata = replace(
        pairs,
        records=(changed_record, *pairs.records[1:]),
        manifest=pairs.manifest.model_copy(
            update={"records": (changed_record, *pairs.records[1:])}
        ),
    )
    metadata_output = tmp_path / "malformed-metadata"
    with pytest.raises(FeatureCacheValidationError, match="metadata does not match"):
        prepare_cmnist_feature_cache(
            construction,
            malformed_metadata,
            DeterministicFakeEncoder(),
            metadata_output,
            normalization="none",
        )
    assert not metadata_output.exists()

    changed_red = pairs.left_red.clone()
    changed_red[0, 0, 0, 0] += 0.25
    malformed_endpoints = replace(pairs, left_red=changed_red)
    endpoints_output = tmp_path / "malformed-endpoints"
    with pytest.raises(FeatureCacheValidationError, match="recoloring invariant"):
        prepare_cmnist_feature_cache(
            construction,
            malformed_endpoints,
            DeterministicFakeEncoder(),
            endpoints_output,
            normalization="none",
        )
    assert not endpoints_output.exists()


def test_fake_feature_cache_is_canonical_validated_and_offline(tmp_path: Path) -> None:
    construction_object, pairs_object = _construction_and_pairs()
    first_root = tmp_path / "first"
    second_root = tmp_path / "second"
    first = prepare_cmnist_feature_cache(
        construction_object,
        pairs_object,
        DeterministicFakeEncoder(seed=13),
        first_root,
        normalization="none",
    )
    second = prepare_cmnist_feature_cache(
        construction_object,
        pairs_object,
        DeterministicFakeEncoder(seed=13),
        second_root,
        normalization="none",
    )
    assert first == second
    assert (
        CmnistFeatureCacheManifest.model_validate_json(first.canonical_json()) == first
    )
    cache = load_cmnist_feature_cache(
        first_root,
        expected_source_manifest_digest=construction_object.manifest.canonical_digest(),
        expected_pair_manifest_digest=pairs_object.manifest.canonical_digest(),
        expected_normalization="none",
    )
    assert tuple(table.name for table in cache.training_tables()) == (
        "train_e01",
        "train_e02",
    )
    assert tuple(table.name for table in cache.validation_tables()) == (
        "val_e01",
        "val_e02",
        "val_e05",
    )
    assert cache.train_e01.features.shape == (20, 512)
    red, green = cache.pair_tables()
    assert red.source_ids == green.source_ids
    assert not torch.equal(red.features, green.features)


def test_tuning_cache_does_not_materialize_or_expose_final_table(
    tmp_path: Path,
) -> None:
    construction, pairs = _construction_and_pairs()
    root = tmp_path / "tuning"
    manifest = prepare_cmnist_feature_cache(
        construction,
        pairs,
        DeterministicFakeEncoder(seed=13),
        root,
        normalization="none",
    )
    tuning = load_cmnist_tuning_feature_cache(
        root,
        expected_source_manifest_digest=manifest.source_manifest_digest,
        expected_pair_manifest_digest=manifest.pair_manifest_digest,
        expected_normalization="none",
    )
    assert len(tuning.training_tables()) == 2
    assert len(tuning.validation_tables()) == 3
    assert len(tuning.pair_tables()) == 2
    assert not hasattr(tuning, "issue_final_handle")
    assert not hasattr(tuning, "open_final_table")


def test_l2_cache_is_explicit_and_cannot_mix_with_primary(tmp_path: Path) -> None:
    construction_object, pairs_object = _construction_and_pairs()
    root = tmp_path / "normalized"
    prepare_cmnist_feature_cache(
        construction_object,
        pairs_object,
        DeterministicFakeEncoder(seed=2),
        root,
        normalization="l2",
    )
    cache = load_cmnist_feature_cache(root, expected_normalization="l2")
    norms = torch.sqrt(cache.train_e01.features.square().sum(dim=1))
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-6, rtol=0.0)
    with pytest.raises(FeatureCacheValidationError, match="normalization"):
        load_cmnist_feature_cache(root, expected_normalization="none")


def test_final_view_is_bound_to_exact_feature_cache_manifest(tmp_path: Path) -> None:
    construction, pairs = _construction_and_pairs()
    first_root = tmp_path / "first"
    second_root = tmp_path / "second"
    prepare_cmnist_feature_cache(
        construction,
        pairs,
        DeterministicFakeEncoder(seed=1),
        first_root,
        normalization="none",
    )
    prepare_cmnist_feature_cache(
        construction,
        pairs,
        DeterministicFakeEncoder(seed=2),
        second_root,
        normalization="none",
    )
    first = load_cmnist_feature_cache(first_root)
    second = load_cmnist_feature_cache(second_root)
    source_ids = first.manifest.tables[5].source_ids
    assert source_ids == second.manifest.tables[5].source_ids
    assert first.manifest.canonical_digest() != second.manifest.canonical_digest()
    view = FinalTestView(
        descriptor=FinalTestSplitDescriptor(
            dataset_id="cmnist",
            manifest_id=first.manifest.canonical_digest(),
            name="test_ood",
            role="final_test",
            source_partition_id="test_sources",
            view_id="test_ood",
        ),
        examples=tuple(
            ExampleIdentity(
                example_id=f"{source_id}:view:test_ood",
                source_id=source_id,
                view_id="test_ood",
            )
            for source_id in source_ids
        ),
        authorization_id="authorization:test",
        run_id="run:test",
        candidate_id="candidate:test",
        scientific_config_digest="sha256:test-config",
        checkpoint_id="checkpoint:test",
        epoch=0,
        method_id="erm",
        seed=301,
    )
    assert first.open_final_table(view).source_ids == source_ids
    with pytest.raises(FeatureCacheValidationError, match="feature cache manifest"):
        second.open_final_table(view)


def test_feature_cache_fails_helpfully_for_missing_or_mismatched_files(
    tmp_path: Path,
) -> None:
    with pytest.raises(FeatureCacheValidationError, match="manifest is missing"):
        load_cmnist_feature_cache(tmp_path / "absent")

    construction_object, pairs_object = _construction_and_pairs()
    root = tmp_path / "cache"
    prepare_cmnist_feature_cache(
        construction_object,
        pairs_object,
        DeterministicFakeEncoder(),
        root,
        normalization="none",
    )
    with pytest.raises(FeatureCacheValidationError, match="source manifest"):
        load_cmnist_feature_cache(
            root,
            expected_source_manifest_digest="sha256:not-the-source",
        )
    (root / "train_e01" / "features.npy").unlink()
    with pytest.raises(FeatureCacheValidationError, match="file is missing"):
        load_cmnist_feature_cache(root)


def test_official_clip_adapter_requires_explicit_download_permission(
    tmp_path: Path,
) -> None:
    encoder = OfficialOpenAiClipEncoder(
        weights_root=tmp_path / "weights",
        allow_download=False,
        batch_size=1,
    )
    with pytest.raises(FileNotFoundError, match="--allow-download"):
        encoder.encode(torch.zeros((1, 3, 4, 4), dtype=torch.float32))


def test_cuda_preflight_fails_before_writing_when_cuda_is_unavailable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    weights_root = tmp_path / "weights"
    encoder = OfficialOpenAiClipEncoder(
        weights_root=weights_root,
        allow_download=False,
        device="cuda",
        batch_size=2,
    )

    with pytest.raises(RuntimeError, match="CUDA is unavailable"):
        encoder.preflight()
    assert not weights_root.exists()


def test_cuda_device_index_out_of_range_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)
    weights_root = tmp_path / "weights"
    encoder = OfficialOpenAiClipEncoder(
        weights_root=weights_root,
        allow_download=False,
        device="cuda:2",
        batch_size=2,
    )

    with pytest.raises(RuntimeError, match="only 2 CUDA device"):
        encoder.preflight()
    assert not weights_root.exists()


def test_cuda_device_string_parsing() -> None:
    assert feature_module.cuda_device_index("cuda") == 0
    assert feature_module.cuda_device_index("cuda:1") == 1
    with pytest.raises(ValueError, match="cuda:N"):
        feature_module.cuda_device_index("cuda:x")


def test_feature_runtime_rejects_cross_backend_details() -> None:
    with pytest.raises(ValidationError, match="CPU feature runtime contains CUDA"):
        FeatureExtractionRuntime(
            requested_device="cpu",
            resolved_device="cpu",
            computation_dtype="torch.float32",
            deterministic_algorithms=True,
            tf32_enabled=False,
            mixed_precision=False,
            batch_size=2,
            torch_version="2.test",
            cuda_runtime_version="12.8",
            device_name="unexpected-gpu",
            compute_capability=(8, 6),
        )


def test_official_clip_cpu_adapter_batches_and_records_runtime(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    weights_root = tmp_path / "weights"
    weights_root.mkdir()
    (weights_root / "ViT-B-32.pt").write_bytes(b"pinned-fixture")
    batch_shapes: list[tuple[int, ...]] = []
    load_devices: list[str] = []

    class FakeModel:
        def eval(self) -> object:
            return self

        def float(self) -> object:
            return self

        def encode_image(self, images: torch.Tensor) -> torch.Tensor:
            assert images.device.type == "cpu"
            assert images.dtype == torch.float32
            batch_shapes.append(tuple(int(value) for value in images.shape))
            return torch.ones((len(images), 512), dtype=torch.float32)

    class FakeClip:
        def load(
            self, name: str, device: str, jit: bool, download_root: str
        ) -> tuple[FakeModel, object]:
            assert name == "ViT-B/32"
            assert jit is False
            assert download_root == str(weights_root)
            load_devices.append(device)

            def preprocess(_image: Image.Image) -> torch.Tensor:
                return torch.zeros((3, 4, 4), dtype=torch.float32)

            return FakeModel(), preprocess

    real_import = feature_module.importlib.import_module

    def fake_import(name: str) -> object:
        return FakeClip() if name == "clip" else real_import(name)

    def fake_hash(_path: Path) -> str:
        return f"sha256:{feature_module.OPENAI_CLIP_WEIGHTS_SHA256}"

    monkeypatch.setattr(
        feature_module.importlib,
        "import_module",
        fake_import,
    )
    monkeypatch.setattr(feature_module, "_file_sha256", fake_hash)
    encoder = OfficialOpenAiClipEncoder(
        weights_root=weights_root,
        allow_download=False,
        device="cpu",
        batch_size=2,
    )
    images = tuple(Image.new("RGB", (4, 4)) for _ in range(3))

    features = encoder.encode_pil(images)

    assert load_devices == ["cpu"]
    assert batch_shapes == [(2, 3, 4, 4), (1, 3, 4, 4)]
    assert features.shape == (3, 512)
    assert features.device.type == "cpu"
    runtime = encoder.extraction_runtime
    assert runtime.requested_device == "cpu"
    assert runtime.batch_size == 2
    assert runtime.cuda_runtime_version is None


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_official_clip_cuda_adapter_uses_cuda_and_returns_cpu_features(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    weights_root = tmp_path / "weights"
    weights_root.mkdir()
    (weights_root / "ViT-B-32.pt").write_bytes(b"pinned-fixture")

    class FakeCudaModel:
        def eval(self) -> object:
            return self

        def float(self) -> object:
            return self

        def encode_image(self, images: torch.Tensor) -> torch.Tensor:
            assert images.device.type == "cuda"
            assert images.dtype == torch.float32
            return torch.ones(
                (len(images), 512), dtype=torch.float32, device=images.device
            )

    class FakeCudaClip:
        def load(
            self, name: str, device: str, jit: bool, download_root: str
        ) -> tuple[FakeCudaModel, object]:
            assert name == "ViT-B/32"
            assert device.startswith("cuda:")
            assert jit is False
            assert download_root == str(weights_root)

            def preprocess(_image: Image.Image) -> torch.Tensor:
                return torch.zeros((3, 4, 4), dtype=torch.float32)

            return FakeCudaModel(), preprocess

    real_import = feature_module.importlib.import_module

    def fake_import(name: str) -> object:
        return FakeCudaClip() if name == "clip" else real_import(name)

    def fake_hash(_path: Path) -> str:
        return f"sha256:{feature_module.OPENAI_CLIP_WEIGHTS_SHA256}"

    monkeypatch.setattr(
        feature_module.importlib,
        "import_module",
        fake_import,
    )
    monkeypatch.setattr(feature_module, "_file_sha256", fake_hash)
    encoder = OfficialOpenAiClipEncoder(
        weights_root=weights_root,
        allow_download=False,
        device="cuda",
        batch_size=2,
    )

    features = encoder.encode_pil((Image.new("RGB", (4, 4)),) * 2)

    assert features.device.type == "cpu"
    runtime = encoder.extraction_runtime
    assert runtime.requested_device == "cuda"
    assert runtime.resolved_device.startswith("cuda:")
    assert runtime.cuda_runtime_version is not None
    assert runtime.compute_capability is not None
    assert runtime.tf32_enabled is False
    assert runtime.mixed_precision is False


def test_pair_free_cache_loading_needs_no_pair_feature_files(tmp_path: Path) -> None:
    construction, pairs = _construction_and_pairs()
    root = tmp_path / "features"
    manifest = prepare_cmnist_feature_cache(
        construction,
        pairs,
        DeterministicFakeEncoder(),
        root,
        normalization="none",
    )
    for table in manifest.tables:
        if table.role == "pair_projection":
            for file in table.files:
                (root / file.relative_path).unlink()
    for loader in (load_cmnist_feature_cache, load_cmnist_tuning_feature_cache):
        cache = loader(root, include_pairs=False)
        assert sum(table.features.shape[0] for table in cache.training_tables()) == 40
        assert all(table.features.shape[0] == 10 for table in cache.validation_tables())
        with pytest.raises(ValueError, match="without pair access"):
            cache.pair_tables()
        with pytest.raises(FeatureCacheValidationError):
            loader(root)
