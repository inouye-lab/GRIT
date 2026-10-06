"""Frozen CLIP features for every UrbanCars image and partner, test sealed.

Each JPEG is encoded once with the pinned unnormalized OpenAI CLIP ViT-B/32 (the
shared encoder the CMNIST path uses). Before encoding, the canvas check proves that
the full 256 x 256 canvas reaches the model: CLIP resizes the shorter side to 224
and centre-crops 224, which removes nothing from a square image, and a framed probe
image must keep its frame after preprocessing. `load_urbancars_tuning_tables` never
opens the test table; only `load_urbancars_test_table` does.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any, Literal, Protocol, TypeAlias, cast

import numpy as np
import torch
from numpy.typing import NDArray
from PIL import Image

from grit.data.urbancars import CANVAS, UrbanCarsDataset, file_sha256
from grit.features.cmnist import (
    FEATURE_DIMENSION,
    EncoderIdentity,
    FeatureExtractionRuntime,
)
from grit.schemas import canonical_digest_value

CACHE_SCHEMA = "grit.urbancars-features/v1"
Normalization: TypeAlias = Literal["none", "l2"]
TableRole: TypeAlias = Literal[
    "training", "validation", "final_test", "pair_projection"
]


class _TensorToNumpy(Protocol):
    def __call__(self, *, force: bool = False) -> NDArray[np.generic]: ...


class PilEncoder(Protocol):
    @property
    def identity(self) -> EncoderIdentity: ...

    @property
    def extraction_runtime(self) -> FeatureExtractionRuntime: ...

    def encode_pil(self, images: tuple[Image.Image, ...]) -> torch.Tensor: ...


# (table, role, image predicate) in encoding order.
TABLES: tuple[tuple[str, TableRole, Callable[[dict[str, Any]], bool]], ...] = (
    ("train", "training", lambda r: r["role"] == "standard" and r["split"] == "train"),
    (
        "validation",
        "validation",
        lambda r: r["role"] == "standard" and r["split"] == "val",
    ),
    ("test", "final_test", lambda r: r["role"] == "standard" and r["split"] == "test"),
    ("partner_b", "pair_projection", lambda r: r["role"] == "partner_b"),
    ("partner_o", "pair_projection", lambda r: r["role"] == "partner_o"),
)
CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)
_CHUNK = 1024


def _unnormalize(inputs: torch.Tensor) -> torch.Tensor:
    mean = torch.tensor(CLIP_MEAN).view(1, 3, 1, 1)
    std = torch.tensor(CLIP_STD).view(1, 3, 1, 1)
    return (inputs * std + mean).clamp(0, 1)


def canvas_check(
    preprocess: Callable[[tuple[Image.Image, ...]], torch.Tensor],
    samples: Sequence[Image.Image],
    output: Path,
) -> dict[str, Any]:
    """Assert that no centre crop removes canvas; save what the model sees.

    A grey 256 x 256 probe with an 8-pixel red frame must still be red at the edge
    midpoints of the model input. The contact sheet shows the first samples as the
    model receives them (unnormalized for display).
    """

    for image in samples:
        if image.size != (CANVAS, CANVAS):
            raise AssertionError(f"image is {image.size}, not {CANVAS} x {CANVAS}")
    probe = np.full((CANVAS, CANVAS, 3), 128, dtype=np.uint8)
    for frame in (probe[:8], probe[-8:], probe[:, :8], probe[:, -8:]):
        frame[...] = (255, 0, 0)
    seen = _unnormalize(preprocess((Image.fromarray(probe),)))[0]
    size = int(seen.shape[-1])
    middle = size // 2
    edges = {
        "top": seen[:, 0, middle],
        "bottom": seen[:, -1, middle],
        "left": seen[:, middle, 0],
        "right": seen[:, middle, -1],
    }
    red = {k: bool(v[0] > 0.8 and v[1] < 0.2 and v[2] < 0.2) for k, v in edges.items()}
    if not all(red.values()):
        raise AssertionError(f"CLIP preprocessing crops the canvas: {red}")
    inputs = _unnormalize(preprocess(tuple(samples)))
    tiles = [
        Image.fromarray(
            cast(
                np.ndarray[Any, np.dtype[np.uint8]],
                (t.permute(1, 2, 0).numpy() * 255).round().astype(np.uint8),  # pyright: ignore[reportUnknownMemberType]
            )
        )
        for t in inputs
    ]
    sheet = Image.new("RGB", (size * len(tiles), size), "white")
    for i, tile in enumerate(tiles):
        sheet.paste(tile, (i * size, 0))
    output.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output)
    return {"model_input_size": size, "frame_kept": red, "sheet": output.name}


def write_feature_table(
    root: Path,
    name: str,
    role: TableRole,
    source_ids: tuple[str, ...],
    targets: tuple[int, ...],
    features: torch.Tensor,
    normalization: Normalization,
) -> dict[str, object]:
    if tuple(features.shape) != (len(source_ids), FEATURE_DIMENSION):
        raise ValueError(f"{name}: features do not align with source IDs")
    if not bool(torch.isfinite(features).all()):
        raise ValueError(f"{name}: features must be finite")
    if normalization == "l2":
        features = features / features.square().sum(dim=1, keepdim=True).sqrt()
    (root / name).mkdir(parents=True, exist_ok=True)
    files: dict[str, str] = {}
    arrays: tuple[tuple[str, NDArray[np.generic]], ...] = (
        ("features", cast(_TensorToNumpy, features.contiguous().numpy)()),
        ("targets", np.asarray(targets, dtype=np.int64)),
    )
    for field, array in arrays:
        path = root / name / f"{field}.npy"
        np.save(path, array, allow_pickle=False)
        files[field] = file_sha256(path)
    return {
        "name": name,
        "role": role,
        "rows": len(source_ids),
        "source_ids": list(source_ids),
        "source_ids_digest": canonical_digest_value(list(source_ids)),
        "files": files,
    }


def prepare_urbancars_feature_cache(
    dataset: UrbanCarsDataset,
    encoder: PilEncoder,
    preprocess: Callable[[tuple[Image.Image, ...]], torch.Tensor],
    output_dir: Path,
) -> dict[str, Any]:
    """Encode every table and write `manifest.json` last."""

    if (output_dir / "manifest.json").exists():
        raise FileExistsError(f"UrbanCars feature cache already exists: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    train = dataset.images(TABLES[0][2])
    check = canvas_check(
        preprocess,
        [Image.open(dataset.root / r["path"]).convert("RGB") for r in train[:8]],
        output_dir / "clip-inputs.png",
    )
    tables: list[dict[str, object]] = []
    for name, role, predicate in TABLES:
        records = dataset.images(predicate)
        for record in records:
            if file_sha256(dataset.root / record["path"]) != record["sha256"]:
                raise ValueError(f"image changed since the build: {record['path']}")
        chunks = [records[i : i + _CHUNK] for i in range(0, len(records), _CHUNK)]
        features = torch.cat(
            [
                encoder.encode_pil(
                    tuple(
                        Image.open(dataset.root / r["path"]).convert("RGB")
                        for r in chunk
                    )
                )
                .detach()
                .cpu()
                .to(torch.float32)
                for chunk in chunks
            ]
        )
        tables.append(
            write_feature_table(
                output_dir,
                name,
                role,
                tuple(r["image_id"] for r in records),
                tuple(int(r["label"]) for r in records),
                features,
                "none",
            )
        )
        print(f"encoded {name}: {len(records)} images", flush=True)
    manifest: dict[str, Any] = {
        "schema_version": CACHE_SCHEMA,
        "dataset_manifest_digest": dataset.digest,
        "encoder": encoder.identity.model_dump(mode="json"),
        "extraction_runtime": encoder.extraction_runtime.model_dump(mode="json"),
        "normalization": "none",
        "feature_dimension": FEATURE_DIMENSION,
        "canvas_check": check,
        "tables": tables,
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


def _manifest(root: Path, dataset: UrbanCarsDataset) -> dict[str, Any]:
    manifest = cast(dict[str, Any], json.loads((root / "manifest.json").read_text()))
    if manifest.get("schema_version") != CACHE_SCHEMA:
        raise ValueError(f"not an UrbanCars feature cache: {root}")
    if manifest["dataset_manifest_digest"] != dataset.digest:
        raise ValueError("feature cache was built from another UrbanCars dataset")
    return manifest


def _load(root: Path, item: dict[str, Any]) -> tuple[tuple[str, ...], torch.Tensor]:
    path = root / item["name"] / "features.npy"
    if not path.is_file() or file_sha256(path) != item["files"]["features"]:
        raise ValueError(f"UrbanCars cache file is missing or damaged: {path}")
    features = torch.tensor(np.load(path, allow_pickle=False), dtype=torch.float32)
    if tuple(features.shape) != (int(item["rows"]), FEATURE_DIMENSION):
        raise ValueError(
            f"UrbanCars cache table {item['name']} has inconsistent shapes"
        )
    return tuple(item["source_ids"]), features


def load_urbancars_tuning_tables(
    root: Path, dataset: UrbanCarsDataset
) -> dict[str, tuple[tuple[str, ...], torch.Tensor]]:
    """Training, validation and partner features by table; test is not opened."""

    return {
        item["name"]: _load(root, item)
        for item in _manifest(root, dataset)["tables"]
        if item["role"] != "final_test"
    }


def load_urbancars_test_table(
    root: Path, dataset: UrbanCarsDataset
) -> tuple[tuple[str, ...], torch.Tensor]:
    """The sealed test features, for evaluation after winners are frozen."""

    (item,) = (
        t for t in _manifest(root, dataset)["tables"] if t["role"] == "final_test"
    )
    return _load(root, item)
