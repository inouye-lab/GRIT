"""Explicitly non-reportable Waterbirds-like assets for offline smoke execution."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from PIL import Image

from grit.data.waterbirds import (
    FixtureWaterbirdsProfile,
    ParsedWaterbirdsAssets,
    WaterbirdsGroupCounts,
    load_waterbirds_assets,
)


@dataclass(frozen=True, slots=True)
class WaterbirdsSmokeAssets:
    assets: ParsedWaterbirdsAssets
    profile: FixtureWaterbirdsProfile


def make_waterbirds_smoke_assets(
    root: Path,
    *,
    construction_seed: int = 17,
) -> WaterbirdsSmokeAssets:
    """Create a 10/4/4 fixture that cannot satisfy the production profile."""

    released_root = root / "waterbird_fixture"
    cub_root = root / "CUB_200_2011"
    masks_root = root / "segmentations"
    places_root = root / "places365"
    released_root.mkdir(parents=True)
    (cub_root / "images").mkdir(parents=True)
    masks_root.mkdir(parents=True)
    identities = (
        *((index, 0, 0, 0) for index in range(1, 5)),
        *((index, 0, 1, 0) for index in range(5, 7)),
        (7, 1, 0, 0),
        *((index, 1, 1, 0) for index in range(8, 11)),
        (11, 0, 0, 1),
        (12, 0, 1, 1),
        (13, 1, 0, 1),
        (14, 1, 1, 1),
        (15, 0, 0, 2),
        (16, 0, 1, 2),
        (17, 1, 0, 2),
        (18, 1, 1, 2),
    )
    image_lines: list[str] = []
    label_lines: list[str] = []
    box_lines: list[str] = []
    metadata_rows: list[dict[str, str]] = []
    for image_id, label, background, split in identities:
        species = "001.Landbird" if label == 0 else "002.Waterbird"
        relative_path = f"{species}/image_{image_id:04d}.jpg"
        image_lines.append(f"{image_id} {relative_path}")
        label_lines.append(f"{image_id} {label + 1}")
        box_lines.append(f"{image_id} 1 1 4 3")
        pixels = np.zeros((6, 8, 3), dtype=np.uint8)
        pixels[:, :, :] = (
            (image_id * 17) % 251,
            (image_id * 29) % 251,
            (image_id * 43) % 251,
        )
        pixels[1:5, 2:6, :] = (190 + 20 * label, 40, 60)
        mask = np.zeros((6, 8, 3), dtype=np.uint8)
        mask[1:5, 2:6, :] = 255
        _save_rgb(cub_root / "images" / relative_path, pixels)
        _save_rgb(released_root / relative_path, pixels)
        _save_rgb(masks_root / Path(relative_path).with_suffix(".png"), mask)
        metadata_rows.append(
            {
                "img_id": str(image_id),
                "img_filename": relative_path,
                "y": str(label),
                "place": str(background),
                "split": str(split),
                "place_filename": f"released/place_{image_id:04d}.jpg",
            }
        )
    (cub_root / "images.txt").write_text(
        "\n".join(image_lines) + "\n", encoding="utf-8"
    )
    (cub_root / "image_class_labels.txt").write_text(
        "\n".join(label_lines) + "\n", encoding="utf-8"
    )
    (cub_root / "classes.txt").write_text(
        "1 001.Landbird\n2 002.Waterbird\n", encoding="utf-8"
    )
    (cub_root / "bounding_boxes.txt").write_text(
        "\n".join(box_lines) + "\n", encoding="utf-8"
    )
    with (released_root / "metadata.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        fieldnames: list[str] = [
            "img_id",
            "img_filename",
            "y",
            "place",
            "split",
            "place_filename",
        ]
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(metadata_rows)
    categories = (
        "bamboo_forest",
        "forest/broadleaf",
        "lake/natural",
        "ocean",
    )
    for category_index, category in enumerate(categories):
        directory = places_root / "data_large" / category[0] / category
        for image_index in range(3):
            pixels = np.full(
                (9 + image_index, 12 + category_index, 3),
                fill_value=30 + 40 * category_index + image_index,
                dtype=np.uint8,
            )
            _save_rgb(directory / f"background_{image_index}.jpg", pixels)
    profile = FixtureWaterbirdsProfile(
        kind="fixture",
        non_reportable=True,
        base_artifact_name="waterbird_fixture",
        construction_seed=construction_seed,
        train_group_counts=WaterbirdsGroupCounts(
            landbird_land=4,
            landbird_water=2,
            waterbird_land=1,
            waterbird_water=3,
        ),
        landbird_pair_count=2,
        waterbird_pair_count=1,
        validation_count=4,
        test_count=4,
    )
    assets = load_waterbirds_assets(
        released_root=released_root,
        cub_root=cub_root,
        masks_root=masks_root,
        places_root=places_root,
        artifact_name=profile.base_artifact_name,
    )
    return WaterbirdsSmokeAssets(assets=assets, profile=profile)


def _save_rgb(path: Path, pixels: NDArray[np.uint8]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(pixels.astype(np.uint8)).save(path)
