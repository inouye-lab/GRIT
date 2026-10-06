"""UrbanCars reconstruction with recorded components (App. G.1, G.5).

UrbanCars (Li et al., "A Whac-A-Mole Dilemma", CVPR 2023) composites a Stanford Cars
car (label: urban or country body type) and an LVIS/COCO co-occurring object onto a
Places365 background. Upstream does not record which components make each image, so
this module replays the upstream generator with a registry of stable component IDs:

- **Pools.** Per kind (car, background, co-object) and category, the upstream pool
  order (concatenated source lists), shuffled with ``random.Random(0)``, split by
  index: train [0, 4000), val [4000, 4500), test [4500, 5000).
- **Standard splits.** The upstream composition loop, consuming each pool in order.
- **Renderer.** The upstream crop/rescale helpers and placement constants, vendored
  from ``facebookresearch/Whac-A-Mole`` (``data_utils/object_scale.py`` and
  ``create_datasets/urbancars/gen_urbancars.py``, CC BY-NC 4.0, Meta Platforms) with
  explicit clipping added; the standard placement never clips.
- **Partners.** Reserved training anchors each get a B-partner (background from the
  opposite category's train pool) and an O-partner (co-object likewise).

Car masks (MaskFormer) and the LVIS co-object pools (``lvis`` API masks and filters)
come from the isolated generation environment (``scripts/urbancars_gen/``), which
writes ``export/components.json``.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
from collections.abc import Callable, Iterable, Iterator, Sequence
from concurrent.futures import Executor
from dataclasses import asdict, dataclass
from itertools import product
from pathlib import Path
from typing import Any, Literal, Protocol, TypeAlias, cast

import numpy as np
import torch
import torch.nn.functional as F
from numpy.typing import NDArray
from PIL import Image
from pydantic import Field

from grit.geometry import group_index
from grit.schemas import StrictBoundaryModel, canonical_digest_value

UPSTREAM_REPOSITORY = "https://github.com/facebookresearch/Whac-A-Mole.git"
UPSTREAM_COMMIT = "c91b097043246bc726fccd70f9860873e28d6122"
MASKFORMER_REPOSITORY = "https://github.com/facebookresearch/MaskFormer.git"
EXPORT_SCHEMA = "grit.urbancars-components/v1"
MANIFEST_SCHEMA = "grit.urbancars-dataset/v1"
CATEGORIES: tuple[str, str] = ("urban", "country")
# Upstream meta_data.json, in its order (the concatenation order of each pool).
PLACES_CLASSES: dict[str, tuple[str, ...]] = {
    "urban": (
        "alley",
        "street",
        "downtown",
        "crosswalk",
        "gas_station",
        "garage-outdoor",
        "driveway",
    ),
    "country": ("desert_road", "field_road", "forest_road"),
}
LVIS_CLASSES: dict[str, tuple[tuple[int, str], ...]] = {
    "urban": (
        (445, "fireplug"),
        (1019, "stop_sign"),
        (1026, "street_sign"),
        (766, "parking_meter"),
        (1112, "traffic_light"),
    ),
    "country": ((80, "cow"), (569, "horse"), (943, "sheep")),
}
POOL_SPLITS: dict[str, tuple[int, int]] = {
    "train": (0, 4000),
    "val": (4000, 4500),
    "test": (4500, 5000),
}
MINIMUM_POOL = 5000
CANVAS = 256
STANFORD_COUNTS = {"train": 8144, "test": 8041}
# Entries of metadata/stanford_cars/{train,test}.txt (the files have no final newline,
# so `wc -l` reports one fewer).
FILTERED_COUNTS = {"train": 7488, "test": 7426}

Kind: TypeAlias = Literal["car", "background", "object"]
PoolSplit: TypeAlias = Literal["train", "val", "test"]


class Construction(StrictBoundaryModel):
    """Placement and correlations. ``standard`` is upstream (the paper's construction);
    the two others are nested fallbacks (the second keeps the first's co-object and
    raises its correlation)."""

    name: Literal["standard", "f1", "f2"]
    car_scale: float = 0.5
    object_scale: float
    # (row, column) centre on the 256 x 256 canvas, as upstream int(fraction * 256).
    object_center: tuple[int, int]
    b_ratio: float = Field(default=0.95, gt=0.5, lt=1.0)
    o_ratio: float = Field(gt=0.5, lt=1.0)

    def upstream_folder(self, split: str) -> str | None:
        """Upstream's output folder for a split, when upstream can generate it.

        Upstream names the folder by the split's ratios, and val and test are
        balanced (0.5), so they sit under ``bg-0.5_co_occur_obj-0.5``.
        """

        if self.object_scale != 0.25 or self.object_center != (128, 224):
            return None
        b, o = (self.b_ratio, self.o_ratio) if split == "train" else (0.5, 0.5)
        return f"bg-{b}_co_occur_obj-{o}"


CONSTRUCTIONS: dict[str, Construction] = {
    "standard": Construction(
        name="standard", object_scale=0.25, object_center=(128, 224), o_ratio=0.95
    ),
    "f1": Construction(
        name="f1", object_scale=0.35, object_center=(128, 210), o_ratio=0.95
    ),
    "f2": Construction(
        name="f2", object_scale=0.35, object_center=(128, 210), o_ratio=0.99
    ),
}
CAR_CENTER = (CANVAS // 2, CANVAS // 2)
# Anchor reservation, in order; the unused remainder follows.
ANCHOR_POOLS: tuple[tuple[str, int], ...] = (
    ("calibration", 1024),
    ("reference", 2048),
    ("bank0", 1024),
    ("bank1", 1024),
    ("bank2", 1024),
)


# Source layout: the upstream ``data/`` tree, so upstream code runs against it too.


@dataclass(frozen=True)
class SourceLayout:
    """``root`` is upstream's ``data/`` tree; the pinned clone sits beside it (never
    inside, since the clone links ``data`` back to ``root``)."""

    root: Path

    @property
    def stanford_cars(self) -> Path:
        return self.root / "stanford_cars"

    @property
    def coco(self) -> Path:
        return self.root / "coco"

    @property
    def lvis(self) -> Path:
        return self.root / "lvis"

    @property
    def places(self) -> Path:
        return self.root / "places"

    @property
    def upstream(self) -> Path:
        """The pinned Whac-A-Mole clone (metadata and generation code)."""
        return self.root.parent / "Whac-A-Mole"

    @property
    def metadata(self) -> Path:
        return self.upstream / "create_datasets" / "urbancars" / "metadata"

    @property
    def export(self) -> Path:
        return self.root / "export"

    @property
    def upstream_output(self) -> Path:
        """Where an unmodified upstream run writes (``data/urbancars``)."""
        return self.root / "urbancars"

    def car_image(self, split: str, fname: str) -> Path:
        return self.stanford_cars / f"cars_{split}" / fname

    def car_mask(self, split: str, fname: str) -> Path:
        stem = Path(fname).stem
        return self.stanford_cars / "mask" / f"cars_{split}" / f"{stem}_mask.png"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return f"sha256:{digest.hexdigest()}"


def normalize_stanford_cars(mirror: Path, target: Path) -> dict[str, str]:
    """Link a Stanford Cars mirror into the original layout, wherever it nests files.

    Needed: ``cars_train/`` and ``cars_test/`` (JPEGs), ``devkit/cars_train_annos.mat``,
    ``devkit/cars_meta.mat`` and ``cars_test_annos_withlabels.mat``. Filenames are
    never renamed; a missing or ambiguous file stops with its name.
    """

    def image_dir(name: str) -> Path:
        found = [
            p
            for p in mirror.rglob(name)
            if p.is_dir() and any(p.glob("0*.jpg")) and "mask" not in p.parts
        ]
        if len(found) != 1:
            raise FileNotFoundError(f"{name}/ with JPEGs: found {found} in {mirror}")
        return found[0]

    def single(pattern: str) -> Path:
        found = sorted(p for p in mirror.rglob(pattern) if p.is_file())
        if not found:
            raise FileNotFoundError(f"{pattern} not found in {mirror}")
        digests = {file_sha256(p) for p in found}
        if len(digests) != 1:
            raise ValueError(f"ambiguous {pattern}: {found}")
        return found[0]

    links = {
        "cars_train": image_dir("cars_train"),
        "cars_test": image_dir("cars_test"),
        "devkit/cars_train_annos.mat": single("cars_train_annos.mat"),
        "devkit/cars_meta.mat": single("cars_meta.mat"),
        "cars_test_annos_withlabels.mat": single("cars_test_annos_withlabels*.mat"),
    }
    for name, source in links.items():
        path = target / name
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.is_symlink() or path.exists():
            if path.resolve() != source.resolve():
                raise FileExistsError(f"{path} already points elsewhere")
            continue
        path.symlink_to(source.resolve(), target_is_directory=source.is_dir())
    return {name: str(source) for name, source in links.items()}


# Components and pools.


@dataclass(frozen=True)
class Component:
    component_id: str
    kind: Kind
    category: int
    # Kind-specific source identity: Stanford split/fname/class; Places rel path;
    # LVIS annotation, COCO image and file, mask file.
    source: dict[str, Any]


@dataclass(frozen=True)
class Registry:
    """Shuffled pools in upstream order, keyed by (kind, category)."""

    pools: dict[tuple[str, int], tuple[Component, ...]]
    export_digest: str
    metadata_digest: str

    def split(
        self, kind: Kind, category: int, split: PoolSplit
    ) -> tuple[Component, ...]:
        start, stop = POOL_SPLITS[split]
        return self.pools[(kind, category)][start:stop]

    def pool_split(self) -> dict[str, PoolSplit]:
        result: dict[str, PoolSplit] = {}
        for pool in self.pools.values():
            for split, (start, stop) in POOL_SPLITS.items():
                for component in pool[start:stop]:
                    result[component.component_id] = cast(PoolSplit, split)
        return result

    def by_id(self) -> dict[str, Component]:
        return {c.component_id: c for pool in self.pools.values() for c in pool}


def upstream_shuffle(items: Sequence[Component]) -> tuple[Component, ...]:
    """``random.Random(0).shuffle`` of the index list, as upstream subsamples."""

    indices = list(range(len(items)))
    random.Random(0).shuffle(indices)
    return tuple(items[i] for i in indices)


def places_pool(metadata: Path, category: str) -> list[Component]:
    """Upstream ``Places``: per class, train then val, each filter list sorted."""

    result: list[Component] = []
    for places_class in PLACES_CLASSES[category]:
        for split in ("train", "val"):
            listing = (
                metadata / "places" / "filter_objs" / split / f"{places_class}.txt"
            )
            names = sorted(line.strip() for line in listing.read_text().splitlines())
            for name in names:
                rel = f"{split}/{places_class}/{name}"
                result.append(
                    Component(
                        f"bg:{rel}",
                        "background",
                        CATEGORIES.index(category),
                        {"places_path": rel, "places_class": places_class},
                    )
                )
    return result


def lvis_component_id(lvis_split: str, annotation_id: int) -> str:
    """LVIS train and val reuse annotation numbers, so identity is (split, number)."""

    return f"obj:lvis-{lvis_split}-{annotation_id}"


def load_export(layout: SourceLayout) -> dict[str, Any]:
    path = layout.export / "components.json"
    export = cast(dict[str, Any], json.loads(path.read_text()))
    if export.get("schema") != EXPORT_SCHEMA:
        raise ValueError(f"not an UrbanCars component export: {path}")
    if export.get("upstream_commit") != UPSTREAM_COMMIT:
        raise ValueError("component export comes from another upstream commit")
    return export


def build_registry(layout: SourceLayout) -> Registry:
    export = load_export(layout)
    pools: dict[tuple[str, int], tuple[Component, ...]] = {}
    for index, category in enumerate(CATEGORIES):
        cars = [
            Component(
                f"car:{car['split']}/{car['fname']}",
                "car",
                index,
                {"split": car["split"], "fname": car["fname"], "class": car["class"]},
            )
            for car in export["cars"][category]
        ]
        objects = [
            Component(
                lvis_component_id(item["lvis_split"], item["annotation_id"]),
                "object",
                index,
                {
                    key: item[key]
                    for key in (
                        "annotation_id",
                        "image_id",
                        "category_id",
                        "lvis_split",
                        "coco_file",
                        "mask",
                    )
                },
            )
            for item in export["objects"][category]
        ]
        for kind, items in (
            ("car", cars),
            ("background", places_pool(layout.metadata, category)),
            ("object", objects),
        ):
            if len(items) < MINIMUM_POOL:
                raise ValueError(
                    f"{kind} pool {category} has {len(items)} < {MINIMUM_POOL} items"
                )
            ids = [c.component_id for c in items]
            if len(set(ids)) != len(ids):
                raise ValueError(f"{kind} pool {category} repeats a component")
            pools[(kind, index)] = upstream_shuffle(items)
    metadata_files = sorted(p for p in layout.metadata.rglob("*") if p.is_file())
    return Registry(
        pools=pools,
        export_digest=file_sha256(layout.export / "components.json"),
        metadata_digest=canonical_digest_value(
            {
                str(p.relative_to(layout.metadata)): file_sha256(p)
                for p in metadata_files
            }
        ),
    )


def source_file(layout: SourceLayout, component: Component) -> Path:
    source = component.source
    if component.kind == "car":
        return layout.car_image(source["split"], source["fname"])
    if component.kind == "background":
        return layout.places / source["places_path"]
    return layout.coco / source["coco_file"]


def source_id_log(
    layout: SourceLayout, registry: Registry, *, hash_files: bool = True
) -> dict[str, Any]:
    """Every component's source IDs and pool split, with cross-split findings.

    No split is ever changed here. The same annotation or file in two splits is a
    real duplicate to investigate; byte-identical files across splits likewise. A
    COCO photo shared by objects in different splits is only logged.
    """

    splits = registry.pool_split()
    by_source: dict[str, set[str]] = {}
    by_photo: dict[int, set[str]] = {}
    by_bytes: dict[str, set[str]] = {}
    entries: list[dict[str, Any]] = []
    for (kind, category), pool in sorted(registry.pools.items()):
        for component in pool:
            split = splits.get(component.component_id, "unused")
            path = source_file(layout, component)
            key = (
                lvis_component_id(
                    component.source["lvis_split"], component.source["annotation_id"]
                )
                if kind == "object"
                else str(path.relative_to(layout.root))
            )
            digest = file_sha256(path) if hash_files and split != "unused" else None
            entries.append(
                {
                    "component_id": component.component_id,
                    "kind": kind,
                    "category": CATEGORIES[category],
                    "split": split,
                    "source": component.source,
                    "sha256": digest,
                }
            )
            if split == "unused":
                continue
            by_source.setdefault(key, set()).add(split)
            if kind == "object":
                by_photo.setdefault(int(component.source["image_id"]), set()).add(split)
            if digest is not None:
                by_bytes.setdefault(f"{kind}:{digest}", set()).add(split)
    return {
        "entries": entries,
        "duplicate_sources_across_splits": sorted(
            k for k, v in by_source.items() if len(v) > 1
        ),
        "byte_identical_across_splits": sorted(
            k for k, v in by_bytes.items() if len(v) > 1
        ),
        "coco_photos_shared_across_splits": sorted(
            k for k, v in by_photo.items() if len(v) > 1
        ),
    }


def fallback_masks(layout: SourceLayout) -> list[dict[str, Any]]:
    """Listed cars whose mask came from the relaxed fallback (a recorded deviation)."""

    path = layout.export / "fallback_masks.json"
    if not path.is_file():
        return []
    return cast(list[dict[str, Any]], json.loads(path.read_text()))


# Acceptance check G0a (sources).


def check_sources(layout: SourceLayout, export: dict[str, Any]) -> dict[str, Any]:
    """G0a: Stanford counts and filtered files, Places lists, LVIS pools, car masks."""

    result: dict[str, Any] = {}
    failures: list[str] = []
    for split, expected in STANFORD_COUNTS.items():
        count = len(list((layout.stanford_cars / f"cars_{split}").glob("*.jpg")))
        listed = (layout.metadata / "stanford_cars" / f"{split}.txt").read_text()
        filtered = [line for line in listed.splitlines() if line]
        missing = [f for f in filtered if not layout.car_image(split, f).is_file()]
        masks = [layout.car_mask(split, f) for f in filtered]
        no_mask = [str(m.name) for m in masks if not m.is_file()]
        empty = [
            str(m.name)
            for m in masks
            if m.is_file() and not np.asarray(Image.open(m)).any()
        ]
        result[f"stanford_{split}"] = {
            "images": count,
            "expected_images": expected,
            "filtered": len(filtered),
            "expected_filtered": FILTERED_COUNTS[split],
            "filtered_missing": missing[:20],
            "masks_missing": no_mask[:20],
            "masks_missing_count": len(no_mask),
            "masks_empty": empty[:20],
            "masks_empty_count": len(empty),
        }
        if count != expected or len(filtered) != FILTERED_COUNTS[split] or missing:
            failures.append(f"stanford_{split}")
        if no_mask or empty:
            failures.append(f"car_masks_{split}")
    places: dict[str, Any] = {}
    for category, classes in PLACES_CLASSES.items():
        for places_class in classes:
            for split in ("train", "val"):
                listing = (
                    layout.metadata
                    / "places"
                    / "filter_objs"
                    / split
                    / f"{places_class}.txt"
                )
                names = listing.read_text().split() if listing.is_file() else []
                absent = [
                    n
                    for n in names
                    if not (layout.places / split / places_class / n).is_file()
                ]
                places[f"{split}/{places_class}"] = {
                    "list_present": listing.is_file(),
                    "listed": len(names),
                    "missing": absent[:20],
                    "missing_count": len(absent),
                }
                if not listing.is_file() or absent:
                    failures.append(f"places_{split}_{places_class}")
        _ = category
    result["places"] = places
    result["lvis_pools"] = {
        category: len(export["objects"][category]) for category in CATEGORIES
    }
    for category, count in result["lvis_pools"].items():
        if count < MINIMUM_POOL:
            failures.append(f"lvis_{category}")
    result["segmenter"] = export.get("segmenter")
    result["fallback_masks"] = fallback_masks(layout)
    result["failures"] = failures
    result["passes"] = not failures
    return result


# Renderer: vendored upstream helpers (Whac-A-Mole, CC BY-NC 4.0) plus clipping.


class _ToTensor(Protocol):
    def __call__(self, pic: Image.Image) -> torch.Tensor: ...


def _to_tensor() -> _ToTensor:
    import torchvision.transforms.functional as functional  # pyright: ignore[reportMissingTypeStubs]

    return cast(_ToTensor, functional.to_tensor)  # pyright: ignore[reportUnknownMemberType]


def crop_mask_and_img(
    mask: torch.Tensor, image: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Upstream ``crop_mask_and_img``: binarize at 0.5 and crop to the mask box."""

    if mask.ndim != 3 or image.ndim != 3 or mask.shape[1:] != image.shape[1:]:
        raise ValueError("mask [1, H, W] and image [C, H, W] must align")
    height, width = mask.shape[1:]
    binary = (mask > 0.5).long()
    indices = binary[0].nonzero()
    if len(indices) == 0:
        raise ValueError("empty mask")
    top, bottom = int(indices[:, 0].min()), int(indices[:, 0].max())
    left, right = int(indices[:, 1].min()), int(indices[:, 1].max())
    rows = slice(top, min(bottom + 1, height))
    columns = slice(left, min(right + 1, width))
    return binary[:, rows, columns], image[:, rows, columns]


def rescale_cropped_mask_and_img(
    cropped_mask: torch.Tensor,
    cropped_img: torch.Tensor,
    scale: float,
    center: tuple[int, int],
    target_side_size: int = CANVAS,
) -> tuple[torch.Tensor, torch.Tensor, bool]:
    """Upstream ``rescale_cropped_mask_and_img`` with explicit clipping.

    Upstream places the rescaled patch at ``max(0, centre - size // 2)`` and fails
    with a shape error if it crosses the far edge; here the patch is clipped to the
    canvas instead and the returned flag says so.
    """

    if not 0 < scale <= 1:
        raise ValueError("scale must lie in (0, 1]")
    longer_side = max(cropped_mask.shape[1:])
    scale_factor = scale * target_side_size / longer_side
    rescaled_mask = F.interpolate(
        cropped_mask.unsqueeze(0).float(),
        scale_factor=scale_factor,
        mode="nearest-exact",
    ).squeeze(0)
    rescaled_img = F.interpolate(
        cropped_img.unsqueeze(0),
        scale_factor=scale_factor,
        mode="bilinear",
        align_corners=False,
        antialias=True,
    ).squeeze(0)
    height, width = rescaled_mask.shape[1:]
    top = max(0, center[0] - height // 2)
    left = max(0, center[1] - width // 2)
    rows = min(height, target_side_size - top)
    columns = min(width, target_side_size - left)
    clipped = rows < height or columns < width
    target_mask = torch.zeros((1, target_side_size, target_side_size), dtype=torch.long)
    target_mask[:, top : top + rows, left : left + columns] = rescaled_mask[
        :, :rows, :columns
    ].long()
    target_img = torch.zeros((3, target_side_size, target_side_size), dtype=torch.float)
    target_img[:, top : top + rows, left : left + columns] = rescaled_img[
        :, :rows, :columns
    ]
    return target_mask, target_img, clipped


def save_jpeg(image: torch.Tensor, path: Path) -> None:
    """torchvision ``save_image`` for one [3, H, W] image in [0, 1] (as upstream)."""

    array = cast(
        NDArray[np.uint8],
        image.mul(255)
        .add_(0.5)
        .clamp_(0, 255)
        .permute(1, 2, 0)
        .to("cpu", torch.uint8)
        .numpy(),  # pyright: ignore[reportUnknownMemberType]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(array).save(path)


@dataclass(frozen=True)
class RenderJob:
    """Absolute source paths for one image; picklable for process pools."""

    image_id: str
    output: str
    car_image: str
    car_mask: str
    background: str
    object_image: str
    object_mask: str
    car_scale: float
    object_scale: float
    object_center: tuple[int, int]


def render(job: RenderJob) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, bool]:
    """Composite as upstream: co-object over background, then the car over that."""

    to_tensor = _to_tensor()
    car_pil = Image.open(job.car_image).convert("RGB")
    car = to_tensor(car_pil)
    car_mask = to_tensor(Image.open(job.car_mask).resize(car_pil.size))
    background = to_tensor(Image.open(job.background).convert("RGB"))
    if tuple(background.shape) != (3, CANVAS, CANVAS):
        raise ValueError(f"background is not {CANVAS} x {CANVAS}: {job.background}")
    co_image = to_tensor(Image.open(job.object_image))
    co_mask = torch.from_numpy(  # pyright: ignore[reportUnknownMemberType]
        (np.asarray(Image.open(job.object_mask)) > 127).astype(np.int64)
    ).unsqueeze(0)
    car_crop_mask, car_crop = crop_mask_and_img(car_mask, car)
    car_mask_canvas, car_canvas, car_clipped = rescale_cropped_mask_and_img(
        car_crop_mask, car_crop, job.car_scale, CAR_CENTER
    )
    co_crop_mask, co_crop = crop_mask_and_img(co_mask, co_image)
    co_mask_canvas, co_canvas, co_clipped = rescale_cropped_mask_and_img(
        co_crop_mask, co_crop, job.object_scale, job.object_center
    )
    blend = co_mask_canvas * co_canvas + (1 - co_mask_canvas) * background
    output = car_mask_canvas * car_canvas + (1 - car_mask_canvas) * blend
    return output, car_mask_canvas, co_mask_canvas, car_clipped or co_clipped


def render_and_save(job: RenderJob) -> dict[str, Any]:
    output, car_mask, co_mask, clipped = render(job)
    path = Path(job.output)
    save_jpeg(output, path)
    co_pixels = int(co_mask.sum())
    visible = int((co_mask * (1 - car_mask)).sum())
    return {
        "image_id": job.image_id,
        "sha256": file_sha256(path),
        "object_pixels": co_pixels,
        "object_visible_fraction": visible / co_pixels if co_pixels else 0.0,
        "clipped": clipped,
    }


# Standard splits (upstream composition replay), anchors and partners.


@dataclass(frozen=True)
class ImageSpec:
    image_id: str
    role: Literal["standard", "partner_b", "partner_o"]
    split: PoolSplit
    label: int
    b_category: int
    o_category: int
    car: str
    background: str
    obj: str
    composition: str | None = None
    index: int | None = None
    anchor: str | None = None

    @property
    def group(self) -> int:
        return group_index(
            self.label, self.b_category == self.label, self.o_category == self.label
        )

    @property
    def path(self) -> str:
        return f"images/{self.image_id}.jpg"


def composition_name(label: int, b: int, o: int) -> str:
    return f"obj-{CATEGORIES[label]}_bg-{CATEGORIES[b]}_co_occur_obj-{CATEGORIES[o]}"


def composition_count(
    split: PoolSplit, label: int, b: int, o: int, construction: Construction
) -> int:
    """Upstream ``int(ratio * num_img_split)``, at least 1; val/test are balanced."""

    num_split = POOL_SPLITS[split][1] - POOL_SPLITS[split][0]
    b_ratio, o_ratio = (
        (construction.b_ratio, construction.o_ratio) if split == "train" else (0.5, 0.5)
    )
    ratio = (b_ratio if b == label else 1 - b_ratio) * (
        o_ratio if o == label else 1 - o_ratio
    )
    return max(1, int(ratio * num_split))


def standard_images(registry: Registry, construction: Construction) -> list[ImageSpec]:
    """Replay upstream's composition loop, consuming every pool in order."""

    images: list[ImageSpec] = []
    for split in cast(tuple[PoolSplit, ...], ("train", "val", "test")):
        iterators: dict[tuple[str, int], Iterator[Component]] = {
            (kind, category): iter(registry.split(kind, category, split))
            for kind in ("car", "background", "object")
            for category in range(2)
        }
        for label, b, o in product(range(2), repeat=3):
            name = composition_name(label, b, o)
            for i in range(composition_count(split, label, b, o, construction)):
                images.append(
                    ImageSpec(
                        image_id=f"{split}/{name}/{i:03d}",
                        role="standard",
                        split=split,
                        label=label,
                        b_category=b,
                        o_category=o,
                        car=next(iterators[("car", label)]).component_id,
                        background=next(iterators[("background", b)]).component_id,
                        obj=next(iterators[("object", o)]).component_id,
                        composition=name,
                        index=i,
                    )
                )
    return images


def _hash_int(*parts: object) -> int:
    payload = "\0".join(str(p) for p in parts).encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def _uniform(*parts: object) -> float:
    return _hash_int(*parts) / 2**64


def proportional_sequence(counts: Sequence[int]) -> list[int]:
    """Labels 0..len(counts)-1 spread evenly (largest deficit), exact counts."""

    total = sum(counts)
    assigned = [0] * len(counts)
    sequence: list[int] = []
    for t in range(total):
        deficits = [
            c * (t + 1) / total - a for c, a in zip(counts, assigned, strict=True)
        ]
        best = max(range(len(counts)), key=lambda k: (deficits[k], -k))
        sequence.append(best)
        assigned[best] += 1
    return sequence


def _stratified_order(members: Sequence[ImageSpec], seed: int) -> list[ImageSpec]:
    """Order so that every prefix is close to proportional by group."""

    by_group: dict[int, list[ImageSpec]] = {}
    for spec in sorted(members, key=lambda s: _hash_int("order", seed, s.image_id)):
        by_group.setdefault(spec.group, []).append(spec)
    groups = sorted(by_group)
    order = proportional_sequence([len(by_group[g]) for g in groups])
    queues = {g: iter(by_group[g]) for g in groups}
    return [next(queues[groups[label]]) for label in order]


def reserve_anchors(
    train: Sequence[ImageSpec], seed: int
) -> dict[str, list[ImageSpec]]:
    """Anchor pools: calibration, reference, three banks, unused; stratified by group.

    Training images sorted by group (then a seeded hash) are dealt onto an evenly
    spread pool sequence, so every pool holds each group in proportion (within one
    image). Within a pool, the order makes every prefix stratified, so a bank's
    first 256 positions are too.
    """

    sizes = [size for _, size in ANCHOR_POOLS]
    names = [name for name, _ in ANCHOR_POOLS] + ["unused"]
    if sum(sizes) > len(train):
        raise ValueError("not enough training images for the anchor pools")
    sequence = proportional_sequence([*sizes, len(train) - sum(sizes)])
    ordered = sorted(
        train, key=lambda s: (s.group, _hash_int("pool", seed, s.image_id))
    )
    pools: dict[str, list[ImageSpec]] = {name: [] for name in names}
    for label, spec in zip(sequence, ordered, strict=True):
        pools[names[label]].append(spec)
    return {name: _stratified_order(members, seed) for name, members in pools.items()}


def partners(
    anchors: Iterable[ImageSpec], registry: Registry, seed: int
) -> list[ImageSpec]:
    """For each anchor, B- and O-partners from the opposite category's train pool."""

    result: list[ImageSpec] = []
    for anchor in anchors:
        b = 1 - anchor.b_category
        o = 1 - anchor.o_category
        b_pool = registry.split("background", b, "train")
        o_pool = registry.split("object", o, "train")
        new_b = b_pool[_hash_int("b-partner", seed, anchor.image_id) % len(b_pool)]
        new_o = o_pool[_hash_int("o-partner", seed, anchor.image_id) % len(o_pool)]
        result.append(
            ImageSpec(
                image_id=f"partner-b/{anchor.image_id}",
                role="partner_b",
                split="train",
                label=anchor.label,
                b_category=b,
                o_category=anchor.o_category,
                car=anchor.car,
                background=new_b.component_id,
                obj=anchor.obj,
                anchor=anchor.image_id,
            )
        )
        result.append(
            ImageSpec(
                image_id=f"partner-o/{anchor.image_id}",
                role="partner_o",
                split="train",
                label=anchor.label,
                b_category=anchor.b_category,
                o_category=o,
                car=anchor.car,
                background=anchor.background,
                obj=new_o.component_id,
                anchor=anchor.image_id,
            )
        )
    return result


def assert_split_isolation(images: Sequence[ImageSpec], registry: Registry) -> None:
    """No val or test component in any training image or partner."""

    splits = registry.pool_split()
    for spec in images:
        for component in (spec.car, spec.background, spec.obj):
            if splits[component] != spec.split:
                raise AssertionError(
                    f"integrity: {spec.image_id} ({spec.split}) uses {component} from "
                    f"the {splits[component]} pool"
                )


def group_counts(images: Sequence[ImageSpec]) -> dict[str, list[int]]:
    counts: dict[str, list[int]] = {}
    for spec in images:
        if spec.role == "standard":
            counts.setdefault(spec.split, [0] * 8)[spec.group] += 1
    return counts


def expected_group_counts(construction: Construction) -> dict[str, list[int]]:
    result: dict[str, list[int]] = {}
    for split in cast(tuple[PoolSplit, ...], ("train", "val", "test")):
        counts = [0] * 8
        for label, b, o in product(range(2), repeat=3):
            counts[group_index(label, b == label, o == label)] = composition_count(
                split, label, b, o, construction
            )
        result[split] = counts
    return result


# Build.


def render_jobs(
    layout: SourceLayout,
    registry: Registry,
    images: Sequence[ImageSpec],
    construction: Construction,
    output_root: Path,
) -> list[RenderJob]:
    by_id = registry.by_id()
    jobs: list[RenderJob] = []
    for spec in images:
        car, obj = by_id[spec.car], by_id[spec.obj]
        jobs.append(
            RenderJob(
                image_id=spec.image_id,
                output=str(output_root / spec.path),
                car_image=str(source_file(layout, car)),
                car_mask=str(layout.car_mask(car.source["split"], car.source["fname"])),
                background=str(source_file(layout, by_id[spec.background])),
                object_image=str(source_file(layout, obj)),
                object_mask=str(layout.export / obj.source["mask"]),
                car_scale=construction.car_scale,
                object_scale=construction.object_scale,
                object_center=construction.object_center,
            )
        )
    return jobs


def compare_with_upstream(
    layout: SourceLayout,
    construction: Construction,
    images: Sequence[ImageSpec],
    output_root: Path,
    per_split: int = 200,
) -> dict[str, Any]:
    """G0b: decoded-pixel differences against an unmodified upstream run.

    Not blocking: a deterministic, distributionally equivalent regeneration is
    accepted when pixel identity is impractical.
    """

    if construction.upstream_folder("train") is None:
        return {"status": "not applicable (upstream cannot generate this construction)"}
    result: dict[str, Any] = {"status": "compared"}
    for split in ("train", "val", "test"):
        folder = construction.upstream_folder(split)
        assert folder is not None
        root = layout.upstream_output / folder
        if not (root / split).is_dir():
            result[split] = {"status": f"no upstream output at {root / split}"}
            continue
        members = [s for s in images if s.role == "standard" and s.split == split]
        step = max(1, len(members) // per_split)
        chosen = members[::step][:per_split]
        maxima: list[int] = []
        means: list[float] = []
        missing = 0
        for spec in chosen:
            assert spec.composition is not None and spec.index is not None
            theirs_path = root / split / spec.composition / f"{spec.index:03d}.jpg"
            if not theirs_path.is_file():
                missing += 1
                continue
            ours = np.asarray(Image.open(output_root / spec.path), dtype=np.int16)
            theirs = np.asarray(Image.open(theirs_path), dtype=np.int16)
            difference = np.abs(ours - theirs)
            maxima.append(int(difference.max()))
            means.append(float(difference.mean()))
        result[split] = {
            "upstream_root": str(root),
            "compared": len(maxima),
            "missing_upstream": missing,
            "identical": sum(m == 0 for m in maxima),
            "max_abs_difference": max(maxima, default=None),
            "mean_abs_difference": float(np.mean(means)) if means else None,
            "worst_image_mean": max(means, default=None),
        }
    return result


def write_preview(
    output_root: Path, images: Sequence[ImageSpec], name: str, columns: int = 8
) -> Path:
    """A contact sheet of rendered images (row-major), for visual inspection."""

    tiles = [Image.open(output_root / s.path).convert("RGB") for s in images]
    rows = math.ceil(len(tiles) / columns)
    sheet = Image.new("RGB", (columns * CANVAS, rows * CANVAS), "white")
    for i, tile in enumerate(tiles):
        sheet.paste(tile, ((i % columns) * CANVAS, (i // columns) * CANVAS))
    path = output_root / "preview" / name
    path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(path, quality=90)
    return path


def build_urbancars(
    layout: SourceLayout,
    construction: Construction,
    output_root: Path,
    *,
    seed: int,
    executor: Executor | None = None,
    hash_sources: bool = True,
) -> dict[str, Any]:
    """Build: registry, source log, standard splits, anchors, partners, renders.

    Writes ``dataset-manifest.json`` last; everything before it is resumable.
    """

    if (output_root / "dataset-manifest.json").exists():
        raise FileExistsError(f"UrbanCars dataset already built: {output_root}")
    export = load_export(layout)
    acceptance = check_sources(layout, export)
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "g0a.json").write_text(json.dumps(acceptance, indent=2) + "\n")
    if not acceptance["passes"]:
        raise ValueError(f"G0a failed: {acceptance['failures']} (see g0a.json)")
    registry = build_registry(layout)
    log = source_id_log(layout, registry, hash_files=hash_sources)
    (output_root / "source-log.json").write_text(json.dumps(log) + "\n")

    standard = standard_images(registry, construction)
    counts, expected = group_counts(standard), expected_group_counts(construction)
    if counts != expected:
        raise AssertionError(f"integrity: group counts {counts} != {expected}")
    train = [s for s in standard if s.split == "train"]
    anchors = reserve_anchors(train, seed)
    reserved = [s for name, pool in anchors.items() if name != "unused" for s in pool]
    partner_specs = partners(reserved, registry, seed)
    images = [*standard, *partner_specs]
    assert_split_isolation(images, registry)

    jobs = render_jobs(layout, registry, images, construction, output_root)
    outcomes = list(
        executor.map(render_and_save, jobs, chunksize=16)
        if executor is not None
        else map(render_and_save, jobs)
    )
    by_image = {o["image_id"]: o for o in outcomes}
    clipped = [o["image_id"] for o in outcomes if o["clipped"]]
    if construction.name == "standard" and clipped:
        raise AssertionError(f"integrity: standard placement clipped {clipped[:5]}")
    visible = np.array([o["object_visible_fraction"] for o in outcomes])

    pools_record = {
        name: [
            {
                "anchor": spec.image_id,
                "group": spec.group,
                "u": _uniform("u", seed, name, spec.image_id),
            }
            for spec in members
        ]
        for name, members in anchors.items()
    }
    records = [
        {
            **asdict(spec),
            "group": spec.group,
            "path": spec.path,
            **by_image[spec.image_id],
        }
        for spec in images
    ]
    first_of_each = [
        next(s for s in standard if s.split == "train" and s.composition == c)
        for c in dict.fromkeys(s.composition for s in standard if s.split == "train")
    ]
    grid = [
        s
        for c in first_of_each
        for s in [x for x in standard if x.composition == c.composition][:4]
    ]
    write_preview(output_root, grid, "train-compositions.jpg")
    pair_preview = [
        s
        for anchor in anchors["bank0"][:8]
        for s in (
            anchor,
            *(p for p in partner_specs if p.anchor == anchor.image_id),
        )
    ]
    write_preview(output_root, pair_preview, "anchor-b-o-partners.jpg", columns=6)
    manifest = {
        "schema_version": MANIFEST_SCHEMA,
        "construction": construction.model_dump(mode="json"),
        "seed": seed,
        "upstream_commit": UPSTREAM_COMMIT,
        "export_digest": registry.export_digest,
        "metadata_digest": registry.metadata_digest,
        "segmenter": export.get("segmenter"),
        "fallback_masks": acceptance["fallback_masks"],
        "group_counts": counts,
        "anchor_pools": pools_record,
        "clipped": clipped,
        "object_visible_fraction": {
            "min": float(visible.min()),
            "p05": float(np.quantile(visible, 0.05)),
            "median": float(np.median(visible)),
            "fraction_below_half": float((visible < 0.5).mean()),
        },
        "source_findings": {k: v for k, v in log.items() if k != "entries"},
        "upstream_fidelity": compare_with_upstream(
            layout, construction, standard, output_root
        ),
        "images": records,
    }
    manifest["digest"] = canonical_digest_value(manifest)
    (output_root / "dataset-manifest.json").write_text(json.dumps(manifest) + "\n")
    return manifest


@dataclass(frozen=True)
class UrbanCarsDataset:
    root: Path
    manifest: dict[str, Any]

    @property
    def digest(self) -> str:
        return cast(str, self.manifest["digest"])

    def images(
        self, predicate: Callable[[dict[str, Any]], bool]
    ) -> list[dict[str, Any]]:
        return [r for r in self.manifest["images"] if predicate(r)]


def load_urbancars_dataset(root: Path) -> UrbanCarsDataset:
    manifest = cast(
        dict[str, Any], json.loads((root / "dataset-manifest.json").read_text())
    )
    if manifest.get("schema_version") != MANIFEST_SCHEMA:
        raise ValueError(f"not an UrbanCars dataset: {root}")
    stored = manifest.pop("digest")
    if canonical_digest_value(manifest) != stored:
        raise ValueError(f"UrbanCars manifest digest mismatch: {root}")
    manifest["digest"] = stored
    return UrbanCarsDataset(root, manifest)
