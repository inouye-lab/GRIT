"""UrbanCars rebuild: upstream replay, renderer, anchors, partners, isolation, cache."""

# pyright: reportPrivateUsage=false
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw

from grit.data import urbancars as uc
from grit.features.cmnist import DeterministicFakeEncoder
from grit.features.urbancars import (
    CLIP_MEAN,
    CLIP_STD,
    load_urbancars_test_table,
    load_urbancars_tuning_tables,
    prepare_urbancars_feature_cache,
)


def test_standard_and_f2_group_counts_match_upstream() -> None:
    standard = uc.expected_group_counts(uc.CONSTRUCTIONS["standard"])
    assert standard["train"] == [3610, 190, 190, 10] * 2
    assert standard["val"] == standard["test"] == [125] * 8
    f2 = uc.expected_group_counts(uc.CONSTRUCTIONS["f2"])
    assert f2["train"] == [3762, 198, 38, 2] * 2


def _upstream_rescale(
    mask: torch.Tensor, image: torch.Tensor, scale: float, center: tuple[int, int]
) -> tuple[torch.Tensor, torch.Tensor]:
    """Upstream's helper without clipping, for comparison."""

    factor = scale * 256 / max(mask.shape[1:])
    m = F.interpolate(
        mask.unsqueeze(0).float(), scale_factor=factor, mode="nearest-exact"
    ).squeeze(0)
    i = F.interpolate(
        image.unsqueeze(0),
        scale_factor=factor,
        mode="bilinear",
        align_corners=False,
        antialias=True,
    ).squeeze(0)
    h, w = m.shape[1:]
    top, left = max(0, center[0] - h // 2), max(0, center[1] - w // 2)
    target_mask = torch.zeros((1, 256, 256), dtype=torch.long)
    target_mask[:, top : top + h, left : left + w] = m
    target_img = torch.zeros((3, 256, 256))
    target_img[:, top : top + h, left : left + w] = i
    return target_mask, target_img


def test_vendored_rescale_matches_upstream_and_clips_instead_of_failing() -> None:
    generator = torch.Generator().manual_seed(0)
    image = torch.rand((3, 90, 140), generator=generator)
    mask = (torch.rand((1, 90, 140), generator=generator) > 0.3).long()
    for construction in uc.CONSTRUCTIONS.values():
        center = construction.object_center
        mask_ours, image_ours, clipped = uc.rescale_cropped_mask_and_img(
            mask, image, construction.object_scale, center
        )
        assert not clipped
        mask_up, image_up = _upstream_rescale(
            mask, image, construction.object_scale, center
        )
        assert torch.equal(mask_ours, mask_up) and torch.equal(image_ours, image_up)
    _, _, clipped = uc.rescale_cropped_mask_and_img(mask, image, 0.5, (128, 250))
    assert clipped
    with pytest.raises(RuntimeError):
        _ = _upstream_rescale(mask, image, 0.5, (128, 250))


def test_proportional_sequence_is_exact_and_even() -> None:
    sequence = uc.proportional_sequence([1024, 2048, 1024, 1024, 1024, 1856])
    assert [sequence.count(k) for k in range(6)] == [1024, 2048, 1024, 1024, 1024, 1856]
    prefix = sequence[:800]
    assert abs(prefix.count(1) - 800 * 2048 / 8000) <= 1


def _spec(image_id: str, label: int, b: int, o: int) -> uc.ImageSpec:
    return uc.ImageSpec(image_id, "standard", "train", label, b, o, "c", "b", "o")


def test_anchor_pools_are_stratified_by_group() -> None:
    counts = uc.expected_group_counts(uc.CONSTRUCTIONS["standard"])["train"]
    train: list[uc.ImageSpec] = []
    for group, count in enumerate(counts):
        label, within = divmod(group, 4)
        b_aligned, o_aligned = uc_within(within)
        b = label if b_aligned else 1 - label
        o = label if o_aligned else 1 - label
        train += [_spec(f"g{group}/{i}", label, b, o) for i in range(count)]
    pools = uc.reserve_anchors(train, 1729)
    assert [len(pools[n]) for n, _ in uc.ANCHOR_POOLS] == [1024, 2048, 1024, 1024, 1024]
    assert len(pools["unused"]) == 8000 - 6144
    for name, size in uc.ANCHOR_POOLS:
        members = pools[name]
        for group, count in enumerate(counts):
            observed = sum(s.group == group for s in members)
            assert abs(observed - count * size / 8000) <= 1
        head = members[:256]
        rare = sum(s.group in (3, 7) for s in head)
        assert rare <= 2  # 20 of 8000 overall
    assert pools == uc.reserve_anchors(list(reversed(train)), 1729)


def uc_within(position: int) -> tuple[bool, bool]:
    from grit.geometry import WITHIN_LABEL

    return WITHIN_LABEL[position]


# A small synthetic source tree, with the pool constants shrunk to fit.


def _jpeg(path: Path, size: tuple[int, int], colour: tuple[int, int, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, colour).save(path)


def _mask(path: Path, size: tuple[int, int], box: tuple[int, int, int, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = Image.new("L", size, 0)
    ImageDraw.Draw(image).ellipse(box, fill=255)
    image.save(path)


@pytest.fixture
def small(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> uc.SourceLayout:
    monkeypatch.setattr(
        uc, "POOL_SPLITS", {"train": (0, 40), "val": (40, 48), "test": (48, 56)}
    )
    monkeypatch.setattr(uc, "MINIMUM_POOL", 56)
    monkeypatch.setattr(
        uc,
        "ANCHOR_POOLS",
        (
            ("calibration", 8),
            ("reference", 16),
            ("bank0", 8),
            ("bank1", 8),
            ("bank2", 8),
        ),
    )
    layout = uc.SourceLayout(tmp_path / "data" / "urbancars")
    metadata = layout.metadata
    rng = np.random.default_rng(0)
    cars: dict[str, list[dict[str, object]]] = {"urban": [], "country": []}
    listed: dict[str, list[str]] = {"train": [], "test": []}
    for n in range(120):
        split = "train" if n % 2 == 0 else "test"
        fname = f"{n + 1:05d}.jpg"
        colour = tuple(int(v) for v in rng.integers(0, 255, 3))
        _jpeg(layout.car_image(split, fname), (120, 80), colour)  # pyright: ignore[reportArgumentType]
        _mask(layout.car_mask(split, fname), (120, 80), (10, 10, 110, 70))
        cars[uc.CATEGORIES[n % 4 // 2]].append(
            {"split": split, "fname": fname, "class": n % 4, "bbox": [0, 0, 1, 1]}
        )
        listed[split].append(fname)
    for split, names in listed.items():
        (metadata / "stanford_cars").mkdir(parents=True, exist_ok=True)
        (metadata / "stanford_cars" / f"{split}.txt").write_text("\n".join(names))
    monkeypatch.setattr(uc, "STANFORD_COUNTS", {"train": 60, "test": 60})
    monkeypatch.setattr(uc, "FILTERED_COUNTS", {"train": 60, "test": 60})
    for category, classes in uc.PLACES_CLASSES.items():
        per_class = 63 // len(classes) + 1
        for places_class in classes:
            for split, count in (("train", per_class - 1), ("val", 1)):
                names = [f"{places_class}-{split}-{i:03d}.jpg" for i in range(count)]
                for name in names:
                    colour = (0, 90, 0) if category == "country" else (90, 90, 90)
                    _jpeg(
                        layout.places / split / places_class / name, (256, 256), colour
                    )
                listing = (
                    metadata / "places" / "filter_objs" / split / f"{places_class}.txt"
                )
                listing.parent.mkdir(parents=True, exist_ok=True)
                listing.write_text("\n".join(names))
    objects: dict[str, list[dict[str, object]]] = {"urban": [], "country": []}
    for n in range(120):
        category = uc.CATEGORIES[n % 2]
        name = f"{n:012d}.jpg"
        # LVIS train and val reuse annotation numbers: n and n + 60 share one, in
        # the same category pool, with different photos and mask sizes.
        lvis_split, number = ("train", n) if n < 60 else ("val", n - 60)
        size = (100, 90) if n < 60 else (70, 110)
        mask = f"lvis_masks/{lvis_split}-{number}.png"
        _jpeg(layout.coco / "train2017" / name, size, (200, 40 * (n % 2), 0))
        _mask(layout.export / mask, size, (10, 10, size[0] - 10, size[1] - 5))
        objects[category].append(
            {
                "annotation_id": number,
                "image_id": n,
                "category_id": 1,
                "lvis_split": lvis_split,
                "coco_file": f"train2017/{name}",
                "mask": mask,
            }
        )
    (layout.export / "components.json").write_text(
        json.dumps(
            {
                "schema": uc.EXPORT_SCHEMA,
                "upstream_commit": uc.UPSTREAM_COMMIT,
                "segmenter": "synthetic",
                "cars": cars,
                "objects": objects,
            }
        )
    )
    return layout


class _FakePilEncoder(DeterministicFakeEncoder):
    def encode_pil(self, images: tuple[Image.Image, ...]) -> torch.Tensor:
        return self.encode(_preprocess(images)[:, :, ::28, ::28].clamp(0, 1))


def _preprocess(images: tuple[Image.Image, ...]) -> torch.Tensor:
    mean = torch.tensor(CLIP_MEAN).view(3, 1, 1)
    std = torch.tensor(CLIP_STD).view(3, 1, 1)
    tensors = [
        (
            torch.tensor(
                np.asarray(i.convert("RGB").resize((224, 224))), dtype=torch.float32
            ).permute(2, 0, 1)
            / 255
            - mean
        )
        / std
        for i in images
    ]
    return torch.stack(tensors)


def test_build_records_components_isolates_splits_and_seals_test(
    small: uc.SourceLayout, tmp_path: Path
) -> None:
    root = tmp_path / "built"
    construction = uc.CONSTRUCTIONS["standard"]
    manifest = uc.build_urbancars(small, construction, root, seed=1729)
    assert manifest["group_counts"] == uc.expected_group_counts(construction)
    assert manifest["clipped"] == []
    findings = manifest["source_findings"]
    assert findings["duplicate_sources_across_splits"] == []
    registry = uc.build_registry(small)
    splits = registry.pool_split()
    for record in manifest["images"]:
        for key in ("car", "background", "obj"):
            assert splits[record[key]] == record["split"]
        if record["role"] == "partner_b":
            anchor = next(
                r for r in manifest["images"] if r["image_id"] == record["anchor"]
            )
            assert record["car"] == anchor["car"] and record["obj"] == anchor["obj"]
            assert record["b_category"] == 1 - anchor["b_category"]
    reserved = sum(len(v) for k, v in manifest["anchor_pools"].items() if k != "unused")
    partners = [r for r in manifest["images"] if r["role"] != "standard"]
    assert len(partners) == 2 * reserved == 96
    assert (root / "preview" / "train-compositions.jpg").is_file()
    again = uc.build_urbancars(small, construction, tmp_path / "again", seed=1729)
    assert [r["sha256"] for r in again["images"]] == [
        r["sha256"] for r in manifest["images"]
    ]

    ids = {c.component_id for c in registry.by_id().values() if c.kind == "object"}
    assert "obj:lvis-train-0" in ids and "obj:lvis-val-0" in ids
    assert len(ids) == 120

    dataset = uc.load_urbancars_dataset(root)
    cache = root / "features"
    features = prepare_urbancars_feature_cache(
        dataset, _FakePilEncoder(), _preprocess, cache
    )
    assert all(features["canvas_check"]["frame_kept"].values())
    tables = load_urbancars_tuning_tables(cache, dataset)
    assert set(tables) == {"train", "validation", "partner_b", "partner_o"}
    ids, test = load_urbancars_test_table(cache, dataset)
    assert len(ids) == test.shape[0] == 16


def test_center_crop_is_detected(tmp_path: Path) -> None:
    from grit.features.urbancars import canvas_check

    def cropping(images: tuple[Image.Image, ...]) -> torch.Tensor:
        return _preprocess(tuple(i.crop((16, 16, 240, 240)) for i in images))

    sample = Image.new("RGB", (256, 256), (10, 20, 30))
    with pytest.raises(AssertionError, match="crops the canvas"):
        _ = canvas_check(cropping, [sample], tmp_path / "x.png")


def test_fidelity_reads_each_splits_upstream_folder(
    small: uc.SourceLayout, tmp_path: Path
) -> None:
    """Upstream writes val and test under bg-0.5_co_occur_obj-0.5."""

    root = tmp_path / "built"
    construction = uc.CONSTRUCTIONS["standard"]
    manifest = uc.build_urbancars(small, construction, root, seed=1729)
    standard = [
        uc.ImageSpec(**{k: r[k] for k in uc.ImageSpec.__dataclass_fields__})
        for r in manifest["images"]
        if r["role"] == "standard"
    ]
    for spec in standard:
        folder = construction.upstream_folder(spec.split)
        assert folder is not None and spec.composition is not None
        target = small.upstream_output / folder / spec.split / spec.composition
        target.mkdir(parents=True, exist_ok=True)
        (target / f"{spec.index:03d}.jpg").write_bytes((root / spec.path).read_bytes())
    assert construction.upstream_folder("val") == "bg-0.5_co_occur_obj-0.5"
    result = uc.compare_with_upstream(small, construction, standard, root)
    for split in ("train", "val", "test"):
        assert result[split]["missing_upstream"] == 0
        assert result[split]["identical"] == result[split]["compared"] > 0
    assert uc.CONSTRUCTIONS["f1"].upstream_folder("train") is None
