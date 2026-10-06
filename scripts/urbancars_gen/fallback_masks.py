"""Fallback car masks for listed Stanford Cars photos that got none (a deviation).

Upstream keeps the panoptic segment of category car or truck (COCO panoptic 2 or 7)
with the largest IoU with the Stanford box, and writes no mask when none exists. Our
rerun of that step left a few listed photos without a mask, which upstream's own
generator cannot load. For exactly those photos, this reruns the same predictor and
resizing with MaskFormer's object-confidence threshold lowered step by step (0.5,
0.3, 0.1 instead of 0.8). At the first step that has one, it keeps the car or truck
segment (then bus) with the largest box IoU, provided that IoU is at least 0.3 and
at least 80% of the segment lies inside the box (padded by 5%). A photo with no such
segment gets a GrabCut mask initialized from the Stanford box. Every other mask is
untouched, so the pools and their order stay upstream's. Masks written by an earlier
fallback run are redone.

Writes the masks where upstream expects them, `export/fallback_masks.json` (file,
category, IoU, threshold) and `export/fallback_masks.jpg` (overlays for inspection).
Run from the pinned Whac-A-Mole clone in the generation environment:

    bash scripts/urbancars_gen/gen.sh fallback
"""

import argparse
import json
import os

import cv2
import numpy as np
import scipy.io as sio
import torch
import torch.nn.functional as F
from create_datasets.urbancars.maskformer.mask_former.config import (
    add_mask_former_config,
)
from detectron2.config import get_cfg
from detectron2.data.detection_utils import read_image
from detectron2.engine.defaults import DefaultPredictor
from detectron2.projects.deeplab import add_deeplab_config

METADATA = "create_datasets/urbancars/metadata/stanford_cars"
CONFIG = f"{METADATA}/maskformer_panoptic_swin_large_IN21k_384_bs64_554k.yaml"
VEHICLES = ((2, 7), (5,))  # car and truck first, then bus
THRESHOLDS = (0.5, 0.3, 0.1)
MIN_IOU = 0.3
MIN_INSIDE = 0.8
BOX_PAD = 0.05


def boxes(root, split):
    path = (
        os.path.join(root, "devkit", "cars_train_annos.mat")
        if split == "train"
        else os.path.join(root, "cars_test_annos_withlabels.mat")
    )
    return {
        a["fname"]: [
            int(a["bbox_x1"]),
            int(a["bbox_y1"]),
            int(a["bbox_x2"]),
            int(a["bbox_y2"]),
        ]
        for a in sio.loadmat(path, squeeze_me=True)["annotations"]
    }


@torch.no_grad()
def predict(predictor, image_path, bbox):
    """Upstream's resizing and box-IoU rule at lowered object-confidence thresholds."""

    img = read_image(image_path, format="BGR")
    h, w = img.shape[:2]
    shorter = min(h, w)
    resized = img
    if shorter > 1024:
        scale = 1024 / shorter
        resized = cv2.resize(img, (int(w * scale), int(h * scale)))
    for threshold in THRESHOLDS:
        predictor.model.object_mask_threshold = threshold
        predictor.model.overlap_threshold = threshold
        panoptic, segments = predictor(resized)["panoptic_seg"]
        if shorter > 1024:
            panoptic = F.interpolate(
                panoptic.unsqueeze(0).unsqueeze(0).float(), (h, w), mode="nearest"
            )[0, 0].int()
        ph, pw = panoptic.shape[:2]
        x1, y1, x2, y2 = bbox
        box = torch.zeros((ph, pw), dtype=torch.long, device=panoptic.device)
        box[y1 : y2 + 1, x1 : x2 + 1] = 1
        box_area = int(box.sum())
        px, py = int(BOX_PAD * (x2 - x1)), int(BOX_PAD * (y2 - y1))
        padded = torch.zeros_like(box)
        padded[max(0, y1 - py) : y2 + py + 1, max(0, x1 - px) : x2 + px + 1] = 1
        for categories in VEHICLES:
            best = None
            for seg in segments:
                if seg["category_id"] not in categories:
                    continue
                mask = panoptic == seg["id"]
                area = int(mask.sum())
                if area == ph * pw:
                    continue
                inter = int((mask * box).sum())
                iou = inter / (area + box_area - inter + 1e-6)
                if int((mask * padded).sum()) < MIN_INSIDE * area:
                    continue
                if best is None or iou > best[0]:
                    best = (iou, seg, mask, threshold)
            if best is not None and best[0] >= MIN_IOU:
                return img, best
    return img, None


def grabcut(img, bbox):
    """GrabCut (OpenCV, 5 iterations) initialized from the Stanford box."""

    x1, y1, x2, y2 = bbox
    mask = np.zeros(img.shape[:2], np.uint8)
    background = np.zeros((1, 65), np.float64)
    foreground = np.zeros((1, 65), np.float64)
    rect = (x1, y1, x2 - x1 + 1, y2 - y1 + 1)
    cv2.setRNGSeed(0)
    cv2.grabCut(img, mask, rect, background, foreground, 5, cv2.GC_INIT_WITH_RECT)
    return np.isin(mask, (cv2.GC_FGD, cv2.GC_PR_FGD)).astype(np.uint8)


def overlay_tile(img, np_mask, bbox, label):
    overlay = img.copy()
    overlay[np_mask > 0] = (0.5 * overlay[np_mask > 0] + [0, 0, 127]).astype(np.uint8)
    x1, y1, x2, y2 = bbox
    cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 255, 0), 3)
    tile = cv2.resize(overlay, (320, int(320 * overlay.shape[0] / overlay.shape[1])))
    tile = cv2.copyMakeBorder(
        tile, 0, max(0, 240 - tile.shape[0]), 0, 0, cv2.BORDER_CONSTANT
    )[:240]
    cv2.putText(
        tile, label, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1
    )
    return tile


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="the upstream data/ directory")
    args = parser.parse_args()
    cfg = get_cfg()
    add_deeplab_config(cfg)
    add_mask_former_config(cfg)
    cfg.merge_from_file(CONFIG)
    cfg.freeze()
    predictor = DefaultPredictor(cfg)
    root = os.path.join(args.data, "stanford_cars")
    out = os.path.join(args.data, "export")
    previous_path = os.path.join(out, "fallback_masks.json")
    previous = set()
    if os.path.exists(previous_path):
        with open(previous_path) as f:
            previous = {(r["split"], r["fname"]) for r in json.load(f)}
    records, tiles = [], []
    for split in ("train", "test"):
        with open(os.path.join(METADATA, f"{split}.txt")) as f:
            listed = [line.strip() for line in f if line.strip()]
        box_of = boxes(root, split)
        for fname in listed:
            mask_path = os.path.join(
                root, "mask", f"cars_{split}", fname[:-4] + "_mask.png"
            )
            if os.path.exists(mask_path) and (split, fname) not in previous:
                continue
            img, best = predict(
                predictor, os.path.join(root, f"cars_{split}", fname), box_of[fname]
            )
            record = {"split": split, "fname": fname, "bbox": box_of[fname]}
            if best is None:
                np_mask = grabcut(img, box_of[fname])
                cv2.imwrite(mask_path, np_mask * 255)
                record.update(status="fallback", method="grabcut")
                label = f"{split}/{fname} grabcut"
                tiles.append(overlay_tile(img, np_mask, box_of[fname], label))
            else:
                iou, seg, mask, threshold = best
                np_mask = mask.long().cpu().numpy().astype(np.uint8)
                cv2.imwrite(mask_path, np_mask * 255)
                record.update(
                    status="fallback",
                    method="maskformer",
                    category_id=int(seg["category_id"]),
                    iou=round(iou, 4),
                    object_mask_threshold=threshold,
                )
                label = (
                    f"{split}/{fname} cat {seg['category_id']} "
                    f"iou {iou:.2f} t {threshold}"
                )
                tiles.append(overlay_tile(img, np_mask, box_of[fname], label))
            records.append(record)
            print(record, flush=True)
    os.makedirs(out, exist_ok=True)
    with open(previous_path, "w") as f:
        json.dump(records, f, indent=1)
    if tiles:
        blank = np.zeros_like(tiles[0])
        rows = [
            np.hstack(tiles[i : i + 4] + [blank] * (4 - len(tiles[i : i + 4])))
            for i in range(0, len(tiles), 4)
        ]
        cv2.imwrite(os.path.join(out, "fallback_masks.jpg"), np.vstack(rows))
    made = sum(r["status"] == "fallback" for r in records)
    print(f"{made} fallback masks; {len(records) - made} still missing")


if __name__ == "__main__":
    main()
