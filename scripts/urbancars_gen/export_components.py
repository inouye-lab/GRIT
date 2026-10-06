"""Export UrbanCars components in upstream order (runs in the generation environment).

Run from the pinned Whac-A-Mole clone (`gen.sh export` does this). Upstream's own
dataset classes decide the pools, so their order and filters are exactly upstream's:

- cars: `StanfordCars(split, body type)` samples, train then test (the upstream
  ConcatDataset order), with Stanford split, filename, class and box;
- co-objects: `LVISDataset(category, area threshold, bbox/mask standard)` per LVIS
  class in `meta_data.json` order, keeping upstream's single-component filter. Each
  kept annotation's mask comes from the same LVIS API call upstream uses
  (`ann_to_mask`) and is written as `lvis_masks/<lvis split>-<annotation id>.png`
  (0/255); LVIS train and val reuse annotation numbers.

Backgrounds need no special library and are rebuilt by `grit.data.urbancars`.
Writes `<data>/export/components.json` last.
"""

import argparse
import json
import os
import subprocess

import cv2
import numpy as np
from create_datasets.urbancars.src_dataset.lvis import LVISDataset
from create_datasets.urbancars.src_dataset.stanford_cars import StanfordCars

SCHEMA = "grit.urbancars-components/v1"
META = "create_datasets/urbancars/metadata/urbancars/meta_data.json"
# gen_urbancars.py: urban co-objects by bbox area >= 4500, country by mask area >= 3500.
THRESHOLDS = {"urban": (4500, "bbox"), "country": (3500, "mask")}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="the upstream data/ directory")
    parser.add_argument("--segmenter", required=True, help="car-mask model identity")
    args = parser.parse_args()
    with open(META) as f:
        meta = json.load(f)
    export_root = os.path.join(args.data, "export")
    mask_root = os.path.join(export_root, "lvis_masks")
    os.makedirs(mask_root, exist_ok=True)

    cars = {}
    for obj in meta["object"]:
        samples = []
        for split in ("train", "test"):
            dataset = StanfordCars("data", obj["id"], split=split)
            for path, target, bbox in dataset._samples:
                samples.append(
                    {
                        "split": split,
                        "fname": os.path.basename(path),
                        "class": int(target),
                        "bbox": [int(v) for v in bbox],
                    }
                )
        cars[obj["name"]] = samples
        print(f"cars {obj['name']}: {len(samples)}", flush=True)

    objects = {}
    for co in meta["co_occur_object"]:
        area, standard = THRESHOLDS[co["name"]]
        items = []
        for lvis_class in co["class_list"]:
            dataset = LVISDataset(
                "data",
                lvis_class["id"],
                area_thresh=area,
                threshold_standard=standard,
            )
            train_meta = dataset.split_to_meta_data["train"]
            val_meta = dataset.split_to_meta_data["val"]
            for ann in dataset.ann_list:
                image_id = ann["image_id"]
                # The same resolution order as LVISDataset.__getitem__.
                name = f"{image_id:012d}.jpg"
                if os.path.exists(os.path.join("data", "coco", "train2017", name)):
                    coco_file = f"train2017/{name}"
                elif os.path.exists(os.path.join("data", "coco", "val2017", name)):
                    coco_file = f"val2017/{name}"
                else:
                    raise FileNotFoundError(name)
                if image_id in train_meta.imgs:
                    lvis_split, mask = "train", train_meta.ann_to_mask(ann)
                elif image_id in val_meta.imgs:
                    lvis_split, mask = "val", val_meta.ann_to_mask(ann)
                else:
                    raise KeyError(image_id)
                # LVIS train and val reuse annotation numbers: key by both.
                mask_file = f"lvis_masks/{lvis_split}-{ann['id']}.png"
                cv2.imwrite(
                    os.path.join(export_root, mask_file),
                    (np.asarray(mask) > 0).astype(np.uint8) * 255,
                )
                items.append(
                    {
                        "annotation_id": int(ann["id"]),
                        "image_id": int(image_id),
                        "category_id": int(ann["category_id"]),
                        "lvis_split": lvis_split,
                        "coco_file": coco_file,
                        "mask": mask_file,
                        "area": float(ann["area"]),
                        "bbox": [float(v) for v in ann["bbox"]],
                    }
                )
            print(
                f"objects {co['name']}/{lvis_class['name']}: {len(items)}", flush=True
            )
        objects[co["name"]] = items

    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
    ).stdout.strip()
    export = {
        "schema": SCHEMA,
        "upstream_commit": commit,
        "segmenter": args.segmenter,
        "cars": cars,
        "objects": objects,
    }
    temporary = os.path.join(export_root, "components.json.tmp")
    with open(temporary, "w") as f:
        json.dump(export, f)
    os.replace(temporary, os.path.join(export_root, "components.json"))
    print("wrote", os.path.join(export_root, "components.json"))


if __name__ == "__main__":
    main()
