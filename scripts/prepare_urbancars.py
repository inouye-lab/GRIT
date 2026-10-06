"""UrbanCars preparation: sources, the component-recording rebuild, CLIP features.

    uv run scripts/prepare_urbancars.py download --stanford-cars <mirror>
    uv run scripts/prepare_urbancars.py build --construction standard
    uv run scripts/prepare_urbancars.py features --construction standard --device cuda:0

`download` clones upstream at the pinned commit, fetches COCO 2017, LVIS v1, the
Places365-Standard easy-format archive (only the ten UrbanCars classes are
extracted) and the MaskFormer weights into `$PROJECT_SCRATCH/data/urbancars` (the
upstream `data/` layout), recording each archive's SHA-256, and links a Stanford Cars
mirror into the original layout. Car masks and the LVIS export come from the
generation environment (`bash scripts/urbancars_gen/gen.sh ...`) between `download`
and `build`. `build` checks G0a, logs every component's source IDs, replays the
upstream splits, reserves anchors, renders images and partners, and compares with an
upstream run when one exists (G0b). `results/README.md` lists these checks.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing
import os
import shutil
import subprocess
import tarfile
import urllib.request
import zipfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from grit.data.urbancars import (
    CONSTRUCTIONS,
    PLACES_CLASSES,
    UPSTREAM_COMMIT,
    UPSTREAM_REPOSITORY,
    SourceLayout,
    build_urbancars,
    file_sha256,
    load_urbancars_dataset,
    normalize_stanford_cars,
)
from grit.paths import (
    DEFAULT_CONSTRUCTION_SEED,
    auto_device,
    default_clip_weights_root,
    scratch_root,
)

ARCHIVES = {
    "coco/train2017.zip": "http://images.cocodataset.org/zips/train2017.zip",
    "coco/val2017.zip": "http://images.cocodataset.org/zips/val2017.zip",
    "lvis/lvis_v1_train.json.zip": "https://s3-us-west-2.amazonaws.com/dl.fbaipublicfiles.com/LVIS/lvis_v1_train.json.zip",
    "lvis/lvis_v1_val.json.zip": "https://s3-us-west-2.amazonaws.com/dl.fbaipublicfiles.com/LVIS/lvis_v1_val.json.zip",
    "places/places365standard_easyformat.tar": "http://data.csail.mit.edu/places/places365/places365standard_easyformat.tar",
    "weights/maskformer_panoptic_swin_large.pkl": "https://dl.fbaipublicfiles.com/maskformer/panoptic-coco/maskformer_panoptic_swin_large_IN21k_384_bs64_554k/model_final_7505c4.pkl",
}


def fetch(url: str, path: Path) -> None:
    """Stream to `<path>.part`, resuming a partial download, then rename."""

    if path.is_file():
        return
    part = path.with_name(path.name + ".part")
    path.parent.mkdir(parents=True, exist_ok=True)
    start = part.stat().st_size if part.exists() else 0
    request = urllib.request.Request(url, headers={"Range": f"bytes={start}-"})
    with urllib.request.urlopen(request) as response:  # noqa: S310
        mode = "ab" if start and response.status == 206 else "wb"
        with part.open(mode) as stream:
            shutil.copyfileobj(response, stream, 1 << 24)
    part.rename(path)
    print(f"downloaded {path}", flush=True)


def extract(archive: Path, layout: SourceLayout) -> None:
    name = archive.name
    marker = archive.with_name(name + ".extracted")
    if marker.exists():
        return
    if name.endswith(".pkl"):
        pass
    elif name.endswith(".zip"):
        with zipfile.ZipFile(archive) as bundle:
            bundle.extractall(archive.parent)
    else:
        # Places: only the ten UrbanCars classes, into places/{train,val}/<class>/
        # (upstream's own extraction step puts the tree in the wrong place).
        wanted = {c for classes in PLACES_CLASSES.values() for c in classes}
        with tarfile.open(archive) as bundle:
            for member in bundle:
                parts = Path(member.name).parts
                if member.isfile() and len(parts) == 4 and parts[2] in wanted:
                    member.name = str(Path(*parts[1:]))
                    bundle.extract(member, layout.places)
    marker.write_text("ok\n")
    print(f"extracted {archive}", flush=True)


def download(args: argparse.Namespace, layout: SourceLayout) -> None:
    clone = layout.upstream
    if not (clone / ".git").is_dir():
        subprocess.run(["git", "clone", UPSTREAM_REPOSITORY, str(clone)], check=True)
    subprocess.run(
        ["git", "-C", str(clone), "checkout", "-q", UPSTREAM_COMMIT], check=True
    )
    manifest_path = layout.root / "downloads-manifest.json"
    manifest: dict[str, object] = (
        json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    )
    for relative, url in ARCHIVES.items():
        if args.skip_archives:
            break
        path = layout.root / relative
        fetch(url, path)
        if relative not in manifest:
            manifest[relative] = {"url": url, "sha256": file_sha256(path)}
            manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
        extract(path, layout)
    if args.stanford_cars is not None:
        links = normalize_stanford_cars(args.stanford_cars, layout.stanford_cars)
        manifest["stanford_cars_mirror"] = {
            name: {"source": source, "sha256": file_sha256(Path(source))}
            for name, source in links.items()
            if Path(source).is_file()
        }
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"sources": str(layout.root), "upstream": str(clone)}, indent=2))
    print("next: bash scripts/urbancars_gen/gen.sh setup (then masks, export)")


def _one_thread() -> None:
    import torch

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("step", choices=("download", "build", "features"))
    parser.add_argument(
        "--sources", type=Path, help="default: $PROJECT_SCRATCH/data/urbancars"
    )
    parser.add_argument("--stanford-cars", type=Path, help="a Stanford Cars mirror")
    parser.add_argument("--skip-archives", action="store_true")
    parser.add_argument(
        "--construction", choices=tuple(CONSTRUCTIONS), default="standard"
    )
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--seed", type=int, default=DEFAULT_CONSTRUCTION_SEED)
    parser.add_argument("--workers", type=int, default=os.cpu_count() or 1)
    parser.add_argument("--device", help="cpu, cuda, or cuda:N (default: auto)")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--clip-weights-root", type=Path)
    parser.add_argument("--no-download", action="store_true", help="CLIP weights")
    args = parser.parse_args()
    layout = SourceLayout(args.sources or scratch_root() / "data" / "urbancars")
    output_root: Path = args.output_root or (
        scratch_root() / "artifacts" / f"urbancars-v1-{args.construction}"
    )
    if args.step == "download":
        download(args, layout)
        return
    if args.step == "build":
        # One thread per render worker; otherwise each spawned worker's torch starts
        # a thread per core and the workers oversubscribe the machine.
        for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
            os.environ[key] = "1"
        context = multiprocessing.get_context("spawn")
        with ProcessPoolExecutor(
            max_workers=args.workers, mp_context=context, initializer=_one_thread
        ) as pool:
            manifest = build_urbancars(
                layout,
                CONSTRUCTIONS[args.construction],
                output_root,
                seed=args.seed,
                executor=pool,
            )
        summary = {
            key: manifest[key]
            for key in (
                "group_counts",
                "clipped",
                "object_visible_fraction",
                "source_findings",
                "upstream_fidelity",
            )
        }
        print(json.dumps(summary, indent=2))
        print(f"inspect {output_root / 'preview'} and {output_root / 'g0a.json'}")
        return
    from grit.features.cmnist import OfficialOpenAiClipEncoder
    from grit.features.urbancars import prepare_urbancars_feature_cache

    dataset = load_urbancars_dataset(output_root)
    encoder = OfficialOpenAiClipEncoder(
        weights_root=args.clip_weights_root or default_clip_weights_root(),
        allow_download=not args.no_download,
        device=args.device or auto_device(),
        batch_size=args.batch_size,
    )
    encoder.preflight()
    manifest = prepare_urbancars_feature_cache(
        dataset, encoder, encoder.model_inputs, output_root / "features"
    )
    print(json.dumps({"canvas_check": manifest["canvas_check"]}, indent=2))


if __name__ == "__main__":
    main()
