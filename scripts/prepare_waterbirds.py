"""Build Waterbirds-CF from local assets and cache CLIP features.

    uv run scripts/prepare_waterbirds.py

Needs released Waterbirds-95, CUB images, CUB masks, and the four Places365
categories under $PROJECT_SCRATCH/data/{waterbirds,cub,cub-masks,places} or the
matching flags. Never downloads those; only CLIP weights are fetched if missing.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Literal, cast

from grit.paths import (
    DEFAULT_CONSTRUCTION_SEED,
    auto_device,
    default_clip_weights_root,
    scratch_root,
)
from grit.search.waterbirds_runner import prepare_server_waterbirds


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--released-root", type=Path)
    parser.add_argument("--cub-root", type=Path)
    parser.add_argument("--masks-root", type=Path)
    parser.add_argument("--places-root", type=Path)
    parser.add_argument("--clip-weights-root", type=Path)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument(
        "--construction-seed", type=int, default=DEFAULT_CONSTRUCTION_SEED
    )
    parser.add_argument("--normalization", choices=("none", "l2"), default="none")
    parser.add_argument("--device", help="cpu, cuda, or cuda:N (default: auto)")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--no-download", action="store_true")
    args = parser.parse_args()

    normalization = cast(Literal["none", "l2"], args.normalization)
    data = scratch_root() / "data"
    output_root = args.output_root or scratch_root() / "artifacts" / (
        f"waterbirds-{normalization}"
    )
    device = args.device or auto_device()
    print(f"output:  {output_root}\ndevice:  {device}")
    prepare_server_waterbirds(
        released_root=args.released_root or data / "waterbirds",
        cub_root=args.cub_root or data / "cub",
        masks_root=args.masks_root or data / "cub-masks",
        places_root=args.places_root or data / "places",
        clip_weights_root=args.clip_weights_root or default_clip_weights_root(),
        output_root=output_root,
        construction_seed=args.construction_seed,
        normalization=normalization,
        allow_clip_download=not args.no_download,
        feature_device=device,
        clip_batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()
