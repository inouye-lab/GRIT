"""Build ColoredMNIST: partitions, 256 oracle pairs, and the CLIP feature cache.

    uv run scripts/prepare_cmnist.py

Downloads MNIST and the pinned CLIP weights if missing. Paths default under
$PROJECT_SCRATCH.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Literal, cast

from grit.paths import (
    DEFAULT_CONSTRUCTION_SEED,
    DEFAULT_PAIR_SEED,
    auto_device,
    default_clip_weights_root,
    scratch_root,
)
from grit.schemas import held_out_validation_name
from grit.search.cmnist_runner import prepare_official_cmnist


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, help="raw MNIST")
    parser.add_argument("--clip-weights-root", type=Path)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument(
        "--construction-seed", type=int, default=DEFAULT_CONSTRUCTION_SEED
    )
    parser.add_argument("--pair-seed", type=int, default=DEFAULT_PAIR_SEED)
    parser.add_argument(
        "--pair-count",
        type=int,
        default=256,
        help="oracle pairs to bank; a config may use any prefix of this bank",
    )
    parser.add_argument("--normalization", choices=("none", "l2"), default="none")
    parser.add_argument(
        "--held-out-flip-prob",
        type=float,
        default=0.5,
        help="color flip rate of the third validation rendering (0.3 to 0.7); "
        "0.5 is the protocol default, others write to a separate artifact root",
    )
    parser.add_argument("--device", help="cpu, cuda, or cuda:N (default: auto)")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--no-download", action="store_true")
    args = parser.parse_args()

    normalization = cast(Literal["none", "l2"], args.normalization)
    held_out = held_out_validation_name(args.held_out_flip_prob)
    data_root = args.data_root or scratch_root() / "data" / "mnist"
    clip_root = args.clip_weights_root or default_clip_weights_root()
    suffix = "" if held_out == "val_e05" else f"-{held_out.removeprefix('val_')}"
    output_root = args.output_root or scratch_root() / "artifacts" / (
        f"cmnist-{normalization}{suffix}"
    )
    device = args.device or auto_device()
    print(f"mnist:   {data_root}\nclip:    {clip_root}")
    print(f"output:  {output_root}\ndevice:  {device}")
    prepare_official_cmnist(
        data_root=data_root,
        clip_weights_root=clip_root,
        output_root=output_root,
        construction_seed=args.construction_seed,
        pair_seed=args.pair_seed,
        normalization=normalization,
        allow_download=not args.no_download,
        feature_device=device,
        clip_batch_size=args.batch_size,
        pair_count=args.pair_count,
        held_out_flip_prob=args.held_out_flip_prob,
    )


if __name__ == "__main__":
    main()
