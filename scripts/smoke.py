"""Hermetic end-to-end check with a fake encoder. Never report its numbers.

    uv run scripts/smoke.py cmnist
    uv run scripts/smoke.py waterbirds
"""

from __future__ import annotations

import argparse
from pathlib import Path

from grit.paths import REPO_ROOT


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", choices=("cmnist", "waterbirds"))
    parser.add_argument("config", type=Path, nargs="?")
    args = parser.parse_args()
    config = args.config or REPO_ROOT / "configs" / args.dataset / "smoke.yaml"
    if args.dataset == "cmnist":
        from grit.search.cmnist_runner import load_cmnist_smoke_config, run_cmnist_smoke

        print(run_cmnist_smoke(load_cmnist_smoke_config(config)).canonical_json())
        return
    from grit.search.waterbirds_runner import (
        load_waterbirds_smoke_config,
        run_waterbirds_smoke,
    )

    print(run_waterbirds_smoke(load_waterbirds_smoke_config(config)).canonical_json())


if __name__ == "__main__":
    main()
