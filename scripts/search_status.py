"""Report completed search work without training.

uv run scripts/search_status.py CONFIG.yaml
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from grit.search.plan import load_production_search_config
from grit.search.run import (
    completed_task_revisions,
    is_dataset_search,
    production_search_status,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    args = parser.parse_args()
    config_path = Path(args.config)
    if is_dataset_search(config_path, "urbancars"):
        from grit.search.urbancars import urbancars_status

        print(json.dumps(urbancars_status(config_path), indent=2))
        return
    print(production_search_status(config_path).canonical_json())
    output_root = Path(load_production_search_config(config_path).output_root)
    revisions = completed_task_revisions(output_root)
    if revisions:
        summary = ", ".join(f"{rev}: {n}" for rev, n in sorted(revisions.items()))
        print(f"completed tasks by code revision: {summary}", file=sys.stderr)
        if len(revisions) > 1:
            print(
                "note: this output directory mixes code revisions; use a fresh "
                "output_root if training code changed between them",
                file=sys.stderr,
            )


if __name__ == "__main__":
    main()
