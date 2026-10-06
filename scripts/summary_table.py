"""Print a completed CMNIST or Waterbirds search's results from its saved summary.

uv run scripts/summary_table.py cmnist-erm-interventions-representation-v1-e03

Each argument is an output tree, either a path or a name under
$PROJECT_SCRATCH/outputs. For every method (and CMNIST selector) it prints the test
metric's mean with its 95% interval, the validation-selected settings and the value
at each final seed. It reads saved summaries only: no training, no new test access.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

_HIDDEN = {"candidate_id", "scientific_config_digest", "method_id"}


def _root(name: str) -> Path:
    path = Path(name)
    if path.is_dir():
        return path
    return Path(os.environ.get("PROJECT_SCRATCH", ".")) / "outputs" / name


def _interval(summary: dict[str, Any]) -> str:
    return (
        f"{100 * summary['mean']:.2f} "
        f"[{100 * summary['ci95_lower']:.2f}, {100 * summary['ci95_upper']:.2f}]"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("outputs", nargs="+")
    args = parser.parse_args()
    for name in args.outputs:
        root = _root(name)
        plan = json.loads((root / "search-plan.json").read_text())
        candidates = {item["candidate_id"]: item for item in plan["candidates"]}
        cmnist = (root / "summaries" / "cmnist-summary.json").exists()
        file = "cmnist-summary.json" if cmnist else "waterbirds-summary.json"
        summary = json.loads((root / "summaries" / file).read_text())
        print(f"== {root.name}")
        for method in summary["methods"]:
            chosen = candidates[method["selected_candidate_id"]]
            settings = {k: v for k, v in chosen.items() if k not in _HIDDEN}
            if cmnist:
                label = f"{method['method_id']} [{method['selector']}]"
                metrics = f"test_ood {_interval(method['accuracy_summary'])}"
                seeds = [
                    (item["seed"], item["test_ood_accuracy"])
                    for item in method["final_observations"]
                ]
            else:
                label = method["method_id"]
                metrics = (
                    f"worst_group {_interval(method['worst_group_summary'])}  "
                    f"adjusted {_interval(method['adjusted_average_summary'])}  "
                    f"raw {_interval(method['raw_average_summary'])}"
                )
                seeds = [tuple(item) for item in method["worst_group_by_seed"]]
            print(f"{label}: {metrics}")
            print(f"  selected: {json.dumps(settings)}")
            print(f"  by seed: {json.dumps(seeds)}")


if __name__ == "__main__":
    main()
