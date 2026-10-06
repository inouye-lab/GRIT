"""Plan and run (or continue) a production search from a YAML config.

    uv run scripts/run_search.py CONFIG.yaml --pilot
    uv run scripts/run_search.py CONFIG.yaml

Completed tasks in the output directory are reused, so interrupting and rerunning
is safe. --pilot runs one tuning task for each configured method and stops.
UrbanCars configs fit on validation only and print the frozen winners' hash;
--unseal-test <hash> then evaluates test.

Setting WANDB_PROJECT enables supported W&B mirrors; see the README.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

from grit.methods.types import IMPLEMENTED_METHODS
from grit.search.execution import ExecutionOptions
from grit.search.run import (
    ProductionExecutionLimits,
    ProductionSearchStatus,
    is_dataset_search,
    pilot_candidates,
    plan_production_search,
    production_search_status,
    run_production_search,
)
from grit.tracking import tracking_settings_from_environment


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument(
        "--dry-run", action="store_true", help="write the plan and print status only"
    )
    parser.add_argument("--pilot", action="store_true")
    parser.add_argument("--limit", type=int, help="run at most N new tasks, then stop")
    parser.add_argument("--stop-after", choices=("tuning",))
    parser.add_argument(
        "--only",
        choices=IMPLEMENTED_METHODS,
        help="tuning-only filter",
    )
    parser.add_argument("--candidate-id", action="append", default=[])
    parser.add_argument("--seed", type=int, help="tuning-only seed filter")
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="concurrent CPU fits (default: 1, direct sequential execution)",
    )
    parser.add_argument(
        "--threads-per-worker",
        type=int,
        help="CPU threads per fit (default: unchanged sequentially, 1 in parallel)",
    )
    parser.add_argument(
        "--unseal-test",
        metavar="WINNERS_HASH",
        help="UrbanCars: evaluate test for the frozen winners' hash",
    )
    args = parser.parse_args(argv)
    try:
        execution = ExecutionOptions(args.workers, args.threads_per_worker)
    except ValueError as error:
        parser.error(str(error))

    config = Path(args.config)
    if is_dataset_search(config, "urbancars"):
        from grit.search.urbancars import run_urbancars

        result = run_urbancars(
            config, execution=execution, unseal_test=args.unseal_test
        )
        print(json.dumps(result, indent=2))
        return 0
    if args.unseal_test is not None:
        parser.error("--unseal-test is only for UrbanCars configs")
    plan = plan_production_search(config)
    print(
        f"{plan.dataset}: {len(plan.candidates)} candidates, "
        f"output {plan.resolved_config.output_root}",
        file=sys.stderr,
    )
    tracking = tracking_settings_from_environment()
    if tracking.enabled:
        print(
            f"tracking: W&B project {tracking.project} ({tracking.mode}), "
            f"files under {tracking.directory}",
            file=sys.stderr,
        )
    if args.dry_run:
        print(production_search_status(config).canonical_json())
        return 0

    candidate_ids = tuple(args.candidate_id)
    seed = args.seed
    stop_after = args.stop_after
    limit = args.limit
    if args.pilot:
        candidates = pilot_candidates(plan)
        candidate_ids = tuple(candidate.candidate_id for candidate in candidates)
        seed = plan.seeds.stages.tuning[0]
        stop_after = "tuning"
        limit = len(candidates)
    filtered = args.only is not None or bool(candidate_ids) or seed is not None
    if filtered:
        stop_after = "tuning"
    limits = (
        ProductionExecutionLimits(
            stop_after=stop_after,
            method=args.only or "all",
            candidate_ids=candidate_ids,
            tuning_seed=seed,
            max_new_runs=limit,
        )
        if filtered or stop_after is not None or limit is not None
        else None
    )
    result = (
        run_production_search(config, limits)
        if execution == ExecutionOptions()
        else run_production_search(config, limits, execution=execution)
    )
    if isinstance(result, ProductionSearchStatus):
        print(result.canonical_json())
    else:
        print(f"completed {result.schema_version} for plan {result.plan_digest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
