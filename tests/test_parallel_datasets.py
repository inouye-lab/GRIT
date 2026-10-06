"""Exercise real dataset runners with tiny caches in actual spawned workers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal, cast

import pytest

from grit.search.cmnist import build_cmnist_hooks
from grit.search.execution import ExecutionOptions
from grit.search.outputs import CmnistProductionSummary, WaterbirdsProductionSummary
from grit.search.plan import SearchPlan
from grit.search.run import (
    ProductionExecutionLimits,
    ProductionSearchStatus,
    run_production_search,
)
from grit.search.waterbirds import build_waterbirds_hooks
from tests.test_cmnist_lifecycle import fake_cache as cmnist_cache
from tests.test_search_plan import write_cmnist_production_config
from tests.test_waterbirds_paper_table import fake_cache as waterbirds_cache
from tests.test_waterbirds_paper_table import write_waterbirds_production_config


# These importable factories install the same tiny cache in parent and children.
# No production loader or pickle behavior is bypassed by a fork-only monkeypatch.
def cmnist_hooks(plan: SearchPlan, limits: ProductionExecutionLimits | None):
    cache = cmnist_cache(plan)

    def load(_plan: SearchPlan, *, tuning_only: bool = False):
        return cache

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr("grit.search.cmnist._load_cmnist_cache", load)
        return build_cmnist_hooks(plan, limits)


def waterbirds_hooks(plan: SearchPlan, limits: ProductionExecutionLimits | None):
    cache = waterbirds_cache(
        plan, Path(plan.resolved_config.config.artifacts.feature_cache_manifest)
    )

    def load(_plan: SearchPlan, *, tuning_only: bool = False):
        return cache

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr("grit.search.waterbirds._load_cache", load)
        return build_waterbirds_hooks(plan, limits)


@pytest.mark.parametrize("dataset", ["cmnist", "waterbirds"])
def test_dataset_parallel_fits_match_sequential(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    dataset: Literal["cmnist", "waterbirds"],
) -> None:
    monkeypatch.setenv("WANDB_MODE", "disabled")
    space: dict[str, object] = {
        "methods": ["erm", "grit"],
        "learning_rates": [0.001],
        "weight_decays": [0.0, 0.0001, 0.001],
        "projection_ranks": [2],
    }
    if dataset == "cmnist":
        path, _ = write_cmnist_production_config(
            tmp_path,
            output_root=tmp_path / "sequential",
            overrides={
                "search_space": space,
                "max_epochs": 1,
                "selectors": ["primary_robust", "secondary_source"],
            },
        )
        monkeypatch.setattr("grit.search.cmnist.build_cmnist_hooks", cmnist_hooks)
    else:
        path, _ = write_waterbirds_production_config(
            tmp_path,
            output_root=tmp_path / "sequential",
            search_space=space,
            selectors=("waterbirds_validation_worst_group",),
        )
        monkeypatch.setattr(
            "grit.search.waterbirds.build_waterbirds_hooks", waterbirds_hooks
        )

    first = run_production_search(path, execution=ExecutionOptions(1, 1))
    payload = json.loads(path.read_text())
    payload["output_root"] = str(tmp_path / "parallel")
    path.write_text(json.dumps(payload))
    limited = run_production_search(
        path,
        ProductionExecutionLimits(max_new_runs=2),
        execution=ExecutionOptions(2),
    )
    assert isinstance(limited, ProductionSearchStatus)
    assert limited.tuning_complete == 2 and limited.final_complete == 0
    assert not tuple((tmp_path / "parallel").rglob("*winner.json"))
    second = run_production_search(path, execution=ExecutionOptions(2, 1))
    assert isinstance(first, CmnistProductionSummary | WaterbirdsProductionSummary)
    assert isinstance(second, CmnistProductionSummary | WaterbirdsProductionSummary)
    assert without_run_ids(first.model_dump(mode="json")["methods"]) == without_run_ids(
        second.model_dump(mode="json")["methods"]
    )
    assert_matching_fits(tmp_path / "sequential", tmp_path / "parallel")


def assert_matching_fits(first_root: Path, second_root: Path) -> None:
    first_paths = tuple(first_root.rglob("result.json"))
    assert first_paths
    assert len(first_paths) == len(tuple(second_root.rglob("result.json")))
    for path in first_paths:
        other = second_root / path.relative_to(first_root)
        first, second = json.loads(path.read_text()), json.loads(other.read_text())
        for field in (
            "validation_metrics",
            "checkpoint_decisions",
            "checkpoint_decision",
            "diagnostic_metrics",
        ):
            assert without_run_ids(first.get(field)) == without_run_ids(
                second.get(field)
            )
        for weights in (path.parent / "selected-checkpoint").glob("*.npy"):
            assert (
                weights.read_bytes()
                == (other.parent / "selected-checkpoint" / weights.name).read_bytes()
            )


def without_run_ids(value: object) -> object:
    # Different output roots intentionally have different plans and run IDs.
    # Keep candidate IDs, seeds, epochs, all measurements, and selection outcomes.
    if isinstance(value, dict):
        return {
            key: without_run_ids(item)
            for key, item in cast(dict[str, object], value).items()
            if key
            not in {
                "run_id",
                "record_id",
                "metric_record_id",
                "checkpoint_id",
                "decision_id",
                "contributing_record_ids",
                "contributing_record_id",
            }
        }
    if isinstance(value, list):
        return [without_run_ids(item) for item in cast(list[object], value)]
    return value
