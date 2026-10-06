"""The held-out validation rendering is a preparation-time choice, not a constant."""

from __future__ import annotations

from pathlib import Path

import pytest

from grit.data.cmnist import construct_cmnist
from grit.schemas import (
    CmnistSelector,
    SeedStage,
    held_out_validation_flip_prob,
    held_out_validation_name,
)
from grit.search.cmnist import materialize_cmnist_candidate_config
from grit.search.outputs import CmnistProductionSummary
from grit.search.plan import SearchPlan
from grit.search.run import plan_production_search, run_production_search
from grit.selection.cmnist import ValidationMetricRecord, select_checkpoint
from tests.test_cmnist import (
    _small_pools,  # pyright: ignore[reportPrivateUsage]
    _small_targets,  # pyright: ignore[reportPrivateUsage]
)
from tests.test_cmnist_lifecycle import fake_cache
from tests.test_search_plan import write_cmnist_production_config


def test_held_out_names_encode_approved_tenths_only() -> None:
    assert [held_out_validation_name(p) for p in (0.3, 0.4, 0.5, 0.6, 0.7)] == [
        "val_e03",
        "val_e04",
        "val_e05",
        "val_e06",
        "val_e07",
    ]
    assert held_out_validation_flip_prob("val_e04") == 0.4
    for bad in (0.45, 0.2, 0.8):
        with pytest.raises(ValueError):
            held_out_validation_name(bad)


def test_construction_renders_the_requested_held_out_rate() -> None:
    train, test = _small_pools()
    construction = construct_cmnist(
        train,
        test,
        construction_seed=11,
        targets=_small_targets(),
        held_out_flip_prob=0.4,
    )
    assert construction.val_held_out.name == "val_e04"
    assert construction.manifest.held_out_validation_name == "val_e04"
    held_out = construction.manifest.environments[4]
    assert held_out.name == "val_e04" and float(held_out.color_flip_prob) == 0.4
    views = construction.validation_views()
    assert tuple(view.descriptor.name for view in views) == (
        "val_e01",
        "val_e02",
        "val_e04",
    )
    default = construct_cmnist(
        train, test, construction_seed=11, targets=_small_targets()
    )
    assert default.val_held_out.name == "val_e05"
    assert default.manifest != construction.manifest


def _record(split: str, value: float) -> ValidationMetricRecord:
    return ValidationMetricRecord(
        record_id=f"metric:{split}",
        run_id="run:a",
        candidate_id="candidate:a",
        method_id="erm",
        scientific_config_digest="sha256:a",
        checkpoint_id="checkpoint:1",
        epoch=1,
        seed=101,
        value=value,
        sample_count=10,
        metric_kind="validation",
        seed_stage=SeedStage.TUNING,
        split_name=split,  # pyright: ignore[reportArgumentType]
        metric_name="accuracy",
        projection_rank=None,
    )


def test_primary_selector_uses_whichever_held_out_rendering_is_present() -> None:
    records = (
        _record("val_e01", 0.9),
        _record("val_e02", 0.8),
        _record("val_e04", 0.6),
    )
    decision = select_checkpoint(records, CmnistSelector.PRIMARY_ROBUST)
    assert decision.objective_value == 0.6
    assert decision.contributing_record_ids[-1] == "metric:val_e04"
    secondary = select_checkpoint(records, CmnistSelector.SECONDARY_SOURCE)
    assert secondary.objective_value == 0.8
    with pytest.raises(ValueError, match="exactly one held-out"):
        select_checkpoint(
            (*records, _record("val_e05", 0.5)), CmnistSelector.PRIMARY_ROBUST
        )


def test_plans_carry_the_held_out_split_and_default_lineage_is_unchanged(
    tmp_path: Path,
) -> None:
    default_path, _ = write_cmnist_production_config(
        tmp_path / "default", output_root=tmp_path / "default-out"
    )
    e04_path, _ = write_cmnist_production_config(
        tmp_path / "e04", output_root=tmp_path / "e04-out", held_out_flip_prob=0.4
    )
    default_plan = plan_production_search(default_path)
    e04_plan = plan_production_search(e04_path)
    assert default_plan.resolved_config.lineage.held_out_validation_split == "val_e05"
    assert "held_out" not in default_plan.resolved_config.lineage.canonical_json()
    assert e04_plan.resolved_config.lineage.held_out_validation_split == "val_e04"
    assert "val_e04" in e04_plan.resolved_config.lineage.canonical_json()
    resolved = materialize_cmnist_candidate_config(
        e04_plan, e04_plan.candidates[0], CmnistSelector.PRIMARY_ROBUST
    )
    assert resolved.dataset.validation_split_names == ("val_e01", "val_e02", "val_e04")
    assert {c.scientific_config_digest for c in e04_plan.candidates}.isdisjoint(
        {c.scientific_config_digest for c in default_plan.candidates}
    )


def test_held_out_e04_tree_runs_the_ordinary_lifecycle(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config_path, _ = write_cmnist_production_config(
        tmp_path,
        output_root=tmp_path / "output",
        overrides={
            "search_space": {
                "methods": ["erm"],
                "learning_rates": [0.01],
                "weight_decays": [0.0, 0.0001, 0.001],
            },
            "max_epochs": 2,
        },
        held_out_flip_prob=0.4,
    )
    plan = plan_production_search(config_path)
    cache = fake_cache(plan)
    assert cache.validation_tables()[2].name == "val_e04"

    def fake_cache_loader(_plan: SearchPlan, *, tuning_only: bool = False):
        return cache

    monkeypatch.setattr("grit.search.cmnist._load_cmnist_cache", fake_cache_loader)
    summary = run_production_search(config_path)
    assert isinstance(summary, CmnistProductionSummary)
    observation = summary.methods[0].final_observations[0]
    assert observation.selector is CmnistSelector.PRIMARY_ROBUST
