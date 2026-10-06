"""A full paired CMNIST production lifecycle on a tiny in-memory feature cache."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal, cast

import pytest
import torch

from grit.features.cmnist import CmnistFeatureCache, FeatureTable
from grit.results import OrdinaryRunResult
from grit.schemas import CmnistSelector
from grit.search.outputs import CmnistProductionSummary
from grit.search.plan import SearchPlan
from grit.search.run import (
    plan_production_search,
    production_search_status,
    run_production_search,
)
from grit.search.scheduler import CmnistCompletedStageRun
from tests.test_search_plan import write_cmnist_production_config


class _ManifestIdentity:
    def __init__(self, digest: str) -> None:
        self._digest = digest

    def canonical_digest(self) -> str:
        return self._digest


def _table(name: str, role: str, seed: int, rows: int = 8) -> FeatureTable:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    features = torch.randn((rows, 512), generator=generator)
    targets = (features[:, 0] > 0).to(torch.int64)
    return FeatureTable(
        name=name,
        role=cast(Literal["training"], role),
        source_ids=tuple(f"source:{name}:{index}" for index in range(rows)),
        features=features,
        digits=targets * 5,
        clean_labels=targets,
        targets=targets,
        colors=torch.arange(rows) % 2,
    )


def fake_cache(plan: SearchPlan) -> CmnistFeatureCache:
    manifest = _ManifestIdentity(
        plan.resolved_config.lineage.feature_cache_manifest_digest
    )
    held_out = plan.resolved_config.lineage.held_out_validation_split
    pair_red = _table("oracle_pair_red", "pair_projection", 7)
    pair_green = FeatureTable(
        name="oracle_pair_green",
        role="pair_projection",
        source_ids=pair_red.source_ids,
        features=pair_red.features + 1.0,
        digits=pair_red.digits,
        clean_labels=pair_red.clean_labels,
        targets=pair_red.targets,
        colors=1 - pair_red.colors,
    )
    return CmnistFeatureCache(
        train_e01=_table("train_e01", "training", 1),
        train_e02=_table("train_e02", "training", 2),
        val_e01=_table("val_e01", "validation", 3),
        val_e02=_table("val_e02", "validation", 4),
        val_held_out=_table(held_out, "validation", 5),
        _test_ood=_table("test_ood", "final_test", 6),
        oracle_pair_red=pair_red,
        oracle_pair_green=pair_green,
        manifest=cast(object, manifest),  # pyright: ignore[reportArgumentType]
        root=Path("."),
    )


def test_ordinary_track_runs_the_full_paired_lifecycle(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    selectors = ("primary_robust", "secondary_source")
    output_root = tmp_path / "output"
    config_path, _ = write_cmnist_production_config(
        tmp_path,
        output_root=output_root,
        overrides={
            "experiment_name": "cmnist-track-check",
            "search_space": {
                "methods": ["erm", "grit"],
                "learning_rates": [0.01],
                "weight_decays": [0.0, 0.0001, 0.001],
                "projection_ranks": [1, 2, 3],
            },
            "selectors": list(selectors),
            "max_epochs": 2,
        },
    )
    plan = plan_production_search(config_path)
    assert plan.selectors == selectors
    assert "finalist_union" in {s.artifact_kind for s in plan.output_schemas}

    cache = fake_cache(plan)

    def fake_cache_loader(
        _plan: SearchPlan, *, tuning_only: bool = False
    ) -> CmnistFeatureCache:
        return cache

    monkeypatch.setattr("grit.search.cmnist._load_cmnist_cache", fake_cache_loader)
    summary = run_production_search(config_path)
    assert isinstance(summary, CmnistProductionSummary)
    assert summary.selectors == tuple(CmnistSelector(s) for s in selectors)
    assert [(m.method_id, m.selector.value) for m in summary.methods] == [
        (method, selector) for method in ("erm", "grit") for selector in selectors
    ]
    assert [p.selector.value for p in summary.paired_selectors] == list(selectors)
    assert production_search_status(config_path).phase == "complete"

    stage_runs = [
        CmnistCompletedStageRun.model_validate_json(path.read_text(encoding="utf-8"))
        for path in sorted(output_root.rglob("result.json"))
    ]
    assert (
        len(stage_runs)
        == plan.expected_run_counts.tuning
        + len(tuple(output_root.glob("runs/confirmation/*/*/*")))
        + plan.expected_run_counts.final
    )
    final_results = [
        path.read_text(encoding="utf-8")
        for path in sorted(output_root.rglob("final-result.json"))
    ]
    assert len(final_results) == plan.expected_run_counts.final
    for text in final_results:
        payload = json.loads(text)
        result = OrdinaryRunResult.model_validate_json(text)
        assert "diagnostic_metrics" not in payload
        assert result.candidate_selection is not None
    selection_files = sorted(
        path.name for path in (output_root / "selection" / "erm").iterdir()
    )
    assert selection_files == [
        "confirmation-union.json",
        "primary-tuning-finalists.json",
        "primary_robust-winner.json",
        "secondary-tuning-finalists.json",
        "secondary_source-winner.json",
    ]
