"""Waterbirds validation-selected baselines through the production lifecycle."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from grit.config import (
    FishrAlgorithmConfig,
    GroupDroAlgorithmConfig,
    IrmAlgorithmConfig,
    LisaAlgorithmConfig,
    RdmAlgorithmConfig,
    RexAlgorithmConfig,
    SdAlgorithmConfig,
)
from grit.features.waterbirds import (
    WaterbirdsFeatureCache,
    WaterbirdsFeatureCacheManifest,
)
from grit.methods.invariance import environment_balanced_epoch_batches
from grit.methods.types import WATERBIRDS_METHODS
from grit.search.outputs import WaterbirdsProductionSummary
from grit.search.plan import (
    SearchPlan,
    WaterbirdsProductionSearchConfig,
    load_production_search_config,
)
from grit.search.run import (
    plan_production_search,
    production_search_status,
    run_production_search,
)
from grit.search.scheduler import WaterbirdsCompletedStageRun
from grit.search.waterbirds import materialize_waterbirds_candidate_config
from grit.search.waterbirds_contracts import WaterbirdsRunResult
from tests.production_artifact_fixtures import (
    write_manifest_only_waterbirds_production,
)

# The standalone regression matrix. Combination lifecycles are exercised separately
# in test_objective_intervention_lifecycle.py.
STANDALONE_METHODS = tuple(method for method in WATERBIRDS_METHODS if "_" not in method)

FULL_GRID: dict[str, object] = {
    "methods": list(STANDALONE_METHODS),
    "learning_rates": [0.01],
    "weight_decays": [0.0, 0.0001, 0.001],
    "projection_ranks": [2],
    "groupdro_step_sizes": [0.01],
    "rex_penalty_weights": [10.0],
    "rex_penalty_anneal_updates": 100,
    "irm_penalty_weights": [100.0],
    "irm_penalty_anneal_updates": 190,
    "lisa_selection_probs": [0.5],
    "sd_penalty_weights": [0.1],
    "fishr_penalty_weights": [100.0],
    # These checks train for one or two epochs, so the warm-up is shortened
    # explicitly; production keeps the protocol's 1,500 updates.
    "fishr_penalty_anneal_updates": 2,
    "fishr_ema": 0.95,
    "rdm_penalty_weights": [1.0],
    "rdm_penalty_anneal_updates": 2,
    "rdm_variance_weight": 0.004,
}


def write_waterbirds_production_config(
    root: Path,
    *,
    output_root: Path,
    selectors: tuple[str, ...] = ("waterbirds_validation_worst_group",),
    search_space: dict[str, object] | None = None,
    max_epochs: int = 1,
) -> tuple[Path, tuple[Path, Path, Path]]:
    artifact_paths = write_manifest_only_waterbirds_production(root / "prepared")
    payload: dict[str, object] = {
        "schema_version": "grit.production-search/v1",
        "dataset": "waterbirds_cf",
        "protocol_id": "waterbirds_cf/v1",
        "experiment_name": "waterbirds-paper-table-check",
        "experiment_variant": "primary_unnormalized",
        "normalization": "none",
        "artifacts": {
            "dataset_manifest": artifact_paths[0].as_posix(),
            "feature_cache_manifest": artifact_paths[1].as_posix(),
            "oracle_pair_manifest": artifact_paths[2].as_posix(),
        },
        "seeds": {
            "construction": 17,
            "pairs": 2718,
            "stages": {
                "tuning": [1101, 1102, 1103],
                "confirmation": [2101, 2102],
                "final": [3101, 3102, 3103, 3104, 3105, 3106, 3107, 3108, 3109, 3110],
            },
        },
        "search_space": search_space or dict(FULL_GRID),
        "selectors": list(selectors),
        "pair_count": 240,
        "batch_size": 256,
        "max_epochs": max_epochs,
        "output_root": output_root.as_posix(),
        "runtime": {"device": "cpu", "deterministic_algorithms": True, "workers": 1},
        "relative_singular_value_tolerance": 1e-12,
    }
    config_path = root / "production.yaml"
    config_path.write_text(json.dumps(payload), encoding="utf-8")
    return config_path, artifact_paths


def fake_cache(plan: SearchPlan, feature_manifest_path: Path) -> WaterbirdsFeatureCache:
    """The real production manifest over random features with a weak label signal."""

    manifest = WaterbirdsFeatureCacheManifest.model_validate_json(
        feature_manifest_path.read_text(encoding="utf-8")
    )
    assert manifest.canonical_digest() == (
        plan.resolved_config.lineage.feature_cache_manifest_digest
    )
    generator = torch.Generator().manual_seed(0)
    features = torch.randn((len(manifest.records), 512), generator=generator)
    labels = torch.tensor([record.bird_label for record in manifest.records])
    backgrounds = torch.tensor([record.background for record in manifest.records])
    features[:, 0] += 2.0 * labels.to(torch.float32)
    features[:, 1] += 3.0 * backgrounds.to(torch.float32)
    return WaterbirdsFeatureCache(
        manifest=manifest, features=features, root=feature_manifest_path.parent
    )


def test_unequal_environments_are_balanced_and_exhaust_the_largest_once() -> None:
    ids = torch.tensor([0] * 3554 + [1] * 1241)
    batches = environment_balanced_epoch_batches(
        ids,
        environment_count=2,
        batch_size=256,
        generator=torch.Generator().manual_seed(1),
    )
    assert len(batches) == 28
    assert [len(batch) for batch in batches][-2:] == [256, 196]
    for batch in batches:
        assert int((ids[batch] == 0).sum()) == int((ids[batch] == 1).sum())
    land_rows = torch.cat([batch[ids[batch] == 0] for batch in batches])
    assert torch.equal(land_rows.sort().values, torch.arange(3554))
    equal = torch.tensor([0] * 100 + [1] * 100)
    reference = torch.Generator().manual_seed(3)
    shuffled = tuple(
        torch.nonzero(equal == e).flatten()[torch.randperm(100, generator=reference)]
        for e in (0, 1)
    )
    expected = tuple(
        torch.cat([rows[i : i + 32] for rows in shuffled]) for i in range(0, 100, 32)
    )
    observed = environment_balanced_epoch_batches(
        equal,
        environment_count=2,
        batch_size=64,
        generator=torch.Generator().manual_seed(3),
    )
    assert all(torch.equal(a, b) for a, b in zip(observed, expected, strict=True))


def test_plan_binds_every_method_to_waterbirds_environments_and_groups(
    tmp_path: Path,
) -> None:
    config_path, _ = write_waterbirds_production_config(
        tmp_path, output_root=tmp_path / "output"
    )
    plan = plan_production_search(config_path)
    assert plan.methods == STANDALONE_METHODS
    assert len(plan.candidates) == 3 * len(STANDALONE_METHODS)
    for candidate in plan.candidates:
        if candidate.method_id == "grit":
            continue
        resolved = materialize_waterbirds_candidate_config(plan, candidate, None)
        assert resolved.selector == "waterbirds_validation_worst_group"
        assert resolved.algorithm.kind == candidate.method_id
        assert resolved.pair_manifest_digest is None
        algorithm = resolved.algorithm
        if isinstance(
            algorithm,
            RexAlgorithmConfig
            | IrmAlgorithmConfig
            | FishrAlgorithmConfig
            | RdmAlgorithmConfig,
        ):
            assert algorithm.environment_names == (
                "background_land",
                "background_water",
            )
        if isinstance(algorithm, SdAlgorithmConfig):
            assert algorithm.sampling == "uniform_without_replacement"
        if isinstance(algorithm, GroupDroAlgorithmConfig | LisaAlgorithmConfig):
            assert algorithm.group_definition == "target_background"


def test_validation_methods_run_through_the_lifecycle(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output_root = tmp_path / "output"
    config_path, artifact_paths = write_waterbirds_production_config(
        tmp_path, output_root=output_root, search_space=dict(FULL_GRID)
    )
    plan = plan_production_search(config_path)
    cache = fake_cache(plan, artifact_paths[1])

    def fake_cache_loader(
        _plan: SearchPlan, *, tuning_only: bool = False
    ) -> WaterbirdsFeatureCache:
        return cache

    monkeypatch.setattr("grit.search.waterbirds._load_cache", fake_cache_loader)
    summary = run_production_search(config_path)
    assert isinstance(summary, WaterbirdsProductionSummary)
    assert summary.selector == "waterbirds_validation_worst_group"
    assert tuple(item.method_id for item in summary.methods) == STANDALONE_METHODS
    assert summary.paired, "ERM and GRIT are both present"
    assert production_search_status(config_path).phase == "complete"

    stage_runs = [
        WaterbirdsCompletedStageRun.model_validate_json(
            path.read_text(encoding="utf-8")
        )
        for path in sorted(output_root.rglob("result.json"))
    ]
    assert len(stage_runs) == (
        plan.expected_run_counts.tuning
        + plan.expected_run_counts.confirmation_minimum
        + plan.expected_run_counts.final
    )
    final_texts = [
        path.read_text(encoding="utf-8")
        for path in sorted(output_root.rglob("final-result.json"))
    ]
    assert len(final_texts) == plan.expected_run_counts.final
    for text in final_texts:
        payload = json.loads(text)
        result = WaterbirdsRunResult.model_validate_json(text)
        assert "diagnostic_metrics" not in payload
        assert result.candidate_selection.selector == (
            "waterbirds_validation_worst_group"
        )
    selection_files = sorted(
        path.name for path in (output_root / "selection" / "grit").iterdir()
    )
    assert selection_files == ["tuning-finalists.json", "winner.json"]
    assert (output_root / "summaries" / "waterbirds-paired-differences.json").is_file()


def test_checked_waterbirds_configs_cover_every_method(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("PROJECT_SCRATCH", "/scratch")
    root = Path(__file__).resolve().parents[1] / "configs" / "waterbirds"
    seen: set[str] = set()
    for path in sorted(root.glob("*-search*.yaml")):
        config = load_production_search_config(path)
        assert isinstance(config, WaterbirdsProductionSearchConfig)
        assert config.artifacts.dataset_manifest.startswith(
            "/scratch/artifacts/waterbirds-none/"
        )
        assert config.selectors == ("waterbirds_validation_worst_group",)
        assert not seen.intersection(config.search_space.methods)
        seen.update(config.search_space.methods)
    assert seen == set(WATERBIRDS_METHODS)
