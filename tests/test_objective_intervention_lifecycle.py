"""Carry each objective's three variants through the actual production lifecycle."""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import replace
from pathlib import Path
from typing import Literal, TypeAlias, cast

import pytest
import torch
from pydantic import ValidationError

from grit.config import (
    ComposedAlgorithmConfig,
    GroupDroAlgorithmConfig,
    LisaAlgorithmConfig,
    algorithm_method_id,
)
from grit.features.cmnist import CmnistFeatureCache
from grit.features.waterbirds import WaterbirdsFeatureCache
from grit.methods.training import LinearProbeAlgorithm, PersistedLinearCheckpointStore
from grit.methods.types import MethodId, consumes_pairs, pair_intervention
from grit.results import CodeProvenance, OrdinaryRunResult
from grit.search.outputs import CmnistProductionSummary, WaterbirdsProductionSummary
from grit.search.plan import SearchPlan
from grit.search.run import (
    ProductionExecutionLimits,
    ProductionSearchStatus,
    plan_production_search,
    production_search_status,
    run_production_search,
)
from grit.search.waterbirds_contracts import WaterbirdsRunResult
from tests.test_cmnist_lifecycle import fake_cache
from tests.test_search_plan import write_cmnist_production_config
from tests.test_waterbirds_paper_table import fake_cache as waterbirds_fake_cache
from tests.test_waterbirds_paper_table import write_waterbirds_production_config

Objective = Literal["erm", "rex", "irm", "fishr", "groupdro", "sd", "rdm", "lisa"]
Dataset = Literal["cmnist", "waterbirds"]
ROWS: dict[Objective, tuple[MethodId, MethodId, MethodId]] = {
    "erm": ("erm", "grit", "erm_consistency"),
    "rex": ("rex", "rex_grit", "rex_consistency"),
    "irm": ("irm", "irm_grit", "irm_consistency"),
    "fishr": ("fishr", "fishr_grit", "fishr_consistency"),
    "groupdro": ("groupdro", "groupdro_grit", "groupdro_consistency"),
    "sd": ("sd", "sd_grit", "sd_consistency"),
    "rdm": ("rdm", "rdm_grit", "rdm_consistency"),
    "lisa": ("lisa", "lisa_grit", "lisa_consistency"),
}
# The second field replaces GroupDRO's and LISA's label x attribute groups with source
# environments, alone or crossed with the label.
GroupOverride: TypeAlias = Literal["training_environment", "target_environment"] | None
MATRIX_ROWS: tuple[tuple[Objective, GroupOverride], ...] = (
    *((objective, None) for objective in ROWS),
    ("groupdro", "training_environment"),
    ("groupdro", "target_environment"),
    ("lisa", "target_environment"),
)
GROUP_OBJECTIVES = ("groupdro", "lisa")
ENVIRONMENT_OBJECTIVES = ("rex", "irm", "fishr", "rdm")


@pytest.fixture(autouse=True)
def _fixed_code_provenance(  # pyright: ignore[reportUnusedFunction]
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    code = CodeProvenance(git_revision="1" * 40, git_dirty=True)
    monkeypatch.setattr("grit.search.plan._code_provenance", lambda: code)


def _space(objective: Objective) -> dict[str, object]:
    space: dict[str, object] = {
        "methods": list(ROWS[objective]),
        "learning_rates": [0.01],
        "weight_decays": [0.0, 0.0001, 0.001],
        "projection_ranks": [1],
        "consistency_weights": [0.1],
    }
    if objective in ("rex", "irm", "fishr"):
        space[f"{objective}_penalty_weights"] = [10.0]
        # Exercise activation and the Fishr optimizer reset in this tiny run.
        space[f"{objective}_penalty_anneal_updates"] = 2
    if objective == "fishr":
        space["fishr_ema"] = 0.95
    if objective == "groupdro":
        space["groupdro_step_sizes"] = [0.1]
    if objective == "sd":
        space["sd_penalty_weights"] = [0.1]
    if objective == "rdm":
        space.update(
            rdm_penalty_weights=[1.0],
            # Crosses the RDM activation and its optimizer reset in this tiny run.
            rdm_penalty_anneal_updates=2,
            rdm_variance_weight=0.004,
        )
    if objective == "lisa":
        space["lisa_selection_probs"] = [0.5]
    return space


def _forbid_access(*_args: object, **_kwargs: object) -> None:
    raise AssertionError("this phase/objective must not access this information")


@pytest.mark.parametrize("dataset", ["cmnist", "waterbirds"])
@pytest.mark.parametrize(("objective", "group_definition"), MATRIX_ROWS)
def test_matrix_row_runs_selection_restoration_and_reporting(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    dataset: Dataset,
    objective: Objective,
    group_definition: GroupOverride,
) -> None:
    if group_definition == "target_environment" and dataset == "waterbirds":
        pytest.skip("Waterbirds label x environment groups are label x background")
    output_root = tmp_path / "output"
    space = _space(objective)
    uses_groups = objective in GROUP_OBJECTIVES
    uses_environments = objective in ENVIRONMENT_OBJECTIVES
    if group_definition is not None:
        space[f"{objective}_group_definition"] = group_definition
        # Waterbirds GroupDRO then reads backgrounds, not label/background groups.
        uses_groups = dataset == "cmnist" or objective == "lisa"
        uses_environments = uses_environments or dataset == "waterbirds"
    methods: tuple[MethodId, ...] = tuple(method for method in ROWS[objective])
    if dataset == "cmnist":
        config_path, _ = write_cmnist_production_config(
            tmp_path,
            output_root=output_root,
            overrides={
                "search_space": space,
                "batch_size": 8,
                "max_epochs": 2,
            },
        )
        plan = plan_production_search(config_path)
        cache = fake_cache(plan)

        def cmnist_loader(_plan: SearchPlan, *, tuning_only: bool = False):
            return cache

        monkeypatch.setattr("grit.search.cmnist._load_cmnist_cache", cmnist_loader)
    else:
        config_path, artifacts = write_waterbirds_production_config(
            tmp_path, output_root=output_root, search_space=space, max_epochs=1
        )
        plan = plan_production_search(config_path)
        cache = waterbirds_fake_cache(plan, artifacts[1])

        def waterbirds_loader(_plan: SearchPlan, *, tuning_only: bool = False):
            return cache

        monkeypatch.setattr("grit.search.waterbirds._load_cache", waterbirds_loader)
        # Interventions grant no metadata beyond what the base objective uses.
        if not uses_groups:
            monkeypatch.setattr(
                WaterbirdsFeatureCache, "training_group_ids", _forbid_access
            )
        if not uses_environments:
            monkeypatch.setattr(
                WaterbirdsFeatureCache, "training_environment_ids", _forbid_access
            )

    assert Counter(candidate.method_id for candidate in plan.candidates) == {
        method: 3 for method in methods
    }
    with monkeypatch.context() as guarded:
        cache_type = type(cache)
        guarded.setattr(cache_type, "issue_final_handle", _forbid_access)
        status = run_production_search(
            config_path, ProductionExecutionLimits(stop_after="tuning")
        )
    assert isinstance(status, ProductionSearchStatus)
    assert status.tuning_complete == status.tuning_expected == 9 * len(methods)
    assert status.final_complete == 0
    assert not tuple(output_root.rglob("final-result.json"))

    summary = run_production_search(config_path)
    assert isinstance(summary, CmnistProductionSummary | WaterbirdsProductionSummary)
    selector_count = 2 if dataset == "cmnist" else 1
    assert Counter(item.method_id for item in summary.methods) == {
        method: selector_count for method in methods
    }
    expected_contrasts = {
        (methods[1], methods[0]),
        (methods[2], methods[0]),
        (methods[1], methods[2]),
    }
    diagnostic_paths = tuple(output_root.rglob("representation-diagnostics.json"))
    assert diagnostic_paths
    for diagnostic_path in diagnostic_paths:
        assert "_consistency" in str(diagnostic_path)
        history = json.loads(diagnostic_path.read_text())
        assert history
        for values in history.values():
            # Prediction-consistency rows log their unweighted pair penalty only.
            assert set(values) == {"pairs/prediction_consistency"}
    comparisons = summary.intervention_comparisons
    assert {(c.minuend, c.subtrahend) for c in comparisons} == expected_contrasts
    assert len(comparisons) == len(expected_contrasts) * (
        2 if dataset == "cmnist" else 3
    )
    for comparison in comparisons:
        assert comparison.base_objective == objective
        assert tuple(seed for seed, _ in comparison.differences_by_seed) == (
            plan.seeds.stages.final
        )
        assert comparison.ci95_lower <= comparison.mean <= comparison.ci95_upper
    status = production_search_status(config_path)
    assert status.phase == "complete"
    assert status.frozen_winner_count == len(methods) * selector_count
    assert status.final_complete == len(methods) * selector_count * 10
    for method in methods:
        assert len(
            tuple((output_root / "selection" / method).glob("*winner.json"))
        ) == (selector_count)

    # Reload one predictor per combination and compare its predictions to the
    # saved final metric, with no separately supplied projection object.
    checked: set[MethodId] = set()
    pair_digests: set[str] = set()
    for path in sorted(output_root.rglob("final-result.json")):
        result = (
            OrdinaryRunResult.model_validate_json(path.read_text())
            if dataset == "cmnist"
            else WaterbirdsRunResult.model_validate_json(path.read_text())
        )
        resolved = result.resolved_config
        method = algorithm_method_id(resolved.algorithm)
        if method in checked:
            continue
        checked.add(method)
        assert result.restoration is not None
        assert result.checkpoint_selection is not None
        assert result.restoration.checkpoint == result.checkpoint_selection.checkpoint
        if isinstance(result, OrdinaryRunResult):
            lineage = result.resolved_config.artifact_lineage
            assert lineage is not None
            pair_digest = lineage.pair_manifest_digest
            assert isinstance(cache, CmnistFeatureCache)
            features = cache._test_ood.features  # pyright: ignore[reportPrivateUsage]
            targets = cache._test_ood.targets  # pyright: ignore[reportPrivateUsage]
            assert result.final_test_metrics is not None
            saved_accuracy = float(result.final_test_metrics[0].value)
        else:
            pair_digest = result.resolved_config.pair_manifest_digest
            assert isinstance(cache, WaterbirdsFeatureCache)
            records = [
                r for r in cache.manifest.records if r.split_role == "final_test"
            ]
            features = cache.features[torch.tensor([r.row_index for r in records])]
            targets = torch.tensor([r.bird_label for r in records])
            saved_accuracy = float(result.final_test_metric.raw_average_accuracy)
        assert (pair_digest is not None) == consumes_pairs(method)
        if pair_digest is not None:
            pair_digests.add(pair_digest)
        store = PersistedLinearCheckpointStore(path.parent / "selected-checkpoint")
        stored = store.load(store.store_id.removeprefix("linear-checkpoint:"))
        basis = stored.state.projection_basis
        assert basis is not None
        assert basis.shape == (
            512,
            1 if pair_intervention(method) == "grit" else 0,
        )
        restored = LinearProbeAlgorithm(
            resolved.training, model_seed=0, projection=None
        )
        restored.restore_inference_state(stored.state)
        predictions = restored.predict(features)
        assert (
            float((predictions == targets).to(torch.float64).mean()) == saved_accuracy
        )
        if pair_intervention(method) == "grit":
            assert torch.equal(
                predictions, restored.predict(features + 10.0 * basis[:, 0])
            ), "restored predictor must remain invariant to the removed direction"
        if isinstance(resolved.algorithm, ComposedAlgorithmConfig) and (
            objective in ENVIRONMENT_OBJECTIVES
        ):
            payload = resolved.model_dump()
            payload["algorithm"]["base_objective"]["environment_names"] = (
                ("background_land", "background_water")
                if dataset == "cmnist"
                else ("train_e01", "train_e02")
            )
            with pytest.raises(ValidationError, match="invariant methods"):
                type(resolved).model_validate(payload)
        base = (
            resolved.algorithm.base_objective
            if isinstance(resolved.algorithm, ComposedAlgorithmConfig)
            else resolved.algorithm
        )
        if isinstance(base, GroupDroAlgorithmConfig | LisaAlgorithmConfig):
            attribute = "target_color" if dataset == "cmnist" else "target_background"
            assert base.group_definition == (group_definition or attribute)
        if isinstance(resolved.algorithm, ComposedAlgorithmConfig) and (
            objective in GROUP_OBJECTIVES
        ):
            payload = resolved.model_dump()
            payload["algorithm"]["base_objective"]["group_definition"] = (
                "target_background" if dataset == "cmnist" else "target_color"
            )
            with pytest.raises(ValidationError, match="GroupDRO uses|LISA uses"):
                type(resolved).model_validate(payload)
    assert checked == set(methods)
    assert len(pair_digests) == 1, "both interventions use the same bank"


@pytest.mark.parametrize("objective", ["rex", "irm", "fishr"])
def test_matrix_candidates_jointly_search_objective_and_intervention(
    tmp_path: Path, objective: Objective
) -> None:
    space = _space(objective)
    space[f"{objective}_penalty_weights"] = [1.0, 10.0]
    space["projection_ranks"] = [1, 2]
    space["consistency_weights"] = [0.1, 1.0, 10.0]
    config_path, _ = write_cmnist_production_config(
        tmp_path,
        output_root=tmp_path / "output",
        overrides={"search_space": space},
    )
    plan = plan_production_search(config_path)
    vanilla, projected, consistency = ROWS[objective]
    assert Counter(candidate.method_id for candidate in plan.candidates) == {
        vanilla: 3 * 2,
        projected: 3 * 2 * 2,
        consistency: 3 * 2 * 3,
    }
    assert {
        (candidate.penalty_weight, candidate.requested_rank)
        for candidate in plan.candidates
        if candidate.method_id == projected
    } == {(weight, rank) for weight in (1.0, 10.0) for rank in (1, 2)}
    assert {
        (candidate.penalty_weight, candidate.consistency_weight)
        for candidate in plan.candidates
        if candidate.method_id == consistency
    } == {(weight, strength) for weight in (1.0, 10.0) for strength in (0.1, 1.0, 10.0)}


# The shipped matrix configurations: each base objective's own grid, crossed with
# vanilla, GRIT and prediction consistency on the same optimizer grid.
MATRIX_GRIDS: dict[Objective, tuple[str | None, tuple[float, ...], int]] = {
    "erm": (None, (), 144),
    "rex": ("rex_penalty_weights", (10.0, 100.0, 1000.0, 10000.0), 576),
    "irm": ("irm_penalty_weights", (100.0, 1000.0, 10000.0, 100000.0), 576),
    "fishr": ("fishr_penalty_weights", (10.0, 100.0, 1000.0, 10000.0), 576),
    "groupdro": ("groupdro_step_sizes", (1e-3, 1e-2, 1e-1), 432),
    "sd": ("sd_penalty_weights", (1e-5, 1e-4, 1e-3, 1e-2, 1e-1), 720),
    "rdm": ("rdm_penalty_weights", (0.1, 1.0, 5.0, 10.0), 576),
    "lisa": ("lisa_selection_probs", (0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0), 1008),
}
# CMNIST's GroupDRO and LISA rows use source environments (crossed with the label);
# Waterbirds' use the default label x background groups.
CONFIG_VARIANTS: dict[Dataset, dict[Objective, str]] = {
    "cmnist": {"groupdro": "-label-source-env", "lisa": "-source-env"},
    "waterbirds": {},
}


@pytest.mark.parametrize("dataset", ["cmnist", "waterbirds"])
def test_matrix_configs_search_the_protocol_grids(
    monkeypatch: pytest.MonkeyPatch, dataset: Dataset
) -> None:
    from grit.search.plan import (
        _method_settings_grid,  # pyright: ignore[reportPrivateUsage]
        load_production_search_config,
    )

    monkeypatch.setenv("PROJECT_SCRATCH", "/scratch")
    configs = Path(__file__).resolve().parents[1] / "configs"
    root = configs / ("cmnist/held-out-e03" if dataset == "cmnist" else "waterbirds")
    output_roots: set[str] = set()
    for base, (field, values, candidate_count) in MATRIX_GRIDS.items():
        variant = CONFIG_VARIANTS[dataset].get(base, "")
        config = load_production_search_config(
            root / f"{base}{variant}-representation-interventions-search.yaml"
        )
        source_groups = "target_environment" if variant else None
        assert config.search_space.groupdro_group_definition == (
            source_groups if base == "groupdro" else None
        )
        assert config.search_space.lisa_group_definition == (
            source_groups if base == "lisa" else None
        )
        space = config.search_space
        assert space.methods == ROWS[base]
        if field is not None:
            assert getattr(space, field) == values
        assert space.projection_ranks == (2, 8, 16, 24)
        assert space.consistency_weights == (0.01, 0.1, 1.0, 10.0)
        if base in ("fishr", "rdm"):
            assert getattr(space, f"{base}_penalty_anneal_updates") == 1500
        if base == "rdm":
            assert space.rdm_variance_weight == 0.004
        optimizers = len(space.learning_rates) * len(space.weight_decays)
        assert optimizers == 16
        assert candidate_count == sum(
            optimizers * len(_method_settings_grid(space, method))
            for method in space.methods
        )
        assert (config.pair_count, config.batch_size, config.max_epochs) == (
            (256, 256, 40) if dataset == "cmnist" else (240, 256, 100)
        )
        if dataset == "cmnist":
            assert "cmnist-none-e03" in config.artifacts.dataset_manifest
            assert config.selectors == ("primary_robust", "secondary_source")
        else:
            assert config.selectors == ("waterbirds_validation_worst_group",)
        output_roots.add(config.output_root)
    assert len(output_roots) == len(MATRIX_GRIDS)
    assert len(tuple(root.glob("*-search.yaml"))) == len(MATRIX_GRIDS)


def test_composed_rdm_rejects_a_warm_up_that_would_never_activate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    space = _space("rdm")
    space.update(methods=["rdm_consistency"], rdm_penalty_anneal_updates=1500)
    del space["projection_ranks"]
    config_path, _ = write_cmnist_production_config(
        tmp_path,
        output_root=tmp_path / "output",
        overrides={"search_space": space, "batch_size": 8, "max_epochs": 2},
    )
    plan = plan_production_search(config_path)
    cache = fake_cache(plan)

    def loader(_plan: SearchPlan, *, tuning_only: bool = False):
        return cache

    monkeypatch.setattr("grit.search.cmnist._load_cmnist_cache", loader)
    with pytest.raises(ValueError, match="leaves no penalized update"):
        run_production_search(config_path)


@pytest.mark.parametrize(
    ("objective", "definition"),
    [
        ("groupdro", "training_environment"),
        ("groupdro", "target_environment"),
        ("lisa", "target_environment"),
    ],
)
def test_cmnist_source_environment_groups_use_sources_not_color(
    tmp_path: Path,
    objective: Literal["groupdro", "lisa"],
    definition: Literal["training_environment", "target_environment"],
) -> None:
    from grit.methods.baselines import LisaLinearProbeMethod
    from grit.methods.groupdro import cmnist_group_ids
    from grit.methods.training import GroupDroLinearProbeMethod
    from grit.schemas import CmnistSelector
    from grit.search.cmnist import (
        _cmnist_base_method,  # pyright: ignore[reportPrivateUsage]
        _CmnistRuntimeCandidate,  # pyright: ignore[reportPrivateUsage]
        materialize_cmnist_candidate_config,
    )
    from grit.search.scheduler import SearchRunTask

    def bind(
        cache: CmnistFeatureCache, plan: SearchPlan
    ) -> GroupDroLinearProbeMethod | LisaLinearProbeMethod:
        candidate = plan.candidates[0]
        runtime = _CmnistRuntimeCandidate(
            candidate,
            materialize_cmnist_candidate_config(
                plan, candidate, CmnistSelector.PRIMARY_ROBUST
            ),
            None,
        )
        method = _cmnist_base_method(cache, runtime, cast(SearchRunTask, None))
        assert isinstance(method, GroupDroLinearProbeMethod | LisaLinearProbeMethod)
        return method

    groups: dict[bool, torch.Tensor] = {}
    for source_environment in (False, True):
        space = _space(objective)
        space["methods"] = [objective]
        for field in ("projection_ranks", "consistency_weights"):
            del space[field]
        if source_environment:
            space[f"{objective}_group_definition"] = definition
        path, _ = write_cmnist_production_config(
            tmp_path / str(source_environment),
            output_root=tmp_path / str(source_environment) / "output",
            overrides={"search_space": space},
        )
        plan = plan_production_search(path)
        cache = fake_cache(plan)
        method = bind(cache, plan)
        groups[source_environment] = method.group_ids
        tables = cache.training_tables()
        targets = torch.cat([table.targets for table in tables])
        sources = torch.cat(
            [torch.full((len(t.source_ids),), i) for i, t in enumerate(tables)]
        )
        colors = torch.cat([table.colors for table in tables])
        expected = (
            cmnist_group_ids(targets, colors)
            if not source_environment
            else sources
            if definition == "training_environment"
            else 2 * targets + sources
        )
        assert torch.equal(method.group_ids, expected)
        assert method.group_count == (
            2 if source_environment and definition == "training_environment" else 4
        )
        if source_environment and definition == "target_environment":
            assert all(bool((method.group_ids == g).any()) for g in range(4))
        # Flipping every training color, clean label and digit moves label x color
        # groups but never source-based ones.
        flipped = replace(
            cache,
            **{
                name: replace(
                    table,
                    colors=1 - table.colors,
                    clean_labels=1 - table.clean_labels,
                    digits=(table.digits + 1) % 10,
                )
                for name, table in zip(("train_e01", "train_e02"), tables, strict=True)
            },
        )
        assert torch.equal(bind(flipped, plan).group_ids, method.group_ids) == (
            source_environment
        )
    assert not torch.equal(groups[False], groups[True])


def test_source_environment_groups_are_validated_per_method_and_dataset(
    tmp_path: Path,
) -> None:
    space = _space("groupdro")
    space.update(methods=["groupdro"], groupdro_group_definition="target_environment")
    for field in ("projection_ranks", "consistency_weights"):
        del space[field]
    # Label x source GroupDRO is a CMNIST option; on Waterbirds it would duplicate
    # the default label x background groups under another identity.
    plan_production_search(
        write_cmnist_production_config(
            tmp_path / "a",
            output_root=tmp_path / "a-out",
            overrides={"search_space": space},
        )[0]
    )
    with pytest.raises(ValidationError, match="leave groupdro_group_definition"):
        plan_production_search(
            write_waterbirds_production_config(
                tmp_path / "a2", output_root=tmp_path / "a2-out", search_space=space
            )[0]
        )
    unknown = {**space, "groupdro_group_definition": "target_color"}
    with pytest.raises(ValidationError):
        plan_production_search(
            write_cmnist_production_config(
                tmp_path / "a3",
                output_root=tmp_path / "a3-out",
                overrides={"search_space": unknown},
            )[0]
        )
    lisa = _space("lisa")
    lisa.update(methods=["lisa"], lisa_group_definition="target_environment")
    for field in ("projection_ranks", "consistency_weights"):
        del lisa[field]
    with pytest.raises(
        ValidationError, match="backgrounds, which are its environments"
    ):
        plan_production_search(
            write_waterbirds_production_config(
                tmp_path / "b", output_root=tmp_path / "b-out", search_space=lisa
            )[0]
        )
    erm = _space("erm")
    erm["groupdro_group_definition"] = "training_environment"
    with pytest.raises(ValidationError, match="requires GroupDRO"):
        plan_production_search(
            write_cmnist_production_config(
                tmp_path / "c",
                output_root=tmp_path / "c-out",
                overrides={"search_space": erm},
            )[0]
        )
    # Each grouping is part of every candidate's scientific identity.
    identities: dict[str | None, set[str]] = {}
    for definition in (None, "training_environment", "target_environment"):
        grouped = _space("groupdro")
        if definition is not None:
            grouped["groupdro_group_definition"] = definition
        grouped_plan = plan_production_search(
            write_cmnist_production_config(
                tmp_path / f"id-{definition}",
                output_root=tmp_path / f"id-{definition}-out",
                overrides={"search_space": grouped},
            )[0]
        )
        identities[definition] = {c.candidate_id for c in grouped_plan.candidates}
        identities[f"{definition}-digest"] = {
            c.scientific_config_digest for c in grouped_plan.candidates
        }
    for a, b in (
        (None, "training_environment"),
        (None, "target_environment"),
        ("training_environment", "target_environment"),
    ):
        assert not identities[a] & identities[b]
        assert not identities[f"{a}-digest"] & identities[f"{b}-digest"]
    # Unset options stay out of the authored config, so existing identities hold.
    default = _space("groupdro")
    config_path, _ = write_cmnist_production_config(
        tmp_path / "d",
        output_root=tmp_path / "d-out",
        overrides={"search_space": default},
    )
    plan = plan_production_search(config_path)
    dumped = plan.resolved_config.config.search_space.model_dump(mode="json")
    assert "groupdro_group_definition" not in dumped
    assert "lisa_group_definition" not in dumped
