"""Production summary and authoritative local-index regression tests."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest
from pydantic import ValidationError

from grit.results import CodeProvenance
from grit.schemas import CmnistSelector
from grit.search.outputs import (
    CmnistFinalSeedObservation,
    CmnistMethodSelectorSummary,
    CmnistPairedSeedDifference,
    CmnistPairedSelectorSummary,
    CmnistProductionSummary,
    ExperimentIndex,
    IndexedArtifact,
    make_cmnist_accuracy_summary,
    verify_experiment_index,
)
from grit.search.plan import SearchLineage, build_search_plan
from tests.test_search_plan import resolved_search_fixture


def _lineage() -> SearchLineage:
    return SearchLineage(
        dataset_manifest_digest="sha256:dataset",
        feature_cache_manifest_digest="sha256:features",
        pair_manifest_digest="sha256:pairs",
        normalization="none",
        adjusted_weight_spec_digest=None,
    )


def _method_summary(
    method: str,
    selector: CmnistSelector,
    offset: float,
) -> CmnistMethodSelectorSummary:
    seeds = (301, 302, 303, 304, 305, 306, 307, 308, 309, 310)
    observations = tuple(
        CmnistFinalSeedObservation.model_validate(
            {
                "seed": seed,
                "method_id": method,
                "selector": selector,
                "result_path": f"runs/{method}/{selector.value}/{seed}.json",
                "metric_record_id": f"metric:{method}:{selector.value}:{seed}",
                "test_ood_accuracy": 0.60 + offset + index / 100,
            }
        )
        for index, seed in enumerate(seeds)
    )
    values = tuple(float(item.test_ood_accuracy) for item in observations)
    selected = f"candidate:{method}:a"
    return CmnistMethodSelectorSummary.model_validate(
        {
            "method_id": method,
            "selector": selector,
            "lineage": _lineage(),
            "selected_candidate_id": selected,
            "finalist_candidate_ids": (
                f"candidate:{method}:a",
                f"candidate:{method}:b",
                f"candidate:{method}:c",
            ),
            "configured_final_seeds": seeds,
            "final_observations": observations,
            "accuracy_summary": make_cmnist_accuracy_summary(
                "test_ood_accuracy", values
            ),
        }
    )


def _paired(
    selector: CmnistSelector,
    erm: CmnistMethodSelectorSummary,
    grit: CmnistMethodSelectorSummary,
) -> CmnistPairedSelectorSummary:
    erm_values = {
        item.seed: float(item.test_ood_accuracy) for item in erm.final_observations
    }
    grit_values = {
        item.seed: float(item.test_ood_accuracy) for item in grit.final_observations
    }
    differences = tuple(
        CmnistPairedSeedDifference(
            seed=seed,
            selector=selector,
            grit_minus_erm_test_ood_accuracy=(
                grit_values[seed] - erm_values[seed]
            ),
        )
        for seed in erm.configured_final_seeds
    )
    return CmnistPairedSelectorSummary(
        selector=selector,
        configured_final_seeds=erm.configured_final_seeds,
        paired_differences=differences,
        difference_summary=make_cmnist_accuracy_summary(
            "grit_minus_erm_test_ood_accuracy",
            tuple(
                float(item.grit_minus_erm_test_ood_accuracy)
                for item in differences
            ),
        ),
    )


def test_cmnist_production_summary_pairs_methods_by_seed() -> None:
    primary_erm = _method_summary("erm", CmnistSelector.PRIMARY_ROBUST, 0.0)
    secondary_erm = _method_summary("erm", CmnistSelector.SECONDARY_SOURCE, 0.01)
    primary_grit = _method_summary("grit", CmnistSelector.PRIMARY_ROBUST, 0.02)
    secondary_grit = _method_summary(
        "grit", CmnistSelector.SECONDARY_SOURCE, 0.03
    )
    summary = CmnistProductionSummary(
        schema_version="grit.cmnist-production-summary/v1",
        reportable=True,
        plan_digest="sha256:plan",
        lineage=_lineage(),
        methods=(primary_erm, secondary_erm, primary_grit, secondary_grit),
        paired_selectors=(
            _paired(CmnistSelector.PRIMARY_ROBUST, primary_erm, primary_grit),
            _paired(
                CmnistSelector.SECONDARY_SOURCE,
                secondary_erm,
                secondary_grit,
            ),
        ),
    )
    assert CmnistProductionSummary.model_validate_json(summary.canonical_json()) == (
        summary
    )
    payload = summary.model_dump(mode="json")
    payload["methods"][2]["final_observations"] = list(
        reversed(payload["methods"][2]["final_observations"])
    )
    with pytest.raises(ValidationError, match="configured seeds exactly once"):
        CmnistProductionSummary.model_validate_json(
            __import__("json").dumps(payload)
        )


def test_experiment_index_round_trip_and_digest_tampering(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "grit.search.plan._code_provenance",
        lambda: CodeProvenance(git_revision="1" * 40, git_dirty=False),
    )
    plan = build_search_plan(resolved_search_fixture())
    plan_path = tmp_path / "search-plan.json"
    resolved_path = tmp_path / "resolved-config.json"
    plan_path.write_text(plan.canonical_json() + "\n", encoding="utf-8")
    resolved_path.write_text(
        plan.resolved_config.canonical_json() + "\n", encoding="utf-8"
    )
    artifact_path = tmp_path / "runs" / "result.json"
    artifact_path.parent.mkdir(parents=True)
    artifact_path.write_text('{"schema_version":"test.run/v1"}\n', encoding="utf-8")
    paths = (resolved_path, artifact_path, plan_path)
    artifacts = tuple(
        IndexedArtifact(
            kind="test_artifact",
            relative_path=path.relative_to(tmp_path).as_posix(),
            sha256=f"sha256:{hashlib.sha256(path.read_bytes()).hexdigest()}",
            schema_version="test.run/v1" if path == artifact_path else None,
        )
        for path in sorted(paths)
    )
    index = ExperimentIndex(
        schema_version="grit.experiment-index/v1",
        dataset="cmnist",
        reportable=True,
        plan_digest=plan.canonical_digest(),
        resolved_config_digest=plan.resolved_config.canonical_digest(),
        artifacts=artifacts,
    )
    assert verify_experiment_index(tmp_path, index) == index
    assert ExperimentIndex.model_validate_json(index.canonical_json()) == index
    wrong_identity = index.model_copy(update={"plan_digest": "sha256:forged"})
    with pytest.raises(ValueError, match="plan/config identity"):
        verify_experiment_index(tmp_path, wrong_identity)
    extra = tmp_path / "unindexed.json"
    extra.write_text("{}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="complete output tree"):
        verify_experiment_index(tmp_path, index)
    extra.unlink()
    artifact_path.write_text("tampered\n", encoding="utf-8")
    with pytest.raises(ValueError, match="digest mismatch"):
        verify_experiment_index(tmp_path, index)


def test_experiment_index_rejects_escape_and_duplicate_paths() -> None:
    artifact = IndexedArtifact(
        kind="result",
        relative_path="run.json",
        sha256="sha256:value",
        schema_version=None,
    )
    with pytest.raises(ValidationError, match="inside the output root"):
        IndexedArtifact(
            kind="result",
            relative_path="../run.json",
            sha256="sha256:value",
            schema_version=None,
        )
    with pytest.raises(ValidationError, match="unique"):
        ExperimentIndex(
            schema_version="grit.experiment-index/v1",
            dataset="cmnist",
            reportable=True,
            plan_digest="sha256:plan",
            resolved_config_digest="sha256:config",
            artifacts=(artifact, artifact),
        )
