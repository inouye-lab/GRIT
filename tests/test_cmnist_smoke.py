"""Hermetic end-to-end CMNIST ERM/oracle-GRIT vertical-slice smoke test."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import cast

import pytest

from grit.config import CmnistSourceCounts, SeedSets
from grit.methods.projection import ProjectionDiagnostics
from grit.results import OrdinaryRunResult
from grit.schemas import CmnistSelector
from grit.search.cmnist_runner import (
    CmnistSmokeRunConfig,
    CmnistSmokeSummary,
    run_cmnist_smoke,
)
from grit.selection.cmnist import (
    FinalTestMetricRecord,
    ValidationMetricRecord,
    select_checkpoint,
)


def _config(output_root: Path) -> CmnistSmokeRunConfig:
    return CmnistSmokeRunConfig(
        schema_version="grit.cmnist-smoke/v1",
        non_reportable=True,
        output_root=output_root.as_posix(),
        construction_seed=17,
        pair_seed=23,
        fake_encoder_seed=31,
        image_size=4,
        source_counts=CmnistSourceCounts(
            train_e01=128,
            train_e02=128,
            validation=32,
            test=32,
        ),
        label_flip_prob=0.25,
        normalization="none",
        pair_count=256,
        projection_rank=1,
        relative_singular_value_tolerance=1e-12,
        learning_rates=(0.001, 0.003, 0.01),
        weight_decay=0.0,
        batch_size=64,
        max_epochs=1,
        seed_sets=SeedSets(
            tuning=(101, 102, 103),
            confirmation=(201, 202),
            final=(301, 302, 303, 304, 305, 306, 307, 308, 309, 310),
        ),
    )


def test_hermetic_cmnist_smoke_runs_both_methods_through_final_gate(
    tmp_path: Path,
) -> None:
    output_root = tmp_path / "smoke"
    summary = run_cmnist_smoke(_config(output_root))
    assert (
        CmnistSmokeSummary.model_validate_json(
            (output_root / "smoke-result.json").read_text(encoding="utf-8")
        )
        == summary
    )
    assert tuple(method.method_id for method in summary.methods) == ("erm", "grit")
    assert all(len(method.result_paths) == 20 for method in summary.methods)
    assert all(len(method.final_accuracies) == 20 for method in summary.methods)
    assert all(len(method.finalist_union_ids) >= 3 for method in summary.methods)

    results = tuple(
        OrdinaryRunResult.model_validate_json(
            (output_root / relative_path).read_text(encoding="utf-8")
        )
        for method in summary.methods
        for relative_path in method.result_paths
    )
    assert len(results) == 40
    assert {result.resolved_config.algorithm.kind for result in results} == {
        "erm",
        "grit",
    }
    assert all(result.status.kind == "succeeded" for result in results)
    assert all(result.restoration is not None for result in results)
    assert all(result.final_test_metrics is not None for result in results)
    assert all(result.resolved_config.reportable is False for result in results)

    projection = ProjectionDiagnostics.model_validate_json(
        (output_root / "runs/grit/projection-1.json").read_text(encoding="utf-8")
    )
    assert projection.pair_manifest_digest == summary.pair_manifest_digest
    assert projection.feature_cache_manifest_digest == summary.feature_manifest_digest

    final_metrics = results[0].final_test_metrics
    if final_metrics is None:
        raise AssertionError("successful smoke result lacks final metrics")
    invalid = cast(
        Sequence[ValidationMetricRecord],
        cast(Sequence[FinalTestMetricRecord], final_metrics),
    )
    with pytest.raises(TypeError, match="ValidationMetricRecord"):
        select_checkpoint(invalid, CmnistSelector.PRIMARY_ROBUST)


def test_cmnist_smoke_is_reproducible_for_identical_seeds(tmp_path: Path) -> None:
    first = run_cmnist_smoke(_config(tmp_path / "first"))
    second = run_cmnist_smoke(_config(tmp_path / "second"))
    assert first.dataset_manifest_digest == second.dataset_manifest_digest
    assert first.feature_manifest_digest == second.feature_manifest_digest
    assert first.pair_manifest_digest == second.pair_manifest_digest
    assert tuple(method.final_accuracies for method in first.methods) == tuple(
        method.final_accuracies for method in second.methods
    )
    assert tuple(method.primary_candidate_id for method in first.methods) == tuple(
        method.primary_candidate_id for method in second.methods
    )
    assert tuple(method.secondary_candidate_id for method in first.methods) == tuple(
        method.secondary_candidate_id for method in second.methods
    )
