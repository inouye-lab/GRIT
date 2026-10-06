"""Within-objective paired estimates retain seed, selector, and pair identity."""

from __future__ import annotations

from math import isclose

import pytest
from pydantic import ValidationError

from grit.methods.types import IMPLEMENTED_METHODS, MethodId, consumes_pairs
from grit.schemas import ORDINARY_CMNIST_SELECTORS, CmnistSelector
from grit.search.outputs import (
    CmnistMethodSelectorSummary,
    CmnistProductionSummary,
    WaterbirdsProductionMethodSummary,
    WaterbirdsProductionSummary,
    make_cmnist_intervention_comparisons,
    make_waterbirds_intervention_comparisons,
)
from grit.search.waterbirds_contracts import make_waterbirds_metric_summary
from tests.test_search_outputs import (
    _lineage,  # pyright: ignore[reportPrivateUsage]
    _method_summary,  # pyright: ignore[reportPrivateUsage]
    _paired,  # pyright: ignore[reportPrivateUsage]
)

ROWS: tuple[tuple[MethodId, MethodId, MethodId], ...] = (
    ("erm", "grit", "erm_consistency"),
    ("rex", "rex_grit", "rex_consistency"),
    ("irm", "irm_grit", "irm_consistency"),
    ("fishr", "fishr_grit", "fishr_consistency"),
    ("groupdro", "groupdro_grit", "groupdro_consistency"),
    ("sd", "sd_grit", "sd_consistency"),
    ("rdm", "rdm_grit", "rdm_consistency"),
    ("lisa", "lisa_grit", "lisa_consistency"),
)


def _cmnist_method(
    method: MethodId, selector: CmnistSelector, offset: float
) -> CmnistMethodSelectorSummary:
    return _method_summary(method, selector, offset).model_copy(
        update={"pair_count": 256 if consumes_pairs(method) else None}
    )


@pytest.mark.parametrize("row", ROWS)
def test_cmnist_all_contrasts_preserve_selectors_and_join_by_seed(
    row: tuple[MethodId, MethodId, MethodId],
) -> None:
    methods = tuple(
        _cmnist_method(method, selector, offset)
        for method, offset in zip(row, (0.0, 0.05, 0.02), strict=True)
        for selector in ORDINARY_CMNIST_SELECTORS
    )
    # A different ordering of a method's configured seed series must not change
    # which seed observations are subtracted.
    right = methods[-1]
    methods = (
        *methods[:-1],
        right.model_copy(
            update={
                "configured_final_seeds": tuple(reversed(right.configured_final_seeds)),
                "final_observations": tuple(reversed(right.final_observations)),
            }
        ),
    )
    comparisons = make_cmnist_intervention_comparisons(methods)
    assert len(comparisons) == 6
    for item in comparisons:
        expected = (
            0.05
            if item.minuend == row[1] and item.subtrahend == row[0]
            else (0.02 if item.minuend == row[2] else 0.03)
        )
        assert isclose(item.mean, expected, abs_tol=1e-12)
        assert set(dict(item.differences_by_seed)) == set(range(301, 311))
        assert all(
            isclose(value, expected, abs_tol=1e-12)
            for _, value in item.differences_by_seed
        )
        assert isclose(item.ci95_lower, expected, abs_tol=1e-12)
        assert isclose(item.ci95_upper, expected, abs_tol=1e-12)
        assert item.pair_count == 256
    assert {item.selector for item in comparisons} == {
        selector.value for selector in ORDINARY_CMNIST_SELECTORS
    }


@pytest.mark.parametrize(
    "field,value",
    [
        ("pair_manifest_digest", "sha256:other-pairs"),
        ("feature_cache_manifest_digest", "sha256:other-encoder"),
        ("dataset_manifest_digest", "sha256:other-dataset"),
        ("normalization", "l2"),
        ("held_out_validation_split", "val_e03"),
    ],
)
def test_comparisons_reject_incompatible_lineage(field: str, value: str) -> None:
    left = _cmnist_method("rex_grit", CmnistSelector.PRIMARY_ROBUST, 0.02)
    right = _cmnist_method("rex_consistency", CmnistSelector.PRIMARY_ROBUST, 0.0)
    right = right.model_copy(
        update={
            "lineage": right.lineage.model_copy(update={field: value}),
        }
    )
    with pytest.raises(ValueError, match="matching lineage and selector"):
        make_cmnist_intervention_comparisons((left, right))


@pytest.mark.parametrize("budget", [None, 128])
def test_comparisons_reject_unknown_or_unequal_pair_budgets(budget: int | None) -> None:
    left = _cmnist_method("irm_grit", CmnistSelector.PRIMARY_ROBUST, 0.02)
    right = _cmnist_method("irm_consistency", CmnistSelector.PRIMARY_ROBUST, 0.0)
    right = right.model_copy(update={"pair_count": budget})
    with pytest.raises(ValueError, match="identical explicit pair budgets"):
        make_cmnist_intervention_comparisons((left, right))


def test_matrix_summary_requires_complete_correct_comparisons() -> None:
    methods = tuple(
        sorted(
            (
                _cmnist_method(method, selector, 0.01 * index)
                for index, method in enumerate(ROWS[0])
                for selector in ORDINARY_CMNIST_SELECTORS
            ),
            key=lambda item: (
                IMPLEMENTED_METHODS.index(item.method_id),
                ORDINARY_CMNIST_SELECTORS.index(item.selector),
            ),
        )
    )
    paired = tuple(
        _paired(selector, methods[index], methods[index + 2])
        for index, selector in enumerate(ORDINARY_CMNIST_SELECTORS)
    )
    with pytest.raises(ValidationError, match="comparisons are incomplete"):
        CmnistProductionSummary(
            schema_version="grit.cmnist-production-summary/v1",
            reportable=True,
            plan_digest="sha256:plan",
            lineage=_lineage(),
            methods=methods,
            paired_selectors=paired,
        )
    comparisons = make_cmnist_intervention_comparisons(methods)
    summary = CmnistProductionSummary(
        schema_version="grit.cmnist-production-summary/v1",
        reportable=True,
        plan_digest="sha256:plan",
        lineage=_lineage(),
        methods=methods,
        paired_selectors=paired,
        intervention_comparisons=comparisons,
    )
    forged = comparisons[0].model_copy(update={"mean": 0.9})
    with pytest.raises(ValidationError, match="uncertainty is inconsistent"):
        CmnistProductionSummary.model_validate(
            summary.model_copy(
                update={
                    "intervention_comparisons": (forged, *comparisons[1:]),
                }
            )
        )


def _waterbirds_method(
    method: MethodId, offset: float
) -> WaterbirdsProductionMethodSummary:
    seeds = tuple(range(301, 311))
    values = tuple(0.5 + offset + index * 0.01 for index in range(10))
    return WaterbirdsProductionMethodSummary(
        method_id=method,
        selector="waterbirds_validation_worst_group",
        lineage=_lineage(),
        pair_count=240 if consumes_pairs(method) else None,
        selected_candidate_id=f"{method}:a",
        finalist_candidate_ids=(f"{method}:a", f"{method}:b", f"{method}:c"),
        configured_final_seeds=seeds,
        result_paths_by_seed=tuple((seed, f"{method}/{seed}.json") for seed in seeds),
        worst_group_by_seed=tuple(zip(seeds, values, strict=True)),
        adjusted_average_by_seed=tuple(zip(seeds, values, strict=True)),
        raw_average_by_seed=tuple(zip(seeds, values, strict=True)),
        worst_group_summary=make_waterbirds_metric_summary(
            "worst_group_accuracy", values
        ),
        adjusted_average_summary=make_waterbirds_metric_summary(
            "adjusted_average_accuracy", values
        ),
        raw_average_summary=make_waterbirds_metric_summary(
            "raw_average_accuracy", values
        ),
    )


@pytest.mark.parametrize("row", ROWS)
def test_waterbirds_reports_all_three_metrics_and_contrasts(
    row: tuple[MethodId, MethodId, MethodId],
) -> None:
    methods = tuple(
        _waterbirds_method(method, offset)
        for method, offset in zip(row, (0.0, 0.05, 0.02), strict=True)
    )
    comparisons = make_waterbirds_intervention_comparisons(methods)
    assert len(comparisons) == 9
    assert {item.metric_name for item in comparisons} == {
        "worst_group_accuracy",
        "adjusted_average_accuracy",
        "raw_average_accuracy",
    }
    assert all(
        isclose(item.mean, expected, abs_tol=1e-12)
        for item, expected in zip(
            comparisons, (0.05,) * 3 + (0.02,) * 3 + (0.03,) * 3, strict=True
        )
    )
    if row[0] != "erm":
        summary = WaterbirdsProductionSummary(
            schema_version="grit.waterbirds-production-summary/v2",
            reportable=True,
            plan_digest="sha256:plan",
            lineage=_lineage(),
            selector="waterbirds_validation_worst_group",
            methods=methods,
            paired_worst_group_by_seed=None,
            paired_worst_group_summary=None,
            intervention_comparisons=comparisons,
        )
        assert len(summary.intervention_comparisons) == 9


def test_waterbirds_rejects_mixed_selectors() -> None:
    left = _waterbirds_method("fishr_grit", 0.02)
    right = _waterbirds_method("fishr_consistency", 0.0).model_copy(
        update={
            "selector": "another_selector",
        }
    )
    with pytest.raises(ValueError, match="matching lineage and selector"):
        make_waterbirds_intervention_comparisons((left, right))


def test_comparisons_reject_different_seed_identities() -> None:
    left = _cmnist_method("irm_grit", CmnistSelector.PRIMARY_ROBUST, 0.02)
    right = _cmnist_method("irm_consistency", CmnistSelector.PRIMARY_ROBUST, 0.0)
    right = right.model_copy(update={"configured_final_seeds": tuple(range(401, 411))})
    with pytest.raises(ValueError, match="identical final seed identities"):
        make_cmnist_intervention_comparisons((left, right))


def test_uncertainty_is_computed_from_paired_differences() -> None:
    from math import sqrt
    from statistics import fmean, stdev

    left = _waterbirds_method("fishr_grit", 0.02)
    right = _waterbirds_method("fishr_consistency", 0.0)
    left = left.model_copy(
        update={
            "worst_group_by_seed": tuple(
                (seed, value + index * 0.005)
                for index, (seed, value) in enumerate(left.worst_group_by_seed)
            ),
        }
    )
    comparison = make_waterbirds_intervention_comparisons((left, right))[0]
    differences = tuple(value for _, value in comparison.differences_by_seed)
    mean = fmean(differences)
    deviation = stdev(differences)
    assert deviation > 0.0
    assert comparison.mean == mean
    assert comparison.sample_standard_deviation == deviation
    half_width = 2.2621571627409915 * deviation / sqrt(10)
    assert comparison.ci95_lower == mean - half_width
    assert comparison.ci95_upper == mean + half_width
