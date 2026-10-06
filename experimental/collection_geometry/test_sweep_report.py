"""Check pairing and mathematical interpretation in the new sweep report."""

from __future__ import annotations

import numpy as np
import pytest
from sweep_report import metrics_for, summarize_sweeps


def make_row(replicate: int, sweep: str, value: float, missed: float) -> dict:
    return {
        "dataset": "synthetic",
        "replicate": replicate,
        "sweep": sweep,
        "value": value,
        "method": "grit",
        "k": int(value) if sweep == "pairs" else 256,
        "sigma": value if sweep == "noise" else 0.1,
        "lambda_min": value if sweep == "coverage" else 1.0,
        "lambda_max": 1.0,
        "missed_u": missed,
        "sensitivity": missed,
        "coefficient_norm": 2.0,
        "test": {"both_shift": 0.75, "both_loss": 0.3},
    }


def test_normalization_and_geometric_bound() -> None:
    row = make_row(0, "coverage", 1.0, 0.2)
    assert metrics_for(row)["normalized_sensitivity"] == 0.1
    row["sensitivity"] = 1.0
    with pytest.raises(ValueError, match="geometric upper bound"):
        metrics_for(row)


def test_endpoint_interval_preserves_paired_replicates() -> None:
    rows = [
        make_row(rep, "coverage", value, 0.2 + 0.1 * rep - 0.05 * value)
        for rep in range(3)
        for value in (0.5, 1.0)
    ]
    summary = summarize_sweeps(rows[::-1], {"slope_min_k": 512})
    difference = next(
        row for row in summary["paired_endpoints"] if row["metric"] == "missed_u"
    )
    np.testing.assert_allclose(difference["mean"], -0.025)
    np.testing.assert_allclose(difference["ci95_mean"], [-0.025, -0.025])
    with pytest.raises(ValueError, match="matching replicates"):
        summarize_sweeps(rows[:-1], {})
    with pytest.raises(ValueError, match="Duplicate"):
        summarize_sweeps(rows + [rows[0]], {})


def test_slopes_use_prespecified_range_and_clean_control_is_separate() -> None:
    rows = [
        make_row(rep, "pairs", k, (rep + 1) / np.sqrt(k))
        for rep in range(3)
        for k in (64, 256, 512, 1024, 2048)
    ]
    for row in rows:
        if row["k"] < 512:
            row["missed_u"] *= 2
            row["sensitivity"] *= 2
    rows += [
        make_row(rep, "noise", sigma, sigma / 2)
        for rep in range(3)
        for sigma in (0.0, 0.1, 1.0)
    ]
    rows += [make_row(rep, "clean_control", 0.0, 0.0) for rep in range(3)]
    rows += [make_row(rep, "nonspanning_control", 0.0, 1.0) for rep in range(3)]
    summary = summarize_sweeps(rows, {"slope_min_k": 512})
    for result in summary["slopes"]:
        np.testing.assert_allclose(result["mean"], -0.5)
        assert result["k_values"] == [512, 1024, 2048]
    noise = [row for row in summary["paired_endpoints"] if row["sweep"] == "noise"]
    assert all(row["from_value"] == 0.1 for row in noise)
    assert any(
        row["sweep"] == "noise" and row["value"] == 0.0 for row in summary["groups"]
    )
    assert (
        len([row for row in summary["groups"] if row["sweep"].endswith("control")]) == 2
    )
    assert not any(
        row["sweep"].endswith("control") for row in summary["paired_endpoints"]
    )
