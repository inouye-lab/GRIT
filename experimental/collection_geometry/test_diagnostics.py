"""Check A1-preserving data construction and the empirical transport identity."""

from __future__ import annotations

import math

from core import Probe
from diagnostics import diagnostics_for_synthetic

from grit.data.collection_geometry import SimulationSpec, make_geometry


def test_stable_probe_has_only_bridge_covariate_shift() -> None:
    spec = SimulationSpec(dimension=12, stable_rank=2, b_rank=1, o_rank=1)
    geometry = make_geometry(spec, seed=51, replicate=0)
    weight = geometry.label_direction(spec)
    probe = Probe(weight=weight, bias=0.3, gradient=0.0, iterations=0)
    rows = diagnostics_for_synthetic(geometry, spec, 51, 0, {"oracle": probe})
    assert len(rows) == 4
    for row in rows:
        assert float(row["raw_u_sensitivity"]) < 1e-12
        assert float(row["direct_absolute_loss_change"]) < 1e-12
        assert float(row["max_stable_displacement"]) < 1e-12
        assert abs(float(row["telescoping_residual"])) < 1e-12
        assert float(row["risk_decomposition_slack"]) >= -1e-12
        assert float(row["bridge_term"]) > 0.01
    assert len({row["target_weights_sha256"] for row in rows}) == 1
    assert len({row["source_features_sha256"] for row in rows}) == 1
    assert rows == diagnostics_for_synthetic(geometry, spec, 51, 0, {"oracle": probe})


def test_unstable_probe_obeys_nested_transport_bounds() -> None:
    spec = SimulationSpec(dimension=12, stable_rank=2, b_rank=1, o_rank=1)
    geometry = make_geometry(spec, seed=52, replicate=1)
    weight = 0.5 * geometry.label_direction(spec) + geometry.u_b[:, 0]
    probe = Probe(weight=weight, bias=0.0, gradient=0.0, iterations=0)
    rows = diagnostics_for_synthetic(geometry, spec, 52, 1, {"unstable": probe})
    for row in rows:
        assert math.isclose(float(row["raw_u_sensitivity"]), 1.0)
        direct = float(row["direct_absolute_loss_change"])
        transport = float(row["direct_score_transport"])
        moment_bound = float(row["rho_times_sensitivity"])
        assert direct <= transport + 1e-12 <= moment_bound + 2e-12
        assert float(row["target_loss"]) <= float(row["risk_upper_bound"]) + 1e-12
        assert abs(float(row["telescoping_residual"])) < 1e-12
    assert float(rows[0]["rho"]) == 0.0
    assert float(rows[-1]["bridge_loss"]) > float(rows[1]["bridge_loss"])
