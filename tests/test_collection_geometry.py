"""Collection-geometry simulation: construction and integrity checks."""

from __future__ import annotations

import math

import numpy as np

from grit import geometry as g
from grit.data import collection_geometry as sim

SMALL = sim.SimulationSpec(
    dimension=24,
    train_size=600,
    validation_per_group=10,
    test_per_group=10,
    bank_size=32,
    calibration_pairs=64,
)


def _close(value: float, expected: float, tolerance: float = 1e-9) -> bool:
    return math.isclose(value, expected, rel_tol=1e-9, abs_tol=tolerance)


def test_training_weights_match_the_protocol_table() -> None:
    within = g.training_group_weights(0.95, 0.95)[:4] * 2
    assert np.allclose(within, [0.9025, 0.0475, 0.0475, 0.0025])
    within = g.training_group_weights(0.95, 0.99)[:4] * 2
    assert np.allclose(within, [0.9405, 0.0495, 0.0095, 0.0005])
    assert math.isclose(float(g.training_group_weights(0.9, 0.8).sum()), 1.0)


def test_group_metrics_follow_group_order() -> None:
    accuracies = [1.0, 0.8, 0.6, 0.4, 0.9, 0.7, 0.5, 0.3]
    metrics = g.group_metrics(accuracies, g.training_group_weights(0.95, 0.95))
    assert _close(metrics["b_shift"], 0.75)
    assert _close(metrics["o_shift"], 0.55)
    assert _close(metrics["both_shift"], 0.35)
    assert metrics["worst_group"] == 0.3


def test_minority_count_and_unreachable_levels() -> None:
    assert g.minority_count(256, 0.5, 29.0, 20.0) == round(256 * 29 / 49)
    assert g.minority_count(16, 1.0, 29.0, 20.0) == 16
    assert g.minority_count(16, 0.0, 29.0, 20.0) == 0
    assert g.minority_count(4, 0.01, 1.0, 100.0) == 0  # unreachable at s > 0


def test_coupled_positions_are_nested_in_count() -> None:
    draws = np.random.default_rng(0).random(64)
    previous = np.zeros(32, dtype=bool)
    for n_o in range(33):
        chosen = g.o_positions(draws, 32, n_o)
        assert chosen.sum() == n_o and bool((chosen >= previous).all())
        previous = chosen


def test_pairs_vary_one_factor_and_cross_its_category() -> None:
    geometry = sim.make_geometry(SMALL, 3, 0)
    train = sim.training_split(SMALL, geometry, 3, 0)
    bank = sim.make_bank(SMALL, geometry, train, 3, 0)
    for delta, basis, cue, mean in (
        (bank.delta_b, geometry.u_b, train.q_b, geometry.m_b),
        (bank.delta_o, geometry.u_o, train.q_o, geometry.m_o),
    ):
        residual = delta - (delta @ basis) @ basis.T
        assert np.abs(residual).max() < 1e-12
        # b - b' = 2 q m + noise: mostly on the anchor's side of the category.
        side = ((delta @ basis) @ mean) * cue[: SMALL.bank_size]
        assert (side > 0).mean() > 0.9


def test_balanced_splits_have_exact_group_counts() -> None:
    geometry = sim.make_geometry(SMALL, 3, 0)
    for test in (False, True):
        split = sim.balanced_split(SMALL, geometry, 3, 0, test=test)
        counts = np.bincount(split.groups, minlength=8)
        assert (counts == 10).all()


def test_clean_span_recovery_is_exact_and_partial_below_rank() -> None:
    """Integrity check: n_F >= r_F recovers the factor span exactly."""

    geometry = sim.make_geometry(SMALL, 5, 0)
    train = sim.training_split(SMALL, geometry, 5, 0)
    bank = sim.make_bank(SMALL, geometry, train, 5, 0)
    moments = {f: geometry.second_moment(f, SMALL) for f in ("b", "o")}
    for n_o, expected_o in ((2, 1.0), (5, 1.0)):
        left, right, _ = bank.collection(g.o_positions(bank.draws, 16, n_o))
        projection = g.fit_projection(left, right, 4, digest="t")
        basis = g.removed_basis(projection)
        assert projection.diagnostics.numerical_rank == 4
        assert _close(g.captured_energy(basis, moments["b"]), 1.0, 1e-10)
        assert _close(g.captured_energy(basis, moments["o"]), expected_o, 1e-10)
    left, right, _ = bank.collection(g.o_positions(bank.draws, 16, 1))
    projection = g.fit_projection(left, right, 4, digest="t")
    assert projection.diagnostics.effective_rank == 3  # capped at numerical rank
    assert g.captured_energy(g.removed_basis(projection), moments["o"]) < 0.999
    left, right, _ = bank.collection(g.o_positions(bank.draws, 16, 0))
    assert g.fit_projection(left, right, 4, digest="t").diagnostics.effective_rank == 2


def test_overlap_tilts_the_first_o_direction_by_theta() -> None:
    spec = SMALL.model_copy(update={"overlap_degrees": 30.0})
    base, tilted = sim.make_geometry(SMALL, 9, 0), sim.make_geometry(spec, 9, 0)
    assert np.allclose(tilted.u_o.T @ tilted.u_o, np.eye(2))
    cosine = float(base.u_o[:, 0] @ tilted.u_o[:, 0])
    assert _close(math.degrees(math.acos(cosine)), 30.0, 1e-6)
    # Everything else is shared, so stressor levels are paired.
    assert np.array_equal(base.stable, tilted.stable)
