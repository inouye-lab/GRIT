"""Checks for changing one theoretical factor without changing the other two."""

import numpy as np
import pytest
from sweeps import CONFIG, conditions, signal_design


def test_spectrum_is_exact_and_independent_of_pair_budget() -> None:
    u = np.linalg.qr(np.random.default_rng(51).standard_normal((12, 2)))[0]
    for g in [0, *CONFIG["coverage_values"]]:
        for k in CONFIG["pair_values"]:
            clean = signal_design(u, k, g)
            measured = np.linalg.eigvalsh(u.T @ clean.T @ clean @ u / k)
            np.testing.assert_allclose(measured, [g, 1], atol=1e-12)
            np.testing.assert_allclose(clean, (clean @ u) @ u.T, atol=1e-12)


def test_noisy_measurements_have_nested_prefixes_without_normalization() -> None:
    rng = np.random.default_rng(72)
    u = np.linalg.qr(rng.standard_normal((12, 2)))[0]
    noise = rng.standard_normal((1024, 12))
    full = signal_design(u, 1024, 0.25) + 0.3 * noise
    for k in [64, 96, 128, 256, 512]:
        current = signal_design(u, k, 0.25) + 0.3 * noise[:k]
        np.testing.assert_array_equal(current, full[:k])
        np.testing.assert_allclose(current - signal_design(u, k, 0.25), 0.3 * noise[:k])


def test_each_sweep_changes_only_its_declared_factor() -> None:
    for row in conditions(CONFIG):
        assert row["lambda_max"] == 1
        if row["sweep"] == "coverage":
            assert row["k"] == 256 and row["sigma"] == 0.1
        elif row["sweep"] == "noise":
            assert row["k"] == 256 and row["lambda_min"] == 1
        elif row["sweep"] == "pairs":
            assert row["sigma"] == 0.1 and row["lambda_min"] == 1
    with pytest.raises(ValueError):
        signal_design(np.eye(4)[:, :2], 7, 1)
