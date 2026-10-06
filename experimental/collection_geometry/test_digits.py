"""Identity isolation and exact geometry checks for the MNIST extension."""

from __future__ import annotations

import numpy as np
from digits import (
    DigitData,
    _regional_features,
    extract_canvas,
    make_digits,
    render_digit,
)


def _data() -> DigitData:
    rng = np.random.default_rng(12)
    return DigitData(
        train_images=rng.integers(0, 256, (5000, 28, 28), dtype=np.uint8),
        train_digits=np.tile(np.arange(10, dtype=np.int64), 500),
        test_images=rng.integers(0, 256, (1200, 28, 28), dtype=np.uint8),
        test_digits=np.tile(np.arange(10, dtype=np.int64), 120),
        metadata={},
    )


def test_renderer_features_and_single_cue_geometry() -> None:
    rng = np.random.default_rng(14)
    images = rng.integers(0, 256, (3, 28, 28), dtype=np.uint8)
    b, o = np.array([2.7, -2.3, 0.8]), np.array([-2.1, 2.8, -1.4])
    nuisance = rng.standard_normal((3, 30)) * 0.2
    batch = _regional_features(images, b, o, nuisance)
    for row in range(3):
        observed = extract_canvas(
            render_digit(images[row], b[row], o[row], nuisance[row])
        )
        np.testing.assert_allclose(observed, batch[row], rtol=1e-13, atol=1e-13)
    first = extract_canvas(render_digit(images[0], 2.5, -2.5, nuisance[0]))
    partner = extract_canvas(
        render_digit(images[0], 2.5 - np.sqrt(2.0), -2.5, nuisance[0])
    )
    expected = np.zeros(96)
    expected[64] = np.sqrt(2.0)
    np.testing.assert_allclose(first - partner, expected, rtol=1e-13, atol=1e-13)


def test_content_identity_isolation_and_test_sealing() -> None:
    data = _data()
    problem = make_digits(1729, 0, data)
    assert not np.intersect1d(problem.train_ids, problem.validation_ids).size
    assert len(np.unique(problem.train_ids)) == 3000
    assert len(np.unique(problem.validation_ids)) == 1000
    np.testing.assert_array_equal(np.bincount(problem.train.labels), [1500, 1500])
    np.testing.assert_array_equal(
        np.bincount(problem.validation.groups), np.full(8, 500)
    )
    assert "test_ids" not in problem.metadata
    # Poisoned official test arrays cannot alter any fitting/selection inputs.
    changed = DigitData(
        train_images=data.train_images,
        train_digits=data.train_digits,
        test_images=np.zeros_like(data.test_images),
        test_digits=np.full_like(data.test_digits, 9),
        metadata=data.metadata,
    )
    repeated = make_digits(1729, 0, changed)
    np.testing.assert_array_equal(problem.train.features, repeated.train.features)
    np.testing.assert_array_equal(
        problem.validation.features, repeated.validation.features
    )
    np.testing.assert_allclose(problem.u.T @ problem.u, np.eye(2), atol=1e-14)
    np.testing.assert_allclose(problem.u.T @ problem.label_direction, 0, atol=1e-14)
    coordinates = problem.validation.features @ problem.rotation
    stable_columns = np.r_[np.arange(64), np.arange(66, 96)]
    for block in range(1, 4):
        np.testing.assert_allclose(
            coordinates[:1000, stable_columns],
            coordinates[1000 * block : 1000 * (block + 1), stable_columns],
            atol=1e-13,
        )
