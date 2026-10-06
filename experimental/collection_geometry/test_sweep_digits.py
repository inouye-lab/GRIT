"""Realizable clean cue differences for the independent signal-strength sweep.

The runner assigns analytic feature differences to training anchors. These checks
construct float-rendered clean endpoints as a realizability witness; they do not
assert that the runner pre-renders pairs or that its Gaussian noise is pixel noise.
"""

from __future__ import annotations

import numpy as np
import pytest
from digits import DigitData, DigitProblem, extract_canvas, make_digits, render_digit
from sweeps import signal_design


@pytest.fixture(scope="module")
def digit_problem() -> tuple[DigitData, DigitProblem]:
    """Use generated digit-shaped arrays; no dataset files or download needed."""
    rng = np.random.default_rng(37)
    data = DigitData(
        train_images=rng.integers(0, 256, (5000, 28, 28), dtype=np.uint8),
        train_digits=np.tile(np.arange(10, dtype=np.int64), 500),
        test_images=rng.integers(0, 256, (1200, 28, 28), dtype=np.uint8),
        test_digits=np.tile(np.arange(10, dtype=np.int64), 120),
        metadata={},
    )
    return data, make_digits(20260927, 0, data)


@pytest.mark.parametrize("g", [0.0, 1 / 64, 1 / 8, 0.5, 7 / 8, 1.0])
def test_signal_design_has_renderable_digit_partners(
    digit_problem: tuple[DigitData, DigitProblem], g: float
) -> None:
    data, problem = digit_problem
    # Four training identities from both classes, each used once for B and once
    # for O, with nonconsecutive/repeated anchor positions in the pair collection.
    rows = np.concatenate(
        [np.flatnonzero(problem.train.labels == label)[:2] for label in (0, 1)]
    )
    anchor_rows = rows[[0, 1, 2, 3, 1, 0, 3, 2]]
    clean = signal_design(problem.u, len(anchor_rows), g)
    coordinates = problem.train.features @ problem.rotation

    for position, row in enumerate(anchor_rows):
        identity = int(problem.train_ids[row])
        assert identity not in problem.validation_ids
        original_image = data.train_images[identity]
        b, o = float(problem.train.b[row, 0]), float(problem.train.o[row, 0])
        nuisance = coordinates[row, 66:]
        left = render_digit(original_image, b, o, nuisance)
        b_partner = b - np.sqrt(2.0) if position % 2 == 0 else b
        o_partner = o if position % 2 == 0 else o - np.sqrt(2.0 * g)
        right = render_digit(original_image, b_partner, o_partner, nuisance)

        # The digit and nuisance patches are literally unchanged. The other cue
        # is also fixed, including at g=0 when an O-pair is an identical endpoint.
        np.testing.assert_array_equal(left[6:34, 10:38], right[6:34, 10:38])
        np.testing.assert_array_equal(left[38:40, :30], right[38:40, :30])
        unchanged_cue = (
            (slice(0, 4), slice(8, 40))
            if position % 2 == 0
            else (slice(4, 6), slice(8, 40))
        )
        np.testing.assert_array_equal(left[unchanged_cue], right[unchanged_cue])
        if position % 2 and g == 0:
            np.testing.assert_array_equal(left, right)

        endpoints = np.stack([extract_canvas(left), extract_canvas(right)])
        endpoints[:, :64] = (endpoints[:, :64] - problem.center) * problem.scale
        embedded = endpoints @ problem.rotation.T
        np.testing.assert_allclose(
            embedded[0], problem.train.features[row], rtol=1e-12, atol=1e-12
        )
        np.testing.assert_allclose(
            embedded[0] - embedded[1], clean[position], rtol=1e-12, atol=1e-12
        )

    assert "test_ids" not in problem.metadata
