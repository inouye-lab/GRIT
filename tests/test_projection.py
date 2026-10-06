"""Numerical contract tests for the CMNIST linear nuisance projection."""

from __future__ import annotations

import pytest
import torch

from grit.methods.projection import (
    FittedLinearProjection,
    ProjectionDiagnostics,
    fit_linear_projection,
)

_PAIR_MANIFEST_DIGEST = "sha256:test-pairs"
_FEATURE_CACHE_MANIFEST_DIGEST = "sha256:test-features"


def _fit(
    left: torch.Tensor,
    right: torch.Tensor,
    *,
    requested_rank: int,
    relative_singular_value_tolerance: float = 1e-12,
) -> FittedLinearProjection:
    return fit_linear_projection(
        left,
        right,
        requested_rank=requested_rank,
        pair_manifest_digest=_PAIR_MANIFEST_DIGEST,
        feature_cache_manifest_digest=_FEATURE_CACHE_MANIFEST_DIGEST,
        relative_singular_value_tolerance=relative_singular_value_tolerance,
    )


def _pair_features() -> tuple[torch.Tensor, torch.Tensor]:
    left = torch.tensor(
        [
            [2.0, 1.0, 0.0, 1.0],
            [3.0, 0.0, 1.0, 2.0],
            [4.0, 1.0, 1.0, 0.0],
        ],
        dtype=torch.float32,
    )
    right = left - torch.tensor(
        [
            [1.0, 0.0, 0.0, 0.0],
            [2.0, 0.0, 0.0, 0.0],
            [3.0, 0.0, 0.0, 0.0],
        ]
    )
    return left, right


def test_projection_is_orthogonal_idempotent_and_annihilates_nuisance() -> None:
    left, right = _pair_features()
    fitted = _fit(left, right, requested_rank=1)
    projector = fitted.projector()
    identity = torch.eye(4, dtype=torch.float64)
    assert torch.allclose(projector, projector.T, atol=1e-14, rtol=0.0)
    assert torch.allclose(projector @ projector, projector, atol=1e-14, rtol=0.0)
    assert torch.allclose(
        fitted.basis.T @ fitted.basis,
        torch.eye(1, dtype=torch.float64),
        atol=1e-14,
        rtol=0.0,
    )
    differences = left.to(torch.float64) - right.to(torch.float64)
    assert torch.allclose(differences @ projector, torch.zeros_like(differences))
    assert not torch.equal(projector, identity)


def test_projection_subspace_is_orientation_invariant_and_deterministic() -> None:
    left, right = _pair_features()
    forward = _fit(left, right, requested_rank=1)
    reverse = _fit(right, left, requested_rank=1)
    repeated = _fit(left, right, requested_rank=1)
    assert torch.allclose(
        forward.projector(), reverse.projector(), atol=1e-14, rtol=0.0
    )
    assert torch.equal(forward.projector(), repeated.projector())
    assert forward.diagnostics == repeated.diagnostics


def test_rank_zero_is_exact_shape_dtype_and_device_preserving_identity() -> None:
    left, right = _pair_features()
    fitted = _fit(left, right, requested_rank=0)
    batch = torch.arange(24, dtype=torch.float32).reshape(2, 3, 4)
    transformed = fitted.transform(batch)
    assert torch.equal(transformed, batch)
    assert transformed.shape == batch.shape
    assert transformed.dtype == batch.dtype
    assert transformed.device == batch.device
    assert fitted.basis.shape == (4, 0)
    assert fitted.diagnostics.requested_rank == 0
    assert fitted.diagnostics.effective_rank == 0


def test_projection_records_complete_spectrum_and_numerical_rank() -> None:
    left, right = _pair_features()
    fitted = _fit(
        left,
        right,
        requested_rank=2,
        relative_singular_value_tolerance=1e-10,
    )
    assert fitted.diagnostics.requested_rank == 2
    assert fitted.diagnostics.numerical_rank == 1
    assert fitted.diagnostics.effective_rank == 1
    assert len(fitted.diagnostics.singular_values) == 3
    assert fitted.diagnostics.operation == "uncentered_left_minus_right"
    assert fitted.diagnostics.fitting_dtype == "torch.float64"
    assert fitted.diagnostics.pair_manifest_digest == _PAIR_MANIFEST_DIGEST
    assert (
        fitted.diagnostics.feature_cache_manifest_digest
        == _FEATURE_CACHE_MANIFEST_DIGEST
    )
    assert (
        ProjectionDiagnostics.model_validate_json(fitted.diagnostics.canonical_json())
        == fitted.diagnostics
    )


def test_projection_uses_uncentered_pair_differences() -> None:
    left = torch.tensor([[2.0, 0.0], [2.0, 1.0]], dtype=torch.float64)
    right = torch.zeros_like(left)
    fitted = _fit(left, right, requested_rank=1)
    # Centering would remove the constant first coordinate and select the second axis.
    assert abs(float(fitted.basis[0, 0])) > abs(float(fitted.basis[1, 0]))
    assert fitted.diagnostics.operation == "uncentered_left_minus_right"


@pytest.mark.parametrize("rank", (-1, 4))
def test_projection_rejects_impossible_ranks(rank: int) -> None:
    left, right = _pair_features()
    with pytest.raises(ValueError, match="requested_rank"):
        _fit(left, right, requested_rank=rank)


def test_projection_rejects_invalid_shape_dtype_and_nonfinite_values() -> None:
    left, right = _pair_features()
    with pytest.raises(ValueError, match="share shape"):
        _fit(left, right[:, :3], requested_rank=1)
    invalid = left.clone()
    invalid[0, 0] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        _fit(invalid, right, requested_rank=1)
    fitted = _fit(left, right, requested_rank=1)
    with pytest.raises(TypeError, match="floating-point"):
        fitted.transform(torch.ones((2, 4), dtype=torch.int64))
    with pytest.raises(ValueError, match="feature dimension"):
        fitted.transform(torch.ones((2, 3), dtype=torch.float32))
