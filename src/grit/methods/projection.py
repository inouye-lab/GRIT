"""Classifier-independent linear nuisance projection for CMNIST GRIT."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Annotated, Literal, Protocol, TypeAlias, cast

import torch
from pydantic import Field, FiniteFloat, StrictInt, StrictStr

from grit.schemas import StrictBoundaryModel

NonNegativeInt: TypeAlias = Annotated[StrictInt, Field(ge=0)]
NonNegativeFloat: TypeAlias = Annotated[FiniteFloat, Field(ge=0.0)]
PositiveFloat: TypeAlias = Annotated[FiniteFloat, Field(gt=0.0)]
NonEmptyStr: TypeAlias = Annotated[StrictStr, Field(min_length=1)]


class _SvdOperation(Protocol):
    def __call__(
        self,
        inputs: torch.Tensor,
        full_matrices: bool = True,
        *,
        driver: str | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]: ...


_full_svd = cast(_SvdOperation, torch.linalg.svd)


class ProjectionDiagnostics(StrictBoundaryModel):
    schema_version: Literal["grit.linear-projection/v2"]
    operation: Literal["uncentered_left_minus_right"]
    backend: Literal["torch.linalg.svd"]
    fitting_device: Literal["cpu"]
    fitting_dtype: Literal["torch.float64"]
    pair_manifest_digest: NonEmptyStr
    feature_cache_manifest_digest: NonEmptyStr
    pair_count: NonNegativeInt
    feature_dimension: NonNegativeInt
    requested_rank: NonNegativeInt
    numerical_rank: NonNegativeInt
    effective_rank: NonNegativeInt
    relative_singular_value_tolerance: PositiveFloat
    absolute_singular_value_threshold: NonNegativeFloat
    singular_values: tuple[NonNegativeFloat, ...]
    removed_energy_fraction: Annotated[FiniteFloat, Field(ge=0.0, le=1.0)]
    orthonormality_residual: NonNegativeFloat
    symmetry_residual: NonNegativeFloat
    idempotence_residual: NonNegativeFloat


@dataclass(frozen=True, slots=True)
class FittedLinearProjection:
    """A fitted CPU-float64 nuisance basis with explicit runtime conversion."""

    basis: torch.Tensor
    diagnostics: ProjectionDiagnostics

    def transform(self, inputs: torch.Tensor) -> torch.Tensor:
        """Remove the fitted subspace while preserving shape, dtype, and device."""

        if not inputs.is_floating_point():
            raise TypeError("projection inputs must be floating-point tensors")
        if inputs.dtype not in {torch.float32, torch.float64}:
            raise TypeError("projection inputs must use float32 or float64")
        if (
            inputs.ndim == 0
            or int(inputs.shape[-1]) != self.diagnostics.feature_dimension
        ):
            raise ValueError(
                "projection input feature dimension does not match the fitted object"
            )
        if not bool(torch.isfinite(inputs).all()):
            raise ValueError("projection inputs must be finite")
        if self.diagnostics.effective_rank == 0:
            return inputs.clone()
        basis = self.basis.to(device=inputs.device, dtype=inputs.dtype)
        return inputs - (inputs @ basis) @ basis.transpose(0, 1)

    def projector(self) -> torch.Tensor:
        """Return the CPU-float64 orthogonal complement projector for diagnostics."""

        dimension = self.diagnostics.feature_dimension
        identity = torch.eye(dimension, dtype=torch.float64)
        if self.diagnostics.effective_rank == 0:
            return identity
        return identity - self.basis @ self.basis.transpose(0, 1)


def fit_linear_projection(
    left: torch.Tensor,
    right: torch.Tensor,
    *,
    requested_rank: int,
    pair_manifest_digest: str,
    feature_cache_manifest_digest: str,
    relative_singular_value_tolerance: float = 1e-12,
) -> FittedLinearProjection:
    """Fit the approved deterministic full SVD to uncentered pair differences."""

    if left.ndim != 2 or right.ndim != 2 or left.shape != right.shape:
        raise ValueError("pair endpoint features must share shape [pairs, features]")
    pair_count, feature_dimension = (int(value) for value in left.shape)
    if pair_count == 0 or feature_dimension == 0:
        raise ValueError("pair endpoint feature matrices must be non-empty")
    maximum_rank = min(pair_count, feature_dimension)
    if requested_rank < 0 or requested_rank > maximum_rank:
        raise ValueError(
            f"requested_rank must lie in [0, {maximum_rank}], got {requested_rank}"
        )
    if (
        not torch.isfinite(torch.tensor(relative_singular_value_tolerance))
        or relative_singular_value_tolerance <= 0.0
    ):
        raise ValueError(
            "relative singular-value tolerance must be finite and positive"
        )
    differences = left.detach().to(
        device="cpu", dtype=torch.float64
    ) - right.detach().to(device="cpu", dtype=torch.float64)
    if not bool(torch.isfinite(differences).all()):
        raise ValueError("pair endpoint features must be finite")

    _, singular_values, right_vectors_h = _full_svd(
        differences,
        full_matrices=False,
    )
    largest = float(singular_values[0].item())
    absolute_threshold = largest * relative_singular_value_tolerance
    numerical_rank = int((singular_values > absolute_threshold).sum().item())
    effective_rank = min(requested_rank, numerical_rank)
    basis = right_vectors_h[:effective_rank].transpose(0, 1).contiguous()

    energy = singular_values.square()
    total_energy = float(energy.sum().item())
    removed_energy = float(energy[:effective_rank].sum().item())
    removed_fraction = 0.0 if total_energy == 0.0 else removed_energy / total_energy
    orthonormality, symmetry, idempotence = _projection_residuals(
        basis,
        feature_dimension,
    )
    diagnostics = ProjectionDiagnostics(
        schema_version="grit.linear-projection/v2",
        operation="uncentered_left_minus_right",
        backend="torch.linalg.svd",
        fitting_device="cpu",
        fitting_dtype="torch.float64",
        pair_manifest_digest=pair_manifest_digest,
        feature_cache_manifest_digest=feature_cache_manifest_digest,
        pair_count=pair_count,
        feature_dimension=feature_dimension,
        requested_rank=requested_rank,
        numerical_rank=numerical_rank,
        effective_rank=effective_rank,
        relative_singular_value_tolerance=relative_singular_value_tolerance,
        absolute_singular_value_threshold=absolute_threshold,
        singular_values=tuple(float(value) for value in torch.unbind(singular_values)),
        removed_energy_fraction=removed_fraction,
        orthonormality_residual=orthonormality,
        symmetry_residual=symmetry,
        idempotence_residual=idempotence,
    )
    return FittedLinearProjection(basis=basis, diagnostics=diagnostics)


def _projection_residuals(
    basis: torch.Tensor,
    feature_dimension: int,
) -> tuple[float, float, float]:
    identity = torch.eye(feature_dimension, dtype=torch.float64)
    rank = int(basis.shape[1])
    orthonormality = 0.0
    if rank:
        orthonormality = _frobenius_norm(
            basis.transpose(0, 1) @ basis
            - torch.eye(rank, dtype=torch.float64)
        )
    projector = identity if rank == 0 else identity - basis @ basis.transpose(0, 1)
    symmetry = _frobenius_norm(projector - projector.transpose(0, 1))
    idempotence = _frobenius_norm(projector @ projector - projector)
    return orthonormality, symmetry, idempotence


def _frobenius_norm(values: torch.Tensor) -> float:
    return float(torch.sqrt(values.square().sum()).item())
