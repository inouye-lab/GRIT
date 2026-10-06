"""Shared definitions for the collection-geometry experiments (App. G and H).

Two unstable binary factors, B and O, each either aligned with the label or not, give
eight groups (label x B-aligned x O-aligned). This module holds what the simulation
and UrbanCars share: the fixed-budget collection rule, the group
metrics and both selectors' scores, the geometry measurements, and paired verdict
statistics. It knows nothing about how features or pairs are produced.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any, Literal, Protocol, TypeAlias, cast

import numpy as np
import torch
from numpy.typing import NDArray

from grit.methods.projection import FittedLinearProjection, fit_linear_projection

FloatArray: TypeAlias = NDArray[np.float64]


class _FromNumpy(Protocol):
    def __call__(self, ndarray: NDArray[Any]) -> torch.Tensor: ...


class _ToNumpy(Protocol):
    def __call__(self, *, force: bool = False) -> NDArray[Any]: ...


_from_numpy = cast(_FromNumpy, torch.from_numpy)


def to_tensor(array: NDArray[Any]) -> torch.Tensor:
    return _from_numpy(np.ascontiguousarray(array))


def to_array(tensor: torch.Tensor) -> NDArray[Any]:
    return cast(_ToNumpy, tensor.detach().cpu().contiguous().numpy)()


# Group order within a label: (B aligned, O aligned). Index = 4 * label + position.
WITHIN_LABEL: tuple[tuple[bool, bool], ...] = (
    (True, True),
    (False, True),
    (True, False),
    (False, False),
)
GROUP_COUNT = 8

Selector: TypeAlias = Literal["worst_group", "train_weighted"]
# Primary first. Both choose configuration and checkpoint independently.
SELECTORS: tuple[Selector, ...] = ("worst_group", "train_weighted")

# Numerical rank: eigenvalues of the uncentered second moment above 1e-8 of the
# largest, i.e. singular values of the difference matrix above 1e-4 of the largest.
EIGENVALUE_TOLERANCE = 1e-8
SINGULAR_VALUE_TOLERANCE = math.sqrt(EIGENVALUE_TOLERANCE)
# Below this norm the effective classifier is treated as constant and the normalized
# residual sensitivity is undefined.
CLASSIFIER_NORM_TOLERANCE = 1e-6


def group_index(label: int, b_aligned: bool, o_aligned: bool) -> int:
    return 4 * label + WITHIN_LABEL.index((b_aligned, o_aligned))


def group_ids(
    labels: NDArray[np.int64],
    b_aligned: NDArray[np.bool_],
    o_aligned: NDArray[np.bool_],
) -> NDArray[np.int64]:
    position = np.where(b_aligned, np.where(o_aligned, 0, 2), np.where(o_aligned, 1, 3))
    return (4 * labels + position).astype(np.int64)


def training_group_weights(p_b: float, p_o: float) -> FloatArray:
    """Training probabilities of the eight groups, labels weighted equally.

    ``p_b`` and ``p_o`` are the finalized training probabilities that each cue
    matches the label, independent given the label.
    """

    within = {
        (True, True): p_b * p_o,
        (False, True): (1 - p_b) * p_o,
        (True, False): p_b * (1 - p_o),
        (False, False): (1 - p_b) * (1 - p_o),
    }
    return np.array(
        [0.5 * within[WITHIN_LABEL[g % 4]] for g in range(GROUP_COUNT)],
        dtype=np.float64,
    )


def group_accuracies(
    predictions: NDArray[np.int64], labels: NDArray[np.int64], groups: NDArray[np.int64]
) -> list[float]:
    result: list[float] = []
    for group in range(GROUP_COUNT):
        mask = groups == group
        if not bool(mask.any()):
            raise ValueError(f"evaluation split has no example of group {group}")
        result.append(float((predictions[mask] == labels[mask]).mean()))
    return result


def _label_mean(accuracies: Sequence[float], position: int) -> float:
    return 0.5 * (accuracies[position] + accuracies[4 + position])


def group_metrics(accuracies: Sequence[float], weights: FloatArray) -> dict[str, float]:
    """Primary metrics and both selector scores from eight group accuracies.

    In-distribution accuracy weights the groups by their training probabilities,
    which is also the secondary selector's score. Shift accuracies average the two
    labels' groups equally. Gaps are secondary and labelled as such.
    """

    if len(accuracies) != GROUP_COUNT:
        raise ValueError("group metrics need eight group accuracies")
    in_distribution = float(np.dot(weights, np.asarray(accuracies)))
    b_shift = _label_mean(accuracies, 1)
    o_shift = _label_mean(accuracies, 2)
    both_shift = _label_mean(accuracies, 3)
    return {
        "in_distribution": in_distribution,
        "b_shift": b_shift,
        "o_shift": o_shift,
        "both_shift": both_shift,
        "worst_group": min(accuracies),
        "aligned": _label_mean(accuracies, 0),
        "gap_b_shift": b_shift - in_distribution,
        "gap_o_shift": o_shift - in_distribution,
        "gap_both_shift": both_shift - in_distribution,
    }


def selector_score(metrics: dict[str, float], selector: Selector) -> float:
    return (
        metrics["worst_group"]
        if selector == "worst_group"
        else metrics["in_distribution"]
    )


# Collections at a fixed budget.


def minority_count(k: int, share: float, energy_b: float, energy_o: float) -> int:
    """n_O for O's energy share ``share`` of a k-pair collection.

    ``energy_b`` and ``energy_o`` are mean per-pair energies from a calibration set
    disjoint from the banks. n_O = 0 at share > 0 means the level is unreachable.
    """

    if not 0.0 <= share <= 1.0:
        raise ValueError("energy share must lie in [0, 1]")
    if energy_b <= 0.0 or energy_o <= 0.0:
        raise ValueError("calibration energies must be positive")
    denominator = share * energy_b + (1.0 - share) * energy_o
    return int(round(k * share * energy_b / denominator))


def o_positions(draws: FloatArray, k: int, n_o: int) -> NDArray[np.bool_]:
    """Coupled construction: among the first k positions, the n_O smallest draws.

    True marks a position that uses its O-partner; the rest use their B-partner.
    """

    if not 0 <= n_o <= k <= len(draws):
        raise ValueError("need 0 <= n_O <= k <= bank size")
    chosen = np.zeros(k, dtype=np.bool_)
    chosen[np.argsort(draws[:k], kind="stable")[:n_o]] = True
    return chosen


def realized_share(differences: FloatArray, is_o: NDArray[np.bool_]) -> float:
    energy = np.square(differences).sum(axis=1)
    total = float(energy.sum())
    return float(energy[is_o].sum()) / total if total > 0 else 0.0


# Projections and geometry.


def fit_projection(
    left: FloatArray, right: FloatArray, requested_rank: int, *, digest: str
) -> FittedLinearProjection:
    """The shared GRIT projection, capped at the numerical rank of 0.4."""

    maximum = min(left.shape)
    return fit_linear_projection(
        to_tensor(left),
        to_tensor(right),
        requested_rank=min(requested_rank, maximum),
        pair_manifest_digest=digest,
        feature_cache_manifest_digest=digest,
        relative_singular_value_tolerance=SINGULAR_VALUE_TOLERANCE,
    )


def subspace_projection(basis: FloatArray, *, digest: str) -> FittedLinearProjection:
    """A fitted projection removing exactly span(basis), for reference rows."""

    rows = np.ascontiguousarray(basis.T)
    return fit_projection(rows, np.zeros_like(rows), rows.shape[0], digest=digest)


def removed_basis(projection: FittedLinearProjection) -> FloatArray:
    return to_array(projection.basis).astype(np.float64)


def captured_energy(basis: FloatArray, second_moment: FloatArray) -> float:
    """tr(P Sigma_F) / tr(Sigma_F) for the removed subspace spanned by ``basis``."""

    total = float(np.trace(second_moment))
    if basis.shape[1] == 0 or total == 0.0:
        return 0.0
    return float(np.trace(basis.T @ second_moment @ basis)) / total


def label_energy(basis: FloatArray, direction: FloatArray) -> float:
    """Fraction of the balanced class-mean difference lying in the removed subspace."""

    norm = float(direction @ direction)
    if basis.shape[1] == 0 or norm == 0.0:
        return 0.0
    coordinates = basis.T @ direction
    return float(coordinates @ coordinates) / norm


def top_eigenvectors(differences: FloatArray, rank: int) -> FloatArray:
    """V-hat_r of the uncentered pair second moment, capped at its numerical rank."""

    _, values, right = np.linalg.svd(differences, full_matrices=False)
    numerical = int((values > values[0] * SINGULAR_VALUE_TOLERANCE).sum())
    return right[: min(rank, numerical)].T


def classifier_contrast(weight: torch.Tensor) -> FloatArray:
    """The logit contrast w_1 - w_0 that a two-class softmax probe actually uses."""

    rows: FloatArray = to_array(weight.to(torch.float64))
    if rows.shape[0] != 2:
        raise ValueError("residual sensitivity is defined for two-class probes")
    return rows[1] - rows[0]


def residual_sensitivity(
    contrast: FloatArray,
    *,
    removed: FloatArray | None,
    factor_bases: dict[str, FloatArray],
    estimated: dict[int, FloatArray],
) -> dict[str, float | None]:
    """0.3: norm of W_eff, normalized factor sensitivity, and enforcement residuals.

    ``removed`` is the projected-input model's removed basis (None otherwise), so
    W_eff = W (I - P). ``estimated`` maps a rank r to V-hat_r of the collection.
    """

    effective = (
        contrast if removed is None else contrast - (contrast @ removed) @ removed.T
    )
    norm = float(np.linalg.norm(effective))
    result: dict[str, float | None] = {"w_eff_norm": norm}
    for name, basis in factor_bases.items():
        result[f"sensitivity_{name}"] = (
            None
            if norm < CLASSIFIER_NORM_TOLERANCE
            else float(np.linalg.norm(effective @ basis)) / norm
        )
    for rank, basis in estimated.items():
        result[f"enforcement_r{rank}"] = float(np.linalg.norm(effective @ basis))
    return result


# Paired contrast statistics.

_T_975 = {9: 2.2621571627409915, 19: 2.093024054408263, 29: 2.045229642132703}


def paired_interval(values: Sequence[float]) -> dict[str, float | int | None]:
    """Mean, SD, 95% t interval and a two-sided exact sign-test p over the pairs."""

    n = len(values)
    if n < 2:
        raise ValueError("a paired interval needs at least two pairs")
    mean = float(np.mean(values))
    sd = float(np.std(values, ddof=1))
    t = _T_975.get(n - 1)
    half = None if t is None else t * sd / math.sqrt(n)
    positive = sum(v > 0 for v in values)
    negative = sum(v < 0 for v in values)
    trials = positive + negative
    tail = min(positive, negative)
    sign_p = (
        1.0
        if trials == 0
        else min(
            1.0, 2 * sum(math.comb(trials, i) for i in range(tail + 1)) / 2**trials
        )
    )
    return {
        "n": n,
        "mean": mean,
        "sd": sd,
        "ci_low": None if half is None else mean - half,
        "ci_high": None if half is None else mean + half,
        "positive": positive,
        "negative": negative,
        "sign_p": sign_p,
    }
