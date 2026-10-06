"""Small deterministic fits; existing GRIT projection and group conventions."""

# NumPy linear-algebra/transpose stubs lose known float64 array types.
# pyright: reportUnknownMemberType=false, reportUnknownArgumentType=false, reportUnknownVariableType=false
from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np

from grit.data.collection_geometry import Split
from grit.geometry import FloatArray, group_metrics, to_array, to_tensor
from grit.methods.projection import fit_linear_projection


def digest_array(x: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest()


@dataclass
class Probe:
    weight: FloatArray
    bias: float
    gradient: float
    iterations: int

    def scores(self, x: FloatArray) -> FloatArray:
        return x @ self.weight + self.bias


def fit_probe(
    x: FloatArray,
    y: np.ndarray,
    *,
    removed: FloatArray | None = None,
    penalty: FloatArray | None = None,
    alpha: float = 0.0,
    ridge: float = 1e-4,
) -> Probe:
    """Minimize BCE + ridge/2 ||w||² + alpha/(2 tr(M)) w'Mw.

    The CR coefficient matches upstream two-logit consistency at symmetric logits
    (-f/2,f/2). Hard projection is a null-space parameterization, not a large penalty.
    Bias is unpenalized. Newton steps use Armijo backtracking and checked convergence.
    """
    n, d = x.shape
    if removed is None or removed.shape[1] == 0:
        retained = np.eye(d)
    else:
        retained = np.linalg.qr(removed, mode="complete")[0][:, removed.shape[1] :]
    z = np.column_stack((x @ retained, np.ones(n)))
    a = np.zeros((z.shape[1], z.shape[1]))
    a[:-1, :-1] = ridge * np.eye(retained.shape[1])
    if penalty is not None and alpha:
        a[:-1, :-1] += alpha * (retained.T @ penalty @ retained) / np.trace(penalty)
    beta = np.zeros(z.shape[1])

    def objective(value: FloatArray) -> float:
        score = z @ value
        return float(
            np.mean(np.logaddexp(0, score) - y * score) + value @ a @ value / 2
        )

    gradient = float("inf")
    iterations = 0
    for iteration in range(80):
        iterations = iteration + 1
        score = z @ beta
        probability = np.exp(-np.logaddexp(0, -score))
        grad = z.T @ (probability - y) / n + a @ beta
        gradient = float(np.max(np.abs(grad)))
        if gradient <= 1e-7:
            break
        hessian = (z.T * (probability * (1 - probability))) @ z / n + a
        step = np.linalg.solve(hessian, grad)
        before = objective(beta)
        scale = 1.0
        while objective(beta - scale * step) > before - 1e-4 * scale * (grad @ step):
            scale /= 2
            if scale < 1e-12:
                raise RuntimeError("Newton line search failed")
        beta -= scale * step
    if gradient > 1e-7:
        raise RuntimeError(f"Unconverged fit: gradient={gradient}")
    return Probe(retained @ beta[:-1], float(beta[-1]), gradient, iterations)


def fit_basis(differences: FloatArray, rank: int) -> FloatArray:
    tensor = to_tensor(differences)
    projection = fit_linear_projection(
        tensor,
        tensor * 0,
        requested_rank=rank,
        pair_manifest_digest=digest_array(differences),
        feature_cache_manifest_digest="local-known-geometry",
        relative_singular_value_tolerance=1e-4,
    )
    return to_array(projection.basis)


def pair_design(
    delta_b: FloatArray,
    delta_o: FloatArray,
    draws: FloatArray,
    noise: FloatArray,
    share: float,
    sigma: float,
) -> tuple[FloatArray, FloatArray]:
    """Normalize CLEAN differences, assign nested positions, then add noise."""
    k = len(draws)
    is_o = np.zeros(k, dtype=bool)
    is_o[np.argsort(draws, kind="stable")[: round(k * share)]] = True
    clean = np.where(is_o[:, None], delta_o, delta_b).copy()
    clean *= np.sqrt(2) / np.linalg.norm(clean, axis=1, keepdims=True)
    return clean, clean + sigma * noise


def evaluate(probe: Probe, split: Split, weights: FloatArray) -> dict[str, float]:
    scores = probe.scores(split.features)
    losses = np.logaddexp(0, scores) - split.labels * scores
    correct = (scores >= 0) == split.labels
    accuracies = [float(correct[split.groups == g].mean()) for g in range(8)]
    metrics = group_metrics(accuracies, weights)
    metrics["loss"] = float(losses.mean())
    metrics["both_loss"] = float(losses[np.isin(split.groups, [3, 7])].mean())
    metrics["source_loss"] = float(
        sum(weights[g] * losses[split.groups == g].mean() for g in range(8))
    )
    return metrics


def geometry_metrics(
    basis: FloatArray, u: FloatArray, label: FloatArray
) -> dict[str, float]:
    missed = u - basis @ (basis.T @ u)
    return {
        "missed_u": float(np.linalg.norm(missed, ord=2)),
        "captured_b": float(np.linalg.norm(basis.T @ u[:, 0]) ** 2),
        "captured_o": float(np.linalg.norm(basis.T @ u[:, 1]) ** 2),
        "label_removed": float(np.linalg.norm(basis.T @ label) ** 2 / (label @ label)),
        "effective_rank": int(basis.shape[1]),
    }


def selector(metrics: dict[str, float], rule: str) -> float:
    return metrics["worst_group" if rule == "worst_group" else "in_distribution"]
