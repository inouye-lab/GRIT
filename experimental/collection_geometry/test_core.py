"""Checks for the local data construction and equivalent convex objectives."""

from __future__ import annotations

import inspect

import numpy as np
import torch
from core import fit_basis, fit_probe, geometry_metrics, pair_design, selector

from grit.methods.training import prediction_consistency_penalty


def pair_inputs():
    rng = np.random.default_rng(431)
    k, d = 256, 12
    u = np.linalg.qr(rng.normal(size=(d, 2)))[0]
    b = rng.uniform(0.2, 3.0, size=(k, 1)) * u[:, 0]
    o = rng.uniform(0.2, 3.0, size=(k, 1)) * u[:, 1]
    return b, o, rng.random(k), rng.normal(size=(k, d)), u


def test_equal_clean_energy_and_unmodified_additive_noise() -> None:
    b, o, draws, noise, u = pair_inputs()
    for share, spectrum in (
        (0.0, (2.0, 0.0)),
        (0.0625, (1.875, 0.125)),
        (0.5, (1.0, 1.0)),
    ):
        clean, observed = pair_design(b, o, draws, noise, share, 0.3)
        np.testing.assert_allclose(np.sum(clean**2, axis=1), 2.0, atol=1e-14)
        np.testing.assert_allclose(observed - clean, 0.3 * noise, atol=1e-15)
        np.testing.assert_allclose(
            np.linalg.eigvalsh(u.T @ clean.T @ clean @ u / len(clean))[::-1],
            spectrum,
            atol=1e-14,
        )
        assert np.std(np.sum(observed**2, axis=1)) > 0.1


def test_clean_collection_span_and_missing_direction() -> None:
    b, o, draws, noise, u = pair_inputs()
    label = np.linalg.qr(np.column_stack((u, np.arange(len(u)))))[0][:, 2]
    for share in (0.0, 0.0625, 0.5):
        _, observed = pair_design(b, o, draws, noise, share, 0.0)
        basis = fit_basis(observed, 2)
        metrics = geometry_metrics(basis, u, label)
        if share == 0.0:
            assert basis.shape[1] == 1
            assert abs(metrics["missed_u"] - 1.0) < 1e-12
            assert metrics["captured_o"] < 1e-24
        else:
            assert basis.shape[1] == 2
            assert metrics["missed_u"] < 1e-12
        assert metrics["label_removed"] < 1e-24


def test_newton_solution_matches_upstream_two_logit_objective() -> None:
    rng = np.random.default_rng(72)
    x = rng.normal(size=(240, 7))
    y = (x[:, 0] + 0.4 * x[:, 1] + rng.normal(size=len(x)) > 0).astype(int)
    differences = rng.normal(size=(90, x.shape[1]))
    matrix = differences.T @ differences / len(differences)
    alpha, ridge = 1.7, 0.03
    probe = fit_probe(x, y, penalty=matrix, alpha=alpha, ridge=ridge)
    weight = torch.tensor(probe.weight, dtype=torch.float64, requires_grad=True)
    bias = torch.tensor(probe.bias, dtype=torch.float64, requires_grad=True)
    score = torch.tensor(x) @ weight + bias
    logits = torch.column_stack((-score / 2, score / 2))
    head = torch.stack((-weight / 2, weight / 2))
    upstream = prediction_consistency_penalty(head, torch.tensor(differences))
    scalar = weight @ torch.tensor(matrix) @ weight / 2
    torch.testing.assert_close(upstream, scalar)
    objective = (
        torch.nn.functional.cross_entropy(logits, torch.tensor(y))
        + ridge * weight.square().sum() / 2
        + alpha * upstream / np.trace(matrix)
    )
    objective.backward()
    assert weight.grad is not None and bias.grad is not None
    assert float(weight.grad.abs().max()) <= 1.1e-7
    assert float(bias.grad.abs()) <= 1.1e-7
    np.testing.assert_allclose(
        torch.nn.functional.cross_entropy(logits, torch.tensor(y)).detach().numpy(),
        np.mean(np.logaddexp(0, probe.scores(x)) - y * probe.scores(x)),
    )


def test_projection_is_hard_and_prediction_invariant() -> None:
    rng = np.random.default_rng(18)
    x = rng.normal(size=(300, 9))
    u = np.linalg.qr(rng.normal(size=(9, 2)))[0]
    y = (x[:, 0] + rng.normal(size=len(x)) > 0).astype(int)
    probe = fit_probe(x, y, removed=u)
    assert np.linalg.norm(u.T @ probe.weight) < 1e-12
    shifted = x + 1000 * rng.normal(size=(len(x), 2)) @ u.T
    np.testing.assert_allclose(probe.scores(shifted), probe.scores(x), atol=1e-11)
    retained = np.linalg.qr(u, mode="complete")[0][:, 2:]
    scores = probe.scores(x)
    probability = np.exp(-np.logaddexp(0, -scores))
    gradient = x.T @ (probability - y) / len(x) + 1e-4 * probe.weight
    assert np.max(np.abs(retained.T @ gradient)) <= 1.1e-7


def test_selector_uses_only_given_validation_metrics() -> None:
    # Counterfactual test scores cannot affect either validation selector.
    signature = inspect.signature(selector)
    assert tuple(signature.parameters) == ("metrics", "rule")
    first = {"worst_group": 0.75, "in_distribution": 0.81, "test_worst_group": 0.99}
    second = {"worst_group": 0.70, "in_distribution": 0.91, "test_worst_group": 0.01}
    assert selector(first, "worst_group") > selector(second, "worst_group")
    assert selector(first, "train_weighted") < selector(second, "train_weighted")
    first["test_worst_group"], second["test_worst_group"] = 0.0, 1.0
    assert selector(first, "worst_group") > selector(second, "worst_group")
    assert selector(first, "train_weighted") < selector(second, "train_weighted")
