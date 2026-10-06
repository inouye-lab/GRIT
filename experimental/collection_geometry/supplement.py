"""Postprocess frozen fits into exact synthetic population-oracle diagnostics.

No new fitting, selection, or test evaluation occurs here. The analytic oracle is
the unregularized population stable logistic optimum, distinct from both the
training label-direction proxy and every finite-label ridge-logistic fitted head.
"""

# NumPy linear-algebra stubs lose known array types.
# pyright: reportUnknownMemberType=false, reportUnknownArgumentType=false, reportUnknownVariableType=false
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
from core import digest_array, fit_basis, pair_design

from grit.data.collection_geometry import (
    Geometry,
    SimulationSpec,
    make_bank,
    make_geometry,
    training_split,
)


def _population(geometry: Geometry, spec: SimulationSpec) -> dict[str, Any]:
    """Compute population moments and independently sum the binary cue mixture."""
    if spec.overlap_degrees != 0:
        raise ValueError("The analytic stable Gaussian oracle requires no overlap")
    u = geometry.reference_basis()
    retained = np.linalg.qr(u, mode="complete")[0][:, u.shape[1] :]
    mu = geometry.stable @ (
        spec.stable_mean * np.ones(spec.stable_rank) / np.sqrt(spec.stable_rank)
    )
    mb, mo = geometry.u_b @ geometry.m_b, geometry.u_o @ geometry.m_o
    noise = (geometry.noise_basis * geometry.noise_sd**2) @ geometry.noise_basis.T
    within = (
        geometry.stable @ geometry.stable.T
        + spec.b_spread**2 * (geometry.u_b @ geometry.u_b.T)
        + spec.o_spread**2 * (geometry.u_o @ geometry.u_o.T)
        + noise
    )
    ab, ao = 2 * spec.p_b - 1, 2 * spec.p_o - 1
    conditional_mean = mu + ab * mb + ao * mo
    second = (
        within
        + np.outer(conditional_mean, conditional_mean)
        + (1 - ab**2) * np.outer(mb, mb)
        + (1 - ao**2) * np.outer(mo, mo)
    )
    # Independent derivation: average all eight (Y, q_B, q_O) component means.
    mixture_second = within.copy()
    for y in (-1, 1):
        for qb in (-1, 1):
            for qo in (-1, 1):
                probability = (
                    0.5
                    * (spec.p_b if qb == y else 1 - spec.p_b)
                    * (spec.p_o if qo == y else 1 - spec.p_o)
                )
                component = y * mu + qb * mb + qo * mo
                mixture_second += probability * np.outer(component, component)
    moment_residual = float(np.max(np.abs(second - mixture_second)))
    if moment_residual > 1e-12:
        raise AssertionError("Independent source-second-moment formulas disagree")
    stable_covariance = retained.T @ within @ retained
    stable_mean = retained.T @ mu
    inverse_mean = np.linalg.solve(stable_covariance, stable_mean)
    oracle = 2 * retained @ inverse_mean
    information = float(stable_mean @ inverse_mean)

    def quadrature(order: int) -> float:
        nodes, weights = np.polynomial.hermite.hermgauss(order)
        signed_score = 2 * information + np.sqrt(8 * information) * nodes
        return float(weights @ np.logaddexp(0, -signed_score) / np.sqrt(np.pi))

    risk, check_risk = quadrature(80), quadrature(120)
    if abs(risk - check_risk) > 1e-9:
        raise AssertionError("Stable-oracle risk quadrature has not converged")
    if np.linalg.norm(u.T @ oracle) > 1e-10:
        raise AssertionError("The population stable oracle leaves the stable space")
    return {
        "oracle": oracle,
        "second_moment": second,
        "L_z": float(np.sqrt(np.linalg.eigvalsh(second)[-1])),
        "stable_oracle_risk": risk,
        "oracle_norm": float(np.linalg.norm(oracle)),
        "information": information,
        "moment_formula_residual": moment_residual,
        "quadrature_residual": abs(risk - check_risk),
    }


def make_supplement(output: Path) -> None:
    """Write theory-components.json and appendix-theory-table.tex from frozen fits."""
    config = json.loads((output / "config.json").read_text())
    results = json.loads((output / "results.json").read_text())
    sigma = config["primary_sigma"]
    rows: list[dict[str, Any]] = []
    population_rows: list[dict[str, Any]] = []
    sensitivity_checks = 0
    for result in results:
        if result["dataset"] != "synthetic":
            continue
        replicate = result["replicate"]
        seed = config["seed"]
        spec = SimulationSpec.model_validate(result["input_metadata"]["spec"])
        geometry = make_geometry(spec, seed, replicate)
        train = training_split(spec, geometry, seed, replicate)
        bank = make_bank(spec, geometry, train, seed, replicate)
        noise = np.random.default_rng([seed, replicate, 77]).standard_normal(
            (spec.bank_size, spec.dimension)
        )
        _, observed = pair_design(
            bank.delta_b, bank.delta_o, bank.draws, noise, 0.5, sigma
        )
        records = result["records"]
        if digest_array(observed) != records[f"balanced:s{sigma}:r2"]["pair_hash"]:
            raise AssertionError("Regenerated pair inputs differ from the frozen fit")
        basis = fit_basis(observed, max(config["ranks"]))
        population = _population(geometry, spec)
        oracle = population["oracle"]
        population_rows.append(
            {
                "replicate": replicate,
                **{
                    key: value
                    for key, value in population.items()
                    if key not in {"oracle", "second_moment"}
                },
            }
        )
        for record in records.values():
            if record["method"] not in {"grit", "grit_weak_ridge"}:
                continue
            ceiling = record["missed_u"] * record["coefficient_norm"]
            if record["sensitivity"] > ceiling + 1e-10:
                raise AssertionError(
                    "Hard-projection residual sensitivity exceeds geometry bound"
                )
            sensitivity_checks += 1
        with np.load(output / f"synthetic-{replicate:02d}-weights.npz") as weights:
            for rank in config["ranks"]:
                key = f"balanced:s{sigma}:r{rank}"
                record = records[key]
                fitted = weights[key]
                if digest_array(fitted) != record["weight_hash"]:
                    raise AssertionError(
                        "Saved probe differs from the frozen weight hash"
                    )
                removed = basis[:, :rank]
                actual_sensitivity = float(
                    np.linalg.norm(geometry.reference_basis().T @ fitted[:-1])
                )
                if abs(actual_sensitivity - record["sensitivity"]) > 1e-12:
                    raise AssertionError(
                        "Saved probe sensitivity differs from its record"
                    )
                extra = basis[:, 2:rank] if rank >= 2 else None
                extra_vector = None if extra is None else extra @ (extra.T @ oracle)
                extra_norm = (
                    None
                    if extra_vector is None
                    else float(np.linalg.norm(extra_vector))
                )
                rows.append(
                    {
                        "replicate": replicate,
                        "rank": rank,
                        "missed_u": record["missed_u"],
                        "sensitivity": actual_sensitivity,
                        "coefficient_norm": record["coefficient_norm"],
                        "geometry_times_norm": record["missed_u"]
                        * record["coefficient_norm"],
                        "stable_label_direction_removed_fraction": record[
                            "label_removed"
                        ],
                        "population_oracle_removed_norm": float(
                            np.linalg.norm(removed.T @ oracle)
                        ),
                        "extra_oracle_removed_norm": extra_norm,
                        "extra_rank_term": None
                        if extra_norm is None
                        else population["L_z"] * extra_norm,
                        "extra_removal_source_l2": None
                        if extra_vector is None
                        else float(
                            np.sqrt(
                                max(
                                    0,
                                    extra_vector
                                    @ population["second_moment"]
                                    @ extra_vector,
                                )
                            )
                        ),
                        "stable_oracle_risk": population["stable_oracle_risk"],
                        "L_z": population["L_z"],
                    }
                )
    record = {
        "scope": (
            "Postprocessing only; exact analytic population stable oracle, "
            "distinct from fitted empirical ridge heads. Extra-rank term is "
            "defined only for r>=r*=2. No full finite-label risk certificate "
            "is asserted."
        ),
        "oracle_formula": (
            "w*_S=2 V(V' C_within V)^(-1)V'mu, where V spans U-perp; bias=0"
        ),
        "risk_formula": (
            "a=mu_S' C_S^(-1) mu_S; Y w*_S'Z ~ N(2a,4a); "
            "80-node Gauss-Hermite logistic risk checked against 120 nodes"
        ),
        "input_hashes": {
            name: hashlib.sha256((output / name).read_bytes()).hexdigest()
            for name in ("config.json", "results.json")
        },
        "supplement_source_sha256": hashlib.sha256(
            Path(__file__).read_bytes()
        ).hexdigest(),
        "sensitivity_inequality_checks": sensitivity_checks,
        "population": population_rows,
        "rows": rows,
    }
    (output / "theory-components.json").write_text(
        json.dumps(record, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    lines = [
        r"\begin{table}[t]",
        r"\centering\small",
        r"\caption{Synthetic population-oracle diagnostics for balanced pairs at "
        r"$\sigma_\xi=0.1$. Entries average replicates. The extra-rank term uses "
        r"the analytic unregularized stable population optimum; fitted heads use "
        r"finite labels and ridge regularization. Dashes denote ranks outside "
        r"the sufficient-rank theorem.}",
        r"\label{tab:collection-population-components}",
        r"\begin{tabular}{rcccc}",
        r"\toprule",
        r"$r$ & Missed $\mathcal U$ & $\|P_{\mathcal U}w_r\|$ & "
        r"$\|(P_r-P_2)w^*_{\mathcal S}\|$ & "
        r"$L_z\|(P_r-P_2)w^*_{\mathcal S}\|$ \\",
        r"\midrule",
    ]
    for rank in config["ranks"]:
        selected = [row for row in rows if row["rank"] == rank]
        values = []
        for metric in (
            "missed_u",
            "sensitivity",
            "extra_oracle_removed_norm",
            "extra_rank_term",
        ):
            numbers = [row[metric] for row in selected if row[metric] is not None]
            values.append("---" if not numbers else f"{float(np.mean(numbers)):.3f}")
        lines.append(str(rank) + " & " + " & ".join(values) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    (output / "appendix-theory-table.tex").write_text("\n".join(lines) + "\n")
