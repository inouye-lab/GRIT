"""Fixed-probe empirical bridge and growing-shift diagnostics.

The new heldout stream uses seed + 700001, with 8192 source examples. A bridge
changes only the two true unstable coordinates: at scale one it reverses each
cue's class-dependent mean, retaining within-cue noise and all stable features.
The final target reweights bridge examples by
``h(P_S z) = 0.1 + exp(-0.5 * (v_S' z)^2)``, where ``v_S`` is the unit stable
class-mean direction. The strictly positive, bounded weight depends on observed
stable features alone. Consequently the underlying target has the same label
kernel given the stable projection (A1), and the same unstable conditional given
stable features and labels as the bridge. The empirical bridge term is computed
exactly for these finite samples; it is not a population confidence certificate.

No diagnostic fits a model, selects a hyperparameter, or modifies shared code.
"""

# NumPy's generic transpose/linear-algebra stubs lose the existing float64 type.
# pyright: reportUnknownVariableType=false, reportUnknownMemberType=false, reportUnknownArgumentType=false
from __future__ import annotations

import numpy as np
from core import Probe, digest_array

from grit.data.collection_geometry import Geometry, SimulationSpec, training_split

HOLDOUT_SEED_OFFSET = 700001
HOLDOUT_SAMPLE_SIZE = 8192
SHIFT_SCALES = (0.0, 1.0, 2.0, 4.0)

DiagnosticValue = str | int | float


def diagnostics_for_synthetic(
    geometry: Geometry,
    spec: SimulationSpec,
    seed: int,
    replicate: int,
    probes: dict[str, Probe],
) -> list[dict[str, DiagnosticValue]]:
    """Evaluate frozen probes on coupled source, bridge and weighted targets.

    ``Probe.weight`` already includes any fitted hard projection. Intercepts
    cancel in direct score changes. All reported losses are unpenalized binary
    logistic risk. ``rho`` is computed from the empirical displacement second
    moment, so the logistic direct-transport and telescoping inequalities hold
    on the exact evaluated sample, without a sampling or optimizer assumption.
    """
    diagnostic_seed = seed + HOLDOUT_SEED_OFFSET
    diagnostic_spec = spec.model_copy(update={"train_size": HOLDOUT_SAMPLE_SIZE})
    source = training_split(diagnostic_spec, geometry, diagnostic_seed, replicate)
    x, y = source.features, source.labels
    u = geometry.reference_basis()
    direction = geometry.label_direction(spec)
    direction = direction - u @ (u.T @ direction)
    direction_norm = float(np.linalg.norm(direction))
    if direction_norm <= 1e-12:
        raise ValueError("The bridge diagnostic needs a nonzero stable label direction")
    direction /= direction_norm
    stable_score = x @ direction
    target_weights = 0.1 + np.exp(-0.5 * stable_score**2)
    target_weights /= target_weights.sum()
    effective_sample_size = float(1.0 / (target_weights @ target_weights))
    delta = (
        -2.0 * (source.q_b[:, None] * geometry.m_b) @ geometry.u_b.T
        - 2.0 * (source.q_o[:, None] * geometry.m_o) @ geometry.u_o.T
    )
    coordinates = delta @ u
    stable_delta = delta - coordinates @ u.T
    max_stable_displacement = float(np.max(np.linalg.norm(stable_delta, axis=1)))
    if max_stable_displacement > 1e-10:
        raise AssertionError("The bridge displacement must lie in the true U")
    second_moment = coordinates.T @ coordinates / len(x)
    base_rho = float(np.sqrt(max(0.0, np.linalg.eigvalsh(second_moment)[-1])))
    inputs = {
        "source_features_sha256": digest_array(x),
        "source_labels_sha256": digest_array(y),
        "displacement_sha256": digest_array(delta),
        "target_weights_sha256": digest_array(target_weights),
    }
    rows: list[dict[str, DiagnosticValue]] = []
    for name, probe in probes.items():
        source_scores = probe.scores(x)
        source_losses = np.logaddexp(0.0, source_scores) - y * source_scores
        source_loss = float(source_losses.mean())
        source_accuracy = float(np.mean((source_scores >= 0.0) == y))
        unit_score_change = delta @ probe.weight
        sensitivity = float(np.linalg.norm(u.T @ probe.weight))
        for scale in SHIFT_SCALES:
            bridge_scores = source_scores + scale * unit_score_change
            bridge_losses = np.logaddexp(0.0, bridge_scores) - y * bridge_scores
            bridge_loss = float(bridge_losses.mean())
            target_loss = float(target_weights @ bridge_losses)
            correct = (bridge_scores >= 0.0) == y
            direct_signed = bridge_loss - source_loss
            bridge_signed = target_loss - bridge_loss
            bridge_term = abs(bridge_signed)
            rho = scale * base_rho
            sensitivity_bound = rho * sensitivity
            score_transport = float(np.mean(np.abs(scale * unit_score_change)))
            risk_upper_bound = source_loss + bridge_term + sensitivity_bound
            tolerance = 1e-10 * max(1.0, risk_upper_bound)
            if abs(direct_signed) > score_transport + tolerance:
                raise AssertionError("Logistic direct transport bound failed")
            if score_transport > sensitivity_bound + tolerance:
                raise AssertionError("Empirical displacement moment bound failed")
            if target_loss > risk_upper_bound + tolerance:
                raise AssertionError("Empirical risk decomposition bound failed")
            rows.append(
                {
                    "replicate": replicate,
                    "probe": name,
                    "diagnostic_seed": diagnostic_seed,
                    "sample_count": len(x),
                    "shift_scale": scale,
                    "source_loss": source_loss,
                    "bridge_loss": bridge_loss,
                    "target_loss": target_loss,
                    "source_accuracy": source_accuracy,
                    "bridge_accuracy": float(correct.mean()),
                    "target_accuracy": float(target_weights @ correct),
                    "direct_signed_loss_change": direct_signed,
                    "direct_absolute_loss_change": abs(direct_signed),
                    "bridge_signed_loss_change": bridge_signed,
                    "bridge_term": bridge_term,
                    "rho": rho,
                    "raw_u_sensitivity": sensitivity,
                    "direct_score_transport": score_transport,
                    "rho_times_sensitivity": sensitivity_bound,
                    "risk_upper_bound": risk_upper_bound,
                    "risk_decomposition_slack": risk_upper_bound - target_loss,
                    "telescoping_residual": (
                        target_loss - source_loss - direct_signed - bridge_signed
                    ),
                    "target_effective_sample_size": effective_sample_size,
                    "max_stable_displacement": max_stable_displacement,
                    **inputs,
                }
            )
    return rows
