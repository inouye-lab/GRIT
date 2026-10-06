"""Known-geometry simulation for collection coverage.

Each replicate draws new bases, labeled data and one pair bank. An example is
``z = S s + U_B b + U_O o + eps``; a B-pair changes only ``b`` (to the opposite cue
category with fresh within-category variation), an O-pair only ``o``. Every random
quantity comes from its own stream of ``SeedSequence([seed, replicate, stream])``,
so the stressors that share a draw key see the same latents and differ only in
geometry, and the test split is drawn only when it is unsealed.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Annotated

import numpy as np
from numpy.typing import NDArray
from pydantic import Field, StrictInt

from grit.geometry import WITHIN_LABEL, FloatArray, group_ids
from grit.schemas import StrictBoundaryModel

PositiveInt = Annotated[StrictInt, Field(gt=0)]
# YAML authors may write 30 or 1e-3 for a float.
Real = Annotated[float, Field(strict=False, allow_inf_nan=False)]
Probability = Annotated[Real, Field(gt=0.5, lt=1.0)]

# Independent random streams within one replicate.
GEOMETRY, TRAIN, VALIDATION, TEST, BANK, BANK_DRAWS, CALIBRATION, PAIR_NOISE = range(8)


class SimulationSpec(StrictBoundaryModel):
    """Default simulation settings."""

    dimension: PositiveInt = 128
    stable_rank: PositiveInt = 4
    stable_mean: Real = 1.28
    b_rank: PositiveInt = 2
    b_mean: Real = 2.5
    b_spread: Real = 1.0
    o_rank: PositiveInt = 2
    o_mean: Real = 2.0
    o_spread: Real = 1.0
    noise_scale: Real = 1.0
    # Training Pr(q_F = y), independent given y.
    p_b: Probability = 0.95
    p_o: Probability = 0.95
    # Useful-overlap stressor: angle of the first O direction toward the stable mean.
    overlap_degrees: Annotated[Real, Field(ge=0.0, lt=90.0)] = 0.0
    # One common isotropic pair-noise variance sigma_xi^2 (0 for clean pairs).
    pair_noise_variance: Annotated[Real, Field(ge=0.0)] = 0.0
    train_size: PositiveInt = 8000
    validation_per_group: PositiveInt = 250
    test_per_group: PositiveInt = 1000
    bank_size: PositiveInt = 256
    calibration_pairs: PositiveInt = 1024


def _rng(seed: int, replicate: int, stream: int) -> np.random.Generator:
    return np.random.default_rng(np.random.SeedSequence([seed, replicate, stream]))


def _orthonormal(rng: np.random.Generator, dimension: int) -> FloatArray:
    q, r = np.linalg.qr(rng.standard_normal((dimension, dimension)))
    return q * np.sign(np.diag(r))


@dataclass(frozen=True)
class Geometry:
    stable: FloatArray
    u_b: FloatArray
    u_o: FloatArray
    noise_basis: FloatArray
    noise_sd: FloatArray
    m_b: FloatArray
    m_o: FloatArray

    @property
    def dimension(self) -> int:
        return int(self.stable.shape[0])

    def second_moment(self, factor: str, spec: SimulationSpec) -> FloatArray:
        """True uncentered second moment of clean factor-F pair differences.

        b - b' = 2 q m + (zeta - zeta'), so E[(b - b')(b - b')^T] = 4 m m^T + 2 tau^2 I.
        """

        basis, mean, spread = (
            (self.u_b, self.m_b, spec.b_spread)
            if factor == "b"
            else (self.u_o, self.m_o, spec.o_spread)
        )
        inner = 4 * np.outer(mean, mean) + 2 * spread**2 * np.eye(len(mean))
        return basis @ inner @ basis.T

    def reference_basis(self) -> FloatArray:
        """Orthonormal basis of span(U_B, U_O) for the true-subspace reference."""

        q, _ = np.linalg.qr(np.concatenate([self.u_b, self.u_o], axis=1))
        return q

    def label_direction(self, spec: SimulationSpec) -> FloatArray:
        """Balanced class-mean difference E[z | y=1] - E[z | y=0].

        Under balanced groups the cue means cancel, so this is 2 S mu_S / sqrt(r_S);
        with overlap the O means still cancel, so it is unchanged.
        """

        ones = np.ones(spec.stable_rank) / math.sqrt(spec.stable_rank)
        return 2 * spec.stable_mean * (self.stable @ ones)


def make_geometry(spec: SimulationSpec, seed: int, replicate: int) -> Geometry:
    rng = _rng(seed, replicate, GEOMETRY)
    d, rs, rb, ro = spec.dimension, spec.stable_rank, spec.b_rank, spec.o_rank
    if rs + rb + ro > d:
        raise ValueError("stable and nuisance ranks exceed the dimension")
    q = _orthonormal(rng, d)
    stable, u_b, u_o = q[:, :rs], q[:, rs : rs + rb], q[:, rs + rb : rs + rb + ro]
    if spec.overlap_degrees > 0.0:
        theta = math.radians(spec.overlap_degrees)
        stable_mean = stable @ (np.ones(rs) / math.sqrt(rs))
        tilted = math.cos(theta) * u_o[:, 0] + math.sin(theta) * stable_mean
        # Gram-Schmidt keeping the tilted vector first.
        columns = [tilted / np.linalg.norm(tilted)]
        for j in range(1, ro):
            v = u_o[:, j].copy()
            for c in columns:
                v -= (c @ v) * c
            columns.append(v / np.linalg.norm(v))
        u_o = np.stack(columns, axis=1)
    noise_basis = _orthonormal(rng, d)
    noise_sd = np.sqrt(spec.noise_scale / np.arange(1, d + 1, dtype=np.float64))
    return Geometry(
        stable=stable,
        u_b=u_b,
        u_o=u_o,
        noise_basis=noise_basis,
        noise_sd=noise_sd,
        m_b=np.full(rb, spec.b_mean / math.sqrt(rb)),
        m_o=np.full(ro, spec.o_mean / math.sqrt(ro)),
    )


@dataclass(frozen=True)
class Split:
    features: FloatArray
    labels: NDArray[np.int64]
    groups: NDArray[np.int64]
    # Latents that pairs need: signed cues and the nuisance coordinates.
    q_b: FloatArray
    q_o: FloatArray
    b: FloatArray
    o: FloatArray


def _sample(
    spec: SimulationSpec,
    geometry: Geometry,
    rng: np.random.Generator,
    labels: NDArray[np.int64],
    b_aligned: NDArray[np.bool_],
    o_aligned: NDArray[np.bool_],
) -> Split:
    n = len(labels)
    signed = 2.0 * labels - 1.0
    q_b = np.where(b_aligned, signed, -signed)
    q_o = np.where(o_aligned, signed, -signed)
    ones = np.ones(spec.stable_rank) / math.sqrt(spec.stable_rank)
    s = signed[:, None] * spec.stable_mean * ones + rng.standard_normal(
        (n, spec.stable_rank)
    )
    b = q_b[:, None] * geometry.m_b + spec.b_spread * rng.standard_normal(
        (n, spec.b_rank)
    )
    o = q_o[:, None] * geometry.m_o + spec.o_spread * rng.standard_normal(
        (n, spec.o_rank)
    )
    eps = (rng.standard_normal((n, spec.dimension)) * geometry.noise_sd) @ (
        geometry.noise_basis.T
    )
    z = s @ geometry.stable.T + b @ geometry.u_b.T + o @ geometry.u_o.T + eps
    return Split(
        features=z,
        labels=labels,
        groups=group_ids(labels, b_aligned, o_aligned),
        q_b=q_b,
        q_o=q_o,
        b=b,
        o=o,
    )


def training_split(
    spec: SimulationSpec, geometry: Geometry, seed: int, replicate: int
) -> Split:
    rng = _rng(seed, replicate, TRAIN)
    n = spec.train_size
    labels = rng.integers(0, 2, size=n).astype(np.int64)
    b_aligned = rng.random(n) < spec.p_b
    o_aligned = rng.random(n) < spec.p_o
    return _sample(spec, geometry, rng, labels, b_aligned, o_aligned)


def balanced_split(
    spec: SimulationSpec,
    geometry: Geometry,
    seed: int,
    replicate: int,
    *,
    test: bool,
) -> Split:
    """Exactly balanced over the eight groups. ``test=True`` is the sealed split."""

    rng = _rng(seed, replicate, TEST if test else VALIDATION)
    per_group = spec.test_per_group if test else spec.validation_per_group
    labels = np.repeat(np.arange(2), 4 * per_group).astype(np.int64)
    within = np.tile(np.repeat(np.arange(4), per_group), 2)
    b_aligned = np.array([WITHIN_LABEL[int(w)][0] for w in within], dtype=np.bool_)
    o_aligned = np.array([WITHIN_LABEL[int(w)][1] for w in within], dtype=np.bool_)
    return _sample(spec, geometry, rng, labels, b_aligned, o_aligned)


def _partner_differences(
    spec: SimulationSpec,
    geometry: Geometry,
    anchors: Split,
    rows: slice,
    rng: np.random.Generator,
) -> tuple[FloatArray, FloatArray]:
    """Clean delta_B and delta_O for anchors ``rows`` (both partners of each).

    The partner keeps everything but one factor, which moves to the opposite
    category with fresh within-category variation: b' = -q_B m_B + zeta'_B. The
    stable part and eps cancel exactly, so delta_B = U_B (b - b').
    """

    q_b, q_o = anchors.q_b[rows], anchors.q_o[rows]
    n = len(q_b)
    b_partner = -q_b[:, None] * geometry.m_b + spec.b_spread * rng.standard_normal(
        (n, spec.b_rank)
    )
    o_partner = -q_o[:, None] * geometry.m_o + spec.o_spread * rng.standard_normal(
        (n, spec.o_rank)
    )
    delta_b = (anchors.b[rows] - b_partner) @ geometry.u_b.T
    delta_o = (anchors.o[rows] - o_partner) @ geometry.u_o.T
    return delta_b, delta_o


@dataclass(frozen=True)
class Bank:
    """Ordered positions: anchor features, both clean differences, u_i and xi_i."""

    anchors: FloatArray
    delta_b: FloatArray
    delta_o: FloatArray
    draws: FloatArray
    noise: FloatArray

    def collection(
        self, is_o: NDArray[np.bool_]
    ) -> tuple[FloatArray, FloatArray, FloatArray]:
        """(left, right, observed differences) for the first ``len(is_o)`` positions."""

        k = len(is_o)
        delta = np.where(is_o[:, None], self.delta_o[:k], self.delta_b[:k])
        observed = delta + self.noise[:k]
        left = self.anchors[:k]
        return left, left - observed, observed


def make_bank(
    spec: SimulationSpec,
    geometry: Geometry,
    train: Split,
    seed: int,
    replicate: int,
) -> Bank:
    """The first ``bank_size`` training examples are the anchors."""

    rows = slice(0, spec.bank_size)
    delta_b, delta_o = _partner_differences(
        spec, geometry, train, rows, _rng(seed, replicate, BANK)
    )
    noise = math.sqrt(spec.pair_noise_variance) * _rng(
        seed, replicate, PAIR_NOISE
    ).standard_normal((spec.bank_size, spec.dimension))
    return Bank(
        anchors=train.features[rows],
        delta_b=delta_b,
        delta_o=delta_o,
        draws=_rng(seed, replicate, BANK_DRAWS).random(spec.bank_size),
        noise=noise,
    )


def calibration_energies(
    spec: SimulationSpec,
    geometry: Geometry,
    train: Split,
    seed: int,
    replicate: int,
) -> tuple[float, float]:
    """Mean observed per-pair energies (E-bar_B, E-bar_O) on anchors disjoint from
    the bank: the next ``calibration_pairs`` training examples, noise included."""

    start = spec.bank_size
    rows = slice(start, start + spec.calibration_pairs)
    if rows.stop > spec.train_size:
        raise ValueError("training split is too small for bank plus calibration")
    rng = _rng(seed, replicate, CALIBRATION)
    delta_b, delta_o = _partner_differences(spec, geometry, train, rows, rng)
    sigma = math.sqrt(spec.pair_noise_variance)
    energy_b = np.square(delta_b + sigma * rng.standard_normal(delta_b.shape)).sum(
        axis=1
    )
    energy_o = np.square(delta_o + sigma * rng.standard_normal(delta_o.shape)).sum(
        axis=1
    )
    return float(energy_b.mean()), float(energy_o.mean())


def clean_pair_energy(spec: SimulationSpec, factor: str) -> float:
    """E ||U_F (f - f')||^2 = 4 gamma_F^2 + 2 r_F tau_F^2."""

    if factor == "b":
        return 4 * spec.b_mean**2 + 2 * spec.b_rank * spec.b_spread**2
    return 4 * spec.o_mean**2 + 2 * spec.o_rank * spec.o_spread**2
