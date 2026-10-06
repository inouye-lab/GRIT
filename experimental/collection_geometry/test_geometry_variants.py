"""Frozen-input reconstruction and default-curve equivalence for the supplement."""

from pathlib import Path
from typing import Any

import numpy as np
import pytest
from core import digest_array, fit_basis
from digits import _rng as digits_rng  # pyright: ignore[reportPrivateUsage]
from geometry_variants import build_geometry_variants, reconstruct_inputs
from run import file_hash, write_json
from sweeps import CONFIG, conditions, signal_design

from grit.data.collection_geometry import SimulationSpec, make_geometry


def test_digits_reconstruction_needs_no_images_and_rejects_hash_mismatch() -> None:
    seed, replicate = 20260927, 2
    q, r = np.linalg.qr(digits_rng(seed, replicate, 3).standard_normal((96, 96)))
    signs = np.sign(np.diag(r))  # pyright: ignore[reportUnknownMemberType]
    u = (q * signs)[:, 64:66]
    noise = np.random.default_rng([seed, replicate, 193]).standard_normal((4096, 96))
    frozen: dict[str, Any] = {
        "dataset": "digits",
        "seed": seed,
        "replicate": replicate,
        "config": CONFIG,
        "inputs": {"u": digest_array(u), "master_noise": digest_array(noise)},
    }
    actual_u, actual_noise = reconstruct_inputs(frozen)
    np.testing.assert_array_equal(actual_u, u)
    np.testing.assert_array_equal(actual_noise, noise)
    frozen["inputs"]["master_noise"] = "invalid"
    with pytest.raises(AssertionError, match="master_noise"):
        reconstruct_inputs(frozen)


def test_supplement_matches_defaults_and_preserves_inputs(tmp_path: Path) -> None:
    config: dict[str, Any] = {
        **CONFIG,
        "dimension": 12,
        "synthetic_replicates": 1,
        "digits_replicates": 0,
        "coverage_values": [0.25, 1.0],
        "noise_values": [0.1, 0.2],
        "pair_values": [128, 256, 512],
    }
    run, output = tmp_path / "primary", tmp_path / "supplement"
    run.mkdir()
    spec = SimulationSpec(dimension=12, b_rank=1, o_rank=1)
    u = make_geometry(spec, config["seed"], 0).reference_basis()
    noise = np.random.default_rng([config["seed"], 0, 193]).standard_normal((512, 12))
    anchors = np.arange(512) % 100
    frozen = {
        "dataset": "synthetic",
        "seed": config["seed"],
        "replicate": 0,
        "config": config,
        "input_metadata": {"spec": spec.model_dump()},
        "inputs": {
            "u": digest_array(u),
            "master_noise": digest_array(noise),
            "anchor_indices": anchors.tolist(),
        },
    }
    raw: list[dict[str, Any]] = []
    for item in conditions(config):
        if item["sweep"] not in {"coverage", "noise", "pairs"}:
            continue
        k = item["k"]
        clean = signal_design(u, k, item["lambda_min"], item["lambda_max"])
        observed = clean + item["sigma"] * noise[:k]
        basis = fit_basis(observed, 2)
        raw.append(
            {
                **item,
                "dataset": "synthetic",
                "replicate": 0,
                "method": "grit",
                "missed_u": float(
                    np.linalg.norm(u - basis @ (basis.T @ u), 2)  # pyright: ignore[reportUnknownMemberType, reportUnknownArgumentType]
                ),
                "clean_hash": digest_array(clean),
                "pair_hash": digest_array(observed),
                "noise_hash": digest_array(noise[:k]),
                "anchor_hash": digest_array(anchors[:k]),
            }
        )
    write_json(run / "config.json", config)
    write_json(run / "synthetic-00-frozen.json", frozen)
    write_json(run / "sweep-results.json", raw)
    before = {path: file_hash(path) for path in run.iterdir()}
    result = build_geometry_variants(run, output)
    assert result["checks"]["default_curve_checks"] == 7
    assert result["checks"]["max_default_error"] == 0
    assert len(result["rows"]) == 21
    assert before == {path: file_hash(path) for path in run.iterdir()}
    assert (output / "geometry-variants.json").exists()
    with pytest.raises(FileExistsError):
        build_geometry_variants(run, output)
