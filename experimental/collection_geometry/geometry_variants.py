"""Secondary geometry sweeps from frozen inputs, without new predictor fits.

Only designed pair geometry is reconstructed. No images, labeled feature arrays,
test metrics or fitted coefficient arrays are used. Primary geometry fields are
read solely to verify exact agreement with the original default curves.
"""

# NumPy linear-algebra stubs lose known float64 array types.
# pyright: reportUnknownMemberType=false, reportUnknownArgumentType=false, reportUnknownVariableType=false
from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from core import digest_array, fit_basis

# Exact frozen stream reuse; no public alternative exists in this local module.
from digits import _rng as digits_rng  # pyright: ignore[reportPrivateUsage]
from run import CODE, HERE, file_hash, write_json
from sweeps import signal_design

from grit.data.collection_geometry import SimulationSpec, make_geometry
from grit.geometry import FloatArray

NOISE_SETTINGS = (0.05, 0.1, 0.2)
BUDGET_SETTINGS = (128, 256, 512)


def variant_conditions(config: dict[str, Any]) -> list[dict[str, Any]]:
    """Keep each original x grid; change only its prespecified secondary factor."""
    rows: list[dict[str, Any]] = []
    for sweep, values, secondary, settings in [
        ("coverage", config["coverage_values"], "sigma", NOISE_SETTINGS),
        ("noise", config["noise_values"], "k", BUDGET_SETTINGS),
        ("pairs", config["pair_values"], "sigma", NOISE_SETTINGS),
    ]:
        for setting in settings:
            for value in values:
                item = {
                    "sweep": sweep,
                    "value": value,
                    "lambda_min": config["lambda_min"],
                    "lambda_max": config["lambda_max"],
                    "sigma": config["primary_sigma"],
                    "k": config["pairs"],
                    "variant_key": f"{secondary}={setting:g}",
                    "variant_label": f"{secondary}={setting:g}",
                    "is_default": setting
                    == config["pairs" if secondary == "k" else "primary_sigma"],
                }
                item[secondary] = setting
                item[
                    {"coverage": "lambda_min", "noise": "sigma", "pairs": "k"}[sweep]
                ] = value
                rows.append(item)
    return rows


def reconstruct_inputs(frozen: dict[str, Any]) -> tuple[FloatArray, FloatArray]:
    """Reconstruct only U and the independent master-noise draw; verify both."""
    config = frozen["config"]
    seed, replicate = frozen["seed"], frozen["replicate"]
    dimension = config["dimension"]
    if frozen["dataset"] == "digits":
        if dimension != 96:
            raise ValueError("The frozen MNIST encoder has exactly 96 coordinates")
        q, r = np.linalg.qr(
            digits_rng(seed, replicate, 3).standard_normal((dimension, dimension))
        )
        rotation = q * np.sign(np.diag(r))
        u = rotation[:, 64:66]
    elif frozen["dataset"] == "synthetic":
        spec = SimulationSpec.model_validate(frozen["input_metadata"]["spec"])
        u = make_geometry(spec, seed, replicate).reference_basis()
    else:
        raise ValueError(f"Unknown dataset: {frozen['dataset']}")
    noise = np.random.default_rng([seed, replicate, 193]).standard_normal(
        (max(config["pair_values"]), dimension)
    )
    for name, array in [("u", u), ("master_noise", noise)]:
        if digest_array(array) != frozen["inputs"][name]:
            raise AssertionError(f"Reconstructed {name} differs from frozen input")
    return u, noise


def aggregate_geometry(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Replicate means and sample SD, with each secondary condition distinct."""
    grouped: dict[tuple[str, str, str, float], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[
            (row["dataset"], row["sweep"], row["variant_key"], row["value"])
        ].append(row)
    result: list[dict[str, Any]] = []
    for key in sorted(grouped):
        current = grouped[key]
        if len({row["replicate"] for row in current}) != len(current):
            raise ValueError("Duplicate replicate within geometry summary group")
        values = np.array([row["missed_u"] for row in current])
        result.append(
            {
                **{
                    name: current[0][name]
                    for name in [
                        "dataset",
                        "sweep",
                        "value",
                        "variant_key",
                        "variant_label",
                        "is_default",
                        "k",
                        "sigma",
                        "lambda_min",
                        "lambda_max",
                    ]
                },
                "n": len(current),
                "missed_u": {
                    "mean": float(values.mean()),
                    "sd": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
                },
            }
        )
    return result


def build_geometry_variants(run: Path, output: Path) -> dict[str, Any]:
    """Save geometry-only supplemental evidence alongside or outside a full run."""
    started = time.perf_counter()
    torch.set_num_threads(1)
    destination = output / "geometry-variants.json"
    if destination.exists():
        raise FileExistsError("Preserve existing geometry-variants evidence")
    config = json.loads((run / "config.json").read_text())
    if (
        config["pairs"],
        config["primary_sigma"],
        config["lambda_min"],
        config["lambda_max"],
        config["rank"],
    ) != (256, 0.1, 1.0, 1.0, 2):
        raise ValueError("This supplement requires the frozen primary reference")
    raw = json.loads((run / "sweep-results.json").read_text())
    defaults = {
        (row["dataset"], row["replicate"], row["sweep"], row["value"]): row
        for row in raw
        if row["method"] == "grit" and row["sweep"] in {"coverage", "noise", "pairs"}
    }
    rows: list[dict[str, Any]] = []
    evidence: list[dict[str, Any]] = []
    errors: list[float] = []
    spectra: list[float] = []
    unique_designs = 0
    for dataset in ["synthetic", "digits"]:
        for replicate in range(config[f"{dataset}_replicates"]):
            path = run / f"{dataset}-{replicate:02d}-frozen.json"
            frozen = json.loads(path.read_text())
            if frozen["config"] != config or frozen["dataset"] != dataset:
                raise AssertionError("Frozen configuration or dataset differs")
            if frozen["replicate"] != replicate or frozen["seed"] != config["seed"]:
                raise AssertionError("Frozen replicate or seed differs")
            u, noise = reconstruct_inputs(frozen)
            anchors = np.array(frozen["inputs"]["anchor_indices"], dtype=np.int64)
            if len(anchors) != len(noise):
                raise AssertionError("Frozen anchor and noise lengths differ")
            evidence.append(
                {
                    "dataset": dataset,
                    "replicate": replicate,
                    "frozen_file": path.name,
                    "frozen_sha256": file_hash(path),
                    "u_hash": digest_array(u),
                    "master_noise_hash": digest_array(noise),
                    "anchor_hash": digest_array(anchors),
                }
            )
            cache: dict[tuple[int, float, float, float], dict[str, Any]] = {}
            for item in variant_conditions(config):
                k, sigma = int(item["k"]), float(item["sigma"])
                g, largest = float(item["lambda_min"]), float(item["lambda_max"])
                key = (k, sigma, g, largest)
                if key not in cache:
                    clean = signal_design(u, k, g, largest)
                    observed = clean + sigma * noise[:k]
                    eigenvalues = np.linalg.eigvalsh(u.T @ clean.T @ clean @ u / k)
                    spectrum_error = float(np.max(np.abs(eigenvalues - [g, largest])))
                    if spectrum_error > 1e-11:
                        raise AssertionError("Secondary design has incorrect spectrum")
                    spectra.append(spectrum_error)
                    basis = fit_basis(observed, config["rank"])
                    cache[key] = {
                        "missed_u": float(np.linalg.norm(u - basis @ (basis.T @ u), 2)),
                        "actual_clean_eigenvalues": eigenvalues.tolist(),
                        "clean_hash": digest_array(clean),
                        "pair_hash": digest_array(observed),
                        "noise_hash": digest_array(noise[:k]),
                        "anchor_hash": digest_array(anchors[:k]),
                    }
                row = {
                    **item,
                    **cache[key],
                    "dataset": dataset,
                    "replicate": replicate,
                    "seed": config["seed"],
                }
                if item["is_default"]:
                    reference_key = (dataset, replicate, item["sweep"], item["value"])
                    reference = defaults[reference_key]
                    fields = ["clean_hash", "pair_hash", "noise_hash", "anchor_hash"]
                    for field in fields:
                        if row[field] != reference[field]:
                            raise AssertionError(f"Default {field} differs")
                    difference = abs(row["missed_u"] - reference["missed_u"])
                    if difference > 1e-12:
                        raise AssertionError("Default geometry differs from primary")
                    errors.append(difference)
                rows.append(row)
            unique_designs += len(cache)
    sources = [
        HERE / name
        for name in [
            "geometry_variants.py",
            "sweeps.py",
            "core.py",
            "digits.py",
            "run.py",
            "SWEEP-PROTOCOL.md",
        ]
    ] + [
        CODE / "src/grit/data/collection_geometry.py",
        CODE / "src/grit/methods/projection.py",
        CODE / "src/grit/geometry.py",
    ]
    # Public provenance identifies the run without exposing a local checkout path.
    try:
        run_location = str(run.resolve().relative_to(CODE.resolve()))
    except ValueError:
        run_location = run.resolve().name
    result: dict[str, Any] = {
        "schema_version": 1,
        "scope": "Geometry-only supplement; no new predictor fits or test evaluation",
        "config": {
            "primary": config,
            "coverage_noise_settings": list(NOISE_SETTINGS),
            "noise_budget_settings": list(BUDGET_SETTINGS),
            "pair_noise_settings": list(NOISE_SETTINGS),
        },
        "provenance": {
            "run": run_location,
            "primary_results_sha256": file_hash(run / "sweep-results.json"),
            "primary_config_sha256": file_hash(run / "config.json"),
            "source_hashes": {str(p.relative_to(CODE)): file_hash(p) for p in sources},
            "frozen_inputs": evidence,
        },
        "rows": rows,
        "groups": aggregate_geometry(rows),
        "checks": {
            "reconstructed_input_hash_pairs": len(evidence),
            "default_curve_checks": len(errors),
            "max_default_error": max(errors),
            "unique_pair_designs": unique_designs,
            "max_clean_spectrum_error": max(spectra),
        },
        "elapsed_seconds": time.perf_counter() - started,
    }
    output.mkdir(parents=True, exist_ok=True)
    write_json(destination, result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        raise SystemExit("Use a fresh output directory for supplemental evidence")
    result = build_geometry_variants(args.run.resolve(), args.output.resolve())
    print(
        f"Geometry supplement: {len(result['rows'])} rows; "
        f"{result['elapsed_seconds']:.1f}s; checks={result['checks']}"
    )


if __name__ == "__main__":
    main()
