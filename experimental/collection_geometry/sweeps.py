"""Independent weakest-signal, noise and pair-count sweeps with frozen selection."""

# NumPy linear-algebra stubs lose known float64 array types.
# pyright: reportUnknownMemberType=false, reportUnknownArgumentType=false, reportUnknownVariableType=false
from __future__ import annotations

import argparse
import platform
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
from core import Probe, digest_array, evaluate, fit_basis, fit_probe, geometry_metrics
from digits import DigitData, make_digits, prepare_digits, test_digits
from run import CODE, HERE, file_hash, write_json

from grit.data.collection_geometry import (
    SimulationSpec,
    balanced_split,
    make_geometry,
    training_split,
)
from grit.geometry import FloatArray, training_group_weights

CONFIG: dict[str, Any] = {
    "seed": 20260927,
    "dimension": 96,
    "pairs": 256,
    "primary_sigma": 0.1,
    "lambda_min": 1.0,
    "lambda_max": 1.0,
    "ridge": 0.1,
    "rank": 2,
    "synthetic_replicates": 20,
    "digits_replicates": 10,
    "coverage_values": [1 / 64, 1 / 32, 1 / 16, 1 / 8, 1 / 4, 1 / 2, 3 / 4, 7 / 8, 1.0],
    "noise_values": [0.025, 0.05, 0.075, 0.1, 0.15, 0.2, 0.3, 0.45, 0.6, 0.8, 1.0],
    "pair_values": [64, 96, 128, 192, 256, 384, 512, 1024, 2048, 4096],
    "alphas": [0.0, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0],
    "selector": "worst_group",
    "slope_min_k": 512,
}


def signal_design(u: FloatArray, k: int, g: float, largest: float = 1.0) -> FloatArray:
    """Equal cue counts with EXACT (largest,g) clean empirical eigenvalues."""
    if k < 2 or k % 2 or not 0 <= g <= largest:
        raise ValueError("Require even k>=2 and 0<=g<=largest")
    if not np.allclose(u.T @ u, np.eye(2), atol=1e-10):
        raise ValueError("The two known cue directions must be orthonormal")
    clean = np.empty((k, u.shape[0]))
    clean[0::2] = np.sqrt(2 * largest) * u[:, 0]
    clean[1::2] = np.sqrt(2 * g) * u[:, 1]
    return clean


def conditions(config: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for sweep, values, key in [
        ("coverage", config["coverage_values"], "lambda_min"),
        ("noise", config["noise_values"], "sigma"),
        ("pairs", config["pair_values"], "k"),
        ("nonspanning_control", [0.0], "lambda_min"),
        ("clean_control", [0.0], "sigma"),
    ]:
        for value in values:
            item = {
                "sweep": sweep,
                "value": value,
                "lambda_min": config["lambda_min"],
                "lambda_max": config["lambda_max"],
                "sigma": config["primary_sigma"],
                "k": config["pairs"],
            }
            item[key] = value
            rows.append(item)
    return rows


def condition_key(item: dict[str, Any]) -> str:
    return f"k{item['k']}:s{item['sigma']}:g{item['lambda_min']}:l{item['lambda_max']}"


def run_replicate(
    dataset: str,
    replicate: int,
    config: dict[str, Any],
    output: Path,
    data: DigitData | None = None,
) -> list[dict[str, Any]]:
    seed = config["seed"]
    problem = None
    spec = None
    if dataset == "digits":
        assert data is not None
        problem = make_digits(seed, replicate, data)
        train, validation = problem.train, problem.validation
        geometry, u, label = problem.geometry, problem.u, problem.label_direction
        metadata = problem.metadata
    else:
        spec = SimulationSpec(
            dimension=96,
            b_rank=1,
            o_rank=1,
            train_size=3000,
            validation_per_group=160,
            test_per_group=400,
            bank_size=256,
        )
        geometry = make_geometry(spec, seed, replicate)
        train = training_split(spec, geometry, seed, replicate)
        validation = balanced_split(spec, geometry, seed, replicate, test=False)
        u, label = geometry.reference_basis(), geometry.label_direction(spec)
        metadata = {"spec": spec.model_dump()}
    rng = np.random.default_rng([seed, replicate, 193])
    max_k = max(config["pair_values"])
    noise = rng.standard_normal((max_k, config["dimension"]))
    anchors = np.arange(max_k) % len(train.labels)
    weights = training_group_weights(0.95, 0.95)
    probes: dict[str, Probe] = {}
    records: dict[str, dict[str, Any]] = {}
    chosen: dict[str, str] = {}
    geometries: dict[str, dict[str, Any]] = {}

    def register(key: str, probe: Probe, details: dict[str, Any]) -> None:
        probes[key] = probe
        records[key] = dict(
            details,
            validation=evaluate(probe, validation, weights),
            gradient=probe.gradient,
            iterations=probe.iterations,
            coefficient_norm=float(np.linalg.norm(probe.weight)),
            sensitivity=float(np.linalg.norm(u.T @ probe.weight)),
            weight_hash=digest_array(np.r_[probe.weight, probe.bias]),
        )

    for method, removed in [("erm", None), ("oracle", u)]:
        register(
            method,
            fit_probe(
                train.features, train.labels, removed=removed, ridge=config["ridge"]
            ),
            {"method": method},
        )
    for item in conditions(config):
        key = condition_key(item)
        if key in geometries:
            continue
        k, g, largest, sigma = (
            item["k"],
            item["lambda_min"],
            item["lambda_max"],
            item["sigma"],
        )
        clean = signal_design(u, k, g, largest)
        observed = clean + sigma * noise[:k]
        clean_cov = clean.T @ clean / k
        matrix = observed.T @ observed / k
        actual = np.linalg.eigvalsh(u.T @ clean_cov @ u)
        assert np.allclose(actual, [g, largest], rtol=1e-11, atol=1e-11)
        basis = fit_basis(observed, config["rank"])
        info = {
            "clean_trace": float(np.trace(clean_cov)),
            "observed_trace": float(np.trace(matrix)),
            "actual_clean_eigenvalues": actual.tolist(),
            "covariance_error": float(
                np.linalg.norm(
                    matrix - clean_cov - sigma**2 * np.eye(config["dimension"]), ord=2
                )
            ),
            "clean_hash": digest_array(clean),
            "pair_hash": digest_array(observed),
            "noise_hash": digest_array(noise[:k]),
            "anchor_hash": digest_array(anchors[:k]),
        }
        geometries[key] = info
        probe = fit_probe(
            train.features, train.labels, removed=basis, ridge=config["ridge"]
        )
        assert np.linalg.norm(basis.T @ probe.weight) < 1e-8
        register(
            f"{key}:grit",
            probe,
            dict(
                info,
                method="grit",
                rank=config["rank"],
                alpha=0.0,
                **geometry_metrics(basis, u, label),
            ),
        )
        if records[f"{key}:grit"]["sensitivity"] > (
            records[f"{key}:grit"]["missed_u"] * np.linalg.norm(probe.weight) + 1e-8
        ):
            raise AssertionError("Projected predictor violates sensitivity inequality")
        choices: list[tuple[float, str]] = []
        for alpha in config["alphas"]:
            candidate = f"{key}:cr{alpha}"
            register(
                candidate,
                fit_probe(
                    train.features,
                    train.labels,
                    penalty=matrix,
                    alpha=alpha,
                    ridge=config["ridge"],
                ),
                dict(info, method="cr", alpha=alpha, rank=-1),
            )
            choices.append((alpha, candidate))
        chosen[key] = max(
            choices,
            key=lambda pair: (records[pair[1]]["validation"]["worst_group"], -pair[0]),
        )[1]

    # Freeze every candidate and CR choice before any test features are requested.
    frozen = {
        "dataset": dataset,
        "replicate": replicate,
        "seed": seed,
        "config": config,
        "input_metadata": metadata,
        "inputs": {
            "train": digest_array(train.features),
            "train_labels": digest_array(train.labels),
            "validation": digest_array(validation.features),
            "validation_labels": digest_array(validation.labels),
            "u": digest_array(u),
            "master_noise": digest_array(noise),
            "anchor_indices": anchors.tolist(),
        },
        "selections": chosen,
        "candidates": records,
    }
    frozen_path = output / f"{dataset}-{replicate:02d}-frozen.json"
    write_json(frozen_path, frozen)
    np.savez_compressed(
        output / f"{dataset}-{replicate:02d}-weights.npz",
        **{key: np.r_[probe.weight, probe.bias] for key, probe in probes.items()},
    )
    if dataset == "digits":
        assert problem is not None and data is not None
        test = test_digits(problem, data)
    else:
        assert spec is not None
        test = balanced_split(spec, geometry, seed, replicate, test=True)
    # Evaluate each reported fitted candidate once, after the frozen file exists.
    requested = {
        "erm",
        "oracle",
        *chosen.values(),
        *(f"{key}:grit" for key in geometries),
    }
    test_metrics = {
        key: evaluate(probes[key], test, weights) for key in sorted(requested)
    }
    rows = []
    for item in conditions(config):
        key = condition_key(item)
        for candidate in [f"{key}:grit", chosen[key], "erm", "oracle"]:
            rows.append(
                {
                    **records[candidate],
                    **item,
                    "dataset": dataset,
                    "replicate": replicate,
                    "seed": seed,
                    "candidate": candidate,
                    "test": test_metrics[candidate],
                    "clean_trace": geometries[key]["clean_trace"],
                }
            )
    write_json(
        output / f"{dataset}-{replicate:02d}-results.json",
        {
            "frozen_sha256": file_hash(frozen_path),
            "test_inputs": {
                "features": digest_array(test.features),
                "labels": digest_array(test.labels),
                "groups": digest_array(test.groups),
                "ids": problem.metadata["test_ids"] if problem is not None else None,
            },
            "rows": rows,
        },
    )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    default = HERE / "sweep-results"
    output = (
        args.output.resolve()
        if args.output
        else (
            default
            if not default.exists()
            else HERE
            / "sweep-reruns"
            / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        )
    )
    if output.exists() and any(output.iterdir()):
        raise SystemExit("Use a fresh output directory to preserve previous evidence")
    output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    started = time.perf_counter()
    write_json(output / "config.json", CONFIG)
    sources = [
        HERE / name
        for name in [
            "sweeps.py",
            "geometry_variants.py",
            "sweep_report.py",
            "core.py",
            "digits.py",
            "run.py",
            "report.py",
            "run_sweeps.sh",
            "SWEEP-PROTOCOL.md",
        ]
    ] + [
        CODE / name
        for name in [
            "uv.lock",
            "src/grit/methods/projection.py",
            "src/grit/geometry.py",
            "src/grit/data/collection_geometry.py",
        ]
    ]
    manifest: dict[str, Any] = {
        "commit": "anonymous-source",
        "source_hashes": {str(p.relative_to(CODE)): file_hash(p) for p in sources},
        "python": platform.python_version(),
        "numpy": np.__version__,
        "torch": torch.__version__,
        "device": "cpu-float64",
        "platform": platform.platform(),
        "status": "running",
        "mps_available": torch.backends.mps.is_available(),
    }
    write_json(output / "manifest.json", manifest)
    rows = []
    for dataset in ["synthetic", "digits"]:
        data = None
        if dataset == "digits":
            data = prepare_digits(CODE / "scratch/collection-geometry/mnist")
            write_json(output / "dataset-inputs.json", data.metadata)
        count = CONFIG[f"{dataset}_replicates"]
        for replicate in range(count):
            rows.extend(run_replicate(dataset, replicate, CONFIG, output, data))
            elapsed = time.perf_counter() - started
            print(
                f"{dataset} {replicate + 1}/{count}: {elapsed:.1f}s",
                flush=True,
            )
    write_json(output / "sweep-results.json", rows)
    from geometry_variants import build_geometry_variants
    from sweep_report import make_sweep_report

    build_geometry_variants(output, output)
    make_sweep_report(output, rows, CONFIG)
    manifest.update(
        status="complete",
        elapsed_seconds=time.perf_counter() - started,
        results_sha256=file_hash(output / "sweep-results.json"),
    )
    write_json(output / "manifest.json", manifest)
    print(f"Complete: {output} ({manifest['elapsed_seconds']:.1f}s)", flush=True)


if __name__ == "__main__":
    main()
