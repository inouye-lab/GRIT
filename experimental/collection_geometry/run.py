"""Run the complete small experiment with one command; no shared code changes."""

# NumPy linear-algebra/transpose stubs lose known float64 array types.
# pyright: reportUnknownMemberType=false, reportUnknownArgumentType=false, reportUnknownVariableType=false
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from digits import DigitData, DigitProblem

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/grit-matplotlib")

import numpy as np
import torch
from core import (
    Probe,
    digest_array,
    evaluate,
    fit_basis,
    fit_probe,
    geometry_metrics,
    pair_design,
    selector,
)

from grit.data.collection_geometry import (
    SimulationSpec,
    balanced_split,
    make_bank,
    make_geometry,
    training_split,
)
from grit.geometry import training_group_weights

HERE = Path(__file__).resolve().parent
CODE = HERE.parents[1]
CONFIG: dict[str, Any] = {
    "seed": 20260925,
    "dimension": 96,
    "pairs": 256,
    "synthetic_replicates": 20,
    "digits_replicates": 10,
    "geometry_replicates": 32,
    "sigmas": [0.0, 0.1, 0.3, 0.6],
    "collections": {"one": 0.0, "uneven": 0.0625, "balanced": 0.5},
    "ranks": [0, 1, 2, 4, 8, 16],
    "alphas": [0.0, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0],
    "ridge": 0.1,
    "primary_sigma": 0.1,
    "rate_k": [64, 128, 256, 512, 1024, 2048, 4096],
    "rate_sigmas": [0.1, 0.3, 0.6],
    "slope_min_k": 512,
    "selectors": ["worst_group", "train_weighted"],
}


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def experiment_replicate(
    dataset: str,
    replicate: int,
    config: dict[str, Any],
    output: Path,
    digits_data: DigitData | None = None,
    pilot: bool = False,
) -> dict[str, Any]:
    seed = config["seed"]
    rng = np.random.default_rng([seed, replicate, 77])
    weights = training_group_weights(0.95, 0.95)
    spec: SimulationSpec | None = None
    problem: DigitProblem | None = None
    if dataset == "synthetic":
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
        bank = make_bank(spec, geometry, train, seed, replicate)
        delta_b, delta_o, draws = bank.delta_b, bank.delta_o, bank.draws
        u = geometry.reference_basis()
        label = geometry.label_direction(spec)
        input_metadata = {"spec": spec.model_dump()}
    else:
        from digits import make_digits

        assert digits_data is not None
        problem = make_digits(seed, replicate, digits_data)
        train, validation = problem.train, problem.validation
        geometry, u, label = problem.geometry, problem.u, problem.label_direction
        delta_b = np.tile(u[:, 0], (256, 1))
        delta_o = np.tile(u[:, 1], (256, 1))
        draws = rng.random(256)
        input_metadata = problem.metadata

    noise = rng.standard_normal((256, 96))
    inputs = {
        "train": digest_array(train.features),
        "train_labels": digest_array(train.labels),
        "validation": digest_array(validation.features),
        "validation_labels": digest_array(validation.labels),
        "delta_b": digest_array(delta_b),
        "delta_o": digest_array(delta_o),
        "noise": digest_array(noise),
        "draws": digest_array(draws),
    }
    candidates: dict[str, Probe] = {}
    records: dict[str, dict[str, Any]] = {}
    chosen: dict[str, str] = {}

    def register(key: str, probe: Probe, metadata: dict[str, Any]) -> None:
        candidates[key] = probe
        records[key] = dict(
            metadata,
            validation=evaluate(probe, validation, weights),
            gradient=probe.gradient,
            iterations=probe.iterations,
            weight_hash=digest_array(np.r_[probe.weight, probe.bias]),
            sensitivity=float(np.linalg.norm(u.T @ probe.weight)),
            coefficient_norm=float(np.linalg.norm(probe.weight)),
            stable_coefficient_norm=float(
                np.linalg.norm(probe.weight - u @ (u.T @ probe.weight))
            ),
        )

    for method, basis in [("erm", None), ("oracle", u)]:
        register(
            method,
            fit_probe(
                train.features, train.labels, removed=basis, ridge=config["ridge"]
            ),
            {
                "method": method,
                "collection": "reference",
                "sigma": 0.0,
                "rank": 0 if basis is None else 2,
                "alpha": 0.0,
            },
        )
    for sigma in config["sigmas"]:
        for name, share in config["collections"].items():
            clean, observed = pair_design(delta_b, delta_o, draws, noise, share, sigma)
            full_basis = fit_basis(observed, max(config["ranks"]))
            matrix = observed.T @ observed / len(observed)
            clean_matrix = clean.T @ clean / len(clean)
            geometry_info = {
                "collection": name,
                "sigma": sigma,
                "clean_trace": float(np.trace(clean_matrix)),
                "observed_trace": float(np.trace(matrix)),
                "clean_gap": 2 * min(share, 1 - share),
                "covariance_error": float(
                    np.linalg.norm(matrix - clean_matrix - sigma**2 * np.eye(96), ord=2)
                ),
                "pair_hash": digest_array(observed),
            }
            groups: dict[str, list[tuple[float, str]]] = {"grit": [], "cr": []}
            for rank in config["ranks"]:
                basis = full_basis[:, :rank]
                key = f"{name}:s{sigma}:r{rank}"
                probe = fit_probe(
                    train.features, train.labels, removed=basis, ridge=config["ridge"]
                )
                register(
                    key,
                    probe,
                    dict(
                        geometry_info,
                        method="grit",
                        rank=rank,
                        alpha=0.0,
                        **geometry_metrics(basis, u, label),
                    ),
                )
                assert np.linalg.norm(basis.T @ probe.weight) < 1e-8
                groups["grit"].append((rank, key))
            for alpha in config["alphas"]:
                key = f"{name}:s{sigma}:a{alpha}"
                probe = fit_probe(
                    train.features,
                    train.labels,
                    penalty=matrix,
                    alpha=alpha,
                    ridge=config["ridge"],
                )
                register(
                    key, probe, dict(geometry_info, method="cr", rank=-1, alpha=alpha)
                )
                groups["cr"].append((alpha, key))
            for method, keys in groups.items():
                for rule in config["selectors"]:
                    selected = max(
                        keys,
                        key=lambda pair: (
                            selector(records[pair[1]]["validation"], rule),
                            -pair[0],
                        ),
                    )[1]
                    chosen[f"{name}:s{sigma}:{method}:{rule}"] = selected
            if name == "balanced" and sigma in [0.1, 0.3]:
                basis = full_basis[:, :2]
                register(
                    f"stress:s{sigma}:r2",
                    fit_probe(train.features, train.labels, removed=basis, ridge=1e-4),
                    dict(
                        geometry_info,
                        method="grit_weak_ridge",
                        rank=2,
                        alpha=0.0,
                        ridge=1e-4,
                        **geometry_metrics(basis, u, label),
                    ),
                )

    frozen = {
        "dataset": dataset,
        "replicate": replicate,
        "seed": seed,
        "config": config,
        "inputs": inputs,
        "input_metadata": input_metadata,
        "selections": chosen,
        "candidates": records,
    }
    freeze_path = output / f"{dataset}-{replicate:02d}-frozen.json"
    write_json(freeze_path, frozen)
    frozen_hash = file_hash(freeze_path)
    np.savez_compressed(
        output / f"{dataset}-{replicate:02d}-weights.npz",
        **{key: np.r_[p.weight, p.bias] for key, p in candidates.items()},
    )
    if pilot:
        return {
            "dataset": dataset,
            "replicate": replicate,
            "frozen_hash": frozen_hash,
            "selections": chosen,
            "records": records,
            "pilot": True,
        }
    # First test creation/access; selection above has no test input.
    if dataset == "synthetic":
        assert spec is not None
        test = balanced_split(spec, geometry, seed, replicate, test=True)
    else:
        from digits import preview_digits, test_digits

        assert problem is not None and digits_data is not None
        test = test_digits(problem, digits_data)
        if replicate == 0:
            preview_digits(problem, digits_data, output / "digits-preview.png")
    assert file_hash(freeze_path) == frozen_hash
    for key, probe in candidates.items():
        records[key]["test"] = evaluate(probe, test, weights)
    extra = []
    if dataset == "synthetic":
        from diagnostics import diagnostics_for_synthetic

        assert spec is not None
        diagnostic_probes = {"erm": candidates["erm"], "oracle": candidates["oracle"]}
        sigma = config["primary_sigma"]
        diagnostic_probes.update(
            {
                f"grit-r{r}": candidates[f"balanced:s{sigma}:r{r}"]
                for r in [1, 2, 4, 8, 16]
            }
        )
        diagnostic_probes["cr"] = candidates[
            chosen[f"balanced:s{sigma}:cr:worst_group"]
        ]
        extra = diagnostics_for_synthetic(
            geometry, spec, seed, replicate, diagnostic_probes
        )
    return {
        "dataset": dataset,
        "replicate": replicate,
        "frozen_hash": frozen_hash,
        "test_hash": digest_array(test.features),
        "test_label_hash": digest_array(test.labels),
        "test_group_hash": digest_array(test.groups),
        "input_metadata": input_metadata,
        "selections": chosen,
        "records": records,
        "diagnostics": extra,
    }


def geometry_rates(config: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for rep in range(config["geometry_replicates"]):
        rng = np.random.default_rng([config["seed"], rep, 991])
        u = np.linalg.qr(rng.normal(size=(96, 2)))[0]
        noise = rng.normal(size=(max(config["rate_k"]), 96))
        # Periodic nested design has exact 1:1 or 15:1 counts at every listed k.
        for name in ["balanced", "uneven"]:
            period = 2 if name == "balanced" else 16
            factors = (np.arange(max(config["rate_k"])) % period == 0).astype(int)
            clean = np.sqrt(2) * u[:, factors].T
            for sigma in config["rate_sigmas"]:
                for k in config["rate_k"]:
                    observed = clean[:k] + sigma * noise[:k]
                    basis = fit_basis(observed, 2)
                    covariance = observed.T @ observed / k
                    expected = clean[:k].T @ clean[:k] / k + sigma**2 * np.eye(96)
                    rows.append(
                        {
                            "replicate": rep,
                            "collection": name,
                            "sigma": sigma,
                            "k": k,
                            "missed_u": float(
                                np.linalg.norm(u - basis @ (basis.T @ u), ord=2)
                            ),
                            "covariance_error": float(
                                np.linalg.norm(covariance - expected, ord=2)
                            ),
                        }
                    )
        if rep % 8 == 0:
            print(
                f"geometry replicate {rep + 1}/{config['geometry_replicates']}",
                flush=True,
            )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--pilot",
        action="store_true",
        help="Two validation-only replicates; no test access",
    )
    parser.add_argument(
        "--skip-digits", action="store_true", help="Synthetic-only development run"
    )
    args = parser.parse_args()
    config = dict(CONFIG)
    if args.pilot:
        config.update(seed=20260926, synthetic_replicates=2, digits_replicates=2)
    if args.output is not None:
        output = args.output.resolve()
    elif (HERE / "results").exists():
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        output = HERE / "reruns" / timestamp
    else:
        output = HERE / "results"
    if (output / "manifest.json").exists():
        raise SystemExit(
            "Output already exists; use a fresh --output to preserve run evidence."
        )
    output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    started = time.perf_counter()
    write_json(output / "config.json", config)
    code_files = list(HERE.glob("*.py")) + [
        CODE / "src/grit/methods/projection.py",
        CODE / "src/grit/geometry.py",
        CODE / "src/grit/data/collection_geometry.py",
        CODE / "src/grit/methods/training.py",
        CODE / "uv.lock",
        HERE / "PROTOCOL.md",
    ]
    manifest: dict[str, Any] = {
        "commit": "anonymous-source",
        "source_hashes": {str(p.relative_to(CODE)): file_hash(p) for p in code_files},
        "python": platform.python_version(),
        "numpy": np.__version__,
        "torch": torch.__version__,
        "platform": platform.platform(),
        "device": "cpu-float64",
        "mps_available": torch.backends.mps.is_available(),
        "pilot": args.pilot,
        "status": "running",
    }
    write_json(output / "manifest.json", manifest)
    all_results: list[dict[str, Any]] = []
    data: DigitData | None = None
    for dataset, count in [
        ("synthetic", config["synthetic_replicates"]),
        ("digits", config["digits_replicates"]),
    ]:
        if dataset == "digits":
            if args.skip_digits:
                continue
            from digits import prepare_digits

            data = prepare_digits(CODE / "scratch/collection-geometry/mnist")
            write_json(output / "dataset-inputs.json", data.metadata)
        for replicate in range(count):
            result = experiment_replicate(
                dataset, replicate, config, output, data, args.pilot
            )
            all_results.append(result)
            write_json(output / f"{dataset}-{replicate:02d}-results.json", result)
            print(
                f"{dataset} {replicate + 1}/{count}, "
                f"{time.perf_counter() - started:.1f}s",
                flush=True,
            )
    rates = [] if args.pilot else geometry_rates(config)
    write_json(output / "geometry-rates.json", rates)
    write_json(output / "results.json", all_results)
    if not args.pilot:
        from paper_assets import make_paper_assets
        from report import make_report
        from supplement import make_supplement

        make_report(output, all_results, rates, config)
        make_paper_assets(output)
        make_supplement(output)
    manifest.update(
        status="complete",
        elapsed_seconds=time.perf_counter() - started,
        results_sha256=file_hash(output / "results.json"),
        rates_sha256=file_hash(output / "geometry-rates.json"),
    )
    write_json(output / "manifest.json", manifest)
    print(f"Complete: {output} ({manifest['elapsed_seconds']:.1f}s)", flush=True)


if __name__ == "__main__":
    main()
