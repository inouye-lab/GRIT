"""Rebuild every table and figure from the frozen experiment records."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

COLLECTIONS = ("one", "uneven", "balanced")
LABELS = {"one": "One direction", "uneven": "Uneven (15:1)", "balanced": "Balanced"}
COLORS = {"one": "#b35a37", "uneven": "#be941f", "balanced": "#177e89"}
N_BOOTSTRAP = 5000
BOOTSTRAP_SEED = 260925
RECORD_METRICS = (
    "missed_u",
    "sensitivity",
    "coefficient_norm",
    "stable_coefficient_norm",
    "label_removed",
    "captured_b",
    "captured_o",
    "effective_rank",
    "rank",
    "alpha",
    "clean_trace",
    "observed_trace",
    "covariance_error",
    "clean_gap",
)
TEST_METRICS = (
    "in_distribution",
    "b_shift",
    "o_shift",
    "both_shift",
    "worst_group",
    "aligned",
    "loss",
    "both_loss",
    "source_loss",
)


def statistics(values: list[float], key: str) -> dict:
    """Seed SD and a fixed percentile bootstrap CI of the replicate mean."""
    array = np.asarray(values, dtype=float)
    if not len(array) or not np.all(np.isfinite(array)):
        raise ValueError(f"Nonempty finite observations required for {key}")
    salt = int.from_bytes(hashlib.sha256(key.encode()).digest()[:4], "little")
    rng = np.random.default_rng([BOOTSTRAP_SEED, salt])
    samples = rng.integers(0, len(array), size=(N_BOOTSTRAP, len(array)))
    low, high = np.quantile(array[samples].mean(axis=1), [0.025, 0.975])
    return {
        "n": len(array),
        "mean": float(array.mean()),
        "sd": float(array.std(ddof=1)) if len(array) > 1 else 0.0,
        "ci95_mean": [float(low), float(high)],
    }


def aggregate(records: list[dict], key: str) -> dict:
    answer = {}
    for metric in RECORD_METRICS:
        if all(metric in row for row in records):
            answer[metric] = statistics([row[metric] for row in records], key + metric)
    for metric in TEST_METRICS:
        answer[metric] = statistics(
            [row["test"][metric] for row in records], key + metric
        )
    return answer


def chosen_records(
    results: list[dict], collection: str, sigma: float, method: str, rule: str
) -> list[dict]:
    selection = f"{collection}:s{sigma}:{method}:{rule}"
    return [row["records"][row["selections"][selection]] for row in results]


def fixed_records(
    results: list[dict], collection: str, sigma: float, rank: int = 2
) -> list[dict]:
    return [row["records"][f"{collection}:s{sigma}:r{rank}"] for row in results]


def rate_slopes(rates: list[dict], minimum: int) -> list[dict]:
    summaries = []
    combinations = sorted({(row["collection"], row["sigma"]) for row in rates})
    for collection, sigma in combinations:
        selected = [
            row
            for row in rates
            if row["collection"] == collection
            and row["sigma"] == sigma
            and row["k"] >= minimum
        ]
        for metric in ("missed_u", "covariance_error"):
            slopes = []
            for replicate in sorted({row["replicate"] for row in selected}):
                rows = sorted(
                    [row for row in selected if row["replicate"] == replicate],
                    key=lambda row: row["k"],
                )
                if len(rows) < 2:
                    continue
                slope = np.polyfit(
                    np.log([row["k"] for row in rows]),
                    np.log([row[metric] for row in rows]),
                    1,
                )[0]
                slopes.append({"replicate": replicate, "slope": float(slope)})
            if slopes:
                summaries.append(
                    {
                        "collection": collection,
                        "sigma": sigma,
                        "metric": metric,
                        "minimum_k": minimum,
                        "k_values": sorted({row["k"] for row in selected}),
                        **statistics(
                            [row["slope"] for row in slopes],
                            f"slope:{collection}:{sigma}:{metric}",
                        ),
                        "replicate_slopes": slopes,
                    }
                )
    return summaries


def make_summary(all_results: list[dict], rates: list[dict], config: dict) -> dict:
    summary = {
        "primary_sigma": config["primary_sigma"],
        "ridge": config["ridge"],
        "sigmas": config["sigmas"],
        "uncertainty": {
            "display": "mean plus/minus sample SD across replicates",
            "interval": (
                "percentile bootstrap of replicate means; "
                "paired differences use matching replicates"
            ),
            "bootstrap_resamples": N_BOOTSTRAP,
            "bootstrap_seed": BOOTSTRAP_SEED,
            "digits_scope": "split/bank variation on one fixed MNIST dataset",
            "rate_fit": f"log-log OLS per replicate, k >= {config['slope_min_k']}",
        },
        "tables": [],
        "baselines": [],
        "paired_balanced_minus_uneven": [],
        "rate_slopes": rate_slopes(rates, config["slope_min_k"]),
        "selection_counts": [],
        "rank_paths": [],
        "cr_paths": [],
        "bridge": [],
        "weak_ridge_stress": [],
    }
    for dataset in sorted({row["dataset"] for row in all_results}, reverse=True):
        results = sorted(
            [row for row in all_results if row["dataset"] == dataset],
            key=lambda row: row["replicate"],
        )
        for method in ("erm", "oracle"):
            summary["baselines"].append(
                {
                    "dataset": dataset,
                    "method": method,
                    "metrics": aggregate(
                        [row["records"][method] for row in results],
                        f"{dataset}:{method}",
                    ),
                }
            )
        for sigma in config["sigmas"]:
            for collection in COLLECTIONS:
                key = f"{dataset}:{sigma}:{collection}:"
                entry = {
                    "dataset": dataset,
                    "sigma": sigma,
                    "collection": collection,
                    "fixed_grit": aggregate(
                        fixed_records(results, collection, sigma), key + "fixed"
                    ),
                }
                for method in ("grit", "cr"):
                    for rule in config["selectors"]:
                        rows = chosen_records(results, collection, sigma, method, rule)
                        entry[f"{method}_selected_{rule}"] = aggregate(
                            rows, key + method + rule
                        )
                        parameter = "rank" if method == "grit" else "alpha"
                        counts = Counter(row[parameter] for row in rows)
                        summary["selection_counts"].append(
                            {
                                "dataset": dataset,
                                "sigma": sigma,
                                "collection": collection,
                                "method": method,
                                "selector": rule,
                                "parameter": parameter,
                                "counts": {
                                    str(value): counts[value]
                                    for value in sorted(counts)
                                },
                            }
                        )
                summary["tables"].append(entry)
            for method in (
                "fixed_grit",
                "grit_selected_worst_group",
                "cr_selected_worst_group",
            ):
                if method == "fixed_grit":
                    balanced = fixed_records(results, "balanced", sigma)
                    uneven = fixed_records(results, "uneven", sigma)
                else:
                    name = method.split("_")[0]
                    balanced = chosen_records(
                        results, "balanced", sigma, name, "worst_group"
                    )
                    uneven = chosen_records(
                        results, "uneven", sigma, name, "worst_group"
                    )
                for metric in (
                    "both_shift",
                    "worst_group",
                    "both_loss",
                    "missed_u",
                    "sensitivity",
                ):
                    if metric in ("missed_u", "sensitivity"):
                        if not all(metric in row for row in balanced + uneven):
                            continue
                        differences = [
                            b[metric] - u[metric]
                            for b, u in zip(balanced, uneven, strict=True)
                        ]
                    else:
                        differences = [
                            b["test"][metric] - u["test"][metric]
                            for b, u in zip(balanced, uneven, strict=True)
                        ]
                    summary["paired_balanced_minus_uneven"].append(
                        {
                            "dataset": dataset,
                            "sigma": sigma,
                            "method": method,
                            "metric": metric,
                            **statistics(
                                differences,
                                f"paired:{dataset}:{sigma}:{method}:{metric}",
                            ),
                        }
                    )
        primary_sigma = config["primary_sigma"]
        for rank in config["ranks"]:
            summary["rank_paths"].append(
                {
                    "dataset": dataset,
                    "collection": "balanced",
                    "sigma": primary_sigma,
                    "rank": rank,
                    "metrics": aggregate(
                        fixed_records(results, "balanced", primary_sigma, rank),
                        f"rank:{dataset}:{rank}",
                    ),
                }
            )
        for alpha in config["alphas"]:
            rows = [
                row["records"][f"balanced:s{primary_sigma}:a{alpha}"] for row in results
            ]
            summary["cr_paths"].append(
                {
                    "dataset": dataset,
                    "collection": "balanced",
                    "sigma": primary_sigma,
                    "alpha": alpha,
                    "metrics": aggregate(rows, f"cr:{dataset}:{alpha}"),
                }
            )
        for sigma in config["sigmas"]:
            key = f"stress:s{sigma}:r2"
            if all(key in result["records"] for result in results):
                summary["weak_ridge_stress"].append(
                    {
                        "dataset": dataset,
                        "collection": "balanced",
                        "sigma": sigma,
                        "ridge": 1e-4,
                        "metrics": aggregate(
                            [result["records"][key] for result in results],
                            f"stress:{dataset}:{sigma}",
                        ),
                        "standard_ridge": aggregate(
                            fixed_records(results, "balanced", sigma),
                            f"standard:{dataset}:{sigma}",
                        ),
                    }
                )
    diagnostic_rows = [
        extra for result in all_results for extra in result.get("diagnostics", [])
    ]
    for probe, scale in sorted(
        {(row["probe"], row["shift_scale"]) for row in diagnostic_rows}
    ):
        rows = [
            row
            for row in diagnostic_rows
            if row["probe"] == probe and row["shift_scale"] == scale
        ]
        metrics = {}
        for metric, value in rows[0].items():
            if isinstance(value, (int, float)) and metric not in (
                "replicate",
                "diagnostic_seed",
                "shift_scale",
            ):
                metrics[metric] = statistics(
                    [row[metric] for row in rows], f"bridge:{probe}:{scale}:{metric}"
                )
        summary["bridge"].append(
            {"probe": probe, "shift_scale": scale, "metrics": metrics}
        )
    return summary


def formatted(stats: dict, *, percent: bool = False, tex: bool = False) -> str:
    factor = 100 if percent else 1
    precision = 1 if percent else 3
    separator = r"\pm" if tex else "±"
    value = (
        f"{stats['mean'] * factor:.{precision}f} {separator} "
        f"{stats['sd'] * factor:.{precision}f}"
    )
    return f"${value}$" if tex else value


def save_figure(fig, output: Path, name: str) -> None:
    fig.savefig(output / f"{name}.png", dpi=180, bbox_inches="tight")
    fig.savefig(
        output / f"{name}.pdf",
        bbox_inches="tight",
        metadata={
            "Creator": "GRIT collection geometry report.py",
            "CreationDate": None,
        },
    )
    plt.close(fig)


def curve(
    ax,
    x,
    rows: list[dict],
    metric: str,
    label: str,
    color=None,
    style="-",
    percent=False,
    log=False,
) -> None:
    mean = np.array([row[metric]["mean"] for row in rows]) * (100 if percent else 1)
    sd = np.array([row[metric]["sd"] for row in rows]) * (100 if percent else 1)
    (line,) = ax.plot(
        x, mean, style, marker="o", markersize=3, label=label, color=color
    )
    lower = np.maximum(mean - sd, 1e-10) if log else mean - sd
    ax.fill_between(
        x, lower, mean + sd, alpha=0.12, color=line.get_color(), linewidth=0
    )


def plot_geometry(output: Path, summary: dict) -> None:
    datasets = list(dict.fromkeys(row["dataset"] for row in summary["tables"]))
    fig, axes = plt.subplots(
        len(datasets), 3, figsize=(10.0, 2.7 * len(datasets)), squeeze=False
    )
    noise_colors = ("#777777", "#177e89", "#be941f", "#b35a37")
    for index, dataset in enumerate(datasets):
        for sigma, color in zip(summary["sigmas"], noise_colors, strict=True):
            rows = [
                next(
                    row
                    for row in summary["tables"]
                    if row["dataset"] == dataset
                    and row["sigma"] == sigma
                    and row["collection"] == name
                )
                for name in COLLECTIONS
            ]
            style = "--" if sigma == 0 else "-"
            label = f"σ={sigma:g}"
            curve(
                axes[index, 0],
                range(3),
                [row["fixed_grit"] for row in rows],
                "missed_u",
                label,
                color,
                style,
            )
            for column, method in ((1, "fixed_grit"), (2, "cr_selected_worst_group")):
                curve(
                    axes[index, column],
                    range(3),
                    [row[method] for row in rows],
                    "both_shift",
                    label,
                    color,
                    style,
                    percent=True,
                )
        axes[index, 0].set_ylabel(f"{dataset.title()}\nMissed-U operator norm")
        axes[index, 0].set_title("GRIT r=2 geometry")
        axes[index, 1].set_title("GRIT r=2")
        axes[index, 2].set_title("CR selected")
        axes[index, 0].set_ylim(-0.04, 1.06)
        for column in (1, 2):
            axes[index, column].set_ylabel("Both-shift accuracy (%)")
            axes[index, column].set_ylim(-3, 103)
        for ax in axes[index]:
            ax.set_xticks(range(3), ["One", "Uneven", "Balanced"])
            ax.grid(alpha=0.15)
            ax.legend(fontsize=7)
    fig.tight_layout()
    save_figure(fig, output, "geometry")


def plot_rates(output: Path, rates: list[dict], summary: dict) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(8.0, 5.5), squeeze=False)
    for index, collection in enumerate(("balanced", "uneven")):
        for column, metric in enumerate(("missed_u", "covariance_error")):
            ax = axes[index, column]
            for sigma in sorted({row["sigma"] for row in rates}):
                selected = [
                    row
                    for row in rates
                    if row["collection"] == collection and row["sigma"] == sigma
                ]
                ks = sorted({row["k"] for row in selected})
                rows = [
                    {
                        metric: statistics(
                            [row[metric] for row in selected if row["k"] == k],
                            f"rate:{collection}:{sigma}:{metric}:{k}",
                        )
                    }
                    for k in ks
                ]
                slope = next(
                    row
                    for row in summary["rate_slopes"]
                    if row["collection"] == collection
                    and row["sigma"] == sigma
                    and row["metric"] == metric
                )
                curve(
                    ax,
                    ks,
                    rows,
                    metric,
                    f"σ={sigma:g}; slope={slope['mean']:.2f}",
                    log=True,
                )
            ax.set_xscale("log", base=2)
            ax.set_yscale("log")
            ax.set_xlabel("Pair count k")
            ax.set_ylabel(
                "Missed-U norm"
                if metric == "missed_u"
                else "Covariance fluctuation (op. norm)"
            )
            ax.set_title(LABELS[collection])
            ax.grid(alpha=0.15)
            ax.legend(fontsize=7)
    fig.tight_layout()
    save_figure(fig, output, "rate")


def plot_paths(output: Path, summary: dict, name: str) -> None:
    rows = summary["rank_paths" if name == "rank" else "cr_paths"]
    datasets = list(dict.fromkeys(row["dataset"] for row in rows))
    fig, axes = plt.subplots(
        len(datasets), 3, figsize=(10.0, 2.7 * len(datasets)), squeeze=False
    )
    if name == "rank":
        metrics = ("sensitivity", "label_removed", "both_shift")
        labels = (
            "Raw true-U sensitivity",
            "Stable label-direction removed",
            "Both-shift accuracy (%)",
        )
    else:
        metrics = ("stable_coefficient_norm", "sensitivity", "both_loss")
        labels = (
            "Stable coefficient norm",
            "Raw true-U sensitivity",
            "Both-shift logistic loss",
        )
    parameter = "rank" if name == "rank" else "alpha"
    for index, dataset in enumerate(datasets):
        selected = [row for row in rows if row["dataset"] == dataset]
        x = [row[parameter] for row in selected]
        for column, (metric, label) in enumerate(zip(metrics, labels, strict=True)):
            ax = axes[index, column]
            curve(
                ax,
                x,
                [row["metrics"] for row in selected],
                metric,
                dataset.title(),
                color="#177e89",
                percent=metric == "both_shift",
            )
            ax.set_ylabel(label)
            ax.set_xlabel("Requested rank r" if name == "rank" else "CR strength α")
            ax.set_title(dataset.title())
            ax.grid(alpha=0.15)
            if name == "rank":
                ax.set_xticks(x)
                ax.axvline(2, linestyle=":", color="gray", alpha=0.7)
            else:
                ax.set_xscale("symlog", linthresh=0.001)
                fixed = next(
                    row["fixed_grit"]
                    for row in summary["tables"]
                    if row["dataset"] == dataset
                    and row["sigma"] == summary["primary_sigma"]
                    and row["collection"] == "balanced"
                )
                ax.axhline(
                    fixed[metric]["mean"],
                    color="#b35a37",
                    linestyle="--",
                    label="GRIT r=2",
                )
                ax.legend(fontsize=7)
    fig.tight_layout()
    save_figure(fig, output, "rank" if name == "rank" else "cr-path")


def plot_bridge(output: Path, summary: dict) -> None:
    if not summary["bridge"]:
        return
    fig, axes = plt.subplots(2, 2, figsize=(8.2, 5.8))
    probe_names = {
        "erm": "ERM",
        "grit-r2": "GRIT r=2",
        "cr": "CR selected",
        "oracle": "True-U projection",
    }
    metrics = ("target_loss", "direct_absolute_loss_change", "bridge_term")
    labels = (
        "Weighted target logistic loss",
        "Absolute source-to-bridge loss change",
        "Bridge covariate-shift risk term",
    )
    colors = ("#555555", "#177e89", "#b35a37", "#725199")
    for (probe, label), color in zip(probe_names.items(), colors, strict=True):
        rows = sorted(
            [row for row in summary["bridge"] if row["probe"] == probe],
            key=lambda row: row["shift_scale"],
        )
        for ax, metric, ylabel in zip(axes.flat, metrics, labels, strict=False):
            curve(
                ax,
                [row["shift_scale"] for row in rows],
                [row["metrics"] for row in rows],
                metric,
                label,
                color,
            )
            ax.set_xlabel("Unstable shift scale")
            ax.set_ylabel(ylabel)
            ax.grid(alpha=0.15)
    ax = axes[1, 1]
    rows = [
        next(
            row["metrics"]
            for row in summary["bridge"]
            if row["probe"] == probe and row["shift_scale"] == 1.0
        )
        for probe in probe_names
    ]
    bottom = np.zeros(len(rows))
    for metric, label, color in (
        ("source_loss", "Source risk", "#d3d3d3"),
        ("bridge_term", "Bridge term", "#efc56b"),
        ("rho_times_sensitivity", "Direct-shift bound", "#7abec2"),
    ):
        values = np.array([row[metric]["mean"] for row in rows])
        ax.bar(range(len(rows)), values, bottom=bottom, label=label, color=color)
        bottom += values
    ax.scatter(
        range(len(rows)),
        [row["target_loss"]["mean"] for row in rows],
        marker="x",
        color="black",
        label="Target risk",
        zorder=3,
    )
    ax.set_xticks(range(len(rows)), ["ERM", "GRIT r=2", "CR", "True U"])
    ax.set_ylabel("Risk bound at shift scale 1")
    ax.legend(fontsize=7)
    axes[0, 0].legend(fontsize=7)
    fig.tight_layout()
    save_figure(fig, output, "bridge")


def write_reports(output: Path, summary: dict, config: dict) -> None:
    lines = [
        "# Collection geometry at a fixed pair budget",
        "",
        f"Each collection uses k={config['pairs']} pairs in "
        f"d={config['dimension']} dimensions with clean energy 2 per pair. "
        "Training data, targets, pair count and Gaussian noise draws are shared "
        "across collection arms. Noise is added after clean normalization; "
        "observed differences are not renormalized.",
        "",
        f"The primary noise SD is {config['primary_sigma']:g}; "
        f"all methods use fixed ridge {config['ridge']:g}. "
        "Clean and higher-noise controls are included below. Rank/CR paths use "
        "the primary noise level and balanced coverage.",
        "",
        "Numbers are replicate mean ± sample SD. Accuracy is in percent; "
        "all other metrics use raw units. CR strength is selected by worst "
        "validation-group accuracy. Fixed GRIT uses requested rank 2. "
        "Selection is frozen before test access. The digits setting uses "
        "engineered fixed features from real MNIST, not pretrained representations.",
        "",
        "| Dataset | σ | Collection | Missed U (GRIT r=2) | "
        "GRIT both-shift accuracy | CR both-shift accuracy |",
        "|---|---:|---|---:|---:|---:|",
    ]
    for row in summary["tables"]:
        lines.append(
            f"| {row['dataset']} | {row['sigma']:g} | {LABELS[row['collection']]} | "
            f"{formatted(row['fixed_grit']['missed_u'])} | "
            f"{formatted(row['fixed_grit']['both_shift'], percent=True)} | "
            f"{formatted(row['cr_selected_worst_group']['both_shift'], percent=True)} |"
        )
    lines += [
        "",
        "## Controls",
        "",
        "| Dataset | Control | Both-shift accuracy (%) | Both-shift logistic loss |",
        "|---|---|---:|---:|",
    ]
    for row in summary["baselines"]:
        lines.append(
            f"| {row['dataset']} | {row['method']} | "
            f"{formatted(row['metrics']['both_shift'], percent=True)} | "
            f"{formatted(row['metrics']['both_loss'])} |"
        )
    lines += [
        "",
        "## Paired balanced minus uneven contrasts",
        "",
        "Percentile 95% bootstrap intervals resample matching replicates "
        "(5,000 resamples, fixed seed). Digits intervals quantify split/bank "
        "variation on this fixed dataset. Accuracy contrasts are percentage points.",
        "",
        "| Dataset | σ | Method | Metric | Mean difference | 95% interval |",
        "|---|---:|---|---|---:|---:|",
    ]
    for row in summary["paired_balanced_minus_uneven"]:
        if row["metric"] not in ("both_shift", "missed_u"):
            continue
        factor = 100 if row["metric"] == "both_shift" else 1
        low, high = row["ci95_mean"]
        lines.append(
            f"| {row['dataset']} | {row['sigma']:g} | {row['method']} | "
            f"{row['metric']} | "
            f"{row['mean'] * factor:.3f} | [{low * factor:.3f}, {high * factor:.3f}] |"
        )
    lines += [
        "",
        "## Finite-pair slopes",
        "",
        "Log-log slopes are fit separately per replicate using "
        f"k≥{config['slope_min_k']}. Intervals are bootstrap intervals for mean slope. "
        "These fits diagnose empirical scaling, not theorem constants or proof "
        "validity. A saturated missed-subspace norm need not follow a −1/2 slope "
        "at the displayed pair counts.",
        "",
        "| Collection | σ | Error metric | Slope mean ± SD | Mean-slope 95% interval |",
        "|---|---:|---|---:|---:|",
    ]
    for row in summary["rate_slopes"]:
        low, high = row["ci95_mean"]
        lines.append(
            f"| {LABELS[row['collection']]} | {row['sigma']:g} | {row['metric']} | "
            f"{formatted(row)} | [{low:.3f}, {high:.3f}] |"
        )
    lines += [
        "",
        "## Predictor norm stress control",
        "",
        "Balanced GRIT with requested rank 2 is refit at ridge 1e-4. "
        "This diagnostic was specified before full test access; it illustrates "
        "that geometric recovery alone does not control risk when the predictor "
        "amplifies residual unstable directions.",
        "",
        "| Dataset | σ | Ridge | Coefficient norm | Raw U sensitivity | "
        "Both-shift accuracy (%) |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in summary["weak_ridge_stress"]:
        for ridge, metrics in (
            (row["ridge"], row["metrics"]),
            (config["ridge"], row["standard_ridge"]),
        ):
            lines.append(
                f"| {row['dataset']} | {row['sigma']:g} | {ridge:g} | "
                f"{formatted(metrics['coefficient_norm'])} | "
                f"{formatted(metrics['sensitivity'])} | "
                f"{formatted(metrics['both_shift'], percent=True)} |"
            )
    lines += [
        "",
        "## Figures and complete records",
        "",
        "- `geometry.png` / `.pdf`: coverage and target accuracy at fixed pair budget.",
        "- `rate.png` / `.pdf`: mean error ± seed SD and fitted slopes.",
        "- `rank.png` / `.pdf`: balanced noisy pairs; under/overselection, "
        "stable label removal and target accuracy.",
        "- `cr-path.png` / `.pdf`: balanced noisy pairs; stable coefficient "
        "shrinkage, true-U sensitivity and target loss.",
        "- `bridge.png` / `.pdf`: coupled synthetic source/bridge/target "
        "diagnostics when available.",
        "- `summary.json`: both selectors, all reported metrics, rank/strength "
        "selection counts, paired intervals and full slope records.",
        "- `results.json`, replicate frozen JSON/weights, `geometry-rates.json`, "
        "`manifest.json`: raw evidence and reproducibility information.",
        "",
        "The single-direction arm intentionally violates full coverage. "
        "No experiment certifies the general theory, universal GRIT dominance, "
        "or deployment validity. The bridge diagnostic is an exact empirical "
        "finite-sample calculation for a designed shift; its bound is not a "
        "population confidence certificate. Full paths and clean controls are "
        "retained even when they favor CR.",
        "",
    ]
    (output / "RESULTS.md").write_text("\n".join(lines))
    tex = [
        r"\begin{tabular}{llccc}",
        r"\toprule",
        r"Data & Collection & Missed $U$ $\downarrow$ & GRIT $r=2$ & CR \\",
        r"\midrule",
    ]
    for row in summary["tables"]:
        if row["sigma"] != config["primary_sigma"]:
            continue
        dataset = "Synthetic" if row["dataset"] == "synthetic" else "MNIST styles"
        cr_accuracy = row["cr_selected_worst_group"]["both_shift"]
        tex.append(
            f"{dataset} & {LABELS[row['collection']]} & "
            f"{formatted(row['fixed_grit']['missed_u'], tex=True)} & "
            f"{formatted(row['fixed_grit']['both_shift'], percent=True, tex=True)} & "
            f"{formatted(cr_accuracy, percent=True, tex=True)} " + r"\\"
        )
    tex += [
        r"\bottomrule",
        r"\end{tabular}",
        "% Accuracy: both cues reversed, percent, replicate mean +/- sample SD.",
        f"% Fixed k={config['pairs']}, clean energy=2/pair, "
        f"coordinate noise SD={config['primary_sigma']:g}, ridge={config['ridge']:g}.",
        "% CR uses worst-group validation selection.",
    ]
    (output / "main-table.tex").write_text("\n".join(tex) + "\n")


def make_report(
    output: Path, all_results: list[dict], rates: list[dict], config: dict
) -> None:
    """Produce reproducible summaries without fitting or selecting new models."""
    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    summary = make_summary(all_results, rates, config)
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    write_reports(output, summary, config)
    plot_geometry(output, summary)
    if rates:
        plot_rates(output, rates, summary)
    plot_paths(output, summary, "rank")
    plot_paths(output, summary, "cr")
    plot_bridge(output, summary)
