"""Compact, accessible plots of three separately controlled pair-information sweeps.

Rows contain already evaluated, frozen method choices. This module aggregates
replicates; it never fits a model or selects a configuration from test metrics.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any, TypeAlias, cast

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.ticker import (
    LogFormatterSciNotation,
    LogLocator,
    MaxNLocator,
    NullFormatter,
    ScalarFormatter,
)
from report import BOOTSTRAP_SEED, N_BOOTSTRAP
from report import (
    statistics as _statistics,  # pyright: ignore[reportUnknownVariableType]
)

# The frozen reporter returns JSON statistics; annotate its unparameterized return
# locally without modifying the original experiment or its bootstrap computation.
statistics = cast(Callable[[list[float], str], dict[str, Any]], _statistics)
# NumPy overloads and Matplotlib's **kwargs stubs leave these members partially
# unknown. Call-site ignores suppress only that external-stub diagnostic; local
# container types, member existence, arguments, and return checks remain enabled.
GroupKey: TypeAlias = tuple[str, str, float, str]
Identity: TypeAlias = tuple[str, str, float, str, int]
CurveStyle: TypeAlias = tuple[str, str, str, str]

SWEEPS = ("coverage", "noise", "pairs")
CONTROLS = ("nonspanning_control", "clean_control")
METHODS = ("grit", "cr", "erm", "oracle")
METHOD_STYLE = {
    "grit": ("GRIT", "#007C83", "-", "o"),
    "cr": ("CR", "#B45A2A", "--", "s"),
    "erm": ("ERM", "#626262", ":", "^"),
    "oracle": ("True-U projection", "#7861A8", "-.", "D"),
}
METRIC_STYLE = {
    "missed_u": ("Subspace error", "#007C83", "-", "o"),
    "normalized_sensitivity": ("Normalized instability", "#B45A2A", "--", "s"),
}
METRICS = (
    "missed_u",
    "sensitivity",
    "coefficient_norm",
    "normalized_sensitivity",
    "both_shift",
    "both_loss",
    "in_distribution",
    "source_loss",
    "covariance_error",
    "observed_trace",
    "alpha",
    "rank",
    "effective_cr_lambda",
)


def metrics_for(row: dict[str, Any]) -> dict[str, float]:
    """Extract metrics without treating test outcomes as selection information."""
    metrics = {key: float(row[key]) for key in METRICS if key in row}
    metrics.update(
        {
            key: float(value)
            for key, value in row.get("test", {}).items()
            if key in METRICS
        }
    )
    if row["method"] == "cr" and "alpha" in metrics and "observed_trace" in metrics:
        metrics["effective_cr_lambda"] = metrics["alpha"] / (
            2 * metrics["observed_trace"]
        )
    if "sensitivity" in metrics and "coefficient_norm" in metrics:
        if metrics["coefficient_norm"] <= 1e-12:
            raise ValueError("Normalized instability is undefined for a zero probe")
        metrics["normalized_sensitivity"] = (
            metrics["sensitivity"] / metrics["coefficient_norm"]
        )
    if row["method"] == "grit":
        needed = {"missed_u", "normalized_sensitivity", "both_shift"}
        if not needed.issubset(metrics):
            raise ValueError(f"Missing GRIT metrics: {needed - metrics.keys()}")
        if metrics["normalized_sensitivity"] > metrics["missed_u"] + 1e-9:
            raise ValueError(
                "Hard-projection sensitivity exceeds geometric upper bound"
            )
    if not all(np.isfinite(value) for value in metrics.values()):
        raise ValueError("Sweep metrics must be finite")
    return metrics


def positive_sweep_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Keep sigma=0 as a separately reported control, outside noisy-sweep fits."""
    return [row for row in rows if row["sweep"] != "noise" or row["value"] > 0]


def summarize_sweeps(
    rows: list[dict[str, Any]], config: dict[str, Any]
) -> dict[str, Any]:
    groups: dict[GroupKey, list[dict[str, Any]]] = defaultdict(list)
    unique: set[Identity] = set()
    for row in rows:
        if row["sweep"] not in SWEEPS + CONTROLS or row["method"] not in METHODS:
            raise ValueError("Unrecognized sweep or method; pass frozen method choices")
        identity: Identity = (
            row["dataset"],
            row["sweep"],
            row["value"],
            row["method"],
            row["replicate"],
        )
        if identity in unique:
            raise ValueError(f"Duplicate replicate/method/point: {identity}")
        unique.add(identity)
        item = dict(row, metrics=metrics_for(row))
        groups[identity[:-1]].append(item)
    summary: dict[str, Any] = {
        "groups": [],
        "paired_endpoints": [],
        "slopes": [],
        "uncertainty": {
            "ribbon": "mean plus/minus sample SD across independent replicates",
            "interval": "paired percentile bootstrap of replicate differences",
            "bootstrap_resamples": N_BOOTSTRAP,
            "bootstrap_seed": BOOTSTRAP_SEED,
            "digits_scope": "split/construction variability on the fixed MNIST dataset",
        },
        "slope_min_k": config.get("slope_min_k", 512),
        "config": config,
        "guide": {
            "anchor_k": 256,
            "anchor_metric": "missed_u",
            "exponent": -0.5,
            "scope": "Shape guide, not a calibrated bound",
        },
    }
    for key, members in sorted(groups.items()):
        dataset, sweep, value, method = key
        for field in ("k", "sigma", "lambda_min", "lambda_max"):
            if not all(
                np.isclose(  # pyright: ignore[reportUnknownMemberType]
                    member[field], members[0][field], rtol=0, atol=1e-12
                )
                for member in members
            ):
                raise ValueError(f"Inconsistent {field} within one sweep point")
        names = set[str].intersection(
            *(set[str](member["metrics"]) for member in members)
        )
        summary["groups"].append(
            {
                "dataset": dataset,
                "sweep": sweep,
                "value": value,
                "method": method,
                "replicates": sorted(member["replicate"] for member in members),
                "k": members[0]["k"],
                "sigma": members[0]["sigma"],
                "lambda_min": members[0]["lambda_min"],
                "lambda_max": members[0]["lambda_max"],
                "metrics": {
                    metric: statistics(
                        [m["metrics"][metric] for m in members], f"sweep:{key}:{metric}"
                    )
                    for metric in sorted(names)
                },
            }
        )
    combinations = sorted({key[:2] + (key[3],) for key in groups})
    for dataset, sweep, method in combinations:
        if sweep not in SWEEPS:
            continue
        selected = positive_sweep_rows(
            [
                row
                for row in summary["groups"]
                if row["dataset"] == dataset
                and row["sweep"] == sweep
                and row["method"] == method
            ]
        )
        values = sorted(row["value"] for row in selected)
        if len(values) < 2:
            continue
        first = {
            row["replicate"]: row["metrics"]
            for row in groups[(dataset, sweep, values[0], method)]
        }
        last = {
            row["replicate"]: row["metrics"]
            for row in groups[(dataset, sweep, values[-1], method)]
        }
        if first.keys() != last.keys():
            raise ValueError("Endpoint contrasts require exactly matching replicates")
        for metric in ("missed_u", "normalized_sensitivity", "both_shift", "both_loss"):
            if not all(metric in row for row in [*first.values(), *last.values()]):
                continue
            differences = [
                last[rep][metric] - first[rep][metric] for rep in sorted(first)
            ]
            summary["paired_endpoints"].append(
                {
                    "dataset": dataset,
                    "sweep": sweep,
                    "method": method,
                    "metric": metric,
                    "from_value": values[0],
                    "to_value": values[-1],
                    **statistics(
                        differences, f"endpoint:{dataset}:{sweep}:{method}:{metric}"
                    ),
                }
            )
    for dataset in sorted({row["dataset"] for row in rows}):
        selected = [
            row
            for row in rows
            if row["dataset"] == dataset
            and row["sweep"] == "pairs"
            and row["method"] == "grit"
            and row["k"] >= summary["slope_min_k"]
        ]
        for metric in ("missed_u", "normalized_sensitivity"):
            fitted: list[dict[str, Any]] = []
            for replicate in sorted({row["replicate"] for row in selected}):
                members = sorted(
                    [row for row in selected if row["replicate"] == replicate],
                    key=lambda row: row["k"],
                )
                if len(members) < 2:
                    continue
                y = np.array([metrics_for(row)[metric] for row in members])
                if np.any(y <= 0):  # pyright: ignore[reportUnknownMemberType]
                    raise ValueError("Log-log slopes require positive measurements")
                slope = np.polyfit(np.log([row["k"] for row in members]), np.log(y), 1)[
                    0
                ]
                fitted.append({"replicate": replicate, "slope": float(slope)})
            if fitted:
                summary["slopes"].append(
                    {
                        "dataset": dataset,
                        "metric": metric,
                        "minimum_k": summary["slope_min_k"],
                        "k_values": sorted({row["k"] for row in selected}),
                        "replicate_slopes": fitted,
                        **statistics(
                            [item["slope"] for item in fitted],
                            f"sweep-slope:{dataset}:{metric}",
                        ),
                        "scope": (
                            "Empirical subspace-error rate"
                            if metric == "missed_u"
                            else "Descriptive fitted-predictor rate; no rate guarantee"
                        ),
                    }
                )
    return summary


def panel_rows(
    summary: dict[str, Any], dataset: str, sweep: str, method: str
) -> list[dict[str, Any]]:
    return sorted(
        [
            row
            for row in summary["groups"]
            if row["dataset"] == dataset
            and row["sweep"] == sweep
            and row["method"] == method
        ],
        key=lambda row: row["value"],
    )


def draw_curve(
    ax: Axes,
    rows: list[dict[str, Any]],
    metric: str,
    style: CurveStyle,
    *,
    percent: bool = False,
    logarithmic: bool = False,
) -> None:
    label, color, linestyle, marker = style
    factor = 100 if percent else 1
    noisy = positive_sweep_rows(rows)
    x = [row["value"] for row in noisy]
    mean = np.array([row["metrics"][metric]["mean"] for row in noisy]) * factor
    sd = np.array([row["metrics"][metric]["sd"] for row in noisy]) * factor
    ax.plot(  # pyright: ignore[reportUnknownMemberType]
        x,
        mean,
        label=label,
        color=color,
        linestyle=linestyle,
        marker=marker,
        linewidth=1.35,
        markersize=3.0,
        markeredgewidth=0.5,
    )
    lower = np.maximum(mean - sd, 1e-12 if logarithmic else 0)
    upper = np.minimum(mean + sd, 100 if percent else 1)
    ax.fill_between(  # pyright: ignore[reportUnknownMemberType]
        x, lower, upper, color=color, alpha=0.13, linewidth=0
    )
    clean = [row for row in rows if row["sweep"] == "noise" and row["value"] == 0]
    if clean:
        value = clean[0]["metrics"][metric]
        ax.errorbar(  # pyright: ignore[reportUnknownMemberType]
            [0],
            [factor * value["mean"]],
            yerr=[factor * value["sd"]],
            color=color,
            marker=marker,
            markerfacecolor="white",
            markersize=4,
            markeredgewidth=0.8,
            linestyle="none",
            linewidth=0.7,
            capsize=1.5,
        )


def format_axes(axes: Sequence[Axes], *, accuracy: bool) -> None:
    xlabels = (
        r"(a) Smallest eigenvalue $\lambda_{\min}$",
        r"(b) Noise SD $\sigma_\xi$",
        r"(c) Pair count $k$",
    )
    for index, ax in enumerate(axes):
        ax.set_xlabel(  # pyright: ignore[reportUnknownMemberType]
            xlabels[index], fontsize=7.5, labelpad=2
        )
        ax.tick_params(  # pyright: ignore[reportUnknownMemberType]
            labelsize=7, length=2, pad=2
        )
        ax.grid(alpha=0.17, linewidth=0.5)  # pyright: ignore[reportUnknownMemberType]
        ax.spines[["top", "right"]].set_visible(False)
        if index != 2 or accuracy:
            ax.set_ylim(-2 if accuracy else -0.025, 102 if accuracy else 1.025)
            ax.set_yticks(  # pyright: ignore[reportUnknownMemberType]
                [0, 50, 100] if accuracy else [0, 0.5, 1]
            )
        if index == 0:
            ax.set_xscale("log", base=2)  # pyright: ignore[reportUnknownMemberType]
            ax.set_xticks(  # pyright: ignore[reportUnknownMemberType]
                [1 / 64, 1 / 16, 1 / 4, 1], [r"$2^{-6}$", r"$2^{-4}$", r"$2^{-2}$", "1"]
            )
        elif index == 1:
            ax.set_xticks(  # pyright: ignore[reportUnknownMemberType]
                [0, 0.25, 0.5, 0.75, 1.0], ["0", ".25", ".5", ".75", "1"]
            )
        else:
            ax.set_xscale("log", base=2)  # pyright: ignore[reportUnknownMemberType]
            ax.set_xticks(  # pyright: ignore[reportUnknownMemberType]
                [64, 256, 1024, 4096], ["64", "256", "1024", "4096"]
            )
            if not accuracy:
                ax.set_yscale("log")  # pyright: ignore[reportUnknownMemberType]
        if index == 1:
            ax.set_yticklabels([])  # pyright: ignore[reportUnknownMemberType]
    axes[0].set_ylabel(  # pyright: ignore[reportUnknownMemberType]
        "Target accuracy (%)" if accuracy else "Error / instability",
        fontsize=7.5,
        labelpad=2,
    )


def save_plots(
    output: Path, summary: dict[str, Any], dataset: str, *, accuracy: bool
) -> None:
    fig, raw_axes = plt.subplots(1, 3, figsize=(5.5, 1.9))
    axes = cast(Sequence[Axes], raw_axes)
    for index, sweep in enumerate(SWEEPS):
        if accuracy:
            for method in METHODS:
                draw_curve(
                    axes[index],
                    panel_rows(summary, dataset, sweep, method),
                    "both_shift",
                    METHOD_STYLE[method],
                    percent=True,
                )
        else:
            rows = panel_rows(summary, dataset, sweep, "grit")
            for metric, style in METRIC_STYLE.items():
                draw_curve(axes[index], rows, metric, style, logarithmic=index == 2)
            if sweep == "pairs":
                anchor = next(row for row in rows if row["k"] == 256)
                x = np.array([row["k"] for row in rows], dtype=float)
                guide = anchor["metrics"]["missed_u"]["mean"] * np.sqrt(256 / x)
                axes[index].plot(  # pyright: ignore[reportUnknownMemberType]
                    x,
                    guide,
                    color="#252525",
                    linestyle=":",
                    linewidth=1.2,
                    label=r"$k^{-1/2}$ shape guide",
                    zorder=4,
                )
    format_axes(axes, accuracy=accuracy)
    handles, labels = axes[0].get_legend_handles_labels()
    if not accuracy:
        handle, label = axes[2].get_legend_handles_labels()
        handles.append(handle[-1])
        labels.append(label[-1])
    fig.legend(  # pyright: ignore[reportUnknownMemberType]
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.52, 1.025),
        ncol=4 if accuracy else 3,
        frameon=False,
        fontsize=7,
        handlelength=2.2,
        handletextpad=0.35,
        columnspacing=0.9,
    )
    fig.subplots_adjust(left=0.08, right=0.995, bottom=0.22, top=0.75, wspace=0.26)
    name = f"sweep-{'accuracy-' if accuracy else 'instability-'}{dataset}"
    fig.savefig(  # pyright: ignore[reportUnknownMemberType]
        output / f"{name}.png", dpi=240, bbox_inches="tight", pad_inches=0.02
    )
    fig.savefig(  # pyright: ignore[reportUnknownMemberType]
        output / f"{name}.pdf",
        bbox_inches="tight",
        pad_inches=0.02,
        metadata={"Creator": "GRIT controlled sweep report", "CreationDate": None},
    )
    plt.close(fig)


def save_stacked_plot(
    output: Path,
    summary: dict[str, Any],
    dataset: str,
    geometry_variants: dict[str, Any] | None = None,
) -> None:
    """Align geometric recovery with held-out accuracy without mixing y metrics."""
    fig, raw_axes = plt.subplots(2, 3, figsize=(5.5, 2.55), sharex="col")
    axes = cast(Sequence[Sequence[Axes]], raw_axes)
    top, bottom = axes
    all_top_rows: dict[str, list[dict[str, Any]]] = {}
    for index, sweep in enumerate(SWEEPS):
        rows = panel_rows(summary, dataset, sweep, "grit")
        if geometry_variants is None:
            all_top_rows[sweep] = rows
            draw_curve(
                top[index], rows, "missed_u", METHOD_STYLE["grit"],
                logarithmic=index == 2,
            )
        else:
            varying = "k" if sweep == "noise" else "sigma"
            groups = [
                group for group in geometry_variants["groups"]
                if group["dataset"] == dataset and group["sweep"] == sweep
            ]
            values = sorted({group[varying] for group in groups})
            all_top_rows[sweep] = []
            for value, marker, linestyle in zip(
                values, ("^", "o", "s"), ("--", "-", "-."), strict=True
            ):
                curve = [
                    dict(group, metrics={"missed_u": group["missed_u"]})
                    for group in sorted(groups, key=lambda group: group["value"])
                    if group[varying] == value
                ]
                all_top_rows[sweep].extend(curve)
                label = (
                    rf"$k={value:g}$" if varying == "k"
                    else rf"$\sigma_\xi={value:g}$"
                )
                draw_curve(
                    top[index], curve, "missed_u",
                    (label, "#007C83", linestyle, marker), logarithmic=index == 2,
                )
            handles, labels = top[index].get_legend_handles_labels()
            top[index].legend(  # pyright: ignore[reportUnknownMemberType]
                handles, labels, loc="upper left" if index == 1 else "upper right",
                fontsize=5.5, handlelength=1.8, handletextpad=0.3,
                labelspacing=0.2, borderpad=0.25, borderaxespad=0.25,
                framealpha=0.9, edgecolor="none",
            )
        for method in ("grit", "cr"):
            draw_curve(
                bottom[index], panel_rows(summary, dataset, sweep, method),
                "both_shift", METHOD_STYLE[method], percent=True,
            )
        if sweep == "pairs":
            anchor = next(row for row in rows if row["k"] == 256)
            x = np.array([row["k"] for row in rows], dtype=float)
            top[index].plot(  # pyright: ignore[reportUnknownMemberType]
                x, anchor["metrics"]["missed_u"]["mean"] * np.sqrt(256 / x),
                color="#252525", linestyle=":", linewidth=1.2,
                label=r"$k^{-1/2}$ guide", zorder=4,
            )
    format_axes(top, accuracy=False)
    format_axes(bottom, accuracy=True)
    # Use every displayed method and its SD ribbon when choosing each y range.
    # Only x axes are shared; each panel keeps its own explicit y tick labels.
    for index, sweep in enumerate(SWEEPS):
        for ax, metric, methods, factor in (
            (top[index], "missed_u", ("grit",), 1.0),
            (bottom[index], "both_shift", ("grit", "cr"), 100.0),
        ):
            metric_rows = (
                all_top_rows[sweep] if metric == "missed_u"
                else [
                    row for method in methods
                    for row in panel_rows(summary, dataset, sweep, method)
                ]
            )
            stats = [row["metrics"][metric] for row in metric_rows]
            lower = min(max(0.0, item["mean"] - item["sd"]) for item in stats) * factor
            upper = max(min(1.0, item["mean"] + item["sd"]) for item in stats) * factor
            if ax.get_yscale() == "log":
                padding = (upper / lower) ** 0.08
                ax.set_ylim(lower / padding, upper * padding)
                ax.yaxis.set_major_locator(LogLocator(base=10))
                ax.yaxis.set_major_formatter(LogFormatterSciNotation())
                ax.yaxis.set_minor_formatter(NullFormatter())
            else:
                padding = max(upper - lower, factor * 0.01) * 0.08
                ax.set_ylim(lower - padding, upper + padding)
                ax.yaxis.set_major_locator(MaxNLocator(nbins=3, min_n_ticks=3))
                ax.yaxis.set_major_formatter(ScalarFormatter())
    for ax in top:
        ax.set_xlabel("")  # pyright: ignore[reportUnknownMemberType]
        ax.tick_params(labelbottom=False)  # pyright: ignore[reportUnknownMemberType]
    top[0].set_ylabel(  # pyright: ignore[reportUnknownMemberType]
        "Subspace error", fontsize=7.5, labelpad=2,
    )
    handles, labels = bottom[0].get_legend_handles_labels()
    guide_handles, guide_labels = top[2].get_legend_handles_labels()
    fig.legend(  # pyright: ignore[reportUnknownMemberType]
        handles + [guide_handles[-1]], labels + [guide_labels[-1]],
        loc="upper center", bbox_to_anchor=(0.53, 1.015), ncol=3,
        frameon=False, fontsize=7, handlelength=2.2, handletextpad=0.35,
        columnspacing=1.3,
    )
    fig.subplots_adjust(
        left=0.095, right=0.99, bottom=0.18, top=0.90, wspace=0.36, hspace=0.15,
    )
    for suffix in ("png", "pdf"):
        fig.savefig(  # pyright: ignore[reportUnknownMemberType]
            output / f"sweep-{dataset}.{suffix}", dpi=240,
            bbox_inches="tight", pad_inches=0.025,
            metadata={"Creator": "GRIT controlled sweep report", "CreationDate": None}
            if suffix == "pdf" else None,
        )
    plt.close(fig)


def caption(
    summary: dict[str, Any], dataset: str, *,
    accuracy: bool = False, secondary: bool = True,
) -> str:
    """Keep interpretation in captions and technical specifications in the setup."""
    name = "CueMNIST" if dataset == "digits" else "the Gaussian synthetic task"
    if accuracy:
        return (
            "\\textbf{Better pair information improves GRIT's target accuracy; "
            "CR is competitive in some conditions.} "
            f"Results on {name}; ERM and true-subspace projection provide controls. "
            "Each column varies one quantity from defaults $k=256$, "
            "$\\sigma_\\xi=0.1$, and clean spectrum $(1,1)$. "
            "Lines and bands show means and sample SD."
        )
    n = len(panel_rows(summary, dataset, "coverage", "grit")[0]["replicates"])
    secondary_note = (
        "Top-panel legends vary a second setting; bottom panels use the defaults. "
        if secondary else "Each column varies one quantity from these defaults. "
    )
    return (
        "\\textbf{Stronger signal in the weakest direction and less noise improve "
        "recovery and target accuracy; more pairs yield near-$k^{-1/2}$ subspace error "
        "and a GRIT advantage over CR at larger budgets.} "
        f"On {name}, the top row shows GRIT's subspace error; "
        "the bottom compares target accuracy. Defaults are $k=256$, "
        "$\\sigma_\\xi=0.1$, and clean spectrum $(1,1)$. "
        f"{secondary_note}Bands show sample SD across {n} replicates; "
        "the dotted line is a shape guide. "
        "Y-axis ranges differ by panel."
    )


def instability_caption(summary: dict[str, Any], dataset: str) -> str:
    name = "CueMNIST" if dataset == "digits" else "the Gaussian synthetic task"
    return (
        "\\textbf{GRIT's fitted instability follows the recovery trends and "
        "decreases faster than subspace error as pair count grows.} "
        f"Mirror of the three sweeps on {name}, adding normalized predictor "
        "sensitivity to the subspace-error curves. Defaults are $k=256$, "
        "$\\sigma_\\xi=0.1$, and clean spectrum $(1,1)$; bands show sample SD. "
        "The dotted $k^{-1/2}$ line describes subspace recovery, "
        "not a predictor-rate guarantee."
    )


def write_outputs(output: Path, summary: dict[str, Any]) -> None:
    (output / "sweep-summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    lines = [
        "# Controlled sweeps of pair information",
        "",
        "All methods and CR choices were frozen before test evaluation. "
        "Plots report replicate means with sample-SD ribbons. "
        "Zero-noise and nonspanning controls are tabulated separately.",
        "",
        "Coverage holds the largest clean eigenvalue fixed, so clean trace "
        "varies. Noise and pair-count sweeps hold both clean eigenvalues fixed.",
        "",
        "## Paired endpoint contrasts",
        "",
        "Differences are last minus first positive-noise sweep point, or last "
        "minus first coverage/pair-count point. Intervals use 5,000 fixed-seed "
        "paired bootstrap resamples. Accuracy changes are percentage points.",
        "",
        "| Data | Sweep | Method | Metric | From → to | Mean change | 95% CI |",
        "|---|---|---|---|---|---:|---:|",
    ]
    for row in summary["paired_endpoints"]:
        factor = 100 if row["metric"] == "both_shift" else 1
        low, high = row["ci95_mean"]
        lines.append(
            f"| {row['dataset']} | {row['sweep']} | {row['method']} | "
            f"{row['metric']} | {row['from_value']:g} → {row['to_value']:g} | "
            f"{row['mean'] * factor:.4f} | "
            f"[{low * factor:.4f}, {high * factor:.4f}] |"
        )
    lines += [
        "",
        "## Pair-count log-log slopes",
        "",
        "Fits use each replicate separately at prespecified k≥512. "
        "Normalized-sensitivity slopes describe fitted predictors, not a "
        "guaranteed accuracy or sensitivity rate.",
        "",
        "| Data | Metric | Mean slope | Sample SD | Mean-slope 95% CI |",
        "|---|---|---:|---:|---:|",
    ]
    for row in summary["slopes"]:
        low, high = row["ci95_mean"]
        lines.append(
            f"| {row['dataset']} | {row['metric']} | {row['mean']:.4f} | "
            f"{row['sd']:.4f} | [{low:.4f}, {high:.4f}] |"
        )
    lines += [
        "",
        "## Separate controls",
        "",
        "Control points do not enter sweep curves, endpoint contrasts, or slopes.",
        "| Data | Control | Method | Missed U | Normalized instability | Accuracy (%) "
        "|",
        "|---|---|---|---:|---:|---:|",
    ]
    for row in summary["groups"]:
        if row["sweep"] not in CONTROLS:
            continue
        values: list[str] = []
        for metric in ("missed_u", "normalized_sensitivity", "both_shift"):
            stats = row["metrics"].get(metric)
            if stats is None:
                values.append("—")
            else:
                factor = 100 if metric == "both_shift" else 1
                values.append(
                    f"{stats['mean'] * factor:.3f} ± {stats['sd'] * factor:.3f}"
                )
        lines.append(
            f"| {row['dataset']} | {row['sweep']} | {row['method']} | "
            + " | ".join(values)
            + " |"
        )
    lines += [
        "",
        "No test outcome selects a configuration in this report. "
        "The shape guide is not a fitted theorem constant or a confidence bound. "
        "No result certifies the general theorem or a universal method ordering.",
        "",
    ]
    (output / "SWEEP-RESULTS.md").write_text("\n".join(lines))
    with (output / "sweep-table.tsv").open("w", newline="") as stream:
        writer = csv.writer(stream, delimiter="\t")
        writer.writerow(
            [
                "dataset",
                "sweep",
                "value",
                "method",
                "metric",
                "n",
                "mean",
                "sd",
                "ci95_low",
                "ci95_high",
            ]
        )
        for row in summary["groups"]:
            for metric, stats in row["metrics"].items():
                writer.writerow(
                    [
                        row["dataset"],
                        row["sweep"],
                        row["value"],
                        row["method"],
                        metric,
                        stats["n"],
                        stats["mean"],
                        stats["sd"],
                        *stats["ci95_mean"],
                    ]
                )


def make_sweep_report(
    output: Path,
    rows: list[dict[str, Any]],
    config: dict[str, Any],
    geometry_variants: dict[str, Any] | None = None,
) -> None:
    """Create main CueMNIST and synthetic appendix figures from saved sweep rows."""
    plt.rcParams.update({"font.size": 7.5, "pdf.fonttype": 42, "ps.fonttype": 42})
    summary = summarize_sweeps(rows, config)
    if geometry_variants is None and (output / "geometry-variants.json").exists():
        geometry_variants = json.loads((output / "geometry-variants.json").read_text())
    write_outputs(output, summary)
    for dataset in sorted({row["dataset"] for row in rows}):
        save_stacked_plot(output, summary, dataset, geometry_variants)
        (output / f"caption-{dataset}.tex").write_text(
            caption(summary, dataset, secondary=geometry_variants is not None) + "\n"
        )
        save_plots(output, summary, dataset, accuracy=False)
        (output / f"caption-instability-{dataset}.tex").write_text(
            instability_caption(summary, dataset) + "\n"
        )
        save_plots(output, summary, dataset, accuracy=True)
        (output / f"caption-accuracy-{dataset}.tex").write_text(
            caption(summary, dataset, accuracy=True) + "\n"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Replot frozen sweep results; no fitting or selection"
    )
    parser.add_argument(
        "run", type=Path, help="Existing run with raw results and config"
    )
    parser.add_argument("output", type=Path, help="Fresh presentation output directory")
    parser.add_argument(
        "--geometry-variants", type=Path,
        help="Supplemental geometry JSON; defaults to the file in the input run",
    )
    args = parser.parse_args()
    variants_path = args.geometry_variants or (args.run / "geometry-variants.json")
    if args.output.exists() and any(args.output.iterdir()):
        parser.error(
            "Output directory must be empty; frozen results are never overwritten"
        )
    args.output.mkdir(parents=True, exist_ok=True)
    make_sweep_report(
        args.output,
        json.loads((args.run / "sweep-results.json").read_text()),
        json.loads((args.run / "config.json").read_text()),
        json.loads(variants_path.read_text()) if variants_path.exists() else None,
    )
