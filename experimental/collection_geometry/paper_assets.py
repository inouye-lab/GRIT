"""Generate appendix table fragments directly from the frozen run summaries."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def make_paper_assets(output: Path) -> None:
    summary = json.loads((output / "summary.json").read_text())

    def emit(name: str, columns: str, header: str, rows: list[str]) -> None:
        text = [
            rf"\begin{{tabular}}{{{columns}}}",
            r"\toprule",
            header + r" \\",
            r"\midrule",
            *rows,
            r"\bottomrule",
            r"\end{tabular}",
        ]
        (output / name).write_text("\n".join(text) + "\n")

    def f(stat: dict[str, Any], scale: float = 1.0, digits: int = 2) -> str:
        return f"${scale * stat['mean']:.{digits}f}\\pm{scale * stat['sd']:.{digits}f}$"

    rows: list[str] = []
    for row in summary["tables"]:
        data = "Synthetic" if row["dataset"] == "synthetic" else "MNIST"
        label = {"one": "One", "uneven": "15:1", "balanced": "Balanced"}[
            row["collection"]
        ]
        fixed = row["fixed_grit"]
        cr = row["cr_selected_worst_group"]
        rows.append(
            f"{data} & {row['sigma']:g} & {label} & {f(fixed['missed_u'], digits=3)} & "
            f"{f(fixed['both_shift'], 100, 1)} & {f(cr['both_shift'], 100, 1)} " + r"\\"
        )
    emit(
        "appendix-noise-table.tex",
        "lrlrrr",
        r"Data & $\sigma_\xi$ & Collection & Missed $\mathcal U$ & GRIT (\%) & CR (\%)",
        rows,
    )

    rows = []
    for row in summary["rate_slopes"]:
        # The report labels this field metric and keeps all individual slopes.
        label = "Balanced" if row["collection"] == "balanced" else "15:1"
        metric = "Subspace" if row["metric"] == "missed_u" else "Covariance"
        stats = row
        rows.append(
            f"{label} & {row['sigma']:g} & {metric} & {f(stats, digits=3)} & "
            f"$[{stats['ci95_mean'][0]:.3f}, {stats['ci95_mean'][1]:.3f}]$ " + r"\\"
        )
    emit(
        "appendix-rate-table.tex",
        "lrlrr",
        r"Collection & $\sigma_\xi$ & Error & Slope $\pm$ SD & 95\% interval",
        rows,
    )

    rows = []
    for row in summary["rank_paths"]:
        data = "Synthetic" if row["dataset"] == "synthetic" else "MNIST"
        m = row["metrics"]
        rows.append(
            f"{data} & {row['rank']} & {f(m['missed_u'], digits=3)} & "
            f"{f(m['label_removed'], 100, 1)} & {f(m['sensitivity'], digits=3)} & "
            f"{f(m['both_shift'], 100, 1)} " + r"\\"
        )
    emit(
        "appendix-rank-table.tex",
        "lrrrrr",
        r"Data & $r$ & Missed $\mathcal U$ & Label removed (\%) & "
        r"$\|P_{\mathcal U}w\|$ & Accuracy (\%)",
        rows,
    )

    rows = []
    labels = {
        "erm": "ERM",
        "oracle": r"True-$\mathcal U$",
        "grit-r1": "GRIT, $r=1$",
        "grit-r2": "GRIT, $r=2$",
        "grit-r16": "GRIT, $r=16$",
        "cr": "CR",
    }
    for row in summary["bridge"]:
        if row["shift_scale"] != 1 or row["probe"] not in labels:
            continue
        m = row["metrics"]
        cells = [
            f"${m[k]['mean']:.3f}$"
            for k in [
                "source_loss",
                "bridge_loss",
                "target_loss",
                "bridge_term",
                "rho_times_sensitivity",
            ]
        ]
        rows.append(labels[row["probe"]] + " & " + " & ".join(cells) + r" \\")
    emit(
        "appendix-bridge-table.tex",
        "lrrrrr",
        r"Probe & $R_p$ & $R_{\breve q}$ & $R_q$ & $C_{\rm br}$ & "
        r"$\rho_\Delta\|P_{\mathcal U}w\|$",
        rows,
    )


if __name__ == "__main__":
    import sys

    make_paper_assets(Path(sys.argv[1]))
