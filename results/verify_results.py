"""Recompute the manuscript's reported values from the included result records.

    python3 results/verify_results.py              # standard library only
    uv run python results/verify_results.py        # also checks configs and plans

Reads only `results/` and the saved controlled sweeps in
`experimental/collection_geometry/`. Nothing is fitted or selected and no test data are
read. Each check prints the value transcribed from the manuscript next to the value
recomputed from per-seed or per-replicate records. FAIL marks a reported value that
the records do not reproduce; FLAG marks a statement in the text that the records do
not support as worded. The exit status is nonzero only if a check FAILs.

The ColoredMNIST-INV (`cmnist/`) and Waterbirds-INV (`waterbirds/`) trees are trimmed
to the methods the paper reports; `TRIMMING.json` records what was removed and the
hashes the checks below confirm.

With the `grit` package importable, the script also confirms that the shipped YAML
configurations are the configurations behind these (trimmed) records, that the
shipped code regenerates the recorded candidate grids, summaries and UrbanCars
bindings.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import statistics as st
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
SWEEPS = ROOT / "experimental" / "collection_geometry"
T975_DF9 = 2.2621571627982  # Student-t 0.975 quantile, 9 degrees of freedom
FINAL_SEEDS = list(range(3101, 3111))
DISPLAY = {"cmnist": "ColoredMNIST-INV", "waterbirds": "Waterbirds-INV"}

counts = {"PASS": 0, "FAIL": 0, "FLAG": 0}


def report(status: str, section: str, item: str, detail: str) -> None:
    counts[status] += 1
    print(f"{status:4s}  {section:10s} {item:58s} {detail}")


def check(section: str, item: str, expected: Any, got: Any) -> None:
    status = "PASS" if expected == got else "FAIL"
    source = "expected" if section in ("Integrity", "Package", "Inputs") else "manuscript"
    report(status, section, item, f"{source} {expected}  records {got}")


def flag(section: str, item: str, detail: str) -> None:
    report("FLAG", section, item, detail)


def load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def mean_sd(values: list[float]) -> tuple[float, float]:
    return 100 * st.fmean(values), 100 * st.stdev(values)


def paired(a: list[float], b: list[float]) -> tuple[float, float, float]:
    d = [x - y for x, y in zip(a, b, strict=True)]
    m, se = st.fmean(d), st.stdev(d) / math.sqrt(len(d))
    return 100 * m, 100 * (m - T975_DF9 * se), 100 * (m + T975_DF9 * se)


def r(values: tuple[float, ...] | float, digits: int) -> Any:
    if isinstance(values, tuple):
        return tuple(round(v, digits) for v in values)
    return round(values, digits)


def canonical_digest(value: object) -> str:
    payload = json.dumps(
        value, allow_nan=False, ensure_ascii=False, separators=(",", ":"), sort_keys=True
    )
    return "sha256:" + hashlib.sha256(payload.encode("utf-8")).hexdigest()


# --------------------------------------------------------------------------- Tables 1, 3
OBJECTIVES = ("ERM", "SD", "RDM", "IRMv1", "V-REx", "Fishr", "GroupDRO", "LISA")
# (vanilla, +PC, +GRIT) means from Table 1, then sample SDs from Table 3.
TABLE1 = {
    "cmnist": {
        "ERM": (17.02, 57.16, 67.89),
        "SD": (17.02, 56.68, 67.77),
        "RDM": (39.95, 59.55, 67.60),
        "IRMv1": (59.89, 61.28, 70.27),
        "V-REx": (54.83, 59.98, 68.20),
        "Fishr": (65.28, 65.33, 68.91),
        "GroupDRO": (30.90, 61.51, 68.01),
        "LISA": (36.68, 61.62, 67.80),
    },
    "waterbirds": {
        "ERM": (52.69, 66.84, 70.09),
        "SD": (52.06, 66.82, 70.09),
        "RDM": (52.80, 78.59, 76.29),
        "IRMv1": (47.78, 77.86, 78.78),
        "V-REx": (48.27, 78.09, 80.99),
        "Fishr": (47.16, 76.72, 80.84),
        "GroupDRO": (79.83, 81.66, 81.91),
        "LISA": (79.65, 80.77, 82.30),
    },
}
TABLE3_SD = {
    "cmnist": {
        "ERM": (2.68, 0.47, 0.55),
        "SD": (2.68, 0.31, 0.62),
        "RDM": (3.47, 0.89, 0.58),
        "IRMv1": (2.71, 1.19, 0.41),
        "V-REx": (3.45, 1.52, 0.13),
        "Fishr": (1.74, 1.61, 0.18),
        "GroupDRO": (2.46, 0.51, 0.43),
        "LISA": (3.92, 0.68, 0.22),
    },
    "waterbirds": {
        "ERM": (0.49, 1.84, 1.19),
        "SD": (1.12, 1.82, 1.19),
        "RDM": (1.65, 0.91, 1.56),
        "IRMv1": (1.26, 1.04, 0.82),
        "V-REx": (0.67, 1.39, 0.94),
        "Fishr": (0.80, 0.90, 0.61),
        "GroupDRO": (1.75, 0.80, 0.54),
        "LISA": (1.28, 0.89, 0.73),
    },
}
CMNIST_TREES = {
    "ERM": ("cmnist-erm-interventions-representation-v1-e03", "erm"),
    "SD": ("cmnist-sd-interventions-representation-v1-e03", "sd"),
    "RDM": ("cmnist-rdm-interventions-representation-v1-e03", "rdm"),
    "IRMv1": ("cmnist-irm-interventions-representation-v1-e03", "irm"),
    "V-REx": ("cmnist-rex-interventions-representation-v1-e03", "rex"),
    "Fishr": ("cmnist-fishr-interventions-representation-v1-e03", "fishr"),
    "GroupDRO": (
        "cmnist-groupdro-label-source-env-interventions-representation-v1-e03",
        "groupdro",
    ),
    "LISA": ("cmnist-lisa-source-env-interventions-representation-v1-e03", "lisa"),
}
WATERBIRDS_TREES = {
    name: (f"waterbirds-{base}-interventions-representation-v1", base)
    for name, base in (
        ("ERM", "erm"),
        ("SD", "sd"),
        ("RDM", "rdm"),
        ("IRMv1", "irm"),
        ("V-REx", "rex"),
        ("Fishr", "fishr"),
        ("GroupDRO", "groupdro"),
        ("LISA", "lisa"),
    )
}


def method_ids(base: str) -> tuple[str, str, str]:
    if base == "erm":
        return ("erm", "erm_consistency", "grit")
    return (base, f"{base}_consistency", f"{base}_grit")


def cmnist_values(tree: str, method: str) -> list[float]:
    summary = load(RESULTS / "cmnist" / tree / "summaries" / "cmnist-summary.json")
    for entry in summary["methods"]:
        if entry["method_id"] == method and entry["selector"] == "primary_robust":
            observations = sorted(entry["final_observations"], key=lambda o: o["seed"])
            assert [o["seed"] for o in observations] == FINAL_SEEDS
            return [o["test_ood_accuracy"] for o in observations]
    raise KeyError(method)


def waterbirds_values(tree: str, method: str) -> list[float]:
    summary = load(RESULTS / "waterbirds" / tree / "summaries" / "waterbirds-summary.json")
    for entry in summary["methods"]:
        if entry["method_id"] == method:
            assert entry["selector"] == "waterbirds_validation_worst_group"
            pairs = sorted(entry["worst_group_by_seed"])
            assert [p[0] for p in pairs] == FINAL_SEEDS
            return [p[1] for p in pairs]
    raise KeyError(method)


def check_core_tables() -> None:
    values: dict[tuple[str, str, int], list[float]] = {}
    for dataset, trees, reader in (
        ("cmnist", CMNIST_TREES, cmnist_values),
        ("waterbirds", WATERBIRDS_TREES, waterbirds_values),
    ):
        lineages = set()
        for name in OBJECTIVES:
            tree, base = trees[name]
            plan = load(RESULTS / dataset / tree / "search-plan.json")
            lineages.add(json.dumps(plan["resolved_config"]["lineage"], sort_keys=True))
            for column, method in enumerate(method_ids(base)):
                seeds = reader(tree, method)
                values[(dataset, name, column)] = seeds
                got_mean, got_sd = mean_sd(seeds)
                label = f"{DISPLAY[dataset]} {name} {('vanilla', '+PC', '+GRIT')[column]}"
                check(
                    "Table 1/3",
                    f"{label} mean, s.d.",
                    (TABLE1[dataset][name][column], TABLE3_SD[dataset][name][column]),
                    (r(got_mean, 2), r(got_sd, 2)),
                )
        check("Inputs", f"{DISPLAY[dataset]}: one input lineage for all 8 trees", 1, len(lineages))
    for dataset, gain in (("cmnist", 50.87), ("waterbirds", 17.40)):
        check(
            "Sec. 6.1",
            f"{DISPLAY[dataset]} ERM GRIT - vanilla (mean gain)",
            gain,
            r(paired(values[(dataset, "ERM", 2)], values[(dataset, "ERM", 0)])[0], 2),
        )
    wins = [
        (dataset, name)
        for dataset in ("cmnist", "waterbirds")
        for name in OBJECTIVES
        if st.fmean(values[(dataset, name, 2)]) > st.fmean(values[(dataset, name, 1)])
    ]
    losses = sorted(
        {(d, n) for d in ("cmnist", "waterbirds") for n in OBJECTIVES} - set(wins)
    )
    check("Sec. 6.1", "GRIT mean > PC mean (settings of 16)", 15, len(wins))
    check("Sec. 6.1", "the PC win", [("waterbirds", "RDM")], losses)
    above = sum(
        st.fmean(values[(d, n, 2)]) > st.fmean(values[(d, n, 0)])
        for d in ("cmnist", "waterbirds")
        for n in OBJECTIVES
    )
    check("Sec. 6.1", "GRIT mean > vanilla mean (settings of 16)", 16, above)
    for dataset, expected in (
        ("waterbirds", ((2.08, 1.04, 3.12), (0.25, -0.44, 0.95))),
        ("cmnist", ((37.11, 35.35, 38.86), (6.50, 5.99, 7.02))),
    ):
        grit = values[(dataset, "GroupDRO", 2)]
        for other, want in zip((0, 1), expected, strict=True):
            check(
                "App. F.3",
                f"{DISPLAY[dataset]} GroupDRO GRIT - {('vanilla', 'PC')[other]} [95% t]",
                want,
                r(paired(grit, values[(dataset, "GroupDRO", other)]), 2),
            )


# ------------------------------------------------------------------ UrbanCars (App. G)
UC = RESULTS / "urbancars"
METRICS = ("b_shift", "o_shift", "both_shift", "worst_group", "in_distribution")
COLLECTIONS = {  # (n_B, n_O) -> summary collection name
    (256, 0): "b_only",
    (128, 128): "count_50",
    (114, 142): "uneven",
    (54, 202): "balanced",
    (0, 256): "o_only",
}
# Tables 4 (worst-group validation) and 5 (training-weighted validation):
# B shift, O shift, both shift, worst group, training-weighted test accuracy.
TABLE45 = {
    "worst_group": {
        "erm": (45.4, 76.0, 28.5, 20.4, 83.1),
        (256, 0): {"consistency": (85.7, 65.1, 55.1, 52.9, 90.2), "grit": (88.5, 68.4, 60.3, 56.7, 92.3)},
        (128, 128): {"consistency": (76.0, 84.4, 71.7, 68.3, 87.5), "grit": (79.1, 85.1, 74.8, 71.9, 89.3)},
        (114, 142): {"consistency": (75.9, 86.0, 70.7, 67.9, 89.9), "grit": (78.2, 85.2, 74.6, 71.7, 89.1)},
        (54, 202): {"consistency": (68.8, 85.3, 65.7, 63.3, 87.7), "grit": (73.7, 87.5, 70.4, 67.9, 89.5)},
        (0, 256): {"consistency": (54.7, 80.2, 46.9, 42.6, 82.4), "grit": (53.9, 94.1, 45.5, 43.8, 95.5)},
    },
    "train_weighted": {
        "erm": (46.4, 86.5, 16.4, 15.3, 94.6),
        (256, 0): {"consistency": (70.8, 81.7, 32.0, 30.0, 97.2), "grit": (87.3, 69.0, 51.8, 49.3, 95.6)},
        (128, 128): {"consistency": (70.7, 81.8, 32.0, 29.9, 97.2), "grit": (80.8, 78.8, 53.2, 51.7, 95.5)},
        (114, 142): {"consistency": (70.7, 81.7, 32.0, 30.1, 97.2), "grit": (78.0, 80.9, 54.2, 52.0, 95.3)},
        (54, 202): {"consistency": (70.3, 82.9, 34.7, 32.5, 97.2), "grit": (54.9, 93.5, 45.6, 42.8, 95.2)},
        (0, 256): {"consistency": (70.5, 82.7, 34.0, 32.1, 97.1), "grit": (53.9, 93.8, 45.5, 44.0, 95.5)},
    },
}
# Table 2: B shift, O shift, worst group for ERM, +PC, +GRIT.
TABLE2 = {
    (256, 0): ((45.4, 85.7, 88.5), (76.0, 65.1, 68.4), (20.4, 52.9, 56.7)),
    (128, 128): ((45.4, 76.0, 79.1), (76.0, 84.4, 85.1), (20.4, 68.3, 71.9)),
    (0, 256): ((45.4, 54.7, 53.9), (76.0, 80.2, 94.1), (20.4, 42.6, 43.8)),
}
TABLE6 = {
    "worst_group": {
        "erm": (45.4, 76.0, 28.5, 20.4, 83.1),
        "erm_matched": (66.0, 82.5, 25.8, 20.6, 96.8),
        "reference": (84.0, 83.8, 82.0, 77.4, 88.9),
    },
    "train_weighted": {
        "erm": (46.4, 86.5, 16.4, 15.3, 94.6),
        "erm_matched": (69.4, 81.5, 31.3, 27.4, 97.1),
        "reference": (93.7, 66.4, 62.5, 61.4, 95.0),
    },
}


class UrbanCars:
    def __init__(self, tree: str) -> None:
        self.summary = load(UC / tree / "summary.json")
        self.rows: dict[str, Any] = self.summary["rows"]
        self.banks: list[int] = self.summary["config"]["banks"]

    def seeds(self, unit: str, selector: str, metric: str, suffix: str = "") -> list[float]:
        """Per-seed values; pair methods are averaged over the fixed banks first."""
        if "@" not in unit:
            return self.rows[f"{unit}{suffix}|{selector}"]["test"][metric]
        per_bank = [
            self.rows[f"{unit}-b{b}{suffix}|{selector}"]["test"][metric] for b in self.banks
        ]
        return [st.fmean(v) for v in zip(*per_bank, strict=True)]

    def mean(self, unit: str, selector: str, suffix: str = "") -> tuple[float, ...]:
        return tuple(
            round(100 * st.fmean(self.seeds(unit, selector, m, suffix)), 1) for m in METRICS
        )


def check_urbancars() -> None:
    v1 = UrbanCars("urbancars-v1-standard")
    config = v1.summary["config"]
    check("App. G.2", "final seeds", FINAL_SEEDS, config["seeds"]["final"])
    for sel, table in TABLE45.items():
        name = "Table 4" if sel == "worst_group" else "Table 5"
        check(name, "ERM (all allocations)", table["erm"], v1.mean("erm", sel))
        for counts_, methods in table.items():
            if counts_ == "erm":
                continue
            collection = COLLECTIONS[counts_]
            for method, expected in methods.items():
                got = v1.mean(f"{method}@{collection}", sel)
                label = {"consistency": "PC", "grit": "GRIT"}[method]
                check(name, f"n=({counts_[0]},{counts_[1]}) {label}", expected, got)
    for counts_, expected in TABLE2.items():
        collection = COLLECTIONS[counts_]
        units = ("erm", f"consistency@{collection}", f"grit@{collection}")
        got = tuple(
            tuple(v1.mean(u, "worst_group")[metric] for u in units) for metric in (0, 1, 3)
        )
        check("Table 2", f"n=({counts_[0]},{counts_[1]}) B, O, worst group", expected, got)
    for sel, table in TABLE6.items():
        for method, expected in table.items():
            check("Table 6", f"{sel} {method}", expected, v1.mean(method, sel))
    contrasts = (
        ("GRIT - PC, O-only, worst group", "grit@o_only", "consistency@o_only", "worst_group", 3, (1.3, -0.5, 3.0)),
        ("GRIT 202-obj - 142-obj, worst group", "grit@balanced", "grit@uneven", "worst_group", 3, (-3.8, -4.8, -2.9)),
        ("GRIT 202-obj - 142-obj, O shift", "grit@balanced", "grit@uneven", "worst_group", 1, (2.3, 1.8, 2.9)),
        ("GRIT 202-obj - matched ERM, worst group", "grit@balanced", "erm_matched", "worst_group", 3, (47.2, 42.6, 51.9)),
        ("TW: GRIT 202-obj - B-only, worst group", "grit@balanced", "grit@b_only", "train_weighted", 3, (-6.5, -8.3, -4.8)),
    )
    for label, a, b, sel, metric, expected in contrasts:
        got = paired(v1.seeds(a, sel, METRICS[metric]), v1.seeds(b, sel, METRICS[metric]))
        check("App. G.4", f"{label} [95% t]", expected, r(got, 1))
    energies = v1.summary["calibration_energies"]
    check("App. G.1", "E_B, E_O, ratio", (61.0, 16.2, 3.76),
          (round(energies["b"], 1), round(energies["o"], 1), round(energies["b"] / energies["o"], 2)))
    n_o = {c: v1.rows[f"grit@{c}-b0|worst_group"]["collection"]["n_o"] for c in ("uneven", "balanced")}
    check("App. G.1", "frozen n_O for s = 0.25, 0.5", (142, 202), (n_o["uneven"], n_o["balanced"]))
    share = energies["o"] / (energies["o"] + energies["b"])
    check("App. G.1", "equal-count object energy share (%)", 21, round(100 * share))
    optimizers = config["optimizers"]
    check("App. G.2", "frozen LR / WD: ERM, PC, GRIT, reference",
          ((1e-5, 0.0), (1e-2, 1e-4), (1e-3, 1e-4), (3e-4, 0.0)),
          tuple((optimizers[m]["learning_rate"], optimizers[m]["weight_decay"])
                for m in ("erm", "consistency", "grit", "reference")))
    check("App. G.2", "GRIT ranks", [1, 2, 4, 8, 16, 32, 64, 128], config["ranks"])
    check("App. G.2", "PC alpha grid", [10.0 ** e for e in range(-3, 7)],
          [float(a) for a in config["consistency_strengths"]])
    grid = load(UC / "urbancars-v1-standard-optimizer-v2" / "inputs.json")["binding"]["config"]
    check("App. G.2", "extended optimizer grid (LR; WD)",
          ([1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2], [0.0, 1e-4]),
          (sorted(grid["learning_rates"]), sorted(grid["weight_decays"])))
    # G.3 statements about selected checkpoints and penalties.
    early = [e for c in ("b_only", "count_50", "uneven", "balanced") for b in v1.banks
             for e in v1.rows[f"grit@{c}-b{b}|worst_group"]["epochs"]]
    late = [e for c in COLLECTIONS.values() for b in v1.banks
            for e in v1.rows[f"grit@{c}-b{b}|train_weighted"]["epochs"]]
    report("PASS" if sum(e <= 7 for e in early) >= 0.75 * len(early) else "FLAG", "App. G.3",
           "WG GRIT epochs 'usually 1-7' (excluding O-only)",
           f"{sum(e <= 7 for e in early)}/{len(early)} final fits at epochs 1-7, max {max(early)}")
    report("PASS" if sum(e >= 35 for e in late) >= 0.75 * len(late) else "FLAG", "App. G.3",
           "TW GRIT epochs 'typically 35-100'",
           f"{sum(e >= 35 for e in late)}/{len(late)} final fits at epochs 35-100, min {min(late)}")
    smallest = min(config["consistency_strengths"])
    chosen = {
        c: [float(v1.rows[f"consistency@{c}-b{b}|train_weighted"]["candidate_id"].split(":")[1][1:])
            for b in v1.banks]
        for c in COLLECTIONS.values()
    }
    others = tuple(
        (c, bank, alpha)
        for c, alphas in chosen.items()
        for bank, alpha in zip(v1.banks, alphas, strict=True)
        if alpha != smallest
    )
    check("App. G.3", "TW PC smallest penalty: bank-level fits of 15; others",
          (12, (("balanced", 1, 0.1), ("o_only", 0, 0.01), ("o_only", 1, 0.01))),
          (sum(alpha == smallest for alphas in chosen.values() for alpha in alphas), others))
    check("App. G.3", "TW PC smallest penalty", 1e-3, smallest)
    tw_pc = [v1.mean(f"consistency@{c}", "train_weighted")[3] for c in COLLECTIONS.values()]
    check("App. G.3", "TW PC worst-group range", (29.9, 32.5), (min(tw_pc), max(tw_pc)))
    for sel in ("worst_group", "train_weighted"):
        grit = [v1.mean(f"grit@{c}", sel)[3] for c in COLLECTIONS.values()]
        pc = [v1.mean(f"consistency@{c}", sel)[3] for c in COLLECTIONS.values()]
        matched = v1.mean("erm_matched", sel)[3]
        check("App. G.3", f"{sel}: GRIT worst group > PC, every allocation", True,
              all(g > p for g, p in zip(grit, pc, strict=True)))
        check("App. G.3", f"{sel}: GRIT worst group > matched ERM, every allocation", True,
              all(g > matched for g in grit))
    # Frozen winners and checks.
    for tree in ("urbancars-v1-standard",):
        winners = load(UC / tree / "winners.json")
        frozen = {k: winners[k] for k in ("finalists", "winners", "matched")}
        summary = load(UC / tree / "summary.json")
        check("Integrity", f"{tree} winners digest", winners["winners_sha256"], canonical_digest(frozen))
        check("Integrity", f"{tree} summary cites the frozen winners", winners["winners_sha256"], summary["winners_sha256"])
    report_ = load(UC / "urbancars-v1-standard-check" / "check-report.json")
    check("App. G.1", "C1 shortcut-reliance feasibility check passes", True, report_["C1_shortcut_reliance"]["passes"])
    check("App. G.4", "directions for 90% of B, O reference energy", (151, 166),
          (report_["C2_geometry"]["b"]["rank_90"], report_["C2_geometry"]["o"]["rank_90"]))
    header = load(UC / "construction" / "dataset-manifest-header.json")
    check("App. G.1", "train groups per label (both, B-only, O-only, neither misaligned)",
          [3610, 190, 190, 10] * 2, header["group_counts"]["train"])
    check("App. G.1", "val / test examples per group", ([125] * 8, [125] * 8),
          (header["group_counts"]["val"], header["group_counts"]["test"]))
    check("App. G.1", "construction seed", 1729, header["seed"])
    check("App. G.1", "anchor pools: calibration, reference, 3 banks",
          (1024, 2048, 1024, 1024, 1024),
          tuple(header["anchor_pool_sizes"][k] for k in ("calibration", "reference", "bank0", "bank1", "bank2")))
    check("App. G.5", "upstream generator revision", "c91b0970", header["upstream_commit"][:8])
    check("Integrity", "summaries cite this dataset manifest", header["digest"], v1.summary["dataset_digest"])
    methods = [m["method"] for m in header["fallback_masks"]]
    check("App. G.5", "fallback masks: total, MaskFormer, GrabCut", (16, 14, 2),
          (len(methods), methods.count("maskformer"), methods.count("grabcut")))
    g0b = load(UC / "construction" / "g0b.json")
    check("App. G.5", "pixel-identical of 600; max intensity difference", (597, 3),
          (sum(g0b[s]["identical"] for s in ("train", "val", "test")),
           max(g0b[s]["max_abs_difference"] for s in ("train", "val", "test"))))
    audit = load(UC / "construction" / "source-audit.json")
    kinds = [p["kind"] for p in audit["byte_identical_non_object_pairs"]]
    check("App. G.5", "byte-identical file pairs: Places, Stanford Cars", (3, 3),
          (kinds.count("background"), kinds.count("car")))
    check("App. G.5", "COCO photos shared across splits", 908, len(audit["coco_photos_shared_across_splits"]))


# ---------------------------------------------------------------- Appendix H, Figs 2-5
def check_sweeps() -> None:
    manifest = load(SWEEPS / "sweep-results" / "manifest.json")
    results_path = SWEEPS / "sweep-results" / "sweep-results.json"
    check("Integrity", "sweep-results.json matches its manifest digest",
          manifest["results_sha256"], hashlib.sha256(results_path.read_bytes()).hexdigest())
    variants = load(SWEEPS / "sweep-variants" / "geometry-variants-public.json")
    for name, recorded in (
        ("sweep manifest", manifest["source_hashes"]),
        ("geometry variants", variants["provenance"]["source_hashes"]),
    ):
        differing = sorted(
            path for path, sha in recorded.items()
            if hashlib.sha256((ROOT / path).read_bytes()).hexdigest() != sha
        )
        check("Integrity", f"{name}: sources that differ from recorded hashes",
              [], differing)
    rows = load(results_path)
    table: dict[tuple[str, str, float, str, str], float] = {}
    with (SWEEPS / "sweep-stacked" / "sweep-table.tsv").open() as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            key = (row["dataset"], row["sweep"], float(row["value"]), row["method"], row["metric"])
            table[key] = float(row["mean"])

    def metric_of(row: dict[str, Any], metric: str) -> float:
        if metric == "both_shift":
            return row["test"]["both_shift"]
        if metric == "normalized_sensitivity":  # as in sweep_report.metrics_for
            return row["sensitivity"] / row["coefficient_norm"]
        return row[metric]

    def value(ds: str, sweep: str, x: float, method: str, metric: str) -> float:
        out = [
            metric_of(row, metric)
            for row in rows
            if (row["dataset"], row["sweep"], row["method"]) == (ds, sweep, method)
            and row["value"] == x
        ]
        mean = st.fmean(out)
        assert abs(mean - table[(ds, sweep, float(x), method, metric)]) < 1e-12
        return 100 * mean if metric == "both_shift" else mean

    reps = {ds: len({row["replicate"] for row in rows if row["dataset"] == ds}) for ds in ("digits", "synthetic")}
    check("App. H", "replicates: CueMNIST, Gaussian", (10, 20), (reps["digits"], reps["synthetic"]))
    check("App. H", "seed", {20260927}, {row["seed"] for row in rows})
    d, s = "digits", "synthetic"
    expected = [
        ("CueMNIST g=1/64 -> 1 subspace error", (0.535, 0.064), (value(d, "coverage", 1 / 64, "grit", "missed_u"), value(d, "coverage", 1, "grit", "missed_u")), 3),
        ("CueMNIST g=1/64 -> 1 normalized instability", (0.469, 0.029), (value(d, "coverage", 1 / 64, "grit", "normalized_sensitivity"), value(d, "coverage", 1, "grit", "normalized_sensitivity")), 3),
        ("CueMNIST g=1/64 -> 1 target accuracy", (3.43, 75.86), (value(d, "coverage", 1 / 64, "grit", "both_shift"), value(d, "coverage", 1, "grit", "both_shift")), 2),
        ("CueMNIST noise .025 -> 1 subspace error", (0.016, 0.780), (value(d, "noise", 0.025, "grit", "missed_u"), value(d, "noise", 1, "grit", "missed_u")), 3),
        ("CueMNIST noise .025 -> 1 normalized instability", (0.002, 0.673), (value(d, "noise", 0.025, "grit", "normalized_sensitivity"), value(d, "noise", 1, "grit", "normalized_sensitivity")), 3),
        ("CueMNIST k 64 -> 4096 subspace error", (0.129, 0.016), (value(d, "pairs", 64, "grit", "missed_u"), value(d, "pairs", 4096, "grit", "missed_u")), 3),
        ("CueMNIST k 64 -> 4096 target accuracy", (60.45, 79.97), (value(d, "pairs", 64, "grit", "both_shift"), value(d, "pairs", 4096, "grit", "both_shift")), 2),
        ("Gaussian g=1/64 -> 1 subspace error", (0.568, 0.064), (value(s, "coverage", 1 / 64, "grit", "missed_u"), value(s, "coverage", 1, "grit", "missed_u")), 3),
        ("Gaussian g=1/64 -> 1 target accuracy", (55.49, 88.43), (value(s, "coverage", 1 / 64, "grit", "both_shift"), value(s, "coverage", 1, "grit", "both_shift")), 2),
        ("Gaussian k 64 -> 4096 subspace error", (0.128, 0.016), (value(s, "pairs", 64, "grit", "missed_u"), value(s, "pairs", 4096, "grit", "missed_u")), 3),
        ("reference GRIT, CR: CueMNIST", (75.86, 71.58), (value(d, "pairs", 256, "grit", "both_shift"), value(d, "pairs", 256, "cr", "both_shift")), 2),
        ("reference GRIT, CR: Gaussian", (88.43, 86.77), (value(s, "pairs", 256, "grit", "both_shift"), value(s, "pairs", 256, "cr", "both_shift")), 2),
        ("CueMNIST k=64 CR, GRIT", (63.83, 60.45), (value(d, "pairs", 64, "cr", "both_shift"), value(d, "pairs", 64, "grit", "both_shift")), 2),
        ("CueMNIST noise 0.3 CR, GRIT", (39.14, 35.94), (value(d, "noise", 0.3, "cr", "both_shift"), value(d, "noise", 0.3, "grit", "both_shift")), 2),
        ("k=4096 GRIT, CR: CueMNIST", (79.97, 71.40), (value(d, "pairs", 4096, "grit", "both_shift"), value(d, "pairs", 4096, "cr", "both_shift")), 2),
        ("k=4096 GRIT, CR: Gaussian", (89.61, 87.18), (value(s, "pairs", 4096, "grit", "both_shift"), value(s, "pairs", 4096, "cr", "both_shift")), 2),
        ("Sec. 6.2 k=4096 GRIT, CR (CueMNIST, 1 decimal)", (80.0, 71.4), (value(d, "pairs", 4096, "grit", "both_shift"), value(d, "pairs", 4096, "cr", "both_shift")), 1),
        ("CueMNIST k=64 normalized instability CR, GRIT", (0.056, 0.091), (value(d, "pairs", 64, "cr", "normalized_sensitivity"), value(d, "pairs", 64, "grit", "normalized_sensitivity")), 3),
        ("k=4096 normalized instability GRIT, CR: CueMNIST", (0.0022, 0.0503), (value(d, "pairs", 4096, "grit", "normalized_sensitivity"), value(d, "pairs", 4096, "cr", "normalized_sensitivity")), 4),
        ("k=4096 normalized instability GRIT, CR: Gaussian", (0.0021, 0.0386), (value(s, "pairs", 4096, "grit", "normalized_sensitivity"), value(s, "pairs", 4096, "cr", "normalized_sensitivity")), 4),
    ]
    for label, want, got, digits in expected:
        check("App. H", label, want, r(got, digits))

    def control(ds: str, name: str, method: str, metric: str) -> list[float]:
        return [
            metric_of(row, metric)
            for row in rows
            if (row["dataset"], row["sweep"], row["method"]) == (ds, name, method)
        ]

    check("App. H", "nonspanning control: subspace error, accuracy (%)", (0.997, 0.02),
          (round(st.fmean(control(d, "nonspanning_control", "grit", "missed_u")), 3),
           round(100 * st.fmean(control(d, "nonspanning_control", "grit", "both_shift")), 2)))
    check("App. H", "noiseless control reaches true-subspace accuracy (%)", (80.31, 80.31),
          (round(100 * st.fmean(control(d, "clean_control", "grit", "both_shift")), 2),
           round(100 * st.fmean(control(d, "clean_control", "oracle", "both_shift")), 2)))
    check("App. H", "noiseless control subspace error < 1e-14", True,
          max(control(d, "clean_control", "grit", "missed_u")) < 1e-14)
    summary = load(SWEEPS / "sweep-stacked" / "sweep-summary.json")
    slopes = {(x["dataset"], x["metric"]): x for x in summary["slopes"]}
    for (ds, metric), want in {
        ("digits", "missed_u"): (-0.500, -0.519, -0.482),
        ("digits", "normalized_sensitivity"): (-1.004, -1.209, -0.839),
        ("synthetic", "missed_u"): (-0.497, -0.511, -0.482),
    }.items():
        x = slopes[(ds, metric)]
        check("App. H", f"{ds} {metric} log-log slope, k >= 512 [95%]", want,
              r((x["mean"], *x["ci95_mean"]), 3))
    gain = next(x for x in summary["paired_endpoints"]
                if (x["dataset"], x["sweep"], x["method"], x["metric"]) == ("digits", "coverage", "grit", "both_shift"))
    check("App. H", "CueMNIST g sweep paired accuracy gain [95% bootstrap]", (72.43, 70.11, 74.57),
          r((100 * gain["mean"], *(100 * v for v in gain["ci95_mean"])), 2))
    alpha = {
        (ds, k): sum(row["alpha"] == 1000.0 for row in rows
                     if row["dataset"] == ds and row["method"] == "cr" and row["sweep"] == "pairs" and row["k"] == k)
        for ds in ("digits", "synthetic") for k in (64, 4096)
    }
    check("App. H", "CR alpha = 1000 at k=4096 (CueMNIST of 10, Gaussian of 20)", (9, 4),
          (alpha[("digits", 4096)], alpha[("synthetic", 4096)]))
    check("App. H", "CR alpha = 1000 at k=64 (CueMNIST of 10)", 10, alpha[("digits", 64)])
    check("App. H", "primary run under a minute (s)", True, manifest["elapsed_seconds"] < 60)


# ------------------------------------------------------------ trimmed CMNIST/Waterbirds
REMOVED_MARKERS = ("_representation_consistency", "_two_layer", "representation_latent")


def check_trimming() -> None:
    trimming = load(RESULTS / "TRIMMING.json")
    trees = sorted(f"{d}/{p.name}" for d in ("cmnist", "waterbirds") for p in (RESULTS / d).iterdir())
    check("Integrity", "TRIMMING.json covers all 16 trees", (16, True),
          (len(trees), trees == sorted(trimming["trees"])))
    for name, tree in sorted(trimming["trees"].items()):
        root = RESULTS / name
        short = name.split("/")[1].replace("-interventions-representation-v1", "")
        changed = {
            file: hashlib.sha256((root / file).read_bytes()).hexdigest()
            for file in tree["changed_files"]
        }
        check("Integrity", f"{short} trimmed files match TRIMMING.json", True,
              all(changed[f] == v["trimmed_sha256"] for f, v in tree["changed_files"].items())
              and not any((root / f).exists() for f in tree["deleted_files"]))
        texts = [p.read_text(encoding="utf-8") for p in root.rglob("*") if p.suffix in (".json", ".yaml")]
        remaining = sum(marker in text for marker in REMOVED_MARKERS for text in texts)
        plan = load(root / "search-plan.json")
        check("Integrity", f"{short} keeps 3 methods, no removed entries",
              (3, 0, tree["candidates"]["kept"]),
              (len(plan["methods"]), remaining, len(plan["candidates"])))
        digests = {load(p)["plan_digest"] for p in (root / "summaries").glob("*-summary.json")}
        check("Integrity", f"{short} summaries cite the recorded plan digest",
              {tree["recorded_plan_digest"]}, digests)


# --------------------------------------------------------------- optional package checks
def check_with_package() -> None:
    try:
        from grit.search.outputs import CmnistProductionSummary, WaterbirdsProductionSummary
        from grit.search.plan import SearchPlan, load_production_search_config
        from grit.search.urbancars import load_config
        from grit.selection.cmnist import (
            FinalistUnion,
            FrozenCandidateSelection,
            TuningFinalistsArtifact,
        )
        from grit.selection.waterbirds import FrozenWaterbirdsCandidate, WaterbirdsTuningFinalists
    except ImportError:
        print("\nSKIP  package checks (run with `uv run` to include them)")
        return
    os.environ["PROJECT_SCRATCH"] = "/anonymous/scratch"
    selection_models: dict[str, dict[str, Any]] = {
        "cmnist": {
            "winner.json": FrozenCandidateSelection,
            "tuning-finalists.json": TuningFinalistsArtifact,
            "confirmation-union.json": FinalistUnion,
        },
        "waterbirds": {
            "winner.json": FrozenWaterbirdsCandidate,
            "tuning-finalists.json": WaterbirdsTuningFinalists,
        },
    }
    for folder in ("cmnist", "waterbirds"):
        for root in sorted((RESULTS / folder).iterdir()):
            resolved = load(root / "resolved-config.json")
            relative = resolved["authored_config_path"].removeprefix("/anonymous/source/")
            config = load_production_search_config(ROOT / relative)
            recorded = resolved["config"]
            check("Package", f"{relative.split('/')[-1][:44]} = recorded config", True,
                  json.loads(config.canonical_json()) == recorded)
            authored = load_production_search_config(root / "authored-config.yaml")
            check("Package", f"{root.name[:44]} authored copy = shipped config", True,
                  authored.canonical_json() == config.canonical_json())
            # The recorded config digest covers the original absolute paths, which the
            # anonymized copy replaces; recompute it before revalidating the plan.
            plan = load(root / "search-plan.json")
            plan["resolved_config"]["authored_config_digest"] = config.canonical_digest()
            try:
                SearchPlan.model_validate_json(json.dumps(plan))
                ok = True
            except ValueError as error:  # pydantic's ValidationError is a ValueError
                ok = False
                print(error)
            check("Package", f"{root.name[:44]} candidate grid regenerated", True, ok)
            name = f"{folder}-summary.json"
            model = CmnistProductionSummary if folder == "cmnist" else WaterbirdsProductionSummary
            text = (root / "summaries" / name).read_text(encoding="utf-8")
            try:
                # Revalidation recomputes every summary statistic and comparison.
                ok = model.model_validate_json(text).canonical_json() + "\n" == text
            except ValueError as error:
                ok = False
                print(error)
            check("Package", f"{root.name[:44]} summary recomputed", True, ok)
            parsed = 0
            files = sorted((root / "selection").rglob("*.json"))
            for path in files:
                model = next(m for suffix, m in selection_models[folder].items()
                             if path.name.endswith(suffix))
                text = path.read_text(encoding="utf-8")
                parsed += model.model_validate_json(text).canonical_json() + "\n" == text
            check("Package", f"{root.name[:44]} selection artifacts revalidated",
                  len(files), parsed)
    for yaml_name, tree in (
        ("v1.yaml", "urbancars-v1-standard"),
        ("check.yaml", "urbancars-v1-standard-check"),
        ("optimizer.yaml", "urbancars-v1-standard-optimizer"),
        ("optimizer-v2.yaml", "urbancars-v1-standard-optimizer-v2"),
    ):
        config = load_config(ROOT / "configs" / "urbancars" / yaml_name)
        inputs = load(UC / tree / "inputs.json")
        scientific = json.loads(json.dumps(config.scientific()))
        check("Package", f"configs/urbancars/{yaml_name} = recorded binding", True,
              scientific == inputs["binding"]["config"])
        winners = UC / tree / "winners.json"
        if winners.exists():
            binding = {"config": scientific, "dataset_digest": inputs["binding"]["dataset_digest"]}
            check("Package", f"{tree} binding digest", load(winners)["binding_digest"], canonical_digest(binding))


def main() -> int:
    check_core_tables()
    check_urbancars()
    check_trimming()
    check_sweeps()
    check_with_package()
    print(f"\n{counts['PASS']} passed, {counts['FAIL']} failed, {counts['FLAG']} flagged")
    return 1 if counts["FAIL"] else 0


if __name__ == "__main__":
    sys.exit(main())
