# Result records

Compact records of the completed searches behind the paper's ColoredMNIST-INV
(`cmnist/`), Waterbirds-INV (`waterbirds/`) and UrbanCars (`urbancars/`) results. The
saved controlled studies are in `experimental/collection_geometry/` (see its README).

```sh
python3 results/verify_results.py            # standard library only
uv run python results/verify_results.py      # also checks the shipped configs and code
```

`verify_results.py` recomputes every reported number listed below from per-seed or
per-replicate records and prints it beside the value transcribed from the paper. It
fits nothing and reads no data, and it flags nothing. `sha256sum -c results/SHA256SUMS`
(run from the package root) checks the files themselves.

## Paper to records

| Paper | Records | Quantity |
| --- | --- | --- |
| Tables 1 and 3, Sec. 6.1, App. F: ColoredMNIST-INV | `cmnist/<tree>/summaries/cmnist-summary.json` | `methods[]` with `selector == "primary_robust"`; `final_observations[].test_ood_accuracy` at seeds 3101–3110 |
| Tables 1 and 3, Sec. 6.1, App. F: Waterbirds-INV | `waterbirds/<tree>/summaries/waterbirds-summary.json` | `methods[].worst_group_by_seed` (selector `waterbirds_validation_worst_group`) |
| Table 2, Tables 4–6, App. G.3–G.4 | `urbancars/urbancars-v1-standard/summary.json` | `rows["<method>@<collection>-b<bank>\|<selector>"].test`, ten final seeds per row; `rows[...].epochs` and `candidate_id` give the selected checkpoints and settings |
| App. G.1 (energies, C1), G.4 (151 and 166 directions) | `urbancars/urbancars-v1-standard-check/check-report.json` | `C4_energy_calibration`, `C1_shortcut_reliance`, `C2_geometry` |
| App. G.2 (frozen optimizers, extended grid) | `urbancars/urbancars-v1-standard-optimizer{,-v2}/` | `optimizer.json`, `inputs.json` |
| App. G.1, G.5 (groups, anchors, masks, fidelity, overlap) | `urbancars/construction/` | `dataset-manifest-header.json`, `g0a.json`, `g0b.json`, `source-audit.json` |
| Figures 2–5, App. H | `experimental/collection_geometry/sweep-results/`, `sweep-variants/`, `sweep-stacked/` | `sweep-results.json` (per replicate), `sweep-table.tsv`, `sweep-summary.json` |

Rows of Tables 1 and 3, with their output trees:

| Objective | ColoredMNIST-INV (`cmnist/`) | Waterbirds-INV (`waterbirds/`) | Method IDs: vanilla, +PC, +GRIT |
| --- | --- | --- | --- |
| ERM | `cmnist-erm-interventions-representation-v1-e03` | `waterbirds-erm-interventions-representation-v1` | `erm`, `erm_consistency`, `grit` |
| SD | `cmnist-sd-…-v1-e03` | `waterbirds-sd-…-v1` | `sd`, `sd_consistency`, `sd_grit` |
| RDM | `cmnist-rdm-…-v1-e03` | `waterbirds-rdm-…-v1` | `rdm`, `rdm_consistency`, `rdm_grit` |
| IRMv1 | `cmnist-irm-…-v1-e03` | `waterbirds-irm-…-v1` | `irm`, `irm_consistency`, `irm_grit` |
| V-REx | `cmnist-rex-…-v1-e03` | `waterbirds-rex-…-v1` | `rex`, `rex_consistency`, `rex_grit` |
| Fishr | `cmnist-fishr-…-v1-e03` | `waterbirds-fishr-…-v1` | `fishr`, `fishr_consistency`, `fishr_grit` |
| GroupDRO (4 groups) | `cmnist-groupdro-label-source-env-…-v1-e03` | `waterbirds-groupdro-…-v1` | `groupdro`, `groupdro_consistency`, `groupdro_grit` |
| LISA | `cmnist-lisa-source-env-…-v1-e03` | `waterbirds-lisa-…-v1` | `lisa`, `lisa_consistency`, `lisa_grit` |

Each ColoredMNIST-INV summary also contains the source-only selector
(`secondary_source`) described in App. F; Tables 1 and 3 use the primary selector.

UrbanCars names: collections `b_only` (n_B = 256, n_O = 0), `count_50` (128, 128),
`uneven` (114, 142), `balanced` (54, 202) and `o_only` (0, 256); methods `erm`,
`consistency` (PC), `grit`, `erm_matched` (ERM, matched optimizer) and `reference`;
selectors `worst_group` (primary, WG) and `train_weighted` (TW). The metric
`in_distribution` is the paper's training-weighted test accuracy.

UrbanCars check labels used in file and key names:

| Label | Check (App. G) |
| --- | --- |
| C1 | ERM shortcut reliance: each validation shift accuracy at least 3 points below in-distribution accuracy (G.1) |
| C2 | Reference geometry: directions holding 90% of the B and O reference energy (G.4) |
| C3 | The large-pair reference projection |
| C4 | Energy calibration E_B, E_O and the count table (G.1) |
| G0a | Source acceptance: Stanford Cars counts and filtered files, Places lists, LVIS pools, car masks (G.5) |
| G0b | Fidelity of the rebuilt images against an unmodified upstream run (G.5) |

**Uncertainty, as in the paper.** ColoredMNIST-INV and Waterbirds-INV cells are means
and sample standard deviations over the ten final seeds; contrasts are paired by seed
with a 95% Student-t interval on 9 degrees of freedom. For UrbanCars PC and GRIT, each
final seed is first averaged over the three banks; ERM and the reference have no
banks. Statements qualified by "usually" or "typically" hold as worded: under
worst-group validation, 117 of 120 non-O-only GRIT final fits select epochs 1–7 (the
maximum is 33, for B-only); under training-weighted validation, 142 of 150 select
epochs 35–100 (the minimum is 18).

## Contents of each tree

- **Objective-matrix trees** (`cmnist/`, `waterbirds/`):
  - `authored-config.yaml`: the run's configuration file, headed by a two-line note;
    its parsed content equals the shipped file in `configs/`.
  - `resolved-config.json`: the resolved configuration with input-artifact digests.
  - `search-plan.json`: every candidate with its scientific digest, the seed plan and
    the expected run counts.
  - `selection/<method>/`: the tuning finalists, the confirmation union and each
    selector's winner.
  - `summaries/`: per-seed final metrics, paired differences, intervals and the
    within-objective contrasts.
  - `projections/`: the GRIT projection diagnostics for each rank.
- **UrbanCars searches**:
  - `inputs.json`: the binding (scientific configuration and dataset digest).
  - `winners.json`: the finalists, winners and matched ranks, frozen and hashed before
    test data were unsealed.
  - `summary.json`: per-bank, per-seed validation and test metrics, selected epochs,
    residual diagnostics and contrasts.

Per-candidate run folders, checkpoints, datasets, images, CLIP features and weights
are not included; summaries keep each final record's relative `result_path`.

## About the records

**Anonymization.** These records were prepared for double-blind review and are kept
unchanged here. Machine paths are replaced by `/anonymous/scratch` (data and outputs)
and `/anonymous/source` (code), and code revisions by `anonymous-rev-1` … `-8`. Three
recorded digests cover the original strings, so they do not verify against these
copies: `plan_digest`, `resolved_config_digest` and `authored_config_digest`.
`verify_results.py` checks configuration identity structurally instead. All other
digests verify, including the UrbanCars winners, binding and dataset digests and the
controlled-study result digests.

**Trimmed to the reported methods.** The recorded matrix searches also crossed
representation consistency with every objective and, for ERM, V-REx, IRMv1 and
Fishr, a two-layer control; the configuration names' `…-representation-…` label comes
from this. The paper reports neither, so their candidates, method-list entries,
search-space settings, selection folders, summary entries and contrasts were removed,
and the plans' derived run counts recomputed. Each method is tuned and selected on its
own, so the removal changes no kept selection, seed or metric. Of 13,728 recorded
candidates, the 9,216 of the reported methods remain with their original identifiers.
`TRIMMING.json` records each tree's removed methods, counts and file hashes before and
after trimming. `verify_results.py` checks these hashes and, with the package
installed, regenerates each plan's candidate grid and revalidates each summary and
selection record.

**Code versions.** The records were produced by pre-release revisions of this code
(`anonymous-rev-*`). The checks above tie them to the shipped code: every shipped
configuration parses to its recorded configuration, the shipped code regenerates every
recorded candidate identifier and scientific digest, and it reproduces the UrbanCars
binding digests.

**Controlled-study manifests.** `sweep-results/manifest.json` and the provenance block
of `sweep-variants/geometry-variants-public.json` list SHA-256 hashes of the source
files; these are set to the shipped files. The saved sweep ran on a CPU (float64,
Python 3.12, macOS). Rerunning it with this code on Linux (Python 3.10) reproduces all
3,840 fitted rows with identical selections and test accuracies; losses and geometry
agree within 1e-13 relative, and hashes of floating-point arrays can differ across
platforms.
