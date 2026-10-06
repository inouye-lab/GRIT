# Controlled CueMNIST and Gaussian studies (Appendix H, Figures 2–5)

This folder holds the controlled studies of Appendix H. CueMNIST uses real MNIST digit
content with a fixed 96-dimensional encoder and two visible color cues (ten
replicates). The Gaussian task is its synthetic counterpart (twenty replicates). Both
fit linear logistic probes with GRIT and with validation-selected consistency
regularization (CR), using 3,000 source labels, ridge 0.1 and known rank 2. The
production seed is 20260927. Configurations and CR strengths are frozen before each
replicate's test features are created.

The three primary sweeps meet at one reference point: clean spectrum (1, 1), Gaussian
pair-noise SD 0.1 and 256 pairs. Each sweep varies one of these: the weakest clean
eigenvalue, the pair noise, or the number of pairs. `SWEEP-PROTOCOL.md` fixes the
settings and distinguishes the original primary sweeps from the follow-up secondary
geometry settings.

## Run the sweeps

From the package root:

```sh
sh experimental/collection_geometry/run_sweeps.sh
```

This fits both datasets' complete sweeps and controls, then writes the frozen
selections, fitted coefficients, input identities, raw measurements, summary tables,
confidence intervals, captions and figures. It also writes the secondary geometry
measurements separately. If `.venv` is absent, the script runs
`uv sync --locked --group dev`, and the first run downloads MNIST; both need network
access. A CPU with float64 is enough; the primary sweeps take under a minute after
setup. Because `sweep-results/` exists, a rerun writes a new
`sweep-reruns/<timestamp>/` folder; `--output /absolute/fresh/path` picks another
location. Nonempty directories are rejected.

## Regenerate the figures from the saved run

This needs no fitting and no test data:

```sh
uv run python experimental/collection_geometry/sweep_report.py \
  experimental/collection_geometry/sweep-results /tmp/grit-sweep-figures \
  --geometry-variants experimental/collection_geometry/sweep-variants/geometry-variants-public.json
```

Use a fresh output directory. Without `--geometry-variants`, the figures show only the
default geometry curve. To recompute the secondary geometry measurements from the
frozen records, run
`uv run python experimental/collection_geometry/geometry_variants.py experimental/collection_geometry/sweep-results /tmp/grit-geometry-variants`
and pass that directory's `geometry-variants.json` to the reporter.

## Saved results

| Folder | Contents |
| --- | --- |
| `sweep-results/` | The primary run. `sweep-results.json` (every fitted row), per-replicate `*-frozen.json`, `*-weights.npz` and `*-results.json`, `config.json`, `manifest.json` (source hashes, runtime, package versions), `dataset-inputs.json` (MNIST archive identities), `SWEEP-RESULTS.md` and the run's own tables and figures |
| `sweep-variants/` | `geometry-variants-public.json`: the secondary geometry measurements and their default-match checks |
| `sweep-stacked/` | The figures as presented: `sweep-digits` (Figure 2), `sweep-synthetic` (Figure 3), `sweep-instability-{digits,synthetic}` (Figures 4–5), supplementary `sweep-accuracy-*` plots with ERM and true-subspace controls, captions, `sweep-summary.json` and `sweep-table.tsv` |

## The original runner

Appendix H states that the earlier fixed-energy comparison and its rank,
predictor-norm, bridge-shift and full CR-path diagnostics can be reproduced
separately. That is the original runner:

```sh
sh experimental/collection_geometry/run.sh
```

It compares 256 pairs that cover one unstable direction, both directions unevenly
(15:1), or both equally, with clean spectra (2, 0), (1.875, 0.125) and (1, 1). It
writes a new `results/` folder (or `--output /absolute/fresh/path`) with the raw
records, summaries, tables, figures and a `RESULTS.md` report; its outputs are not
included here. `PROTOCOL.md` records its settings and the validation-only amendments
made before test data were generated. `--pilot` (validation-only replicates) and
`--skip-digits` (synthetic task only) are development options, not the reported
experiment.

## Tests

```sh
uv run pytest -q experimental/collection_geometry
```

These check the cue geometry, image-identity and test-data isolation, and the
fitting and reporting code.
