# GRIT: Out-of-Distribution Robustness from Finite and Noisy Invariant Pairs

Code for the paper *GRIT: Out-of-Distribution Robustness from Finite and Noisy
Invariant Pairs* (arXiv link forthcoming). GRIT estimates a linear subspace from
invariant pair differences on fixed pretrained features and projects it out before
predictor fitting, so the same constraint can accompany ERM or a compatible robustness
objective.

This repository contains the code, configurations and result records for the
experiments reported in the paper:

- the frozen-feature objective comparisons on ColoredMNIST-INV and Waterbirds-INV
  (Tables 1 and 3, Sec. 6.1, App. F);
- the UrbanCars pair-allocation study (Tables 2 and 4–6, App. G);
- the controlled CueMNIST/Gaussian studies (Figures 2–5, App. H).

The paper's Appendices F, G and H define the protocols.

In code, configurations and records, ColoredMNIST-INV is `cmnist` and Waterbirds-INV
is `waterbirds` (protocol `waterbirds_cf/v1`). The base objectives are ERM, SD, RDM,
IRMv1, V-REx, Fishr, GroupDRO and LISA. Each runs vanilla (`<base>`), with prediction
consistency (`<base>_consistency`, "+PC") and with GRIT (`<base>_grit`); ERM's two
variants are named `erm_consistency` and `grit`.

## Using GRIT on your own features

GRIT fits a nuisance subspace to invariant pair differences and projects it out
before the predictor is trained. After the setup below:

```python
from grit.methods.projection import fit_linear_projection

# pair_left, pair_right: [n_pairs, d] features of inputs that should receive the
# same prediction (e.g. one image under two backgrounds), from a frozen encoder.
projection = fit_linear_projection(
    pair_left,
    pair_right,
    requested_rank=k,  # the paper selects k on validation data
    pair_manifest_digest="none",  # provenance labels used by the paper pipeline
    feature_cache_manifest_digest="none",
)
train_projected = projection.transform(train_features)
# Fit any classifier on train_projected, and apply projection.transform to
# validation and test features before prediction.
```

`projection.diagnostics` reports the singular values and the fraction of
pair-difference energy removed.

## Check the reported results without data or training

This needs only Python 3.10 or later:

```sh
python3 results/verify_results.py
```

It recomputes every reported value in Tables 1–6, the related appendix contrasts and
statements, and the Appendix H sweep values from the per-seed and per-replicate
records in `results/` and `experimental/collection_geometry/`. Each value is printed
beside its manuscript value. After the setup below,
`uv run python results/verify_results.py` also checks that the shipped configurations
and code regenerate the recorded search plans, summaries and UrbanCars bindings.
`results/README.md` describes the records.

## Environment

Python 3.10.20 is pinned in `.python-version`. Install `uv`, then run:

```sh
git clone https://github.com/inouye-lab/GRIT.git
cd GRIT
uv sync --locked --group dev
export PROJECT_SCRATCH=/absolute/path/to/private/scratch
export WANDB_MODE=disabled
```

`PROJECT_SCRATCH` is a writable directory outside this folder for datasets, prepared
CLIP features and outputs; the configurations resolve their paths through it. Where
supported, the preparation programs download the CLIP weights and public datasets.
Waterbirds-INV also needs the released Waterbirds data, CUB images and masks, and
Places backgrounds. UrbanCars needs its component datasets and the pinned upstream
image-generation dependencies. Source images and prepared features are not
distributed here. Optional W&B logging is off unless `WANDB_PROJECT` is set.

## Paper-to-code-and-results map

| Paper | Source and configuration | Records (`results/` unless noted) | Recompute from records |
| --- | --- | --- | --- |
| Tables 1 and 3, Sec. 6.1, App. F: ColoredMNIST-INV | `scripts/prepare_cmnist.py`; the eight `configs/cmnist/held-out-e03/*-search.yaml` | `cmnist/<tree>/`: plan, selections, per-seed summaries | `verify_results.py`; `scripts/summary_table.py results/cmnist/<tree>` |
| Tables 1 and 3, Sec. 6.1, App. F: Waterbirds-INV | `scripts/prepare_waterbirds.py`; the eight `configs/waterbirds/*-search.yaml` | `waterbirds/<tree>/` | as above |
| Table 2, Tables 4–6, App. G | `scripts/prepare_urbancars.py`, `scripts/urbancars_gen/`, `configs/urbancars/v1.yaml`; setup checks `check.yaml`, `optimizer.yaml`, `optimizer-v2.yaml` | `urbancars/urbancars-v1-standard*/`, `urbancars/construction/` | `verify_results.py` |
| Figures 2–5, App. H | `experimental/collection_geometry/` (see its README) | `experimental/collection_geometry/sweep-results/`, `sweep-variants/`, `sweep-stacked/` | `sweep_report.py` |
| Figure 1 | schematic; no data | — | — |
| App. I | discussion; no additional experiments | — | — |

The ColoredMNIST-INV GroupDRO and LISA rows use `groupdro-label-source-env-…` (four
label-by-source groups) and `lisa-source-env-…`; the Waterbirds-INV rows use
`groupdro-…` (four label-by-background groups) and `lisa-…`. `results/README.md` lists
each tree and its method IDs. Configuration and record names carry an
`…-representation-…` label from the recorded runs; see the note on the records in
`results/README.md`.

Where the methods are implemented:

| Component | Code |
| --- | --- |
| GRIT projection (fit on pair differences, applied to the probe input) | `src/grit/methods/projection.py`; `LinearProbeAlgorithm` in `src/grit/methods/training.py` |
| Prediction consistency and its composition with a base objective | `prediction_consistency_penalty` in `src/grit/methods/training.py`; `src/grit/methods/interventions.py` |
| ERM, GroupDRO, V-REx, IRMv1 | `src/grit/methods/training.py`, `invariance.py`, `groupdro.py` |
| LISA, SD, Fishr, RDM | `src/grit/methods/baselines.py`, `lisa.py`, `invariance.py`, `fishr.py`, `rdm.py` |
| Pair construction | `src/grit/data/cmnist.py`, `src/grit/data/waterbirds_pairs.py`, `src/grit/search/pair_inputs.py` |
| Validation selection (tuning, confirmation, frozen winners) | `src/grit/selection/`, `src/grit/search/{plan,scheduler,lifecycle}.py` |
| Dataset search adapters | `src/grit/search/{cmnist,waterbirds}.py` |
| UrbanCars construction, features and search | `src/grit/data/urbancars.py`, `src/grit/features/urbancars.py`, `src/grit/search/urbancars.py`, `src/grit/geometry.py` |
| Controlled studies | `experimental/collection_geometry/`, `src/grit/data/collection_geometry.py`, `src/grit/geometry.py` |

## Run the frozen-feature experiments

ColoredMNIST-INV uses the held-out `e03` rendering:

```sh
uv run scripts/prepare_cmnist.py --held-out-flip-prob 0.3
uv run scripts/run_search.py configs/cmnist/held-out-e03/erm-representation-interventions-search.yaml
```

Run the other seven objective YAMLs in that directory the same way. For
Waterbirds-INV, put the four asset roots under
`$PROJECT_SCRATCH/data/{waterbirds,cub,cub-masks,places}` or pass the matching flags to
`prepare_waterbirds.py`, then run its eight objective YAMLs:

```sh
uv run scripts/prepare_waterbirds.py
uv run scripts/run_search.py configs/waterbirds/erm-representation-interventions-search.yaml
```

Each complete search writes its plan, validation-selected winners, per-seed final
results and summary under its configured `output_root`. Interrupted searches resume.
`uv run scripts/smoke.py cmnist` and `uv run scripts/smoke.py waterbirds` run a quick
end-to-end check with fake encoders; their numbers are never paper results.
`uv run scripts/summary_table.py OUTPUT_ROOT` prints a completed search's results
without retraining or reading test data again.

## Run the UrbanCars study

1. `uv run scripts/prepare_urbancars.py download --stanford-cars <mirror>` fetches the
   component sources; Stanford Cars must come from an authorized mirror.
2. `bash scripts/urbancars_gen/gen.sh setup`, then `masks`, `fallback` and `export`,
   produce the car masks and object exports in an isolated environment.
3. `uv run scripts/prepare_urbancars.py build --construction standard`, then
   `features --construction standard`, rebuild the images and their CLIP features.
4. `uv run scripts/run_search.py configs/urbancars/v1.yaml` fits on validation data
   only, freezes the winners and prints their hash. Rerunning with
   `--unseal-test WINNERS_HASH` evaluates the test split.

`v1.yaml` covers both selectors, all five allocations, and the optimizer and reference
controls. `check.yaml`, `optimizer.yaml` and `optimizer-v2.yaml` are the setup checks
and optimizer choice it depends on.

## Run the controlled studies

```sh
sh experimental/collection_geometry/run_sweeps.sh
```

`experimental/collection_geometry/README.md` describes the outputs, how to regenerate
the figures from the saved run, and the original runner that Appendix H also cites.

## Tests

`uv run pytest` runs the unit, contract and compact-lifecycle tests on fake encoders
and synthetic data (about 20 minutes on one CPU), and
`uv run pytest experimental/collection_geometry` tests the controlled studies. Three
tests skip: one needs CUDA and two cover Waterbirds group definitions that do not
apply. A fourth needs a Git checkout and skips in a source archive. The W&B
offline-mode test also skips where the environment forbids W&B's local service; no
test contacts W&B servers.

## License

MIT; see `LICENSE`.

## Citation

If you use this code, please cite the paper (see `CITATION.cff`):

```bibtex
@misc{bai2026grit,
  title  = {{GRIT}: Out-of-Distribution Robustness from Finite and Noisy Invariant Pairs},
  author = {Bai, Ruqi and Currie, Easton and Ji, Yao and Kim, Mingyu and Zhou, Zeyu and Inouye, David I.},
  year   = {2026},
  note   = {arXiv preprint, forthcoming}
}
```
