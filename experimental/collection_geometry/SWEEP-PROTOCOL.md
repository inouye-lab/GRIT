# Three components of finite-pair recovery

## Reporting standards — 2026-09-25, before these runs

This revision answers the author's request to replace balance-centered framing
with the smallest clean eigenvalue. The integrated GRIT theorem uses
`(sigma/sqrt(g) + sigma²/g) sqrt(h/k) + (sigma²/g) h/k`, where
`g=lambda_min(Sigma_k|U)` and `h=d+log(1/eta)`. The largest clean eigenvalue is
absent from this recovery factor. This is not a universal risk/accuracy power law.
Earlier fixed-energy results and their sources remain separate, unchanged evidence.

Use the existing synthetic and engineered two-cue MNIST (CueMNIST) constructions
and exact upstream GRIT projection. CueMNIST is the running example; synthetic is appendix
support. Fix d=96, r=2, 3,000 training labels and ridge=.1. Reuse the earlier
eight-group validation/test definitions, fixed encoder, CR alpha grid and
worst-group validation selector. Ten CueMNIST and twenty synthetic replicates use
seed 20260927, fixed before evaluation. No test metric chooses a sweep range,
reference setting, predictor, figure subset or fitting interval.

Each clean pair changes just one cue. The runner constructs its feature
difference analytically and assigns the endpoints to recorded training anchors;
it does not materialize the noisy image endpoints. Existing render/extract
operations realize the clean cue interventions exactly. Alternate the two directions with equal
counts, with displacement sqrt(2 lambda_max) or sqrt(2 g). Then the actual clean
empirical spectrum is exactly (lambda_max,g) for every even k. Here lambda_max=1.
Training anchors cycle when k exceeds the 3,000 training identities. Every noisy
pair measurement receives independent Gaussian acquisition noise; k counts noisy
measurements, not distinct identities. Master noise and anchors have nested
prefixes across pair counts. Common data, directions and noise draws are used
across sweep conditions within each replicate.

Three independently varied axes share the reference (g=1,sigma=.1,k=256):

| Axis | Values | Fixed quantities |
| --- | --- | --- |
| Smallest clean eigenvalue g | 1/64,1/32,1/16,1/8,1/4,1/2,3/4,7/8,1 | lambda_max=1, sigma=.1, k=256 |
| Pair-noise SD sigma | .025,.05,.075,.1,.15,.2,.3,.45,.6,.8,1 | (lambda_max,g)=(1,1), k=256 |
| Pair count k | 64,96,128,192,256,384,512,1024,2048,4096 | (lambda_max,g)=(1,1), sigma=.1 |

Add g=0 at the reference noise/budget as an explicitly nonspanning control,
and sigma=0 at the reference spectrum/budget as an exact-geometry control. These
controls are reported separately, not assigned artificial log coordinates.
Clean trace is 1+g in the eigenvalue sweep: this is NOT a fixed-energy comparison.
Do not normalize observed noisy differences. At fixed g and sigma, extending k
does not change the exact empirical clean eigenvalues.

Freeze every fitted candidate, validation-selected CR strength and input hash for
a replicate before creating its test features. Fixed-rank GRIT is the primary
geometry comparison, not a claim that practical rank is known. ERM and true-U
projection remain controls. No new optimizer or hyperparameter search is needed.

Main figure: three labeled side-by-side panels, 5.5 inches wide, showing GRIT
subspace error and normalized residual sensitivity `||U' w||/||w||`. These obey a
deterministic ordering for the projected predictor, but their empirical values
need not be equal. Distinguish curves with both markers and line styles, plus
accessible colors. Means and sample seed SD are displayed. The pair-count panel
includes a k^-1/2 shape reference anchored to the measured mean subspace error at
the predeclared reference k=256. It is not a calibrated theorem bound. Estimate
log-log slopes separately per replicate only at k>=512, chosen before the runs.
Endpoint comparisons use matched replicate differences with 5,000 bootstrap
resamples and a fixed seed. Retain all points and adverse outcomes.

The appendix includes the same three panels for target accuracy with GRIT,
validation-selected CR, ERM and true-U projection, and the synthetic counterparts.
Main captions lead with visible, panel-specific findings; setup, metrics, fixed
controls and uncertainty follow those claims. Avoid condition-number or balance
language as an independent cause at fixed weakest signal. The earlier study's
fixed-energy result can be interpreted as increasing its weakest eigenvalue.

Scope: finite-label ridge fitting, engineered global linear cues and feature-space
Gaussian pair noise. Neither fitted accuracy rates nor numerical certification
of the complete population-risk theorem are claimed. Weak-signal/high-noise
subspace error can saturate at one. The full bound has a 1/k term; the reference
line and slope are specifically large-k diagnostics.

## Reuse and local changes

New `sweeps.py`, `sweep_report.py`, focused sweep tests and `run_sweeps.sh` reuse the
original local solver, MNIST encoder, upstream generator, metrics and SVD.
No shared implementation is copied, monkey-patched or changed. The old runner,
its frozen outputs and original paper insertion remain available as historical
evidence; this protocol governs the new primary sweep figure.

## Author-requested presentation revision — 2026-09-25, after frozen results

The main figure now has two aligned rows: GRIT subspace error above, GRIT and
validation-selected CR target accuracy below. The three columns retain every
original point and the same axes/controls. The Gaussian task has an analogous
appendix stack; subspace error plus normalized sensitivity move to appendix
mirror figures. Other-method accuracy controls remain in the saved tables and
supplemental plot outputs. This replaces the presentation described above, not
the pre-run grids, estimators, selectors, metrics or fitting intervals.

Use concise claim-first captions with only essential interpretation/defaults.
Put full technical definitions and setup in the methods text; put column labels
below the plots. The fixed-grid CR comparison retains its upper-grid-boundary
qualification. This is a presentation revision after viewing frozen outcomes,
not a new pre-registration or fresh experiment.

## Secondary geometry settings — 2026-09-25, before supplemental computation

Following author review of the primary plots, add two nearby settings around
its reference curve in each top panel. These are geometry-only additions:

- Smallest-eigenvalue sweep: noise SD 0.05, 0.1 (default), 0.2; k=256.
- Noise sweep: k=128, 256 (default), 512; clean spectrum (1,1).
- Pair-count sweep: noise SD 0.05, 0.1 (default), 0.2; clean spectrum (1,1).

Retain every original x-axis value, replicate, dimension, geometry and Gaussian
noise stream. Reuse noise prefixes across pair counts. Reconstruct the known
subspaces from the original deterministic construction and verify their hashes
against the frozen input records. Recompute subspace error using the existing
projection estimator; verify that all default points match the original raw
results. No predictor is fitted or reselected, and no new test evaluation is
performed for the two added curves. Bottom-row GRIT/CR accuracies remain those
of the original defaults. This is a disclosed follow-up after primary results,
not a claim that the additional settings were part of the original protocol.

Use one color with distinct markers and line styles for the three geometry
curves, and a small within-panel legend naming the secondary settings. Preserve
independent y ranges including all means and sample-SD bands, and shared x axes
within columns. The caption distinguishes secondary top-row settings from the
bottom-row defaults. Save raw supplemental measurements and summaries separately
from the frozen primary outputs.

## Acceptance and public packaging — 2026-09-25

The author accepted the revised experiment as CueMNIST and requested integration
into the active manuscript. This naming and publication step does not change the
pre-run settings, reported measurements, or disclosed follow-up scope. The public
supplement is `sweep-variants/geometry-variants-public.json`: all measurements and
recorded hashes are unchanged, while its run-location metadata uses a
repository-relative path instead of an absolute local checkout path. Subsequent
runs record a repository-relative run location, or only the output directory's
name when it is outside the code repository. Canonical results and validation
pilots are retained; duplicate numerical reruns and machine-specific build/cache
files are excluded from the public package.
