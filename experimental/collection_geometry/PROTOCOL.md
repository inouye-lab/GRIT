# Small collection-geometry and residual-instability experiment

## Reporting standards (2026-09-25, before local runs)

This self-contained experiment reuses the generator, projection and group metrics
from `src/grit/`. It is isolated from those shared implementation files.

Primary question: at fixed k=256 and clean difference energy 2 per pair, compare
one direction, 15:1 uneven coverage, and balanced coverage of two orthogonal
unstable directions. Their clean spectra are (2,0), (1.875,.125), and (1,1).
Add identical independent Gaussian noise draws across collection arms, with
coordinate standard deviation .3; also report the clean control. The single
direction arm intentionally fails the spanning assumption. Do not renormalize
observed differences, which would change the noise model.

Synthetic features have d=96, stable rank 4, unstable rank 2, 3,000 labels,
160 validation observations/group and 400 test observations/group. Reuse the
upstream latent generator with one dimension per unstable factor; keep all other
upstream source-model defaults. Normalize its clean bank differences locally.
Twenty replicates redraw bases, labels and pairs. Semi-synthetic MNIST replaces
Gaussian task content with real handwritten digits and two designed visible style
cues; its precise construction is recorded in the local README. Use ten split/bank
replicates. This is engineered fixed-feature evidence, not pretrained-feature evidence.

Fit binary logistic probes with intercept and fixed ridge coefficient 1e-4 by a
local deterministic Newton solver (gradient infinity norm <=1e-7), avoiding
optimizer/checkpoint search. Reuse the exact upstream GRIT SVD. Quadratic CR uses
the upstream sum-over-logits convention with symmetric logits (-f/2,f/2):
alpha/(2 tr(M)) w' M w. Alpha grid: 0,.001,.01,.1,1,10,100,1000.
GRIT ranks: 0,1,2,4,8,16. The primary selector maximizes worst validation-group
accuracy; secondary maximizes source-weighted validation accuracy. Ties prefer
smaller rank/strength. Fixed rank two is the primary geometry contrast.
ERM and true-U projection are controls. All candidates and selection records are
frozen and hashed before generating/evaluating each replicate's test data.
Test metrics cannot select any configuration. Main test distributions include
single-cue reversals and both reversed, with equal label weights. Report both-shift
accuracy and missed-subspace operator norm; retain all group metrics and logistic loss.
Report means and seed SD, plus paired 95% intervals using a fixed bootstrap over
replicates. MNIST intervals describe split/bank variation on a fixed dataset.

Finite-pair geometry: k=64,128,256,512,1024,2048,4096; sigma=.1,.3,.6;
32 independent noise/basis replicates; balanced and uneven designs. Fit log-log
slopes on k>=512, fixed before results, and report slope uncertainty across
replicates. Measure covariance fluctuation and missed-U norm, not an accuracy rate.

Appendix diagnostics: under/overselection, raw true-U predictor sensitivity,
stable label-direction removal, all CR strength paths, stable-only reweighting
of coupled source/bridge examples, and growing unstable shift. These are mechanism
diagnostics, not numerical certificates or theorem/proof validation. The synthetic
finite-label optimizer differs from the population optimizer in the theorem.

Keep all results, including adverse comparisons. Pilot changes require a dated
amendment before full-test access. Record source/config/data hashes, elapsed time,
package versions, and all seeds. CPU float64 is preferred for these small exact
linear algebra operations; GPU use is unnecessary unless measured beneficial.

## Local modifications to reused code

No shared implementation is edited or monkey-patched. Local changes are generator
parameters, clean-difference normalization, and a convex Newton solver for the
same logistic/projection/quadratic-penalty objectives. Additional diagnostics and
MNIST construction live here.

## Pre-test amendment (2026-09-25)

The two-replicate synthetic validation pilot completed in 1.2 seconds, with
all Newton fits converged. This permits adding sigma=.1 and .6 to prediction
diagnostics alongside the predeclared primary .3 and clean control. Primary setup,
grids and claims are unchanged. This is a runtime-driven extension before any
local test evaluation, not a choice among test outcomes. The first pilot remains
preserved with its original configuration and source hashes.

The same heldout pilot exposed coefficient amplification: at sigma=.3, balanced
geometry had missed-U norm .189 but weak-ridge synthetic both-shift validation
accuracy .328 (oracle .884). Low-variance retained features supported large probe
norms. An explicit validation-only comparison retained the original generator,
examined feature noise alternatives (not adopted), and compared ridge 1e-4,.01,.1.
The final fixed ridge is **.1 for every method** and primary sigma is **.1** for
both datasets; all four noise levels remain fully reported. This choice provides
an interpretable moderate-noise case, and sigma=.3/.6 expose deterioration rather
than disappearing from the report. There is no test-based setup choice.
Balanced rank-two fits at ridge=1e-4 and sigma=.1/.3 are additionally retained as
the coefficient-amplification stress case. All pilot observations and alternatives
are saved, including adverse results. These amendments supersede the initial
ridge and primary-noise values above. Ridge fitting is an empirical variant of the
unregularized population-fitting theorem and is not claimed to instantiate it exactly.
