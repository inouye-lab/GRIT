# Controlled sweeps of pair information

All methods and CR choices were frozen before test evaluation. Plots report replicate means with sample-SD ribbons. Zero-noise and nonspanning controls are tabulated separately.

Coverage holds the largest clean eigenvalue fixed, so clean trace varies. Noise and pair-count sweeps hold both clean eigenvalues fixed.

## Paired endpoint contrasts

Differences are last minus first positive-noise sweep point, or last minus first coverage/pair-count point. Intervals use 5,000 fixed-seed paired bootstrap resamples. Accuracy changes are percentage points.

| Data | Sweep | Method | Metric | From → to | Mean change | 95% CI |
|---|---|---|---|---|---:|---:|
| digits | coverage | cr | normalized_sensitivity | 0.015625 → 1 | -0.5047 | [-0.5350, -0.4834] |
| digits | coverage | cr | both_shift | 0.015625 → 1 | 65.5000 | [62.9600, 68.0500] |
| digits | coverage | cr | both_loss | 0.015625 → 1 | -0.1445 | [-0.1534, -0.1360] |
| digits | coverage | erm | normalized_sensitivity | 0.015625 → 1 | 0.0000 | [0.0000, 0.0000] |
| digits | coverage | erm | both_shift | 0.015625 → 1 | 0.0000 | [0.0000, 0.0000] |
| digits | coverage | erm | both_loss | 0.015625 → 1 | 0.0000 | [0.0000, 0.0000] |
| digits | coverage | grit | missed_u | 0.015625 → 1 | -0.4710 | [-0.5043, -0.4368] |
| digits | coverage | grit | normalized_sensitivity | 0.015625 → 1 | -0.4395 | [-0.4706, -0.4066] |
| digits | coverage | grit | both_shift | 0.015625 → 1 | 72.4300 | [70.1100, 74.5700] |
| digits | coverage | grit | both_loss | 0.015625 → 1 | -0.9388 | [-1.0133, -0.8587] |
| digits | coverage | oracle | normalized_sensitivity | 0.015625 → 1 | 0.0000 | [0.0000, 0.0000] |
| digits | coverage | oracle | both_shift | 0.015625 → 1 | 0.0000 | [0.0000, 0.0000] |
| digits | coverage | oracle | both_loss | 0.015625 → 1 | 0.0000 | [0.0000, 0.0000] |
| digits | noise | cr | normalized_sensitivity | 0.025 → 1 | 0.9296 | [0.9227, 0.9349] |
| digits | noise | cr | both_shift | 0.025 → 1 | -77.6700 | [-78.6103, -76.7697] |
| digits | noise | cr | both_loss | 0.025 → 1 | 2.5147 | [2.4994, 2.5297] |
| digits | noise | erm | normalized_sensitivity | 0.025 → 1 | 0.0000 | [0.0000, 0.0000] |
| digits | noise | erm | both_shift | 0.025 → 1 | 0.0000 | [0.0000, 0.0000] |
| digits | noise | erm | both_loss | 0.025 → 1 | 0.0000 | [0.0000, 0.0000] |
| digits | noise | grit | missed_u | 0.025 → 1 | 0.7643 | [0.7255, 0.8036] |
| digits | noise | grit | normalized_sensitivity | 0.025 → 1 | 0.6705 | [0.6218, 0.7264] |
| digits | noise | grit | both_shift | 0.025 → 1 | -79.8900 | [-80.6400, -79.1700] |
| digits | noise | grit | both_loss | 0.025 → 1 | 2.0529 | [1.9437, 2.1761] |
| digits | noise | oracle | normalized_sensitivity | 0.025 → 1 | 0.0000 | [0.0000, 0.0000] |
| digits | noise | oracle | both_shift | 0.025 → 1 | 0.0000 | [0.0000, 0.0000] |
| digits | noise | oracle | both_loss | 0.025 → 1 | 0.0000 | [0.0000, 0.0000] |
| digits | pairs | cr | normalized_sensitivity | 64 → 4096 | -0.0059 | [-0.0169, 0.0048] |
| digits | pairs | cr | both_shift | 64 → 4096 | 7.5700 | [3.6400, 11.6400] |
| digits | pairs | cr | both_loss | 64 → 4096 | 0.0148 | [-0.0163, 0.0466] |
| digits | pairs | erm | normalized_sensitivity | 64 → 4096 | 0.0000 | [0.0000, 0.0000] |
| digits | pairs | erm | both_shift | 64 → 4096 | 0.0000 | [0.0000, 0.0000] |
| digits | pairs | erm | both_loss | 64 → 4096 | 0.0000 | [0.0000, 0.0000] |
| digits | pairs | grit | missed_u | 64 → 4096 | -0.1133 | [-0.1179, -0.1078] |
| digits | pairs | grit | normalized_sensitivity | 64 → 4096 | -0.0888 | [-0.0952, -0.0823] |
| digits | pairs | grit | both_shift | 64 → 4096 | 19.5200 | [17.7400, 21.3800] |
| digits | pairs | grit | both_loss | 64 → 4096 | -0.1963 | [-0.2129, -0.1788] |
| digits | pairs | oracle | normalized_sensitivity | 64 → 4096 | 0.0000 | [0.0000, 0.0000] |
| digits | pairs | oracle | both_shift | 64 → 4096 | 0.0000 | [0.0000, 0.0000] |
| digits | pairs | oracle | both_loss | 64 → 4096 | 0.0000 | [0.0000, 0.0000] |
| synthetic | coverage | cr | normalized_sensitivity | 0.015625 → 1 | -0.3540 | [-0.3760, -0.3335] |
| synthetic | coverage | cr | both_shift | 0.015625 → 1 | 35.4062 | [33.3812, 37.3880] |
| synthetic | coverage | cr | both_loss | 0.015625 → 1 | -0.2264 | [-0.2637, -0.1880] |
| synthetic | coverage | erm | normalized_sensitivity | 0.015625 → 1 | 0.0000 | [0.0000, 0.0000] |
| synthetic | coverage | erm | both_shift | 0.015625 → 1 | 0.0000 | [0.0000, 0.0000] |
| synthetic | coverage | erm | both_loss | 0.015625 → 1 | 0.0000 | [0.0000, 0.0000] |
| synthetic | coverage | grit | missed_u | 0.015625 → 1 | -0.5042 | [-0.5205, -0.4886] |
| synthetic | coverage | grit | normalized_sensitivity | 0.015625 → 1 | -0.3629 | [-0.3773, -0.3483] |
| synthetic | coverage | grit | both_shift | 0.015625 → 1 | 32.9437 | [31.3250, 34.7125] |
| synthetic | coverage | grit | both_loss | 0.015625 → 1 | -0.4215 | [-0.4396, -0.4034] |
| synthetic | coverage | oracle | normalized_sensitivity | 0.015625 → 1 | 0.0000 | [0.0000, 0.0000] |
| synthetic | coverage | oracle | both_shift | 0.015625 → 1 | 0.0000 | [0.0000, 0.0000] |
| synthetic | coverage | oracle | both_loss | 0.015625 → 1 | 0.0000 | [0.0000, 0.0000] |
| synthetic | noise | cr | normalized_sensitivity | 0.025 → 1 | 0.5863 | [0.5497, 0.6263] |
| synthetic | noise | cr | both_shift | 0.025 → 1 | -81.7188 | [-82.9437, -80.3562] |
| synthetic | noise | cr | both_loss | 0.025 → 1 | 0.7565 | [0.6053, 0.9331] |
| synthetic | noise | erm | normalized_sensitivity | 0.025 → 1 | 0.0000 | [0.0000, 0.0000] |
| synthetic | noise | erm | both_shift | 0.025 → 1 | 0.0000 | [0.0000, 0.0000] |
| synthetic | noise | erm | both_loss | 0.025 → 1 | 0.0000 | [0.0000, 0.0000] |
| synthetic | noise | grit | missed_u | 0.025 → 1 | 0.8133 | [0.7760, 0.8516] |
| synthetic | noise | grit | normalized_sensitivity | 0.025 → 1 | 0.5951 | [0.5672, 0.6239] |
| synthetic | noise | grit | both_shift | 0.025 → 1 | -80.2562 | [-81.7125, -78.8000] |
| synthetic | noise | grit | both_loss | 0.025 → 1 | 1.3411 | [1.2718, 1.4099] |
| synthetic | noise | oracle | normalized_sensitivity | 0.025 → 1 | 0.0000 | [0.0000, 0.0000] |
| synthetic | noise | oracle | both_shift | 0.025 → 1 | 0.0000 | [0.0000, 0.0000] |
| synthetic | noise | oracle | both_loss | 0.025 → 1 | 0.0000 | [0.0000, 0.0000] |
| synthetic | pairs | cr | normalized_sensitivity | 64 → 4096 | -0.0040 | [-0.0120, 0.0039] |
| synthetic | pairs | cr | both_shift | 64 → 4096 | 2.7937 | [1.9875, 3.6939] |
| synthetic | pairs | cr | both_loss | 64 → 4096 | 0.0479 | [0.0234, 0.0757] |
| synthetic | pairs | erm | normalized_sensitivity | 64 → 4096 | 0.0000 | [0.0000, 0.0000] |
| synthetic | pairs | erm | both_shift | 64 → 4096 | 0.0000 | [0.0000, 0.0000] |
| synthetic | pairs | erm | both_loss | 64 → 4096 | 0.0000 | [0.0000, 0.0000] |
| synthetic | pairs | grit | missed_u | 64 → 4096 | -0.1116 | [-0.1142, -0.1086] |
| synthetic | pairs | grit | normalized_sensitivity | 64 → 4096 | -0.0602 | [-0.0656, -0.0549] |
| synthetic | pairs | grit | both_shift | 64 → 4096 | 4.9250 | [4.4375, 5.3813] |
| synthetic | pairs | grit | both_loss | 64 → 4096 | -0.0780 | [-0.0834, -0.0725] |
| synthetic | pairs | oracle | normalized_sensitivity | 64 → 4096 | 0.0000 | [0.0000, 0.0000] |
| synthetic | pairs | oracle | both_shift | 64 → 4096 | 0.0000 | [0.0000, 0.0000] |
| synthetic | pairs | oracle | both_loss | 64 → 4096 | 0.0000 | [0.0000, 0.0000] |

## Pair-count log-log slopes

Fits use each replicate separately at prespecified k≥512. Normalized-sensitivity slopes describe fitted predictors, not a guaranteed accuracy or sensitivity rate.

| Data | Metric | Mean slope | Sample SD | Mean-slope 95% CI |
|---|---|---:|---:|---:|
| digits | missed_u | -0.5001 | 0.0313 | [-0.5187, -0.4816] |
| digits | normalized_sensitivity | -1.0037 | 0.3212 | [-1.2089, -0.8390] |
| synthetic | missed_u | -0.4968 | 0.0341 | [-0.5111, -0.4818] |
| synthetic | normalized_sensitivity | -0.8641 | 0.3014 | [-0.9963, -0.7332] |

## Separate controls

Control points do not enter sweep curves, endpoint contrasts, or slopes.
| Data | Control | Method | Missed U | Normalized instability | Accuracy (%) |
|---|---|---|---:|---:|---:|
| digits | clean_control | cr | — | 0.002 ± 0.000 | 80.130 ± 1.380 |
| digits | clean_control | erm | — | 0.937 ± 0.005 | 0.000 ± 0.000 |
| digits | clean_control | grit | 0.000 ± 0.000 | 0.000 ± 0.000 | 80.310 ± 1.342 |
| digits | clean_control | oracle | — | 0.000 ± 0.000 | 80.310 ± 1.342 |
| digits | nonspanning_control | cr | — | 0.937 ± 0.005 | 0.000 ± 0.000 |
| digits | nonspanning_control | erm | — | 0.937 ± 0.005 | 0.000 ± 0.000 |
| digits | nonspanning_control | grit | 0.997 ± 0.005 | 0.869 ± 0.014 | 0.020 ± 0.042 |
| digits | nonspanning_control | oracle | — | 0.000 ± 0.000 | 80.310 ± 1.342 |
| synthetic | clean_control | cr | — | 0.007 ± 0.005 | 89.350 ± 1.096 |
| synthetic | clean_control | erm | — | 0.795 ± 0.012 | 4.037 ± 0.820 |
| synthetic | clean_control | grit | 0.000 ± 0.000 | 0.000 ± 0.000 | 89.731 ± 1.177 |
| synthetic | clean_control | oracle | — | 0.000 ± 0.000 | 89.731 ± 1.177 |
| synthetic | nonspanning_control | cr | — | 0.663 ± 0.020 | 28.250 ± 2.457 |
| synthetic | nonspanning_control | erm | — | 0.795 ± 0.012 | 4.037 ± 0.820 |
| synthetic | nonspanning_control | grit | 0.995 ± 0.005 | 0.640 ± 0.021 | 36.281 ± 2.547 |
| synthetic | nonspanning_control | oracle | — | 0.000 ± 0.000 | 89.731 ± 1.177 |

No test outcome selects a configuration in this report. The shape guide is not a fitted theorem constant or a confidence bound. No result certifies the general theorem or a universal method ordering.
