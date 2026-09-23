# TGRS nonpolar classical/template baseline audit (current repaired data)

Status: **valid final output**. Use this `v4` directory.

## Protocol and leakage result

- Split lock: `tgrs_nested_nonpolar_v2`.
- Held geometry: four strict object-disjoint folds, 20 held objects in total, and all 60 unique nonpolar directions for every held object (1,200 predictions per method). No train/dev/test sample overlap and no train/test or dev/test object overlap was found.
- ID: sample-disjoint only; the same 20 objects occur in train/dev/test. Each object has 40/10/10 directions in train/dev/test, so the ID test contains 10 directions per object, **not** all 60. The three-way union is all 60 directions per object, and each ID split collectively covers the 60 direction classes.
- Leakage details: `leakage_audit.json` (`status=ok`).

## Held-object results that can be cited

All intervals are a fixed-seed, 50,000-replicate bootstrap over 20 object-level means. The 1,200 angle rows are not treated as independent experimental units.

| Deterministic template method | Mean GCE | Object-bootstrap 95% CI | Top-1 | P90 / P95 GCE | Worst object mean |
|---|---:|---:|---:|---:|---:|
| Operational amplitude/sin/cos phase 6-channel 1-NN | **27.25 deg** | [16.19, 39.54] deg | **70.50%** | 120.00 / 138.59 deg | 85.01 deg |
| Operational 6-channel, common-global-phase invariant 1-NN | 28.37 deg | [17.24, 40.75] deg | 69.50% | 120.00 / 138.59 deg | 82.13 deg |
| Operational 6-channel, common-phase + integer-range-cell invariant 1-NN | 30.95 deg | [19.74, 42.95] deg | 67.00% | 120.00 / 138.59 deg | 80.54 deg |
| Complex coherent correlation 1-NN | 34.43 deg | [23.37, 46.27] deg | 70.08% | 128.68 / 180.00 deg | 90.02 deg |
| Complex coherent correlation, common-global-phase invariant 1-NN | 33.57 deg | [22.23, 45.94] deg | 70.00% | 128.68 / 180.00 deg | 93.70 deg |
| Complex matched filter, common-phase + integer-range-cell invariant 1-NN | 33.51 deg | [22.15, 45.76] deg | 69.58% | 128.68 / 180.00 deg | 90.70 deg |

The strongest deterministic nonlearned baseline remains operational 6-channel 1-NN. Removing one common phase costs 1.11 deg mean GCE (paired object-bootstrap 95% CI [0.12, 2.18] deg), and removing both common phase and discrete range cell costs 3.69 deg [1.98, 5.51] deg. Absolute phase/range reference is therefore materially predictive in this simulator corpus; invariance is not a free robustness improvement. The invariant variants slightly lower the worst-object mean but do not repair the catastrophic tail.

The dev-only class-template top-k controls selected `k` independently in each fold from `{1,3,5,7,10}` and did not improve held-object accuracy (35.89--35.94 deg mean GCE). No test labels were used for selecting `k`.

## ID results and seed policy

Operational 6-channel 1-NN obtains 33.72 deg mean GCE and 62.50% Top-1 on the 200-query ID test. Its common-phase and phase+range invariant versions obtain 35.87/59.50% and 39.18/57.00%, respectively. ID must not be described as object-disjoint or as a 60-direction-per-object test.

All methods are deterministic and nonlearned. Multiple training seeds are inapplicable; the relevant replication axes are the four fixed held-object folds and 20 object-level contributions. The bootstrap seed affects only Monte Carlo approximation of confidence limits and is fixed in provenance.

## Audit of the old `baselines_fixed` directory

The 2026-08-02 remote result predates the 480-sample repair: exactly 480 of the 1,200 evaluated current NPZ files have modification times after the old `summary.json`. Against current data, old predictions mismatch 142/1,200 operational held queries and 162/1,200 complex-correlation held queries. That directory has no dataset content digest and must not be cited for the repaired corpus.

The repository's original `evaluate_tgrs_nested_baselines.py` was independently rerun on current data. The new evaluator reproduces its operational and complex-correlation predictions exactly for held (1,200/1,200) and ID (200/200), with zero prediction or GCE mismatch. See `current_reference_audit.json`.

## Reproducibility and limitations

- Evaluator: `code/tools/evaluate_tgrs_nested_template_correlations.py`.
- Full predictions: `samples.csv`; per-object metrics: `objects.csv`; pooled metrics: `pooled.csv`; selections: `selections.json`.
- Dataset/split/script hashes and exact command/environment: `provenance.json`.
- Evaluated NPZ combined path+content SHA-256: `3b7a0f0fadcabb9d36be8e38fac710f417596d27a2c61c4e52cc4fe7d3c818fb`.
- Range alignment searches all 91 integer circular FFT range cells with one common shift for both receive polarizations. It does not interpolate sub-bin delays and preserves relative frequency and cross-polarization phase.
- These controls remain limited to the same PEC/CST simulated corpus and are not evidence of measured-data or cross-solver generalization.
