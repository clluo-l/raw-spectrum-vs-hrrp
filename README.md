# Raw Spectrum vs. HRRP

Code and evidence for the paper

> **Raw Spectrum Beats Processed HRRP for Neural Radar Look-Direction Inference
> from a Single Coherent Sweep**
> (IEEE Transactions on Aerospace and Electronic Systems, 2026)

The name mirrors the paper's central matched comparison — the raw complex
frequency spectrum versus the processed rectangular-IFFT high-resolution range
profile (HRRP) — with the verdict left to the data. The repository lets you
independently check the numbers reported in the paper:
the frozen train/development/test splits, the sample-level prediction rows behind
every aggregate, the table/figure inputs, and the regeneration code. It does not
attempt end-to-end retraining or full-wave re-simulation.

## What is inside

```
code/                 regeneration and audit scripts (Python 3, NumPy/Pandas/Matplotlib)
evidence/             frozen evidence produced by the audited runs
  dataset/            centered-HRRP dataset audit (per-object IFFT centering checks)
  deterministic/      deterministic cosine-1NN baseline and invariance controls
  direction_tail/     held-mesh direction-resolved tails, confusion, paired effects
  fig3_predictions/   per-seed prediction rows behind the main-paper object-effects figure
  formal_stress/      measurement-domain stress suite (AWGN / phase / range / calibration)
  function_match/     conjugation audit between the two bases (35 checkpoints)
  hann/               one-factor Hann-window secondary control
  review_data/        20-object fixed-direction raw complex-response subset (CSV)
  splits/             every frozen split list + archive-relative split manifest
  strict_rect91/      primary strict matched-transform results
audit/scientific_gate.json   machine-audit receipts (row counts, gate statuses)
```

The manuscript itself is not redistributed here; its figures and numerical
values are reproducible from `code/` and `evidence/`.

## What is *not* included

Raw ShapeNet meshes, the complete 1680-file complex NPZ corpus, CST simulation
projects, and trained checkpoints are excluded for source-license and size
reasons. The split lists and the released raw-response subset still pin the
audited corpus exactly.

Native controller, audit, provenance, and figure-input records that contain
machine paths, hostnames, or account names were omitted rather than redacted;
their scientific values are represented by the derived CSV/JSON files.

## Minimal result audit

1. Recompute split counts, direction support, object disjointness, and zero
   sample overlap from `evidence/splits/`
   (`code/generate_tgrs_nested_splits.py` documents the construction).
2. Regenerate object-centered and paired summaries from the released prediction
   rows (`code/evaluate_hrrp_class_predictions.py`,
   `code/audit_tgrs_matched_rect91_mlp.py`).
3. Reproduce the table/figure inputs
   (`code/make_fig3_matched_object_effects.py`,
   `code/make_hrrp_dataset_overview_figure.py`, ...).
4. Check the rectangular-IFFT construction on
   `evidence/review_data/tgrs_complex_response_subset.csv`
   (20 objects x 91 complex samples at (90 deg, 0 deg)) with
   `code/export_tgrs_repaired_complex_subset.py`.

## Provenance

The split lists are deterministic: for each object `o` (lexicographic index),
polar index `t`, and `s = o mod 12`, the two ID-test azimuth indices at polar
index `t` are `(2t+s) mod 12` and `(2t+6+s) mod 12`, and the two development
indices are `(2t+3+s) mod 12` and `(2t+9+s) mod 12`. Outer folds hold five
objects each (supplementary material, Table S2) with zero train--test,
development--test, and train--development overlap.

The original audit artifact shipped a per-file SHA-256 manifest for offline
integrity checks. In this repository, git history and tagged releases serve
as the authoritative version record.

## Citation

If this repository helps your work, please cite the paper above. A BibTeX
entry will be added upon acceptance.

## License

Code is released under the MIT License (see `LICENSE`). Derived evidence files
(CSV/JSON) are provided for verification of the paper's reported numbers.
