# Internal Evaluation Repair

> Historical run status: metrics from the private 2026-07-11 five-fold result were independently recomputed from the complete OOF files, but its run-start audit remained literally `running`; no contemporaneous terminal audit or public timestamp attestation exists.

> **Public implementation update, 2026-09-04:** The repository now includes the Torch-free evaluation utilities, nested evaluator, synthetic regression tests, lightweight CI, and a privacy-safe [aggregate receipt](results/repaired_internal_evaluation_summary.json). No model was retrained. The `2026-09-04-v5` code performed a read-only aggregate reanalysis of frozen private OOF predictions from a legacy run asserted to date from 2026-07-11; these numbers are not an end-to-end v5 model run and have no contemporaneous public timestamp attestation.

## Why This Exists

The historical internal estimate (`0.988 +/- 0.008`) used patient-grouped folds, but it remained image-level and reused each fold for checkpoint selection and scoring. The model configuration was also selected after comparing configurations on the single held-out test set. A data audit additionally found exact duplicate image files under different HYGD patient IDs.

The historical result remains part of the project history. It is not the preferred estimate for PI-facing or research-facing use.

## Prospective v5 Protocol

- Detect exact image duplicates with SHA-256.
- Link patient IDs that share an exact image into one independent evaluation group.
- Count each exact image hash once.
- Fix the model configuration before outer evaluation.
- Use five stratified outer folds over independent evaluation groups.
- Use a separate stratified inner validation split for best-epoch and threshold selection.
- Never calculate or display outer-test metrics during training.
- Save every out-of-fold image prediction.
- Make the equal-weight mean of outer-fold group AUROCs the primary discrimination estimand; retain pooled OOF AUROC only as a continuity metric.
- Report image-level and duplicate-aware group-level AUROC, sensitivity, and specificity.
- Bootstrap whole evaluation groups within each fixed outer fold for the primary interval.
- Treat the threshold as cross-validated reporting output, not as a deployment recommendation.

This is the required behavior of the prospective v5 evaluator, not a description of how the legacy model run was executed. Its configuration is fixed before any future v5 outer evaluation, but it came from earlier HYGD development. A future complete run would therefore remain a repaired **post-development internal resampling estimate**, not a prospectively untouched estimate of the entire model-development process.

## Public reproduction path

The executable path is public. A full numerical reproduction still requires the
HYGD dataset and private model-run artifacts; the synthetic suite verifies the
integrity contracts without claiming to reproduce the AUROC. The public
[aggregate receipt](results/repaired_internal_evaluation_summary.json) publishes
metrics and SHA-256 commitments, not row-level predictions.

Data/split audit only:

```bash
python validation/internal_evaluation_repair.py --audit-only
```

Complete evaluation:

```bash
python validation/internal_evaluation_repair.py
```

Tests:

```bash
python -m unittest discover -s validation -p 'test_*.py' -v
```

Frozen-OOF reanalysis contract (safe to inspect without private files):

```bash
python validation/reanalyze_frozen_oof.py --describe-contract
```

The reanalysis entry point is read-only and aggregate-only: it neither trains nor
runs inference nor writes files. Private input locations are supplied only via
local environment variables and are excluded from its scientific command and
public receipt.

## Local outputs (git-ignored, never publication artifacts)

- `results/internal_evaluation_repair.json`
- `results/internal_evaluation_repair_audit.json`
- `results/internal_evaluation_repair_group_folds.csv`
- `results/internal_evaluation_repair_oof_images.csv`
- `results/internal_evaluation_repair_oof_groups.csv`

`--audit-only` automatically uses `internal_evaluation_repair_audit_only*`; a
partial `--max-folds N` run uses `internal_evaluation_repair_smoke_Nof5*`, has no
confidence interval, and cannot emit the preferred schema. Runs refuse to
overwrite any existing bundle entry. Choose a new direct prefix such as
`--output-prefix results/internal_evaluation_repair_20260831a` to preserve prior
artifacts.

Only the locked full defaults (5 folds, 10 epochs, batch size 32, seed
`20260711`, inner fraction 0.15, sensitivity target 0.95, and 5,000 bootstrap
samples) can emit `status: complete` with the preferred schema. Any altered
training or evaluation settings require a proper-subset smoke run and remain
noncanonical.

The 2026-07-11 workspace audit file was a run-start manifest and retained the
literal status `running`; the complete private result plus independent OOF
integrity recomputation are the basis for treating the model run as completed.
They do not provide a contemporaneous terminal audit or public timestamp. The public v5 implementation gives audit-only and
partial smoke runs distinct noncanonical namespaces, refuses every pre-existing
bundle target, writes a terminal audit before publishing the result as the bundle
commit marker, and records hashes for each referenced artifact. A complete
terminal bundle is accepted only after the evaluator recomputes the canonical
group-to-fold assignment from the audited dataset and matches it against both
the fold manifest and OOF rows; the terminal audit has an exact closed schema
and is bound to the complete result. This is
prospective code hardening, not a model rerun.

## Final Results

The complete run evaluated every unique hash and independent group exactly once:

- source rows: 747;
- unique SHA-256 image hashes: 737;
- exact duplicate groups: 10, including 6 spanning different supplied patient IDs;
- independent linked evaluation groups: 283;
- label conflicts across exact duplicates: 0.

An exploratory perceptual-hash/pixel-correlation scan was also run across different groups. Fundus framing created many similarity false positives; visual review of the strongest candidates showed different vascular anatomy. No probabilistic near-duplicate merge was made. Exact SHA-256 equality remains the auditable grouping rule.

| Analysis level | N | AUROC (95% conditional CI) | Sensitivity | Specificity | Confusion matrix (TN, FP, FN, TP) |
|---|---:|---:|---:|---:|---|
| **Equal-weight mean outer-fold group AUROC (primary)** | **283** | **0.9908 (0.9790-0.9990)** | **0.9563** | **0.9700** | 97, 3, 8, 175 |
| Pooled group OOF probabilities (continuity only) | 283 | 0.9904 (0.9797-0.9980) | — | — | — |
| Image (secondary) | 737 | 0.9837 (0.9731-0.9925) | 0.9500 | 0.9289 | 183, 14, 27, 513 |

Outer-fold group AUROCs: 0.9757, 0.9932, 0.9851, 1.0000, 1.0000. Thresholds recorded as inner-validation-selected in the legacy result were 0.7240, 0.9479, 0.4829, 0.8898, and 0.7767; their selection was not independently rerun because the inner prediction rows and bound checkpoint bytes are unavailable. The threshold spread is a calibration warning, not a clinical operating-point recommendation.

The independent post-run check re-read the OOF CSVs, confirmed 737 unique hashes and 283 unique groups with one outer-fold assignment each, and reproduced the pooled group AUROC and confusion matrix exactly. The private read-only v5 aggregate reanalysis then reproduced fold AUROCs 0.9757, 0.9932, 0.9851, 1.0000, and 1.0000; their equal-weight mean is 0.9908 (group-count-weighted sensitivity analysis: 0.9907). It also checked the legacy result's recorded thresholds, best epochs, history structure, and run-start audit for post-hoc consistency; it did not rerun inner selection or bind legacy checkpoint bytes. Exact source-file commitments and aggregate values are recorded in [`results/repaired_internal_evaluation_summary.json`](results/repaired_internal_evaluation_summary.json).

## Interpretation Rule

The equal-weight mean outer-fold group AUROC is the preferred internal discrimination estimate; pooled OOF AUROC and image-level performance are secondary. This avoids ranking raw probabilities across separately fitted fold models. It supports strong in-distribution discrimination after correcting the identified evaluation defects. Residual post-selection optimism remains possible because the model recipe was historically informed by HYGD. It does not repair external target adaptivity, prove calibration, or establish clinical utility.

The primary 5,000-sample percentile interval resamples whole evaluation groups
within each fixed outer fold and averages fold AUROCs equally. The deterministic
Monte Carlo contract uses seed `20269714`, stable within-fold group ordering, and
yielded 5,000/5,000 valid replicates. Sensitivity and specificity pool
fold-specific decisions recorded in the legacy result. All intervals condition
on the frozen OOF predictions and recorded fold/decision choices; they exclude
uncertainty from retraining, checkpoint and threshold selection, recipe selection,
transportability, and deployment. The pooled group and image intervals instead use a
global group-cluster bootstrap over fixed pooled scores; both AUROCs are
secondary because they compare score scales from separately fitted fold models.
