# Internal Evaluation Repair

> Status: complete on 2026-07-11; tests, full five-fold run, and independent OOF integrity check passed.

## Why This Exists

The historical internal estimate (`0.988 +/- 0.008`) used patient-grouped folds, but it remained image-level and reused each fold for checkpoint selection and scoring. The model configuration was also selected after comparing configurations on the single held-out test set. A data audit additionally found exact duplicate image files under different HYGD patient IDs.

The historical result remains part of the project history. It is not the preferred estimate for PI-facing or research-facing use.

## Fixed Protocol

- Detect exact image duplicates with SHA-256.
- Link patient IDs that share an exact image into one independent evaluation group.
- Count each exact image hash once.
- Fix the model configuration before outer evaluation.
- Use five stratified outer folds over independent evaluation groups.
- Use a separate stratified inner validation split for best-epoch and threshold selection.
- Never calculate or display outer-test metrics during training.
- Save every out-of-fold image prediction.
- Report image-level and duplicate-aware group-level AUROC, sensitivity, and specificity.
- Use cluster bootstrap over independent evaluation groups.
- Treat the threshold as cross-validated reporting output, not as a deployment recommendation.

The configuration was frozen before the repaired outer-fold run, but it came from earlier HYGD development. Therefore this is a repaired **post-development internal resampling estimate**, not a prospectively untouched estimate of the entire model-development process.

## Full-Workspace Reproduction Record

The commands and outputs below belong to the full audited research workspace.
This first public reviewer snapshot includes the narrative result but not the
later executable or row-level result bundle, so these commands are not expected
to run from the public snapshot alone.

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
python -m unittest validation.test_internal_evaluation_repair
```

## Outputs

- `results/internal_evaluation_repair.json`
- `results/internal_evaluation_repair_audit.json`
- `results/internal_evaluation_repair_group_folds.csv`
- `results/internal_evaluation_repair_oof_images.csv`
- `results/internal_evaluation_repair_oof_groups.csv`

Provenance note: the `*_audit.json` file is the run-start data/split manifest
and retains the literal status `running`; it is not the post-run verdict. The
completed result file records all five outer folds, and a separate read-only
OOF recomputation verified the row/group coverage, AUROC, and confusion matrix.

## Final Results

The complete run evaluated every unique hash and independent group exactly once:

- source rows: 747;
- unique SHA-256 image hashes: 737;
- exact duplicate groups: 10, including 6 spanning different supplied patient IDs;
- independent linked evaluation groups: 283;
- label conflicts across exact duplicates: 0.

An exploratory perceptual-hash/pixel-correlation scan was also run across different groups. Fundus framing created many similarity false positives; visual review of the strongest candidates showed different vascular anatomy. No probabilistic near-duplicate merge was made. Exact SHA-256 equality remains the auditable grouping rule.

| Analysis level | N | AUROC (95% cluster-bootstrap CI) | Sensitivity | Specificity | Confusion matrix (TN, FP, FN, TP) |
|---|---:|---:|---:|---:|---|
| **Duplicate-aware group (primary)** | **283** | **0.9904 (0.9797-0.9980)** | **0.9563** | **0.9700** | 97, 3, 8, 175 |
| Image (secondary) | 737 | 0.9837 (0.9731-0.9925) | 0.9500 | 0.9289 | 183, 14, 27, 513 |

Outer-fold group AUROCs: 0.9757, 0.9932, 0.9851, 1.0000, 1.0000. Inner-validation-selected thresholds: 0.7240, 0.9479, 0.4829, 0.8898, 0.7767. The threshold spread is a calibration warning; it is not a clinical operating-point recommendation.

The independent post-run check re-read the OOF CSVs, confirmed 737 unique hashes and 283 unique groups with one outer-fold assignment each, and reproduced the group AUROC and confusion matrix exactly.

## Interpretation Rule

The duplicate-aware group-level estimate is the preferred internal descriptive result; image-level performance is secondary. It supports strong in-distribution discrimination after correcting the identified evaluation defects. Residual post-selection optimism remains possible because the model recipe was historically informed by HYGD. It does not repair external target adaptivity, prove calibration, or establish clinical utility.
