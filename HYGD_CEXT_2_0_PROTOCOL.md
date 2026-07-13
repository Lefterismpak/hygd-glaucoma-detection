# HYGD-CEXT-2.0 Source-Only Frozen-Feature Protocol

**Status:** pre-download frozen on 2026-07-13

**Protocol ID:** `HYGD-CEXT-2.0`

**Purpose:** one bounded source-only feasibility test after the v1.1 transportability failure

**Target access:** prohibited

**Model/repository/weight download:** awaiting explicit project-owner approval

## Question

Can one fixed, full-frame DINOv2-S/14 representation reduce source-domain shortcut dependence enough to meet predeclared three-source leave-one-source-out performance and dataset-origin gates, without target data, target anatomy, fine-tuning, or recipe search?

## Fixed Prior Evidence

- `HYGD-CEXT-1.1` remains closed and unchanged. Candidate A equal-source mean AUROC was 0.6227 [0.5822, 0.6632]; HYGD and PAPILA were near chance. Candidate B was geometry-ineligible and no classifier was trained.
- `HYGD-MANUAL-GEOM-1` showed strong center localization but failed diameter and combined gates. The original manual-source masks used a synthetic 0.25-image-side diameter.
- Post-hoc scalar correction repaired resubstitution geometry but left dataset-origin accuracy at 0.9420 versus majority/chance 0.4541. It is not a candidate in this protocol.

Frozen local inputs:

| Input | SHA-256 |
|---|---|
| `results/confirmatory/source_audit.json` | `9e504a7627174abc45f8f6ad6e6f45677ce204e469d88af08541680bfb092735` |
| `results/confirmatory/candidate_A/qualification_summary.json` | `9653bda8fb16a069d64567a42114ec66d1d7f8c1413e02074658d501bd90887c` |
| `validation/confirmatory_protocol_lock_v1.1.json` | `755d093f537104bd25b127e32155ecab18a74fc629b6316ebf7a3c4b47a604b5` |
| `results/hygd_boundary_adjudication_v1.json` | `1f53bb1963a723c4a9e99e22c95f71fc50e61a8b4cc5002acf5bc4ea1769413a` |
| `results/hygd_scale_calibration_feasibility.json` | `944dbb0f156d34d63457cdf9bb4c70ef1afee5a655027a12fedb0a6ea35f43cb` |

## Single Candidate

`F0 = dinov2_vits14 frozen CLS features + one fixed logistic-regression head`

- Official entrypoint: `dinov2_vits14`.
- Architecture stated by the official model card: ViT-S/14, 21M parameters, 384-dimensional embedding.
- Repository license checked on 2026-07-13: Apache License 2.0.
- Weight-specific URL, resolved repository commit, byte size, and SHA-256 remain `needs-proof` until an explicitly approved first download creates a local receipt.
- DINOv2 was not trained or validated for glaucoma diagnosis. This protocol tests representation transportability only.

Official sources:

- https://github.com/facebookresearch/dinov2
- https://github.com/facebookresearch/dinov2/blob/main/MODEL_CARD.md
- https://github.com/facebookresearch/dinov2/blob/main/LICENSE

## Fixed Data And Units

- Reuse the exact audited HYGD, PAPILA, and RIM-ONE rows, content hashes, duplicate clusters, labels, and eligibility rules from v1.1.
- Reuse the exact three leave-one-source-out folds.
- HYGD is evaluated on linked duplicate-aware groups.
- PAPILA is evaluated at eye level with patient-cluster bootstrap.
- RIM-ONE remains image-level with true subject independence labelled `needs-proof`.
- No new source, target, image exclusion, manual correction, pseudo-label, or target-derived statistic is allowed.

## Fixed Preprocessing

For every source image independently:

1. Decode as RGB.
2. Preserve the full image; no optic-disc crop, segmentation, or localizer is used.
3. Resize the longest side to 518 pixels with bicubic interpolation.
4. Reflect-pad the shorter side symmetrically to 518 x 518.
5. Convert to float tensor and use fixed ImageNet normalization: mean `[0.485, 0.456, 0.406]`, standard deviation `[0.229, 0.224, 0.225]`.
6. Extract the final normalized CLS token once in evaluation mode and L2-normalize it.

No CLAHE, colour constancy, augmentation, test-time augmentation, multi-crop view, segmentation mask, learned preprocessing, or per-dataset rule is allowed.

## Fixed Classifier

- One scikit-learn logistic regression per held-out source.
- `solver=liblinear`, `C=1.0`, `penalty=l2`, `max_iter=5000`, `random_state=20260713`.
- Training sample weights equalize the total contribution of each available `source x class` cell.
- No hyperparameter search, feature selection, calibration, MLP, fine-tuning, or seed selection.
- The existing Candidate A metrics are the fixed comparator; Candidate A is not rerun or retuned.

## Evaluation

Primary disease metric:

- duplicate-/unit-aware AUROC on each held-out source;
- equal-source mean AUROC;
- 2,000 fixed-seed cluster-bootstrap replicates using the same unit logic as v1.1.

Shortcut metric:

- three-way dataset-origin logistic-regression probe on the frozen F0 features;
- four group-aware folds, fixed `C=1.0`, no hyperparameter search;
- report accuracy and majority/chance.

Integrity checks:

- exact row/hash/fold reconciliation with the frozen source audit;
- no train/evaluation unit overlap;
- finite 384-dimensional feature for every eligible image;
- deterministic rerun hash for features and predictions;
- one label-permutation negative control per held-out source, reported but not used for tuning.

## Promotion Gate

F0 passes only if all conditions hold:

1. Source-audit and fold-integrity checks pass.
2. Every held-out source AUROC is at least 0.60.
3. Equal-source mean AUROC is at least 0.70.
4. At least two of three held-out-source 95% confidence-interval lower bounds exceed 0.50.
5. Dataset-origin probe accuracy is below 0.75.
6. No classifier, preprocessing, feature, or metric deviation occurred.

There is no fallback candidate. A failed condition closes v2.0 without fine-tuning, another foundation model, target access, or post-result recipe change.

## Authorization Boundary

Before explicit project-owner approval:

- do not clone or install the DINOv2 repository;
- do not download or load model weights;
- do not add dependencies or alter the environment;
- do not inspect a new target;
- do not train or evaluate F0.

After approval, the first allowed action is a receipt-only acquisition that records official URL, resolved commit, license copy, byte size, SHA-256, and local path. If that receipt differs from this protocol's model identity or carries an incompatible notice, execution stops.

Even a v2.0 pass does not authorize clinical use, publication claims, external submission, challenge registration, target access, or a new Lock B. Each requires a separate project-owner decision.
