# HYGD-SHORTCUT-MAP-1 Protocol

**Status:** frozen before diagnostic execution

**Frozen:** 2026-07-13

**Owner:** project owner

**Scope:** source-shortcut mechanism attribution only

## 1. Decision Question

Can fixed, source-blind preprocessing materially reduce the overwhelming dataset-origin signal observed in the closed `HYGD-CEXT-2.0` run without destroying its cross-source glaucoma signal?

This is a diagnostic study. It is not a new qualification, a model-promotion attempt, Lock B, external validation, or clinical evidence. `HYGD-CEXT-2.0` remains closed as `failed_closed_no_fallback` regardless of this result.

## 2. Frozen Inputs

- The 1,642 exact-hash representative rows already audited for `HYGD-CEXT-2.0`: HYGD 737, PAPILA 420, RIM-ONE DL 485.
- The three unchanged leave-one-source-out fold manifests and subject/hash leakage controls.
- The existing local official DINOv2 ViT-S/14 repository and weight receipt used by `HYGD-CEXT-2.0`.
- The exact baseline feature matrix and primary qualification summary from `HYGD-CEXT-2.0`.

No new data, target, repository, weight, model, dependency, annotation, account action, download, or external submission is authorized.

## 3. Frozen Representation And Classifiers

- Backbone: frozen DINOv2 ViT-S/14, official entrypoint `dinov2_vits14`.
- Input: 518 x 518 RGB after the branch-specific deterministic transform.
- Feature: `forward_features(...)["x_norm_clstoken"]`, then row-wise L2 normalization.
- Feature dimension/dtype: 384 / float32.
- Runtime: CPU, batch size 4, eight CPU threads, deterministic PyTorch algorithms.
- Disease head, source/class weights, LOSO folds, evaluation units, 2,000-replicate cluster bootstrap, and within-training-source permutation controls: unchanged from `HYGD-CEXT-2.0`.
- Origin probe: four-fold `StratifiedGroupKFold`, shuffled with seed 20260713, fixed multinomial logistic regression, mean held-out fold accuracy.

## 4. Frozen Branches

### S0 - Closed CEXT Baseline

Reuse, without re-extraction or reinterpretation, the verified `HYGD-CEXT-2.0` primary features and results:

- equal-source mean AUROC: 0.7104536193472292;
- dataset-origin mean-fold accuracy: 0.9993917274939172.

### S1 - Retinal-Support Center Square

For each decoded RGB image:

1. Compute float32 luminance `0.299 R + 0.587 G + 0.114 B` on the original 0-255 pixels.
2. Mark provisional foreground where luminance is strictly greater than 10.
3. A row is active when at least 5% of its pixels are foreground; a column is active under the analogous rule.
4. The support bounding box runs from the first through the last active row and column. Missing support, a support fraction below 10%, or a box dimension below 64 pixels is a hard error.
5. Take the largest centered square inside that box. Its side is `min(box_width, box_height)`; integer left/top coordinates use floor of center minus half-side.
6. Resize that square directly to 518 x 518 with Pillow bicubic interpolation. No padding, reflection, rotation, augmentation, or dataset-specific rule is allowed.

### S2 - S1 Plus Fixed Per-Image Grayscale Standardization

Apply S1, then:

1. Recompute luminance with the same coefficients.
2. Define support as luminance strictly greater than 10. Support below 10% or support standard deviation below 1.0 is a hard error.
3. Compute the population mean and standard deviation over support pixels only.
4. Standardize every support pixel, clip z-scores to `[-2.5, 2.5]`, and map linearly to `[0, 255]` using `(z + 2.5) / 5 * 255`.
5. Set non-support pixels to zero, round to the nearest integer, cast to uint8, and repeat the single channel three times.

This deliberately removes color and first-order per-image luminance variation. It is a mechanism test, not a claim that this is clinically optimal preprocessing.

### S3 - LEACE Feature-Space Mechanism Probes On S0

S3 uses only the frozen S0 features. The implementation follows the closed-form whitening/projection construction in the official EleutherAI LEACE implementation, with explicit supported settings: affine `true`, covariance shrinkage `false`, covariance-trace constraint `false`, and SVD tolerance `0.01`.

Two analyses are fixed:

- **S3-CF:** cross-fitted erasure. For the origin probe, fit the eraser only on each origin-training fold and apply it to that train/evaluation pair. For each disease LOSO direction, fit the eraser only on the two training sources and apply it to training and held-out features.
- **S3-GLOBAL:** fit one three-source eraser on all 1,642 source labels, then rerun the origin probe and disease LOSO analyses. This is explicitly transductive and mechanism-only because held-out source identity contributes to the eraser.

S3 uses one-hot source labels, float64 covariance algebra, no post-erasure L2 renormalization, and the same fixed downstream classifiers. No S3 result can promote a model or satisfy the preprocessing repair gate.

Primary method sources:

- LEACE paper: https://papers.nips.cc/paper_files/paper/2023/hash/d066d21c619d0a78c5b557fa3291a8f4-Abstract-Conference.html
- Official implementation: https://github.com/EleutherAI/concept-erasure

## 5. Frozen Outputs And Metrics

For S1 and S2, run independent `primary` and `rerun` feature extractions. Exact feature-file SHA-256 equality is required. Re-evaluate both runs and require exact hashes for all disease, permutation-control, and origin prediction files.

Report for every branch:

- three held-out-source AUROCs and cluster-bootstrap 95% confidence intervals;
- equal-source mean AUROC and its cluster-bootstrap 95% confidence interval;
- four origin fold accuracies, mean-fold accuracy, and pooled OOF accuracy;
- fixed within-source permutation-control AUROCs;
- deltas from S0;
- all file, feature, and prediction hashes needed for independent audit.

## 6. Predeclared Diagnostic Gates

A preprocessing branch is a **bounded repair candidate** only when both conditions hold:

1. origin mean-fold accuracy is at least 0.10 lower than S0, therefore `<= 0.8993917274939172`; and
2. equal-source mean AUROC loses no more than 0.03 from S0, therefore `>= 0.6804536193472292`.

This label authorizes only discussion of a separately frozen next protocol. It does not reopen or modify CEXT-2.0.

If neither S1 nor S2 passes both gates, the fixed full-frame preprocessing repair lane closes. Do not iterate thresholds, add transforms, swap backbones, or inspect a new target under this protocol.

S3 is interpreted only as attribution:

- a large source-accuracy reduction with preserved disease AUROC supports a linearly encoded source shortcut that may be separable from disease signal;
- a large disease collapse supports entanglement of disease and source signal;
- residual origin accuracy after LEACE shows remaining non-erased or fold-shift source information.

These interpretations remain non-causal and `needs-proof` outside the frozen datasets.

## 7. Hard Boundaries

- No target access or target-time adaptation.
- No fine-tuning, neural training, model selection, hyperparameter search, or fallback.
- No result-dependent branch, transform, threshold, seed, or erasure setting.
- No public, publication, clinical, or deployment claim.
- RIM-ONE DL subject independence and broader multi-dataset training compatibility remain `needs-proof`.
- Any implementation deviation closes the protocol and must be documented; it cannot be silently repaired after results are visible.
