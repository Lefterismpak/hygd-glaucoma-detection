# HYGD-CEXT-2.0 Execution Specification

**Status:** implementation-frozen before feature extraction

**Protocol:** `HYGD-CEXT-2.0`

**Scientific recipe:** unchanged from `HYGD_CEXT_2_0_PROTOCOL.md`

This file resolves implementation details that the protocol intentionally described at a higher level. It was written before any source-image DINOv2 feature was extracted or any F0 classifier was fitted.

## Provenance

- Official model entrypoint: `dinov2_vits14` from Meta's DINOv2 repository.
- Repository commit: `7764ea0f912e53c92e82eb78a2a1631e92725fc8`.
- Weight SHA-256: `b938bf1bc15cd2ec0feacfe3a1bb553fe8ea9ca46a7e1d8d00217f29aef60cd9`.
- The official entrypoint is instantiated with `pretrained=False`; the receipted state dictionary is then loaded locally with `torch.load(..., weights_only=True)` and `strict=True`. This avoids an extra Torch-Hub cache copy or network call.
- No dependency is added. CPU inference uses the existing project environment.

## Deterministic Preprocessing

1. Rows are the 1,642 exact-hash representatives in the frozen source manifest, sorted by `source_row_id`.
2. RGB images are resized with Pillow bicubic interpolation. The scale is `518 / max(width, height)`; the shorter integer dimension uses Python `round`, with the longest dimension forced to 518.
3. Odd padding puts `floor(delta / 2)` pixels on the top or left and the remainder on the bottom or right.
4. Padding uses NumPy `reflect` mode, channel axes unpadded.
5. Pixel values are converted to float32 in `[0,1]`, then normalized by the fixed ImageNet mean and standard deviation.
6. `forward_features(...)["x_norm_clstoken"]` is L2-normalized and stored as float32.
7. Batch size is 4, DataLoader workers are 0, CPU threads are 8, and deterministic Torch algorithms are enabled.

## Disease Classifier

- Each fold uses the frozen fold CSV unchanged.
- Logistic regression is exactly `solver=liblinear`, `C=1.0`, `penalty=l2`, `max_iter=5000`, `random_state=20260713`.
- Each available `source x class` cell receives equal total weight. Per-row weights are then rescaled to mean 1, preserving the conventional interpretation of `C=1.0`.
- HYGD predictions are averaged within the frozen linked subject cluster before AUROC.
- PAPILA remains eye-level for AUROC, with patient cluster resampling for confidence intervals.
- RIM-ONE remains image-level and its subject independence remains `needs-proof`.
- The 2,000-replicate source and equal-source-mean bootstraps reuse seed `20260711` from v1.1.

## Dataset-Origin Probe

- Labels are fixed as `HYGD=0`, `PAPILA=1`, `RIMONE=2`.
- Four folds use `StratifiedGroupKFold(n_splits=4, shuffle=True, random_state=20260713)` with frozen subject clusters as groups.
- Each fold uses unweighted multinomial-capable logistic regression with `solver=lbfgs`, `C=1.0`, `penalty=l2`, `max_iter=5000`, `random_state=20260713`.
- The gate metric is the unweighted mean of the four held-out fold accuracies. Pooled out-of-fold accuracy is reported as a secondary check.

## Negative Controls And Determinism

- One negative control per held-out source permutes disease labels independently within each training source, preserving every source-class count. Seeds are `20260714`, `20260715`, and `20260716` for HYGD, PAPILA, and RIM-ONE respectively.
- Feature extraction is run twice from a fresh process. Exact `.npy` SHA-256 equality is required.
- Evaluation is run once from each feature file. Exact prediction CSV SHA-256 equality is required for all disease, permutation-control, and origin-probe predictions.
- Any mismatch, missing feature, non-finite value, fold leakage, receipt mismatch, or protocol deviation fails integrity and blocks promotion.

## No-Fallback Rule

No result can trigger feature tuning, another model, a different image view, target access, or threshold adjustment inside this protocol. A failed gate closes `HYGD-CEXT-2.0`.
