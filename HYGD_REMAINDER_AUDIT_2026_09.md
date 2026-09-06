# HYGD remainder audit: data, training, anatomy and inference

**Audit date:** 2026-09-06. **Public baseline:** `1c9e8e917d7ef77daeb98d65a8661f029fd5ad67`.

The broader review found four reproducible defects: batch-dependent weighted-loss aggregation, malformed segmentation targets, permissive clinical-label parsing and a clamped temperature optimizer. They have different consequences. Correcting them does not automatically invalidate every previous result, and does not establish a better clinical model.

A complete v6 run repairs checkpoint-loss aggregation. It selects exactly the same epochs as v5 and reproduces its OOF CSV bytes. The performance result remains **0.9908 [0.9790, 0.9990]** mean outer-fold linked-group AUROC, with **97 TN / 3 FP / 4 FN / 179 TP**. There is **no measured performance gain** from this repair in the fixed run. The value is correctness and clearer evidence boundaries.

## Audit coverage and adjudication

| Component | Work performed | Finding / disposition |
| --- | --- | --- |
| Clinical question and labels | Rechecked HYGD and PAPILA primary descriptions; rebuilt PAPILA from the actual two clinical workbooks | HYGD predicts clinical GON labels from images. PAPILA 0=healthy, 1=glaucoma, 2=suspect; excluding suspects changes the target population. Public data do not establish a screening-population prevalence. |
| Data identity | Decoded and hashed all 737 byte-unique HYGD images; checked EXIF orientation; exploratory similarity/registration screen | 737 pixel-unique images, no additional exact-pixel duplicates, no nondefault EXIF orientation. Fifty-seven cross-group pHash pairs are **candidates only**, not patient-identity proof. |
| Training and checkpoint selection | Actual Torch batch-partition tests, a constructed checkpoint reversal, complete fixed v6 execution | Corrected global weighted CE. v5 replay remains explicitly available; v6 is separately identified. |
| Frozen parameters and augmentation | Traced the model/training mode and ran a real BatchNorm buffer test | Backbone parameters can be frozen while running statistics adapt. Augmentation was a modeling choice, not demonstrated clinical invariance. Descriptions corrected; v6 retains these behaviors. |
| Clinical-table ingestion | Invalid/ambiguous input tests plus actual PAPILA reconstruction | Rejects unknown/fractional diagnoses, malformed IDs, duplicates and missing included images. Actual cohort labels remain unchanged: 420 eyes, 210 supplied IDs, 87 glaucoma / 333 healthy. |
| Segmentation target preparation | Inspected all 970 RIM-ONE disc/cup masks and exercised the real dataset-to-BCE path with synthetic inputs | Every inspected mask uses 0/255. The retained two-channel builder passed 255 into BCE targets. Explicit 0/1 normalization added. |
| Anatomy/target provenance | Traced disc and VCDR builders into the forward/reverse disease losses | Upstream segmenters used PAPILA/RIM-ONE anatomy; the claim that PAPILA was excluded from segmenter training was wrong. Predicted HYGD VCDR is masked out of the final auxiliary loss. |
| Inference and parity | Added verified-image loading; reran the actual 99-image historical parity check with a hash-bound checkpoint | Passed: maximum single/batch difference **2.77e-6**, below 1e-4. Parity does not prove old training provenance, calibration or clinical validity. |
| Temperature scaling | Compared 12 fixed synthetic cases to an independent scalar optimizer | One counterexample trapped the old optimizer at T=0.001. Fixed positive fitting matches the independent oracle within 1e-7 NLL in all 12 cases. No frequency-of-failure estimate is implied. |
| CEXT / preprocessing / LEACE | Recomputed 18 source AUROCs and six means; reviewed fitting boundaries | Point estimates match. Cross-fitted disease and origin diagnostics use different fitted erasers; their pair of scores is not one deployable erased model. |
| Manual geometry | Recomputed the 36-case retained annotation/prediction join and gate metrics | 35 predictions; center rate 1.0000 among available predictions, diameter rate 0.6286, combined 0.6111 over all 36. The scale/combined failure remains. No fresh blinding or expert adjudication was established. |
| SWA | Inspected actual buffer averaging and the installed PyTorch contract | Buffer averaging is a supported alternative to a post-training BN refresh, not automatically a bug. Documented the actual variant; no retuning or claimed gain. |
| Artifact and runtime surfaces | Added fresh-output guards, declared missing legacy preparation dependencies, added CPU Torch/OpenCV CI | Existing label/coordinate/crop manifests are not silently replaced. Core tests and real numerical/image paths have distinct CI jobs; no local packages were installed. |

Detailed aggregate evidence and input hashes are in [the audit receipt](results/remainder_audit_20260906.json). This is a bounded component audit with the coverage above, not a guarantee that every possible defect or dependency has been ruled out. No independent human or agent peer review is claimed.

## 1. Weighted cross entropy: denominator matters

For integer targets and class weights, PyTorch's mean-reduced cross entropy divides the weighted loss sum by the **sum of target-class weights**. The older loops multiplied each batch mean by its image count and then divided by the total image count. With different class mixtures across batches, this does not recover the full-set weighted loss. [PyTorch CrossEntropyLoss](https://docs.pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html).

The real `src.train.train` loop, with a fixed toy model and identical examples, reported 0.220095 or 0.204567 solely depending on batch partition. A two-checkpoint counterexample makes the consequence explicit: the old aggregation prefers losses averaging 0.425 over 0.55, while the proper weighted objective prefers 0.52 over 0.65.

`src/loss_utils.py` now provides the correct denominator and accumulator. The versioned internal runner exposes `--protocol v6`; plain `--protocol v5` preserves the old batch-mean definition for replay. A fixed v6 run changes only reported epoch-loss aggregation and consequent checkpoint selection. Per-batch gradients, augmentation, BatchNorm adaptation, fold construction and threshold policy stay the same.

The [v6 execution plan](results/v6_execution_plan_20260906.json) was recorded locally before training and binds eight source files plus the cached weight bytes. It is published after execution, not independently timestamped preregistration. All eight hashes matched after completion. v6 used 5 folds × 10 epochs, batch 32, seed 20260711 and Apple MPS, with network denied in the process.

| Comparison | v5 | v6 |
| --- | --- | --- |
| Checkpoint epochs | 4, 10, 3, 9, 1 | 4, 10, 3, 9, 1 |
| OOF image and group CSVs | Retained reference | **Byte-identical** |
| Mean-fold AUROC | 0.9908 | 0.9908 |
| Conditional 95% interval | 0.9790–0.9990 | 0.9790–0.9990 |
| Confusion TN / FP / FN / TP | 97 / 3 / 4 / 179 | 97 / 3 / 4 / 179 |
| Recorded validation losses | Batch-dependent means | Correct global weighting |

The defect therefore **did not drive the reported discrimination in this run**. A separate NumPy pairwise-AUROC calculation and all 5,000 primary bootstrap draws matched the v6 result, as did Brier score and log loss. These intervals still condition on fixed predictions/folds and omit model-development, retraining and selection uncertainty. [Complete v6 aggregate](results/v6_complete_run_20260906.json).

## 2. Masks: 255 is an intensity, not a probability target

All 970 inspected RIM-ONE mask files contain values 0 and 255. `build_vcdr.py` previously resized/cast these masks to float without binarizing them. Its BCE criterion therefore received target values of 255. For example, BCEWithLogitsLoss at logit 1 and target 255 is approximately **−253.687**, demonstrating a malformed objective. BCE targets must lie between 0 and 1. [Official loss contract](https://docs.pytorch.org/docs/stable/generated/torch.nn.modules.loss.BCEWithLogitsLoss.html).

The corrected dataset path uses `binary_training_mask` before resizing/stacking. It accepts the documented 0/1 and 0/255 encodings, rejects unknown levels/nonfinite/wrong-shape arrays, and does not claim anatomical correctness. A regression exercises the real image loading, resizing, DataLoader and BCE-input path; only the heavyweight pretrained model constructor is replaced by a tiny synthetic model.

**Scope of impact:** this is a defect in the retained cup/disc generator code. Its exact effect on historical pseudo-label artifacts requires their execution/weight provenance. The final forward/reverse Attempt-B losses mask predicted HYGD VCDR out, so it is wrong to conclude that this defect automatically invalidates all disease AUROCs. No mixed-source segmentation retraining was performed.

## 3. The upstream anatomy path was target-assisted

The disc-localizer builder uses PAPILA and RIM-ONE images/contours to produce HYGD disc coordinates. The cup/disc builder also uses both datasets to produce HYGD pseudo-VCDR. PAPILA was therefore not excluded from **all upstream training**, even when it was absent from the final forward disease-label loss.

| Input / mechanism | Downstream use | Interpretation |
| --- | --- | --- |
| PAPILA + RIM-ONE images and disc contours | HYGD disc-localizer training → HYGD crops | Target-domain anatomy entered preprocessing in the adaptive chronology |
| Two-channel cup/disc generator | HYGD pseudo-VCDR | Old mask-encoding defect; final disease scripts mask these targets out of auxiliary loss |
| RIM-ONE/PAPILA source contours | Forward/reverse auxiliary VCDR target | Anatomy-related measurement, not a guaranteed source-invariant or shortcut-proof target |
| Two PAPILA expert contours | Rasterized union | Not an average or independently adjudicated consensus boundary |

The old generators used ordered image splits, not verified patient-disjoint random splits. Their Dice numbers are not independent-subject transfer validation. The inspected official HYGD description identifies digital fundus images and a TOPCON camera; it does not establish the stronger SLO-specific explanation previously used in the code comments. That attribution has been removed from current guidance. [HYGD primary record](https://physionet.org/content/hillel-yaffe-glaucoma-dataset/1.1.0/), [PAPILA primary paper](https://pmc.ncbi.nlm.nih.gov/articles/PMC9184612/).

## 4. Clinical import and inference integrity

The old PAPILA parser could truncate fractional diagnosis values, map unknown codes to healthy, truncate malformed identifiers, and silently drop missing images. The new parser fails instead. It recognizes the actual staggered two-row workbook header and excludes code 2 deliberately. Rebuilding both original workbooks gives exactly the old patient/eye/label/stem assignments; no historical label corruption was found in those retained files.

The old PAPILA and RIM-ONE label CSVs contain 905 paths that no longer resolve after the workspace relocation. Their bytes remain frozen. A new private PAPILA derivative contains the same labels with 420 valid current paths. External archived pipelines still require deliberate input reconstruction, applicable subject mappings and data-rights checks before a new run; this audit does not claim that every archived script is a one-command fresh-clone reproduction.

Single-image inference now decodes the same no-follow, hash-verified bytes as the batch reference, rejects training-mode/nonfinite outputs, and accepts an expected image digest. The parity CLI requires an explicit checkpoint and checksum. A real 99-image check passed with maximum difference 2.771615982e-6. The displayed scalar is an uncalibrated research score, not a clinical risk estimate or an image-quality/eligibility assessment.

## 5. Calibration optimizer: a demonstrated boundary trap

The old temperature fit optimized an unconstrained scalar through `clamp_min(0.001)` and an epsilon-modified probability loss. It could enter a flat clamped region and return extreme, incorrect scaling. A fixed synthetic seed-10 example yields T=0.001 and true Bernoulli NLL **45.5305**, whereas a scalar numerical oracle finds T≈**0.42577**, NLL≈**0.195446**.

The replacement optimizes exact, numerically stable Bernoulli NLL in bounded log-temperature space (T from 0.001 to 1000), explicitly includes T=1 and the bounds, checks convergence and records the fitting method/bounds. All 12 fixed synthetic cases agree with the independent oracle within 1e-7 NLL. A boundary solution is not proof of calibration quality. This experiment does not establish that a historical patient-output calibration failed, nor does a corrected fit establish prospective calibration.

## 6. Historical results that survived the review

Read-only replay of the bound prediction artifacts reproduced the following equal-source means:

| Analysis | Recomputed mean source AUROC |
| --- | ---: |
| CEXT-1.1 | 0.622720 |
| CEXT-2.0 | 0.710454 |
| S1 | 0.707189 |
| S2 | 0.736384 |
| S3-CF | 0.709571 |
| S3-GLOBAL | 0.701727 |

This checks point estimates from retained predictions, not a new fit or a full independent re-execution of each confidence interval. Those intervals remain conditional on selected sources, evaluation units and fitted models; they are not uncertainty intervals over arbitrary new hospitals. The retained manual geometry values also match: 36 records, 35 predictions, coverage 0.9722, center pass 1.0000 among available predictions, diameter pass 0.6286 and combined pass 0.6111 over all records. No new annotation or blinding claim follows.

For S3-CF, the disease analysis fits a separate eraser in each leave-one-source-out split, using its two training sources. The origin probe fits other erasers within four group folds over all three sources. Its disease AUROC and origin accuracy therefore do not characterize one common frozen erased model. S3-GLOBAL uses one common map fitted to all known sources and remains transductive. Neither result supplies an independently qualified clinical model.

The SWA variants average running buffers as part of the state. PyTorch explicitly supports buffer averaging as an alternative to a later activation-statistics update. The absence of `update_bn()` is therefore not by itself a demonstrated bug; a superiority claim would require an appropriate comparison. [PyTorch AveragedModel](https://docs.pytorch.org/docs/main/generated/torch.optim.swa_utils.AveragedModel.html).

## 7. Reproduce the corrected paths

```bash
# Existing full suite; CI also runs the real Torch/OpenCV regressions separately.
python -m unittest discover -s validation -p 'test_*.py' -v

# Public HYGD v1.1.0 in data/raw/, matching dependencies and cached/authorized weights:
python validation/internal_evaluation_repair.py --protocol v6 --device mps \
  --output-prefix results/internal_evaluation_repair_my_v6
python validation/evaluation_v6.py --result results/internal_evaluation_repair_my_v6.json \
  --expected-result-sha256 YOUR_OWN_COMPLETED_RESULT_SHA256

# Explicit historical v5 replay remains available:
python validation/internal_evaluation_repair.py --protocol v5

# A fresh private label derivative; existing outputs are refused:
python validation/make_papila_labels.py --dataset-root PATH_TO_PAPILA \
  --out validation/data/papila_labels_current.csv

# Existing local checkpoint; no pretrained download occurs in this loader:
python validation/verify_parity.py --raw-dir PATH_TO_HYGD \
  --checkpoint PRIVATE_CHECKPOINT --checkpoint-sha256 ITS_SHA256
```

The exact source-result receipt is verifiable only with its private bound bundle. A fresh training run requires public HYGD data, the specified recipe, appropriate weights and the recorded software environment; different hardware/runtime can change predictions. The standard full requirements now declare the previously implicit OpenPyXL/OpenCV/segmentation dependencies. Use a fresh environment and one OpenCV wheel variant; this task used already installed local packages.

The public tests/aggregates contain no real patient images, masks, row identifiers, row-level predictions or trained weights; tests use synthetic inputs. Existing legacy results and their provenance remain intact. Original source experiment outputs remain intact; correction backflow uses additive pointers after the private source audit.

## Decision and next proof

The strengthened result is a more trustworthy research artifact with actual v6 execution and demonstrated bug fixes. Its discrimination did not improve, and none of the changes establishes transportability, clinical utility, biological-subject independence or permission for new mixed-source work. The [RIM-ONE terms](https://github.com/miag-ull/rim-one-dl) remain a boundary for new mixed-source training/tuning.

A new study should test a defined clinical task and justified nuisance interventions or subject-exchangeable nulls, rather than treating source decoding alone as proof of harmful reliance. Keep the historical gates closed; have an ophthalmology PI review the corrected data/protocol assumptions before committing to another optimization study or a new target.
