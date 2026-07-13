# HYGD Manual Geometry Result

**Gate:** `HYGD-MANUAL-GEOM-1`

**Status:** closed - geometry metrics failed

**Classifier authorized:** no

**Scope:** exploratory HYGD source-domain geometry only; this does not reopen or alter frozen `HYGD-CEXT-1.1`.

## Answer

The manual-source localizer learned optic-disc **center** well but did not learn anatomical **diameter**. Its masks inherited the synthetic training diameter, so the resulting crops remain too large and highly dataset-separable. A one-scalar post-hoc correction can repair geometry on the same 36 images, but it does not repair the dataset shortcut. The scalar-only lane is therefore closed.

## Locked Evaluation

All 36 preselected, subject-disjoint images were accepted under the blinded five-click annotation contract. The locked evaluator verified the manifest, prediction, script, lock, and annotation hashes before computing the predeclared metrics.

| Metric | Result | Gate | Decision |
|---|---:|---:|---|
| Prediction coverage | 0.9722 (35/36) | >=0.95 | pass |
| Center pass rate | 1.0000 | >=0.90 | pass |
| Diameter pass rate | 0.6286 | >=0.85 | fail |
| Combined pass rate | 0.6111 | >=0.90 | fail |
| Median center error | 0.0778 reference diameters | <=0.50 | pass |
| Median diameter ratio | 1.7583 | 0.75-1.30 | fail |

Twelve predictions were above the per-image 1.80 diameter-ratio ceiling, one (`93_0`) was below 0.50, and one (`114_2`) was missing. These outcomes are retained; no image was replaced.

## Deterministic Overlay Review

The 36-tile overlay shows the human center/boundary in green and the frozen prediction in orange. It confirms that the numeric failure is not a coordinate-transform or rendering artifact:

- orange and green centers generally coincide;
- the orange diameter is systematically larger across the set;
- `93_0` is a genuine opposite-direction outlier in the recorded evidence;
- `114_2` remains visibly marked as missing;
- no post-lock annotation, threshold, sample, or prediction was changed.

Overlay: `results/hygd_boundary_adjudication_v1_overlay.png`

SHA-256: `63e76f3cc79fbef212b45c802158189235575f7ac7858f55f95bc337d53c53be`

## Root Cause

The earlier 100 `data/hygd_manual_disc_centers.csv` records were center clicks with a synthetic diameter equal to exactly `0.25 * max(image_width, image_height)` in 100/100 images. The localizer was trained against circular masks built from those values.

The new anatomical annotations resolve the old `needs-proof` diameter question:

- synthetic training diameter fraction: exactly 0.2500;
- frozen prediction median fraction: 0.2542;
- human reference-diameter median fraction: 0.1427 (IQR 0.1346-0.1502).

The model therefore reproduced its training target. This is a label-design failure for scale, not evidence that the center localizer failed.

## Scalar-Only Feasibility

A post-hoc scalar was derived as `median(reference / prediction) = 0.568744156295`. This is resubstitution evidence and is not a qualification result.

On the same 36 images:

- median calibrated diameter ratio: 1.0000;
- diameter pass rate: 0.9714;
- combined pass rate across all 36: 0.9444.

After applying the same scalar to all 728 available HYGD manual-source predictions and rebuilding all 1,603 source crops, the dataset-origin probe remained **0.9420** versus majority/chance **0.4541**. The pre-existing pass-ish boundary is `<0.75`.

Therefore anatomical scale correction alone does not remove the dominant dataset shortcut. The small change from 0.9563 to 0.9420 is insufficient to justify another boundary-annotation round for the same scalar question.

## Decision

1. Preserve `HYGD-CEXT-1.1` as a closed negative source-only qualification.
2. Close the manual-source scalar-only lane; do not train a glaucoma classifier from this manifest.
3. Do not ask for more HYGD boundary clicks merely to reconfirm the same scale factor.
4. Treat dataset separability as the load-bearing blocker.
5. If the project owner approves a new lane, test one pre-frozen full-frame foundation representation across the same three source LOSO folds before any new target access.

## Model-Selection Adversarial Check

DINOv3 was checked after the v2.0 freeze to ensure that DINOv2 was not chosen merely from stale familiarity. DINOv3 is newer and its official release claims stronger broad visual features, but its code and weights use a custom DINOv3 License rather than Apache-2.0. The official weight path also requires an access request and e-mail delivery of model URLs. The local environment has compatible `timm`, but that does not remove the weight-access and license-acceptance gates.

Decision: keep DINOv2-S/14 as the single first baseline because it is a small 21M-parameter official model under the repository's Apache-2.0 license and asks the narrowest scientific question with the smallest approval surface. DINOv3 is not a fallback in `HYGD-CEXT-2.0`; testing it would require a new protocol and separate license/access approval.

Official comparison sources:

- https://github.com/facebookresearch/dinov2/blob/main/MODEL_CARD.md
- https://github.com/facebookresearch/dinov2/blob/main/LICENSE
- https://github.com/facebookresearch/dinov3
- https://github.com/facebookresearch/dinov3/blob/main/LICENSE.md

## Evidence

- `validation/HYGD_BOUNDARY_ADJUDICATION_V1.md`
- `data/hygd_boundary_annotations_v1.csv`
- `results/hygd_boundary_adjudication_v1.json`
- `results/hygd_boundary_adjudication_v1_details.csv`
- `results/hygd_boundary_adjudication_v1_overlay.json`
- `results/hygd_scale_calibration_feasibility.json`
- `validation/make_hygd_boundary_adjudication_overlays.py`
- `validation/evaluate_hygd_scale_calibration_feasibility.py`
