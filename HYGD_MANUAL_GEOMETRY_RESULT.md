# HYGD Manual Geometry Result

**Gate:** `HYGD-MANUAL-GEOM-1`

**Status:** closed - geometry metrics failed

**Classifier authorized:** no

**Scope:** exploratory HYGD source-domain geometry only; this does not reopen or alter frozen `HYGD-CEXT-1.1`.

## Answer

In a post-lock exploratory 36-image manual-reference audit, the predeclared center-error criterion passed, whereas the diameter and combined-geometry criteria failed. The original diameter labels were algorithmically fixed at 25% of image size. A scalar estimated and evaluated on the same subset improved descriptive geometry metrics but did not pass the subsequent dataset-origin gate. This is resubstitution and mechanism-supporting negative evidence only—not independent localization validation, an anatomical ground-truth study, or a qualified preprocessing repair. No classifier was authorized.

## Locked Evaluation

All 36 preselected, linked-evaluation-group-disjoint images were accepted under the blinded five-click annotation contract. Here, disjointness is defined by supplied HYGD IDs plus exact-hash links; it is not independent proof of biological-subject identity. The locked evaluator verified the manifest, prediction, script, lock, and annotation hashes before computing the predeclared metrics.

| Metric | Result | Gate | Decision |
|---|---:|---:|---|
| Prediction coverage | 0.9722 (35/36) | >=0.95 | pass |
| Center pass rate | 1.0000 | >=0.90 | pass |
| Diameter pass rate | 0.6286 | >=0.85 | fail |
| Combined pass rate | 0.6111 | >=0.90 | fail |
| Median center error | 0.0778 reference diameters | <=0.50 | pass |
| Median diameter ratio | 1.7583 | 0.75-1.30 | fail |

Twelve predictions were above the per-image 1.80 diameter-ratio ceiling, one was below 0.50, and one was missing. These aggregate outcomes are retained; no image was replaced.

## Deterministic Overlay Review

The 36-tile overlay shows the human center/boundary in green and the frozen prediction in orange. It confirms that the numeric failure is not a coordinate-transform or rendering artifact:

- orange and green centers generally coincide;
- the orange diameter is systematically larger across the set;
- one case is an opposite-direction outlier in the private recorded evidence;
- one case remains marked as missing in the private overlay;
- no post-lock annotation, threshold, sample, or prediction was changed.

The row-level overlay is private and is not included in the public repository.

SHA-256: `63e76f3cc79fbef212b45c802158189235575f7ac7858f55f95bc337d53c53be`

## Root Cause

The earlier 100 `data/hygd_manual_disc_centers.csv` records were center clicks with a synthetic diameter equal to exactly `0.25 * max(image_width, image_height)` in 100/100 images. The localizer was trained against circular masks built from those values.

The new anatomical annotations narrow the old `needs-proof` diameter question:

- synthetic training diameter fraction: exactly 0.2500;
- frozen prediction median fraction: 0.2542;
- human reference-diameter median fraction: 0.1427 (IQR 0.1346-0.1502).

The values are consistent with reproduction of the fixed scale target and support a label-design-failure hypothesis for scale. They are not causal proof and do not show that the center localizer failed.

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
5. **Historical next-step record:** this proposed full-frame foundation-representation lane was subsequently executed as HYGD-CEXT-2.0 and is now closed after the dataset-origin gate failed; it is not a current authorization.

## Model-Selection Adversarial Check

DINOv3 was checked after the v2.0 freeze to ensure that DINOv2 was not chosen merely from stale familiarity. DINOv3 is newer and its official release claims stronger broad visual features, but its code and weights use a custom DINOv3 License rather than Apache-2.0. The official weight path also requires an access request and e-mail delivery of model URLs. The local environment has compatible `timm`, but that does not remove the weight-access and license-acceptance gates.

Decision: keep DINOv2-S/14 as the single first baseline because it is a small 21M-parameter official model under the repository's Apache-2.0 license and asks the narrowest scientific question with the smallest approval surface. DINOv3 is not a fallback in `HYGD-CEXT-2.0`; testing it would require a new protocol and separate license/access approval.

Official comparison sources:

- https://github.com/facebookresearch/dinov2/blob/main/MODEL_CARD.md
- https://github.com/facebookresearch/dinov2/blob/main/LICENSE
- https://github.com/facebookresearch/dinov3
- https://github.com/facebookresearch/dinov3/blob/main/LICENSE.md

## Evidence

The aggregate statements above are bound to a private, read-only source set. Each
record has exactly `{"role": <table path>, "sha256": <lowercase digest>,
"size_bytes": <integer>}`. Records are sorted by the UTF-8 `role` value and
serialized with Python `json.dumps(records, ensure_ascii=True,
separators=(",", ":"), sort_keys=True)`. The manifest is SHA-256 over the ASCII
domain separator `HYGD_MANUAL_GEOMETRY_PRIVATE_SOURCES_V1`, a newline, those
canonical JSON bytes, and a final newline.

**Private-source manifest SHA-256:** `8022b0ce2723b0f842bf8a68e40ab52d7dc9664015c794209cf585ec8a7d765b`

| Private source role | SHA-256 | Bytes |
|---|---|---:|
| `validation/HYGD_BOUNDARY_ADJUDICATION_V1.md` | `03531587052c2026f3c90c2e4f8a3dd719547983f561d1e82a3a124aadb2b6fb` | 4,148 |
| `data/hygd_boundary_annotations_v1.csv` | `22d191ddf3c0d5e94bf9bc4e4904741ede6a553d7f289e181a592ea0fcf66220` | 23,021 |
| `results/hygd_boundary_adjudication_v1.json` | `1f53bb1963a723c4a9e99e22c95f71fc50e61a8b4cc5002acf5bc4ea1769413a` | 1,734 |
| `results/hygd_boundary_adjudication_v1_details.csv` | `8115302126e36f4b3b3728e5286cc877165c8785c8a10ef13b41636ced524af0` | 17,862 |
| `results/hygd_boundary_adjudication_v1_overlay.json` | `85be53f99bd12a809b361b6f3ac4da607f924d98d59c3df73f9d4fc14dd59c13` | 7,897 |
| `results/hygd_boundary_adjudication_v1_overlay.png` | `63e76f3cc79fbef212b45c802158189235575f7ac7858f55f95bc337d53c53be` | 3,704,620 |
| `results/hygd_scale_calibration_feasibility.json` | `944dbb0f156d34d63457cdf9bb4c70ef1afee5a655027a12fedb0a6ea35f43cb` | 1,966 |
| `validation/make_hygd_boundary_adjudication_overlays.py` | `0c431b2c2fe8497ff05fa77a4f31ab7364f6bf346b5ae432c986d6158e326961` | 8,236 |
| `validation/evaluate_hygd_scale_calibration_feasibility.py` | `ab1920c531431b675937b0e513274f0922de0307960a9e80b2fcc885145ab87d` | 5,927 |

The hashes permit private workspace-level provenance checking without publishing
row-level annotations, identifiers, images, overlays, or predictions. The lock
chronology and mapping from these private, unpublished files to the public
aggregates remain externally `needs-proof`; the files are not distributable
evidence from this repository.

- `validation/HYGD_BOUNDARY_ADJUDICATION_V1.md`
- `data/hygd_boundary_annotations_v1.csv`
- `results/hygd_boundary_adjudication_v1.json`
- `results/hygd_boundary_adjudication_v1_details.csv`
- `results/hygd_boundary_adjudication_v1_overlay.json`
- `results/hygd_scale_calibration_feasibility.json`
- `validation/make_hygd_boundary_adjudication_overlays.py`
- `validation/evaluate_hygd_scale_calibration_feasibility.py`
