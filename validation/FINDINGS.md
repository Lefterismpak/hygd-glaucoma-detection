# GlaucoGen — External Validation Findings (PAPILA, zero-shot)

> Run 2026-07-05. This is the honest external-validation result. It is on the
> `external-validation` branch and **not yet published** — how/whether to surface
> it publicly is a framing decision for the repo owner.

## Headline

The HYGD model that scores **AUROC 0.988 in patient-level cross-validation** does
**not transfer to PAPILA**: zero-shot **AUROC ≈ 0.51 (95% CI 0.44–0.58) — chance.**
It collapses to predicting *glaucoma for almost everything* (96.7% of PAPILA eyes,
mean P 0.93), with healthy and glaucoma probabilities essentially identical
(healthy 0.935 vs glaucoma 0.901 — no separation). See `external_prediction_collapse.png`.

| Setting | AUROC | Notes |
|---|---|---|
| HYGD, patient-level 5-fold CV (in-distribution) | 0.988 ± 0.008 | reported result |
| HYGD test via `predict.py` (parity check) | 0.976 | pipeline is correct |
| **PAPILA, zero-shot (as-trained preprocessing)** | **0.508** | CI [0.44, 0.58] |
| PAPILA, zero-shot + field-of-view standardization | 0.58 | small bump, still near-chance |

## This is a real failure, not a bug — ruled out:

- **Pipeline correct:** the same `predict.py` gives AUROC 0.976 on the HYGD test set (parity gate already PASS at 2.8e-06).
- **Labels correct:** 420 eyes = 333 healthy + 87 glaucoma (suspects excluded), matches PAPILA's published composition exactly.
- **Not just framing:** HYGD is square (1893²), PAPILA landscape (2576×1934). Standardizing the field of view (center-square crop) only moves AUROC 0.51 → 0.58 — the failure is deeper than aspect ratio.

## Interpretation (honest)

The model learned features specific to HYGD (single hospital, single camera — TOPCON DRI OCT Triton — and a 73% glaucoma base rate) that do not exist or differ on PAPILA. It is saturated toward "glaucoma," consistent with the high training prevalence plus dataset-specific shortcuts. **Recalibration** (temperature/Platt) fixes the calibration dramatically (ECE 0.74 → 0.08 with Platt) but — as stated in the protocol — **cannot restore discrimination**: recalibrating a chance-level ranking is still chance.

This is exactly what external validation exists to reveal, and it is the norm, not the exception, for single-dataset fundus models. It is not a flaw in how the model was *built* (the in-distribution work is clean and honest); it is the true, measured limit of what one small single-hospital dataset can produce.

## Why this is a *strong* outcome for the portfolio

A student who builds a model, externally validates it, finds it collapses, and reports the collapse transparently — with the mechanism characterized — demonstrates exactly the maturity a skeptical DACH ophthalmology PI is looking for. It is a better story than a modest, cherry-picked drop. It also directly motivates real next work.

## Next experiments (post-freeze, if pursued)

1. **Disc-centred crop + light fine-tune** (Tier-3): does a small amount of PAPILA/RIM-ONE fine-tuning on disc-standardized input recover transferable signal? (train RIM-ONE → test PAPILA, and vice-versa; never fine-tune and test on the same set.)
2. **Shortcut-learning analysis:** Grad-CAM on PAPILA to see what the saturated model attends to — likely border/field artifacts, not the disc.
3. Add **RIM-ONE DL** as the second external set to confirm the pattern generalizes across datasets.

## Files
- `results/external_papila.json` — full metrics (zero-shot + recalibration).
- `validation/external_prediction_collapse.png` — the prediction-distribution figure.
- `validation/make_papila_labels.py`, `validation/eval_external.py` — reproducible.
- PAPILA images are git-ignored (GPL-3.0+, not redistributed); the label CSV is derived metadata.

---

## Update — cross-dataset fine-tune (Tier-3): does a short fine-tune fix transfer? NO.

Ran the honest test: fine-tune the HYGD model (layer4+fc, class-weighted, early-stopped) on ONE external dataset, evaluate on the OTHER (fully held out, never seen in fine-tuning). RIM-ONE zero-shot also confirmed the collapse generalizes (AUROC 0.606, still saturates — sens 1.0 / spec 0.0).

| Fine-tune direction | target zero-shot | target after fine-tune | Δ | source-val (fit) |
|---|---|---|---|---|
| RIM-ONE → PAPILA | 0.508 | **0.683** | +0.18 | 0.961 |
| PAPILA → RIM-ONE | 0.606 | **0.467** | −0.14 | 0.925 |

**Verdict: fine-tuning on one dataset does NOT produce a model that transfers to another.** The best case (RIM-ONE→PAPILA) recovers to only 0.68 — still weak; the other direction actually *degrades* below chance (0.47). Crucially, the model fits each source dataset excellently (val AUROC 0.92–0.96) — so glaucoma IS learnable in each set individually. The failure is specifically **transfer**: each dataset teaches the model dataset-specific shortcuts (camera, field, processing), not universal glaucoma features.

## Complete, honest conclusion

A single small single-hospital dataset — even with fine-tuning on a second external set — is **not enough to build a glaucoma model that generalizes across fundus datasets.** In-distribution AUROC 0.988 is real but does not survive contact with other cameras/populations. This is consistent with the literature and is precisely why the field moved to foundation models (e.g. RETFound) pretrained on very large multi-dataset corpora. It is not a flaw in how this model was built; it is the honest ceiling of the data.
