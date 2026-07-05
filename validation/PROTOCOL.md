# GlaucoGen — External Validation Protocol (pre-registered)

> **Written 2026-07-05, BEFORE any external-dataset probability was computed.** The point of pre-registering is credibility: every analysis decision below is fixed *before* seeing the numbers, so the eventual "domain-shift drop" cannot be a story fitted after the fact. Deviations, if any, will be logged in a dated "Deviations" section at the bottom — never silently.

## Objective

Measure, honestly, how the published HYGD glaucoma model (`finetune_layer4_aug`, patient-level CV AUROC 0.988 ± 0.008 in-distribution) generalizes to **independent** public fundus datasets, and how much of any drop is recoverable by recalibration/threshold-tuning vs is a true loss of discrimination.

## Datasets (external, held-out — never used in HYGD training)

| Dataset | License | Use | Redistribution |
|---|---|---|---|
| **PAPILA** | CC BY 4.0 | primary external test | label CSV + derived figures committable |
| **RIM-ONE DL** | research/education, no redistribution | secondary external test | ship a regeneration script only; data git-ignored |

- ORIGA is **excluded** (no clean public license). REFUGE, if ever added, results/figures only — never redistributed images.
- Glaucoma-"suspect" images: **primary analysis excludes suspects** (binary glaucoma vs healthy). A secondary sensitivity analysis may include suspects-as-positive, reported separately and labelled as such.

## Fixed pre-registered decisions

1. **Model is frozen.** The exact committed checkpoint is used as-is for the zero-shot analysis. No retraining before the zero-shot numbers are reported.
2. **Inference preprocessing is `validation/predict.py`** — proven to match the HYGD eval pipeline to < 1e-4 by `validation/verify_parity.py` (PASS, max Δ = 2.8e-06 on 2026-07-05). Any external inference uses this path only.
3. **Everything is patient-level.** Metrics, splits, and any calibration split group by patient ID. No image-level pooling that could mix a patient across roles.
4. **Primary metric = AUROC** with a 2000-sample bootstrap 95% CI. Secondary: AUPRC (base rates differ across datasets), sensitivity/specificity **at the pre-set 0.40 screening threshold** carried over from HYGD, and calibration.
5. **Calibration is a first-class outcome, not an afterthought.** Report a reliability diagram, Expected Calibration Error (ECE), calibration slope/intercept, and Brier score for every dataset. Hypothesis (stated in advance): the HYGD training base rate (73% glaucoma) is far above the external base rate, so raw probabilities will be **mis-calibrated even if discrimination holds** — i.e. we expect the visible "failure" to be calibration, not necessarily AUROC.
6. **Recalibration is evaluated honestly.** Temperature scaling and Platt scaling are fit on a **patient-stratified 30% calibration split** of each external set and evaluated on the remaining 70%. It will be stated explicitly in the writeup that **AUROC is invariant to any monotone recalibration** — so recalibration's win is calibration/threshold usefulness, not discrimination.
7. **Threshold re-selection**: choose the operating threshold on the calibration split targeting **sensitivity ≥ 0.90**, then report the resulting specificity on the held-out eval split. Never select the threshold on the same data used to report it.
8. **Optional Tier-3 light fine-tune** (only if time permits, reported separately): disc-crop + short `layer4`+`fc` fine-tune, **cross-dataset** (train on RIM-ONE → test on PAPILA and vice-versa). Never fine-tune and test on the same dataset. ≤1 short run per direction.

## Correctness gates (a broken gate voids the affected result)

- (a) Inference preprocessing exactly matches training/eval — **enforced by `verify_parity.py`** (already PASS).
- (b) Patient-level grouping everywhere; assert zero patient overlap before every split.
- (c) Never fine-tune and evaluate on the same dataset.
- (d) State that AUROC is unchanged by monotone recalibration; do not present recalibration as a discrimination gain.

## Deliverables

- `validation/predict.py` (done), `validation/verify_parity.py` (done, PASS).
- `validation/eval_external.py` — zero-shot + calibration + threshold, one function per dataset (post-freeze).
- `VALIDATION.md` external-validation section: one summary TABLE + reliability/ROC FIGURES.
- `TRIPOD-AI-checklist.md` filled for the external-validation study.
- README "External Validation on 2 independent datasets" section — the concrete PI-outreach artifact.

## Prior art to cite (not re-derive)

- TRIPOD+AI reporting guideline (BMJ 2024) — the checklist basis.
- Van Calster et al. 2019 (calibration: the Achilles heel of predictive analytics).
- Guo et al. 2017 (temperature scaling / modern-network calibration).
- RETFound-vs-CNN glaucoma external comparison (Ophthalmology Science 2025) — situate results against it.
- The patient-vs-image data-leakage failure mode (cite an established reference) — motivates the patient-level insistence.

## needs-proof before running (verify at execution time)
- PAPILA download URL (Figshare mirror) and its image-file license line (reported as unusually GPL-3.0+ for the images — confirm before any redistribution of derived crops).
- RIM-ONE DL download (github.com/miag-ull/rim-one-dl; bit.ly redirect — confirm it resolves) and its exact split/augmentation terms.
- External glaucoma base rates (used only to frame the calibration hypothesis, not as an input).

---

### Deviations from protocol
*(none yet — append dated entries here if any decision above changes, with the reason.)*
