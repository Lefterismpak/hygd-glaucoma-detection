# GlaucoGen — Historical External Validation Protocol (pre-registered)

> **Written 2026-07-05, BEFORE any external-dataset probability was computed.** The point of pre-registering is credibility: every analysis decision below is fixed *before* seeing the numbers, so the eventual "domain-shift drop" cannot be a story fitted after the fact. Deviations, if any, will be logged in a dated "Deviations" section at the bottom — never silently.

> **Retrospective status correction (2026-08-31):** This file preserves the original historical protocol; it is not the current claim hierarchy. Its `0.988 +/- 0.008` HYGD CV baseline is superseded and non-canonical, and later target visibility/resources made the recovery work adaptive rather than prospectively untouched. Historical external intervals used eye/image-row bootstrap rather than patient-cluster bootstrap; the calibration split was grouped but not outcome-stratified; RIM-ONE subject independence is `needs-proof`. PAPILA Figshare v2 reports GPL 3.0+; the earlier permissive-license attribution was wrong. RIM-ONE permits research/education use, prohibits copying/redistribution, and instructs users not to add other databases for training or tuning; compatibility of the mixed-source experiments therefore remains `needs-proof`. Use [`../INTERNAL_EVALUATION_REPAIR.md`](../INTERNAL_EVALUATION_REPAIR.md), [`../results/repaired_internal_evaluation_summary.json`](../results/repaired_internal_evaluation_summary.json), and [`../HYGD_FAILURE_FIRST_RESEARCH_BRIEF.md`](../HYGD_FAILURE_FIRST_RESEARCH_BRIEF.md) for the adjudicated evidence.

## Historical objective

Measure, honestly, how the published HYGD glaucoma model (using the **historical, superseded** patient-level CV AUROC 0.988 +/- 0.008 as the then-current in-distribution baseline) performs on independent public fundus datasets, and how much of any drop appears recoverable by recalibration/threshold-tuning versus loss of discrimination.

## Historical external datasets

| Dataset | License | Use | Redistribution |
|---|---|---|---|
| **PAPILA** | Figshare v2: GPL 3.0+ | historical primary external test | images/crops not redistributed; derived redistribution compatibility `needs-proof` |
| **RIM-ONE DL** | research/education only; copying/redistribution prohibited | historical secondary external test | data git-ignored; mixed-source training/tuning compatibility `needs-proof` |

- ORIGA is **excluded** (no clean public license). REFUGE, if ever added, results/figures only — never redistributed images.
- Glaucoma-"suspect" images: **primary analysis excludes suspects** (binary glaucoma vs healthy). A secondary sensitivity analysis may include suspects-as-positive, reported separately and labelled as such.

## Fixed decisions as written in 2026-07-05

The numbered items below are retained verbatim as the intended protocol. The
dated deviations section records where execution did not satisfy them.

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

## Historical deliverables and current status

- `validation/predict.py` and `validation/verify_parity.py`: present; historical parity claim retained as provenance.
- `validation/eval_external.py`: present; historical outputs only partially satisfied the patient-level, stratification, and provenance contract. The current code writes to a private patient-aware namespace. It emits no canonical subject-cluster interval: an operator-supplied mapping must be hash-bound, and its biological correctness remains `needs-proof` unless independently sourced and verified.
- `VALIDATION.md`: present, now adjudicated as an adaptive chronology.
- `TRIPOD-AI-checklist.md`: not present in this public repository.
- README external evidence section: replaced by a failure-first claim hierarchy; no PI-outreach external-validation claim is made.

## Prior art to cite (not re-derive)

- TRIPOD+AI reporting guideline (BMJ 2024) — the checklist basis.
- Van Calster et al. 2019 (calibration: the Achilles heel of predictive analytics).
- Guo et al. 2017 (temperature scaling / modern-network calibration).
- RETFound-vs-CNN glaucoma external comparison (Ophthalmology Science 2025) — situate results against it.
- The patient-vs-image data-leakage failure mode (cite an established reference) — motivates the patient-level insistence.

## needs-proof before any new run
- PAPILA access is documented on Figshare v2 as GPL 3.0+; legal compatibility for redistribution of derived crops remains `needs-proof`.
- RIM-ONE DL access terms permit research/education use, prohibit copy/redistribution, and request original partitions without adding other databases for training/tuning. Written clarification is required before any new mixed-source run.
- A verified RIM-ONE subject mapping is required; image stems are not accepted as proof of subject independence.
- External glaucoma base rates (used only to frame the calibration hypothesis, not as an input).

---

### Deviations from protocol

- **2026-07-14:** Target AUROC was visible during iterative development, and target-domain anatomical resources affected preprocessing. The recovery chronology is therefore adaptive development evidence, not untouched external validation.
- **2026-08-30:** The original HYGD five-fold baseline was reclassified as historical and superseded after identifying test-set configuration selection, same-fold checkpoint/scoring reuse, and exact duplicates spanning supplied patient IDs. The duplicate-aware inner/outer evaluator is the preferred internal path.
- **2026-08-31:** Historical PAPILA/RIM-ONE intervals were adjudicated as eye/image-row bootstrap rather than patient-cluster intervals. Historical external calibration used a grouped, non-stratified split; fine-tuning used a row split; RIM-ONE subject independence was not verified. The affected outputs remain numeric history with the literal `partial_with_deviations` or `needs-proof` labels, not canonical external evidence.
- **2026-08-31:** PAPILA's public record was corrected from the earlier permissive-license attribution to Figshare v2 GPL 3.0+. RIM-ONE's official research/education, non-redistribution, and no-database-mixing instructions were added. Mixed-source compatibility remains `needs-proof`; no data or derived images are published here.
