# Validation & Honesty Statement

> **Status correction (2026-09-04):** The historical `0.988 +/- 0.008` CV is superseded because the configuration was chosen after a development-test comparison and each fold was reused for checkpoint selection and scoring. The preferred internal estimate below is a v5 read-only aggregate reanalysis of frozen predictions from a private model run asserted to date from 2026-07-11; no contemporaneous public timestamp attests that run and no end-to-end v5 model run is claimed. External-recovery results remain adaptive development evidence, not untouched external validation. Later locked source-only qualifications did not establish transportability. See the [aggregate receipt](results/repaired_internal_evaluation_summary.json) and [HYGD_FAILURE_FIRST_RESEARCH_BRIEF.md](HYGD_FAILURE_FIRST_RESEARCH_BRIEF.md).

This document separates the evidence that remains usable from the historical artifacts that are retained only for provenance. A high AUC is easy to inflate; the point is to preserve the failures and limitations alongside the attractive numbers.

Historical numbers are read from committed development artifacts. The repaired result is documented in [INTERNAL_EVALUATION_REPAIR.md](INTERNAL_EVALUATION_REPAIR.md). Its row-level audit/OOF outputs are deliberately git-ignored and are not published; the public [aggregate receipt](results/repaired_internal_evaluation_summary.json) contains only metrics, limitations, and cryptographic commitments to the private source bytes.

## What was tested

- **Dataset:** Hillel Yaffe Glaucoma Dataset (HYGD) — 747 colour fundus images associated with 288 supplied patient IDs, with dataset-author reference labels based on a reported ophthalmic work-up (visual acuity, IOP, OCT, visual fields, ≥1 year follow-up), not image review alone.
- **Historical model comparison:** ResNet-18 pretrained on ImageNet, with three configurations compared — a frozen-backbone baseline, a frozen-backbone augmented model, and a partially fine-tuned (`layer4` + head) augmented model with a class-weighted loss for the 73%/27% imbalance.
- **Preferred internal metrics:** duplicate-aware evaluation-group AUROC, sensitivity, specificity, confusion matrix, and group-cluster-bootstrap 95% confidence intervals from outer-fold OOF predictions.
- **Historical development metrics:** one test split, image-row bootstrap, threshold sweep, and the invalidated five-fold robustness estimate. They document model development; they are not a final untouched test.

## Internal evaluation hierarchy

The original work grouped supplied patient IDs, but a later audit found three residual threats: configurations were compared on the development test split; the old five-fold loop selected a checkpoint and reported performance on the same fold; and exact images appeared under different supplied patient IDs.

The repaired protocol hashes every image, counts each exact hash once, links patient IDs that share a hash into one independent evaluation group, fixes the model recipe before outer evaluation, uses a separate inner group split for checkpoint and threshold selection, and scores each outer group once.

| Metric | Value |
|---|---|
| **Mean outer-fold group AUROC (preferred internal)** | **0.9908, fold-stratified group-bootstrap 95% CI [0.9790, 0.9990]** |
| Pooled group OOF AUROC (continuity only) | 0.9904, group-cluster 95% CI [0.9797, 0.9980] |
| Group sensitivity / specificity | 0.9563 / 0.9700 (TN 97, FP 3, FN 8, TP 175) |
| Image AUROC (secondary descriptive) | 0.9837, group-cluster 95% CI [0.9731, 0.9925] |
| Historical five-fold CV | 0.988 +/- 0.008 — **superseded and non-canonical** |
| Historical best single-split AUROC | 0.976 — development comparison, not final test |

The primary discrimination estimand averages the five group-level outer-fold AUROCs equally and therefore does not compare raw score scales from separately fitted fold models. Its percentile interval resamples whole linked groups within each fixed fold using seed `20269714`, stable group ordering, and 5,000/5,000 valid replicates. Sensitivity/specificity pool OOF binary decisions made with thresholds recorded in the legacy result as inner-validation-selected; selection was not independently rerun. The intervals condition on the frozen OOF predictions and recorded fold/decision choices; they exclude uncertainty from retraining, checkpoint and threshold selection, recipe selection, transportability, and deployment. The pooled 0.9904 group AUROC and image-level AUROC use global group-cluster intervals over fixed pooled scores and remain secondary because both compare score scales from separately fitted fold models.

## What this repo does NOT claim

- **The in-distribution numbers above are single-dataset** — one hospital (Hillel Yaffe Medical Center), one camera (TOPCON DRI OCT Triton). They do not establish transportability.
- **It is not a clinical device** and must never be used for real diagnostic decisions.
- **Calibration, clinical utility, and deployment performance are unproved.** Fold-specific inner-validation thresholds ranged from 0.483 to 0.948; no fixed operating point is recommended.
- **The historical 44-ID split and its image-row intervals are development artifacts**, not an untouched test or the uncertainty source for the preferred result.
- **Historical Grad-CAM is not validation evidence.** The post hoc saliency review had no predeclared localization endpoint or independent expert adjudication. It cannot establish causal feature use, correct localization, clinical reasoning, or absence of shortcut learning; its row-level panels and interpretations were removed from the public evidence package.

## The error analysis is honest too

The selected single-split development model made 5 image-level errors. Their mean FundusQ-Net score was 5.48 versus 6.04 among correct predictions; an exploratory one-sided Mann-Whitney comparison yielded p = 0.45. The analysis was post hoc, underpowered, image-level despite repeated-patient structure, and unadjusted for selection or multiplicity. It provides no evidence for or against a quality-error association and does not justify a quality threshold or causal failure taxonomy.

## Reproduce it (from a fresh clone)

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
# download HYGD into data/raw/ (see README §3), then:
python -m unittest discover -s validation -p 'test_*.py' -v
python validation/internal_evaluation_repair.py --audit-only
python validation/internal_evaluation_repair.py
python validation/reanalyze_frozen_oof.py --describe-contract
python run_v2_experiments.py --describe-policy
```

The `2026-09-04-v5` integrity patch publishes code, synthetic guardrail tests, and an [aggregate-only receipt](results/repaired_internal_evaluation_summary.json). It does not retrain the legacy models: a private read-only v5 reanalysis recomputed the scale-robust 0.9908 estimand from bound frozen OOF predictions. The receipt binds the private source manifest, canonical OOF identity, canonical group-to-fold assignment, implementation, script, scientific command, and runtime while publishing no private paths or row-level records. Complete terminal bundles must reproduce the canonical fold assignment and satisfy a closed, artifact-bound audit schema. Generated repair audits, fold manifests, OOF predictions, and full result JSON remain local and git-ignored. Audit-only and partial smoke runs use distinct noncanonical namespaces, and every bundle refuses to overwrite existing targets. Only the locked documented defaults can emit a complete preferred-result schema; changed settings require a proper-subset smoke run. To reproduce only the historical single-split comparison, an explicit `--run-historical-comparison` acknowledgement is required; that route never emits a CV estimate. This is not an end-to-end v5 training receipt.

## Historical external stress tests and adaptive recovery

The naive model failed zero-shot on PAPILA and RIM-ONE. Later anatomy-assisted development recovered benchmark AUROC, but the recipe evolved after target results were visible and target-domain anatomical resources affected preprocessing. These are adaptive benchmark results, not prospectively untouched external validation or proof of transportability.

| Stage | Held-out PAPILA AUROC | Notes |
|---|---|---|
| HYGD historical five-fold CV | 0.988 +/- 0.008 | superseded, non-canonical development artifact |
| **PAPILA, zero-shot** | **0.51** (historical eye-row interval 0.44–0.58) | noncanonical uncertainty; adaptive chronology |
| RIM-ONE DL, zero-shot | 0.61 (historical image-row interval 0.55–0.66) | subject independence `needs-proof` |
| Single-source naive fine-tune | 0.68 | doesn't transfer |
| + disc-crop + colour-norm + multi-source + SWA | 0.79 (historical eye-row interval 0.74–0.84) | adaptive; mixed-source terms `needs-proof` |
| + test-time augmentation | 0.83 (historical eye-row interval 0.77–0.87) | adaptive; interval noncanonical |
| + VCDR multi-task head | 0.857 +/- 0.019 (five-run sample mean +/- SD, not a CI; best historical eye-row interval 0.82–0.91) | adaptive, not confirmatory |

The legacy recovery figure was removed because its standalone headline overstated the current evidence boundary. The corrected adaptive chronology remains in self-adjudicating text and aggregate JSON.

> **Symmetric adaptive check.** Running the same disease-label-held-out recipe in reverse reached RIM-ONE AUROC 0.915 +/- 0.012 across five runs; this is a descriptive sample mean +/- SD, not a confidence interval. It is mechanism-supporting development evidence, not a reset of the research history: target metrics were visible, RIM-ONE subject independence is `needs-proof`, and mixed-dataset use compatibility requires clarification.

**Zero-shot, the model saturates** — it calls almost everything glaucoma, healthy and glaucoma probabilities indistinguishable — and a naive cross-dataset fine-tune fits each source (val 0.92–0.96) but does not transfer. That pattern is consistent with source-specific shortcut reliance and is insufficient evidence of source-invariant disease features; it does not prove that no universal signal exists.

**Adaptive recovery recipe:** disc-centred, disc-size-standardized cropping; colour/illumination normalization; multi-source training; heavy augmentation; SWA; test-time augmentation; and a VCDR auxiliary head. Disease labels from the nominal target were excluded from the training loss and source-validation checkpoint criterion, but target AUROC was displayed during development and target anatomical resources entered preprocessing. The historical, superseded 0.988 internal CV and these external adaptive values must not be combined into a deployment or transportability claim.

Later locked source-only qualifications failed to establish transportability: HYGD-CEXT-1.1 reached equal-source mean AUROC 0.6227 [0.5822, 0.6632] under its fallback and was not promoted; HYGD-CEXT-2.0 reached mean AUROC 0.7105 [0.6746, 0.7423] but failed its decisive dataset-origin gate with accuracy 0.9994. The project supports in-distribution discrimination and a documented failure/recovery chronology, not calibration, clinical utility, deployment, or cross-site readiness.

## Attribution

Dataset: Abramovich O, Pizem H, Fhima J, et al. *Hillel Yaffe Glaucoma Dataset (HYGD)*, PhysioNet (Open Data Commons Attribution License v1.0); and *GONet: A Generalizable Deep Learning Model for Glaucoma Detection*, IEEE Transactions on Biomedical Engineering. 2026;73(1):32–39. doi:10.1109/TBME.2025.3576688. Quality scores: Abramovich O, et al. *FundusQ-Net*, Comput Methods Programs Biomed 2023;239:107522.
