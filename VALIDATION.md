# Validation & Honesty Statement

This document explains **why the headline numbers in this repo are trustworthy, what they do *not* claim, and exactly what would need to happen next** to make them clinically meaningful. It exists because a high AUC is easy to report and easy to inflate — the point of this project is to show the reporting is honest, not just high.

All numbers below are read directly from the committed result files (`results/v2_comparison.json`, `results/cv_results.json`, `results/threshold_sweep.json`, `results/error_analysis.json`), not from memory.

## What was tested

- **Dataset:** Hillel Yaffe Glaucoma Dataset (HYGD) — 747 colour fundus images from 288 patients, with **gold-standard labels** derived from a full ophthalmic work-up (visual acuity, IOP, OCT, visual fields, ≥1 year follow-up), not from image review alone.
- **Model:** ResNet-18 pretrained on ImageNet, with two configurations compared — a frozen-backbone baseline and a partially fine-tuned (`layer4` + head) model with light augmentation and a class-weighted loss for the 73%/27% imbalance.
- **Reported metrics:** AUROC, sensitivity, specificity, a confusion matrix, bootstrap 95% confidence intervals, a 5-fold **patient-level** cross-validation, a decision-threshold sweep, and a per-error clinical analysis.

## Why the AUC is honest (the part most student projects get wrong)

**The split is at the patient level, not the image level.** HYGD contains ~2.6 images per patient (up to 14 for one patient). If images from the same patient land in both the training and test sets — the default when you split naively — the model can memorise a patient rather than learn the disease, and the reported AUC is silently inflated. This is a well-documented failure mode in medical imaging ML.

Every split in this repo uses `sklearn`'s `GroupShuffleSplit` / `GroupKFold` **grouped by patient ID**, and the code asserts zero patient overlap between splits before training. The headline result is therefore a 5-fold **patient-level** cross-validation, which measures stability across different patient partitions rather than the luck of one split:

| Metric | Value |
|---|---|
| **5-fold patient-level CV AUROC** | **0.988 ± 0.008** (folds: 0.995, 0.995, 0.980, 0.978, 0.991) |
| Best single-split model (fine-tuned `layer4` + aug), test AUROC | 0.976, 95% CI [0.943, 0.998] |
| — sensitivity @0.5 | 0.954, 95% CI [0.90, 1.00] |
| — specificity @0.5 | 0.941, 95% CI [0.853, 1.00] |
| Frozen-backbone baseline (v1), test AUROC | 0.952 |

Confidence intervals are reported precisely **because the test set is small** (99 images / 44 held-out patients) — the CIs are wide, and that is stated rather than hidden.

## What this repo does NOT claim

- **It is not externally validated.** Every number above comes from a single dataset, a single hospital (Hillel Yaffe Medical Center), and a single camera (TOPCON DRI OCT Triton). Performance on a different population/device is unknown and would almost certainly be lower.
- **It is not a clinical device** and must never be used for real diagnostic decisions.
- **The metrics are indicative, not precise** — a 44-patient test set cannot pin down performance tightly, which is exactly why the CIs are wide.
- **The threshold is a starting point, not a deployment setting.** A screening tool should minimise missed disease, so the operating point matters more than the default 0.5. On this test set, a **0.40 threshold** catches 63/65 glaucoma cases (2 missed vs 3 at 0.5) for one extra false alarm — but a real deployment threshold must be re-derived on a larger, external, calibrated set.
- **Grad-CAM is a sanity check, not proof of clinical reasoning.** It usefully shows the model attends to the optic disc on most correct cases, but saliency maps are known to be imperfect explanations — they are used here to catch gross failures, not to claim the model "reasons like a clinician."

## The error analysis is honest too

The best model makes 5 errors (3 missed glaucoma, 2 false alarms). These are **not** simply explained by poor image quality (Mann–Whitney U on FundusQ score, p = 0.45). They split into two failure modes — *localization* (the model's attention was off the disc, including one very-low-quality image) and *interpretation* (attention on the disc, wrong call). Full write-up in `notebooks/04_explainability.ipynb`. The disc-level descriptions there are framed as observational hypotheses for model behaviour, **not diagnoses.**

## Reproduce it (from a fresh clone)

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
# download HYGD into data/raw/ (see README §3), then:
python run_v2_experiments.py     # regenerates all models, CIs, CV, threshold sweep
```

Model checkpoints are git-ignored (too large); everything else (metrics JSON, figures, notebooks) is committed so the claims can be checked without re-training.

## External validation (done) — and it does NOT generalize

External validation was run on two independent public datasets. The honest result: **the in-distribution AUROC of 0.988 does not survive contact with other cameras and populations.** Full analysis, code, and figures are in [`validation/`](validation/) (see [`validation/FINDINGS.md`](validation/FINDINGS.md)); the pipeline is parity-verified so the drop is real, not a preprocessing bug.

| Setting | AUROC | Notes |
|---|---|---|
| HYGD, patient-level 5-fold CV (in-distribution) | **0.988 ± 0.008** | reported result |
| **PAPILA, zero-shot** | **0.51** (95% CI 0.44–0.58) | chance — collapses |
| **RIM-ONE DL, zero-shot** | **0.61** (95% CI 0.55–0.66) | poor; saturates toward "glaucoma" |
| RIM-ONE → PAPILA, after cross-dataset fine-tune | 0.68 | still weak |
| PAPILA → RIM-ONE, after cross-dataset fine-tune | 0.47 | *degrades* |

On both external sets the model saturates — it calls almost everything glaucoma, with healthy and glaucoma probabilities essentially indistinguishable. Recalibration (temperature/Platt) fixes the calibration (ECE 0.74 → 0.08) but, as expected, **cannot restore discrimination** (AUROC is invariant to monotone recalibration). A short cross-dataset fine-tune fits each source dataset well (val AUROC 0.92–0.96) but does **not** produce a model that transfers — each dataset teaches its own shortcuts (camera, field of view, processing), not universal glaucoma features.

**Honest conclusion.** One small single-hospital dataset — even with light fine-tuning on a second external set — is not enough to build a glaucoma classifier that generalizes across fundus datasets. The in-distribution result is real and cleanly reported, but it is the ceiling of this data, not a deployable model. This is consistent with the literature and is precisely why the field moved to foundation models pretrained on very large multi-dataset corpora (e.g. RETFound). Reporting this transparently — rather than a cherry-picked drop — is the point of the exercise.

## Attribution

Dataset: Abramovich O, Pizem H, Fhima J, et al. *Hillel Yaffe Glaucoma Dataset (HYGD)*, PhysioNet (Open Data Commons Attribution License v1.0); and *GONet: A Generalizable Deep Learning Model for Glaucoma Detection*, arXiv 2025. Quality scores: Abramovich O, et al. *FundusQ-Net*, Comput Methods Programs Biomed 2023;239:107522.
