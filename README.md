# HYGD Glaucoma Detection from Fundus Images: Baseline Classification, Explainability, and Clinical Error Analysis

> **Current status (2026-07-14):** the repaired duplicate-aware internal evaluation reports group AUROC **0.9904** (95% CI **0.9797-0.9980**) across 283 linked evaluation groups. This is strong single-site, post-development resampling evidence from one hospital and one camera.
>
> **Bottom line:** transportability is not established. Later locked source-only tests and shortcut controls prevented the attractive development results from being promoted as external validation or clinical evidence.
>
> **Start here:** [HYGD Failure-First Glaucoma AI Audit](HYGD_FAILURE_FIRST_RESEARCH_BRIEF.md) - a concise map of what was tested, what failed, what the evidence supports, and what would justify a genuinely different next study.

## 1. Clinical context

Glaucoma is a leading cause of irreversible blindness worldwide. It is often asymptomatic until significant, permanent optic nerve damage has already occurred, which makes photographic screening of the optic disc (via fundus imaging) a clinically important early-detection tool — a cheap, non-invasive image that a model can flag for a human specialist to review, not replace.

## 2. Project objective

Build a clean, reproducible baseline classifier for glaucoma detection from retinal fundus images, paired with an honest explainability and clinical error analysis — not a state-of-the-art benchmark, not a clinical device.

**Scope / non-goals (deliberate).** This is a clean, honest, reproducible baseline — explicitly *not* a PhD-level contribution, *not* SOTA-chasing, *not* a multi-dataset study, *not* OCT segmentation (a possible separate later project), and *not* a black-box tutorial clone. Every step is meant to be understandable and defensible rather than maximally performant.

### Current evidence status

| Status | Evidence | Meaning |
|---|---|---|
| `historical` | Single-split AUROC 0.976; CV AUROC 0.988 +/- 0.008 | Development results retained but superseded as the preferred internal estimate |
| `preferred internal` | [Duplicate-aware repair](INTERNAL_EVALUATION_REPAIR.md): group AUROC 0.9904 [0.9797-0.9980] | Strong in-distribution discrimination; single-site post-development resampling only |
| `adaptive development` | PAPILA/RIM-ONE recovery chronology in [VALIDATION.md](VALIDATION.md) and [validation/FINDINGS.md](validation/FINDINGS.md) | Historical development evidence, not untouched external validation |
| `failed confirmatory` | [HYGD-CEXT-1.1](SOURCE_ONLY_QUALIFICATION_REPORT.md) and [HYGD-CEXT-2.0](HYGD_CEXT_2_0_RESULT.md) | Source-only qualification did not establish transportability; the confirmatory target-access gate remains blocked |
| `mechanism only` | [Shortcut Map 1](HYGD_SHORTCUT_MAP_1_RESULT.md) | Source signal is strongly encoded; no bounded preprocessing repair was found |
| `needs-proof` | RIM-ONE permutation mechanism and mixed-dataset use compatibility | Control behavior limits disease-AUROC interpretation; historical adaptive values should not support an external claim without clarification |

## 3. Dataset

**Hillel Yaffe Glaucoma Dataset (HYGD)** v1.1.0 — PhysioNet, DOI [10.13026/m92s-0z95](https://doi.org/10.13026/m92s-0z95), Open Data Commons Attribution License v1.0 (open access, no credentialing required beyond citation).

- 747 fundus images from 288 patients (ages 36–95), captured with a TOPCON DRI OCT Triton retinal camera (45° field of view).
- Labels are **gold-standard**: based on full ophthalmic work-up (visual acuity, IOP, OCT, visual field, ≥1 year follow-up) rather than image-review alone — a real strength of this dataset over many public glaucoma sets.
- Class balance: 548 GON+ (73.4%) / 199 GON- (26.6%) — a real imbalance, handled via class-weighted loss (see Methods).
- Each image has a FundusQ-Net quality score (1–10); mean 5.9, range 2.0–7.7.
- Patients contribute 1–14 images each (mean 2.6) — this is why the train/val/test split is done at the **patient** level, not the image level (see Methods).

Download: `https://physionet.org/content/hillel-yaffe-glaucoma-dataset/get-zip/1.1.0/` (~118MB). Not vendored into this repo — download it yourself into `data/raw/` (expects `data/raw/Images/` + `data/raw/Labels.csv`; see `.gitignore`).

## 4. Methods

- **Split:** patient-level `GroupShuffleSplit` (`sklearn`), 70/15/15 train/val/test, seed 42. Verified zero patient overlap across splits. Train 535 img/200 patients (74.8% GON+), val 113 img/44 patients (73.5% GON+), test 99 img/44 patients (65.7% GON+) — the test split's GON+ rate is somewhat lower than train/val since `GroupShuffleSplit` balances patient *counts* per split, not the label distribution across groups; with only 288 patients this residual skew is expected and noted as a limitation rather than engineered away.
- **Preprocessing:** resize to 224×224, ImageNet mean/std normalization (matches the pretrained backbone). No aggressive augmentation for this baseline.
- **Model:** ResNet18, ImageNet-pretrained, **frozen backbone** + new trainable 2-class linear head. Freezing was a deliberate choice for a 535-image training set — fine-tuning the whole network risks overfitting fast on this little data.
- **Loss:** class-weighted `CrossEntropyLoss` (weights `[1.98, 0.67]` for GON-/GON+) to counter the 73/27 imbalance, instead of oversampling.
- **Training:** 8 epochs, Adam, lr=1e-3, batch size 32, CPU. ~5 minutes wall-clock.

## 5. Results

### Historical development comparison

The table below is retained as project history. All metrics are on the **same development test set** (99 images / 44 patients, zero supplied-patient overlap with train/val). The model configuration was selected after this comparison, and the confidence intervals were image-level despite repeated images per patient. These numbers are therefore not the preferred internal estimate.

| Model | AUC | Sensitivity | Specificity |
|---|---|---|---|
| v1 — frozen backbone, head only (baseline) | 0.952 | 0.892 | 0.971 |
| v2 — frozen backbone + augmentation | 0.965 | 0.938 | 0.882 |
| **v2 — fine-tuned `layer4` + augmentation (best)** | **0.976** | **0.954** | **0.941** |

Best model — 95% CIs: AUC [0.943, 0.998], sensitivity [0.90, 1.00], specificity [0.85, 1.00]. Confusion matrix at 0.5: TN=32, FP=2, FN=3, TP=62. (Source of truth: `results/v2_comparison.json`.)

![Model comparison](figures/11_model_comparison.png)
![v2 ROC](figures/12_roc_v2.png)
![v2 confusion matrix](figures/13_confusion_v2.png)

Within that historical development comparison, progressively unfreezing the last residual block and adding light augmentation improved every reported metric over the frozen-head baseline. All three configurations remain deliberately modest - this is a baseline study, not a maximum-performance benchmark.

**Fine-tuning honesty note:** the fine-tuned model's train loss drops toward zero while val loss plateaus and then drifts up (classic mild overfitting on 535 images) — so training keeps the best-validation checkpoint (epoch 9), not the last one. This is expected on a small dataset and is why the backbone is only *partially* unfrozen (`layer4`), not fully.

![v2 training curves](figures/14_training_curves_v2.png)

### Preferred internal estimate: duplicate-aware inner/outer evaluation

The historical CV AUROC of `0.988 +/- 0.008` grouped supplied patient IDs, but it reused each fold for checkpoint selection and scoring, and it missed exact images duplicated under different patient IDs. It is superseded, not erased.

The repaired protocol links patient IDs that share an exact image, counts each SHA-256 hash once, fixes the model recipe before outer evaluation, uses a separate inner validation split for checkpoint and threshold selection, and evaluates every independent outer group once.

| Analysis level | N | AUROC (95% cluster-bootstrap CI) | Sensitivity | Specificity |
|---|---:|---:|---:|---:|
| **Duplicate-aware evaluation group (primary)** | **283** | **0.9904 (0.9797-0.9980)** | **0.9563** | **0.9700** |
| Image (secondary) | 737 | 0.9837 (0.9731-0.9925) | 0.9500 | 0.9289 |

The model recipe was historically informed by HYGD, so this remains a repaired **post-development internal resampling estimate**. It supports strong in-distribution discrimination, not performance at a new hospital. Full protocol and audit: [INTERNAL_EVALUATION_REPAIR.md](INTERNAL_EVALUATION_REPAIR.md).

![Historical CV folds, superseded by the repaired evaluation](figures/15_cv_folds.png)

## 6. Explainability

Grad-CAM (last conv block of ResNet18) on 5 correct + 5 incorrect test predictions:

![Grad-CAM correct](figures/08_gradcam_correct.png)
![Grad-CAM incorrect](figures/09_gradcam_wrong.png)

On the historical development predictions, the heatmap concentrates on the optic disc / peripapillary region in many correct cases. This is a limited localization sanity signal, not evidence that the model ignores acquisition artifacts; the later dataset-origin probes show that strong source information remains encoded.

A full per-error clinical write-up is in `notebooks/04_explainability.ipynb`, with all 5 errors shown as original + Grad-CAM in `figures/16_the_5_errors_annotated.png`. It separates the errors into two distinct failure modes: **localization** (in 2 of 3 misses the model's attention is off the disc — one an image-quality artifact, one a detection miss) and **interpretation** (both false alarms look at the disc but over-call, plausibly on a large physiological cup or co-existing findings). The disc descriptions there are explicitly framed as observational hypotheses for *model behaviour*, not diagnoses.

**Historical data-driven error analysis.** The selected development model made 5 errors on the test split (3 missed glaucoma / false negatives, 2 false alarms / false positives). The misclassified images had a slightly lower mean FundusQ-Net quality score (5.48 vs 6.04 for correct), but the difference was **not statistically significant** (Mann-Whitney U, one-sided, p = 0.45). This small retrospective analysis does not explain the later source-shortcut findings. See `results/error_analysis.json` and the figure below.

![Error vs quality](figures/10_error_vs_quality.png)

## 7. Baseline reproducibility

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Download the dataset into `data/raw/` (see §3 above), then run the notebooks in order: `01_eda` → `02_preprocessing` → `03_baseline_model` → `04_explainability`. These instructions reproduce the original baseline workflow.

The first failure-first reviewer packet adds the audited narrative reports but intentionally does not publish the later executable/result bundle. It therefore does not claim that the newer qualification runs are reproducible from this public snapshot alone. Start with [HYGD_FAILURE_FIRST_RESEARCH_BRIEF.md](HYGD_FAILURE_FIRST_RESEARCH_BRIEF.md) and its Evidence Map.

## 8. Historical threshold analysis

The original development analysis compared sensitivity and specificity on the 44-patient test split. It is retained to show how the threshold trade-off was explored, not to recommend a clinical operating point.

The partially fine-tuned development model reported sensitivity 0.954 and specificity 0.941 at the default 0.5 threshold.

The retrospective threshold sweep on that same split was:

| threshold | sensitivity | specificity | missed (FN) | false alarms (FP) |
|---|---|---|---|---|
| 0.30 | 0.969 | 0.824 | 2 | 6 |
| 0.40 (historical candidate) | 0.969 | 0.912 | 2 | 3 |
| 0.50 (default) | 0.954 | 0.941 | 3 | 2 |
| 0.64 | 0.938 | 0.941 | 4 | 2 |

In the repaired internal evaluation, thresholds selected independently inside the five training folds ranged from 0.483 to 0.948. That spread is a calibration warning. No fixed clinical threshold is justified by this project.

## 9. Limitations

- **Transportability is not established.** The PAPILA/RIM-ONE recovery chronology is adaptive development evidence. Target AUROC was displayed during development, and target anatomical resources affected preprocessing. It is not an untouched external test.
- **The first locked source-only qualification failed.** HYGD-CEXT-1.1 produced equal-source mean AUROC 0.6227 [0.5822-0.6632]; its geometry candidate was ineligible before glaucoma-classifier training. See [SOURCE_ONLY_QUALIFICATION_REPORT.md](SOURCE_ONLY_QUALIFICATION_REPORT.md).
- **A stronger representation did not solve the shortcut.** Frozen DINOv2 features reached equal-source mean AUROC 0.7105 [0.6746-0.7423], but dataset-origin accuracy was 0.9994 against a required value below 0.75. See [HYGD_CEXT_2_0_RESULT.md](HYGD_CEXT_2_0_RESULT.md).
- **Bounded preprocessing did not repair source decoding.** The best fixed branch reported equal-source mean AUROC 0.7364 while origin accuracy remained 0.9933. LEACE is mechanism evidence only. See [HYGD_SHORTCUT_MAP_1_RESULT.md](HYGD_SHORTCUT_MAP_1_RESULT.md).
- **The RIM-ONE negative control remains unresolved.** Permutation AUROC remained 0.6759-0.7513 across later branches. Its mechanism is `needs-proof` and limits interpretation of the corresponding disease AUROCs.
- **All preferred internal evidence remains single-site.** The repaired evaluation uses 737 unique images and 283 linked groups, but every group comes from one hospital and one camera.
- **Modest by design** — even the best model is only a partially-unfrozen ResNet18 (`layer4` + head), 10 epochs, no hyperparameter search, no architecture search. Explicitly not an attempt at maximum achievable performance (see the Scope note in §2). The point is a clean, honest, reproducible pipeline, not a leaderboard number.
- **Calibration and clinical utility are untested.** The historical threshold sweep is not a deployment recommendation, and no patient-impact or prospective study was performed.
- **This is a student portfolio/research artifact, not a clinical device, and must never be used for real diagnostic decisions.**

## Repository structure

```
data/raw/               # HYGD dataset (download yourself — not committed, see .gitignore)
notebooks/              # 01_eda, 02_preprocessing, 03_baseline_model, 04_explainability
src/                    # data_utils, train, evaluate, visualize (baseline) + experiments (v2 harness)
run_v2_experiments.py   # historical development comparison and threshold sweep
analyze_errors.py       # data-driven error analysis (errors vs image quality score)
figures/                # EDA + results + explainability + v2 comparison figures (committed)
results/                # metrics.json, v2_comparison.json, cv_results.json, error_analysis.json,
                        #   run logs (model checkpoints are git-ignored — too large)
HYGD_FAILURE_FIRST_RESEARCH_BRIEF.md  # current reviewer-facing evidence map
INTERNAL_EVALUATION_REPAIR.md         # preferred internal estimate and audit boundary
SOURCE_ONLY_QUALIFICATION_REPORT.md   # closed HYGD-CEXT-1.1 source-only result
HYGD_MANUAL_GEOMETRY_RESULT.md        # closed center/diameter/shortcut investigation
HYGD_CEXT_2_0_PROTOCOL.md             # frozen DINOv2 source-only protocol
HYGD_CEXT_2_0_EXECUTION_SPEC.md       # execution and provenance boundary
HYGD_CEXT_2_0_RESULT.md               # failed dataset-origin gate
HYGD_SHORTCUT_MAP_1_PROTOCOL.md       # frozen preprocessing/attribution protocol
HYGD_SHORTCUT_MAP_1_RESULT.md         # no bounded preprocessing repair candidate
references/             # (empty — no external papers/notes added yet)
```

## Citation

If you use the HYGD dataset, cite the PhysioNet resource (DOI 10.13026/m92s-0z95) and the underlying Abramovich et al. 2026 paper — see `data/raw/HYGD_README.md` (added after download) for the exact citation text.
