# HYGD Glaucoma Detection from Fundus Images: Baseline Classification, Explainability, and Clinical Error Analysis

> **Evidence status (updated 2026-09-04):** the preferred internal discrimination estimate is a read-only aggregate reanalysis of frozen OOF predictions from a private five-fold model run asserted to date from 2026-07-11: mean outer-fold linked-group AUROC **0.9908** (conditional 95% CI **0.9790-0.9990**; 283 groups). No contemporaneous public timestamp attests that legacy run, and no model was retrained. The current public implementation is `2026-09-04-v5`; no end-to-end v5 model run is claimed for these numbers. See the privacy-safe [aggregate receipt](results/repaired_internal_evaluation_summary.json).
>
> **Bottom line:** transportability is not established. Later locked source-only tests and shortcut controls prevented the attractive development results from being promoted as external validation or clinical evidence.
>
> **Start here:** [HYGD Failure-First Glaucoma AI Audit](HYGD_FAILURE_FIRST_RESEARCH_BRIEF.md) - a concise map of what was tested, what failed, what the evidence supports, and what would justify a genuinely different next study.

## 1. Clinical context

Glaucoma is a leading cause of irreversible blindness worldwide. It is often asymptomatic until significant, permanent optic nerve damage has already occurred, which makes photographic screening of the optic disc (via fundus imaging) a clinically important early-detection tool — a cheap, non-invasive image that a model can flag for a human specialist to review, not replace.

## 2. Project objective

Build a reproducible baseline classifier for glaucoma detection from retinal fundus images, paired with a failure-first audit of the historical evaluation and explainability work — not a state-of-the-art benchmark, not a clinical device.

**Original baseline scope / non-goals (deliberate).** The HYGD baseline began as a clean, understandable single-dataset classifier — explicitly *not* a PhD-level contribution, *not* SOTA-chasing, *not* OCT segmentation, and *not* a black-box tutorial clone. Later multi-dataset stress tests and adaptive experiments are a separate audit chronology; they did not convert the baseline into a transportable model.

### Current evidence status

| Status | Evidence | Meaning |
|---|---|---|
| `historical` | Single-split AUROC 0.976; CV AUROC 0.988 +/- 0.008 | Development results retained but superseded as the preferred internal estimate |
| `preferred internal` | [Duplicate-aware repair](INTERNAL_EVALUATION_REPAIR.md): mean outer-fold group AUROC 0.9908 [0.9790-0.9990] | Scale-robust strong in-distribution discrimination; single-site post-development resampling only |
| `adaptive development` | PAPILA/RIM-ONE recovery chronology in [VALIDATION.md](VALIDATION.md) and [validation/FINDINGS.md](validation/FINDINGS.md) | Historical development evidence, not untouched external validation |
| `failed confirmatory` | [HYGD-CEXT-1.1](SOURCE_ONLY_QUALIFICATION_REPORT.md) and [HYGD-CEXT-2.0](HYGD_CEXT_2_0_RESULT.md) | Source-only qualification did not establish transportability; the confirmatory target-access gate remains blocked |
| `mechanism only` | [Shortcut Map 1](HYGD_SHORTCUT_MAP_1_RESULT.md) | Source signal is strongly encoded; no bounded preprocessing repair was found |
| `needs-proof` | RIM-ONE permutation mechanism and mixed-dataset use compatibility | Control behavior limits disease-AUROC interpretation; historical adaptive values should not support an external claim without clarification |

## 3. Dataset

**Hillel Yaffe Glaucoma Dataset (HYGD)** v1.1.0 — PhysioNet, DOI [10.13026/m92s-0z95](https://doi.org/10.13026/m92s-0z95), distributed under the Open Data Commons Attribution License v1.0. Anyone may access the files subject to the current license terms; users must provide the required attribution and citations.

- 747 fundus images associated with 288 supplied patient IDs (reported ages 36–95), captured with a TOPCON DRI OCT Triton retinal camera (45° field of view).
- Dataset-author reference labels are based on a reported ophthalmic work-up (visual acuity, IOP, OCT, visual field, ≥1 year follow-up), rather than image review alone.
- Class balance: 548 GON+ (73.4%) / 199 GON- (26.6%) — a real imbalance, handled via class-weighted loss (see Methods).
- Each image has a FundusQ-Net quality score (1–10); mean 5.9, range 2.0–7.7.
- Supplied IDs contribute 1–14 images each (mean 2.6) — this is why the historical train/val/test split grouped the supplied patient-ID field rather than image rows (see Methods).

Download: `https://physionet.org/content/hillel-yaffe-glaucoma-dataset/get-zip/1.1.0/` (~118MB). Not vendored into this repo — download it yourself into `data/raw/` (expects `data/raw/Images/` + `data/raw/Labels.csv`; see `.gitignore`).

## 4. Methods

- **Historical development split:** patient-level `GroupShuffleSplit` (`sklearn`), 70/15/15 train/val/test, seed 42. It had zero supplied-patient overlap, but configurations were compared on its test partition and its image-row confidence intervals did not account for repeated images.
- **Preferred internal evaluation:** exact SHA-256 duplicates are counted once; supplied patient IDs sharing a hash are linked into one evaluation group; five stratified outer group folds are separated from the inner group split used for checkpoint and threshold selection; each outer fold is evaluated once. Confidence intervals resample whole evaluation groups.
- **Historical frozen-head baseline recipe:** resize to 224×224 and apply ImageNet normalization; train a new two-class ResNet18 head over a frozen ImageNet backbone for 8 epochs with class-weighted `CrossEntropyLoss`, Adam at 1e-3, batch size 32, on CPU. The historical split produced weights `[1.98, 0.67]` for GON-/GON+.
- **Preferred evaluator's fixed recipe:** resize to 224×224, apply light training-only augmentation (horizontal flip, rotation, mild colour jitter) and ImageNet normalization; fine-tune ResNet18 `layer4` at 1e-4 and its head at 1e-3 for at most 10 epochs, batch size 32. Class weights are recomputed inside each outer-training partition, and the checkpoint is selected solely by minimum inner-validation loss before the outer fold is evaluated.

## 5. Results

### Historical development comparison

The table below is retained as project history. All metrics are on the **same development test set** (99 images / 44 supplied patient IDs, zero supplied-ID overlap with train/val). The model configuration was selected after this comparison, and the confidence intervals were image-level despite repeated images per supplied ID. These numbers are therefore not the preferred internal estimate.

| Model | AUC | Sensitivity | Specificity |
|---|---|---|---|
| v1 — frozen backbone, head only (baseline) | 0.952 | 0.892 | 0.971 |
| v2 — frozen backbone + augmentation | 0.965 | 0.938 | 0.882 |
| **v2 — fine-tuned `layer4` + augmentation (best)** | **0.976** | **0.954** | **0.941** |

Best model — historical image-row 95% CIs: AUC [0.943, 0.998], sensitivity [0.90, 1.00], specificity [0.85, 1.00]. Confusion matrix at 0.5: TN=32, FP=2, FN=3, TP=62. (Historical artifact: `results/v2_comparison.json`; it is not the canonical internal result.)

The original standalone model-comparison, ROC, and confusion-matrix charts were removed because the images did not carry their now-required historical/noncanonical status. The aggregate values remain in self-adjudicating JSON and text records.

Within that historical development comparison, progressively unfreezing the last residual block and adding light augmentation improved every reported metric over the frozen-head baseline. All three configurations remain deliberately modest - this is a baseline study, not a maximum-performance benchmark.

**Fine-tuning honesty note:** the fine-tuned model's train loss drops toward zero while val loss plateaus and then drifts up (classic mild overfitting on 535 images) — so training keeps the best-validation checkpoint (epoch 9), not the last one. This is expected on a small dataset and is why the backbone is only *partially* unfrozen (`layer4`), not fully.

The historical training chart was removed for the same reason; the run-level losses remain provenance, not proof against overfitting.

### Preferred internal estimate: duplicate-aware inner/outer evaluation

The historical CV AUROC of `0.988 +/- 0.008` grouped supplied patient IDs, but it reused each fold for checkpoint selection and scoring, and it missed exact images duplicated under different patient IDs. It is superseded, not erased.

The repaired protocol links patient IDs that share an exact image, counts each SHA-256 hash once, fixes the model recipe before outer evaluation, uses a separate inner validation split for checkpoint and threshold selection, and evaluates every independent outer group once.

| Analysis level | N | AUROC (95% conditional CI) | Sensitivity | Specificity |
|---|---:|---:|---:|---:|
| **Evaluation group: equal-weight mean of 5 outer-fold AUROCs (primary)** | **283** | **0.9908 (0.9790-0.9990)** | **0.9563** | **0.9700** |
| Pooled group OOF probabilities (continuity metric) | 283 | 0.9904 (0.9797-0.9980) | — | — |
| Image (secondary) | 737 | 0.9837 (0.9731-0.9925) | 0.9500 | 0.9289 |

The model recipe was historically informed by HYGD, so this remains a repaired **post-development internal resampling estimate**. It supports strong in-distribution discrimination, not performance at a new hospital. Full protocol and audit: [INTERNAL_EVALUATION_REPAIR.md](INTERNAL_EVALUATION_REPAIR.md).

The primary AUROC averages the five group-level outer-fold AUROCs equally, so it does not compare raw score scales from separately fitted fold models. Its 5,000-sample percentile interval resamples whole evaluation groups within each fixed fold; the deterministic Monte Carlo contract uses seed `20269714`, canonical group ordering, and produced 5,000/5,000 valid replicates. Sensitivity/specificity pool the OOF binary decisions made with thresholds recorded in the legacy result as inner-validation-selected; that selection was not independently rerun. The intervals condition on the frozen OOF predictions and recorded fold/decision choices; they exclude uncertainty from retraining, checkpoint and threshold selection, recipe selection, transportability, and deployment. The pooled 0.9904 AUROC and the image-level AUROC use global group-cluster intervals over pooled fixed scores; both remain secondary because their AUROCs compare score scales from separately fitted fold models and are not invariant to fold-specific monotone rescaling.

The legacy fold plot was removed because its standalone presentation did not disclose the same-fold checkpoint/scoring defect.

## 6. Explainability

Historical Grad-CAM and case-level error review were post hoc analyses of one selected development split. The row-level images, saliency maps, identifiers, and case interpretations are not part of the public evidence package. Grad-CAM was not evaluated against a predeclared localization endpoint or independent expert annotations and cannot establish causal feature use, localization accuracy, clinical reasoning, or absence of shortcut learning. No preprocessing or quality gate is justified by that review.

**Historical exploratory image-row error analysis.** The selected development split contained five image-level errors. Their mean FundusQ-Net score was 5.48 versus 6.04 among correct predictions; an exploratory one-sided Mann-Whitney comparison yielded p = 0.45. The analysis was post hoc, underpowered, image-level despite repeated-patient structure, and unadjusted for selection or multiplicity. It provides no evidence for or against a quality-error association and does not justify a quality threshold. See the self-adjudicating aggregate record in `results/error_analysis.json`.

## 7. Reproducibility and integrity checks

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Download the dataset into `data/raw/` (see §3 above), then run the notebooks in order: `01_eda` → `02_preprocessing` → `03_baseline_model` → `04_explainability`. Notebook `05_v2_experiments` is historical-only and noncanonical. These instructions reproduce historical development paths locally; they do not generate canonical evidence, and row-level outputs must not be committed.

The repaired internal evaluator and its dataset-free regression suite are now public:

```bash
python -m unittest discover -s validation -p 'test_*.py' -v
python validation/internal_evaluation_repair.py --audit-only
python validation/internal_evaluation_repair.py
python validation/reanalyze_frozen_oof.py --describe-contract
```

Generated audit, fold, OOF, and result files stay under the git-ignored `results/internal_evaluation_repair*` namespace and are not publication artifacts. The historical single-split runner is gated behind `--run-historical-comparison`, emits no cross-validation estimate, and is not the canonical path; inspect its policy with `python run_v2_experiments.py --describe-policy`.

Audit-only and partial `--max-folds` runs use distinct noncanonical filename namespaces. Every run refuses to overwrite any existing bundle target; use a new direct prefix such as `--output-prefix results/internal_evaluation_repair_20260831a` when preserving an earlier run.

Only the documented defaults (5 folds, 10 epochs, batch size 32, seed `20260711`, inner fraction 0.15, sensitivity target 0.95, and 5,000 bootstrap samples) can emit a complete preferred-result schema. Altered settings require a proper-subset `--max-folds` smoke run and remain noncanonical.

The complete evaluator recomputes the locked patient/duplicate-linked group-to-fold assignment from the audited dataset before accepting a terminal bundle; coherent reassignment of both the fold manifest and OOF rows is therefore rejected. The terminal audit is also schema-closed and cryptographically bound to the exact result, OOF, fold, and dataset-contract records.

The frozen OOF predictions came from a private run asserted to date from 2026-07-11; no contemporaneous public timestamp attests it. On 2026-09-04, the v5 aggregate code re-read the bound private OOF artifacts and reproduced the scale-robust 0.9908 estimate without training or inference. The public [aggregate receipt](results/repaired_internal_evaluation_summary.json) binds those private source bytes, canonical OOF identity, canonical group-to-fold assignment, implementation, script, scientific command, and runtime by SHA-256 or exact value while publishing no row-level predictions or private paths. `--describe-contract` prints the location-free scientific command without reading private inputs. The receipt is not an end-to-end v5 model-run receipt or a public numerical reproduction. Passing synthetic tests alone does not reproduce the model run or its numbers. Start with [HYGD_FAILURE_FIRST_RESEARCH_BRIEF.md](HYGD_FAILURE_FIRST_RESEARCH_BRIEF.md) and its Evidence Map.

## 8. Historical threshold analysis

The original development analysis compared sensitivity and specificity on the 44-supplied-ID test split. It is retained to show how the threshold trade-off was explored, not to recommend a clinical operating point.

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
- **Modest by design** — the preferred evaluator uses a partially-unfrozen ResNet18 (`layer4` + head), at most 10 epochs, no hyperparameter search, and no architecture search. It is not an attempt at maximum achievable performance (see the Scope note in §2).
- **Calibration and clinical utility are untested.** The historical threshold sweep is not a deployment recommendation, and no patient-impact or prospective study was performed.
- **This is a student portfolio/research artifact, not a clinical device, and must never be used for real diagnostic decisions.**

## Repository structure

```
data/raw/               # HYGD dataset (download yourself — not committed, see .gitignore)
notebooks/              # 01-04 historical path; 05_v2_experiments is historical-only
src/                    # baseline/model helpers; ambiguous legacy CV/bootstrap entry points are disabled
run_v2_experiments.py   # explicitly gated historical single-split comparison; never canonical CV
validation/evaluation_utils.py            # Torch-free grouping/split/metric/bootstrap guardrails
validation/internal_evaluation_repair.py  # preferred duplicate-aware inner/outer evaluator
validation/test_*.py                      # synthetic regression and public-claim boundary tests
analyze_errors.py       # historical post-hoc quality/error analysis; outputs stay local
figures/                # aggregate EDA charts only; generated result panels stay local
results/                # historical committed artifacts; repair outputs and checkpoints are ignored
results/repaired_internal_evaluation_summary.json  # aggregate-only public receipt; no row-level data
HYGD_FAILURE_FIRST_RESEARCH_BRIEF.md  # current reviewer-facing evidence map
INTERNAL_EVALUATION_REPAIR.md         # preferred internal estimate and audit boundary
SOURCE_ONLY_QUALIFICATION_REPORT.md   # closed HYGD-CEXT-1.1 source-only result
HYGD_MANUAL_GEOMETRY_RESULT.md        # closed center/diameter/shortcut investigation
HYGD_CEXT_2_0_PROTOCOL.md             # annotated public copy of the frozen DINOv2 protocol
HYGD_CEXT_2_0_EXECUTION_SPEC.md       # execution and provenance boundary
HYGD_CEXT_2_0_RESULT.md               # failed dataset-origin gate
HYGD_SHORTCUT_MAP_1_PROTOCOL.md       # frozen preprocessing/attribution protocol
HYGD_SHORTCUT_MAP_1_RESULT.md         # no bounded preprocessing repair candidate
references/             # (empty — no external papers/notes added yet)
```

## Citation

If you use HYGD, cite its [current PhysioNet record](https://physionet.org/content/hillel-yaffe-glaucoma-dataset/1.1.0/) and DOI [10.13026/m92s-0z95](https://doi.org/10.13026/m92s-0z95). Take the exact underlying-article citation from that current official record rather than from a file that is absent in a fresh clone.
