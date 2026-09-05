# GlaucoGen — Historical PAPILA/RIM-ONE Stress-Test and Adaptive-Development Chronology

> **Status correction (2026-08-31):** This file is a chronological development record. Its external intervals used eye/image-row bootstrap, not patient-cluster bootstrap; its fine-tuning source split was row-level; RIM-ONE subject independence and mixed-source license compatibility remain `needs-proof`. Later recovery sections are target-adaptive experiments, not prospectively untouched external validation, and they do not establish a transportable glaucoma model. See [HYGD_FAILURE_FIRST_RESEARCH_BRIEF.md](../HYGD_FAILURE_FIRST_RESEARCH_BRIEF.md).

> Historical run dated 2026-07-05. The aggregate artifact is now public with an
> explicit `partial_with_deviations` label; row-level predictions, data, and
> checkpoints are not published.

## Headline

The HYGD model whose **historical, superseded CV reported AUROC 0.988**
showed near-chance ranking on the historical PAPILA zero-shot run: **AUROC ≈ 0.51**
(historical noncanonical eye-row interval 0.44–0.58). The private run summary
reported 96.7% of PAPILA eyes above its historical threshold and mean probabilities
of 0.935 versus 0.901; those row-level values are not public and remain `needs-proof`
beyond the aggregate artifact.

| Setting | AUROC | Notes |
|---|---|---|
| HYGD, historical five-fold CV (in-distribution) | 0.988 +/- 0.008 | superseded, non-canonical result |
| HYGD test via `predict.py` (historical parity check) | 0.976 | same-path aggregate; not proof of full pipeline correctness |
| **PAPILA, zero-shot (as-trained preprocessing)** | **0.508** | historical eye-row interval [0.44, 0.58], noncanonical |
| PAPILA, zero-shot + field-of-view standardization | 0.58 | private historical aggregate, `needs-proof` |

## Checks performed in the historical run

- **Historical parity check:** the same `predict.py` path reported AUROC 0.976 on the HYGD development test and a maximum output difference of 2.8e-06 in a private parity run. This narrows, but does not rule out, implementation defects.
- **Aggregate label-count check:** 420 eyes = 333 healthy + 87 glaucoma (suspects excluded), matching PAPILA's published aggregate composition. This does not independently verify row-level label semantics.
- **Framing sensitivity:** HYGD is square while PAPILA is landscape. A private center-square sensitivity run reportedly moved AUROC from 0.51 to 0.58; that observation alone does not identify a causal mechanism.

## Interpretation (honest)

The observed behavior is consistent with reliance on HYGD-specific acquisition or prevalence cues, but it does not identify which features caused the drop. The historical Platt summary reported ECE 0.74 → 0.08 on a grouped but non-stratified split. That is a noncanonical development result, not proof that calibration was fixed. Monotone recalibration cannot improve AUROC ranking.

This is exactly what external stress testing exists to reveal. The historical internal path was later repaired for duplicate and inner/outer separation defects; see [`../INTERNAL_EVALUATION_REPAIR.md`](../INTERNAL_EVALUATION_REPAIR.md). The zero-shot collapse remains useful development evidence, not proof of a universal limit.

## Why this is useful development evidence

The transparent report of an external stress-test collapse, with observed behavior separated from unproved mechanisms, is more decision-useful than a selectively reported performance drop. It directly motivates the next experiments without making an audience or prestige claim.

## Next experiments (post-freeze, if pursued)

1. **Disc-centred crop + light fine-tune** (Tier-3): does a small amount of PAPILA/RIM-ONE fine-tuning on disc-standardized input recover transferable signal? (train RIM-ONE → test PAPILA, and vice-versa; never fine-tune and test on the same set.)
2. **Shortcut-learning analysis:** any future saliency work requires a predeclared endpoint and independent adjudication. Grad-CAM alone cannot identify a causal shortcut or distinguish border/field artifacts from disease-relevant features.
3. Add **RIM-ONE DL** as a second stress-test set, subject to verified subject identity and use-term compatibility; a second dataset would not by itself confirm a general mechanism.

## Files
- `results/external_papila.json` — full metrics (zero-shot + recalibration).
- The legacy collapse chart was removed because its standalone labels overstated a partial historical stress test; the aggregate JSON and corrected narrative remain.
- `validation/make_papila_labels.py` and the current hardened `validation/eval_external.py` document a reproducible prospective path; the exact historical producer and full historical run reproduction remain `needs-proof`.
- PAPILA images are git-ignored (GPL-3.0+, not redistributed); the label CSV is derived metadata.

---

## Update — cross-dataset fine-tune (Tier-3): does a short fine-tune fix transfer? NO.

The historical experiment fine-tuned the HYGD model (layer4+fc, class-weighted, early-stopped) on one external dataset and evaluated aggregate disease metrics on the other. Target disease rows were not used for the corresponding fine-tune, but the source validation split was row-level, broader target results were visible, and RIM-ONE subject identity was not verified. The RIM-ONE zero-shot aggregate was AUROC 0.606; it does not confirm a general mechanism.

| Fine-tune direction | target zero-shot | target after fine-tune | Δ | source-val (fit) |
|---|---|---|---|---|
| RIM-ONE → PAPILA | 0.508 | **0.683** | +0.18 | 0.961 |
| PAPILA → RIM-ONE | 0.606 | **0.467** | −0.14 | 0.925 |

**Verdict: this fine-tuning experiment did NOT produce a model that transferred to the other dataset.** The best case (RIM-ONE→PAPILA) recovers to only 0.68 — still weak; the other direction actually *degrades* below chance (0.47). The high source-validation fit with poor transfer is consistent with source-specific shortcut reliance and is insufficient evidence of source-invariant disease features; it does not prove that no universal signal exists.

## Complete, honest conclusion

A single small single-hospital dataset — even with fine-tuning on a second external set — did **not** establish a glaucoma model that transports across sources. The historical, superseded in-distribution CV AUROC 0.988 did not predict the zero-shot result on other cameras/populations. This is a project-specific finding, not proof of an absolute data ceiling.

---

## Update 2 — adaptive benchmark recovery (PAPILA 0.51 → 0.83)

> **Evidence boundary:** Every recovery result below is adaptive, mechanism-supporting development evidence. It does not establish transportability; the later locked HYGD-CEXT-1.1 and HYGD-CEXT-2.0 qualifications did not establish it either.

After the zero-shot collapse and failed naive fine-tune, a literature-grounded domain-generalization pipeline improved the PAPILA adaptive benchmark. PAPILA disease labels were excluded from the loss and source-validation checkpoint criterion in the final runs, but target AUROC was displayed during development and target anatomical resources affected preprocessing; this is not confirmatory transfer.

| Stage | Held-out PAPILA AUROC |
|---|---|
| Zero-shot (original failure) | 0.508 |
| Single-source naive fine-tune | 0.683 |
| + disc-crop + colour-norm + multi-source + SWA | 0.794 [historical eye-row interval 0.74–0.84] |
| **+ test-time augmentation (historical run)** | **0.825 [historical eye-row interval 0.77–0.87]** |

The legacy recovery chart was removed because its standalone headline overstated the current evidence boundary; the corrected adaptive chronology below is the public record.

**Historical adaptive recipe (the final-run operations were label-free, but the development process was not target-blind):**
1. **Automatic disc-centred, disc-size-standardized cropping.** A U-Net disc segmenter trained on PAPILA+RIM-ONE masks reported validation Dice 0.958 in private development. Crops used a square of 2.2× estimated disc diameter. This does not prove successful localization for every image or elimination of field-of-view shortcuts.
2. **Colour/illumination normalization** — Shades-of-Gray colour constancy + CLAHE, applied identically at train and test, intended to reduce camera colour-cast.
3. **Multi-source training** — train on HYGD + RIM-ONE (two domains) with domain- and class-balanced sampling + a base-rate fix. PAPILA disease images were excluded from the final training loss, while target metrics/resources had already influenced the development chronology.
4. **Heavy colour/geometry augmentation** — pushes the model toward camera-invariant morphology.
5. **SWA (stochastic weight averaging)** for source-validation model selection — label-free within each final run; it does not undo earlier target visibility.
6. **Test-time augmentation** — label-free averaging over flips/rotations at inference.

**Updated guardrail interpretation:** PAPILA disease labels were excluded from the training loss and source-validation checkpoint criterion in the final runs. However, target AUROC was displayed during development and target anatomical resources entered preprocessing. Patient-grouped splits and non-redistribution still matter, but they do not convert this adaptive chronology into untouched external validation.

**Honest caveats.** PAPILA contains both eyes from 210 patients; the historical 0.77–0.87 interval resampled eye rows and is not a patient-cluster CI. The 0.83 adaptive benchmark is not the historical, superseded 0.988 in-distribution CV and is not confirmatory generalization. A private dataset probe reportedly showed residual source separability near 0.90 (`needs-proof`).

## Historical execution path (inputs are not public)

`validation/build_disc_coords.py` (disc localization) → `validation/build_crops.py` (crop + colour-norm + dataset-probe gate) → historical `validation/train_generalize.py` route. The current script is prospectively hardened, writes to a new patient-aware namespace, and must not be treated as the exact producer of every field in `results/generalize_attemptA.json`; the historical TTA producer is `needs-proof`.

### Update 3 — VCDR auxiliary head (Attempt B): held-out PAPILA 0.825 → 0.871

Adding a vertical cup-to-disc-ratio (VCDR) regression head in the historical multi-source run produced a PAPILA adaptive benchmark of **0.871** with a noncanonical eye-row interval **[0.82–0.91]**. VCDR was supervised from RIM-ONE masks while HYGD was masked out of that auxiliary loss. PAPILA disease images were excluded from the final training loss, but target AUROC and anatomical resources were part of the wider development history; mixed-source license compatibility remains `needs-proof`. `results/generalize_attemptB.json`.

---

## Update 4 — seed robustness + symmetric adaptive check

The Attempt-B headline (0.871) was a single run. Two adaptive checks followed:
seed sensitivity, and the same disease-label-held-out recipe in the reverse
direction? Each seed varies weight init, augmentation, sampler, AND the source
train/val split (`--seed` → `torch.manual_seed` + `np.random.seed` +
historical `GroupShuffleSplit`). The checked-in seed-0 summary matches 0.8708 numerically, but the exact historical producer identity remains `needs-proof`.

**(a) Seed robustness — held-out PAPILA (forward: train HYGD+RIM-ONE):**

| seed | held-out PAPILA AUROC (TTA) |
|---|---|
| 0 (published) | 0.871 |
| 1 | 0.838 |
| 2 | 0.850 |
| 3 | 0.884 |
| 4 | 0.845 |
| **sample mean ± sample SD** | **0.857 ± 0.019** (range 0.838–0.884; descriptive, not a CI) |

The adaptive benchmark is stable across these five seeds — every run clears
0.83, above the earlier zero-shot and single-source values. The published 0.871
sits near the top of these five runs; the descriptive summary is **0.86 +/- 0.02 sample SD**, not a confidence interval.

**(b) Symmetric direction — held-out RIM-ONE (reverse: train HYGD+PAPILA):**

Same pipeline, roles swapped: RIM-ONE disease images were excluded from the
final training loss; VCDR was supervised on values derived from PAPILA's
dataset-provided two-grader contours (HYGD masked out). This historical selection
rule does not independently establish measurement validity or reliability. This
direction remains adaptive because the broader target history was visible.

| seed | held-out RIM-ONE AUROC (TTA) |
|---|---|
| 0 | 0.927 |
| 1 | 0.905 |
| 2 | 0.922 |
| 3 | 0.900 |
| 4 | 0.922 |
| **sample mean ± sample SD** | **0.915 ± 0.012** (range 0.900–0.927; descriptive, not a CI) |

The RIM-ONE adaptive benchmark changed from zero-shot 0.61 to a five-run descriptive mean of **0.915 +/- 0.012 sample SD**.
The bidirectional pattern is mechanism-supporting evidence, not proof of
prospective transportability or absence of target-informed development.

**Honest caveat.** Held-out RIM-ONE scores higher than held-out PAPILA. RIM-ONE DL
is distributed as tightly disc-cropped images and has higher glaucoma prevalence
(0.35 vs 0.21), both of which plausibly make it an easier held-out target. The
observation is **bidirectional adaptive recovery**, not that the two datasets are
equally hard or that transportability is established.

Historical execution references: `validation/train_generalize_vcdr.py --seed {0..4}`
(forward) and `validation/train_generalize_vcdr_reverse.py --seed {0..4}`
(reverse), aggregated by `validation/aggregate_seeds.py` and
`aggregate_seeds_reverse.py`. The current scripts are prospectively hardened and
write to a new patient-aware namespace; they do not retroactively validate the
historical summaries in `results/seed_robustness_attemptB.json` and
`results/seed_robustness_reverse.json`.
