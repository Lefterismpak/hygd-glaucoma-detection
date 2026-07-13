# HYGD-CEXT-1.1 Source-Only Qualification Report

Date: 2026-07-12

Status: **complete; independently audited; transportability gate failed**

Target access: **none**

## Decision

The locked source-only qualification did not produce a pipeline that is credible enough to expose to a new confirmatory target.

- Candidate B is **ineligible** because its source-trained optic-disc localizer failed predeclared geometry-transfer gates in two of three leave-one-source-out folds. No Candidate B glaucoma classifier was trained.
- Candidate A is the formal v1.1 selection only because Candidate B became ineligible. Candidate A is **not promoted**: two of three held-out sources were close to chance and the pooled high-sensitivity operating point had unusably low specificity.
- Lock B is therefore **blocked**. No target dataset, target labels, account action, download, registration, leaderboard result, or submission was accessed.
- HYGD-CEXT-1.1 is closed without post-result tuning. Any further work requires a separately frozen v2.0 protocol.

This is a useful negative result: the gate prevented a weak source-only pipeline from consuming a genuinely new target.

## Frozen Protocol Identity

| Item | SHA-256 |
|---|---|
| Base protocol v1.0 | `ca74f726ff841177462d3ecef325fac6f05677d0cacaf67fd488b940ad2aa7a3` |
| Addendum v1.1 | `667483f7d0fcbc36832c0dda48c304d84b0b373aaeb09a41dbe9afd148978f97` |
| Combined protocol identity | `51b0fcfab7daabba4b95b073d722374044f11c51c70bb0ed6b30850777160c84` |

The addendum was frozen before qualification because PAPILA has seven patients with discordant eye-level labels and because Candidate B needed an exact localizer recipe and numeric geometry gates. The base v1.0 file remains unchanged.

## Source Audit

| Source | Supplied rows | Unique hashes | Evaluation units / clusters | Locked evaluation level |
|---|---:|---:|---:|---|
| HYGD | 747 | 737 | 283 / 283 | linked subject-group |
| PAPILA | 420 | 420 | 420 eyes / 210 patients | eye AUROC; patient-cluster bootstrap |
| RIM-ONE DL | 485 | 485 | 485 / 485 | image AUROC; true subject independence `needs-proof` |
| **Total** | **1,652** | **1,642** | **978 clusters** | source-specific |

The audit found 10 exact-duplicate groups, all inside HYGD, and zero cross-dataset duplicate groups. Fold manifests have zero image-hash or subject-cluster overlap between training and held-out evaluation data. Source of truth: `results/confirmatory/source_audit.json`.

## Candidate B: Geometry Qualification

Candidate B used the v1.1 U-Net localizer and could proceed to classifier training only if every held-out fold passed its geometry gates.

| Held-out source | Best source-val Dice | Held-out geometry result | Decision |
|---|---:|---|---|
| HYGD | 0.9512 | coverage 0.9796; median diameter fraction 0.1633; plausible fraction 0.9917 | automatic gates passed; visual gate not needed after other folds failed |
| PAPILA | 0.9567 | center pass 0.9524, but median diameter ratio 4.4685 and combined pass 0.0000 | fail |
| RIM-ONE | 0.9557 | center pass 1.0000, but median diameter ratio 0.5061 and combined pass 0.5093 | fail |

High in-source segmentation Dice did not guarantee held-out scale transfer. Candidate B is therefore ineligible and `classifier_training_allowed` is false. Source of truth: `results/confirmatory/candidate_B/geometry_summary.json` and the three fold-level `geometry_qc.json` files.

## Candidate A: Full-Frame Qualification

Candidate A completed the locked 15-run grid: three held-out sources, five seeds each, 20 epochs, SWA, and fixed five-view TTA.

| Held-out source | AUROC | 95% cluster-bootstrap CI | Evaluation level |
|---|---:|---:|---|
| HYGD | 0.5220 | 0.4537-0.5901 | linked subject-group |
| PAPILA | 0.5651 | 0.4727-0.6588 | eye, clustered by patient |
| RIM-ONE | 0.7811 | 0.7347-0.8242 | image; subject independence `needs-proof` |
| **Mean across sources** | **0.6227** | **0.5822-0.6632** | equal source mean |

The locked source-derived threshold was `0.265418`: sensitivity `0.9502`, specificity `0.1609`. This is not a clinically usable operating point and no calibration was fitted.

Candidate A is selected **only by the predeclared fallback rule** after Candidate B became ineligible. Selection is not promotion: the source evidence does not justify Lock B or exposure to a new target.

## Independent Verification

The independent audit passed:

- 15 of 15 checkpoints load and match their recorded SHA-256 values.
- All prediction rows, labels, image hashes, and source clusters match the locked fold manifests.
- No training/evaluation hash or cluster leakage was found.
- AUROCs, confidence intervals, equal-source mean, and threshold were independently recomputed to numerical equality.
- Candidate B classifier artifact count is zero.
- Recorded production runtime is 47,481.478 seconds (about 13.19 hours).

Audit source: `results/confirmatory/independent_audit.json`.

## Failure Analysis

The weak performance is stable across seeds rather than being one unlucky initialization. HYGD and PAPILA ensemble probabilities show little label separation, while seed predictions remain highly correlated. RIM-ONE transfers better, but it is distributed in a much more disc-focused format than the two full-frame sources, so the equal-source mean hides substantial heterogeneity.

The localizer results reveal the same failure from another angle: excellent source Dice coexists with catastrophic held-out diameter scaling. The bottleneck is representation and acquisition-domain transport, not a threshold adjustment.

Therefore the defensible project claim is now:

> The repaired HYGD evaluation shows excellent in-distribution discrimination; the earlier PAPILA/RIM-ONE recovery is anatomy-assisted adaptive evidence; and the first locked source-only leave-one-source-out qualification failed to establish transportability.

## Out-of-Protocol Artifact Boundary

`results/source_only_localizer_forward_manual_source_prep.json` belongs to a later exploratory manual-source recipe change. Its initial HYGD center-QC failure compared against historical HYGD coordinates produced by an older non-HYGD U-Net, not anatomical ground truth, so that center verdict was rejected. Manual clicks support centering on the small adjudicated subset, but true disc diameter remains `needs-proof`, 19/747 HYGD predictions are missing, and dataset-probe accuracy remains 0.9563 versus 0.4541 majority/chance. No classifier was run.

`validation/HYGD_BOUNDARY_ADJUDICATION_V1.md` now freezes a separate blinded 36-image, subject-disjoint manual center-and-boundary gate (`HYGD-MANUAL-GEOM-1`). It is post-lock exploratory source geometry only, leaves `classifier_authorized=false`, and cannot modify or rescue HYGD-CEXT-1.1.

## Evidence-Informed v2.0 Options

No v2.0 classifier/backbone work is authorized or frozen yet. Complete the cheaper manual anatomy gate first if the project owner chooses to continue this lane.

1. **Preferred bounded feasibility sprint:** compare a license-light DINOv2 backbone with the current full-frame baseline under the same source-only leave-one-source-out folds. Start with frozen features/linear probing, then permit one tightly specified fine-tuning recipe only if the predeclared gate is met. DINOv2 code and pretrained models are published under Apache 2.0.
2. **Research-only comparator:** RETFound is retina-specific and is a reasonable scientific comparator, but its official repository is CC BY-NC 4.0. That restriction must remain explicit; it is not a clean commercial foundation. RetBench's glaucoma recommendation for DINORET is useful source-informed evidence, not sufficient reason to choose a model blindly.
3. **Anatomy route:** replace the plain source U-Net with a domain-generalized optic-disc/cup segmentation design, then re-run geometry qualification before any disease classifier. DoFE and newer retinal foundation-segmentation work support this direction, but they do not remove dataset, license, or external-validation requirements.

Official/primary references:

- RETFound paper: https://www.nature.com/articles/s41586-023-06555-x
- RETFound repository and license: https://github.com/rmaphoh/RETFound and https://github.com/rmaphoh/RETFound/blob/main/LICENSE
- DINOv2 repository: https://github.com/facebookresearch/dinov2
- RetBench record: https://discovery.ucl.ac.uk/id/eprint/10220057/
- Foundation-model demographic generalizability study: https://discovery.ucl.ac.uk/id/eprint/10203510/
- DoFE: https://arxiv.org/abs/2010.06208
- FunduSegmenter: https://pmc.ncbi.nlm.nih.gov/articles/PMC13206755/

## Post-Lock Geometry Resolution And Completed Next Gate

The separately frozen 36-image blinded boundary annotation is complete. `HYGD-MANUAL-GEOM-1` passed center localization but failed diameter and combined geometry; the original 100 manual-source masks used a synthetic diameter fixed at 25% of image size. A post-hoc scalar repaired same-sample geometry but left dataset-origin accuracy at 0.9420, so the scalar-only lane is closed and no classifier is authorized. See `HYGD_MANUAL_GEOMETRY_RESULT.md`.

The separately frozen `HYGD-CEXT-2.0` test was subsequently approved and completed without target access. Frozen full-frame DINOv2-S/14 features improved equal-source mean AUROC to 0.7105 [0.6746, 0.7423] and passed the disease gates, but dataset-origin accuracy was 0.9994 against a required `<0.75`. The independent audit reproduced the result. v2.0 is closed without fallback and Lock B remains blocked; see `HYGD_CEXT_2_0_RESULT.md`.
