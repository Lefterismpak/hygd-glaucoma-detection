# HYGD Failure-First Glaucoma AI Audit

**Status:** reviewer brief; claim hierarchy updated 2026-09-06

**New:** [September reassessment](HYGD_REASSESSMENT_2026_09.md) — complete v5 model execution, descriptive probability checks, a measured permutation-structure defect and reproducible control diagnostics.

**Research conclusion:** the project demonstrates strong discrimination inside one
single-site fundus dataset, but it does not establish transportability to a new
clinic or camera. The main contribution is the evaluation and audit sequence
that exposed this boundary, prevented weak candidates from being promoted, and
preserved the negative results.

This is a student research and portfolio artifact. It is not a clinical device
and must not be used for diagnosis, screening, or treatment decisions.

## Research Question

Can a glaucoma classifier trained on public fundus datasets retain clinically
relevant disease signal while avoiding dataset- and acquisition-specific
shortcuts?

The work began as a reproducible ResNet18 baseline on the Hillel Yaffe Glaucoma
Dataset (HYGD). Later evaluation changed the question from "how high is the
AUROC?" to "what evidence would justify trusting the signal outside the source
dataset?" That required duplicate-aware internal evaluation, explicit
development-versus-confirmatory boundaries, source-only leave-one-dataset-out
tests, dataset-origin probes, permutation controls, frozen protocols, and stop
rules.

## Why The Evaluation Changed

The original project used patient-grouped splits and reported high HYGD
performance. A later audit found three reasons to narrow that interpretation:

1. The historical cross-validation reused each fold for checkpoint selection
   and scoring, and the configuration had already been selected after a
   development test-set comparison.
2. HYGD contained exact duplicate images under different supplied patient IDs.
   A repaired protocol linked those IDs, counted each hash once, separated
   inner validation from outer scoring, and stored one out-of-fold prediction
   per linked evaluation group; biological independence remains unproved.
3. Attractive PAPILA and RIM-ONE recovery results came from an adaptive
   development process. Target AUROC was displayed during development, and
   target anatomical resources affected preprocessing. Those results are useful
   historical evidence, but they are not prospectively untouched external
   validation.

The project then moved to locked source-only qualifications. A geometry-based
candidate was stopped before glaucoma-classifier training when held-out optic
disc scale transfer failed. A later 36-image exploratory manual audit passed its
predeclared center-error criterion but failed diameter and combined criteria;
the original diameter target was synthetic. Same-sample scale correction did
not pass the predeclared dataset-origin gate. A frozen
DINOv2 representation improved cross-source disease AUROC, yet made dataset
origin almost perfectly decodable. Fixed image-space preprocessing did not
remove that signal.

## Decision-Relevant Evidence

| Stage | Result | Current interpretation |
| --- | --- | --- |
| Historical HYGD development | Test AUROC 0.976; CV AUROC 0.988 +/- 0.008 | Superseded as the preferred internal estimate |
| Complete v5 execution (2026-09-06) | Mean outer-fold group AUROC **0.9908 [0.9790–0.9990]**; sensitivity **0.9781**, specificity **0.9700** | Current fixed recipe actually executed; internal post-development evidence, not a new population |
| Previous repaired HYGD reanalysis | Mean outer-fold duplicate-aware group AUROC **0.9908**, conditional 95% CI **0.9790-0.9990**, 283 groups | Read-only v5 aggregate reanalysis of frozen private OOF predictions from a run asserted to date from 2026-07-11; no contemporaneous public timestamp, retraining, or end-to-end v5 model run |
| Historical adaptive recovery | PAPILA **0.857 +/- 0.019**; RIM-ONE **0.915 +/- 0.012** | Five-run sample mean +/- SD, not CIs; target-adaptive development, noncanonical per-run intervals, subject/license boundaries `needs-proof` |
| HYGD-CEXT-1.1 | Equal-source mean AUROC **0.6227**, 95% CI **0.5822-0.6632** | Failed source-only qualification; confirmatory target-access gate ("Lock B") blocked |
| HYGD-CEXT-2.0 | Equal-source mean AUROC **0.7105**, 95% CI **0.6746-0.7423**; origin accuracy **0.9994** | Disease gate passed; decisive shortcut gate failed |
| Shortcut S2 | Equal-source mean AUROC **0.7364**; origin accuracy **0.9933** | Fixed preprocessing did not repair source decoding |
| Cross-fitted LEACE | Equal-source mean AUROC **0.7096**; origin accuracy **0.4488** | Mechanism evidence only, not a deployable repair |
| RIM-ONE permutation controls | AUROC **0.6759-0.7513** across later branches | September audit: row shuffling breaks group label structure; high-score mechanism still `needs-proof` |

The critical pattern is not that every disease AUROC collapsed. It is that
disease discrimination could improve while source identity remained nearly
perfectly encoded. AUROC alone was therefore insufficient for promotion.

The adaptive-recovery row is retained only to document the development
chronology. Its historical PAPILA intervals resampled eyes rather than patients,
and RIM-ONE subject independence is not established by image stems. Compatibility
of the mixed-dataset training recipe with the RIM-ONE DL use terms remains
`needs-proof`; those values should not support a paper, competition, or external
performance claim without written clarification.

## What The Evidence Supports

- The model strongly discriminates HYGD dataset labels under duplicate-aware,
  linked-group resampling; the biological and acquisition contributions are not isolated.
- Single-site ranking performance can remain high after repairing known split,
  duplicate, and threshold-selection defects.
- Acquisition and dataset identity are strongly represented in both the
  baseline and frozen DINOv2 feature spaces.
- A high source-validation segmentation Dice does not guarantee held-out
  anatomical scale transfer.
- Linear concept erasure suggests that part of the source signal is separable
  from the measured disease signal in the frozen representation.
- Frozen gates, independent refits, exact artifact hashes, negative controls,
  and explicit stop decisions materially improved the honesty of the project.

## What It Does Not Support

- It does not show that the classifier transports to a new hospital, camera,
  population, or prevalence setting.
- It does not validate a fixed probability threshold or demonstrate calibrated
  risk estimates.
- It does not establish clinical utility, patient benefit, safety, fairness, or
  deployment readiness.
- LEACE does not identify a causal image feature and does not convert the
  representation into a qualified glaucoma model.
- The high RIM-ONE permutation AUROC remains unexplained. Until its mechanism is
  resolved, the corresponding disease AUROCs cannot support a causal or
  clinical claim.
- Historical adaptive recovery does not become confirmatory evidence merely
  because it was robust across seeds or evaluated in two directions.

## My Contribution

I directed and audited an AI-assisted medical-imaging research workflow: I
defined the questions and stop rules, required duplicate- and patient-aware
evaluation, separated development evidence from confirmatory evidence,
introduced source-origin and permutation controls, and preserved negative
results when promotion gates failed. This repository does not claim that I
independently invented a new generalizable glaucoma model.

The practical work included reconciling duplicated records, freezing protocols
before evaluation, specifying source-only folds and promotion gates, checking
held-out geometry, separating inner selection from outer scoring, requiring
independent recomputation, and documenting why a numerically attractive branch
was not promoted.

## Questions For A PI Or Clinical Reviewer

1. Which controlled intervention and subject-exchangeable null can test whether
   the disease predictor actually relies on a nuisance factor, beyond decoding source identity?
2. Which independent dataset, evaluation unit, and clinically meaningful
   endpoint would make a next study decision-useful rather than another
   benchmark exercise?
3. Which components require ophthalmologist adjudication or PI collaboration
   before this could become a paper-level study?

## Current Stop Rule And Revisit Trigger

The September internal rerun and retrospective diagnostics are complete; the historical qualification gates remain closed. Further model optimization needs a separately designed study. It should reopen only when at least one of
the following becomes available:

- a properly licensed retina-specific or general foundation model with
  reproducible cross-domain glaucoma evidence;
- a new independent dataset and a predeclared untouched-target protocol; or
- PI or clinical guidance that supplies a different scientific hypothesis,
  appropriate data, or a clinically meaningful evaluation design.

A newer model name or a higher benchmark score alone is not enough. Any new
lane needs a separate frozen protocol, explicit data and license boundaries,
and a promotion gate that tests more than disease AUROC.

## Evidence Map

The public repository now includes the evaluator, synthetic integrity tests,
historical aggregate artifacts with explicit evidence labels, and a privacy-safe
aggregate receipt. Private row-level predictions, checkpoints, data, and run
bundles remain unpublished. Their SHA-256 commitments do not make the numerical
result publicly reproducible or prove a contemporaneous 2026-07-11 timestamp.

- [September reassessment and next-study recommendation](HYGD_REASSESSMENT_2026_09.md)
- [Complete v5 run aggregate receipt](results/v5_complete_run_20260906.json)
- [Internal evaluation repair](INTERNAL_EVALUATION_REPAIR.md)
- [Aggregate-only internal evidence receipt](results/repaired_internal_evaluation_summary.json)
- [Source-only qualification report](SOURCE_ONLY_QUALIFICATION_REPORT.md)
- [Manual geometry result](HYGD_MANUAL_GEOMETRY_RESULT.md)
- [HYGD-CEXT-2.0 protocol](HYGD_CEXT_2_0_PROTOCOL.md)
- [HYGD-CEXT-2.0 execution specification](HYGD_CEXT_2_0_EXECUTION_SPEC.md)
- [HYGD-CEXT-2.0 result](HYGD_CEXT_2_0_RESULT.md)
- [Shortcut Map 1 protocol](HYGD_SHORTCUT_MAP_1_PROTOCOL.md)
- [Shortcut Map 1 result](HYGD_SHORTCUT_MAP_1_RESULT.md)
