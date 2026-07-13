# HYGD Failure-First Glaucoma AI Audit

**Status:** reviewer brief, 2026-07-14

**Bottom line:** the project demonstrates strong discrimination inside one
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
   per independent evaluation group.
3. Attractive PAPILA and RIM-ONE recovery results came from an adaptive
   development process. Target AUROC was displayed during development, and
   target anatomical resources affected preprocessing. Those results are useful
   historical evidence, but they are not prospectively untouched external
   validation.

The project then moved to locked source-only qualifications. A geometry-based
candidate was stopped before glaucoma-classifier training when held-out optic
disc scale transfer failed. A later manual audit showed that its center
localization was accurate but its diameter target was synthetic; correcting the
scale on the same sample still left dataset-origin accuracy at 0.9420. A frozen
DINOv2 representation improved cross-source disease AUROC, yet made dataset
origin almost perfectly decodable. Fixed image-space preprocessing did not
remove that signal.

## Decision-Relevant Evidence

| Stage | Result | Current interpretation |
| --- | --- | --- |
| Historical HYGD development | Test AUROC 0.976; CV AUROC 0.988 +/- 0.008 | Superseded as the preferred internal estimate |
| Repaired HYGD evaluation | Duplicate-aware group AUROC **0.9904**, 95% CI **0.9797-0.9980**, 283 groups | Strong single-site discrimination; post-development resampling only |
| Historical adaptive recovery | PAPILA **0.857 +/- 0.019**; RIM-ONE **0.915 +/- 0.012** | Development evidence, not untouched validation or current transportability proof |
| HYGD-CEXT-1.1 | Equal-source mean AUROC **0.6227**, 95% CI **0.5822-0.6632** | Failed source-only qualification; confirmatory target-access gate ("Lock B") blocked |
| HYGD-CEXT-2.0 | Equal-source mean AUROC **0.7105**, 95% CI **0.6746-0.7423**; origin accuracy **0.9994** | Disease gate passed; decisive shortcut gate failed |
| Shortcut S2 | Equal-source mean AUROC **0.7364**; origin accuracy **0.9933** | Fixed preprocessing did not repair source decoding |
| Cross-fitted LEACE | Equal-source mean AUROC **0.7096**; origin accuracy **0.4488** | Mechanism evidence only, not a deployable repair |
| RIM-ONE permutation controls | AUROC **0.6759-0.7513** across later branches | Mechanism `needs-proof`; limits disease-AUROC interpretation |

The critical pattern is not that every disease AUROC collapsed. It is that
disease discrimination could improve while source identity remained nearly
perfectly encoded. AUROC alone was therefore insufficient for promotion.

The adaptive-recovery row is retained only to document the development
chronology. Compatibility of the mixed-dataset training recipe with the
RIM-ONE DL use terms remains `needs-proof`; those values should not support a
paper, competition, or external performance claim without written
clarification.

## What The Evidence Supports

- HYGD contains a strong in-distribution glaucoma signal under duplicate-aware,
  linked-group resampling.
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

1. Given near-perfect source decoding, which anatomy-aware or domain-invariance
   hypothesis is scientifically worth freezing next?
2. Which independent dataset, evaluation unit, and clinically meaningful
   endpoint would make a next study decision-useful rather than another
   benchmark exercise?
3. Which components require ophthalmologist adjudication or PI collaboration
   before this could become a paper-level study?

## Current Stop Rule And Revisit Trigger

Further model iteration is parked. It should reopen only when at least one of
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

This first public snapshot contains the reviewer brief and narrative audit
reports only. Lower-level scripts, locks, predictions, and aggregate result
files named inside those reports are provenance references from the full
audited workspace; they are not included here and are not claimed to be
publicly reproducible from this snapshot alone.

- [Internal evaluation repair](INTERNAL_EVALUATION_REPAIR.md)
- [Source-only qualification report](SOURCE_ONLY_QUALIFICATION_REPORT.md)
- [Manual geometry result](HYGD_MANUAL_GEOMETRY_RESULT.md)
- [HYGD-CEXT-2.0 protocol](HYGD_CEXT_2_0_PROTOCOL.md)
- [HYGD-CEXT-2.0 execution specification](HYGD_CEXT_2_0_EXECUTION_SPEC.md)
- [HYGD-CEXT-2.0 result](HYGD_CEXT_2_0_RESULT.md)
- [Shortcut Map 1 protocol](HYGD_SHORTCUT_MAP_1_PROTOCOL.md)
- [Shortcut Map 1 result](HYGD_SHORTCUT_MAP_1_RESULT.md)
