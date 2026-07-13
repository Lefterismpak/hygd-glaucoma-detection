# HYGD-CEXT-2.0 Frozen-Feature Result

**Protocol:** `HYGD-CEXT-2.0`

**Candidate:** `F0 = dinov2_vits14 frozen CLS features + fixed logistic regression`

**Status:** closed - dataset-origin shortcut gate failed

**Promotion:** no

**Target access:** none

## Answer

The frozen full-frame DINOv2 representation improved disease discrimination across all three held-out sources and passed every predeclared AUROC and confidence-interval condition. It nevertheless retained almost perfect dataset identity: the four-fold group-aware dataset-origin probe reached **0.9994** accuracy against a `<0.75` gate. `HYGD-CEXT-2.0` therefore fails and closes without a fallback candidate.

This is a useful negative result. A stronger generic representation can raise source-only disease AUROC while remaining dominated by acquisition/source information. The result does not establish a transportable glaucoma model and does not authorize target access, clinical use, publication claims, fine-tuning, or model hopping.

## Locked Provenance

| Item | Frozen value |
|---|---|
| Official entrypoint | `dinov2_vits14` |
| Repository commit | `7764ea0f912e53c92e82eb78a2a1631e92725fc8` |
| Repository license | Apache License 2.0 |
| Weight bytes | `88,283,115` |
| Weight SHA-256 | `b938bf1bc15cd2ec0feacfe3a1bb553fe8ea9ca46a7e1d8d00217f29aef60cd9` |
| Protocol SHA-256 | `edea2a4c991c8b2e3c44ae23a3e26b601b82fa33ad889f7b8cd96b0865b4a721` |
| Execution-lock SHA-256 | `185e17cca2fa338003cca481123b7536b5fc47de23aa9d2982b10afec9de37a5` |

The official receipt was captured before first model load. The existing project environment was used without dependency changes. The official architecture was instantiated locally, the receipted state dictionary was loaded with `weights_only=True` and `strict=True`, and all 22,056,576 parameters matched without missing or unexpected keys.

## Reproducibility And Integrity

- The frozen audit reconciled 1,642 exact-hash representative images: HYGD 737, PAPILA 420, RIM-ONE 485.
- Every source image byte size and SHA-256 was checked before each extraction.
- Two fresh-process CPU extractions produced the exact same 384-dimensional float32 feature file SHA-256: `d99d9b056150d5e45ab6a007f22ea723e02fef1510c60f949b22c17154bf41a5`.
- All features were finite and L2-normalized.
- Three LOSO disease predictions, three fixed permutation-control predictions, and the origin-probe predictions had exact matching hashes on rerun.
- An independent script refitted all disease, permutation, and origin classifiers and reproduced every reported probability, AUROC, confidence interval, fold assignment, and final gate verdict.
- No training/evaluation hash or subject-cluster overlap was present.

## Disease Results

| Held-out source | F0 AUROC (95% cluster-bootstrap CI) | v1.1 Candidate A | Fixed permutation control |
|---|---:|---:|---:|
| HYGD | **0.6261** [0.5580, 0.6910] | 0.5220 | 0.5166 |
| PAPILA | **0.7596** [0.6957, 0.8172] | 0.5651 | 0.3820 |
| RIM-ONE | **0.7457** [0.6997, 0.7911] | 0.7811 | **0.7468** |
| Equal-source mean | **0.7105** [0.6746, 0.7423] | 0.6227 | not pooled |

All three source AUROCs were at least 0.60, the equal-source mean exceeded 0.70, and all three source CI lower bounds exceeded 0.50. The disease-performance portion of the gate therefore passed.

## Shortcut Gate

The fixed four-fold `StratifiedGroupKFold` origin probe produced:

- mean fold accuracy: **0.9993917**;
- pooled out-of-fold accuracy: **0.9993910**;
- majority chance: **0.4488429**;
- fold accuracies: `1.0000`, `1.0000`, `0.9976`, `1.0000`;
- required gate: `<0.75`.

The shortcut gate failed overwhelmingly. The DINOv2 features make the three datasets almost perfectly distinguishable.

## Negative-Control Caution

The fixed RIM-ONE permutation control reached AUROC **0.7468**, essentially the same as the disease model's **0.7457**, despite a low correlation between their image probabilities (`-0.0783`). This control is not a promotion gate and one fixed permutation cannot identify the mechanism. It is, however, independently reproducible and materially weakens any causal interpretation of the RIM-ONE disease AUROC. The exact cause remains `needs-proof`; the near-perfect origin probe makes residual source/acquisition structure the leading concern.

## Gate Verdict

| Predeclared condition | Result |
|---|---|
| Source audit and fold integrity | pass |
| Every source AUROC >=0.60 | pass |
| Equal-source mean AUROC >=0.70 | pass |
| At least two source CI lower bounds >0.50 | pass (3/3) |
| Dataset-origin accuracy <0.75 | **fail (0.9994)** |
| No protocol deviation | pass |

**Final verdict:** `failed_closed_no_fallback`.

## Decision

1. Preserve `HYGD-CEXT-2.0` as a completed negative qualification.
2. Do not fine-tune DINOv2, substitute DINOv3/RETFound, alter preprocessing, or consume a new target inside this protocol.
3. Keep Lock B blocked.
4. Use the result to strengthen the project's failure-first validation narrative: disease metrics alone can improve while source shortcut remains decisive.
5. Park further model development until a PI/mentor supplies a genuinely different scientific hypothesis, appropriate data, or a preregistered domain-invariance design.

## Evidence

- `HYGD_CEXT_2_0_PROTOCOL.md`
- `HYGD_CEXT_2_0_EXECUTION_SPEC.md`
- `validation/hygd_cext_2_0_protocol_lock.json`
- `validation/hygd_cext_2_0_execution_lock.json`
- `artifacts/dinov2_vits14_receipt/receipt.json`
- `validation/cext_2_0.py`
- `validation/audit_cext_2_0.py`
- `results/cext_2_0/feature_extraction_primary.json`
- `results/cext_2_0/feature_extraction_rerun.json`
- `results/cext_2_0/evaluation_primary/qualification_summary.json`
- `results/cext_2_0/evaluation_rerun/qualification_summary.json`
- `results/cext_2_0/independent_audit.json`
