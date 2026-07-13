# HYGD-SHORTCUT-MAP-1 Result

**Completed:** 2026-07-13

**Status:** `closed_no_preprocessing_repair_candidate`

**Scope:** source-shortcut mechanism attribution only

**Promotion authorized:** no

## Adjudicated Conclusion

Neither frozen image-space branch materially reduced dataset-origin decoding. `S1` and `S2` preserved the equal-source disease signal, but source identity remained almost perfectly recoverable. The fixed full-frame preprocessing repair lane therefore closes under the predeclared protocol.

LEACE reduced the fixed linear origin probe to approximately the source-majority level while preserving most of the disease AUROC. This supports a strong, linearly encoded source component that is partly separable from the measured disease signal in the frozen representation. It is mechanism evidence only: it does not identify a causal image feature, establish unseen-domain transportability, reopen `HYGD-CEXT-2.0`, authorize Lock B, or produce a deployable model.

## Frozen Results

| Branch | HYGD AUROC | PAPILA AUROC | RIM-ONE AUROC | Equal-source mean AUROC [95% CI] | Origin mean-fold accuracy | Decision |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| S0 CEXT baseline | 0.6261 | 0.7596 | 0.7457 | 0.7105 [0.6746, 0.7423] | 0.9994 | Closed CEXT baseline |
| S1 retinal-support square | 0.5859 | 0.7569 | 0.7787 | 0.7072 [0.6697, 0.7415] | 0.9982 | Failed origin gate |
| S2 S1 + grayscale standardization | 0.6799 | 0.7598 | 0.7694 | 0.7364 [0.7022, 0.7707] | 0.9933 | Failed origin gate |
| S3-CF LEACE | 0.6068 | 0.7599 | 0.7620 | 0.7096 [0.6748, 0.7417] | 0.4488 | Attribution only |
| S3-GLOBAL LEACE | 0.6159 | 0.7458 | 0.7435 | 0.7017 [0.6654, 0.7340] | 0.4367 | Transductive attribution only |

The preprocessing gate required origin accuracy `<= 0.8993917274939172` and equal-source mean AUROC `>= 0.6804536193472292`. S1 and S2 passed only the AUROC-preservation condition. No bounded preprocessing repair candidate was identified.

## Negative-Control Warning

RIM-ONE permutation AUROC remained high across branches: S1 0.7081, S2 0.6759, S3-CF 0.7513, and S3-GLOBAL 0.7439. Its exact mechanism remains `needs-proof`. These controls limit interpretation of the RIM-ONE disease AUROCs and are one reason not to promote any branch.

## Runtime Correction

The first execution lock copied PyTorch `2.12.1` from the earlier CEXT receipt instead of inspecting the unchanged current runtime. The first S1 feature pass exposed PyTorch `2.13.0` before any diagnostic metric was evaluated. That pass is retained under `results/shortcut_map_1/pre_metric_runtime_smoke_s1/` and excluded from counted primary/rerun claims.

The lock was corrected additively, no dependency was installed or changed, and the excluded smoke feature hash exactly matched the counted S1 hash: `ff537a3f71d456d36b7a14841db604d0ca2d30e2c344c5d0daad469225172537`.

## Verification

- S1 primary/rerun feature files matched exactly at SHA-256 `ff537a3f71d456d36b7a14841db604d0ca2d30e2c344c5d0daad469225172537`.
- S2 primary/rerun feature files matched exactly at SHA-256 `4c43337ce46539ffee47f0358dd0a7f6bf17028e927f270fd6a9f87ce9dc53b2`.
- All S1/S2 disease, permutation, and origin prediction artifacts matched their reruns exactly.
- The independent auditor refit all disease, permutation, origin, cross-fitted LEACE, and global LEACE analyses; reproduced the confidence intervals, leakage checks, runtime correction, and closed status; and wrote `results/shortcut_map_1/independent_audit.json` with `status: passed`.
- Execution lock SHA-256: `fc50cb21b0a66993b421993886a2ccd2d80d1453c1da9e09eb4cf29ae3c813f7`.

The auditor emitted only a scikit-learn future-deprecation warning for the explicit logistic-regression `penalty` argument. The frozen implementation is unchanged; compatibility with a future scikit-learn release remains a maintenance `needs-proof`, not a result deviation.

## Boundaries And Next Decision

- `HYGD-CEXT-2.0` remains `failed_closed_no_fallback`; Lock B remains blocked.
- No target, target-time adaptation, fine-tuning, neural training, new model, new weight, new repository, dependency change, account action, submission, commit, or push occurred.
- Do not iterate image thresholds, add transforms, sweep backbones, or reinterpret these results inside this protocol.
- A future anatomy-standardized lane would require a separately frozen protocol, explicit approval for any new weights/license exposure, reliable held-out optic-disc geometry, and more suitable source-domain evidence. Its benefit remains `needs-proof`.

**Next decision:** park further model experimentation and convert the failure-first result into a concise PI/mentor review packet before considering a genuinely different, separately approved protocol.
