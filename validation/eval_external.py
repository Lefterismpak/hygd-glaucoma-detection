"""Historical external stress-test evaluator, hardened for future local runs.

Input: a label CSV with columns image_path, patient_id, label (1=glaucoma).
Runs the frozen HYGD model (via validation/predict.py, parity-verified) zero-shot,
then reports an image/eye-level point AUROC and descriptive calibration metrics.
Patient-cluster uncertainty is emitted only when an operator supplies a
hash-bound image-to-subject map. That digest proves which mapping bytes were
used, not the biological truth of the mapping; subject independence therefore
remains ``needs-proof``. The historical 0.40 point is labelled test-derived and
nondeployment. Recalibration uses a deterministic group-stratified 30/70 split
and records the Platt direction instead of assuming monotonicity.
  - threshold re-selection targeting sensitivity >= 0.90 on the calibration split,
    specificity reported on the eval split.

Usage:
  python validation/eval_external.py --labels validation/data/papila_labels.csv --name PAPILA
"""

import argparse
import io
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from validation.predict import load_model, predict_probs  # noqa: E402
from validation.evaluation_utils import (  # noqa: E402
    assert_fresh_output_bundle,
    atomic_write_text,
    load_hash_bound_subject_map,
    patient_cluster_auc_ci,
    read_verified_file_bytes,
    resolve_private_output_path,
    stratified_group_holdout,
    validate_external_label_frame,
)

HYGD_THRESHOLD = 0.40
EPS = 1e-6


def _logit(p):
    p = np.clip(p, EPS, 1 - EPS)
    return np.log(p / (1 - p))


def sens_spec(y, p, thr):
    yp = (p >= thr).astype(int)
    tp = int(((yp == 1) & (y == 1)).sum()); fn = int(((yp == 0) & (y == 1)).sum())
    tn = int(((yp == 0) & (y == 0)).sum()); fp = int(((yp == 1) & (y == 0)).sum())
    sens = tp / (tp + fn) if (tp + fn) else float("nan")
    spec = tn / (tn + fp) if (tn + fp) else float("nan")
    return sens, spec, (tn, fp, fn, tp)


def ece(y, p, n_bins=10):
    """Expected Calibration Error."""
    bins = np.linspace(0, 1, n_bins + 1)
    e = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (p >= lo) & (p < hi) if hi < 1 else (p >= lo) & (p <= hi)
        if m.sum() == 0:
            continue
        e += (m.mean()) * abs(y[m].mean() - p[m].mean())
    return float(e)


def brier(y, p):
    return float(np.mean((p - y) ** 2))


def fit_temperature(logits, y, iters=200, lr=0.05):
    """1-param temperature scaling on the effective logit (minimize NLL)."""
    import torch
    z = torch.tensor(logits, dtype=torch.float64)
    t = torch.ones(1, dtype=torch.float64, requires_grad=True)
    yy = torch.tensor(y, dtype=torch.float64)
    opt = torch.optim.LBFGS([t], lr=lr, max_iter=iters)

    def closure():
        opt.zero_grad()
        p = torch.sigmoid(z / t.clamp_min(1e-3))
        nll = -(yy * torch.log(p + EPS) + (1 - yy) * torch.log(1 - p + EPS)).mean()
        nll.backward()
        return nll

    opt.step(closure)
    return max(float(t.detach().item()), 1e-3)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", required=True)
    ap.add_argument("--name", default="external")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--checkpoint-sha256", required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--subject-map")
    ap.add_argument("--subject-map-sha256")
    ap.add_argument("--out")
    args = ap.parse_args()
    if bool(args.subject_map) != bool(args.subject_map_sha256):
        raise ValueError("--subject-map and --subject-map-sha256 are required together")
    requested_out = (
        args.out or f"results/patient_aware/external_{args.name.lower()}_v2.json"
    )
    out = resolve_private_output_path(ROOT, requested_out, "results/patient_aware")
    assert_fresh_output_bundle([out])

    import pandas as pd
    labels_path = Path(args.labels)
    if not labels_path.is_absolute():
        labels_path = ROOT / labels_path
    label_bytes, labels_sha256 = read_verified_file_bytes(labels_path)
    df = pd.read_csv(io.BytesIO(label_bytes))
    df = validate_external_label_frame(df, context="External labels")
    y = df["label"].to_numpy(dtype=int)
    model = load_model(
        args.checkpoint,
        expected_checkpoint_sha256=args.checkpoint_sha256,
    )
    print(f"[{args.name}] {len(df)} images, {df['patient_id'].nunique()} supplied group IDs, "
          f"{y.mean():.1%} glaucoma prevalence")

    p = predict_probs([str(x) for x in df["image_path"]], model)
    z = _logit(p)

    # --- zero-shot discrimination + raw calibration on the full set ---
    auroc = float(roc_auc_score(y, p))
    group_ids = df["patient_id"].astype(str)
    mapping_sha256 = None
    if args.subject_map:
        stems = df["image_path"].map(lambda value: Path(value).stem)
        mapping, mapping_sha256 = load_hash_bound_subject_map(
            stems, args.subject_map, args.subject_map_sha256
        )
        group_ids = stems.map(mapping)
        uncertainty = patient_cluster_auc_ci(
            y,
            p,
            group_ids.astype(str).to_numpy(),
            n_bootstrap=2000,
            seed=args.seed,
        )
        uncertainty.update(
            {
                "status": "operator_asserted_hash_bound_subject_map_needs_proof",
                "canonical_or_confirmatory": False,
                "subject_mapping_sha256": mapping_sha256,
            }
        )
    else:
        uncertainty = {
            "status": "not_computed_subject_independence_needs_proof",
            "point_estimand_unit": "image_or_eye_row",
            "resampling_unit": "verified_subject_id_unavailable",
            "canonical_or_confirmatory": False,
        }
    auprc = float(average_precision_score(y, p))
    sens, spec, cm = sens_spec(y, p, HYGD_THRESHOLD)
    res = {
        "_evidence_status": {
            "status": "operator_specified_external_stress_test_nonconfirmatory",
            "input_selection": "operator_specified_at_runtime",
            "transportability_established": False,
            "full_protocol_compliance": False,
            "license_compatibility": "needs-proof",
            "external_claim_permitted": False,
        },
        "dataset": args.name, "n_images_or_eyes": int(len(y)),
        "n_supplied_group_ids": int(df["patient_id"].nunique()),
        "labels_csv_sha256": labels_sha256,
        "checkpoint_sha256": model.checkpoint_sha256,
        "subject_independence": "needs-proof",
        "subject_mapping_status": (
            "operator_asserted_hash_bound_needs_independent_proof"
            if mapping_sha256
            else "unavailable"
        ),
        "prevalence": float(y.mean()),
        "zero_shot": {
            "image_or_eye_level_auroc": round(auroc, 4),
            "patient_cluster_uncertainty": uncertainty,
            "auprc": round(auprc, 4),
            "historical_test_derived_threshold_0_40_nondeployment": {"sensitivity": round(sens, 4), "specificity": round(spec, 4),
                        "confusion_tn_fp_fn_tp": cm},
            "ece_raw": round(ece(y, p), 4), "brier_raw": round(brier(y, p), 4),
        },
    }

    # --- group-disjoint and group-stratified 30% calibration / 70% evaluation ---
    split_frame = df.assign(_analysis_group=group_ids.astype(str).to_numpy())
    cal_idx, eval_idx = stratified_group_holdout(
        split_frame, "_analysis_group", "label", test_fraction=0.70, seed=args.seed
    )
    if set(split_frame.iloc[cal_idx]["_analysis_group"]) & set(
        split_frame.iloc[eval_idx]["_analysis_group"]
    ):
        raise AssertionError("Calibration and evaluation groups overlap")
    yc, zc = y[cal_idx], z[cal_idx]
    ye, ze, pe = y[eval_idx], z[eval_idx], p[eval_idx]

    recal = {
        "split": {
            "seed": args.seed,
            "group_stratified": True,
            "calibration_rows": int(len(cal_idx)),
            "evaluation_rows": int(len(eval_idx)),
            "group_overlap": 0,
        }
    }
    T = fit_temperature(zc, yc)
    pe_temp = 1 / (1 + np.exp(-ze / T))
    from sklearn.linear_model import LogisticRegression
    platt = LogisticRegression().fit(zc.reshape(-1, 1), yc)
    platt_coefficient = float(platt.coef_[0, 0])
    pe_platt = platt.predict_proba(ze.reshape(-1, 1))[:, 1]

    # threshold on calibration split targeting sensitivity >= 0.90 (raw probs)
    cal_p = p[cal_idx]
    thrs = np.unique(cal_p)
    chosen = HYGD_THRESHOLD
    for t in sorted(thrs):
        s, _, _ = sens_spec(yc, cal_p, t)
        if s >= 0.90:
            chosen = float(t)
    s_eval, sp_eval, _ = sens_spec(ye, pe, chosen)

    recal.update({
        "temperature": round(T, 3),
        "platt_coefficient": platt_coefficient,
        "platt_intercept": float(platt.intercept_[0]),
        "platt_strictly_increasing": platt_coefficient > 0,
        "auroc_invariance_applicable": platt_coefficient > 0,
        "eval_ece_raw": round(ece(ye, pe), 4),
        "eval_ece_temp": round(ece(ye, pe_temp), 4),
        "eval_ece_platt": round(ece(ye, pe_platt), 4),
        "eval_brier_raw": round(brier(ye, pe), 4),
        "eval_brier_temp": round(brier(ye, pe_temp), 4),
        "threshold_selected_on_calibration_split_targeting_sensitivity_gte_0_90": {
            "threshold": round(chosen, 3),
            "target_sensitivity": 0.9,
            "eval_sensitivity": round(s_eval, 4),
            "eval_specificity": round(sp_eval, 4),
            "target_met_on_evaluation_split": bool(s_eval >= 0.90),
        },
    })
    res["recalibration"] = recal

    atomic_write_text(out, json.dumps(res, indent=2) + "\n")
    print(json.dumps(res, indent=2))
    print(f"\nsaved {out}")


if __name__ == "__main__":
    main()
