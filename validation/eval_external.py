"""External-validation evaluation (GlaucoGen Steps 2-4), dataset-agnostic.

Input: a label CSV with columns image_path, patient_id, label (1=glaucoma).
Runs the frozen HYGD model (via validation/predict.py, parity-verified) zero-shot,
then reports — per PROTOCOL.md —:
  - discrimination: AUROC + bootstrap 95% CI, AUPRC
  - operating point: sensitivity/specificity at the HYGD screening threshold 0.40
  - calibration (the expected failure under base-rate shift): ECE, Brier, reliability
  - recalibration: temperature + Platt fit on a patient-grouped 30% split,
    calibration re-measured on the held-out 70% (AUROC is invariant to this — stated)
  - threshold re-selection targeting sensitivity >= 0.90 on the calibration split,
    specificity reported on the eval split.

Usage:
  python validation/eval_external.py --labels validation/data/papila_labels.csv --name PAPILA
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import GroupShuffleSplit

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from validation.predict import load_model, predict_probs  # noqa: E402

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


def bootstrap_auc(y, p, n=2000, seed=42):
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n):
        idx = rng.integers(0, len(y), len(y))
        if len(np.unique(y[idx])) < 2:
            continue
        vals.append(roc_auc_score(y[idx], p[idx]))
    return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]


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
    return float(t.detach().item())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", required=True)
    ap.add_argument("--name", default="external")
    ap.add_argument("--checkpoint", default=str(ROOT / "results/finetune_layer4_aug.pt"))
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    import pandas as pd
    df = pd.read_csv(args.labels)
    y = df["label"].astype(int).values
    model = load_model(args.checkpoint)
    print(f"[{args.name}] {len(df)} images, {df['patient_id'].nunique()} patients, "
          f"{y.mean():.1%} glaucoma prevalence")

    p = predict_probs([str(x) for x in df["image_path"]], model)
    z = _logit(p)

    # --- zero-shot discrimination + raw calibration on the full set ---
    auroc = float(roc_auc_score(y, p))
    ci = bootstrap_auc(y, p, seed=args.seed)
    auprc = float(average_precision_score(y, p))
    sens, spec, cm = sens_spec(y, p, HYGD_THRESHOLD)
    res = {
        "dataset": args.name, "n": int(len(y)), "n_patients": int(df["patient_id"].nunique()),
        "prevalence": float(y.mean()),
        "zero_shot": {
            "auroc": round(auroc, 4), "auroc_95ci": [round(c, 4) for c in ci],
            "auprc": round(auprc, 4),
            "at_0.40": {"sensitivity": round(sens, 4), "specificity": round(spec, 4),
                        "confusion_tn_fp_fn_tp": cm},
            "ece_raw": round(ece(y, p), 4), "brier_raw": round(brier(y, p), 4),
        },
    }

    # --- patient-grouped 30% calibration split for recalibration + threshold ---
    gss = GroupShuffleSplit(n_splits=1, test_size=0.70, random_state=args.seed)
    cal_idx, eval_idx = next(gss.split(df, groups=df["patient_id"].values))
    yc, zc = y[cal_idx], z[cal_idx]
    ye, ze, pe = y[eval_idx], z[eval_idx], p[eval_idx]

    recal = {"note": "AUROC is invariant to monotone recalibration — the win here is calibration/threshold, not discrimination."}
    if len(np.unique(yc)) == 2 and len(np.unique(ye)) == 2:
        T = fit_temperature(zc, yc)
        pe_temp = 1 / (1 + np.exp(-ze / T))
        from sklearn.linear_model import LogisticRegression
        platt = LogisticRegression().fit(zc.reshape(-1, 1), yc)
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
            "eval_ece_raw": round(ece(ye, pe), 4),
            "eval_ece_temp": round(ece(ye, pe_temp), 4),
            "eval_ece_platt": round(ece(ye, pe_platt), 4),
            "eval_brier_raw": round(brier(ye, pe), 4),
            "eval_brier_temp": round(brier(ye, pe_temp), 4),
            "threshold_for_sens>=0.90": {"threshold": round(chosen, 3),
                                          "eval_sensitivity": round(s_eval, 4),
                                          "eval_specificity": round(sp_eval, 4)},
        })
    res["recalibration"] = recal

    out = ROOT / "results" / f"external_{args.name.lower()}.json"
    out.write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))
    print(f"\nsaved {out}")


if __name__ == "__main__":
    main()
