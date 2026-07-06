"""Aggregate the per-seed Attempt-B runs into a mean +/- std robustness summary.

Reads every results/seed_runs/attemptB_seed*.json and reports the held-out
PAPILA AUROC (TTA) point estimate across seeds. This measures TRAINING
stochasticity (weight init + augmentation + source train/val split), which is
complementary to the within-run bootstrap CI (test-set sampling noise).
"""
import glob
import json
import os
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
files = sorted(glob.glob(str(ROOT / "results/seed_runs/attemptB_seed*.json")))
runs = []
for f in files:
    d = json.loads(Path(f).read_text())
    runs.append({"seed": d.get("seed"), "auroc": d["held_out_papila_auroc_TTA"],
                 "ci": d.get("papila_auroc_95ci")})
runs.sort(key=lambda r: (r["seed"] is None, r["seed"]))

aurocs = np.array([r["auroc"] for r in runs], dtype=float)
summary = {
    "attempt": "B_multitask_VCDR",
    "metric": "held_out_PAPILA_AUROC_TTA (PAPILA never used for training/selection)",
    "n_seeds": len(aurocs),
    "seeds": [r["seed"] for r in runs],
    "per_seed_auroc": [round(x, 4) for x in aurocs],
    "mean": round(float(aurocs.mean()), 4),
    "std": round(float(aurocs.std(ddof=1)), 4) if len(aurocs) > 1 else None,
    "min": round(float(aurocs.min()), 4),
    "max": round(float(aurocs.max()), 4),
    "note": "std is seed-to-seed training stochasticity; each run also carries a "
            "within-run bootstrap 95% CI on the test set (see per-seed files).",
}
out = ROOT / "results/seed_robustness_attemptB.json"
out.write_text(json.dumps(summary, indent=2))
print(json.dumps(summary, indent=2))
