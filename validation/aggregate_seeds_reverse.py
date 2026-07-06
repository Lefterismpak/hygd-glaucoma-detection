"""Aggregate the per-seed REVERSE runs (train HYGD+PAPILA, held-out RIM-ONE)."""
import glob, json
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
files = sorted(glob.glob(str(ROOT / "results/seed_runs_reverse/reverse_seed*.json")))
runs = []
for f in files:
    d = json.loads(Path(f).read_text())
    runs.append({"seed": d.get("seed"), "auroc": d["held_out_auroc_TTA"], "ci": d.get("auroc_95ci")})
runs.sort(key=lambda r: (r["seed"] is None, r["seed"]))
a = np.array([r["auroc"] for r in runs], dtype=float)
summary = {"attempt": "B_reverse_multitask_VCDR",
           "metric": "held_out_RIMONE_AUROC_TTA (RIM-ONE never used for training/selection)",
           "n_seeds": len(a), "seeds": [r["seed"] for r in runs],
           "per_seed_auroc": [round(x,4) for x in a],
           "mean": round(float(a.mean()),4),
           "std": round(float(a.std(ddof=1)),4) if len(a)>1 else None,
           "min": round(float(a.min()),4), "max": round(float(a.max()),4),
           "zero_shot_rimone_was": 0.606,
           "note": "Symmetric to forward (held-out PAPILA). std = seed-to-seed training stochasticity."}
(ROOT/"results/seed_robustness_reverse.json").write_text(json.dumps(summary, indent=2))
print(json.dumps(summary, indent=2))
