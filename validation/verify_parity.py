"""Inference-parity gate (GlaucoGen Step 0).

Proves that the standalone single-image `predict.predict_prob` reproduces the
probabilities from the batched eval pipeline used to produce the reported HYGD
results, on the HYGD test split, to < TOL. If this fails, external-validation
"domain-shift" numbers cannot be trusted — fix preprocessing before proceeding.

Run:  python validation/verify_parity.py
"""

from pathlib import Path
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.data_utils import load_dataset_metadata, train_val_test_split  # noqa: E402
from src.experiments import TransformHYGDDataset, build_transforms  # noqa: E402
from validation.predict import load_model, predict_probs  # noqa: E402

TOL = 1e-4


def reference_probs(model, test_df, device="cpu"):
    """Probabilities from the SAME batched DataLoader path used in run_v2_experiments."""
    loader = DataLoader(
        TransformHYGDDataset(test_df, build_transforms(train=False)),
        batch_size=32, shuffle=False,
    )
    probs = []
    model.eval()
    with torch.no_grad():
        for x, _ in loader:
            probs.extend(torch.softmax(model(x.to(device)), dim=1)[:, 1].cpu().numpy().tolist())
    return np.array(probs)


def main():
    df = load_dataset_metadata(str(ROOT / "data/raw"))
    _, _, test_df = train_val_test_split(df, seed=42)
    test_df = test_df.reset_index(drop=True)

    model = load_model(str(ROOT / "results/finetune_layer4_aug.pt"))

    ref = reference_probs(model, test_df)
    standalone = predict_probs([str(p) for p in test_df["image_path"]], model)

    max_diff = float(np.max(np.abs(ref - standalone)))
    mean_diff = float(np.mean(np.abs(ref - standalone)))
    print(f"n = {len(ref)} test images")
    print(f"max |Δ|  = {max_diff:.3e}")
    print(f"mean |Δ| = {mean_diff:.3e}")
    print(f"tolerance = {TOL:.0e}")

    if max_diff < TOL:
        print("\nPARITY GATE: PASS ✓  standalone predict == batched eval pipeline")
        return 0
    print("\nPARITY GATE: FAIL ✗  preprocessing differs — do NOT trust external-validation drops until fixed")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
