"""Inference-parity gate (GlaucoGen Step 0).

Proves that the standalone single-image `predict.predict_prob` reproduces the
probabilities from the batched eval pipeline used to produce the reported HYGD
results, on the HYGD test split, to < TOL. If this fails, external-validation
"domain-shift" numbers cannot be trusted — fix preprocessing before proceeding.

Run with --checkpoint and --checkpoint-sha256; use --raw-dir for a relocated
HYGD copy. The gate compares current execution paths on fixed bytes, not the
provenance of an old training run or biological-subject independence.
"""

from pathlib import Path
import argparse
import hashlib
import json
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.data_utils import load_dataset_metadata, train_val_test_split  # noqa: E402
from src.experiments import TransformHYGDDataset, build_transforms  # noqa: E402
from validation.predict import load_model, predict_probs  # noqa: E402
from validation.evaluation_utils import read_verified_file_bytes  # noqa: E402

TOL = 1e-4


def reference_probs(model, test_df, device="cpu"):
    """Probabilities from the SAME batched DataLoader path used in run_v2_experiments."""
    loader = DataLoader(
        TransformHYGDDataset(test_df, build_transforms(train=False), require_verified_sha256=True),
        batch_size=32, shuffle=False,
    )
    probs = []
    model.eval()
    with torch.no_grad():
        for x, _ in loader:
            probs.extend(torch.softmax(model(x.to(device)), dim=1)[:, 1].cpu().numpy().tolist())
    return np.array(probs)


def check_parity(model, frame, device="cpu"):
    if frame.empty or not {"image_path", "label", "sha256"}.issubset(frame.columns):
        raise ValueError("Parity requires nonempty, hash-bound image metadata")
    model.eval().to(device)
    ref = reference_probs(model, frame, device)
    standalone = predict_probs(frame.image_path.astype(str).tolist(), model, device,
                               expected_image_sha256s=frame.sha256.astype(str).tolist())
    if not np.isfinite(ref).all() or not np.isfinite(standalone).all():
        raise ValueError("Parity cannot accept nonfinite outputs")
    delta = np.abs(ref - standalone)
    return {"status": "passed" if float(delta.max()) < TOL else "failed",
            "images": len(frame), "max_absolute_difference": float(delta.max()),
            "mean_absolute_difference": float(delta.mean()), "tolerance": TOL,
            "scope": "same verified bytes, single-image versus batch execution; not training provenance or clinical validation"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=ROOT / "data/raw")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--checkpoint-sha256", required=True)
    args = parser.parse_args()
    df = load_dataset_metadata(args.raw_dir)
    _, _, test_df = train_val_test_split(df, seed=42)
    test_df = test_df.reset_index(drop=True)

    test_df["sha256"] = [read_verified_file_bytes(p)[1] for p in test_df.image_path]
    model = load_model(args.checkpoint, expected_checkpoint_sha256=args.checkpoint_sha256)
    report = check_parity(model, test_df)
    report["checkpoint_sha256"] = model.checkpoint_sha256
    report["image_byte_sequence_sha256"] = hashlib.sha256("\n".join(test_df.sha256).encode()).hexdigest()
    print(json.dumps(report, indent=2, allow_nan=False))
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
