"""Standalone inference for the HYGD best model, for external validation.

The whole point of external validation is to measure how the model degrades on
*other* datasets. That number is only trustworthy if the inference preprocessing
is IDENTICAL to what the model was trained/evaluated with — otherwise a
preprocessing mismatch fakes a "domain-shift drop" that is really a bug.

To make that guarantee by construction, this module reuses the exact same
`build_transforms(train=False)` and `build_model(mode="finetune_layer4")` from
`src.experiments` that produced the reported HYGD results. `verify_parity.py`
then proves that this single-image path reproduces the batched eval pipeline's
probabilities on the HYGD test set to < 1e-4 — the load-bearing gate for the
whole GlaucoGen study.
"""

from pathlib import Path

import numpy as np
import torch
from PIL import Image

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.experiments import build_model, build_transforms  # noqa: E402

_EVAL_TF = build_transforms(train=False)
DEFAULT_CKPT = "results/finetune_layer4_aug.pt"


def load_model(checkpoint=DEFAULT_CKPT, device="cpu"):
    """Load the HYGD best model (fine-tuned layer4) from a checkpoint."""
    model = build_model(mode="finetune_layer4")
    model.load_state_dict(torch.load(checkpoint, map_location=device))
    model.eval().to(device)
    return model


@torch.no_grad()
def predict_prob(image_path, model, device="cpu"):
    """Return P(glaucoma / GON+) for a single fundus image path.

    Uses the exact eval transform the model was validated with. This is the
    function every external-validation script should call — do not re-implement
    preprocessing anywhere else.
    """
    img = Image.open(image_path).convert("RGB")
    x = _EVAL_TF(img).unsqueeze(0).to(device)
    prob = torch.softmax(model(x), dim=1)[0, 1].item()
    return float(prob)


@torch.no_grad()
def predict_probs(image_paths, model, device="cpu"):
    """Vectorized convenience wrapper: list of paths -> np.array of P(glaucoma)."""
    return np.array([predict_prob(p, model, device) for p in image_paths])


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="P(glaucoma) for one fundus image")
    ap.add_argument("image")
    ap.add_argument("--checkpoint", default=DEFAULT_CKPT)
    args = ap.parse_args()
    m = load_model(args.checkpoint)
    print(f"{args.image}\tP(glaucoma)={predict_prob(args.image, m):.6f}")
