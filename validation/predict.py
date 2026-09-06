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
from src.data_utils import load_verified_rgb_image  # noqa: E402
from validation.evaluation_utils import load_torch_state_dict_safely  # noqa: E402

_EVAL_TF = build_transforms(train=False)
DEFAULT_CKPT = Path(__file__).resolve().parents[1] / "results/finetune_layer4_aug.pt"


def load_model(
    checkpoint=DEFAULT_CKPT,
    device="cpu",
    *,
    expected_checkpoint_sha256=None,
):
    """Load a local weights-only state dict without following filesystem links."""
    model = build_model(mode="finetune_layer4", pretrained=False)
    state, digest = load_torch_state_dict_safely(
        checkpoint,
        map_location=device,
        expected_sha256=expected_checkpoint_sha256,
    )
    model.load_state_dict(state)
    model.checkpoint_sha256 = digest
    model.eval().to(device)
    return model


@torch.no_grad()
def predict_prob(image_path, model, device="cpu", *, expected_image_sha256=None):
    """Return P(glaucoma / GON+) for a single fundus image path.

    Uses the exact eval transform the model was validated with. This is the
    function every external-validation script should call — do not re-implement
    preprocessing anywhere else.
    """
    if model.training:
        raise ValueError("Inference requires a model in evaluation mode")
    img = load_verified_rgb_image(image_path, expected_sha256=expected_image_sha256)
    x = _EVAL_TF(img).unsqueeze(0).to(device)
    prob = torch.softmax(model(x), dim=1)[0, 1].item()
    if not np.isfinite(prob) or not 0 <= prob <= 1:
        raise ValueError("Model emitted an invalid probability")
    return float(prob)


@torch.no_grad()
def predict_probs(image_paths, model, device="cpu", *, expected_image_sha256s=None):
    """Single-image loop returning one bounded probability per verified path."""
    image_paths = list(image_paths)
    digests = [None] * len(image_paths) if expected_image_sha256s is None else list(expected_image_sha256s)
    if len(digests) != len(image_paths):
        raise ValueError("One expected image digest is required per input path")
    return np.array([predict_prob(p, model, device, expected_image_sha256=h)
                     for p, h in zip(image_paths, digests)])


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Research-only uncalibrated GON+ model score for one image")
    ap.add_argument("image")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--checkpoint-sha256", required=True)
    ap.add_argument("--image-sha256")
    args = ap.parse_args()
    m = load_model(
        args.checkpoint,
        expected_checkpoint_sha256=args.checkpoint_sha256,
    )
    print(f"{args.image}\tuncalibrated_GON_plus_score={predict_prob(args.image, m, expected_image_sha256=args.image_sha256):.6f}")
