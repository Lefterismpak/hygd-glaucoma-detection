"""Explainability + clinical error analysis (Phase 5).

Grad-CAM (via the `grad-cam` / pytorch-grad-cam package) was chosen over SHAP
for this image-based clinical task (Grad-CAM fits image data better than SHAP here).
"""

import numpy as np
import torch
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

from src.data_utils import IMAGENET_MEAN, IMAGENET_STD


def grad_cam_overlay(model, image_tensor, target_layer):
    """Return an RGB numpy array (H, W, 3) with the Grad-CAM heatmap overlaid.

    `image_tensor` is a single preprocessed (3, H, W) tensor (already normalized,
    as produced by data_utils.preprocess_image). `target_layer` is typically the
    last conv block, e.g. `model.layer4[-1]` for a torchvision ResNet18.
    """
    cam = GradCAM(model=model, target_layers=[target_layer])
    # requires_grad_(True) is needed even with a frozen backbone: autograd only
    # tracks gradients through a subgraph if something feeding it requires grad,
    # and every backbone weight has requires_grad=False by design (see train.py).
    input_tensor = image_tensor.unsqueeze(0).requires_grad_(True)
    grayscale_cam = cam(input_tensor=input_tensor)[0]

    # Un-normalize back to [0, 1] RGB for the overlay background image.
    chw = image_tensor.numpy()
    hwc = chw.transpose(1, 2, 0)
    rgb = np.clip(hwc * IMAGENET_STD + IMAGENET_MEAN, 0, 1)

    return show_cam_on_image(rgb, grayscale_cam, use_rgb=True)


def review_predictions(y_true, y_prob, n_correct=5, n_wrong=5, seed=42):
    """Sample indices of n_correct correct + n_wrong incorrect predictions for review.

    Returns a dict of {"correct": [indices], "wrong": [indices]}. The actual
    clinical reflection on *why* each wrong prediction may have happened is a
    write-up step, not something this function can generate — do it by hand
    in notebook 04 once the sampled images are visible.
    """
    rng = np.random.default_rng(seed)
    y_pred = (y_prob >= 0.5).astype(int)
    correct_idx = np.where(y_pred == y_true)[0]
    wrong_idx = np.where(y_pred != y_true)[0]

    n_correct = min(n_correct, len(correct_idx))
    n_wrong = min(n_wrong, len(wrong_idx))

    return {
        "correct": rng.choice(correct_idx, size=n_correct, replace=False).tolist() if n_correct else [],
        "wrong": rng.choice(wrong_idx, size=n_wrong, replace=False).tolist() if n_wrong else [],
    }
