"""Historical v2 model-development helpers.

This sits on top of the Phase-4 baseline (src/train.py, src/evaluate.py). The baseline
answered "does a frozen-backbone ResNet18 work at all" (yes, AUC 0.952 on one split).
This module retains the model/transformation helpers needed to reproduce the old
single-split development comparison. It does not expose a valid final-CV or
uncertainty path. For the canonical internal evaluation, use
``validation/internal_evaluation_repair.py``.

Historical questions retained for provenance:
  - Does augmentation or fine-tuning actually help?        -> controlled comparison
  - What operating threshold makes sense for *screening*?  -> threshold analysis
  - Are the errors random or concentrated in low-quality images? -> error analysis
"""

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import confusion_matrix, roc_auc_score
from torch.utils.data import Dataset
from torchvision import models, transforms

from src.data_utils import load_verified_rgb_image

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


# ----------------------------------------------------------------------------
# Data: augmentation-capable dataset (torchvision transforms)
# ----------------------------------------------------------------------------

def build_transforms(train, image_size=224):
    """Train transforms include light augmentation appropriate for fundus images.

    These are fixed historical modeling choices, not a clinically validated
    invariance claim. Their effects on anatomical orientation, laterality and
    disease signal were not independently adjudicated. Preserve this recipe
    when reproducing the fixed v5/v6 comparisons.
    """
    if train:
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(brightness=0.1, contrast=0.1),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


class TransformHYGDDataset(Dataset):
    """HYGD dataset that applies a torchvision transform (for augmentation)."""

    def __init__(self, metadata_df, transform, *, require_verified_sha256=False):
        self.df = metadata_df.reset_index(drop=True)
        self.transform = transform
        self.require_verified_sha256 = bool(require_verified_sha256)
        if self.require_verified_sha256 and (
            "sha256" not in self.df.columns
            or self.df["sha256"].isna().any()
            or not self.df["sha256"].astype(str).str.fullmatch(r"[0-9a-f]{64}").all()
        ):
            raise ValueError("Canonical HYGD decoding requires one verified SHA-256 per row")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        expected_sha256 = str(row["sha256"]) if "sha256" in self.df.columns else None
        img = load_verified_rgb_image(
            row["image_path"], expected_sha256=expected_sha256
        )
        img = self.transform(img)
        label = int(row["label"])
        return img, torch.tensor(label, dtype=torch.long)


# ----------------------------------------------------------------------------
# Model: baseline (frozen) or fine-tuned (unfreeze layer4)
# ----------------------------------------------------------------------------

def build_model(num_classes=2, mode="frozen", *, pretrained=True):
    """Freeze parameters except the head, or unfreeze layer4 plus the head.

    Both modes retain training-mode BatchNorm updates in the training loop,
    including buffers in parameter-frozen layers. "Frozen" therefore describes
    parameters, not an immutable feature extractor or guaranteed overfit control.
    """
    weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
    model = models.resnet18(weights=weights)
    for p in model.parameters():
        p.requires_grad = False

    if mode == "finetune_layer4":
        for p in model.layer4.parameters():
            p.requires_grad = True
    elif mode != "frozen":
        raise ValueError(f"unknown mode {mode!r}")

    model.fc = nn.Linear(model.fc.in_features, num_classes)  # always trainable
    return model


# ----------------------------------------------------------------------------
# Metrics helpers
# ----------------------------------------------------------------------------

def metrics_at_threshold(y_true, y_prob, threshold=0.5):
    """Return auc, sensitivity, specificity, and the confusion matrix at a threshold."""
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    y_pred = (y_prob >= threshold).astype(int)
    auc = roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else float("nan")
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    sens = tp / (tp + fn) if (tp + fn) else float("nan")
    spec = tn / (tn + fp) if (tn + fp) else float("nan")
    return {"auc": auc, "sensitivity": sens, "specificity": spec, "confusion_matrix": cm}


def bootstrap_ci(y_true, y_prob, threshold=0.5, n_boot=2000, seed=42):
    """Hard-deprecated ambiguous image-row bootstrap entry point."""

    raise RuntimeError(
        "bootstrap_ci is disabled because it resampled correlated image rows. "
        "Use validation.evaluation_utils.cluster_bootstrap with "
        "evaluation_group for inference, or historical_image_bootstrap_ci only "
        "when explicitly reproducing the noncanonical development artifact."
    )


def threshold_for_target_sensitivity(y_true, y_prob, target_sensitivity=0.95):
    """Find the highest threshold that still achieves >= target sensitivity.

    Historical helper only: this returns the exact highest observed-probability
    boundary meeting the constraint on the supplied data. It can be an unstable,
    test-derived point and must not be used as a deployment recommendation. The
    repaired evaluator selects fold-specific thresholds on inner validation only.

    For a screening tool you usually fix a minimum acceptable sensitivity
    (catch most disease) and then take the best specificity available at that
    constraint — the opposite of just using 0.5.
    """
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    thresholds = np.unique(y_prob)
    best = None
    for t in sorted(thresholds):
        m = metrics_at_threshold(y_true, y_prob, t)
        if m["sensitivity"] >= target_sensitivity:
            best = {"threshold": float(t), **{k: m[k] for k in ("sensitivity", "specificity")}}
    return best  # highest threshold meeting the constraint (best specificity)


def patient_kfold(metadata, n_splits=5, seed=42):
    """Hard-deprecated same-fold checkpoint/scoring helper."""

    raise RuntimeError(
        "patient_kfold is disabled: the historical loop reused each fold for "
        "checkpoint selection and reporting. Run "
        "validation/internal_evaluation_repair.py for disjoint inner/outer folds."
    )
