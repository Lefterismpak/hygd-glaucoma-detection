"""v2 experimental harness: augmentation, fine-tuning, patient-level cross-validation,
bootstrap confidence intervals, and threshold analysis.

This sits on top of the Phase-4 baseline (src/train.py, src/evaluate.py). The baseline
answered "does a frozen-backbone ResNet18 work at all" (yes, AUC 0.952 on one split).
This module answers the harder, more honest questions:
  - Is that 0.952 a stable estimate or a lucky split?  -> patient-level k-fold CV
  - How wide is the uncertainty on a 44-patient test set? -> bootstrap CIs
  - Does augmentation or fine-tuning actually help?        -> controlled comparison
  - What operating threshold makes sense for *screening*?  -> threshold analysis
  - Are the errors random or concentrated in low-quality images? -> error analysis
"""

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import confusion_matrix, roc_auc_score
from sklearn.model_selection import GroupKFold
from torch.utils.data import Dataset
from torchvision import models, transforms

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


# ----------------------------------------------------------------------------
# Data: augmentation-capable dataset (torchvision transforms)
# ----------------------------------------------------------------------------

def build_transforms(train, image_size=224):
    """Train transforms include light augmentation appropriate for fundus images.

    Fundus photos are roughly rotation/flip invariant for the disc (the optic
    nerve head looks glaucomatous or not regardless of small orientation
    changes), so horizontal flip + small rotation + mild color jitter are safe.
    We deliberately avoid vertical flip (unnatural for retina) and heavy jitter.
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

    def __init__(self, metadata_df, transform):
        self.df = metadata_df.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = Image.open(row["image_path"]).convert("RGB")
        img = self.transform(img)
        label = int(row["label"])
        return img, torch.tensor(label, dtype=torch.long)


# ----------------------------------------------------------------------------
# Model: baseline (frozen) or fine-tuned (unfreeze layer4)
# ----------------------------------------------------------------------------

def build_model(num_classes=2, mode="frozen"):
    """mode: 'frozen' (head only) or 'finetune_layer4' (unfreeze last residual block).

    Fine-tuning only layer4 (not the whole backbone) is the middle ground for a
    small dataset: it lets the highest-level features adapt to fundus images
    without the overfitting risk of unfreezing all 11M parameters.
    """
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
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
    """Bootstrap 95% CIs for auc/sensitivity/specificity by resampling test cases.

    On a 44-patient / ~99-image test set, point estimates are noisy — this
    quantifies how noisy. Resamples at the image level (a simplification;
    a fully rigorous version would resample patients).
    """
    rng = np.random.default_rng(seed)
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    n = len(y_true)
    aucs, senss, specs = [], [], []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        yt, yp = y_true[idx], y_prob[idx]
        if len(np.unique(yt)) < 2:
            continue
        m = metrics_at_threshold(yt, yp, threshold)
        aucs.append(m["auc"])
        senss.append(m["sensitivity"])
        specs.append(m["specificity"])

    def ci(vals):
        vals = np.array([v for v in vals if not np.isnan(v)])
        return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]

    return {"auc_95ci": ci(aucs), "sensitivity_95ci": ci(senss), "specificity_95ci": ci(specs)}


def threshold_for_target_sensitivity(y_true, y_prob, target_sensitivity=0.95):
    """Find the highest threshold that still achieves >= target sensitivity.

    NOTE: this returns the exact highest observed-probability boundary meeting the
    constraint, which can be an unstable single point. For a deployment recommendation
    prefer a full threshold sweep (see run_v2_experiments / results/threshold_sweep.json)
    and pick a round, robust operating point (0.40 was chosen for this project).

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
    """Yield (train_df, val_df) folds split at the patient level (no leakage)."""
    gkf = GroupKFold(n_splits=n_splits)
    groups = metadata["patient_id"].values
    for train_idx, val_idx in gkf.split(metadata, groups=groups):
        yield (
            metadata.iloc[train_idx].reset_index(drop=True),
            metadata.iloc[val_idx].reset_index(drop=True),
        )
