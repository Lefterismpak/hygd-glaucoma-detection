"""Evaluation metrics for the baseline model (Phase 4).

Reports AUC, sensitivity, specificity, and confusion matrix — accuracy alone
is not sufficient for an imbalanced clinical screening task (73% positive rate
means a trivial "always predict GON+" classifier scores 73% accuracy).
"""

import numpy as np
import torch
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve


@torch.no_grad()
def evaluate(model, loader, device="cpu"):
    """Return a dict with auc, sensitivity, specificity, confusion_matrix, y_true, y_prob.

    Positive class = GON+ (label 1). Sensitivity = recall on GON+, specificity = recall on GON-.
    """
    model.eval()
    model.to(device)

    all_labels, all_probs = [], []
    for images, labels in loader:
        images = images.to(device)
        logits = model(images)
        probs = torch.softmax(logits, dim=1)[:, 1]  # P(GON+)
        all_labels.extend(labels.numpy().tolist())
        all_probs.extend(probs.cpu().numpy().tolist())

    y_true = np.array(all_labels)
    y_prob = np.array(all_probs)
    y_pred = (y_prob >= 0.5).astype(int)

    auc = roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else float("nan")
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
    specificity = tn / (tn + fp) if (tn + fp) > 0 else float("nan")

    return {
        "auc": auc,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "confusion_matrix": cm,
        "y_true": y_true,
        "y_prob": y_prob,
    }


def roc_points(y_true, y_prob):
    """Return (fpr, tpr, thresholds) for plotting the ROC curve."""
    return roc_curve(y_true, y_prob)
