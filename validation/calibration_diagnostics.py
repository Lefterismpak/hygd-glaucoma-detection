"""Descriptive group-level probability diagnostics; never fit or select a model.

Brier score and log loss assess overall probabilistic accuracy, not calibration
alone. These summaries condition on frozen OOF predictions. They do not estimate
uncertainty due to training/selection and cannot establish clinical utility.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, log_loss


def _operating_point(labels: np.ndarray, decisions: np.ndarray) -> dict:
    matrix = confusion_matrix(labels, decisions, labels=[0, 1])
    tn, fp, fn, tp = (int(value) for value in matrix.ravel())
    return {
        "confusion_matrix": matrix.tolist(),
        "sensitivity": tp / (tp + fn) if tp + fn else None,
        "specificity": tn / (tn + fp) if tn + fp else None,
        "predicted_positive_fraction": float(np.mean(decisions)),
    }


def describe_group_predictions(frame: pd.DataFrame) -> dict:
    """Summarize one row per linked evaluation group, without exporting IDs.

    Fixed-width reliability bins containing fewer than ten groups suppress their
    counts, outcomes and scores. This is a small-cell reporting rule, not a
    differential-privacy guarantee. Callers remain responsible for publication.
    """
    required = {"evaluation_group", "label", "probability", "selected_threshold", "outer_fold"}
    if not required.issubset(frame.columns) or frame.empty:
        raise ValueError("Nonempty group predictions require all diagnostic columns")
    groups = frame["evaluation_group"]
    if groups.isna().any() or groups.astype(str).str.strip().eq("").any() or groups.astype(str).duplicated().any():
        raise ValueError("Each nonempty evaluation group must occur exactly once")
    arrays = {}
    for column in required - {"evaluation_group"}:
        values = pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=float)
        if not np.isfinite(values).all():
            raise ValueError("Diagnostic inputs must be finite numeric values")
        if column == "outer_fold":
            if np.any(values < 0) or np.any(values != np.floor(values)):
                raise ValueError("Outer folds must be nonnegative integers")
        elif np.any((values < 0) | (values > 1)):
            raise ValueError("Labels, probabilities and thresholds must be in [0, 1]")
        if column == "label" and not np.isin(values, [0, 1]).all():
            raise ValueError("Labels must be binary")
        arrays[column] = values
    labels, scores = arrays["label"].astype(int), arrays["probability"]
    thresholds, folds = arrays["selected_threshold"], arrays["outer_fold"].astype(int)
    for fold in np.unique(folds):
        if len(np.unique(thresholds[folds == fold])) != 1:
            raise ValueError("Each outer fold must have exactly one recorded inner threshold")

    bins = []
    indices = np.minimum((scores * 5).astype(int), 4)
    for index in range(5):
        selected = indices == index
        count = int(selected.sum())
        cell = {"lower": index / 5, "upper": (index + 1) / 5, "suppressed": count < 10}
        if count >= 10:
            cell.update(groups=count, mean_predicted_risk=float(scores[selected].mean()),
                        observed_fraction=float(labels[selected].mean()))
        bins.append(cell)

    observed, predicted = float(labels.mean()), float(scores.mean())
    return {
        "status": "post_development_descriptive_only",
        "used_to_select_model_or_threshold": False,
        "evaluation_groups": int(len(frame)),
        "brier_score": float(np.mean((scores - labels) ** 2)),
        "log_loss": float(log_loss(labels, scores, labels=[0, 1])),
        "observed_fraction": observed,
        "mean_predicted_risk": predicted,
        "mean_risk_minus_observed_fraction": predicted - observed,
        "same_sample_prevalence_reference_brier": observed * (1 - observed),
        "recorded_threshold_range": [float(thresholds.min()), float(thresholds.max())],
        "operating_points": {
            "recorded_inner_thresholds": _operating_point(labels, scores >= thresholds),
            "fixed_0_5_descriptive": _operating_point(labels, scores >= 0.5),
        },
        "reliability_bins": bins,
        "limitations": [
            "Post-development OOF description; no recalibration, threshold search or prospective assessment.",
            "Brier score and log loss combine calibration, discrimination and outcome uncertainty.",
            "The prevalence reference is descriptive and uses these same outcomes; it is not a fitted comparator.",
            "Bins are sparse and noisy; fewer than ten groups suppresses counts, outcomes and scores.",
            "Fold-specific models and repeated development limit pooled probability interpretation.",
            "Threshold variation alone does not prove probability miscalibration.",
            "No clinical operating point or clinical utility is established.",
        ],
    }
