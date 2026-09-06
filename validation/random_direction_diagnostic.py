"""Aggregate-only geometry stress test of an existing frozen representation.

The sampling unit is a random direction, NOT a patient or a shuffled dataset.
Fractions and quantiles are conditional on these fixed features and labels;
they are not permutation p-values, confidence intervals or clinical validation.
"""
from __future__ import annotations

import numpy as np
from scipy.stats import rankdata


def describe_random_directions(features: np.ndarray, labels: np.ndarray, *,
                               seed: int = 20260906, n_directions: int = 1024) -> dict:
    """Draw isotropic directions without fitting, selection or sign reversal.

    Callers must choose and document the evaluation unit before supplying rows.
    The result depends on the frozen feature coordinate system and its scaling.
    A high score from one random direction can coexist with a useful learned
    representation; this test does not diagnose why a trained model succeeds.
    """
    x, y = np.asarray(features, dtype=float), np.asarray(labels, dtype=float)
    if x.ndim != 2 or y.ndim != 1 or x.shape[0] != len(y) or x.shape[1] < 1:
        raise ValueError("Expected a feature matrix and one matching label per row")
    if not np.isfinite(x).all() or not np.isin(y, [0, 1]).all() or len(np.unique(y)) != 2:
        raise ValueError("Features must be finite and labels must contain both binary classes")
    if type(seed) is not int or seed < 0 or type(n_directions) is not int or n_directions < 1:
        raise ValueError("Seed and number of directions must be valid integers")
    rng = np.random.default_rng(seed)
    directions = rng.normal(size=(x.shape[1], n_directions))
    directions /= np.linalg.norm(directions, axis=0, keepdims=True)
    # Centering improves numerical conditioning; it changes each score by only
    # a common offset, so it does not change rank-based AUROC.
    scores = (x - x.mean(axis=0, keepdims=True)) @ directions
    ranks = rankdata(scores, method="average", axis=0)
    positive = y == 1
    n_positive, n_negative = int(positive.sum()), int((~positive).sum())
    auc = (ranks[positive].sum(axis=0) - n_positive * (n_positive + 1) / 2) / (n_positive * n_negative)
    return {
        "status": "retrospective_representation_geometry_only",
        "rows": len(y), "feature_dimensions": x.shape[1],
        "seed": seed, "n_directions": n_directions,
        "quantile_probabilities": [0.025, 0.25, 0.5, 0.75, 0.975],
        "auroc_quantiles": np.quantile(auc, [0.025, 0.25, 0.5, 0.75, 0.975]).tolist(),
        "fraction_auroc_at_least_0_75": float(np.mean(auc >= 0.75)),
        "fraction_auroc_at_most_0_25": float(np.mean(auc <= 0.25)),
        "fractions_are_p_values": False,
        "label_based_direction_selection": False,
        "model_fitting_or_inference": "no fitting; linear projections of existing frozen features only",
        "limitations": [
            "Random directions are not training-label permutations and need not share their distribution.",
            "No causal mechanism, biological independence or transportability is identified.",
            "Directional quantiles condition on these data; they are not sampling confidence intervals.",
            "Results depend on the feature coordinate system and scaling; do not tune them for this diagnostic.",
        ],
    }
