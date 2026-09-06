"""Aggregate mean-reduced losses by their actual denominator, not batch size."""
import math


class LossAccumulator:
    def __init__(self):
        self.numerator = 0.0
        self.denominator = 0.0

    def add(self, mean_loss, normalizer):
        mean_loss, normalizer = float(mean_loss), float(normalizer)
        if not math.isfinite(mean_loss) or mean_loss < 0:
            raise ValueError("Loss must be finite and nonnegative")
        if not math.isfinite(normalizer) or normalizer <= 0:
            raise ValueError("Loss normalizer must be finite and positive")
        self.numerator += mean_loss * normalizer
        self.denominator += normalizer

    def mean(self):
        if self.denominator == 0:
            raise ValueError("Cannot summarize an empty loss stream")
        return self.numerator / self.denominator


def cross_entropy_normalizer(labels, weights=None, *, ignore_index=-100):
    """Denominator of Torch CE(mean) for 1-D integer class targets.

    Soft labels, label smoothing and composite/auxiliary losses require their
    own explicit aggregation contract. The HYGD classifiers use hard targets.
    """
    import torch
    if labels.ndim != 1 or labels.dtype != torch.long:
        raise ValueError("Expected one-dimensional int64 class targets")
    valid = labels[labels != ignore_index]
    if valid.numel() == 0 or (valid < 0).any():
        raise ValueError("No valid nonnegative class targets")
    if weights is None:
        return float(valid.numel())
    if weights.ndim != 1 or not torch.isfinite(weights).all() or (weights <= 0).any():
        raise ValueError("Class weights must be finite and positive")
    if (valid >= weights.numel()).any():
        raise ValueError("Class target is outside the weight vector")
    return float(weights[valid.to(weights.device)].sum().item())
