"""Normalize the explicitly supported binary annotation encodings for BCE."""
import numpy as np


def binary_training_mask(mask):
    """Accept 0/1 or 0/255 masks; reject unknown intensities instead of guessing.

    This checks numeric encoding, not whether a contour is anatomically correct.
    """
    values = np.asarray(mask)
    if values.ndim != 2 or values.size == 0 or not np.isfinite(values).all():
        raise ValueError("A mask must be a nonempty finite two-dimensional array")
    if not np.isin(values, [0, 1, 255]).all():
        raise ValueError("Only explicit 0/1 or 0/255 binary mask encodings are supported")
    return (values > 0).astype(np.float32)
