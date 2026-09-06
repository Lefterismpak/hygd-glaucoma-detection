"""Fit one positive temperature to a designated calibration split only."""
import numpy as np
from scipy.optimize import minimize_scalar

TEMPERATURE_BOUNDS = (0.001, 1000.0)


def _inputs(logits, labels):
    z, y = np.asarray(logits, dtype=float), np.asarray(labels, dtype=float)
    if z.ndim != 1 or y.ndim != 1 or len(z) != len(y) or len(z) < 2:
        raise ValueError("Temperature fitting needs matching one-dimensional calibration inputs")
    if not np.isfinite(z).all() or not np.isin(y, [0, 1]).all() or len(np.unique(y)) != 2:
        raise ValueError("Calibration inputs must be finite and contain both binary classes")
    if np.max(np.abs(z)) > np.finfo(float).max * TEMPERATURE_BOUNDS[0] / 2:
        raise ValueError("Logit magnitude exceeds the bounded numerical contract")
    return z, y


def bernoulli_nll(logits, labels, temperature):
    z, y = _inputs(logits, labels)
    if not np.isfinite(temperature) or temperature <= 0:
        raise ValueError("Temperature must be finite and positive")
    scaled = z / temperature
    return float(np.mean(np.logaddexp(0, scaled) - y * scaled))


def fit_positive_temperature(logits, labels):
    """Bounded exact Bernoulli NLL in log-temperature space.

    The fixed positive bounds avoid a disconnected clamped-parameter gradient.
    Include T=1 and both bounds explicitly, so the returned calibration objective
    cannot be worse than the unscaled reference. A boundary solution can signal
    weak identification; it is not evidence of prospective calibration quality.
    """
    z, y = _inputs(logits, labels)
    lower, upper = TEMPERATURE_BOUNDS
    objective = lambda value: bernoulli_nll(z, y, float(np.exp(value)))
    result = minimize_scalar(objective, bounds=(np.log(lower), np.log(upper)),
                             method="bounded", options={"xatol": 1e-9, "maxiter": 200})
    if not result.success or not np.isfinite(result.fun):
        raise ValueError("Positive temperature optimizer did not converge")
    candidates = [1.0, float(np.exp(result.x)), lower, upper]
    return min(candidates, key=lambda t: bernoulli_nll(z, y, t))
