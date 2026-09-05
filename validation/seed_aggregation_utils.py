"""Fail-closed schema checks shared by the current seed aggregators."""

from __future__ import annotations

import math
import re
from pathlib import Path


EXPECTED_EVIDENCE_STATUS = {
    "status": "adaptive_development_evidence",
    "target_blind": False,
    "transportability_established": False,
    "license_compatibility": "needs-proof",
    "external_claim_permitted": False,
    "rimone_subject_identity": (
        "operator_asserted_hash_bound_needs_independent_proof"
    ),
    "warning": (
        "Target AUROC was displayed during development and target-domain "
        "anatomical resources influenced preprocessing."
    ),
}

PATIENT_CLUSTER_UNCERTAINTY_KEYS = {
    "auroc",
    "auroc_95ci",
    "point_estimand_unit",
    "resampling_unit",
    "scope",
    "canonical_or_confirmatory",
    "seed",
    "attempted_replicates",
    "valid_replicates",
}


def require_exact_keys(payload: object, expected: set[str], context: str) -> dict:
    if not isinstance(payload, dict) or set(payload) != expected:
        raise ValueError(f"{context} must have the exact current schema")
    return payload


def is_lowercase_sha256(value: object) -> bool:
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def require_probability(value: object, context: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or not 0.0 <= float(value) <= 1.0
    ):
        raise ValueError(f"{context} must be a finite numeric value in [0, 1]")
    return float(value)


def validate_evidence_status(value: object, context: str) -> None:
    if value != EXPECTED_EVIDENCE_STATUS:
        raise ValueError(f"{context} lacks the exact adaptive evidence status")


def validate_patient_cluster_uncertainty(
    value: object, *, top_level_auroc: float, context: str
) -> None:
    payload = require_exact_keys(
        value, PATIENT_CLUSTER_UNCERTAINTY_KEYS, context
    )
    auroc = require_probability(payload["auroc"], f"{context} AUROC")
    interval = payload["auroc_95ci"]
    if not isinstance(interval, list) or len(interval) != 2:
        raise ValueError(f"{context} interval must contain exactly two bounds")
    lower = require_probability(interval[0], f"{context} lower bound")
    upper = require_probability(interval[1], f"{context} upper bound")
    if lower > auroc or auroc > upper:
        raise ValueError(f"{context} interval must contain its AUROC")
    if round(auroc, 4) != top_level_auroc:
        raise ValueError(f"{context} AUROC does not match the top-level result")
    expected_literals = {
        "point_estimand_unit": "eye_or_image_row",
        "resampling_unit": "patient",
        "scope": "conditional_on_fixed_adaptive_predictions",
        "canonical_or_confirmatory": False,
        "seed": 42,
        "valid_replicates": 2000,
    }
    for key, expected in expected_literals.items():
        if payload[key] != expected or type(payload[key]) is not type(expected):
            raise ValueError(f"{context} has invalid {key}")
    attempted = payload["attempted_replicates"]
    if (
        type(attempted) is not int
        or attempted < payload["valid_replicates"]
        or attempted > 20_000
    ):
        raise ValueError(f"{context} has invalid attempted_replicates")


def validate_seed_filename(path: Path, *, pattern: str, seed: int, context: str) -> None:
    if path.name != pattern.format(seed=seed):
        raise ValueError(f"{context} seed identity does not match its filename")
