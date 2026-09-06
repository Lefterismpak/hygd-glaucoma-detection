"""Verify a complete v5 bundle and emit only aggregate diagnostics to stdout.

This verifies retained evidence, not an independently attested execution date.
It performs no training, inference, model selection or file writes. Private
locations and row-level artifacts are deliberately absent from the output.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import re
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from validation.calibration_diagnostics import describe_group_predictions
from validation.evaluation_utils import (
    PROTOCOL_VERSION,
    _read_terminal_artifact_bytes,
    read_confined_json,
    read_verified_file_bytes,
    validate_terminal_bundle,
)


def summarize_completed_run(root: Path, result_relative: str, *, expected_result_sha256: str) -> dict:
    """Require complete, hash-bound, semantically checked evidence before export."""
    root = Path(root).resolve()
    # This first read validates the results directory and final component without
    # following symlinks. The pinned byte read below must decode identically.
    result = read_confined_json(root, result_relative)
    raw, digest = read_verified_file_bytes(root / result_relative, expected_sha256=expected_result_sha256)
    if json.loads(raw) != result:
        raise ValueError("Result changed between confined and hash-bound reads")
    validate_terminal_bundle(result, root)
    # Re-read through the same no-follow, hash/size-bound contract. The exact
    # verified bytes, rather than a later unguarded CSV read, feed diagnostics.
    contents = _read_terminal_artifact_bytes(result, root)
    groups = pd.read_csv(io.BytesIO(contents["oof_groups"]))
    diagnostics = describe_group_predictions(groups)
    primary = result["internal_evaluation"]["primary"]
    report = result["internal_evaluation"]
    numeric_primary_keys = ("n", "positives", "negatives", "sensitivity", "specificity",
        "confusion_tn_fp_fn_tp", "auroc", "auroc_95ci", "sensitivity_95ci", "specificity_95ci")
    # Closed, explicit projection: no free-text field from the private result is
    # copied to the public receipt, even if it passes another schema validator.
    runtime = {}
    for key in ("python", "torch", "torchvision", "numpy", "pandas", "scikit_learn", "pillow"):
        value = result["environment"][key]
        if not isinstance(value, str) or not re.fullmatch(r"[0-9][0-9A-Za-z.+_-]{0,50}", value):
            raise ValueError("Runtime versions must be location-free version strings")
        runtime[key] = value
    device = result["configuration_fixed_before_outer_evaluation"]["device"]
    if device not in {"cpu", "mps", "cuda"}:
        raise ValueError("Unrecognized execution device")
    return {
        "schema_version": "hygd-completed-run-summary-v1",
        "protocol_version": PROTOCOL_VERSION,
        "evidence_status": "verified_complete_v5_bundle",
        "execution_date_attested_by_exporter": False,
        "primary": {key: primary[key] for key in numeric_primary_keys},
        "estimand": "equal_weight_mean_of_five_outer_fold_linked_group_aurocs",
        "uncertainty": "5000_fixed_fold_group_bootstrap_replicates_conditional_on_frozen_predictions",
        "secondary": {
            "pooled_group_auroc": report["pooled_oof_group_descriptive"]["auroc"],
            "image_auroc": report["secondary_image_level"]["auroc"],
            "unique_images": report["secondary_image_level"]["n"],
        },
        "probability_diagnostics": diagnostics,
        "runtime": {**runtime, "device": device},
        "private_source_binding": {
            "result_sha256": digest,
            "result_size_bytes": len(raw),
            "artifact_sha256": dict(result["artifact_sha256"]),
            "artifact_size_bytes": dict(result["artifact_size_bytes"]),
        },
        "exporter_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "diagnostic_implementation_sha256": hashlib.sha256((ROOT / "validation/calibration_diagnostics.py").read_bytes()).hexdigest(),
        "privacy": {"row_level_data_included": False, "private_paths_included": False,
                    "images_or_weights_included": False},
        "limitations": [
            "Complete bundle verification is not independent attestation of when or how training occurred.",
            "The recipe was informed by prior HYGD development; this is internal post-development evidence.",
            "The interval excludes retraining, recipe, checkpoint and threshold-selection uncertainty.",
            "Linked evaluation groups are not independently verified biological subjects.",
            "No external transportability, prospective calibration or clinical utility is established.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result", help="One private result JSON directly under results/")
    parser.add_argument("--expected-result-sha256")
    parser.add_argument("--verify-summary", help="Optional aggregate JSON directly under results/")
    parser.add_argument("--describe-contract", action="store_true")
    args = parser.parse_args()
    if args.describe_contract:
        print(json.dumps({"requires": "complete v5 terminal bundle and expected result SHA-256",
            "writes_files": False, "trains_or_runs_inference": False,
            "output": "aggregate-only JSON; no execution-date attestation"}, indent=2))
        return
    if not args.result or not args.expected_result_sha256:
        parser.error("--result and --expected-result-sha256 are required")
    summary = summarize_completed_run(ROOT, args.result, expected_result_sha256=args.expected_result_sha256)
    if args.verify_summary:
        if read_confined_json(ROOT, args.verify_summary) != summary:
            raise ValueError("Published aggregate differs from verified private evidence")
    print(json.dumps(summary, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
