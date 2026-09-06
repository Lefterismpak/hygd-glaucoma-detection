"""v6 contract: globally aggregated weighted CE, with the fixed v5 split recipe.

The v5 utility module and old receipts remain byte-preserved. This adapter checks
the two additional v6 training declarations, then reuses the v5 recipe/OOF
validators on a private in-memory projection. The actual v6 audit and its claim
hash are checked against the unmodified v6 result, never the projection.
"""
from __future__ import annotations

import argparse
import copy
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
from validation import evaluation_utils as legacy
from validation.calibration_diagnostics import describe_group_predictions

PROTOCOL_VERSION = "2026-09-06-v6"
TRAINING_FIELDS = {
    "loss_aggregation": "sum_weighted_nll_over_sum_target_class_weights",
    "batchnorm_policy": "running_statistics_adapt_in_all_training_mode_layers",
}


def _legacy_projection(result):
    if not isinstance(result, dict) or result.get("protocol_version") != PROTOCOL_VERSION:
        raise ValueError("Expected the separately identified v6 protocol")
    config = result.get("configuration_fixed_before_outer_evaluation")
    if not isinstance(config, dict) or any(config.get(k) != v for k, v in TRAINING_FIELDS.items()):
        raise ValueError("v6 requires its exact loss and BatchNorm declarations")
    projected = copy.deepcopy(result)
    projected["protocol_version"] = legacy.PROTOCOL_VERSION
    for key in TRAINING_FIELDS:
        del projected["configuration_fixed_before_outer_evaluation"][key]
    return projected


def validate_result(result):
    legacy.validate_internal_result_payload(_legacy_projection(result))


def validate_terminal(result, root):
    projected = _legacy_projection(result)
    legacy.validate_internal_result_payload(projected)
    contents = legacy._read_terminal_artifact_bytes(result, root)
    audit = json.loads(contents["audit"])
    audit_path = Path(result["artifacts"]["audit"])
    expected_result = str(audit_path.with_name(audit_path.name.removesuffix("_audit.json") + ".json"))
    keys = {"protocol_version", "status", "dataset_identity", "data_quality", "split_summary",
            "group_fold_manifest", "completed_outer_folds", "result_artifact", "result_claim_sha256"}
    if (not isinstance(audit, dict) or set(audit) != keys
            or audit.get("protocol_version") != PROTOCOL_VERSION or audit.get("status") != "complete"
            or audit.get("dataset_identity") != result["dataset_identity"]
            or audit.get("data_quality") != result["data_quality"]
            or audit.get("group_fold_manifest") != result["artifacts"]["group_folds"]
            or audit.get("completed_outer_folds") != result["completed_outer_folds"]
            or audit.get("result_artifact") != expected_result
            or audit.get("result_claim_sha256") != legacy.terminal_result_claim_sha256(result)):
        raise ValueError("v6 terminal audit is inconsistent or not bound to this claim")
    legacy._validate_complete_oof_claims(projected, contents, expected_split_summary=audit["split_summary"])


def summarize(root, result_relative, expected_sha256):
    root = Path(root).resolve()
    result = legacy.read_confined_json(root, result_relative)
    raw, digest = legacy.read_verified_file_bytes(root / result_relative, expected_sha256=expected_sha256)
    if json.loads(raw) != result:
        raise ValueError("v6 result changed between verified reads")
    validate_terminal(result, root)
    contents = legacy._read_terminal_artifact_bytes(result, root)
    groups = pd.read_csv(io.BytesIO(contents["oof_groups"]))
    primary = result["internal_evaluation"]["primary"]
    numeric = ("n", "positives", "negatives", "auroc", "auroc_95ci", "sensitivity", "specificity",
               "sensitivity_95ci", "specificity_95ci", "confusion_tn_fp_fn_tp")
    runtime = {}
    for key in ("python", "torch", "torchvision", "numpy", "pandas", "scikit_learn", "pillow"):
        value = result["environment"][key]
        if not isinstance(value, str) or not re.fullmatch(r"[0-9][0-9A-Za-z.+_-]{0,50}", value):
            raise ValueError("Runtime versions must be location-free version strings")
        runtime[key] = value
    return {
        "schema_version": "hygd-completed-v6-summary-v1", "protocol_version": PROTOCOL_VERSION,
        "evidence_status": "complete_v6_internal_post_development_run",
        "execution_date_independently_attested": False,
        "training_contract": dict(TRAINING_FIELDS),
        "primary": {key: primary[key] for key in numeric},
        "estimand": "equal_weight_mean_of_five_outer_fold_linked_group_aurocs",
        "uncertainty": "5000_fixed_fold_group_bootstrap_replicates_conditional_on_frozen_predictions",
        "probability_diagnostics": describe_group_predictions(groups),
        "runtime": runtime,
        "private_source_binding": {"result_sha256": digest, "result_size_bytes": len(raw),
            "artifact_sha256": dict(result["artifact_sha256"]), "artifact_size_bytes": dict(result["artifact_size_bytes"])},
        "verifier_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "legacy_numerical_validator_sha256": hashlib.sha256((ROOT / "validation/evaluation_utils.py").read_bytes()).hexdigest(),
        "privacy": {"row_level_data_included": False, "private_paths_included": False, "images_or_weights_included": False},
        "limitations": [
            "Only checkpoint/epoch-loss aggregation changed from v5; this is not model or population discovery.",
            "Gradient updates, BatchNorm adaptation, folds and the historically informed recipe remain fixed.",
            "The interval omits retraining and recipe/checkpoint/threshold-selection uncertainty.",
            "Byte/pixel identity checks and supplied IDs do not establish biological-subject independence.",
            "No external transportability, clinical utility or prospective calibration is established.",
            "A terminal bundle verifies retained records, not an independently attested execution date.",
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result")
    parser.add_argument("--expected-result-sha256")
    parser.add_argument("--describe-contract", action="store_true")
    args = parser.parse_args()
    if args.describe_contract:
        print(json.dumps({"protocol_version": PROTOCOL_VERSION, "training_contract": TRAINING_FIELDS,
            "trains_or_runs_inference": False, "writes_files": False, "output": "aggregate-only"}, indent=2))
        return
    if not args.result or not args.expected_result_sha256:
        parser.error("--result and --expected-result-sha256 are required")
    print(json.dumps(summarize(ROOT, args.result, args.expected_result_sha256), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
