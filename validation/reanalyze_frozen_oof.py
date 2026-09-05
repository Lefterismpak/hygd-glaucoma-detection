"""Read-only aggregate reanalysis of the frozen private HYGD OOF bundle.

The script never trains a model, runs inference, or emits row-level values. Private
filesystem locations are execution-environment details rather than scientific
inputs; override them with ``HYGD_PRIVATE_SOURCE_PREFIX`` and
``HYGD_PUBLIC_RECEIPT`` when the private bundle and public checkout differ.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import platform
import sys
from importlib.metadata import version as package_version
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from validation.evaluation_utils import (  # noqa: E402
    DEFAULT_SEED,
    EXPECTED_HYGD_CANONICAL_GROUP_FOLD_SHA256,
    EXPECTED_HYGD_CONTRACT_SHA256,
    EXPECTED_HYGD_DATASET_IDENTITY,
    EXPECTED_HYGD_DEDUPLICATED_OOF_IDENTITY_SHA256,
    GROUP_FOLD_ASSIGNMENT_ALGORITHM,
    OOF_IDENTITY_ALGORITHM,
    aggregate_groups,
    build_primary_report,
    cluster_bootstrap,
    fold_stratified_group_bootstrap,
    frame_metrics,
    group_fold_assignment_fingerprint,
    make_outer_folds,
    oof_identity_fingerprint,
    outer_fold_auc_summary,
    read_verified_file_bytes,
    sha256_file,
    validate_canonical_group_fold_assignment,
    validate_canonical_oof_identity,
)


EXPECTED_SOURCE_MANIFEST_SHA256 = (
    "1163d6ab73cbc3417be69289f81413399970735218508c822f9163c9a1a5d283"
)
SOURCE_SUFFIXES = {
    "audit": "_audit.json",
    "group_folds": "_group_folds.csv",
    "legacy_result_json": ".json",
    "oof_groups": "_oof_groups.csv",
    "oof_images": "_oof_images.csv",
}
CANONICAL_REANALYSIS_ARGV = [
    "python",
    "-B",
    "validation/reanalyze_frozen_oof.py",
    "--expected-source-manifest-sha256",
    EXPECTED_SOURCE_MANIFEST_SHA256,
    "--expected-dataset-contract-sha256",
    EXPECTED_HYGD_CONTRACT_SHA256,
    "--expected-oof-identity-sha256",
    EXPECTED_HYGD_DEDUPLICATED_OOF_IDENTITY_SHA256,
    "--expected-group-fold-assignment-sha256",
    EXPECTED_HYGD_CANONICAL_GROUP_FOLD_SHA256,
    "--outer-folds",
    "5",
    "--seed",
    str(DEFAULT_SEED),
    "--bootstrap",
    "5000",
    "--emit",
    "aggregate-only",
]
EXPECTED_EVIDENCE_STATUS = {
    "status": "preferred_internal_post_development_resampling",
    "preferred_internal_estimate": True,
    "canonical_v5_end_to_end_run": False,
    "model_retrained_for_reanalysis": False,
    "confirmatory_evidence": False,
    "single_site_single_camera": True,
    "transportability_established": False,
    "calibration_established": False,
    "clinical_utility_established": False,
    "deployment_ready": False,
    "public_reproducibility": "aggregate_receipt_only",
}
EXPECTED_PRIVACY = {
    "patient_or_image_identifiers": False,
    "row_level_predictions": False,
    "raw_images": False,
    "model_weights": False,
    "absolute_paths": False,
}
EXPECTED_LIMITATIONS = [
    "The model recipe was informed by earlier HYGD development, so this is post-development internal resampling rather than prospectively untouched validation.",
    "The legacy runner displayed each completed outer-fold outcome before later folds finished. No mid-run adaptation is documented, but its absence is needs-proof; the reported models were not trained by the outcome-blind v5 runner.",
    "Fold-specific thresholds and best epochs are post-hoc consistency-checked against the legacy result record and frozen OOF rows, but their inner-validation selection was not independently rerun.",
    "The legacy threshold tie-breaker differed from the current v5 helper, and the private bundle contains neither the inner-validation predictions nor checkpoint bytes needed to reproduce selection or model execution.",
    "The interval conditions on the frozen OOF predictions and recorded fold/decision choices; it excludes retraining, checkpoint and threshold selection, mid-run adaptation, and recipe-selection uncertainty.",
    "The data come from one hospital and one camera. Transportability, calibration, clinical utility, and deployment readiness remain unestablished.",
    "The private source hashes bind this aggregate to the audited workspace but cannot be reverified from the public repository without the private row-level artifacts.",
]


def _source_bundle(prefix: Path) -> tuple[dict[str, bytes], list[dict[str, object]], str]:
    content_by_role = {}
    records = []
    for role, suffix in sorted(SOURCE_SUFFIXES.items()):
        path = Path(str(prefix) + suffix)
        content, digest = read_verified_file_bytes(path)
        content_by_role[role] = content
        records.append(
            {"role": role, "sha256": digest, "size_bytes": len(content)}
        )
    payload = json.dumps(
        records, ensure_ascii=True, separators=(",", ":"), sort_keys=True
    ).encode("ascii")
    manifest = hashlib.sha256(
        b"HYGD_PRIVATE_SOURCE_MANIFEST_V1\n" + payload + b"\n"
    ).hexdigest()
    return content_by_role, records, manifest


def _read_csv(content: bytes, *, context: str) -> pd.DataFrame:
    try:
        return pd.read_csv(io.BytesIO(content))
    except (UnicodeDecodeError, pd.errors.ParserError, ValueError) as error:
        raise ValueError(f"{context} is not valid CSV") from error


def _assert_frame_equal(actual: pd.DataFrame, expected: pd.DataFrame, key: str) -> None:
    if list(actual.columns) != list(expected.columns) or len(actual) != len(expected):
        raise ValueError("Frozen aggregate artifact schema or row count is inconsistent")
    actual = actual.sort_values(key, kind="mergesort").reset_index(drop=True)
    expected = expected.sort_values(key, kind="mergesort").reset_index(drop=True)
    for column in actual.columns:
        if column in {"probability", "selected_threshold"}:
            left = pd.to_numeric(actual[column], errors="coerce").to_numpy(float)
            right = pd.to_numeric(expected[column], errors="coerce").to_numpy(float)
            if not np.isfinite(left).all() or not np.allclose(
                left, right, rtol=0.0, atol=1e-12
            ):
                raise ValueError("Frozen aggregate numeric values are inconsistent")
        elif not np.array_equal(
            actual[column].astype(str).to_numpy(),
            expected[column].astype(str).to_numpy(),
        ):
            raise ValueError("Frozen aggregate identifiers or values are inconsistent")


def _runtime_versions() -> dict[str, str]:
    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scikit_learn": package_version("scikit-learn"),
    }


def _decode_json(content: bytes, context: str) -> dict[str, object]:
    try:
        payload = json.loads(content.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{context} is not valid UTF-8 JSON") from error
    if not isinstance(payload, dict):
        raise ValueError(f"{context} must be a JSON object")
    return payload


def _assert_legacy_metrics(
    stored: object, actual: dict[str, object], context: str
) -> None:
    keys = {
        "n",
        "positives",
        "negatives",
        "auroc",
        "sensitivity",
        "specificity",
        "confusion_tn_fp_fn_tp",
    }
    if not isinstance(stored, dict) or set(stored) != keys:
        raise ValueError(f"{context} has an invalid metric schema")
    for key in keys:
        if key in {"auroc", "sensitivity", "specificity"}:
            if not np.isclose(float(stored[key]), float(actual[key]), rtol=0, atol=1e-12):
                raise ValueError(f"{context} does not match frozen OOF metrics")
        elif stored[key] != actual[key]:
            raise ValueError(f"{context} does not match frozen OOF counts")


def _validate_legacy_semantics(
    content: dict[str, bytes],
    images: pd.DataFrame,
    groups: pd.DataFrame,
    group_folds: pd.DataFrame,
) -> dict[str, object]:
    """Bind threshold/checkpoint provenance to the frozen legacy records."""

    legacy = _decode_json(content["legacy_result_json"], "Legacy result")
    expected_legacy_keys = {
        "protocol_version",
        "status",
        "completed_outer_folds",
        "configuration_fixed_before_outer_evaluation",
        "data_quality",
        "fold_results",
        "artifacts",
        "environment",
        "runtime_seconds",
        "overall_duplicate_aware_group_level",
        "overall_image_level",
    }
    if (
        set(legacy) != expected_legacy_keys
        or legacy.get("protocol_version") != "2026-07-11-v1"
        or legacy.get("status") != "complete"
        or legacy.get("completed_outer_folds") != [0, 1, 2, 3, 4]
    ):
        raise ValueError("Legacy result has an invalid completion schema")
    configuration = legacy.get("configuration_fixed_before_outer_evaluation")
    expected_configuration = {
        "model": "ImageNet ResNet18, layer4 + head fine-tuned",
        "configuration_origin": "Chosen during prior HYGD development, then frozen before this repair run",
        "residual_post_selection_risk": "Outer folds were hidden from this run, but the architecture and recipe were historically informed by HYGD development/test results",
        "epochs": 10,
        "batch_size": 32,
        "outer_folds": 5,
        "inner_validation_fraction": 0.15,
        "threshold_target_sensitivity": 0.95,
        "seed": DEFAULT_SEED,
        "device": "mps",
        "model_selection": "minimum inner-validation loss",
        "threshold_selection": "inner-validation group-level predictions only",
        "outer_test_visibility_during_training": "none",
    }
    if configuration != expected_configuration:
        raise ValueError("Legacy result fixed-configuration record is inconsistent")
    if legacy.get("artifacts") != {
        "audit": "results/internal_evaluation_repair_audit.json",
        "group_folds": "results/internal_evaluation_repair_group_folds.csv",
        "oof_images": "results/internal_evaluation_repair_oof_images.csv",
        "oof_groups": "results/internal_evaluation_repair_oof_groups.csv",
    }:
        raise ValueError("Legacy result artifact paths are inconsistent")

    fold_results = legacy.get("fold_results")
    if not isinstance(fold_results, list) or len(fold_results) != 5:
        raise ValueError("Legacy result requires five fold records")
    expected_fold_keys = {
        "outer_fold",
        "seed",
        "best_epoch_selected_on_inner_validation_loss",
        "selected_threshold_from_inner_group_validation",
        "counts",
        "outer_test_image_level",
        "outer_test_group_level",
        "training_history",
    }
    expected_count_keys = {
        "inner_train_images",
        "inner_train_groups",
        "inner_validation_images",
        "inner_validation_groups",
        "outer_test_images",
        "outer_test_groups",
    }
    for outer_fold, fold_record in enumerate(fold_results):
        if (
            not isinstance(fold_record, dict)
            or set(fold_record) != expected_fold_keys
            or fold_record.get("outer_fold") != outer_fold
            or fold_record.get("seed") != DEFAULT_SEED + outer_fold * 101
        ):
            raise ValueError("Legacy result fold ordering is inconsistent")
        threshold_record = fold_record.get(
            "selected_threshold_from_inner_group_validation"
        )
        if not isinstance(threshold_record, dict) or set(threshold_record) != {
            "threshold",
            "inner_validation_sensitivity",
            "inner_validation_specificity",
        }:
            raise ValueError("Legacy result threshold record is invalid")
        if any(
            not isinstance(threshold_record[key], (int, float))
            or isinstance(threshold_record[key], bool)
            or not np.isfinite(threshold_record[key])
            or not 0 <= float(threshold_record[key]) <= 1
            for key in threshold_record
        ) or float(threshold_record["inner_validation_sensitivity"]) < 0.95:
            raise ValueError("Legacy result threshold metadata is inconsistent")
        fold_images = images.loc[images["outer_fold"].astype(int) == outer_fold]
        fold_groups = groups.loc[groups["outer_fold"].astype(int) == outer_fold]
        thresholds = fold_images["selected_threshold"].astype(float).unique()
        if len(thresholds) != 1 or not np.isclose(
            thresholds[0], float(threshold_record["threshold"]), rtol=0, atol=1e-12
        ):
            raise ValueError("Frozen OOF threshold does not match legacy fold record")
        history = fold_record.get("training_history")
        best_epoch = fold_record.get("best_epoch_selected_on_inner_validation_loss")
        if not isinstance(history, list) or len(history) != 10:
            raise ValueError("Legacy result training history is incomplete")
        for epoch_number, row in enumerate(history, start=1):
            if (
                not isinstance(row, dict)
                or set(row) != {"epoch", "train_loss", "inner_validation_loss"}
                or row.get("epoch") != epoch_number
                or any(
                    not isinstance(row.get(key), (int, float))
                    or isinstance(row.get(key), bool)
                    or not np.isfinite(row[key])
                    or float(row[key]) < 0
                    for key in ("train_loss", "inner_validation_loss")
                )
            ):
                raise ValueError("Legacy result training history is inconsistent")
        losses = [float(row["inner_validation_loss"]) for row in history]
        if best_epoch != int(np.argmin(losses)) + 1:
            raise ValueError("Legacy best epoch does not match recorded inner loss")
        counts = fold_record.get("counts")
        if (
            not isinstance(counts, dict)
            or set(counts) != expected_count_keys
            or any(type(counts.get(key)) is not int or counts[key] <= 0 for key in counts)
            or counts.get("outer_test_images") != len(fold_images)
            or counts.get("outer_test_groups") != len(fold_groups)
            or counts["inner_train_images"]
            + counts["inner_validation_images"]
            + counts["outer_test_images"]
            != len(images)
            or counts["inner_train_groups"]
            + counts["inner_validation_groups"]
            + counts["outer_test_groups"]
            != len(group_folds)
        ):
            raise ValueError("Legacy fold counts do not match frozen OOF rows")
        _assert_legacy_metrics(
            fold_record.get("outer_test_image_level"),
            frame_metrics(fold_images),
            "Legacy image fold metrics",
        )
        _assert_legacy_metrics(
            fold_record.get("outer_test_group_level"),
            frame_metrics(fold_groups),
            "Legacy group fold metrics",
        )

    audit = _decode_json(content["audit"], "Legacy run-start audit")
    if set(audit) != {
        "protocol_version",
        "status",
        "data_quality",
        "split_summary",
        "group_fold_manifest",
    } or audit.get("protocol_version") != "2026-07-11-v1" or audit.get("status") != "running":
        raise ValueError("Legacy run-start audit schema or status is inconsistent")
    split_summary = [
        {
            "outer_fold": int(fold),
            "groups": int(len(rows)),
            "images": int(rows["n_images"].sum()),
            "positive_groups": int(rows["label"].sum()),
            "negative_groups": int((1 - rows["label"]).sum()),
        }
        for fold, rows in group_folds.groupby("outer_fold", sort=True)
    ]
    if (
        audit.get("data_quality") != legacy.get("data_quality")
        or
        audit.get("split_summary") != split_summary
        or audit.get("group_fold_manifest")
        != "results/internal_evaluation_repair_group_folds.csv"
    ):
        raise ValueError("Legacy run-start audit does not match frozen folds")
    return {
        "legacy_result_protocol_version": "2026-07-11-v1",
        "legacy_result_status": "complete",
        "legacy_run_start_audit_status": "running",
        "thresholds_match_oof_by_outer_fold": True,
        "best_epochs_match_recorded_inner_loss_argmin": True,
        "outer_fold_metrics_match_frozen_oof": True,
        "legacy_result_claimed_outer_test_visibility": "none",
        "separate_legacy_runner_source_review": "fold_outcomes_were_displayed_before_later_folds_completed",
        "absence_of_midrun_adaptation": "needs-proof",
        "legacy_training_code_in_source_manifest": False,
        "legacy_checkpoint_bytes_in_source_manifest": False,
        "threshold_selection_provenance": (
            "recorded_as_inner_group_validation_in_legacy_result_not_independently_rerun"
        ),
    }


def _public_dataset_identity() -> dict[str, object]:
    identity_keys = (
        "schema_version",
        "dataset_id",
        "dataset_name",
        "dataset_version",
        "doi",
        "labels_csv_sha256",
        "row_manifest_algorithm",
        "ordered_image_sha256_patient_id_label_inventory_sha256",
        "contract_sha256",
    )
    return {
        **{key: EXPECTED_HYGD_DATASET_IDENTITY[key] for key in identity_keys},
        "counts": {
            "source_rows": 747,
            "unique_image_hashes": 737,
            "supplied_patient_ids": 288,
            "source_negative_rows": 199,
            "source_positive_rows": 548,
            "removed_exact_duplicate_rows": 10,
            "duplicate_hash_groups": 10,
            "cross_patient_duplicate_hash_groups": 6,
            "independent_evaluation_groups": 283,
        },
    }


def _expected_public_receipt(
    *,
    aggregate: dict[str, object],
    source_records: list[dict[str, object]],
    source_manifest: str,
    legacy_semantics: dict[str, object],
) -> dict[str, object]:
    return {
        "schema_version": "hygd-public-internal-summary-v1",
        "protocol_version": "2026-09-04-v5",
        "evidence_status": EXPECTED_EVIDENCE_STATUS,
        "dataset_identity": _public_dataset_identity(),
        "execution_origin": {
            "model_run_date_assertion": "2026-07-11",
            "model_run_protocol_version": "2026-07-11-v1",
            "legacy_run_start_audit_terminal_status": "running",
            "completion_basis": "Private complete-result JSON, semantic threshold/history checks, and independent OOF integrity recomputation",
            "public_time_attestation": "not_available",
            "model_retrained_for_publication": False,
            "aggregate_reanalysis_date": "2026-09-04",
            "aggregate_reanalysis_protocol_version": "2026-09-04-v5",
            "aggregate_reanalysis_implementation_sha256": sha256_file(
                ROOT / "validation" / "evaluation_utils.py"
            ),
            "aggregate_reanalysis_script_sha256": sha256_file(Path(__file__)),
            "aggregate_reanalysis_argv": CANONICAL_REANALYSIS_ARGV,
            "aggregate_reanalysis_runtime": _runtime_versions(),
            "aggregate_reanalysis_status": "verified_private_read_only",
            "model_training_or_inference_executed_for_reanalysis": False,
            "current_v5_code_used_for_training": False,
            "private_source_only": True,
            "scope": "Aggregate-only reanalysis of frozen legacy out-of-fold predictions; not an end-to-end v5 model rerun.",
        },
        "primary": aggregate["primary"],
        "secondary_descriptive": aggregate["secondary_descriptive"],
        "private_source_binding": {
            "schema_version": "hygd-private-source-binding-v1",
            "manifest_algorithm": "sha256(HYGD_PRIVATE_SOURCE_MANIFEST_V1 newline + canonical JSON source-file records + newline)",
            "source_manifest_sha256": source_manifest,
            "oof_identity_algorithm": OOF_IDENTITY_ALGORITHM,
            "oof_identity_fields": [
                "image_name",
                "patient_id",
                "evaluation_group",
                "sha256",
                "label",
            ],
            "oof_identity_rows": 737,
            "deduplicated_oof_identity_sha256": EXPECTED_HYGD_DEDUPLICATED_OOF_IDENTITY_SHA256,
            "canonical_group_fold_assignment_algorithm": GROUP_FOLD_ASSIGNMENT_ALGORITHM,
            "canonical_group_fold_assignment_fields": [
                "evaluation_group",
                "label",
                "n_images",
                "outer_fold",
            ],
            "canonical_group_fold_assignment_sha256": EXPECTED_HYGD_CANONICAL_GROUP_FOLD_SHA256,
            "outer_fold_assignment_groups": 283,
            "outer_fold_assignment_contract": {
                "splitter": "sklearn.model_selection.StratifiedKFold",
                "scikit_learn_version": _runtime_versions()["scikit_learn"],
                "n_splits": 5,
                "shuffle": True,
                "seed": DEFAULT_SEED,
                "stable_group_order": "evaluation_group ascending",
            },
            "observed_private_mapping_match": True,
            "legacy_semantic_checks": legacy_semantics,
            "source_files": source_records,
            "row_level_artifacts_published": False,
        },
        "privacy": EXPECTED_PRIVACY,
        "limitations": EXPECTED_LIMITATIONS,
    }


def validate_public_receipt(
    receipt: object,
    *,
    aggregate: dict[str, object],
    source_records: list[dict[str, object]],
    source_manifest: str,
    legacy_semantics: dict[str, object],
) -> None:
    """Require exact schema and values for every material public claim."""

    expected = _expected_public_receipt(
        aggregate=aggregate,
        source_records=source_records,
        source_manifest=source_manifest,
        legacy_semantics=legacy_semantics,
    )
    if receipt != expected:
        raise ValueError("Public aggregate receipt does not match the full derived claim")


def reanalyze(
    args: argparse.Namespace, *, validate_receipt_bytes: bool = True
) -> dict[str, object]:
    if (
        args.emit != "aggregate-only"
        or args.expected_source_manifest_sha256 != EXPECTED_SOURCE_MANIFEST_SHA256
        or args.expected_dataset_contract_sha256 != EXPECTED_HYGD_CONTRACT_SHA256
        or args.expected_oof_identity_sha256
        != EXPECTED_HYGD_DEDUPLICATED_OOF_IDENTITY_SHA256
        or args.expected_group_fold_assignment_sha256
        != EXPECTED_HYGD_CANONICAL_GROUP_FOLD_SHA256
        or args.outer_folds != 5
        or args.seed != DEFAULT_SEED
        or args.bootstrap != 5_000
    ):
        raise ValueError("Reanalysis arguments do not match the locked contract")

    source_prefix = Path(
        os.environ.get(
            "HYGD_PRIVATE_SOURCE_PREFIX", "results/internal_evaluation_repair"
        )
    )
    receipt_path = Path(
        os.environ.get(
            "HYGD_PUBLIC_RECEIPT",
            "results/repaired_internal_evaluation_summary.json",
        )
    )
    content, source_records, source_manifest = _source_bundle(source_prefix)
    if source_manifest != args.expected_source_manifest_sha256:
        raise ValueError("Private source manifest does not match the locked contract")

    images = _read_csv(content["oof_images"], context="Frozen OOF images")
    validate_canonical_oof_identity(images)
    if oof_identity_fingerprint(images) != args.expected_oof_identity_sha256:
        raise ValueError("Frozen OOF identity does not match the locked contract")
    required_numeric = [
        "label",
        "outer_fold",
        "probability",
        "selected_threshold",
        "predicted_class",
    ]
    for column in required_numeric:
        if column not in images:
            raise ValueError("Frozen OOF images are missing required fields")
        images[column] = pd.to_numeric(images[column], errors="coerce")
    if (
        images[required_numeric].isna().any().any()
        or not images["label"].isin([0, 1]).all()
        or not images["predicted_class"].isin([0, 1]).all()
        or not images["probability"].between(0, 1).all()
        or not images["selected_threshold"].between(0, 1).all()
        or not np.array_equal(
            images["predicted_class"].astype(int).to_numpy(),
            (images["probability"] >= images["selected_threshold"])
            .astype(int)
            .to_numpy(),
        )
    ):
        raise ValueError("Frozen OOF image values are inconsistent")

    group_folds = _read_csv(content["group_folds"], context="Frozen group folds")
    validate_canonical_group_fold_assignment(group_folds)
    if (
        group_fold_assignment_fingerprint(group_folds)
        != args.expected_group_fold_assignment_sha256
    ):
        raise ValueError("Frozen group-fold identity does not match the locked contract")
    canonical_folds = make_outer_folds(images, args.outer_folds, args.seed)
    _assert_frame_equal(
        group_folds[canonical_folds.columns], canonical_folds, "evaluation_group"
    )
    fold_map = group_folds.set_index("evaluation_group")["outer_fold"]
    expected_image_folds = images["evaluation_group"].map(fold_map)
    if expected_image_folds.isna().any() or not np.array_equal(
        images["outer_fold"].astype(int).to_numpy(),
        expected_image_folds.astype(int).to_numpy(),
    ):
        raise ValueError("Frozen OOF rows do not match the canonical fold assignment")

    groups = aggregate_groups(images)
    groups["predicted_class"] = (
        groups["probability"] >= groups["selected_threshold"]
    ).astype(int)
    stored_groups = _read_csv(content["oof_groups"], context="Frozen OOF groups")
    _assert_frame_equal(
        stored_groups[groups.columns], groups, "evaluation_group"
    )
    legacy_semantics = _validate_legacy_semantics(
        content, images, groups, group_folds
    )
    report = build_primary_report(
        outer_fold_auc=outer_fold_auc_summary(groups),
        fold_stratified_ci=fold_stratified_group_bootstrap(
            groups, args.bootstrap, args.seed + 9_003
        ),
        pooled_group_metrics=frame_metrics(groups),
        pooled_group_cluster_ci=cluster_bootstrap(
            groups, "evaluation_group", args.bootstrap, args.seed + 9_002
        ),
        image_metrics=frame_metrics(images),
        image_cluster_ci=cluster_bootstrap(
            images, "evaluation_group", args.bootstrap, args.seed + 9_001
        ),
    )
    report["primary"]["classification_estimand"] = (
        "pooled_oof_binary_decisions_using_fold_specific_thresholds_"
        "recorded_as_inner_validation_selected_in_legacy_result"
    )
    aggregate = {
        "primary": report["primary"],
        "secondary_descriptive": {
            "pooled_oof_group": {
                key: value
                for key, value in report["pooled_oof_group_descriptive"].items()
                if key != "warning"
            },
            "image_level": {
                key: value
                for key, value in report["secondary_image_level"].items()
                if key != "warning"
            },
        },
    }

    expected_receipt = _expected_public_receipt(
        aggregate=aggregate,
        source_records=source_records,
        source_manifest=source_manifest,
        legacy_semantics=legacy_semantics,
    )
    if not validate_receipt_bytes:
        return expected_receipt

    receipt_bytes, _ = read_verified_file_bytes(receipt_path)
    receipt = _decode_json(receipt_bytes, "Public aggregate receipt")
    if receipt != expected_receipt:
        raise ValueError("Public aggregate receipt does not match the full derived claim")

    return {
        "status": "verified_private_read_only",
        "model_training_or_inference_executed": False,
        "source_manifest_sha256": source_manifest,
        "oof_identity_sha256": args.expected_oof_identity_sha256,
        "group_fold_assignment_sha256": (
            args.expected_group_fold_assignment_sha256
        ),
        "legacy_semantic_checks": legacy_semantics,
        **aggregate,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--describe-contract", action="store_true")
    parser.add_argument(
        "--print-derived-receipt",
        action="store_true",
        help="Read private frozen inputs and print the aggregate-only expected receipt without writing files.",
    )
    parser.add_argument(
        "--expected-source-manifest-sha256",
        default=EXPECTED_SOURCE_MANIFEST_SHA256,
    )
    parser.add_argument(
        "--expected-dataset-contract-sha256",
        default=EXPECTED_HYGD_CONTRACT_SHA256,
    )
    parser.add_argument(
        "--expected-oof-identity-sha256",
        default=EXPECTED_HYGD_DEDUPLICATED_OOF_IDENTITY_SHA256,
    )
    parser.add_argument(
        "--expected-group-fold-assignment-sha256",
        default=EXPECTED_HYGD_CANONICAL_GROUP_FOLD_SHA256,
    )
    parser.add_argument("--outer-folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--bootstrap", type=int, default=5_000)
    parser.add_argument("--emit", choices=["aggregate-only"], default="aggregate-only")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.describe_contract:
        print(
            json.dumps(
                {
                    "argv": CANONICAL_REANALYSIS_ARGV,
                    "private_locations_in_scientific_contract": False,
                    "writes_files": False,
                    "trains_or_runs_inference": False,
                },
                indent=2,
            )
        )
        return
    if args.print_derived_receipt:
        print(
            json.dumps(
                reanalyze(args, validate_receipt_bytes=False),
                indent=2,
                allow_nan=False,
            )
        )
        return
    print(json.dumps(reanalyze(args), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
