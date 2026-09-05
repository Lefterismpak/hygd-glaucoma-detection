"""Pure utilities for leakage-resistant HYGD internal evaluation.

This module intentionally has no Torch or Torchvision dependency. It owns the
scientific bookkeeping that must be testable in a lightweight, dataset-free CI:
exact-duplicate grouping, nested group splits, threshold selection, aggregation,
cluster bootstrap, artifact policy, and result-schema guardrails.
"""

from __future__ import annotations

import copy
import hashlib
import io
import json
import os
import re
import secrets
import stat
import unicodedata
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, roc_auc_score
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit


DEFAULT_SEED = 20_260_711
PROTOCOL_VERSION = "2026-09-04-v5"
_EXPECTED_HYGD_DATASET_IDENTITY_BASE = {
    "schema_version": "hygd-dataset-identity-v1",
    "dataset_id": "physionet-hygd",
    "dataset_name": "Hillel Yaffe Glaucoma Dataset",
    "dataset_version": "1.1.0",
    "doi": "10.13026/m92s-0z95",
    "labels_csv_sha256": (
        "c2f4e6e756f6f5f9d9d900f613cf329116da9897877e480cc9782bc042d4c0f5"
    ),
    "row_manifest_algorithm": "hygd-row-manifest-v1",
    "ordered_image_sha256_patient_id_label_inventory_sha256": (
        "1187ea38d89ee414442113869d5c6d8c575530e935b8169c8c037b3f4c53a291"
    ),
    "inventory_rows": 747,
    "unique_image_hashes": 737,
    "patient_ids": 288,
    "negative_rows": 199,
    "positive_rows": 548,
}
EXPECTED_HYGD_CONTRACT_SHA256 = hashlib.sha256(
    json.dumps(
        _EXPECTED_HYGD_DATASET_IDENTITY_BASE,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")
).hexdigest()
EXPECTED_HYGD_DATASET_IDENTITY = {
    **_EXPECTED_HYGD_DATASET_IDENTITY_BASE,
    "contract_sha256": EXPECTED_HYGD_CONTRACT_SHA256,
}
EXPECTED_HYGD_DATA_QUALITY = {
    "input_images": 747,
    "unique_image_hashes": 737,
    "removed_exact_duplicate_rows": 10,
    "duplicate_hash_groups": 10,
    "cross_patient_duplicate_hash_groups": 6,
    "input_patient_ids": 288,
    "negative_rows": 199,
    "positive_rows": 548,
    "independent_evaluation_groups": 283,
    "label_conflicts": 0,
    "missing_images": 0,
}
OOF_IDENTITY_ALGORITHM = "hygd-deduplicated-oof-identity-v1"
EXPECTED_HYGD_DEDUPLICATED_OOF_IDENTITY_SHA256 = (
    "fbbaa74909dbc242843b708ca6073214e043e3e45a26856c827ce744024f849e"
)
GROUP_FOLD_ASSIGNMENT_ALGORITHM = "hygd-canonical-group-fold-v1"
EXPECTED_HYGD_CANONICAL_GROUP_FOLD_SHA256 = (
    "c9d142c3fcfb213a6c1815468576694aec94db076e9f929c0d7018b0b2333432"
)
CANONICAL_RUN_CONFIG = {
    "outer_folds": 5,
    "epochs": 10,
    "batch_size": 32,
    "seed": DEFAULT_SEED,
    "inner_validation_fraction": 0.15,
    "threshold_target_sensitivity": 0.95,
    "bootstrap_samples": 5_000,
}
PREFERRED_INTERNAL_METRIC_SOURCE = (
    "duplicate_aware_group_macro_outer_fold_auc_with_"
    "fold_stratified_group_bootstrap"
)


def _canonical_config_mismatches(configuration: Mapping[str, object]) -> list[str]:
    """Return locked canonical fields that are missing or have the wrong value."""

    mismatches = []
    for key, expected in CANONICAL_RUN_CONFIG.items():
        if key not in configuration:
            mismatches.append(key)
            continue
        actual = configuration[key]
        if isinstance(expected, int):
            matches = type(actual) is int and actual == expected
        else:
            matches = type(actual) is float and actual == expected
        if not matches:
            mismatches.append(key)
    return mismatches


def classify_run_mode(
    *,
    folds: int,
    epochs: int,
    batch_size: int,
    seed: int,
    inner_val_fraction: float,
    target_sensitivity: float,
    bootstrap: int,
    max_folds: int | None,
    audit_only: bool,
) -> str:
    """Classify a run before any device access, data read, or artifact write.

    Only the locked full recipe may emit the preferred complete-result schema.
    Audit-only and proper-subset smoke runs remain explicitly noncanonical and
    may use diagnostic settings.
    """

    if audit_only:
        if max_folds is not None:
            raise ValueError("audit_only cannot be combined with max_folds")
        return "audit_only"
    if max_folds is not None:
        if max_folds <= 0:
            raise ValueError("max_folds must be positive")
        if max_folds >= folds:
            raise ValueError("max_folds must be a proper subset of outer folds")
        return "partial_smoke_run"

    requested = {
        "outer_folds": folds,
        "epochs": epochs,
        "batch_size": batch_size,
        "seed": seed,
        "inner_validation_fraction": inner_val_fraction,
        "threshold_target_sensitivity": target_sensitivity,
        "bootstrap_samples": bootstrap,
    }
    mismatches = _canonical_config_mismatches(requested)
    if mismatches:
        raise ValueError(
            "The complete preferred-result configuration is locked; changed fields: "
            + ", ".join(mismatches)
            + ". Use --max-folds for a noncanonical smoke run."
        )
    return "complete"


def fold_completion_message(completed_fold: int, total_folds: int) -> str:
    """Return outcome-blind progress while later outer folds remain pending."""

    if total_folds < 1 or completed_fold < 0 or completed_fold >= total_folds:
        raise ValueError("completed_fold must identify one of the planned folds")
    return (
        f"finalized outer fold {completed_fold + 1}/{total_folds}; "
        "held back fold outcomes until terminal bundle publication"
    )


class UnionFind:
    """Minimal union-find for transitively linking supplied patient IDs."""

    def __init__(self, values: Iterable[object]):
        self.parent = {str(value): str(value) for value in values}

    def find(self, value: object) -> str:
        value = str(value)
        if value not in self.parent:
            raise KeyError(f"Unknown union-find value: {value}")
        root = value
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[value] != value:
            parent = self.parent[value]
            self.parent[value] = root
            value = parent
        return root

    def union(self, left: object, right: object) -> None:
        left_root, right_root = self.find(left), self.find(right)
        if left_root != right_root:
            self.parent[right_root] = left_root


def _require_columns(frame: pd.DataFrame, columns: Sequence[str], context: str) -> None:
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise ValueError(f"{context} is missing required columns: {missing}")


def validate_external_label_frame(
    frame: pd.DataFrame, *, context: str
) -> pd.DataFrame:
    """Normalize external labels without coercing invalid rows into evidence."""

    _require_columns(frame, ["image_path", "patient_id", "label"], context)
    normalized = frame.copy()
    for column in ("image_path", "patient_id"):
        if normalized[column].isna().any():
            raise ValueError(f"{context} contains missing {column} values")
        normalized[column] = normalized[column].astype(str).str.strip()
        if normalized[column].eq("").any():
            raise ValueError(f"{context} contains blank {column} values")
    labels = pd.to_numeric(normalized["label"], errors="coerce")
    if (
        labels.isna().any()
        or not np.isfinite(labels.to_numpy(dtype=float)).all()
        or not np.equal(labels, np.floor(labels)).all()
        or not labels.isin([0, 1]).all()
    ):
        raise ValueError(f"{context} labels must be binary integers 0/1")
    normalized["label"] = labels.astype(int)
    return normalized


def _patient_sort_key(patient_id: object) -> tuple[int, object]:
    value = str(patient_id)
    return (0, int(value)) if value.isdigit() else (1, value)


def read_verified_file_bytes(
    path: str | Path,
    *,
    expected_sha256: str | None = None,
) -> tuple[bytes, str]:
    """Return bytes and SHA-256 from the same no-follow regular-file descriptor.

    Callers that decode or parse the returned bytes therefore consume exactly the
    content that was hashed. A single-link requirement excludes mutable aliases
    through hard links; ``O_NOFOLLOW`` excludes final-component symlinks.
    """

    if expected_sha256 is not None and not _is_sha256(expected_sha256):
        raise ValueError("expected_sha256 must be lowercase 64-hex")
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    flags |= getattr(os, "O_NONBLOCK", 0)
    try:
        file_fd = os.open(Path(path), flags)
    except OSError as error:
        raise ValueError("Verified input must be a readable no-follow regular file") from error
    try:
        before = os.fstat(file_fd)
        if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
            raise ValueError("Verified input must be a single-link regular file")
        content = bytearray()
        digest = hashlib.sha256()
        while True:
            block = os.read(file_fd, 1 << 20)
            if not block:
                break
            content.extend(block)
            digest.update(block)
        after = os.fstat(file_fd)
        if (
            before.st_dev,
            before.st_ino,
            before.st_size,
            before.st_mtime_ns,
            before.st_ctime_ns,
        ) != (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
            after.st_ctime_ns,
        ) or len(content) != after.st_size:
            raise ValueError("Verified input changed while it was being read")
        actual_sha256 = digest.hexdigest()
        if expected_sha256 is not None and actual_sha256 != expected_sha256:
            raise ValueError("Verified input SHA-256 mismatch")
        return bytes(content), actual_sha256
    finally:
        os.close(file_fd)


def sha256_file(path: str | Path) -> str:
    _, digest = read_verified_file_bytes(path)
    return digest


def _is_sha256(value: object) -> bool:
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def _is_finite_number(value: object) -> bool:
    return (
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and bool(np.isfinite(value))
    )


def _validate_interval(value: object, context: str) -> None:
    if (
        not isinstance(value, list)
        or len(value) != 2
        or any(not _is_finite_number(bound) for bound in value)
        or not 0 <= float(value[0]) <= float(value[1]) <= 1
    ):
        raise ValueError(f"{context} must be a finite ordered interval in [0, 1]")


def _validate_metric_summary(metrics: Mapping[str, object], context: str) -> None:
    required = {
        "n",
        "positives",
        "negatives",
        "auroc",
        "sensitivity",
        "specificity",
        "confusion_tn_fp_fn_tp",
    }
    if not isinstance(metrics, Mapping) or not required.issubset(metrics):
        raise ValueError(f"{context} is missing metric fields")
    if any(
        type(metrics.get(key)) is not int or metrics[key] <= 0
        for key in ("n", "positives", "negatives")
    ) or metrics["positives"] + metrics["negatives"] != metrics["n"]:
        raise ValueError(f"{context} has inconsistent metric counts")
    if any(
        not _is_finite_number(metrics.get(key))
        or not 0 <= float(metrics[key]) <= 1
        for key in ("auroc", "sensitivity", "specificity")
    ):
        raise ValueError(f"{context} has invalid metric values")
    confusion = metrics.get("confusion_tn_fp_fn_tp")
    if (
        not isinstance(confusion, list)
        or len(confusion) != 4
        or any(type(value) is not int or value < 0 for value in confusion)
        or sum(confusion) != metrics["n"]
        or confusion[2] + confusion[3] != metrics["positives"]
        or confusion[0] + confusion[1] != metrics["negatives"]
        or not np.isclose(
            float(metrics["sensitivity"]),
            confusion[3] / metrics["positives"],
            rtol=0.0,
            atol=1e-12,
        )
        or not np.isclose(
            float(metrics["specificity"]),
            confusion[0] / metrics["negatives"],
            rtol=0.0,
            atol=1e-12,
        )
    ):
        raise ValueError(f"{context} has inconsistent confusion metrics")


def inventory_fingerprint(metadata: pd.DataFrame) -> str:
    """Hash a relocatable, order-independent multiset of HYGD source records."""

    _require_columns(metadata, ["sha256", "patient_id", "label"], "source inventory")
    if metadata.empty or metadata[["sha256", "patient_id", "label"]].isna().any().any():
        raise ValueError("source inventory fields cannot be empty or missing")
    if not metadata["sha256"].map(_is_sha256).all():
        raise ValueError("source inventory sha256 values must be lowercase 64-hex")
    patients = metadata["patient_id"].astype(str).map(
        lambda value: unicodedata.normalize("NFC", value.strip())
    )
    if patients.str.strip().eq("").any():
        raise ValueError("source inventory patient_id cannot be blank")
    if patients.map(lambda value: any(ord(character) < 32 for character in value)).any():
        raise ValueError("source inventory patient_id cannot contain control characters")
    labels = pd.to_numeric(metadata["label"], errors="coerce")
    if labels.isna().any() or not labels.isin([0, 1]).all():
        raise ValueError("source inventory labels must be binary")
    records = sorted(
        [
            [str(digest), str(patient), int(label)]
            for digest, patient, label in zip(
                metadata["sha256"], patients, labels.astype(int)
            )
        ],
        key=lambda item: (item[0], item[1], item[2]),
    )
    digest = hashlib.sha256(b"HYGD_ROW_MANIFEST_V1\n")
    for record in records:
        digest.update(
            json.dumps(record, ensure_ascii=True, separators=(",", ":")).encode(
                "ascii"
            )
        )
        digest.update(b"\n")
    return digest.hexdigest()


def oof_identity_fingerprint(frame: pd.DataFrame) -> str:
    """Hash the official deduplicated row identities without exposing them."""

    columns = ["image_name", "patient_id", "evaluation_group", "sha256", "label"]
    _require_columns(frame, columns, "deduplicated OOF identity")
    if frame.empty or frame[columns].isna().any().any():
        raise ValueError("deduplicated OOF identity fields cannot be empty or missing")
    normalized = frame[columns].copy()
    for column in ("image_name", "patient_id", "evaluation_group", "sha256"):
        normalized[column] = normalized[column].astype(str).map(
            lambda value: unicodedata.normalize("NFC", value.strip())
        )
        if normalized[column].eq("").any() or normalized[column].map(
            lambda value: any(ord(character) < 32 for character in value)
        ).any():
            raise ValueError("deduplicated OOF identity contains invalid strings")
    if (
        normalized["image_name"].duplicated().any()
        or not normalized["image_name"].map(
            lambda value: Path(value).name == value and value not in {".", ".."}
        ).all()
        or normalized["sha256"].duplicated().any()
        or not normalized["sha256"].map(_is_sha256).all()
    ):
        raise ValueError("deduplicated OOF identity contains invalid names or hashes")
    labels = pd.to_numeric(normalized["label"], errors="coerce")
    if labels.isna().any() or not labels.isin([0, 1]).all():
        raise ValueError("deduplicated OOF identity labels must be binary")
    records = sorted(
        [
            [
                str(row.image_name),
                str(row.patient_id),
                str(row.evaluation_group),
                str(row.sha256),
                int(label),
            ]
            for row, label in zip(
                normalized.itertuples(index=False), labels.astype(int)
            )
        ],
        key=lambda record: (record[3], record[0], record[1], record[2], record[4]),
    )
    digest = hashlib.sha256(b"HYGD_DEDUPLICATED_OOF_IDENTITY_V1\n")
    for record in records:
        digest.update(
            json.dumps(record, ensure_ascii=True, separators=(",", ":")).encode(
                "ascii"
            )
        )
        digest.update(b"\n")
    return digest.hexdigest()


def validate_canonical_oof_identity(frame: pd.DataFrame) -> None:
    """Fail closed unless deduplicated OOF identities match official HYGD v1.1.0."""

    if oof_identity_fingerprint(frame) != EXPECTED_HYGD_DEDUPLICATED_OOF_IDENTITY_SHA256:
        raise ValueError("Terminal OOF identity does not match official HYGD v1.1.0")


def group_fold_assignment_fingerprint(
    frame: pd.DataFrame,
    *,
    n_splits: int = CANONICAL_RUN_CONFIG["outer_folds"],
    seed: int = CANONICAL_RUN_CONFIG["seed"],
    oof_identity_sha256: str = EXPECTED_HYGD_DEDUPLICATED_OOF_IDENTITY_SHA256,
) -> str:
    """Hash the protocol-level group-to-fold assignment without exposing IDs."""

    columns = ["evaluation_group", "label", "n_images", "outer_fold"]
    _require_columns(frame, columns, "canonical group-fold assignment")
    if (
        type(n_splits) is not int
        or n_splits < 2
        or type(seed) is not int
        or seed < 0
        or not _is_sha256(oof_identity_sha256)
        or frame.empty
        or frame[columns].isna().any().any()
    ):
        raise ValueError("canonical group-fold assignment contract is invalid")
    groups = frame["evaluation_group"].astype(str).map(
        lambda value: unicodedata.normalize("NFC", value.strip())
    )
    if (
        groups.eq("").any()
        or groups.duplicated().any()
        or groups.map(lambda value: any(ord(character) < 32 for character in value)).any()
    ):
        raise ValueError("canonical group-fold assignment identifiers are invalid")
    labels = pd.to_numeric(frame["label"], errors="coerce")
    image_counts = pd.to_numeric(frame["n_images"], errors="coerce")
    folds = pd.to_numeric(frame["outer_fold"], errors="coerce")
    if (
        labels.isna().any()
        or not labels.isin([0, 1]).all()
        or image_counts.isna().any()
        or not np.equal(image_counts, np.floor(image_counts)).all()
        or (image_counts <= 0).any()
        or folds.isna().any()
        or not np.equal(folds, np.floor(folds)).all()
        or set(folds.astype(int)) != set(range(n_splits))
    ):
        raise ValueError("canonical group-fold assignment values are invalid")
    contract = {
        "algorithm": GROUP_FOLD_ASSIGNMENT_ALGORITHM,
        "n_splits": n_splits,
        "oof_identity_sha256": oof_identity_sha256,
        "seed": seed,
    }
    records = sorted(
        [
            [str(group), int(label), int(n_images), int(fold)]
            for group, label, n_images, fold in zip(
                groups, labels, image_counts, folds
            )
        ],
        key=lambda record: record[0],
    )
    digest = hashlib.sha256(b"HYGD_GROUP_FOLD_ASSIGNMENT_V1\n")
    digest.update(
        json.dumps(contract, ensure_ascii=True, separators=(",", ":"), sort_keys=True)
        .encode("ascii")
    )
    digest.update(b"\n")
    for record in records:
        digest.update(
            json.dumps(record, ensure_ascii=True, separators=(",", ":")).encode(
                "ascii"
            )
        )
        digest.update(b"\n")
    return digest.hexdigest()


def validate_canonical_group_fold_assignment(frame: pd.DataFrame) -> None:
    """Fail closed unless the official locked group-to-fold assignment matches."""

    if (
        group_fold_assignment_fingerprint(frame)
        != EXPECTED_HYGD_CANONICAL_GROUP_FOLD_SHA256
    ):
        raise ValueError("Terminal canonical group-fold assignment is inconsistent")


def build_dataset_identity(
    data_quality: Mapping[str, object],
    *,
    labels_csv_sha256: str,
) -> dict:
    """Build a path-free dataset identity record and its canonical-match verdict."""

    if not _is_sha256(labels_csv_sha256):
        raise ValueError("labels_csv_sha256 must be lowercase 64-hex")
    try:
        identity = {
            "schema_version": "hygd-dataset-identity-v1",
            "dataset_id": "physionet-hygd",
            "dataset_name": "Hillel Yaffe Glaucoma Dataset",
            "dataset_version": "1.1.0",
            "doi": "10.13026/m92s-0z95",
            "labels_csv_sha256": labels_csv_sha256,
            "row_manifest_algorithm": "hygd-row-manifest-v1",
            "ordered_image_sha256_patient_id_label_inventory_sha256": data_quality[
                "ordered_image_sha256_patient_id_label_inventory_sha256"
            ],
            "inventory_rows": int(data_quality["input_images"]),
            "unique_image_hashes": int(data_quality["unique_image_hashes"]),
            "patient_ids": int(data_quality["input_patient_ids"]),
            "negative_rows": int(data_quality["negative_rows"]),
            "positive_rows": int(data_quality["positive_rows"]),
            "contract_sha256": EXPECTED_HYGD_CONTRACT_SHA256,
        }
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("data quality is missing dataset-identity fields") from error
    identity["canonical_hygd_v1_1_0_match"] = all(
        identity.get(key) == expected
        for key, expected in EXPECTED_HYGD_DATASET_IDENTITY.items()
    )
    return identity


def validate_canonical_dataset_identity(identity: Mapping[str, object]) -> None:
    """Fail closed unless the exact expected HYGD v1.1.0 content is present."""

    if not isinstance(identity, Mapping):
        raise ValueError("Canonical result requires HYGD v1.1.0 dataset identity")
    expected_keys = set(EXPECTED_HYGD_DATASET_IDENTITY) | {
        "canonical_hygd_v1_1_0_match"
    }
    mismatches = [
        key
        for key, expected in EXPECTED_HYGD_DATASET_IDENTITY.items()
        if identity.get(key) != expected
    ]
    for key in (
        "inventory_rows",
        "unique_image_hashes",
        "patient_ids",
        "negative_rows",
        "positive_rows",
    ):
        if type(identity.get(key)) is not int:
            mismatches.append(key)
    if set(identity) != expected_keys:
        mismatches.append("schema_fields")
    if identity.get("canonical_hygd_v1_1_0_match") is not True:
        mismatches.append("canonical_hygd_v1_1_0_match")
    if mismatches:
        raise ValueError(
            "Canonical result requires the exact HYGD v1.1.0 content; invalid fields: "
            + ", ".join(sorted(set(mismatches)))
        )


def validate_canonical_data_quality(data_quality: Mapping[str, object]) -> None:
    """Require every ratified HYGD v1.1.0 aggregate before canonical writes."""

    if not isinstance(data_quality, Mapping):
        raise ValueError("Canonical result requires HYGD v1.1.0 data_quality")
    mismatches = [
        key
        for key, expected in EXPECTED_HYGD_DATA_QUALITY.items()
        if data_quality.get(key) != expected
    ]
    for key in EXPECTED_HYGD_DATA_QUALITY:
        if type(data_quality.get(key)) is not int:
            mismatches.append(key)
    if (
        data_quality.get(
            "ordered_image_sha256_patient_id_label_inventory_sha256"
        )
        != EXPECTED_HYGD_DATASET_IDENTITY[
            "ordered_image_sha256_patient_id_label_inventory_sha256"
        ]
    ):
        mismatches.append(
            "ordered_image_sha256_patient_id_label_inventory_sha256"
        )
    duplicate_groups = data_quality.get("duplicate_groups")
    if not isinstance(duplicate_groups, list) or len(duplicate_groups) != int(
        EXPECTED_HYGD_DATA_QUALITY["duplicate_hash_groups"]
    ):
        mismatches.append("duplicate_groups")
    if mismatches:
        raise ValueError(
            "Canonical result requires exact HYGD v1.1.0 data_quality; invalid fields: "
            + ", ".join(sorted(set(mismatches)))
        )


def _assert_group_invariants(
    frame: pd.DataFrame,
    group_column: str,
    invariant_columns: Sequence[str],
) -> None:
    _require_columns(frame, [group_column, *invariant_columns], "grouped frame")
    for column in invariant_columns:
        counts = frame.groupby(group_column, dropna=False)[column].nunique(dropna=False)
        bad = counts[counts != 1]
        if not bad.empty:
            raise ValueError(
                f"{column} must be invariant within each {group_column}; "
                f"violations: {bad.index.astype(str).tolist()}"
            )


def prepare_duplicate_aware_metadata(
    metadata: pd.DataFrame,
) -> tuple[pd.DataFrame, dict]:
    """Hash, link, and deduplicate normalized image metadata.

    ``metadata`` must already use the normalized columns produced by
    ``src.data_utils.load_dataset_metadata``. Exact files are counted once.
    Supplied patient IDs that share any exact hash are linked transitively into
    one independent ``evaluation_group``.
    """

    _require_columns(
        metadata,
        ["image_path", "patient_id", "label"],
        "metadata",
    )
    if metadata.empty:
        raise ValueError("metadata must contain at least one image")

    metadata = metadata.copy()
    if metadata["patient_id"].isna().any():
        raise ValueError("patient_id cannot be missing")
    numeric_labels = pd.to_numeric(metadata["label"], errors="coerce")
    if numeric_labels.isna().any() or not numeric_labels.isin([0, 1]).all():
        raise ValueError("label must contain only binary values 0 and 1")
    metadata["label"] = numeric_labels.astype(int)
    metadata["image_path"] = metadata["image_path"].map(str)
    metadata["image_name"] = metadata["image_path"].map(lambda path: Path(path).name)

    missing = [path for path in metadata["image_path"] if not Path(path).is_file()]
    if missing:
        raise FileNotFoundError(
            f"{len(missing)} HYGD images are missing; first: {missing[0]}"
        )

    metadata["sha256"] = metadata["image_path"].map(sha256_file)
    source_inventory_sha256 = inventory_fingerprint(metadata)
    union_find = UnionFind(metadata["patient_id"].astype(str).unique())
    duplicate_details = []

    for digest, rows in metadata.groupby("sha256", sort=True):
        if len(rows) == 1:
            continue
        duplicate_labels = sorted(rows["label"].unique().tolist())
        if len(duplicate_labels) != 1:
            raise ValueError(
                f"Exact duplicate {digest} has conflicting labels "
                f"{duplicate_labels}; evaluation must stop."
            )
        patients = sorted(
            rows["patient_id"].astype(str).unique(), key=_patient_sort_key
        )
        for patient in patients[1:]:
            union_find.union(patients[0], patient)
        duplicate_details.append(
            {
                "sha256": digest,
                "images": sorted(rows["image_name"].tolist()),
                "patient_ids": patients,
                "label": int(duplicate_labels[0]),
                "extra_rows": int(len(rows) - 1),
                "cross_patient": len(patients) > 1,
            }
        )

    components: dict[str, list[str]] = {}
    for patient in metadata["patient_id"].astype(str).unique():
        components.setdefault(union_find.find(patient), []).append(patient)

    component_names: dict[str, str] = {}
    for members in components.values():
        ordered = sorted(members, key=_patient_sort_key)
        name = "P" + "+".join(ordered)
        for patient in ordered:
            component_names[patient] = name

    metadata["evaluation_group"] = (
        metadata["patient_id"].astype(str).map(component_names)
    )
    _assert_group_invariants(metadata, "evaluation_group", ["label"])

    deduplicated = (
        metadata.sort_values(["sha256", "image_name"], kind="stable")
        .drop_duplicates("sha256", keep="first")
        .sort_values(["evaluation_group", "image_name"], kind="stable")
        .reset_index(drop=True)
    )
    if deduplicated["sha256"].duplicated().any():
        raise AssertionError("Exact hashes remain after deduplication.")

    quality = {
        "input_images": int(len(metadata)),
        "unique_image_hashes": int(metadata["sha256"].nunique()),
        "removed_exact_duplicate_rows": int(len(metadata) - len(deduplicated)),
        "duplicate_hash_groups": int(len(duplicate_details)),
        "cross_patient_duplicate_hash_groups": int(
            sum(item["cross_patient"] for item in duplicate_details)
        ),
        "input_patient_ids": int(metadata["patient_id"].nunique()),
        "negative_rows": int((1 - metadata["label"]).sum()),
        "positive_rows": int(metadata["label"].sum()),
        "independent_evaluation_groups": int(
            deduplicated["evaluation_group"].nunique()
        ),
        "label_conflicts": 0,
        "missing_images": 0,
        "ordered_image_sha256_patient_id_label_inventory_sha256": (
            source_inventory_sha256
        ),
        "duplicate_groups": duplicate_details,
    }
    return deduplicated, quality


def console_data_quality_summary(data_quality: Mapping[str, object]) -> dict:
    """Return aggregate audit counts safe for stdout and captured CI logs."""

    aggregate_keys = (
        "input_images",
        "unique_image_hashes",
        "removed_exact_duplicate_rows",
        "duplicate_hash_groups",
        "cross_patient_duplicate_hash_groups",
        "input_patient_ids",
        "independent_evaluation_groups",
        "label_conflicts",
        "missing_images",
    )
    missing = [key for key in aggregate_keys if key not in data_quality]
    if missing:
        raise ValueError(f"data_quality is missing aggregate keys: {missing}")
    return {key: int(data_quality[key]) for key in aggregate_keys}


def make_outer_folds(
    metadata: pd.DataFrame,
    n_splits: int = 5,
    seed: int = DEFAULT_SEED,
) -> pd.DataFrame:
    """Assign every independent evaluation group to one stratified outer fold."""

    if n_splits < 2:
        raise ValueError("n_splits must be at least 2")
    _assert_group_invariants(metadata, "evaluation_group", ["label"])
    _require_columns(metadata, ["image_name"], "metadata")
    groups = (
        metadata.groupby("evaluation_group", as_index=False)
        .agg(label=("label", "first"), n_images=("image_name", "size"))
        .sort_values("evaluation_group", kind="stable")
        .reset_index(drop=True)
    )
    class_counts = groups["label"].value_counts()
    if len(class_counts) != 2 or int(class_counts.min()) < n_splits:
        raise ValueError(
            "Each class must contain at least n_splits independent evaluation groups"
        )
    splitter = StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=seed,
    )
    groups["outer_fold"] = -1
    for fold, (_, test_indices) in enumerate(splitter.split(groups, groups["label"])):
        groups.loc[test_indices, "outer_fold"] = fold
    if (groups["outer_fold"] < 0).any() or groups["evaluation_group"].duplicated().any():
        raise AssertionError("Outer fold assignment is incomplete or duplicated.")
    return groups


def inner_group_split(
    outer_train: pd.DataFrame,
    validation_fraction: float,
    seed: int,
) -> tuple[set[str], set[str]]:
    """Split outer-training groups into disjoint inner train/validation sets."""

    if not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must be between 0 and 1")
    _assert_group_invariants(outer_train, "evaluation_group", ["label"])
    group_table = (
        outer_train.groupby("evaluation_group", as_index=False)
        .agg(label=("label", "first"))
        .sort_values("evaluation_group", kind="stable")
        .reset_index(drop=True)
    )
    if len(group_table["label"].unique()) != 2:
        raise ValueError("Inner splitting requires positive and negative groups")
    splitter = StratifiedShuffleSplit(
        n_splits=1,
        test_size=validation_fraction,
        random_state=seed,
    )
    train_indices, validation_indices = next(
        splitter.split(group_table, group_table["label"])
    )
    train_groups = set(group_table.iloc[train_indices]["evaluation_group"].astype(str))
    validation_groups = set(
        group_table.iloc[validation_indices]["evaluation_group"].astype(str)
    )
    if train_groups & validation_groups:
        raise AssertionError("Inner training and validation groups overlap.")
    if train_groups | validation_groups != set(
        group_table["evaluation_group"].astype(str)
    ):
        raise AssertionError("Inner split does not cover every outer-training group.")
    return train_groups, validation_groups


def stratified_group_holdout(
    frame: pd.DataFrame,
    group_column: str,
    label_column: str,
    test_fraction: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return deterministic, group-disjoint indices stratified by group label pattern."""

    if not 0 < test_fraction < 1:
        raise ValueError("test_fraction must be between 0 and 1")
    _require_columns(frame, [group_column, label_column], "group holdout frame")
    if frame.empty or frame[[group_column, label_column]].isna().any().any():
        raise ValueError("group holdout fields cannot be empty or missing")
    labels = pd.to_numeric(frame[label_column], errors="coerce")
    if labels.isna().any() or not labels.isin([0, 1]).all():
        raise ValueError("group holdout labels must be binary")
    groups = frame[group_column].astype(str).str.strip()
    if groups.eq("").any():
        raise ValueError("group holdout identifiers cannot be blank")
    table = (
        pd.DataFrame({"group": groups, "label": labels.astype(int)})
        .groupby("group", as_index=False)
        .agg(label_min=("label", "min"), label_max=("label", "max"))
        .sort_values("group", kind="mergesort")
        .reset_index(drop=True)
    )
    table["stratum"] = (
        table["label_min"].astype(str) + "-" + table["label_max"].astype(str)
    )
    if (table["stratum"].value_counts() < 2).any():
        raise ValueError("Every group-label stratum requires at least two groups")
    splitter = StratifiedShuffleSplit(
        n_splits=1,
        test_size=test_fraction,
        random_state=seed,
    )
    try:
        train_group_indices, test_group_indices = next(
            splitter.split(table, table["stratum"])
        )
    except ValueError as error:
        raise ValueError("Unable to construct a stratified group holdout") from error
    train_groups = set(table.iloc[train_group_indices]["group"])
    test_groups = set(table.iloc[test_group_indices]["group"])
    if train_groups & test_groups or train_groups | test_groups != set(table["group"]):
        raise AssertionError("Group holdout is overlapping or incomplete")
    train_indices = np.flatnonzero(groups.isin(train_groups).to_numpy())
    test_indices = np.flatnonzero(groups.isin(test_groups).to_numpy())
    for name, indices in (("train", train_indices), ("test", test_indices)):
        if set(labels.iloc[indices].astype(int).unique()) != {0, 1}:
            raise ValueError(f"Stratified group holdout {name} split lacks a class")
    return train_indices, test_indices


def load_hash_bound_subject_map(
    stems: Sequence[object],
    mapping_csv: str | Path | None,
    expected_sha256: str | None,
) -> tuple[dict[str, str], str]:
    """Load exact mapping bytes and return the map plus its verified digest.

    The digest establishes which operator-supplied mapping was used. It does not
    establish that the asserted subject identities are biologically correct.
    """

    expected_stems = [str(value).strip() for value in stems]
    if not expected_stems or any(not value for value in expected_stems):
        raise ValueError("Expected image stems cannot be empty or blank")
    if mapping_csv is None or expected_sha256 is None:
        raise ValueError("A mapping CSV and expected SHA-256 are required")
    content, digest = read_verified_file_bytes(
        mapping_csv, expected_sha256=expected_sha256
    )
    try:
        mapping = pd.read_csv(
            io.BytesIO(content), dtype={"stem": "string", "subject_id": "string"}
        )
    except (UnicodeDecodeError, pd.errors.ParserError, ValueError) as error:
        raise ValueError("Subject mapping must be valid CSV") from error
    if set(mapping.columns) != {"stem", "subject_id"}:
        raise ValueError("Subject mapping CSV requires exactly stem and subject_id")
    if mapping.isna().any().any():
        raise ValueError("Subject mapping contains missing values")
    mapping["stem"] = mapping["stem"].astype(str).str.strip()
    mapping["subject_id"] = mapping["subject_id"].astype(str).str.strip()
    if (
        mapping.empty
        or mapping["stem"].eq("").any()
        or mapping["subject_id"].eq("").any()
        or mapping["stem"].duplicated().any()
    ):
        raise ValueError("Subject mapping contains blank, missing, or duplicate stems")
    if set(mapping["stem"]) != set(expected_stems):
        raise ValueError("Subject mapping must cover the exact expected stem set")
    return dict(zip(mapping["stem"], mapping["subject_id"])), digest


def aggregate_groups(frame: pd.DataFrame) -> pd.DataFrame:
    """Aggregate image probabilities at the duplicate-aware group grain."""

    _require_columns(
        frame,
        [
            "evaluation_group",
            "label",
            "probability",
            "selected_threshold",
            "outer_fold",
            "image_name",
        ],
        "prediction frame",
    )
    _assert_group_invariants(
        frame,
        "evaluation_group",
        ["label", "selected_threshold", "outer_fold"],
    )
    return (
        frame.groupby("evaluation_group", as_index=False)
        .agg(
            label=("label", "first"),
            probability=("probability", "mean"),
            selected_threshold=("selected_threshold", "first"),
            outer_fold=("outer_fold", "first"),
            n_images=("image_name", "size"),
        )
        .sort_values("evaluation_group", kind="stable")
        .reset_index(drop=True)
    )


def _strict_metric_vector(
    values: Sequence[object] | np.ndarray,
    *,
    context: str,
    binary: bool,
) -> np.ndarray:
    """Return a one-dimensional finite vector without lossy coercion."""

    raw = np.asarray(values)
    if raw.ndim != 1 or not len(raw):
        raise ValueError(f"{context} must be a nonempty one-dimensional vector")
    if raw.dtype.kind in {"b", "S", "U"}:
        qualifier = "binary integers 0/1" if binary else "numeric probabilities"
        raise ValueError(
            f"{context} must use explicit {qualifier}, not booleans or strings"
        )
    if raw.dtype.kind == "O":
        permitted = (int, np.integer) if binary else (int, float, np.integer, np.floating)
        if any(isinstance(value, (bool, np.bool_)) or not isinstance(value, permitted)
               for value in raw.tolist()):
            qualifier = "binary integers 0/1" if binary else "numeric probabilities"
            raise ValueError(
                f"{context} must use explicit {qualifier}, not booleans or strings"
            )
    if binary and raw.dtype.kind not in {"i", "u", "O"}:
        raise ValueError(f"{context} must contain binary integers 0/1")
    if not binary and raw.dtype.kind not in {"i", "u", "f", "O"}:
        raise ValueError(f"{context} must contain finite numeric probabilities")
    numeric = pd.to_numeric(pd.Series(raw), errors="coerce").to_numpy(dtype=float)
    if not np.isfinite(numeric).all():
        raise ValueError(f"{context} must contain finite numeric values")
    if binary:
        if not np.equal(numeric, np.floor(numeric)).all() or not np.isin(
            numeric, [0, 1]
        ).all():
            raise ValueError(f"{context} must contain binary integers 0/1")
        return numeric.astype(int)
    if np.any(numeric < 0) or np.any(numeric > 1):
        raise ValueError(f"{context} probabilities must be within [0, 1]")
    return numeric


def metrics_from_predictions(
    labels: Sequence[int] | np.ndarray,
    probabilities: Sequence[float] | np.ndarray,
    predictions: Sequence[int] | np.ndarray,
) -> dict:
    labels = _strict_metric_vector(labels, context="labels", binary=True)
    probabilities = _strict_metric_vector(
        probabilities, context="probabilities", binary=False
    )
    predictions = _strict_metric_vector(
        predictions, context="predictions", binary=True
    )
    if not (len(labels) == len(probabilities) == len(predictions)) or not len(labels):
        raise ValueError("labels, probabilities, and predictions must have equal nonzero length")
    auc = (
        float(roc_auc_score(labels, probabilities))
        if len(np.unique(labels)) == 2
        else None
    )
    tn, fp, fn, tp = confusion_matrix(labels, predictions, labels=[0, 1]).ravel()
    sensitivity = tp / (tp + fn) if tp + fn else None
    specificity = tn / (tn + fp) if tn + fp else None
    return {
        "n": int(len(labels)),
        "positives": int(labels.sum()),
        "negatives": int((1 - labels).sum()),
        "auroc": auc,
        "sensitivity": float(sensitivity) if sensitivity is not None else None,
        "specificity": float(specificity) if specificity is not None else None,
        "confusion_tn_fp_fn_tp": [int(tn), int(fp), int(fn), int(tp)],
    }


def fixed_threshold_metrics(
    labels: Sequence[int] | np.ndarray,
    probabilities: Sequence[float] | np.ndarray,
    threshold: float,
) -> dict:
    if (
        isinstance(threshold, bool)
        or not isinstance(threshold, (int, float))
        or not np.isfinite(threshold)
        or not 0 <= float(threshold) <= 1
    ):
        raise ValueError("threshold must be a finite value within [0, 1]")
    probabilities = _strict_metric_vector(
        probabilities, context="probabilities", binary=False
    )
    predictions = (probabilities >= threshold).astype(int)
    return metrics_from_predictions(labels, probabilities, predictions)


def select_threshold(
    labels: Sequence[int] | np.ndarray,
    probabilities: Sequence[float] | np.ndarray,
    target_sensitivity: float,
) -> dict:
    """Choose an inner-validation threshold without consulting outer predictions."""

    if (
        isinstance(target_sensitivity, (bool, np.bool_))
        or not isinstance(target_sensitivity, (int, float, np.integer, np.floating))
        or not np.isfinite(target_sensitivity)
        or not 0 < float(target_sensitivity) <= 1
    ):
        raise ValueError("target_sensitivity must be greater than 0 and at most 1")
    labels = _strict_metric_vector(labels, context="labels", binary=True)
    probabilities = _strict_metric_vector(
        probabilities, context="probabilities", binary=False
    )
    if len(labels) != len(probabilities) or not len(labels):
        raise ValueError("labels and probabilities must have equal nonzero length")
    if set(np.unique(labels).tolist()) != {0, 1}:
        raise ValueError("Threshold selection requires positive and negative labels")
    candidates = np.unique(np.concatenate(([0.0], probabilities, [1.0])))
    best: tuple[float, float, float] | None = None
    for threshold in candidates:
        metrics = fixed_threshold_metrics(labels, probabilities, float(threshold))
        if (
            metrics["sensitivity"] is None
            or metrics["sensitivity"] < target_sensitivity
        ):
            continue
        # Specificity is the constrained objective. If tied, prefer the
        # candidate with greater sensitivity, then the higher threshold.
        candidate = (
            float(metrics["specificity"]),
            float(metrics["sensitivity"]),
            float(threshold),
        )
        if best is None or candidate > best:
            best = candidate
    if best is None:
        raise RuntimeError("No inner-validation threshold meets the sensitivity target")
    return {
        "threshold": best[2],
        "inner_validation_sensitivity": best[1],
        "inner_validation_specificity": best[0],
    }


def frame_metrics(frame: pd.DataFrame) -> dict:
    _require_columns(
        frame,
        ["label", "probability", "selected_threshold"],
        "prediction frame",
    )
    probabilities = frame["probability"].to_numpy(dtype=float)
    thresholds = frame["selected_threshold"].to_numpy(dtype=float)
    return metrics_from_predictions(
        frame["label"].to_numpy(dtype=int),
        probabilities,
        (probabilities >= thresholds).astype(int),
    )


def historical_image_bootstrap_ci(
    labels: Sequence[int] | np.ndarray,
    probabilities: Sequence[float] | np.ndarray,
    threshold: float = 0.5,
    n_bootstrap: int = 2_000,
    seed: int = 42,
) -> dict:
    """Reproduce the legacy row-bootstrap interval with an explicit warning.

    This helper exists only for historical artifact compatibility. Repeated
    images from one patient are not independent, so callers must never promote
    these intervals as the canonical internal estimate.
    """

    labels = _strict_metric_vector(labels, context="labels", binary=True)
    probabilities = _strict_metric_vector(
        probabilities, context="probabilities", binary=False
    )
    if len(labels) != len(probabilities) or not len(labels):
        raise ValueError("labels and probabilities must have equal nonzero length")
    if n_bootstrap <= 0:
        raise ValueError("n_bootstrap must be positive")
    rng = np.random.default_rng(seed)
    values = {"auroc": [], "sensitivity": [], "specificity": []}
    for _ in range(n_bootstrap):
        indices = rng.integers(0, len(labels), size=len(labels))
        sampled_labels = labels[indices]
        if len(np.unique(sampled_labels)) < 2:
            continue
        metrics = fixed_threshold_metrics(
            sampled_labels,
            probabilities[indices],
            threshold,
        )
        for key in values:
            if metrics[key] is not None:
                values[key].append(float(metrics[key]))
    missing = [key for key, metric_values in values.items() if not metric_values]
    if missing:
        raise RuntimeError(
            f"No valid historical image-bootstrap replicates for metrics: {missing}"
        )
    intervals = {
        key + "_95ci": [
            float(np.percentile(metric_values, 2.5)),
            float(np.percentile(metric_values, 97.5)),
        ]
        for key, metric_values in values.items()
    }
    return {
        "status": "historical_image_level_only",
        "sampling_unit": "image_row",
        "canonical_internal_estimate": False,
        "warning": (
            "Rows, not independent patient/evaluation groups, were resampled. "
            "Use the repaired group-cluster bootstrap for canonical inference."
        ),
        "intervals": intervals,
    }


def resample_clusters(
    frame: pd.DataFrame,
    cluster_column: str,
    sampled_clusters: Sequence[object],
) -> pd.DataFrame:
    """Materialize a prescribed bootstrap draw using whole clusters."""

    _require_columns(frame, [cluster_column], "bootstrap frame")
    if not sampled_clusters:
        raise ValueError("sampled_clusters cannot be empty")
    rows_by_cluster = {
        cluster: rows.copy()
        for cluster, rows in frame.groupby(cluster_column, sort=False, dropna=False)
    }
    unknown = [cluster for cluster in sampled_clusters if cluster not in rows_by_cluster]
    if unknown:
        raise ValueError(f"Unknown bootstrap clusters: {unknown}")
    return pd.concat(
        [rows_by_cluster[cluster] for cluster in sampled_clusters],
        ignore_index=True,
    )


def cluster_bootstrap(
    frame: pd.DataFrame,
    cluster_column: str,
    n_bootstrap: int,
    seed: int,
) -> dict:
    """Return percentile CIs after resampling independent groups atomically."""

    if n_bootstrap <= 0:
        raise ValueError("n_bootstrap must be positive")
    _require_columns(
        frame,
        [cluster_column, "label", "probability", "selected_threshold"],
        "bootstrap frame",
    )
    clusters = sorted(frame[cluster_column].drop_duplicates().tolist(), key=str)
    if not clusters:
        raise ValueError("bootstrap frame has no clusters")
    cluster_indices = {
        cluster: np.asarray(indices, dtype=int)
        for cluster, indices in frame.groupby(
            cluster_column, sort=False, dropna=False
        ).indices.items()
    }
    labels = frame["label"].to_numpy(dtype=int)
    probabilities = frame["probability"].to_numpy(dtype=float)
    thresholds = frame["selected_threshold"].to_numpy(dtype=float)
    rng = np.random.default_rng(seed)
    values = {"auroc": [], "sensitivity": [], "specificity": []}

    for _ in range(n_bootstrap):
        sampled = rng.choice(clusters, size=len(clusters), replace=True).tolist()
        sampled_indices = np.concatenate(
            [cluster_indices[cluster] for cluster in sampled]
        )
        sampled_labels = labels[sampled_indices]
        if len(np.unique(sampled_labels)) < 2:
            continue
        sampled_probabilities = probabilities[sampled_indices]
        sampled_thresholds = thresholds[sampled_indices]
        metrics = metrics_from_predictions(
            sampled_labels,
            sampled_probabilities,
            (sampled_probabilities >= sampled_thresholds).astype(int),
        )
        for key in values:
            if metrics[key] is not None:
                values[key].append(float(metrics[key]))

    missing = [key for key, metric_values in values.items() if not metric_values]
    if missing:
        raise RuntimeError(f"No valid bootstrap replicates for metrics: {missing}")
    return {
        key + "_95ci": [
            float(np.percentile(metric_values, 2.5)),
            float(np.percentile(metric_values, 97.5)),
        ]
        for key, metric_values in values.items()
    }


def patient_cluster_auc_ci(
    labels: Sequence[int] | np.ndarray,
    probabilities: Sequence[float] | np.ndarray,
    patient_ids: Sequence[object] | np.ndarray,
    n_bootstrap: int,
    seed: int,
) -> dict:
    """Bootstrap whole patients while retaining an eye/image-row AUROC estimand."""

    labels = np.asarray(labels)
    probabilities = np.asarray(probabilities, dtype=float)
    patient_ids = np.asarray(patient_ids, dtype=object)
    if (
        labels.ndim != 1
        or probabilities.ndim != 1
        or patient_ids.ndim != 1
        or not len(labels)
        or not (len(labels) == len(probabilities) == len(patient_ids))
    ):
        raise ValueError("labels, probabilities, and patient_ids must align")
    numeric_labels = pd.to_numeric(pd.Series(labels), errors="coerce")
    if numeric_labels.isna().any() or not numeric_labels.isin([0, 1]).all():
        raise ValueError("patient-cluster labels must be binary")
    labels = numeric_labels.to_numpy(dtype=int)
    if set(np.unique(labels).tolist()) != {0, 1}:
        raise ValueError("patient-cluster AUROC requires both classes")
    if not np.isfinite(probabilities).all() or np.any(probabilities < 0) or np.any(
        probabilities > 1
    ):
        raise ValueError("patient-cluster probabilities must be finite in [0, 1]")
    if pd.isna(pd.Series(patient_ids, dtype=object)).any():
        raise ValueError("patient-cluster identifiers cannot be blank or missing")
    raw_ids = [str(value) for value in patient_ids]
    if any(re.search(r"[\x00-\x1f\x7f]", value) is not None for value in raw_ids):
        raise ValueError("patient-cluster identifiers cannot contain control characters")
    normalized_ids = np.asarray([value.strip() for value in raw_ids])
    if any(
        not value
        or value.lower() == "nan"
        for value in normalized_ids
    ):
        raise ValueError("patient-cluster identifiers cannot be blank or missing")
    if type(n_bootstrap) is not int or n_bootstrap <= 0:
        raise ValueError("n_bootstrap must be a positive integer")

    patients = sorted(np.unique(normalized_ids).tolist(), key=str)
    patient_indices = {
        patient: np.flatnonzero(normalized_ids == patient) for patient in patients
    }
    rng = np.random.default_rng(seed)
    values = []
    attempts = 0
    maximum_attempts = max(n_bootstrap * 10, n_bootstrap + 100)
    while len(values) < n_bootstrap and attempts < maximum_attempts:
        attempts += 1
        sampled_patients = rng.choice(
            patients, size=len(patients), replace=True
        ).tolist()
        sampled_indices = np.concatenate(
            [patient_indices[patient] for patient in sampled_patients]
        )
        sampled_labels = labels[sampled_indices]
        if len(np.unique(sampled_labels)) < 2:
            continue
        values.append(
            float(roc_auc_score(sampled_labels, probabilities[sampled_indices]))
        )
    if len(values) != n_bootstrap:
        raise RuntimeError("Insufficient valid patient-cluster bootstrap replicates")
    return {
        "auroc": float(roc_auc_score(labels, probabilities)),
        "auroc_95ci": [
            float(np.percentile(values, 2.5)),
            float(np.percentile(values, 97.5)),
        ],
        "point_estimand_unit": "eye_or_image_row",
        "resampling_unit": "patient",
        "scope": "conditional_on_fixed_adaptive_predictions",
        "canonical_or_confirmatory": False,
        "seed": int(seed),
        "attempted_replicates": int(attempts),
        "valid_replicates": int(len(values)),
    }


def outer_fold_auc_summary(group_frame: pd.DataFrame) -> dict:
    """Summarize group AUROC within folds without comparing cross-model scores."""

    _require_columns(
        group_frame,
        ["evaluation_group", "outer_fold", "label", "probability"],
        "group OOF frame",
    )
    if group_frame["evaluation_group"].duplicated().any():
        raise ValueError("group OOF frame must contain one row per evaluation group")
    fold_rows = []
    for fold, rows in group_frame.groupby("outer_fold", sort=True):
        if set(rows["label"].astype(int).unique().tolist()) != {0, 1}:
            raise ValueError(f"Outer fold {fold} must contain both outcome classes")
        fold_rows.append(
            {
                "outer_fold": int(fold),
                "n_groups": int(len(rows)),
                "positives": int(rows["label"].sum()),
                "negatives": int((1 - rows["label"]).sum()),
                "auroc": float(roc_auc_score(rows["label"], rows["probability"])),
            }
        )
    if len(fold_rows) < 2:
        raise ValueError("At least two outer folds are required")
    aucs = np.asarray([row["auroc"] for row in fold_rows], dtype=float)
    counts = np.asarray([row["n_groups"] for row in fold_rows], dtype=float)
    return {
        "fold_aurocs": fold_rows,
        "unweighted_mean_auroc": float(aucs.mean()),
        "group_count_weighted_mean_auroc": float(np.average(aucs, weights=counts)),
    }


def fold_stratified_group_bootstrap(
    group_frame: pd.DataFrame,
    n_bootstrap: int,
    seed: int,
) -> dict:
    """Bootstrap whole groups within folds for scale-robust macro discrimination."""

    if n_bootstrap <= 0:
        raise ValueError("n_bootstrap must be positive")
    _require_columns(
        group_frame,
        [
            "evaluation_group",
            "outer_fold",
            "label",
            "probability",
            "selected_threshold",
        ],
        "group OOF frame",
    )
    if group_frame["evaluation_group"].duplicated().any():
        raise ValueError("group OOF frame must contain one row per evaluation group")
    if group_frame["evaluation_group"].isna().any() or not group_frame[
        "evaluation_group"
    ].map(lambda value: isinstance(value, str)).all():
        raise ValueError("evaluation_group must contain non-null strings")
    fold_frames = [
        rows.sort_values("evaluation_group", kind="mergesort").reset_index(drop=True)
        for _, rows in group_frame.groupby("outer_fold", sort=True)
    ]
    if len(fold_frames) < 2 or any(
        set(rows["label"].astype(int).unique().tolist()) != {0, 1}
        for rows in fold_frames
    ):
        raise ValueError("Every outer fold must contain both outcome classes")

    rng = np.random.default_rng(seed)
    values = {"auroc": [], "sensitivity": [], "specificity": []}
    for _ in range(n_bootstrap):
        sampled_folds = []
        fold_aurocs = []
        for rows in fold_frames:
            sampled = rows.iloc[
                rng.integers(0, len(rows), size=len(rows))
            ].reset_index(drop=True)
            if len(np.unique(sampled["label"])) < 2:
                break
            sampled_folds.append(sampled)
            fold_aurocs.append(
                float(roc_auc_score(sampled["label"], sampled["probability"]))
            )
        if len(sampled_folds) != len(fold_frames):
            continue
        pooled_classification = frame_metrics(
            pd.concat(sampled_folds, ignore_index=True)
        )
        values["auroc"].append(float(np.mean(fold_aurocs)))
        values["sensitivity"].append(float(pooled_classification["sensitivity"]))
        values["specificity"].append(float(pooled_classification["specificity"]))

    missing = [key for key, metric_values in values.items() if not metric_values]
    if missing:
        raise RuntimeError(
            f"No valid fold-stratified bootstrap replicates for metrics: {missing}"
        )
    intervals = {
        key + "_95ci": [
            float(np.percentile(metric_values, 2.5)),
            float(np.percentile(metric_values, 97.5)),
        ]
        for key, metric_values in values.items()
    }
    return {
        **intervals,
        "resampling_unit": "evaluation_group_within_outer_fold",
        "fold_weighting": "equal",
        "seed": int(seed),
        "n_bootstrap": int(n_bootstrap),
        "valid_replicates": int(len(values["auroc"])),
    }


def _walk_mapping_keys(value: object) -> Iterable[str]:
    if isinstance(value, Mapping):
        for key, nested in value.items():
            yield str(key)
            yield from _walk_mapping_keys(nested)
    elif isinstance(value, (list, tuple)):
        for nested in value:
            yield from _walk_mapping_keys(nested)


def finalize_historical_comparison(results: Mapping[str, object]) -> dict:
    """Mark a single-split comparison as noncanonical and forbid CV/selection."""

    payload = copy.deepcopy(dict(results))
    keys = {key.lower() for key in _walk_mapping_keys(payload)}
    cv_keys = {"fold_aucs", "mean_auc", "std_auc", "cv_results", "cross_validation"}
    selection_keys = {
        "selected_configuration",
        "best_configuration",
        "best_config",
        "best_name",
        "model_selection",
    }
    if keys & cv_keys:
        raise ValueError(
            "Historical single-split output cannot contain cross-validation results"
        )
    if keys & selection_keys:
        raise ValueError(
            "Historical test-set comparison cannot perform configuration selection"
        )
    if "_provenance" in payload:
        raise ValueError("_provenance is reserved for the historical finalizer")
    payload["_provenance"] = {
        "status": "historical_development_comparison",
        "canonical_internal_estimate": False,
        "configuration_selection_permitted": False,
        "warning": (
            "All configurations were compared on the same development test split. "
            "These values and image-level intervals are descriptive history, not "
            "an untouched final test or a basis for cross-validation selection."
        ),
        "primary_internal_command": "python validation/internal_evaluation_repair.py",
    }
    return payload


def build_primary_report(
    *,
    outer_fold_auc: Mapping[str, object],
    fold_stratified_ci: Mapping[str, object],
    pooled_group_metrics: Mapping[str, object],
    pooled_group_cluster_ci: Mapping[str, object],
    image_metrics: Mapping[str, object],
    image_cluster_ci: Mapping[str, object],
) -> dict:
    """Build the only schema allowed for repaired internal headline metrics."""

    if (
        "unweighted_mean_auroc" not in outer_fold_auc
        or "group_count_weighted_mean_auroc" not in outer_fold_auc
        or "auroc_95ci" not in fold_stratified_ci
        or "auroc_95ci" not in pooled_group_cluster_ci
        or "auroc_95ci" not in image_cluster_ci
        or any(
            key not in fold_stratified_ci
            for key in (
                "resampling_unit",
                "fold_weighting",
                "seed",
                "n_bootstrap",
                "valid_replicates",
            )
        )
    ):
        raise ValueError("Primary and descriptive reports require AUROC intervals")
    pooled_metrics = copy.deepcopy(dict(pooled_group_metrics))
    primary = {
        key: pooled_metrics[key]
        for key in (
            "n",
            "positives",
            "negatives",
            "sensitivity",
            "specificity",
            "confusion_tn_fp_fn_tp",
        )
        if key in pooled_metrics
    }
    primary.update(
        {
            "grain": "evaluation_group",
            "auroc": float(outer_fold_auc["unweighted_mean_auroc"]),
            "auroc_95ci": copy.deepcopy(fold_stratified_ci["auroc_95ci"]),
            "sensitivity_95ci": copy.deepcopy(
                fold_stratified_ci.get("sensitivity_95ci")
            ),
            "specificity_95ci": copy.deepcopy(
                fold_stratified_ci.get("specificity_95ci")
            ),
            "discrimination_estimand": "unweighted_mean_outer_fold_auroc",
            "classification_estimand": (
                "pooled_oof_binary_decisions_using_fold_specific_"
                "inner_validation_thresholds"
            ),
            "uncertainty_source": "fold_stratified_evaluation_group_bootstrap",
            "fold_aurocs": copy.deepcopy(outer_fold_auc.get("fold_aurocs")),
            "group_count_weighted_mean_auroc": float(
                outer_fold_auc["group_count_weighted_mean_auroc"]
            ),
            "bootstrap": {
                key: copy.deepcopy(fold_stratified_ci[key])
                for key in (
                    "resampling_unit",
                    "fold_weighting",
                    "seed",
                    "n_bootstrap",
                    "valid_replicates",
                )
                if key in fold_stratified_ci
            },
        }
    )
    pooled_group = {
        **pooled_metrics,
        **copy.deepcopy(dict(pooled_group_cluster_ci)),
        "grain": "evaluation_group",
        "status": "secondary_descriptive_continuity_metric",
        "uncertainty_source": "global_group_cluster_bootstrap_over_pooled_oof_scores",
        "warning": (
            "Pooled OOF AUROC compares score scales from separately fitted outer-fold "
            "models and is not invariant to fold-specific monotone rescaling. Its "
            "global group-cluster interval is conditional on those fixed scores."
        ),
    }
    secondary = {
        **copy.deepcopy(dict(image_metrics)),
        **copy.deepcopy(dict(image_cluster_ci)),
    }
    secondary.update(
        {
            "grain": "unique_image_hash",
            "uncertainty_source": "evaluation_group_cluster_bootstrap",
            "status": "secondary_descriptive",
            "warning": (
                "Image-level pooled OOF AUROC compares score scales from separately "
                "fitted fold models. Its global group-cluster interval is conditional "
                "on those fixed scores and remains secondary."
            ),
        }
    )
    return {
        "preferred_internal_metric_source": PREFERRED_INTERNAL_METRIC_SOURCE,
        "uncertainty_scope": (
            "The primary percentile interval resamples whole evaluation groups "
            "within each fixed outer fold over fixed OOF predictions. It conditions "
            "on the realized inner split, checkpoint, fold-specific threshold, "
            "fitted models, and fold assignment; it excludes training, "
            "recipe-selection, and transportability uncertainty. Pooled group and "
            "image intervals instead use a global group-cluster bootstrap and are "
            "secondary descriptive summaries."
        ),
        "primary": primary,
        "pooled_oof_group_descriptive": pooled_group,
        "secondary_image_level": secondary,
    }


def build_smoke_diagnostics(
    *,
    fold_execution: Sequence[Mapping[str, object]],
    completed_outer_folds: Sequence[int],
    total_outer_folds: int,
) -> dict:
    """Build an outcome-blind receipt for an incomplete execution smoke run."""

    if type(total_outer_folds) is not int or total_outer_folds < 2:
        raise ValueError("total_outer_folds must be at least 2")
    if any(type(fold) is not int for fold in completed_outer_folds):
        raise ValueError("completed_outer_folds must contain exact integers")
    completed = sorted(set(completed_outer_folds))
    if not completed or completed != list(map(int, completed_outer_folds)):
        raise ValueError("completed_outer_folds must be nonempty, unique, and sorted")
    if completed != list(range(len(completed))) or len(completed) >= total_outer_folds:
        raise ValueError("Smoke diagnostics require a proper leading subset of folds")
    expected_execution_keys = {
        "outer_fold",
        "training_loop_completed",
        "checkpoint_restore_completed",
        "inner_selection_path_completed",
        "outer_inference_path_completed",
        "outer_inference_rows",
        "outer_inference_values_validated",
    }
    if len(fold_execution) != len(completed):
        raise ValueError("Smoke fold_execution must cover every completed fold")
    normalized_execution = []
    for expected_fold, receipt in zip(completed, fold_execution):
        if not isinstance(receipt, Mapping) or set(receipt) != expected_execution_keys:
            raise ValueError("Smoke fold_execution must use the exact outcome-blind schema")
        if (
            type(receipt.get("outer_fold")) is not int
            or receipt.get("outer_fold") != expected_fold
        ):
            raise ValueError("Smoke fold_execution is not aligned to completed folds")
        if any(
            receipt.get(key) is not True
            for key in expected_execution_keys
            if key not in {"outer_fold", "outer_inference_rows"}
        ):
            raise ValueError("Smoke execution checks must all pass")
        rows = receipt.get("outer_inference_rows")
        if type(rows) is not int or rows <= 0:
            raise ValueError("Smoke outer_inference_rows must be positive")
        normalized_execution.append(copy.deepcopy(dict(receipt)))
    return {
        "status": "partial_smoke_noncanonical",
        "canonical_internal_estimate": False,
        "warning": (
            "Incomplete execution receipt only. No evaluation outcomes are serialized, "
            "and this is not an internal estimate."
        ),
        "completed_outer_folds": completed,
        "planned_outer_folds": int(total_outer_folds),
        "fold_execution": normalized_execution,
        "outcomes_emitted": False,
    }


def validate_smoke_prediction_execution(
    probabilities: Sequence[float] | np.ndarray,
    expected_rows: int,
) -> dict:
    """Validate outer inference mechanically without returning any predictions."""

    values = np.asarray(probabilities, dtype=float)
    if (
        type(expected_rows) is not int
        or expected_rows <= 0
        or values.ndim != 1
        or len(values) != expected_rows
        or not np.isfinite(values).all()
        or np.any(values < 0)
        or np.any(values > 1)
    ):
        raise ValueError("smoke predictions must be finite probabilities of expected length")
    return {
        "outer_inference_path_completed": True,
        "outer_inference_rows": int(expected_rows),
        "outer_inference_values_validated": True,
    }


def validate_primary_report(report: Mapping[str, object]) -> None:
    """Fail if a renderer could promote a noncanonical uncertainty source."""

    expected_report_keys = {
        "preferred_internal_metric_source",
        "uncertainty_scope",
        "primary",
        "pooled_oof_group_descriptive",
        "secondary_image_level",
    }
    if not isinstance(report, Mapping) or set(report) != expected_report_keys:
        raise ValueError("Primary report must use the exact scientific schema")
    if report.get("preferred_internal_metric_source") != PREFERRED_INTERNAL_METRIC_SOURCE:
        raise ValueError(
            "preferred_internal_metric_source must be "
            + PREFERRED_INTERNAL_METRIC_SOURCE
        )
    primary = report.get("primary")
    pooled = report.get("pooled_oof_group_descriptive")
    secondary = report.get("secondary_image_level")
    if not all(isinstance(value, Mapping) for value in (primary, pooled, secondary)):
        raise ValueError("Primary report requires primary and descriptive mappings")
    metric_keys = {
        "n",
        "positives",
        "negatives",
        "auroc",
        "sensitivity",
        "specificity",
        "confusion_tn_fp_fn_tp",
    }
    primary_keys = metric_keys | {
        "grain",
        "auroc_95ci",
        "sensitivity_95ci",
        "specificity_95ci",
        "discrimination_estimand",
        "classification_estimand",
        "uncertainty_source",
        "fold_aurocs",
        "group_count_weighted_mean_auroc",
        "bootstrap",
    }
    descriptive_keys = metric_keys | {
        "auroc_95ci",
        "sensitivity_95ci",
        "specificity_95ci",
        "grain",
        "status",
        "uncertainty_source",
        "warning",
    }
    if (
        set(primary) != primary_keys
        or set(pooled) != descriptive_keys
        or set(secondary) != descriptive_keys
    ):
        raise ValueError("Primary report has an invalid exact metric schema")
    _validate_metric_summary(primary, "Primary report")
    _validate_metric_summary(pooled, "Pooled group report")
    _validate_metric_summary(secondary, "Secondary image report")
    for context, values in (
        ("Primary report", primary),
        ("Pooled group report", pooled),
        ("Secondary image report", secondary),
    ):
        for key in ("auroc_95ci", "sensitivity_95ci", "specificity_95ci"):
            _validate_interval(values.get(key), f"{context} {key}")
    if primary.get("grain") != "evaluation_group":
        raise ValueError("Primary report grain must be evaluation_group")
    if primary.get("discrimination_estimand") != "unweighted_mean_outer_fold_auroc":
        raise ValueError("Primary discrimination must be mean outer-fold AUROC")
    if primary.get("uncertainty_source") != "fold_stratified_evaluation_group_bootstrap":
        raise ValueError("Primary uncertainty must be fold-stratified group bootstrap")
    if pooled.get("status") != "secondary_descriptive_continuity_metric":
        raise ValueError("Pooled OOF AUROC must remain a descriptive continuity metric")
    if secondary.get("status") != "secondary_descriptive":
        raise ValueError("Image-level output must remain secondary_descriptive")
    if pooled.get("grain") != "evaluation_group":
        raise ValueError("Pooled OOF report grain must be evaluation_group")
    if secondary.get("grain") != "unique_image_hash":
        raise ValueError("Secondary report grain must be unique_image_hash")
    if (
        primary.get("classification_estimand")
        != "pooled_oof_binary_decisions_using_fold_specific_inner_validation_thresholds"
    ):
        raise ValueError("Primary classification estimand is invalid")
    if (
        pooled.get("uncertainty_source")
        != "global_group_cluster_bootstrap_over_pooled_oof_scores"
        or secondary.get("uncertainty_source")
        != "evaluation_group_cluster_bootstrap"
    ):
        raise ValueError("Descriptive uncertainty source is invalid")
    if not isinstance(report.get("uncertainty_scope"), str) or not report[
        "uncertainty_scope"
    ].strip():
        raise ValueError("Primary report requires an uncertainty scope")
    bootstrap = primary.get("bootstrap")
    if not isinstance(bootstrap, Mapping):
        raise ValueError("Primary report requires bootstrap metadata")
    try:
        n_bootstrap = bootstrap["n_bootstrap"]
        valid_replicates = bootstrap["valid_replicates"]
        seed = bootstrap["seed"]
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("Primary report has invalid bootstrap metadata") from error
    if (
        set(bootstrap)
        != {
            "resampling_unit",
            "fold_weighting",
            "seed",
            "n_bootstrap",
            "valid_replicates",
        }
        or bootstrap.get("resampling_unit")
        != "evaluation_group_within_outer_fold"
        or bootstrap.get("fold_weighting") != "equal"
        or type(n_bootstrap) is not int
        or type(valid_replicates) is not int
        or type(seed) is not int
        or n_bootstrap <= 0
        or not 0 < valid_replicates <= n_bootstrap
        or seed < 0
    ):
        raise ValueError("Primary report has invalid bootstrap metadata")
    fold_aurocs = primary.get("fold_aurocs")
    if not isinstance(fold_aurocs, list) or len(fold_aurocs) < 2:
        raise ValueError("Primary report requires outer-fold AUROC rows")
    for expected_fold, row in enumerate(fold_aurocs):
        if (
            not isinstance(row, Mapping)
            or set(row)
            != {"outer_fold", "n_groups", "positives", "negatives", "auroc"}
            or row.get("outer_fold") != expected_fold
            or any(
                type(row.get(key)) is not int or row[key] <= 0
                for key in ("n_groups", "positives", "negatives")
            )
            or row["positives"] + row["negatives"] != row["n_groups"]
            or not _is_finite_number(row.get("auroc"))
            or not 0 <= float(row["auroc"]) <= 1
        ):
            raise ValueError("Primary report has invalid outer-fold AUROC rows")


def validate_internal_result_payload(result: Mapping[str, object]) -> None:
    """Reject historical selection/CV material from the canonical result JSON."""

    required_keys = {
        "protocol_version",
        "status",
        "completed_outer_folds",
        "configuration_fixed_before_outer_evaluation",
        "dataset_identity",
        "data_quality",
        "fold_results",
        "artifacts",
        "environment",
        "runtime_seconds",
        "internal_evaluation",
    }
    terminal_keys = required_keys | {"artifact_sha256", "artifact_size_bytes"}
    if set(result) not in (required_keys, terminal_keys):
        raise ValueError("Canonical result must use the exact scientific schema")
    if result.get("protocol_version") != PROTOCOL_VERSION:
        raise ValueError(f"Canonical result protocol must be {PROTOCOL_VERSION}")
    if result.get("status") != "complete":
        raise ValueError("Canonical result status must be complete")
    configuration = result.get("configuration_fixed_before_outer_evaluation")
    completed = result.get("completed_outer_folds")
    if not isinstance(configuration, Mapping) or not isinstance(completed, list):
        raise ValueError("Canonical result requires configuration and completed folds")
    expected_configuration_keys = {
        "model",
        "weights",
        "trainable_parameters",
        "optimizer",
        "head_learning_rate",
        "backbone_learning_rate",
        "train_transform",
        "evaluation_transform",
        "configuration_origin",
        "residual_post_selection_risk",
        "epochs",
        "batch_size",
        "outer_folds",
        "inner_validation_fraction",
        "threshold_target_sensitivity",
        "bootstrap_samples",
        "seed",
        "device",
        "checkpoint_selection",
        "threshold_selection",
        "outer_test_visibility_during_training",
    }
    if set(configuration) != expected_configuration_keys:
        raise ValueError("Canonical result has invalid fixed-recipe schema")
    mismatches = _canonical_config_mismatches(configuration)
    if mismatches:
        raise ValueError(
            "Canonical result must use the locked canonical configuration; "
            "invalid fields: " + ", ".join(mismatches)
        )
    total_folds = configuration["outer_folds"]
    if any(type(fold) is not int for fold in completed):
        raise ValueError("Canonical result has invalid outer-fold metadata")
    completed_folds = completed
    if completed_folds != list(range(total_folds)):
        raise ValueError("Canonical result must cover every configured outer fold")
    expected_configuration_text = {
        "model": "ImageNet ResNet18, layer4 + head fine-tuned",
        "weights": "ResNet18_Weights.IMAGENET1K_V1",
        "trainable_parameters": "layer4 + fc",
        "optimizer": "Adam",
        "train_transform": (
            "Resize(224x224), RandomHorizontalFlip(p=0.5), "
            "RandomRotation(15deg), ColorJitter(brightness=0.1,contrast=0.1), "
            "ImageNet normalization"
        ),
        "evaluation_transform": "Resize(224x224), ImageNet normalization",
        "checkpoint_selection": "minimum inner-validation loss",
        "threshold_selection": "inner-validation group-level predictions only",
        "outer_test_visibility_during_training": (
            "withheld_until_terminal_bundle_publication"
        ),
        "configuration_origin": (
            "Chosen during prior HYGD development, then frozen before this repair run"
        ),
        "residual_post_selection_risk": (
            "Outer folds were hidden from this run, but the architecture and recipe "
            "were historically informed by HYGD development/test results"
        ),
    }
    wrong_text = [
        key
        for key, expected in expected_configuration_text.items()
        if configuration.get(key) != expected
    ]
    if wrong_text:
        raise ValueError(
            "Canonical result has invalid fixed-recipe metadata: "
            + ", ".join(wrong_text)
        )
    expected_numeric_recipe = {
        "head_learning_rate": 1e-3,
        "backbone_learning_rate": 1e-4,
    }
    wrong_numeric = [
        key
        for key, expected in expected_numeric_recipe.items()
        if type(configuration.get(key)) is not float
        or configuration.get(key) != expected
    ]
    if wrong_numeric:
        raise ValueError(
            "Canonical result has invalid fixed-recipe numeric metadata: "
            + ", ".join(wrong_numeric)
        )
    validate_canonical_dataset_identity(result.get("dataset_identity"))
    data_quality = result.get("data_quality")
    validate_canonical_data_quality(data_quality)
    expected_quality_keys = set(EXPECTED_HYGD_DATA_QUALITY) | {
        "ordered_image_sha256_patient_id_label_inventory_sha256",
        "duplicate_groups",
    }
    if set(data_quality) != expected_quality_keys:
        raise ValueError("Canonical result has invalid HYGD data_quality schema")
    fold_results = result.get("fold_results")
    if not isinstance(fold_results, list) or len(fold_results) != total_folds:
        raise ValueError("Canonical result requires one fold_result per outer fold")
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
    required_fold_mappings = (
        "selected_threshold_from_inner_group_validation",
        "counts",
        "outer_test_image_level",
        "outer_test_group_level",
    )
    outer_image_total = 0
    outer_group_total = 0
    for expected_fold, fold_result in enumerate(fold_results):
        if not isinstance(fold_result, Mapping):
            raise ValueError("Canonical fold_results entries must be mappings")
        if set(fold_result) != expected_fold_keys:
            raise ValueError("Canonical fold_result has invalid schema")
        if (
            type(fold_result.get("outer_fold")) is not int
            or fold_result.get("outer_fold") != expected_fold
        ):
            raise ValueError("Canonical fold_results must be ordered by outer_fold")
        if (
            type(fold_result.get("seed")) is not int
            or fold_result.get("seed") != configuration["seed"] + expected_fold * 101
        ):
            raise ValueError("Canonical fold_result seed is inconsistent")
        best_epoch = fold_result.get("best_epoch_selected_on_inner_validation_loss")
        if type(best_epoch) is not int or not 1 <= best_epoch <= configuration["epochs"]:
            raise ValueError("Canonical fold_result requires selected best epoch")
        if any(
            not isinstance(fold_result.get(key), Mapping)
            for key in required_fold_mappings
        ) or not isinstance(fold_result.get("training_history"), list):
            raise ValueError("Canonical fold_result is missing scientific fields")
        threshold = fold_result["selected_threshold_from_inner_group_validation"]
        if set(threshold) != {
            "threshold",
            "inner_validation_sensitivity",
            "inner_validation_specificity",
        } or any(
            not _is_finite_number(threshold.get(key))
            or not 0 <= float(threshold[key]) <= 1
            for key in threshold
        ):
            raise ValueError("Canonical fold_result has invalid inner threshold metadata")
        if (
            float(threshold["inner_validation_sensitivity"]) + 1e-12
            < float(configuration["threshold_target_sensitivity"])
        ):
            raise ValueError(
                "Canonical fold_result threshold misses the locked sensitivity target"
            )
        counts = fold_result["counts"]
        expected_count_keys = {
            "inner_train_images",
            "inner_train_groups",
            "inner_validation_images",
            "inner_validation_groups",
            "outer_test_images",
            "outer_test_groups",
        }
        if set(counts) != expected_count_keys or any(
            type(counts.get(key)) is not int or counts[key] <= 0
            for key in expected_count_keys
        ):
            raise ValueError("Canonical fold_result has invalid counts")
        if (
            counts["inner_train_images"]
            + counts["inner_validation_images"]
            + counts["outer_test_images"]
            != EXPECTED_HYGD_DATA_QUALITY["unique_image_hashes"]
            or counts["inner_train_groups"]
            + counts["inner_validation_groups"]
            + counts["outer_test_groups"]
            != EXPECTED_HYGD_DATA_QUALITY["independent_evaluation_groups"]
        ):
            raise ValueError(
                "Canonical fold_result inner/outer partition counts are incomplete"
            )
        outer_image_total += counts["outer_test_images"]
        outer_group_total += counts["outer_test_groups"]
        for metric_key, expected_n in (
            ("outer_test_image_level", counts["outer_test_images"]),
            ("outer_test_group_level", counts["outer_test_groups"]),
        ):
            metrics = fold_result[metric_key]
            metric_keys = {
                "n",
                "positives",
                "negatives",
                "auroc",
                "sensitivity",
                "specificity",
                "confusion_tn_fp_fn_tp",
            }
            if set(metrics) != metric_keys or metrics.get("n") != expected_n:
                raise ValueError("Canonical fold_result has invalid metric schema")
            if any(
                type(metrics.get(key)) is not int or metrics[key] < 0
                for key in ("n", "positives", "negatives")
            ) or metrics["positives"] + metrics["negatives"] != metrics["n"]:
                raise ValueError("Canonical fold_result has inconsistent metric counts")
            if any(
                not _is_finite_number(metrics.get(key))
                or not 0 <= float(metrics[key]) <= 1
                for key in ("auroc", "sensitivity", "specificity")
            ):
                raise ValueError("Canonical fold_result has invalid metrics")
            confusion = metrics.get("confusion_tn_fp_fn_tp")
            if (
                not isinstance(confusion, list)
                or len(confusion) != 4
                or any(type(value) is not int or value < 0 for value in confusion)
                or sum(confusion) != metrics["n"]
                or confusion[2] + confusion[3] != metrics["positives"]
                or confusion[0] + confusion[1] != metrics["negatives"]
            ):
                raise ValueError("Canonical fold_result has invalid confusion counts")
        history = fold_result["training_history"]
        if len(history) != configuration["epochs"]:
            raise ValueError("Canonical fold_result requires complete training history")
        for epoch_number, epoch in enumerate(history, start=1):
            if (
                not isinstance(epoch, Mapping)
                or set(epoch) != {"epoch", "train_loss", "inner_validation_loss"}
                or epoch.get("epoch") != epoch_number
                or not _is_finite_number(epoch.get("train_loss"))
                or not _is_finite_number(epoch.get("inner_validation_loss"))
                or epoch["train_loss"] < 0
                or epoch["inner_validation_loss"] < 0
            ):
                raise ValueError("Canonical fold_result has invalid training history")
        selected_from_history = (
            int(
                np.argmin(
                    [float(epoch["inner_validation_loss"]) for epoch in history]
                )
            )
            + 1
        )
        if best_epoch != selected_from_history:
            raise ValueError(
                "Canonical best epoch must equal the minimum inner-validation loss"
            )
    if (
        outer_image_total != EXPECTED_HYGD_DATA_QUALITY["unique_image_hashes"]
        or outer_group_total
        != EXPECTED_HYGD_DATA_QUALITY["independent_evaluation_groups"]
    ):
        raise ValueError("Canonical fold_results do not cover the ratified evaluation set")
    artifacts = result.get("artifacts")
    required_artifacts = {"audit", "group_folds", "oof_images", "oof_groups"}
    if not isinstance(artifacts, Mapping) or set(artifacts) != required_artifacts:
        raise ValueError("Canonical result requires the exact artifact manifest")
    for relative in artifacts.values():
        path = Path(relative) if isinstance(relative, str) else Path()
        if (
            not isinstance(relative, str)
            or path.is_absolute()
            or len(path.parts) != 2
            or path.parts[0] != "results"
            or not path.name.startswith("internal_evaluation_repair")
        ):
            raise ValueError("Canonical result has invalid artifact path")
    if len(set(artifacts.values())) != len(artifacts):
        raise ValueError("Canonical artifact roles must use distinct paths")
    hashes = result.get("artifact_sha256")
    sizes = result.get("artifact_size_bytes")
    if (hashes is None) != (sizes is None):
        raise ValueError("Canonical artifact hashes and sizes must appear together")
    if hashes is not None:
        _validate_artifact_commitments(artifacts, hashes, sizes)
    artifact_names = {key: Path(value).name for key, value in artifacts.items()}
    canonical_stem = artifact_names["audit"].removesuffix("_audit.json")
    expected_artifact_names = {
        "audit": f"{canonical_stem}_audit.json",
        "group_folds": f"{canonical_stem}_group_folds.csv",
        "oof_images": f"{canonical_stem}_oof_images.csv",
        "oof_groups": f"{canonical_stem}_oof_groups.csv",
    }
    if (
        not canonical_stem.startswith("internal_evaluation_repair")
        or artifact_names != expected_artifact_names
    ):
        raise ValueError("Canonical artifact paths must share one coherent prefix")
    environment = result.get("environment")
    required_environment = {
        "python",
        "torch",
        "torchvision",
        "numpy",
        "pandas",
        "scikit_learn",
        "pillow",
        "platform",
    }
    if not isinstance(environment, Mapping) or set(environment) != required_environment:
        raise ValueError("Canonical result requires complete environment provenance")
    if any(
        not isinstance(environment.get(key), str) or not environment[key].strip()
        for key in required_environment
    ):
        raise ValueError("Canonical result requires nonblank environment provenance")
    if (
        not isinstance(configuration.get("device"), str)
        or not configuration["device"].strip()
    ):
        raise ValueError("Canonical result requires a nonblank device")
    runtime_seconds = result.get("runtime_seconds")
    if not _is_finite_number(runtime_seconds) or runtime_seconds < 0:
        raise ValueError("Canonical result requires a valid runtime_seconds")
    report = result.get("internal_evaluation")
    if not isinstance(report, Mapping):
        raise ValueError("Canonical result requires an internal_evaluation report")
    validate_primary_report(report)
    classification_keys = (
        "n",
        "positives",
        "negatives",
        "sensitivity",
        "specificity",
        "confusion_tn_fp_fn_tp",
    )

    def summed_fold_classification(metric_key: str) -> dict[str, object]:
        confusion = [
            sum(fold[metric_key]["confusion_tn_fp_fn_tp"][index] for fold in fold_results)
            for index in range(4)
        ]
        positives = confusion[2] + confusion[3]
        negatives = confusion[0] + confusion[1]
        return {
            "n": positives + negatives,
            "positives": positives,
            "negatives": negatives,
            "sensitivity": confusion[3] / positives,
            "specificity": confusion[0] / negatives,
            "confusion_tn_fp_fn_tp": confusion,
        }

    expected_group_classification = summed_fold_classification(
        "outer_test_group_level"
    )
    expected_image_classification = summed_fold_classification(
        "outer_test_image_level"
    )
    for context, actual, expected in (
        (
            "Canonical primary classification metrics",
            report["primary"],
            expected_group_classification,
        ),
        (
            "Canonical pooled group classification metrics",
            report["pooled_oof_group_descriptive"],
            expected_group_classification,
        ),
        (
            "Canonical secondary image classification metrics",
            report["secondary_image_level"],
            expected_image_classification,
        ),
    ):
        for key in classification_keys:
            if key == "confusion_tn_fp_fn_tp":
                matches = actual.get(key) == expected[key]
            elif _is_finite_number(expected[key]):
                matches = _is_finite_number(actual.get(key)) and np.isclose(
                    float(actual[key]), float(expected[key]), rtol=0.0, atol=1e-12
                )
            else:
                matches = actual.get(key) == expected[key]
            if not matches:
                raise ValueError(f"{context} are inconsistent")
    bootstrap = report["primary"]["bootstrap"]
    if (
        bootstrap["n_bootstrap"] != configuration["bootstrap_samples"]
        or bootstrap["valid_replicates"] != configuration["bootstrap_samples"]
        or bootstrap["seed"] != configuration["seed"] + 9_003
    ):
        raise ValueError("Canonical result has inconsistent bootstrap metadata")
    fold_auc_rows = report["primary"].get("fold_aurocs")
    if not isinstance(fold_auc_rows, list) or len(fold_auc_rows) != total_folds:
        raise ValueError("Canonical result requires five primary fold AUROCs")
    fold_aucs = []
    for expected_fold, (auc_row, fold_result) in enumerate(
        zip(fold_auc_rows, fold_results)
    ):
        expected_auc = fold_result["outer_test_group_level"]["auroc"]
        if (
            not isinstance(auc_row, Mapping)
            or auc_row.get("outer_fold") != expected_fold
            or auc_row.get("n_groups")
            != fold_result["counts"]["outer_test_groups"]
            or auc_row.get("positives")
            != fold_result["outer_test_group_level"]["positives"]
            or auc_row.get("negatives")
            != fold_result["outer_test_group_level"]["negatives"]
            or not _is_finite_number(auc_row.get("auroc"))
            or abs(float(auc_row["auroc"]) - float(expected_auc)) > 1e-12
        ):
            raise ValueError("Canonical primary fold AUROCs are inconsistent")
        fold_aucs.append(float(auc_row["auroc"]))
    if abs(float(report["primary"]["auroc"]) - float(np.mean(fold_aucs))) > 1e-12:
        raise ValueError("Canonical primary AUROC is not the mean outer-fold AUROC")
    fold_weights = [row["n_groups"] for row in fold_auc_rows]
    weighted_auc = float(np.average(fold_aucs, weights=fold_weights))
    if not np.isclose(
        float(report["primary"]["group_count_weighted_mean_auroc"]),
        weighted_auc,
        rtol=0.0,
        atol=1e-12,
    ):
        raise ValueError("Canonical weighted primary AUROC is inconsistent")
    keys = {key.lower() for key in _walk_mapping_keys(result)}
    forbidden = {
        "historical",
        "legacy",
        "test_auc",
        "bootstrap_95ci",
        "selected_configuration",
        "best_configuration",
        "best_config",
        "best_name",
        "cv_results",
        "fold_aucs",
        "mean_auc",
        "std_auc",
        "cross_validation",
    }
    found = sorted(keys & forbidden)
    if found:
        raise ValueError(
            "Canonical result cannot contain historical/CV selection fields: "
            + ", ".join(found)
        )


def _validate_artifact_commitments(
    artifacts: Mapping[str, object],
    hashes: object,
    sizes: object,
) -> None:
    """Validate one complete hash and size commitment for every artifact role."""

    if not isinstance(hashes, Mapping) or set(hashes) != set(artifacts):
        raise ValueError("Terminal bundle requires one artifact hash per artifact")
    if not isinstance(sizes, Mapping) or set(sizes) != set(artifacts):
        raise ValueError("Terminal bundle requires one artifact size per artifact")
    invalid_hashes = [key for key, value in hashes.items() if not _is_sha256(value)]
    if invalid_hashes:
        raise ValueError(
            "Terminal bundle hashes must be lowercase 64-hex: "
            + ", ".join(sorted(invalid_hashes))
        )
    invalid_sizes = [
        key for key, value in sizes.items() if type(value) is not int or value < 0
    ]
    if invalid_sizes:
        raise ValueError(
            "Terminal bundle sizes must be nonnegative integers: "
            + ", ".join(sorted(invalid_sizes))
        )


def _read_terminal_artifact_bytes(
    result: Mapping[str, object], root: str | Path
) -> dict[str, bytes]:
    """Read all committed artifacts through no-follow, single-link descriptors."""

    artifacts = result["artifacts"]
    hashes = result.get("artifact_sha256")
    sizes = result.get("artifact_size_bytes")
    _validate_artifact_commitments(artifacts, hashes, sizes)

    root = Path(root).resolve()
    logical_results = root / "results"
    directory_flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    directory_flags |= getattr(os, "O_NOFOLLOW", 0)
    try:
        directory_fd = os.open(logical_results, directory_flags)
    except OSError as error:
        raise ValueError(
            "Terminal bundle results directory must be a real confined directory"
        ) from error
    content_by_role: dict[str, bytes] = {}
    try:
        for key, relative in artifacts.items():
            path = Path(relative) if isinstance(relative, str) else Path()
            if (
                not isinstance(relative, str)
                or path.is_absolute()
                or path.parts[:1] != ("results",)
                or len(path.parts) != 2
            ):
                raise ValueError(f"Terminal bundle artifact path is invalid: {key}")
            flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
            flags |= getattr(os, "O_NONBLOCK", 0)
            try:
                artifact_fd = os.open(path.name, flags, dir_fd=directory_fd)
            except OSError as error:
                raise ValueError(
                    f"Terminal bundle artifact {key} is missing or unsafe"
                ) from error
            try:
                metadata = os.fstat(artifact_fd)
                if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
                    raise ValueError(
                        f"Terminal bundle artifact {key} must be a single-link regular file"
                    )
                if metadata.st_size != sizes[key]:
                    raise ValueError(f"Terminal bundle artifact size mismatch: {key}")
                digest = hashlib.sha256()
                content = bytearray()
                while True:
                    block = os.read(artifact_fd, 1 << 20)
                    if not block:
                        break
                    digest.update(block)
                    content.extend(block)
                if digest.hexdigest() != hashes[key]:
                    raise ValueError(f"Terminal bundle artifact hash mismatch: {key}")
                after = os.fstat(artifact_fd)
                if (
                    metadata.st_dev,
                    metadata.st_ino,
                    metadata.st_size,
                    metadata.st_mtime_ns,
                    metadata.st_ctime_ns,
                ) != (
                    after.st_dev,
                    after.st_ino,
                    after.st_size,
                    after.st_mtime_ns,
                    after.st_ctime_ns,
                ):
                    raise ValueError(f"Terminal bundle artifact changed while read: {key}")
                content_by_role[str(key)] = bytes(content)
            finally:
                os.close(artifact_fd)
    finally:
        os.close(directory_fd)
    return content_by_role


def _decode_json_artifact(content: bytes, context: str) -> object:
    try:
        return json.loads(content.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{context} must be valid UTF-8 JSON") from error


def _assert_nested_equivalent(
    actual: object,
    expected: object,
    context: str,
    *,
    tolerance: float = 1e-12,
) -> None:
    """Compare scientific structures without leaking row-level values in errors."""

    if isinstance(expected, Mapping):
        if not isinstance(actual, Mapping) or set(actual) != set(expected):
            raise ValueError(f"{context} schema is inconsistent")
        for key in expected:
            _assert_nested_equivalent(
                actual[key], expected[key], context, tolerance=tolerance
            )
        return
    if isinstance(expected, list):
        if not isinstance(actual, list) or len(actual) != len(expected):
            raise ValueError(f"{context} sequence is inconsistent")
        for actual_item, expected_item in zip(actual, expected):
            _assert_nested_equivalent(
                actual_item, expected_item, context, tolerance=tolerance
            )
        return
    if _is_finite_number(expected):
        if not _is_finite_number(actual) or not np.isclose(
            float(actual), float(expected), rtol=0.0, atol=tolerance
        ):
            raise ValueError(f"{context} numeric value is inconsistent")
        return
    if actual != expected or type(actual) is not type(expected):
        raise ValueError(f"{context} value is inconsistent")


def _read_terminal_csv(
    content: bytes,
    *,
    expected_columns: Sequence[str],
    string_columns: Sequence[str],
    context: str,
) -> pd.DataFrame:
    try:
        frame = pd.read_csv(
            io.BytesIO(content),
            dtype={column: "string" for column in string_columns},
        )
    except (UnicodeDecodeError, pd.errors.ParserError, ValueError) as error:
        raise ValueError(f"{context} must be valid CSV") from error
    if list(frame.columns) != list(expected_columns) or frame.empty:
        raise ValueError(f"{context} has invalid schema")
    return frame


def _strict_integer_series(
    frame: pd.DataFrame,
    column: str,
    context: str,
) -> pd.Series:
    values = pd.to_numeric(frame[column], errors="coerce")
    if (
        values.isna().any()
        or not np.isfinite(values.to_numpy(dtype=float)).all()
        or not np.equal(values, np.floor(values)).all()
    ):
        raise ValueError(f"{context} has invalid integer fields")
    return values.astype(int)


def _strict_probability_series(
    frame: pd.DataFrame,
    column: str,
    context: str,
) -> pd.Series:
    values = pd.to_numeric(frame[column], errors="coerce")
    numeric = values.to_numpy(dtype=float)
    if values.isna().any() or not np.isfinite(numeric).all() or np.any(
        (numeric < 0) | (numeric > 1)
    ):
        raise ValueError(f"{context} has invalid probability fields")
    return values.astype(float)


def _assert_frames_equivalent(
    actual: pd.DataFrame,
    expected: pd.DataFrame,
    *,
    key: str,
    integer_columns: Sequence[str],
    float_columns: Sequence[str],
    context: str,
) -> None:
    if list(actual.columns) != list(expected.columns) or len(actual) != len(expected):
        raise ValueError(f"{context} schema or row count is inconsistent")
    actual = actual.sort_values(key, kind="mergesort").reset_index(drop=True)
    expected = expected.sort_values(key, kind="mergesort").reset_index(drop=True)
    for column in actual.columns:
        if column in float_columns:
            left = pd.to_numeric(actual[column], errors="coerce").to_numpy(dtype=float)
            right = pd.to_numeric(expected[column], errors="coerce").to_numpy(dtype=float)
            if (
                not np.isfinite(left).all()
                or not np.isfinite(right).all()
                or not np.allclose(left, right, rtol=0.0, atol=1e-12)
            ):
                raise ValueError(f"{context} numeric values are inconsistent")
        elif column in integer_columns:
            left = _strict_integer_series(actual, column, context).to_numpy()
            right = _strict_integer_series(expected, column, context).to_numpy()
            if not np.array_equal(left, right):
                raise ValueError(f"{context} integer values are inconsistent")
        else:
            left = actual[column].astype(str).to_numpy()
            right = expected[column].astype(str).to_numpy()
            if not np.array_equal(left, right):
                raise ValueError(f"{context} identifiers are inconsistent")


def _validate_complete_oof_claims(
    result: Mapping[str, object],
    content_by_role: Mapping[str, bytes],
    *,
    expected_split_summary: object,
) -> list[dict[str, int]]:
    """Recompute the complete scientific receipt from committed OOF CSV bytes."""

    image_columns = [
        "image_name",
        "patient_id",
        "evaluation_group",
        "sha256",
        "label",
        "quality_score",
        "outer_fold",
        "probability",
        "selected_threshold",
        "predicted_class",
    ]
    images = _read_terminal_csv(
        content_by_role["oof_images"],
        expected_columns=image_columns,
        string_columns=("image_name", "patient_id", "evaluation_group", "sha256"),
        context="Terminal OOF image artifact",
    )
    if len(images) != EXPECTED_HYGD_DATA_QUALITY["unique_image_hashes"]:
        raise ValueError("Terminal OOF image row count is inconsistent")
    if images.drop(columns=["quality_score"]).isna().any().any():
        raise ValueError("Terminal OOF image artifact contains missing fields")
    for column in ("image_name", "patient_id", "evaluation_group", "sha256"):
        images[column] = images[column].astype(str).str.strip()
        if images[column].eq("").any():
            raise ValueError("Terminal OOF image identifiers cannot be blank")
    if (
        images["image_name"].duplicated().any()
        or not images["image_name"].map(
            lambda value: Path(value).name == value and value not in {".", ".."}
        ).all()
        or images["sha256"].duplicated().any()
        or not images["sha256"].map(_is_sha256).all()
    ):
        raise ValueError("Terminal OOF image identity is inconsistent")
    images["label"] = _strict_integer_series(
        images, "label", "Terminal OOF image artifact"
    )
    validate_canonical_oof_identity(images)
    images["outer_fold"] = _strict_integer_series(
        images, "outer_fold", "Terminal OOF image artifact"
    )
    images["predicted_class"] = _strict_integer_series(
        images, "predicted_class", "Terminal OOF image artifact"
    )
    if (
        not images["label"].isin([0, 1]).all()
        or not images["predicted_class"].isin([0, 1]).all()
        or set(images["outer_fold"]) != set(result["completed_outer_folds"])
    ):
        raise ValueError("Terminal OOF image labels or folds are inconsistent")
    images["probability"] = _strict_probability_series(
        images, "probability", "Terminal OOF image artifact"
    )
    images["selected_threshold"] = _strict_probability_series(
        images, "selected_threshold", "Terminal OOF image artifact"
    )
    expected_predictions = (
        images["probability"] >= images["selected_threshold"]
    ).astype(int)
    if not np.array_equal(images["predicted_class"].to_numpy(), expected_predictions):
        raise ValueError("Terminal OOF image predictions are inconsistent")
    quality = pd.to_numeric(images["quality_score"], errors="coerce")
    if quality.notna().any() and not np.isfinite(
        quality.dropna().to_numpy(dtype=float)
    ).all():
        raise ValueError("Terminal OOF image quality values are inconsistent")
    images["quality_score"] = quality

    group_columns = [
        "evaluation_group",
        "label",
        "probability",
        "selected_threshold",
        "outer_fold",
        "n_images",
        "predicted_class",
    ]
    committed_groups = _read_terminal_csv(
        content_by_role["oof_groups"],
        expected_columns=group_columns,
        string_columns=("evaluation_group",),
        context="Terminal OOF group artifact",
    )
    derived_groups = aggregate_groups(images)
    derived_groups["predicted_class"] = (
        derived_groups["probability"] >= derived_groups["selected_threshold"]
    ).astype(int)
    _assert_frames_equivalent(
        committed_groups,
        derived_groups[group_columns],
        key="evaluation_group",
        integer_columns=("label", "outer_fold", "n_images", "predicted_class"),
        float_columns=("probability", "selected_threshold"),
        context="Terminal OOF group artifact",
    )
    groups = derived_groups[group_columns].copy()
    if len(groups) != EXPECTED_HYGD_DATA_QUALITY["independent_evaluation_groups"]:
        raise ValueError("Terminal OOF group count is inconsistent")

    fold_columns = ["evaluation_group", "label", "n_images", "outer_fold"]
    committed_folds = _read_terminal_csv(
        content_by_role["group_folds"],
        expected_columns=fold_columns,
        string_columns=("evaluation_group",),
        context="Terminal group-fold artifact",
    )
    _assert_frames_equivalent(
        committed_folds,
        groups[fold_columns],
        key="evaluation_group",
        integer_columns=("label", "n_images", "outer_fold"),
        float_columns=(),
        context="Terminal group-fold artifact",
    )
    canonical_folds = make_outer_folds(
        images,
        n_splits=CANONICAL_RUN_CONFIG["outer_folds"],
        seed=CANONICAL_RUN_CONFIG["seed"],
    )
    validate_canonical_group_fold_assignment(canonical_folds)
    _assert_frames_equivalent(
        committed_folds,
        canonical_folds[fold_columns],
        key="evaluation_group",
        integer_columns=("label", "n_images", "outer_fold"),
        float_columns=(),
        context="Terminal canonical group-fold assignment",
    )
    split_summary = [
        {
            "outer_fold": int(fold),
            "groups": int(len(rows)),
            "images": int(rows["n_images"].sum()),
            "positive_groups": int(rows["label"].sum()),
            "negative_groups": int((1 - rows["label"]).sum()),
        }
        for fold, rows in committed_folds.groupby("outer_fold", sort=True)
    ]
    if expected_split_summary != split_summary:
        raise ValueError("Terminal audit split summary is inconsistent")

    for fold_result in result["fold_results"]:
        fold = fold_result["outer_fold"]
        fold_images = images.loc[images["outer_fold"] == fold].reset_index(drop=True)
        fold_groups = groups.loc[groups["outer_fold"] == fold].reset_index(drop=True)
        if (
            len(fold_images) != fold_result["counts"]["outer_test_images"]
            or len(fold_groups) != fold_result["counts"]["outer_test_groups"]
            or fold_images["selected_threshold"].nunique() != 1
            or not np.isclose(
                float(fold_images["selected_threshold"].iloc[0]),
                float(
                    fold_result["selected_threshold_from_inner_group_validation"][
                        "threshold"
                    ]
                ),
                rtol=0.0,
                atol=1e-12,
            )
        ):
            raise ValueError("Terminal OOF fold counts or threshold are inconsistent")
        _assert_nested_equivalent(
            fold_result["outer_test_image_level"],
            frame_metrics(fold_images),
            "Terminal OOF image fold metrics",
        )
        _assert_nested_equivalent(
            fold_result["outer_test_group_level"],
            frame_metrics(fold_groups),
            "Terminal OOF group fold metrics",
        )

    configuration = result["configuration_fixed_before_outer_evaluation"]
    bootstrap_samples = configuration["bootstrap_samples"]
    seed = configuration["seed"]
    expected_report = build_primary_report(
        outer_fold_auc=outer_fold_auc_summary(groups),
        fold_stratified_ci=fold_stratified_group_bootstrap(
            groups, bootstrap_samples, seed + 9_003
        ),
        pooled_group_metrics=frame_metrics(groups),
        pooled_group_cluster_ci=cluster_bootstrap(
            groups, "evaluation_group", bootstrap_samples, seed + 9_002
        ),
        image_metrics=frame_metrics(images),
        image_cluster_ci=cluster_bootstrap(
            images, "evaluation_group", bootstrap_samples, seed + 9_001
        ),
    )
    _assert_nested_equivalent(
        result["internal_evaluation"],
        expected_report,
        "Terminal OOF scientific report",
    )
    return split_summary


def validate_terminal_bundle(result: Mapping[str, object], root: str | Path) -> None:
    """Verify each committed artifact through no-follow directory descriptors."""

    validate_internal_result_payload(result)
    artifacts = result["artifacts"]
    content_by_role = _read_terminal_artifact_bytes(result, root)
    audit_payload = _decode_json_artifact(content_by_role["audit"], "Terminal audit")

    audit_path = Path(artifacts["audit"])
    expected_result = str(
        audit_path.with_name(audit_path.name.removesuffix("_audit.json") + ".json")
    )
    expected_audit_keys = {
        "protocol_version",
        "status",
        "dataset_identity",
        "data_quality",
        "split_summary",
        "group_fold_manifest",
        "result_artifact",
        "result_claim_sha256",
        "completed_outer_folds",
    }
    if (
        not isinstance(audit_payload, Mapping)
        or set(audit_payload) != expected_audit_keys
        or audit_payload.get("protocol_version") != PROTOCOL_VERSION
        or audit_payload.get("status") != "complete"
        or audit_payload.get("dataset_identity") != result.get("dataset_identity")
        or audit_payload.get("data_quality") != result.get("data_quality")
        or audit_payload.get("group_fold_manifest") != artifacts["group_folds"]
        or audit_payload.get("completed_outer_folds")
        != result.get("completed_outer_folds")
        or audit_payload.get("result_artifact") != expected_result
        or audit_payload.get("result_claim_sha256")
        != terminal_result_claim_sha256(result)
    ):
        raise ValueError("Terminal audit is inconsistent with the result marker")
    _validate_complete_oof_claims(
        result,
        content_by_role,
        expected_split_summary=audit_payload.get("split_summary"),
    )


def validate_smoke_result_payload(result: Mapping[str, object]) -> None:
    """Fail if an incomplete execution smoke result resembles canonical evidence."""

    required_result_keys = {
        "protocol_version",
        "status",
        "canonical_internal_estimate",
        "completed_outer_folds",
        "planned_outer_folds",
        "smoke_diagnostics",
        "artifacts",
        "runtime_seconds",
    }
    terminal_result_keys = required_result_keys | {
        "artifact_sha256",
        "artifact_size_bytes",
    }
    if set(result) not in (required_result_keys, terminal_result_keys):
        raise ValueError("Smoke result must use the exact outcome-blind schema")
    if result.get("protocol_version") != PROTOCOL_VERSION:
        raise ValueError("Smoke result has invalid protocol_version")
    if result.get("status") != "partial_smoke_run":
        raise ValueError("Smoke result status must be partial_smoke_run")
    if result.get("canonical_internal_estimate") is not False:
        raise ValueError("Smoke result cannot be a canonical internal estimate")
    if (
        type(result.get("planned_outer_folds")) is not int
        or not isinstance(result.get("completed_outer_folds"), list)
        or any(type(fold) is not int for fold in result["completed_outer_folds"])
    ):
        raise ValueError("Smoke result requires exact integer fold metadata")
    if "internal_evaluation" in result:
        raise ValueError("Smoke result cannot contain internal_evaluation")
    diagnostic = result.get("smoke_diagnostics")
    if not isinstance(diagnostic, Mapping):
        raise ValueError("Smoke result requires smoke_diagnostics")
    if diagnostic.get("status") != "partial_smoke_noncanonical":
        raise ValueError("Smoke diagnostics must be explicitly noncanonical")
    if diagnostic.get("canonical_internal_estimate") is not False:
        raise ValueError("Smoke diagnostics cannot be a canonical internal estimate")
    expected_diagnostic_keys = {
        "status",
        "canonical_internal_estimate",
        "warning",
        "completed_outer_folds",
        "planned_outer_folds",
        "fold_execution",
        "outcomes_emitted",
    }
    if set(diagnostic) != expected_diagnostic_keys:
        raise ValueError("Smoke diagnostics must use the exact outcome-blind schema")
    if diagnostic.get("outcomes_emitted") is not False:
        raise ValueError("Smoke diagnostics must remain outcome-blind")
    if result.get("completed_outer_folds") != diagnostic.get(
        "completed_outer_folds"
    ) or result.get("planned_outer_folds") != diagnostic.get("planned_outer_folds"):
        raise ValueError("Smoke result fold metadata is inconsistent")
    try:
        expected_diagnostic = build_smoke_diagnostics(
            fold_execution=diagnostic["fold_execution"],
            completed_outer_folds=result["completed_outer_folds"],
            total_outer_folds=result["planned_outer_folds"],
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("Smoke diagnostics must use the exact outcome-blind schema") from error
    if dict(diagnostic) != expected_diagnostic:
        raise ValueError("Smoke diagnostics must use the exact outcome-blind schema")
    artifacts = result.get("artifacts")
    if not isinstance(artifacts, Mapping) or set(artifacts) != {
        "audit",
        "group_folds",
    }:
        raise ValueError("Smoke result requires only audit and group_folds artifacts")
    for relative in artifacts.values():
        path = Path(relative) if isinstance(relative, str) else Path()
        if (
            not isinstance(relative, str)
            or path.is_absolute()
            or len(path.parts) != 2
            or path.parts[0] != "results"
            or not path.name.startswith("internal_evaluation_repair")
        ):
            raise ValueError("Smoke result has invalid artifact path")
    if len(set(artifacts.values())) != len(artifacts):
        raise ValueError("Smoke artifact roles must use distinct paths")
    audit_path = Path(artifacts["audit"])
    group_path = Path(artifacts["group_folds"])
    smoke_stem = audit_path.name.removesuffix("_audit.json")
    expected_smoke_suffix = (
        f"_smoke_{len(result['completed_outer_folds'])}of{result['planned_outer_folds']}"
    )
    if (
        not audit_path.name.endswith("_audit.json")
        or not group_path.name.endswith("_group_folds.csv")
        or smoke_stem != group_path.name.removesuffix("_group_folds.csv")
        or not smoke_stem.startswith("internal_evaluation_repair")
        or not smoke_stem.endswith(expected_smoke_suffix)
    ):
        raise ValueError("Smoke artifact paths must share one coherent prefix")
    runtime_seconds = result.get("runtime_seconds")
    if not _is_finite_number(runtime_seconds) or runtime_seconds < 0:
        raise ValueError("Smoke result requires valid runtime_seconds")
    keys = {key.lower() for key in _walk_mapping_keys(result)}
    forbidden_fragments = (
        "auroc",
        "auc",
        "probability",
        "threshold",
        "sensitivity",
        "specificity",
        "confusion",
        "predicted_class",
        "oof",
        "95ci",
        "preferred_internal_metric_source",
        "authoritative_metric_source",
    )
    if any(
        fragment in key for key in keys for fragment in forbidden_fragments
    ):
        raise ValueError("Smoke diagnostics must remain outcome-blind")
    hashes = result.get("artifact_sha256")
    sizes = result.get("artifact_size_bytes")
    if (hashes is None) != (sizes is None):
        raise ValueError("Smoke artifact hashes and sizes must appear together")
    if hashes is not None:
        _validate_artifact_commitments(artifacts, hashes, sizes)


def validate_terminal_smoke_bundle(
    result: Mapping[str, object], root: str | Path
) -> None:
    """Verify a noncanonical smoke marker and its two committed artifacts."""

    validate_smoke_result_payload(result)
    if "artifact_sha256" not in result or "artifact_size_bytes" not in result:
        raise ValueError("Terminal smoke bundle requires hash and size commitments")
    artifacts = result["artifacts"]
    content_by_role = _read_terminal_artifact_bytes(result, root)
    audit = _decode_json_artifact(content_by_role["audit"], "Terminal smoke audit")
    expected_audit_keys = {
        "protocol_version",
        "status",
        "dataset_identity",
        "data_quality",
        "split_summary",
        "group_fold_manifest",
        "result_artifact",
        "result_claim_sha256",
        "completed_outer_folds",
    }
    audit_path = Path(artifacts["audit"])
    expected_result = str(
        audit_path.with_name(audit_path.name.removesuffix("_audit.json") + ".json")
    )
    if (
        not isinstance(audit, Mapping)
        or set(audit) != expected_audit_keys
        or audit.get("protocol_version") != PROTOCOL_VERSION
        or audit.get("status") != "partial_smoke_run"
        or audit.get("completed_outer_folds") != result.get("completed_outer_folds")
        or audit.get("group_fold_manifest") != artifacts["group_folds"]
        or audit.get("result_artifact") != expected_result
        or audit.get("result_claim_sha256")
        != terminal_result_claim_sha256(result)
    ):
        raise ValueError("Terminal smoke audit is inconsistent with the result marker")
    validate_canonical_dataset_identity(audit.get("dataset_identity"))
    validate_canonical_data_quality(audit.get("data_quality"))

    try:
        group_folds = pd.read_csv(
            io.BytesIO(content_by_role["group_folds"]),
            dtype={"evaluation_group": "string"},
        )
    except (UnicodeDecodeError, pd.errors.ParserError, ValueError) as error:
        raise ValueError("Terminal smoke group_folds must be valid CSV") from error
    expected_columns = ["evaluation_group", "label", "n_images", "outer_fold"]
    if list(group_folds.columns) != expected_columns or group_folds.empty:
        raise ValueError("Terminal smoke group_folds has invalid schema")
    if group_folds.isna().any().any():
        raise ValueError("Terminal smoke group_folds cannot contain missing values")
    groups = group_folds["evaluation_group"].astype(str).str.strip()
    labels = pd.to_numeric(group_folds["label"], errors="coerce")
    n_images = pd.to_numeric(group_folds["n_images"], errors="coerce")
    folds = pd.to_numeric(group_folds["outer_fold"], errors="coerce")
    if (
        groups.eq("").any()
        or groups.duplicated().any()
        or labels.isna().any()
        or not labels.isin([0, 1]).all()
        or n_images.isna().any()
        or not np.equal(n_images, np.floor(n_images)).all()
        or (n_images <= 0).any()
        or folds.isna().any()
        or not np.equal(folds, np.floor(folds)).all()
        or set(folds.astype(int)) != set(range(result["planned_outer_folds"]))
        or len(group_folds) != EXPECTED_HYGD_DATA_QUALITY["independent_evaluation_groups"]
        or int(n_images.sum()) != EXPECTED_HYGD_DATA_QUALITY["unique_image_hashes"]
    ):
        raise ValueError("Terminal smoke group_folds is inconsistent")
    normalized = pd.DataFrame(
        {
            "evaluation_group": groups,
            "label": labels.astype(int),
            "n_images": n_images.astype(int),
            "outer_fold": folds.astype(int),
        }
    )
    validate_canonical_group_fold_assignment(normalized)
    split_summary = [
        {
            "outer_fold": int(fold),
            "groups": int(len(rows)),
            "images": int(rows["n_images"].sum()),
            "positive_groups": int(rows["label"].sum()),
            "negative_groups": int((1 - rows["label"]).sum()),
        }
        for fold, rows in normalized.groupby("outer_fold", sort=True)
    ]
    if audit.get("split_summary") != split_summary:
        raise ValueError("Terminal smoke split summary is inconsistent")
    images_by_fold = normalized.groupby("outer_fold")["n_images"].sum().to_dict()
    for receipt in result["smoke_diagnostics"]["fold_execution"]:
        if receipt["outer_inference_rows"] != int(images_by_fold[receipt["outer_fold"]]):
            raise ValueError("Terminal smoke execution row count is inconsistent")


def resolve_repair_output_prefix(root: str | Path, requested: str | Path) -> Path:
    """Resolve output only inside the git-ignored repair-results namespace."""

    root = Path(root).resolve()
    requested = Path(requested)
    candidate = requested if requested.is_absolute() else root / requested
    resolved = candidate.resolve(strict=False)
    logical_results_root = root / "results"
    if logical_results_root.is_symlink():
        raise ValueError(
            "The results directory cannot be a symlink; repair outputs must stay "
            "under results/internal_evaluation_repair*"
        )
    allowed_root = logical_results_root.resolve()
    try:
        relative = resolved.relative_to(allowed_root)
    except ValueError as error:
        raise ValueError(
            "Repair outputs must stay under results/internal_evaluation_repair*"
        ) from error
    if (
        len(relative.parts) != 1
        or not relative.parts[0].startswith("internal_evaluation_repair")
    ):
        raise ValueError(
            "Repair outputs must stay under results/internal_evaluation_repair*"
        )
    return resolved


def resolve_run_output_prefix(
    root: str | Path,
    requested: str | Path,
    *,
    max_folds: int | None,
    total_folds: int,
    audit_only: bool,
) -> Path:
    """Give audits and incomplete smokes distinct, noncanonical namespaces."""

    if total_folds < 2:
        raise ValueError("total_folds must be at least 2")
    prefix = resolve_repair_output_prefix(root, requested)
    if audit_only:
        if max_folds is not None:
            raise ValueError("audit_only cannot be combined with max_folds")
        return Path(f"{prefix}_audit_only")
    if max_folds is None:
        return prefix
    if max_folds <= 0:
        raise ValueError("max_folds must be positive")
    if max_folds >= total_folds:
        raise ValueError("max_folds must be a proper subset of outer folds")
    return Path(f"{prefix}_smoke_{max_folds}of{total_folds}")


def assert_fresh_output_bundle(paths: Sequence[str | Path]) -> None:
    """Refuse to start when any target entry already exists, including a symlink."""

    normalized = [Path(path) for path in paths]
    canonical_names = [os.path.abspath(os.fspath(path)) for path in normalized]
    if len(set(canonical_names)) != len(canonical_names):
        raise ValueError("Output bundle paths must be unique")
    existing = [path for path in normalized if os.path.lexists(path)]
    if existing:
        raise FileExistsError(
            "Output bundle is fail-closed; refusing to overwrite existing target: "
            + str(existing[0])
        )


def resolve_private_output_path(
    root: str | Path,
    requested: str | Path,
    namespace: str | Path,
) -> Path:
    """Confine one generated file directly under a named ignored namespace."""

    root = Path(root).resolve()
    namespace = Path(namespace)
    if namespace.is_absolute() or not namespace.parts or ".." in namespace.parts:
        raise ValueError("Private output namespace must be repository-relative")
    allowed = root / namespace
    for candidate_parent in (root / Path(*namespace.parts[:index]) for index in range(1, len(namespace.parts) + 1)):
        if os.path.lexists(candidate_parent) and candidate_parent.is_symlink():
            raise ValueError("Private output namespace cannot contain symlinks")
    requested = Path(requested)
    candidate = requested if requested.is_absolute() else root / requested
    resolved = candidate.resolve(strict=False)
    allowed_resolved = allowed.resolve(strict=False)
    try:
        relative = resolved.relative_to(allowed_resolved)
    except ValueError as error:
        raise ValueError("Generated output must stay inside its private namespace") from error
    if len(relative.parts) != 1 or relative.name in {"", ".", ".."}:
        raise ValueError("Generated output must be one direct namespace file")
    return resolved


def _open_directory_chain_no_symlinks(directory: str | Path) -> int:
    """Open/create an absolute directory chain one descriptor-bound step at a time."""

    absolute = Path(os.path.abspath(os.fspath(directory)))
    # macOS exposes trusted system roots such as /var through a top-level
    # symlink (/var -> /private/var). Resolve only that root-owned first hop;
    # every task-controlled descendant is still opened with O_NOFOLLOW.
    if len(absolute.parts) > 1:
        first_component = Path(absolute.anchor) / absolute.parts[1]
        if first_component.is_symlink():
            absolute = first_component.resolve(strict=True).joinpath(*absolute.parts[2:])
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(absolute.anchor or os.sep, flags)
    try:
        for component in absolute.parts[1:]:
            try:
                os.mkdir(component, mode=0o755, dir_fd=descriptor)
            except FileExistsError:
                pass
            next_descriptor = os.open(component, flags, dir_fd=descriptor)
            os.close(descriptor)
            descriptor = next_descriptor
        return descriptor
    except BaseException:
        os.close(descriptor)
        raise


def atomic_write_bytes(
    path: str | Path,
    content: bytes,
    *,
    overwrite: bool = False,
) -> None:
    """Publish bytes through a descriptor-bound parent without following links."""

    path = Path(path)
    if path.name in {"", ".", ".."}:
        raise ValueError("Atomic output must name a file")
    directory_fd = _open_directory_chain_no_symlinks(path.parent)
    temporary_name = f"{path.name}.{secrets.token_hex(8)}.tmp"
    temporary_exists = False
    try:
        file_flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
        file_flags |= getattr(os, "O_NOFOLLOW", 0)
        temporary_fd = os.open(
            temporary_name,
            file_flags,
            0o600,
            dir_fd=directory_fd,
        )
        temporary_exists = True
        with os.fdopen(temporary_fd, "wb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        if overwrite:
            os.replace(
                temporary_name,
                path.name,
                src_dir_fd=directory_fd,
                dst_dir_fd=directory_fd,
            )
        else:
            os.link(
                temporary_name,
                path.name,
                src_dir_fd=directory_fd,
                dst_dir_fd=directory_fd,
                follow_symlinks=False,
            )
            os.unlink(temporary_name, dir_fd=directory_fd)
        os.fsync(directory_fd)
        temporary_exists = False
    finally:
        if temporary_exists:
            try:
                os.unlink(temporary_name, dir_fd=directory_fd)
            except FileNotFoundError:
                pass
        os.close(directory_fd)


def atomic_write_text(
    path: str | Path,
    text: str,
    *,
    overwrite: bool = False,
) -> None:
    """Publish UTF-8 text atomically without following the final target."""

    atomic_write_bytes(path, text.encode("utf-8"), overwrite=overwrite)


def load_torch_state_dict_safely(
    checkpoint: str | Path,
    *,
    map_location: str = "cpu",
    expected_sha256: str | None = None,
) -> tuple[Mapping[str, object], str]:
    """Load a single-link checkpoint as weights-only and return its byte digest."""

    content, digest = read_verified_file_bytes(
        checkpoint, expected_sha256=expected_sha256
    )
    import torch  # Optional heavyweight dependency; intentionally lazy.

    state = torch.load(
        io.BytesIO(content), map_location=map_location, weights_only=True
    )
    if not isinstance(state, Mapping):
        raise ValueError("Checkpoint must contain a weights-only state mapping")
    return state, digest


def atomic_torch_save(
    path: str | Path,
    state: Mapping[str, object],
    *,
    overwrite: bool = False,
) -> str:
    """Serialize a state mapping in memory, then atomically publish its bytes."""

    import torch  # Optional heavyweight dependency; intentionally lazy.

    buffer = io.BytesIO()
    torch.save(state, buffer)
    content = buffer.getvalue()
    atomic_write_bytes(path, content, overwrite=overwrite)
    return hashlib.sha256(content).hexdigest()


def read_confined_json(root: str | Path, relative: str | Path) -> dict:
    """Read one results JSON through no-follow descriptors and reject hardlinks."""

    root = Path(root).resolve()
    path = Path(relative)
    if path.is_absolute() or path.parts[:1] != ("results",) or len(path.parts) != 2:
        raise ValueError("Confined JSON path must be one file directly under results")
    directory_flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    directory_flags |= getattr(os, "O_NOFOLLOW", 0)
    file_flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    file_flags |= getattr(os, "O_NONBLOCK", 0)
    try:
        directory_fd = os.open(root / "results", directory_flags)
    except OSError as error:
        raise ValueError("Confined JSON results directory is unsafe") from error
    try:
        try:
            file_fd = os.open(path.name, file_flags, dir_fd=directory_fd)
        except FileNotFoundError:
            raise
        except OSError as error:
            raise ValueError("Confined JSON file is missing or unsafe") from error
        try:
            metadata = os.fstat(file_fd)
            if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
                raise ValueError("Confined JSON must be a single-link regular file")
            content = bytearray()
            while True:
                block = os.read(file_fd, 1 << 20)
                if not block:
                    break
                content.extend(block)
        finally:
            os.close(file_fd)
    finally:
        os.close(directory_fd)
    try:
        payload = json.loads(content.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("Confined JSON must contain valid UTF-8 JSON") from error
    if not isinstance(payload, dict):
        raise ValueError("Confined JSON top level must be an object")
    return payload


def validate_oof_predictions(
    oof: pd.DataFrame,
    metadata: pd.DataFrame,
    *,
    complete: bool,
) -> None:
    """Fail closed on repeated, missing, or reassigned OOF image hashes."""

    required = ["sha256", "evaluation_group", "outer_fold"]
    _require_columns(oof, required, "OOF predictions")
    _require_columns(metadata, required, "evaluation metadata")
    if oof["sha256"].duplicated().any():
        raise AssertionError("OOF predictions contain duplicate hashes")

    expected = metadata[required].drop_duplicates("sha256")
    observed = oof[required]
    expected_hashes = set(expected["sha256"])
    observed_hashes = set(observed["sha256"])
    if complete and (
        len(observed) != len(expected)
        or observed_hashes != expected_hashes
    ):
        raise AssertionError("Complete OOF predictions are incomplete")
    if not observed_hashes.issubset(expected_hashes):
        raise AssertionError("OOF predictions contain unknown hashes")

    compared = observed.merge(
        expected,
        on="sha256",
        how="left",
        suffixes=("_observed", "_expected"),
        validate="one_to_one",
    )
    group_match = (
        compared["evaluation_group_observed"].astype(str)
        == compared["evaluation_group_expected"].astype(str)
    )
    fold_match = (
        compared["outer_fold_observed"].astype(int)
        == compared["outer_fold_expected"].astype(int)
    )
    if not bool((group_match & fold_match).all()):
        raise AssertionError("OOF prediction metadata mismatch")
    if complete and set(observed["evaluation_group"].astype(str)) != set(
        expected["evaluation_group"].astype(str)
    ):
        raise AssertionError("Complete OOF group coverage is incomplete")


def terminal_result_claim_sha256(result: Mapping[str, object]) -> str:
    """Bind the complete pre-publication result without circular audit hashes."""

    claim = copy.deepcopy(dict(result))
    claim.pop("artifact_sha256", None)
    claim.pop("artifact_size_bytes", None)
    payload = json.dumps(
        claim,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
        allow_nan=False,
    ).encode("ascii")
    return hashlib.sha256(b"HYGD_TERMINAL_RESULT_CLAIM_V1\n" + payload + b"\n").hexdigest()


def finalize_audit(
    audit: Mapping[str, object],
    *,
    result_status: str,
    result_path: str,
    completed_outer_folds: Sequence[int],
    result_claim_sha256: str,
) -> dict:
    """Return a terminal audit record after a successful evaluation run."""

    if result_status not in {"complete", "partial_smoke_run"}:
        raise ValueError("result_status must be complete or partial_smoke_run")
    if not _is_sha256(result_claim_sha256):
        raise ValueError("result_claim_sha256 must be lowercase 64-hex")
    finalized = copy.deepcopy(dict(audit))
    finalized.update(
        {
            "status": result_status,
            "result_artifact": str(result_path),
            "completed_outer_folds": [int(fold) for fold in completed_outer_folds],
            "result_claim_sha256": result_claim_sha256,
        }
    )
    return finalized
