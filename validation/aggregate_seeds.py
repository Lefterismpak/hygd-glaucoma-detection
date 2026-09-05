"""Aggregate five current patient-aware forward runs as adaptive evidence only."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from validation.evaluation_utils import (  # noqa: E402
    assert_fresh_output_bundle,
    atomic_write_text,
    read_verified_file_bytes,
    resolve_private_output_path,
)
from validation.seed_aggregation_utils import (  # noqa: E402
    is_lowercase_sha256,
    require_exact_keys,
    require_probability,
    validate_evidence_status,
    validate_patient_cluster_uncertainty,
    validate_seed_filename,
)

NAMESPACE = "results/patient_aware"
INPUT_PATTERN = "attemptB_seed{seed}.json"
DEFAULT_OUTPUT = f"{NAMESPACE}/seed_robustness_attemptB_patient_cluster_v2.json"
EXPECTED_RUN_KEYS = {
    "_evidence_status",
    "rimone_subject_mapping_sha256",
    "attempt",
    "seed",
    "adaptive_papila_eye_level_auroc_TTA",
    "papila_patient_cluster_uncertainty",
    "vcdr_weight",
    "vcdr_supervised_on",
    "attempt_A_was",
}
EXPECTED_VCDR_SOURCE = (
    "RIM-ONE only (dataset-provided mask-derived VCDR; historical "
    "auxiliary-label rule); HYGD masked out"
)


def _load_run(path: Path) -> tuple[dict, bytes]:
    content, _ = read_verified_file_bytes(path)
    try:
        payload = json.loads(content)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"Invalid seed JSON: {path.name}") from error
    if not isinstance(payload, dict):
        raise ValueError(f"Seed artifact must be a JSON object: {path.name}")
    return payload, hashlib.sha256(content).digest()


def summarize_runs(paths: Sequence[Path]) -> dict:
    if len(paths) != 5:
        raise ValueError("Exactly five forward seed artifacts are required")
    runs = []
    mapping_hashes = set()
    source_records = []
    for path in paths:
        payload, source_digest = _load_run(path)
        require_exact_keys(payload, EXPECTED_RUN_KEYS, f"Forward artifact {path.name}")
        validate_evidence_status(
            payload["_evidence_status"], f"Forward artifact {path.name}"
        )
        if (
            payload["attempt"] != "B_multitask_VCDR"
            or type(payload["vcdr_weight"]) is not float
            or payload["vcdr_weight"] != 0.5
            or type(payload["attempt_A_was"]) is not float
            or payload["attempt_A_was"] != 0.8251
            or payload["vcdr_supervised_on"] != EXPECTED_VCDR_SOURCE
        ):
            raise ValueError(f"Seed artifact run-family fields differ: {path.name}")
        seed = payload["seed"]
        auroc = require_probability(
            payload["adaptive_papila_eye_level_auroc_TTA"],
            f"Forward AUROC: {path.name}",
        )
        mapping_hash = payload["rimone_subject_mapping_sha256"]
        if type(seed) is not int or seed not in range(5):
            raise ValueError(f"Invalid seed identity: {path.name}")
        validate_seed_filename(
            path, pattern=INPUT_PATTERN, seed=seed, context="Forward artifact"
        )
        if not is_lowercase_sha256(mapping_hash):
            raise ValueError(f"Invalid subject-map binding: {path.name}")
        validate_patient_cluster_uncertainty(
            payload["papila_patient_cluster_uncertainty"],
            top_level_auroc=auroc,
            context=f"Forward patient-cluster uncertainty: {path.name}",
        )
        mapping_hashes.add(mapping_hash)
        runs.append((seed, auroc))
        source_records.append(
            {"seed": seed, "filename": path.name, "sha256": source_digest.hex()}
        )
    runs.sort()
    if [seed for seed, _ in runs] != list(range(5)):
        raise ValueError("Forward seed artifacts must contain unique seeds 0..4")
    if len(mapping_hashes) != 1:
        raise ValueError("Forward seed artifacts must share one subject-map binding")
    source_records.sort(key=lambda row: row["seed"])
    values = np.asarray([value for _, value in runs], dtype=float)
    return {
        "_evidence_status": {"status": "adaptive_development_evidence", "target_blind": False, "transportability_established": False, "license_compatibility": "needs-proof", "external_claim_permitted": False, "warning": "Target AUROC was visible during development; this five-run summary is descriptive only."},
        "attempt": "B_multitask_VCDR",
        "metric": "adaptive_PAPILA_eye_level_AUROC_TTA",
        "uncertainty_status": "descriptive_sample_sd_not_confidence_interval",
        "seed_only_run_family_identity": "bound_to_current_schema_and_one_subject_map_but_not_confirmatory",
        "rimone_subject_mapping_sha256": next(iter(mapping_hashes)),
        "n_seeds": 5,
        "seeds": list(range(5)),
        "per_seed_auroc": [round(value, 4) for value in values],
        "mean": round(float(values.mean()), 4),
        "std": round(float(values.std(ddof=1)), 4),
        "min": round(float(values.min()), 4),
        "max": round(float(values.max()), 4),
        "input_bundle_algorithm": "sha256(HYGD_FORWARD_SEED_BUNDLE_V1 newline + canonical JSON seed/file/hash records + newline)",
        "input_bundle_sha256": hashlib.sha256(
            b"HYGD_FORWARD_SEED_BUNDLE_V1\n"
            + json.dumps(
                source_records,
                ensure_ascii=True,
                separators=(",", ":"),
                sort_keys=True,
            ).encode("ascii")
            + b"\n"
        ).hexdigest(),
        "note": "Sample SD across five adaptive runs is not a confidence interval; per-run patient-cluster intervals do not make this chronology confirmatory.",
    }


def run(argv: Sequence[str] | None = None, *, root: Path = ROOT) -> dict | None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", default=NAMESPACE)
    parser.add_argument("--out", default=DEFAULT_OUTPUT)
    parser.add_argument("--describe-policy", action="store_true")
    args = parser.parse_args(argv)
    if args.describe_policy:
        policy = {"status": "adaptive_development_only", "input_namespace": NAMESPACE, "output_namespace": NAMESPACE, "required_seeds": list(range(5)), "external_claim_permitted": False}
        print(json.dumps(policy, indent=2))
        return policy
    if Path(args.input_dir).is_absolute() or Path(args.input_dir).parts != Path(NAMESPACE).parts:
        raise ValueError(f"Seed inputs must come from {NAMESPACE}")
    paths = [resolve_private_output_path(root, f"{NAMESPACE}/{INPUT_PATTERN.format(seed=seed)}", NAMESPACE) for seed in range(5)]
    out = resolve_private_output_path(root, args.out, NAMESPACE)
    assert_fresh_output_bundle([out])
    summary = summarize_runs(paths)
    atomic_write_text(out, json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    run()
