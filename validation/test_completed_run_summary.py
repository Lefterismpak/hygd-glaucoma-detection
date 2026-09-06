"""Exercise aggregate export through a real synthetic terminal bundle."""
import copy
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


class CompletedRunSummaryTests(unittest.TestCase):
    def test_partial_and_wrong_hash_cannot_become_complete_public_evidence(self):
        from validation.summarize_completed_run import summarize_completed_run
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "results").mkdir()
            source = root / "results/internal_evaluation_repair.json"
            source.write_text(json.dumps({"status": "partial_smoke_run"}))
            digest = hashlib.sha256(source.read_bytes()).hexdigest()
            for expected in [digest, "0" * 64]:
                with self.subTest(expected=expected), self.assertRaises(ValueError):
                    summarize_completed_run(root, "results/internal_evaluation_repair.json", expected_result_sha256=expected)

    def test_private_directory_and_linked_results_are_rejected(self):
        from validation.summarize_completed_run import summarize_completed_run
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            actual = root / "outside"
            actual.mkdir()
            (root / "results").symlink_to(actual, target_is_directory=True)
            source = actual / "internal_evaluation_repair.json"
            source.write_text('{}')
            with self.assertRaises(ValueError):
                summarize_completed_run(root, "results/internal_evaluation_repair.json", expected_result_sha256=hashlib.sha256(source.read_bytes()).hexdigest())

    def test_verified_bundle_exports_only_aggregates_and_rejects_tampering(self):
        # Reuse only a synthetic fixture, not production expectation builders.
        from validation.test_evaluation_utils import EvaluationUtilsTests
        from validation.evaluation_utils import (group_fold_assignment_fingerprint,
            oof_identity_fingerprint, terminal_result_claim_sha256)
        from validation.summarize_completed_run import summarize_completed_run
        result, images, groups, group_folds = EvaluationUtilsTests().canonical_oof_bundle()
        for key in ("python", "torch", "torchvision", "numpy", "pandas", "scikit_learn", "pillow"):
            result["environment"][key] = "0.0.0+synthetic"
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "results").mkdir()
            audit = {
                "protocol_version": result["protocol_version"], "status": "complete",
                "dataset_identity": result["dataset_identity"], "data_quality": result["data_quality"],
                "split_summary": [{"outer_fold": int(fold), "groups": len(rows),
                    "images": int(rows.n_images.sum()), "positive_groups": int(rows.label.sum()),
                    "negative_groups": int((1 - rows.label).sum())}
                    for fold, rows in group_folds.groupby("outer_fold")],
                "group_fold_manifest": result["artifacts"]["group_folds"],
                "completed_outer_folds": result["completed_outer_folds"],
                "result_artifact": "results/internal_evaluation_repair.json",
                "result_claim_sha256": terminal_result_claim_sha256(result),
            }
            for role, relative in result["artifacts"].items():
                file = root / relative
                if role == "audit":
                    file.write_text(json.dumps(audit))
                else:
                    {"group_folds": group_folds, "oof_groups": groups, "oof_images": images}[role].to_csv(file, index=False)
            result["artifact_sha256"] = {key: hashlib.sha256((root / rel).read_bytes()).hexdigest() for key, rel in result["artifacts"].items()}
            result["artifact_size_bytes"] = {key: (root / rel).stat().st_size for key, rel in result["artifacts"].items()}
            file = root / "results/internal_evaluation_repair.json"
            file.write_text(json.dumps(result))
            digest = hashlib.sha256(file.read_bytes()).hexdigest()
            with patch("validation.evaluation_utils.EXPECTED_HYGD_DEDUPLICATED_OOF_IDENTITY_SHA256", oof_identity_fingerprint(images)), patch("validation.evaluation_utils.EXPECTED_HYGD_CANONICAL_GROUP_FOLD_SHA256", group_fold_assignment_fingerprint(group_folds)):
                summary = summarize_completed_run(root, "results/internal_evaluation_repair.json", expected_result_sha256=digest)
                self.assertEqual(summary["primary"]["n"], 283)
                self.assertEqual(summary["private_source_binding"]["result_sha256"], digest)
                self.assertEqual(summary["probability_diagnostics"]["evaluation_groups"], 283)
                encoded = json.dumps(summary)
                self.assertNotIn(directory, encoded)
                self.assertNotIn("patient_id", encoded)
                self.assertNotIn("image_name", encoded)
                # Real post-completion tampering must stop before any export.
                group_file = root / result["artifacts"]["oof_groups"]
                group_file.write_bytes(group_file.read_bytes() + b"\n")
                with self.assertRaises(ValueError):
                    summarize_completed_run(root, "results/internal_evaluation_repair.json", expected_result_sha256=digest)


if __name__ == "__main__":
    unittest.main()
