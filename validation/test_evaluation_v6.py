"""Version-specific contracts retain the complete v5 numerical guardrails."""
import copy
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


class V6ContractTests(unittest.TestCase):
    def fixture(self):
        from validation.test_evaluation_utils import EvaluationUtilsTests
        from validation.evaluation_v6 import PROTOCOL_VERSION, TRAINING_FIELDS
        result, images, groups, folds = EvaluationUtilsTests().canonical_oof_bundle()
        result["protocol_version"] = PROTOCOL_VERSION
        result["configuration_fixed_before_outer_evaluation"].update(TRAINING_FIELDS)
        return result, images, groups, folds

    def test_v6_requires_explicit_loss_and_batchnorm_contract(self):
        from validation.evaluation_v6 import validate_result
        result, _, _, _ = self.fixture()
        validate_result(result)
        for change in ["loss_aggregation", "batchnorm_policy", "protocol_version"]:
            bad = copy.deepcopy(result)
            if change == "protocol_version":
                bad[change] = "2026-09-04-v5"
            else:
                bad["configuration_fixed_before_outer_evaluation"][change] = "incorrect"
            with self.subTest(change=change), self.assertRaises(ValueError):
                validate_result(bad)

    def test_v6_keeps_legacy_checks_for_recipe_and_outer_completeness(self):
        from validation.evaluation_v6 import validate_result
        result, _, _, _ = self.fixture()
        for alteration in ["partial", "epochs", "extra"]:
            bad = copy.deepcopy(result)
            if alteration == "partial": bad["completed_outer_folds"] = [0, 1]
            elif alteration == "epochs": bad["configuration_fixed_before_outer_evaluation"]["epochs"] = 2
            else: bad["unbound_claim"] = True
            with self.subTest(alteration=alteration), self.assertRaises(ValueError):
                validate_result(bad)

    def test_v6_terminal_verifies_its_own_audit_binding_and_every_oof_metric(self):
        from validation.evaluation_v6 import validate_terminal
        from validation.evaluation_utils import (group_fold_assignment_fingerprint,
            oof_identity_fingerprint, terminal_result_claim_sha256)
        result, images, groups, folds = self.fixture()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); (root / "results").mkdir()
            audit = {"protocol_version": result["protocol_version"], "status": "complete",
                "dataset_identity": result["dataset_identity"], "data_quality": result["data_quality"],
                "split_summary": [{"outer_fold": int(fold), "groups": len(rows), "images": int(rows.n_images.sum()),
                    "positive_groups": int(rows.label.sum()), "negative_groups": int((1-rows.label).sum())}
                    for fold, rows in folds.groupby("outer_fold")],
                "group_fold_manifest": result["artifacts"]["group_folds"],
                "completed_outer_folds": result["completed_outer_folds"],
                "result_artifact": "results/internal_evaluation_repair.json",
                "result_claim_sha256": terminal_result_claim_sha256(result)}
            for role, path in result["artifacts"].items():
                if role == "audit": (root / path).write_text(json.dumps(audit))
                else: {"group_folds": folds, "oof_images": images, "oof_groups": groups}[role].to_csv(root / path, index=False)
            result["artifact_sha256"] = {role: hashlib.sha256((root/path).read_bytes()).hexdigest() for role,path in result["artifacts"].items()}
            result["artifact_size_bytes"] = {role: (root/path).stat().st_size for role,path in result["artifacts"].items()}
            with patch("validation.evaluation_utils.EXPECTED_HYGD_DEDUPLICATED_OOF_IDENTITY_SHA256", oof_identity_fingerprint(images)), patch("validation.evaluation_utils.EXPECTED_HYGD_CANONICAL_GROUP_FOLD_SHA256", group_fold_assignment_fingerprint(folds)):
                validate_terminal(result, root)
                bad = copy.deepcopy(result)
                bad["configuration_fixed_before_outer_evaluation"]["loss_aggregation"] = "legacy_batch_mean"
                with self.assertRaises(ValueError): validate_terminal(bad, root)
                (root / result["artifacts"]["oof_groups"]).write_text("tampered\n")
                with self.assertRaises(ValueError): validate_terminal(result, root)


if __name__ == "__main__":
    unittest.main()
