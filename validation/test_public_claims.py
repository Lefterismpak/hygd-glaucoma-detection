"""Regression checks for the public scientific-claim and artifact boundary."""

from __future__ import annotations

import re
import json
import hashlib
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CLAIM_FILES = (
    ROOT / "README.md",
    ROOT / "VALIDATION.md",
    ROOT / "INTERNAL_EVALUATION_REPAIR.md",
    ROOT / "HYGD_FAILURE_FIRST_RESEARCH_BRIEF.md",
    ROOT / "validation" / "PROTOCOL.md",
    ROOT / "validation" / "FINDINGS.md",
    ROOT / "SOURCE_ONLY_QUALIFICATION_REPORT.md",
    ROOT / "HYGD_MANUAL_GEOMETRY_RESULT.md",
    ROOT / "HYGD_CEXT_2_0_PROTOCOL.md",
    ROOT / "HYGD_CEXT_2_0_EXECUTION_SPEC.md",
    ROOT / "HYGD_CEXT_2_0_RESULT.md",
    ROOT / "HYGD_SHORTCUT_MAP_1_RESULT.md",
)

ADAPTIVE_RESULT_FILES = (
    ROOT / "results" / "generalize_attemptA.json",
    ROOT / "results" / "generalize_attemptB.json",
    ROOT / "results" / "finetune_papila_to_rimone.json",
    ROOT / "results" / "finetune_rimone_to_papila.json",
    ROOT / "results" / "seed_robustness_attemptB.json",
    ROOT / "results" / "seed_robustness_reverse.json",
    *(ROOT / "results" / "seed_runs" / f"attemptB_seed{seed}.json" for seed in range(5)),
    *(ROOT / "results" / "seed_runs_reverse" / f"reverse_seed{seed}.json" for seed in range(5)),
)
EXTERNAL_STRESS_TEST_FILES = (
    ROOT / "results" / "external_papila.json",
    ROOT / "results" / "external_rimone.json",
)
HISTORICAL_INTERNAL_FILES = (
    ROOT / "results" / "metrics.json",
    ROOT / "results" / "history.json",
    ROOT / "results" / "error_analysis.json",
    ROOT / "results" / "cv_results.json",
    ROOT / "results" / "v2_comparison.json",
    ROOT / "results" / "threshold_sweep.json",
)
PUBLIC_INTERNAL_SUMMARY = ROOT / "results" / "repaired_internal_evaluation_summary.json"
REMOVED_PUBLIC_ASSETS = (
    ROOT / "figures" / "04_example_images.png",
    ROOT / "figures" / "05_roc_curve.png",
    ROOT / "figures" / "06_confusion_matrix.png",
    ROOT / "figures" / "07_loss_curves.png",
    ROOT / "figures" / "08_gradcam_correct.png",
    ROOT / "figures" / "09_gradcam_wrong.png",
    ROOT / "figures" / "10_error_vs_quality.png",
    ROOT / "figures" / "11_model_comparison.png",
    ROOT / "figures" / "12_roc_v2.png",
    ROOT / "figures" / "13_confusion_v2.png",
    ROOT / "figures" / "14_training_curves_v2.png",
    ROOT / "figures" / "15_cv_folds.png",
    ROOT / "figures" / "16_the_5_errors_annotated.png",
    ROOT / "figures" / "dg_recovery.png",
    ROOT / "validation" / "external_prediction_collapse.png",
    ROOT / "results" / "error_review_indices.json",
)


class PublicClaimBoundaryTests(unittest.TestCase):
    def test_ci_declares_pillow_and_pins_actions_immutably(self):
        requirements = (ROOT / "requirements-ci.txt").read_text().lower()
        packages = {
            re.split(r"[<>=!~]", line, maxsplit=1)[0].strip()
            for line in requirements.splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        }
        self.assertIn("pillow", packages)

        workflow = (ROOT / ".github" / "workflows" / "evaluation-integrity.yml").read_text()
        self.assertRegex(workflow, r"actions/checkout@[0-9a-f]{40}\s+# v[0-9.]+")
        self.assertRegex(workflow, r"actions/setup-python@[0-9a-f]{40}\s+# v[0-9.]+")
        self.assertIn("persist-credentials: false", workflow)

    def test_read_only_reanalysis_entrypoint_is_bound_to_the_public_receipt(self):
        from validation import reanalyze_frozen_oof

        script = ROOT / "validation" / "reanalyze_frozen_oof.py"
        completed = subprocess.run(
            [sys.executable, "-B", str(script), "--describe-contract"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        contract = json.loads(completed.stdout)
        self.assertFalse(contract["private_locations_in_scientific_contract"])
        self.assertFalse(contract["writes_files"])
        self.assertFalse(contract["trains_or_runs_inference"])
        self.assertEqual(contract["argv"][:3], [
            "python",
            "-B",
            "validation/reanalyze_frozen_oof.py",
        ])

        receipt = json.loads(PUBLIC_INTERNAL_SUMMARY.read_text())
        origin = receipt["execution_origin"]
        binding = receipt["private_source_binding"]
        self.assertEqual(origin["aggregate_reanalysis_argv"], contract["argv"])
        self.assertEqual(
            origin["aggregate_reanalysis_script_sha256"],
            hashlib.sha256(script.read_bytes()).hexdigest(),
        )
        self.assertEqual(
            binding["deduplicated_oof_identity_sha256"],
            "fbbaa74909dbc242843b708ca6073214e043e3e45a26856c827ce744024f849e",
        )
        self.assertEqual(
            binding["canonical_group_fold_assignment_sha256"],
            "c9d142c3fcfb213a6c1815468576694aec94db076e9f929c0d7018b0b2333432",
        )
        self.assertEqual(binding["oof_identity_rows"], 737)
        self.assertEqual(binding["outer_fold_assignment_groups"], 283)
        self.assertTrue(binding["observed_private_mapping_match"])
        self.assertEqual(
            binding["source_manifest_sha256"],
            reanalyze_frozen_oof.EXPECTED_SOURCE_MANIFEST_SHA256,
        )
        self.assertEqual(
            [row["role"] for row in binding["source_files"]],
            sorted(reanalyze_frozen_oof.SOURCE_SUFFIXES),
        )
        self.assertTrue(
            all(
                set(row) == {"role", "sha256", "size_bytes"}
                for row in binding["source_files"]
            )
        )
        argv = origin["aggregate_reanalysis_argv"]
        flag = "--expected-source-manifest-sha256"
        self.assertEqual(argv.count(flag), 1)
        self.assertEqual(
            argv[argv.index(flag) + 1],
            reanalyze_frozen_oof.EXPECTED_SOURCE_MANIFEST_SHA256,
        )

    def test_row_level_assets_and_stale_visual_claims_are_not_public(self):
        tracked = set(
            subprocess.check_output(["git", "ls-files"], cwd=ROOT, text=True).splitlines()
        )
        deleted = set(
            subprocess.check_output(
                ["git", "ls-files", "--deleted"], cwd=ROOT, text=True
            ).splitlines()
        )
        for path in REMOVED_PUBLIC_ASSETS:
            relative = str(path.relative_to(ROOT))
            self.assertNotIn(relative, tracked - deleted)
            ignored = subprocess.run(
                ["git", "check-ignore", "--no-index", "--quiet", relative],
                cwd=ROOT,
                check=False,
            )
            self.assertEqual(ignored.returncode, 0, relative)

        public_sources = (
            ROOT / "README.md",
            ROOT / "VALIDATION.md",
            ROOT / "validation" / "FINDINGS.md",
            *(ROOT / "notebooks" / f"0{index}_{name}.ipynb" for index, name in (
                (1, "eda"),
                (2, "preprocessing"),
                (3, "baseline_model"),
                (4, "explainability"),
                (5, "v2_experiments"),
            )),
        )
        combined = "\n".join(path.read_text() for path in public_sources)
        for path in REMOVED_PUBLIC_ASSETS:
            self.assertNotIn(path.name, combined)
        explainability = (ROOT / "notebooks" / "04_explainability.ipynb").read_text()
        geometry = (ROOT / "HYGD_MANUAL_GEOMETRY_RESULT.md").read_text()
        identifier = r"(?<![A-Za-z0-9])\d{1,4}_[0-9]{1,3}(?![A-Za-z0-9])"
        self.assertIsNone(re.search(identifier, explainability))
        self.assertIsNone(re.search(identifier, geometry))
        for prohibited in (
            "isn't just picking up on unrelated image artifacts",
            "quality gate",
            "would likely help",
            "fixable *upstream*",
        ):
            self.assertNotIn(prohibited, explainability)
        license_text = (ROOT / "LICENSE").read_text()
        self.assertIn("current Git tip/tree includes no raw dataset images", license_text)
        self.assertIn("Earlier Git history", license_text)
        self.assertIn("is not sanitized", license_text)

    def test_public_internal_summary_is_bound_aggregate_only_evidence(self):
        payload = json.loads(PUBLIC_INTERNAL_SUMMARY.read_text())
        self.assertEqual(
            set(payload),
            {
                "schema_version",
                "protocol_version",
                "evidence_status",
                "dataset_identity",
                "execution_origin",
                "primary",
                "secondary_descriptive",
                "private_source_binding",
                "privacy",
                "limitations",
            },
        )
        self.assertEqual(payload["schema_version"], "hygd-public-internal-summary-v1")
        self.assertEqual(payload["protocol_version"], "2026-09-04-v5")
        status = payload["evidence_status"]
        self.assertTrue(status["preferred_internal_estimate"])
        for key in (
            "confirmatory_evidence",
            "canonical_v5_end_to_end_run",
            "model_retrained_for_reanalysis",
            "transportability_established",
            "calibration_established",
            "clinical_utility_established",
            "deployment_ready",
        ):
            self.assertFalse(status[key])

        origin = payload["execution_origin"]
        self.assertEqual(origin["model_run_protocol_version"], "2026-07-11-v1")
        self.assertEqual(origin["legacy_run_start_audit_terminal_status"], "running")
        self.assertEqual(origin["public_time_attestation"], "not_available")
        self.assertEqual(
            origin["aggregate_reanalysis_protocol_version"], "2026-09-04-v5"
        )
        self.assertFalse(origin["model_retrained_for_publication"])
        self.assertFalse(origin["model_training_or_inference_executed_for_reanalysis"])
        self.assertFalse(origin["current_v5_code_used_for_training"])
        self.assertIn("not an end-to-end v5 model rerun", origin["scope"])
        self.assertRegex(
            origin["aggregate_reanalysis_implementation_sha256"], r"^[0-9a-f]{64}$"
        )
        self.assertEqual(
            origin["aggregate_reanalysis_implementation_sha256"],
            hashlib.sha256(
                (ROOT / "validation" / "evaluation_utils.py").read_bytes()
            ).hexdigest(),
        )

        primary = payload["primary"]
        fold_aucs = [row["auroc"] for row in primary["fold_aurocs"]]
        self.assertAlmostEqual(primary["auroc"], sum(fold_aucs) / len(fold_aucs), 15)
        self.assertEqual(sum(row["n_groups"] for row in primary["fold_aurocs"]), 283)
        self.assertEqual(sum(primary["confusion_tn_fp_fn_tp"]), primary["n"])
        self.assertEqual(primary["bootstrap"]["n_bootstrap"], 5000)
        self.assertEqual(primary["bootstrap"]["valid_replicates"], 5000)
        self.assertEqual(primary["bootstrap"]["seed"], 20269714)

        binding = payload["private_source_binding"]
        source_files = binding["source_files"]
        self.assertEqual([row["role"] for row in source_files], sorted(
            row["role"] for row in source_files
        ))
        self.assertTrue(all(re.fullmatch(r"[0-9a-f]{64}", row["sha256"])
                            for row in source_files))
        self.assertTrue(all(type(row["size_bytes"]) is int and row["size_bytes"] > 0
                            for row in source_files))
        manifest_payload = json.dumps(
            source_files,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("ascii")
        manifest = hashlib.sha256(
            b"HYGD_PRIVATE_SOURCE_MANIFEST_V1\n" + manifest_payload + b"\n"
        ).hexdigest()
        self.assertEqual(manifest, binding["source_manifest_sha256"])
        self.assertFalse(binding["row_level_artifacts_published"])
        self.assertTrue(all(value is False for value in payload["privacy"].values()))

        forbidden_keys = {
            "image_name",
            "image_path",
            "patient_id",
            "probability",
            "selected_threshold",
            "artifact_path",
        }
        discovered_keys = set()

        def walk(value):
            if isinstance(value, dict):
                for key, nested in value.items():
                    discovered_keys.add(key)
                    walk(nested)
            elif isinstance(value, list):
                for nested in value:
                    walk(nested)
            elif isinstance(value, str):
                self.assertNotIn("/Users/", value)

        walk(payload)
        self.assertFalse(discovered_keys & forbidden_keys)
        for relative in ("README.md", "VALIDATION.md", "INTERNAL_EVALUATION_REPAIR.md"):
            self.assertIn(
                "results/repaired_internal_evaluation_summary.json",
                (ROOT / relative).read_text(),
            )

    def test_historical_runner_exposes_a_torch_free_noncanonical_policy(self):
        completed = subprocess.run(
            [sys.executable, "-S", "run_v2_experiments.py", "--describe-policy"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        policy = json.loads(completed.stdout)
        self.assertEqual(policy["status"], "historical_development_only")
        self.assertFalse(policy["canonical_internal_estimate"])
        self.assertFalse(policy["generates_cross_validation"])
        self.assertFalse(policy["configuration_selection_permitted"])
        self.assertEqual(
            policy["primary_command"],
            "python validation/internal_evaluation_repair.py",
        )

    def test_historical_runner_requires_explicit_acknowledgement(self):
        completed = subprocess.run(
            [sys.executable, "-S", "run_v2_experiments.py"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(completed.returncode, 2)
        self.assertIn("--run-historical-comparison", completed.stderr)

    def test_historical_runner_is_repo_root_bound_from_a_foreign_cwd(self):
        script = ROOT / "run_v2_experiments.py"
        with tempfile.TemporaryDirectory() as temporary:
            completed = subprocess.run(
                [sys.executable, "-S", str(script), "--describe-policy"],
                cwd=temporary,
                capture_output=True,
                text=True,
                check=False,
            )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertEqual(
            json.loads(completed.stdout)["status"],
            "historical_development_only",
        )

        source = script.read_text()
        self.assertIn("ROOT = Path(__file__).resolve().parent", source)
        self.assertIn('load_dataset_metadata(ROOT / "data" / "raw")', source)
        self.assertIn("resolve_private_output_path", source)
        self.assertIn("assert_fresh_output_bundle", source)
        self.assertIn("atomic_torch_save", source)
        self.assertIn("atomic_write_text", source)
        self.assertIn("results/historical_notebook_runs", source)
        self.assertIn(
            "historical_test_derived_threshold_sens>=0.95_nondeployment",
            source,
        )
        self.assertNotIn('RESULTS_DIR / f"{name}.pt"', source)
        self.assertNotIn('results["screening_threshold_sens>=0.95"]', source)
        self.assertNotIn('sys.path.insert(0, ".")', source)
        self.assertNotIn('open("results/', source)

    def test_every_historical_internal_cv_number_has_a_local_warning(self):
        warning = re.compile(r"historical|superseded|non-canonical|invalid", re.I)
        failures = []
        for path in CLAIM_FILES:
            text = path.read_text()
            for match in re.finditer(r"0\.988", text):
                nearby = text[max(0, match.start() - 240) : match.end() + 240]
                if not warning.search(nearby):
                    failures.append(f"{path.relative_to(ROOT)}:{text[:match.start()].count(chr(10)) + 1}")
        self.assertEqual(
            failures,
            [],
            "0.988 appears without a nearby historical/superseded warning: "
            + ", ".join(failures),
        )

    def test_primary_docs_name_the_repaired_command_and_unproved_boundaries(self):
        for relative in ("README.md", "VALIDATION.md"):
            text = (ROOT / relative).read_text().lower()
            self.assertIn("validation/internal_evaluation_repair.py", text)
            self.assertIn("mean outer-fold", text)
            self.assertIn("0.9908", text)
            self.assertIn("pooled", text)
            self.assertIn("transportability", text)
            self.assertIn("calibration", text)
            self.assertIn("clinical utility", text)
            self.assertIn("deployment", text)

    def test_readme_separates_historical_and_preferred_model_recipes(self):
        readme = (ROOT / "README.md").read_text().lower()
        methods = readme.split("## 4. methods", 1)[1].split("## 5. results", 1)[0]
        self.assertIn("historical frozen-head baseline recipe", methods)
        self.assertIn("preferred evaluator's fixed recipe", methods)
        self.assertIn("fine-tune resnet18 `layer4`", methods)
        self.assertIn("10 epochs", methods)

    def test_public_docs_scope_primary_and_pooled_uncertainty_separately(self):
        for relative in (
            "README.md",
            "VALIDATION.md",
            "INTERNAL_EVALUATION_REPAIR.md",
        ):
            text = (ROOT / relative).read_text().lower()
            self.assertIn("global group-cluster", text)
            self.assertIn("separately fitted fold models", text)
            self.assertIn("20269714", text)

    def test_adaptive_external_chronology_is_not_promoted_as_target_blind(self):
        source_paths = (
            ROOT / "validation" / "FINDINGS.md",
            ROOT / "validation" / "train_generalize.py",
            ROOT / "validation" / "train_generalize_vcdr.py",
            ROOT / "validation" / "train_generalize_vcdr_reverse.py",
            ROOT / "validation" / "aggregate_seeds.py",
            ROOT / "validation" / "aggregate_seeds_reverse.py",
        )
        combined = "\n".join(path.read_text().lower() for path in source_paths)
        prohibited = (
            "recovers genuine transfer",
            "each lever is principled and target-free",
            "papila remains fully held out; swa + tta are target-free",
            "rim-one never used for training/selection",
            '"tuned to papila" to "generalizes across datasets"',
        )
        for phrase in prohibited:
            self.assertNotIn(phrase, combined)
        self.assertIn("target auroc was displayed during development", combined)
        self.assertIn("adaptive", combined)

        for path in ADAPTIVE_RESULT_FILES:
            payload = json.loads(path.read_text())
            status = payload["_evidence_status"]
            self.assertEqual(status["status"], "adaptive_development_evidence")
            self.assertFalse(status["target_blind"])
            self.assertFalse(status["transportability_established"])
            self.assertEqual(status["license_compatibility"], "needs-proof")
            self.assertFalse(status["external_claim_permitted"])

    def test_external_stress_tests_expose_units_licenses_and_protocol_deviations(self):
        papila = json.loads(EXTERNAL_STRESS_TEST_FILES[0].read_text())
        rimone = json.loads(EXTERNAL_STRESS_TEST_FILES[1].read_text())

        for payload in (papila, rimone):
            status = payload["_evidence_status"]
            self.assertEqual(status["status"], "historical_locked_zero_shot_stress_test")
            self.assertFalse(status["transportability_established"])
            self.assertFalse(status["full_protocol_compliance"])
            self.assertEqual(status["protocol_execution_status"], "partial_with_deviations")
            self.assertFalse(payload["_uncertainty_provenance"]["patient_cluster_interval"])
            self.assertIn("row_bootstrap_noncanonical", payload["_uncertainty_provenance"]["status"])
            self.assertFalse(payload["_evidence_status"]["external_claim_permitted"])

        self.assertEqual(papila["evaluation_unit"], "eye")
        self.assertEqual(papila["_evidence_status"]["license"], "GPL-3.0+")
        self.assertEqual(papila["_evidence_status"]["redistribution_of_derived_material"], "needs-proof")
        self.assertNotIn("screening_threshold", papila["zero_shot"])

        self.assertEqual(rimone["evaluation_unit"], "image")
        self.assertEqual(rimone["subject_independence"], "needs-proof")
        self.assertNotIn("n_patients", rimone)
        self.assertEqual(
            rimone["_evidence_status"]["mixed_source_license_compatibility"],
            "needs-proof",
        )
        self.assertFalse(rimone["_evidence_status"]["external_claim_permitted"])

    def test_adaptive_uncertainty_is_locally_adjudicated(self):
        interval_files = (
            ROOT / "results" / "generalize_attemptA.json",
            ROOT / "results" / "generalize_attemptB.json",
            *(ROOT / "results" / "seed_runs" / f"attemptB_seed{seed}.json" for seed in range(5)),
            *(ROOT / "results" / "seed_runs_reverse" / f"reverse_seed{seed}.json" for seed in range(5)),
        )
        for path in interval_files:
            provenance = json.loads(path.read_text())["_uncertainty_provenance"]
            self.assertFalse(provenance["patient_cluster_interval"])
            self.assertIn("row_bootstrap_noncanonical", provenance["status"])

        for name in ("seed_robustness_attemptB.json", "seed_robustness_reverse.json"):
            provenance = json.loads((ROOT / "results" / name).read_text())["_uncertainty_provenance"]
            self.assertEqual(provenance["uncertainty_measure"], "sample_standard_deviation")
            self.assertFalse(provenance["confidence_interval"])
            self.assertEqual(provenance["seed_only_run_family_identity"], "needs-proof")

    def test_historical_origin_artifacts_are_self_adjudicating(self):
        cv = json.loads((ROOT / "results" / "cv_results.json").read_text())
        comparison = json.loads((ROOT / "results" / "v2_comparison.json").read_text())
        threshold = json.loads((ROOT / "results" / "threshold_sweep.json").read_text())
        for payload in (cv, comparison, threshold):
            self.assertFalse(payload["_provenance"]["canonical_internal_estimate"])

        for path in HISTORICAL_INTERNAL_FILES:
            payload = json.loads(path.read_text())
            self.assertIn("_provenance", payload, path.name)
            self.assertFalse(payload["_provenance"]["canonical_internal_estimate"])

        self.assertEqual(cv["_provenance"]["evaluation_unit"], "image_row")
        self.assertFalse(cv["_provenance"]["inner_outer_separation"])
        self.assertFalse(threshold["_provenance"]["deployment_recommendation"])
        self.assertFalse(threshold["_provenance"]["patient_cluster_uncertainty"])
        for model_name in ("frozen_aug", "finetune_layer4_aug"):
            self.assertIn(
                "historical_test_derived_threshold_sens>=0.95_nondeployment",
                comparison[model_name],
            )
            self.assertNotIn("screening_threshold_sens>=0.95", comparison[model_name])

        log_head = (ROOT / "results" / "v2_run.log").read_text().splitlines()[:5]
        self.assertTrue(any("STATUS CORRECTION" in line for line in log_head))

        notebook = (ROOT / "notebooks" / "05_v2_experiments.ipynb").read_text().lower()
        self.assertIn("status correction", notebook)
        self.assertIn("superseded", notebook)
        self.assertNotIn("0.40 (recommended)", notebook)
        self.assertNotIn("**recommended: 0.40**", notebook)

    def test_public_docs_remove_stale_or_overstated_publication_claims(self):
        combined = "\n".join(path.read_text().lower() for path in CLAIM_FILES)
        prohibited = (
            "not yet published",
            "pipeline is correct",
            "fixes the calibration dramatically",
            "confirm the pattern generalizes",
            "localizes the optic disc on every image",
            "destroying the field-of-view",
            "beyond citation",
        )
        for phrase in prohibited:
            self.assertNotIn(phrase, combined)

        readme = (ROOT / "README.md").read_text()
        validation = (ROOT / "VALIDATION.md").read_text()
        findings = (ROOT / "validation" / "FINDINGS.md").read_text()
        protocol = (ROOT / "validation" / "PROTOCOL.md").read_text()
        self.assertNotIn("![Model comparison](figures/11_model_comparison.png)", readme)
        self.assertNotIn("![Historical CV folds", readme)
        self.assertNotIn("![recovery]", validation)
        self.assertNotIn("![recovery]", findings)
        self.assertNotIn("CC BY 4.0", protocol)
        self.assertIn("GPL 3.0+", protocol)
        self.assertIn("mixed-source", protocol)
        self.assertIn("needs-proof", protocol)

    def test_executable_checkpoint_and_output_boundaries_are_hardened(self):
        executable_paths = (
            ROOT / "run_v2_experiments.py",
            ROOT / "analyze_errors.py",
            ROOT / "validation" / "predict.py",
            ROOT / "validation" / "eval_external.py",
            ROOT / "validation" / "finetune_crossdataset.py",
            ROOT / "validation" / "train_generalize.py",
            ROOT / "validation" / "train_generalize_vcdr.py",
            ROOT / "validation" / "train_generalize_vcdr_reverse.py",
            ROOT / "validation" / "aggregate_seeds.py",
            ROOT / "validation" / "aggregate_seeds_reverse.py",
        )
        for path in executable_paths:
            source = path.read_text()
            self.assertNotIn("torch.load(", source, path.name)
            if any(
                token in source
                for token in ("json.dumps(res", "json.dumps(results", "json.dumps(summary")
            ):
                self.assertIn("atomic_write_text", source, path.name)
                self.assertIn("resolve_private_output_path", source, path.name)

        utilities = (ROOT / "validation" / "evaluation_utils.py").read_text()
        self.assertIn("weights_only=True", utilities)
        self.assertIn("io.BytesIO(content)", utilities)
        self.assertIn("atomic_write_bytes(path, content", utilities)

    def test_seed_wrappers_require_bound_subject_maps_and_private_outputs(self):
        for name in ("run_seeds.sh", "run_seeds_reverse.sh"):
            source = (ROOT / "validation" / name).read_text()
            self.assertIn("set -euo pipefail", source)
            self.assertIn("HYGD_RIMONE_SUBJECT_MAP", source)
            self.assertIn("HYGD_RIMONE_SUBJECT_MAP_SHA256", source)
            self.assertIn("--rimone-subject-map-sha256", source)
            self.assertIn("--acknowledge-", source)
            self.assertIn("results/patient_aware/", source)
            self.assertNotIn('results/seed_runs/', source)
            self.assertIn('if [ "$#" -eq 0 ]', source)
            self.assertIn('0|1|2|3|4)', source)
            self.assertIn("duplicate seed", source)
            self.assertNotIn('> "logs/', source)

    def test_historical_notebooks_share_one_safe_local_checkpoint_chain(self):
        baseline = json.loads((ROOT / "notebooks" / "03_baseline_model.ipynb").read_text())
        explain = json.loads((ROOT / "notebooks" / "04_explainability.ipynb").read_text())
        baseline_source = "\n".join(
            "".join(cell.get("source", [])) for cell in baseline["cells"]
        )
        explain_source = "\n".join(
            "".join(cell.get("source", [])) for cell in explain["cells"]
        )
        shared = "../results/historical_notebook_runs/baseline_resnet18.pt"
        self.assertIn(shared, baseline_source)
        self.assertIn(shared, explain_source)
        self.assertIn("atomic_torch_save", baseline_source)
        self.assertIn("atomic_write_text", baseline_source)
        self.assertIn("load_torch_state_dict_safely", explain_source)
        self.assertNotIn("torch.save(", baseline_source)
        self.assertNotIn("torch.load(", explain_source)
        self.assertIn("sum(len(rows) for rows in sample.values())", explain_source)

    def test_protocol_status_and_private_hash_boundaries_are_unambiguous(self):
        for name in ("external_papila.json", "external_rimone.json"):
            payload = json.loads((ROOT / "results" / name).read_text())
            self.assertEqual(
                payload["_evidence_status"]["protocol_execution_status"],
                "partial_with_deviations",
            )
        combined = (
            (ROOT / "validation" / "FINDINGS.md").read_text()
            + (ROOT / "validation" / "PROTOCOL.md").read_text()
        )
        self.assertNotIn("partial_protocol_execution", combined)
        self.assertIn("partial_with_deviations", combined)

        locked_hash = "edea2a4c991c8b2e3c44ae23a3e26b601b82fa33ad889f7b8cd96b0865b4a721"
        cext_protocol_path = ROOT / "HYGD_CEXT_2_0_PROTOCOL.md"
        cext_protocol = cext_protocol_path.read_text()
        cext_result = (ROOT / "HYGD_CEXT_2_0_RESULT.md").read_text()
        self.assertIn("annotated public copy", cext_protocol)
        self.assertIn("not the lock artifact", cext_protocol)
        self.assertIn(locked_hash, cext_protocol)
        self.assertIn("Private locked-protocol SHA-256", cext_result)
        self.assertIn(locked_hash, cext_result)
        self.assertNotEqual(
            hashlib.sha256(cext_protocol_path.read_bytes()).hexdigest(), locked_hash
        )

        geometry = (ROOT / "HYGD_MANUAL_GEOMETRY_RESULT.md").read_text()
        self.assertIn('"role": <table path>', geometry)
        self.assertIn('separators=(",", ":")', geometry)
        self.assertIn(
            "8022b0ce2723b0f842bf8a68e40ab52d7dc9664015c794209cf585ec8a7d765b",
            geometry,
        )
        table_match = re.search(
            r"\| Private source role \| SHA-256 \| Bytes \|(?P<body>.*?)(?:\n\n|\Z)",
            geometry,
            flags=re.S,
        )
        self.assertIsNotNone(table_match)
        records = []
        for role, digest, size in re.findall(
            r"^\| `([^`]+)` \| `([0-9a-f]{64})` \| ([0-9,]+) \|$",
            table_match.group("body"),
            flags=re.M,
        ):
            records.append(
                {"role": role, "sha256": digest, "size_bytes": int(size.replace(",", ""))}
            )
        self.assertEqual(len(records), 9)
        canonical = json.dumps(
            sorted(records, key=lambda row: row["role"].encode("utf-8")),
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("ascii")
        recomputed = hashlib.sha256(
            b"HYGD_MANUAL_GEOMETRY_PRIVATE_SOURCES_V1\n" + canonical + b"\n"
        ).hexdigest()
        self.assertEqual(
            recomputed,
            "8022b0ce2723b0f842bf8a68e40ab52d7dc9664015c794209cf585ec8a7d765b",
        )

    def test_notebooks_publish_no_execution_outputs_or_embedded_media(self):
        for path in sorted((ROOT / "notebooks").glob("*.ipynb")):
            notebook = json.loads(path.read_text())
            self.assertNotIn("attachments", notebook, path.name)
            for cell in notebook.get("cells", []):
                self.assertFalse(cell.get("attachments"), path.name)
                if cell.get("cell_type") == "code":
                    self.assertEqual(cell.get("outputs", []), [], path.name)
                    self.assertIsNone(cell.get("execution_count"), path.name)
            serialized = path.read_text()
            self.assertNotRegex(serialized, r"data:(?:image|audio|video)/")
            self.assertNotIn("/Users/", serialized)

    def test_public_unit_language_does_not_overclaim_identity_or_reference_labels(self):
        combined = "\n".join(
            (ROOT / relative).read_text().lower()
            for relative in ("README.md", "VALIDATION.md")
        )
        for phrase in ("288 patients", "44 patients", "44-patient", "gold-standard"):
            self.assertNotIn(phrase, combined)
        self.assertIn("288 supplied patient ids", combined)
        self.assertIn("dataset-author reference labels", combined)
        source = (ROOT / "src" / "data_utils.py").read_text().lower()
        self.assertNotIn("288 patients", source)
        self.assertIn("288 supplied patient ids", source)
        metrics = json.loads((ROOT / "results" / "metrics.json").read_text())
        self.assertEqual(metrics["n_test_supplied_patient_ids"], 44)
        self.assertIn("supplied_patient_id_count", metrics["n_test_patients_legacy_key_definition"])
        for notebook_name in ("01_eda.ipynb", "02_preprocessing.ipynb"):
            notebook = (ROOT / "notebooks" / notebook_name).read_text().lower()
            self.assertNotIn("288 patients", notebook)
            self.assertNotIn("} patients,", notebook)
            self.assertIn("supplied patient id", notebook)

        reviewed_figure_digests = {
            "01_class_distribution.png": "3f0dc287a141f91e1675e51c9c5e2311ea3c75113cf0f66e9a7809de64e70e60",
            "03_images_per_patient.png": "40707e6472035aa88d4739799c8784c5693ea75dbacd12463ade7a94224238aa",
        }
        for name, expected in reviewed_figure_digests.items():
            digest = hashlib.sha256((ROOT / "figures" / name).read_bytes()).hexdigest()
            self.assertEqual(digest, expected, name)

    def test_vcdr_auxiliary_label_wording_does_not_claim_unproved_reliability(self):
        paths = [
            ROOT / "validation" / "train_generalize_vcdr.py",
            ROOT / "validation" / "train_generalize_vcdr_reverse.py",
            ROOT / "results" / "generalize_attemptB.json",
            *sorted((ROOT / "results" / "seed_runs").glob("attemptB_seed*.json")),
            *sorted((ROOT / "results" / "seed_runs_reverse").glob("reverse_seed*.json")),
        ]
        for path in paths:
            text = path.read_text().lower()
            self.assertNotIn("only (reliable)", text, path)
            self.assertNotIn("expert-derived and reliable", text, path)
            self.assertIn("historical auxiliary-label rule", text, path)

    def test_private_evidence_and_mixed_source_license_boundaries_are_local(self):
        for relative in (
            "SOURCE_ONLY_QUALIFICATION_REPORT.md",
            "HYGD_MANUAL_GEOMETRY_RESULT.md",
            "HYGD_CEXT_2_0_RESULT.md",
            "HYGD_SHORTCUT_MAP_1_RESULT.md",
        ):
            text = (ROOT / relative).read_text().lower()
            self.assertIn("private", text, relative)
            self.assertIn("unpublished", text, relative)
            self.assertIn("needs-proof", text, relative)
        for relative in (
            "SOURCE_ONLY_QUALIFICATION_REPORT.md",
            "HYGD_CEXT_2_0_PROTOCOL.md",
            "HYGD_CEXT_2_0_RESULT.md",
        ):
            text = (ROOT / relative).read_text().lower()
            self.assertIn("rim-one mixed-source use compatibility", text, relative)
            self.assertIn("no external or publication performance claim", text, relative)

    def test_declared_groups_are_not_promoted_to_verified_biological_subjects(self):
        paths = (
            "SOURCE_ONLY_QUALIFICATION_REPORT.md",
            "HYGD_CEXT_2_0_RESULT.md",
            "HYGD_CEXT_2_0_EXECUTION_SPEC.md",
            "HYGD_SHORTCUT_MAP_1_PROTOCOL.md",
            "HYGD_MANUAL_GEOMETRY_RESULT.md",
        )
        forbidden = (
            "subject-cluster overlap",
            "cluster leakage",
            "subject/hash leakage",
            "frozen subject clusters",
            "subject-disjoint",
        )
        for relative in paths:
            text = (ROOT / relative).read_text().lower()
            for phrase in forbidden:
                self.assertNotIn(phrase, text, relative)
            self.assertIn("evaluation-group", text, relative)
        combined = "\n".join((ROOT / relative).read_text().lower() for relative in paths)
        self.assertIn("biological-subject independence remains `needs-proof`", combined)
        self.assertIn("not independent proof of biological-subject identity", combined)

    def test_historical_reanalysis_does_not_claim_oof_files_were_recomputed(self):
        repair = (ROOT / "INTERNAL_EVALUATION_REPAIR.md").read_text()
        self.assertNotIn("OOF files were independently recomputed", repair)
        self.assertIn(
            "were independently recomputed from the complete OOF files",
            repair,
        )

    def test_generated_repair_artifacts_are_git_ignored(self):
        generated = (
            "results/internal_evaluation_repair.json",
            "results/internal_evaluation_repair_audit.json",
            "results/internal_evaluation_repair_group_folds.csv",
            "results/internal_evaluation_repair_oof_images.csv",
            "results/internal_evaluation_repair_oof_groups.csv",
            "results/internal_evaluation_repair_audit_only_audit.json",
            "results/internal_evaluation_repair_smoke_1of5.json",
            "results/historical_notebook_runs/v2_comparison_run.json",
            "results/patient_aware/external_papila_v2.json",
            "figures/local/error_vs_quality.png",
            "data/hygd_manual_disc_centers.csv",
            "data/hygd_boundary_annotations_v1.csv",
            "validation/HYGD_BOUNDARY_ADJUDICATION_V1.md",
            "validation/confirmatory_protocol_lock_v1.1.json",
            "validation/hygd_cext_2_0_protocol_lock.json",
            "validation/hygd_cext_2_0_execution_lock.json",
            "validation/cext_2_0.py",
            "validation/audit_cext_2_0.py",
            "validation/make_hygd_boundary_adjudication_overlays.py",
            "validation/evaluate_hygd_scale_calibration_feasibility.py",
            "artifacts/dinov2_vits14_receipt/receipt.json",
            "results/confirmatory/source_audit.json",
            "results/confirmatory/candidate_A/qualification_summary.json",
            "results/confirmatory/candidate_B/geometry_summary.json",
            "results/confirmatory/candidate_B/fold_0/geometry_qc.json",
            "results/cext_2_0/feature_extraction_primary.json",
            "results/cext_2_0/evaluation_primary/qualification_summary.json",
            "results/cext_2_0/independent_audit.json",
            "results/shortcut_map_1/independent_audit.json",
            "results/shortcut_map_1/pre_metric_runtime_smoke_s1/receipt.json",
            "results/hygd_boundary_adjudication_v1.json",
            "results/hygd_boundary_adjudication_v1_details.csv",
            "results/hygd_boundary_adjudication_v1_overlay.png",
            "results/hygd_scale_calibration_feasibility.json",
            "results/source_only_localizer_forward_manual_source_prep.json",
        )
        failures = []
        for relative in generated:
            completed = subprocess.run(
                ["git", "check-ignore", "--no-index", "--quiet", relative],
                cwd=ROOT,
                check=False,
            )
            if completed.returncode != 0:
                failures.append(relative)
        self.assertEqual(
            failures,
            [],
            "Generated repair artifacts are not ignored: " + ", ".join(failures),
        )


if __name__ == "__main__":
    unittest.main()
