"""Dataset-free regression tests for the repaired HYGD evaluation contract."""

from __future__ import annotations

import subprocess
import sys
import tempfile
import unittest
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal
from sklearn.metrics import roc_auc_score

from validation.evaluation_utils import (
    EXPECTED_HYGD_CONTRACT_SHA256,
    aggregate_groups,
    build_primary_report,
    cluster_bootstrap,
    fold_stratified_group_bootstrap,
    finalize_audit,
    finalize_historical_comparison,
    historical_image_bootstrap_ci,
    inner_group_split,
    make_outer_folds,
    outer_fold_auc_summary,
    prepare_duplicate_aware_metadata,
    resample_clusters,
    resolve_repair_output_prefix,
    select_threshold,
    terminal_result_claim_sha256,
    validate_oof_predictions,
    validate_internal_result_payload,
    validate_primary_report,
)


ROOT = Path(__file__).resolve().parents[1]
TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None
TORCHVISION_AVAILABLE = importlib.util.find_spec("torchvision") is not None
EXPECTED_INVENTORY_SHA256 = (
    "1187ea38d89ee414442113869d5c6d8c575530e935b8169c8c037b3f4c53a291"
)
EXPECTED_LABELS_SHA256 = (
    "c2f4e6e756f6f5f9d9d900f613cf329116da9897877e480cc9782bc042d4c0f5"
)


class EvaluationUtilsTests(unittest.TestCase):
    def make_metadata(self, rows):
        """Create literal local files and a normalized metadata frame.

        Rows are ``(filename, patient_id, label, bytes)``. The bytes need not be
        valid images because this layer only verifies paths and hashes content.
        """

        temporary = tempfile.TemporaryDirectory()
        root = Path(temporary.name)
        records = []
        for filename, patient_id, label, content in rows:
            path = root / filename
            path.write_bytes(content)
            records.append(
                {
                    "image_path": str(path),
                    "patient_id": patient_id,
                    "label": label,
                    "quality_score": 6.0,
                }
            )
        return temporary, pd.DataFrame(records)

    def canonical_result(
        self,
        report,
        *,
        fold_image_counts=None,
        fold_group_counts=None,
        fold_image_positives=None,
        fold_group_positives=None,
    ):
        report = copy.deepcopy(report)
        fold_image_counts = fold_image_counts or [148, 148, 147, 147, 147]
        fold_group_counts = fold_group_counts or [57, 57, 57, 56, 56]
        fold_image_positives = fold_image_positives or [
            count // 2 for count in fold_image_counts
        ]
        fold_group_positives = fold_group_positives or [
            count // 2 for count in fold_group_counts
        ]
        canonical_auc = float(report["primary"]["auroc"])
        report["primary"]["fold_aurocs"] = [
            {
                "outer_fold": fold,
                "n_groups": fold_group_counts[fold],
                "positives": fold_group_positives[fold],
                "negatives": fold_group_counts[fold] - fold_group_positives[fold],
                "auroc": canonical_auc,
            }
            for fold in range(5)
        ]
        report["primary"]["group_count_weighted_mean_auroc"] = canonical_auc
        group_positives = sum(fold_group_positives)
        group_negatives = sum(fold_group_counts) - group_positives
        image_positives = sum(fold_image_positives)
        image_negatives = sum(fold_image_counts) - image_positives
        group_classification = {
            "n": sum(fold_group_counts),
            "positives": group_positives,
            "negatives": group_negatives,
            "sensitivity": 1.0,
            "specificity": 1.0,
            "confusion_tn_fp_fn_tp": [
                group_negatives,
                0,
                0,
                group_positives,
            ],
        }
        image_classification = {
            "n": sum(fold_image_counts),
            "positives": image_positives,
            "negatives": image_negatives,
            "sensitivity": 1.0,
            "specificity": 1.0,
            "confusion_tn_fp_fn_tp": [
                image_negatives,
                0,
                0,
                image_positives,
            ],
        }
        report["primary"].update(group_classification)
        report["pooled_oof_group_descriptive"].update(group_classification)
        report["secondary_image_level"].update(image_classification)
        configuration = {
            "model": "ImageNet ResNet18, layer4 + head fine-tuned",
            "weights": "ResNet18_Weights.IMAGENET1K_V1",
            "trainable_parameters": "layer4 + fc",
            "optimizer": "Adam",
            "head_learning_rate": 1e-3,
            "backbone_learning_rate": 1e-4,
            "train_transform": (
                "Resize(224x224), RandomHorizontalFlip(p=0.5), "
                "RandomRotation(15deg), ColorJitter(brightness=0.1,contrast=0.1), "
                "ImageNet normalization"
            ),
            "evaluation_transform": "Resize(224x224), ImageNet normalization",
            "configuration_origin": (
                "Chosen during prior HYGD development, then frozen before this repair run"
            ),
            "residual_post_selection_risk": (
                "Outer folds were hidden from this run, but the architecture and recipe "
                "were historically informed by HYGD development/test results"
            ),
            "outer_folds": 5,
            "epochs": 10,
            "batch_size": 32,
            "seed": 20_260_711,
            "inner_validation_fraction": 0.15,
            "threshold_target_sensitivity": 0.95,
            "bootstrap_samples": 5_000,
            "device": "cpu",
            "checkpoint_selection": "minimum inner-validation loss",
            "threshold_selection": "inner-validation group-level predictions only",
            "outer_test_visibility_during_training": (
                "withheld_until_terminal_bundle_publication"
            ),
        }
        identity = {
            "schema_version": "hygd-dataset-identity-v1",
            "dataset_id": "physionet-hygd",
            "dataset_name": "Hillel Yaffe Glaucoma Dataset",
            "dataset_version": "1.1.0",
            "doi": "10.13026/m92s-0z95",
            "labels_csv_sha256": EXPECTED_LABELS_SHA256,
            "row_manifest_algorithm": "hygd-row-manifest-v1",
            "ordered_image_sha256_patient_id_label_inventory_sha256": (
                EXPECTED_INVENTORY_SHA256
            ),
            "inventory_rows": 747,
            "unique_image_hashes": 737,
            "patient_ids": 288,
            "negative_rows": 199,
            "positive_rows": 548,
            "contract_sha256": EXPECTED_HYGD_CONTRACT_SHA256,
            "canonical_hygd_v1_1_0_match": True,
        }
        data_quality = {
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
            "ordered_image_sha256_patient_id_label_inventory_sha256": (
                EXPECTED_INVENTORY_SHA256
            ),
            "duplicate_groups": [{"synthetic": index} for index in range(10)],
        }
        fold_results = []
        for fold in range(5):
            image_n = fold_image_counts[fold]
            group_n = fold_group_counts[fold]
            image_positive = fold_image_positives[fold]
            group_positive = fold_group_positives[fold]
            fold_results.append({
                "outer_fold": fold,
                "seed": 20_260_711 + fold * 101,
                "best_epoch_selected_on_inner_validation_loss": 1,
                "selected_threshold_from_inner_group_validation": {
                    "threshold": 0.5,
                    "inner_validation_sensitivity": 0.95,
                    "inner_validation_specificity": 0.9,
                },
                "counts": {
                    "inner_train_images": 737 - image_n - 90,
                    "inner_train_groups": 190,
                    "inner_validation_images": 90,
                    "inner_validation_groups": 283 - group_n - 190,
                    "outer_test_images": image_n,
                    "outer_test_groups": group_n,
                },
                "outer_test_image_level": {
                    "n": image_n,
                    "positives": image_positive,
                    "negatives": image_n - image_positive,
                    "auroc": canonical_auc,
                    "sensitivity": 1.0,
                    "specificity": 1.0,
                    "confusion_tn_fp_fn_tp": [
                        image_n - image_positive,
                        0,
                        0,
                        image_positive,
                    ],
                },
                "outer_test_group_level": {
                    "n": group_n,
                    "positives": group_positive,
                    "negatives": group_n - group_positive,
                    "auroc": canonical_auc,
                    "sensitivity": 1.0,
                    "specificity": 1.0,
                    "confusion_tn_fp_fn_tp": [
                        group_n - group_positive,
                        0,
                        0,
                        group_positive,
                    ],
                },
                "training_history": [
                    {
                        "epoch": epoch,
                        "train_loss": 0.5,
                        "inner_validation_loss": 0.4,
                    }
                    for epoch in range(1, 11)
                ],
            })
        return {
            "protocol_version": "2026-09-04-v5",
            "status": "complete",
            "completed_outer_folds": list(range(5)),
            "configuration_fixed_before_outer_evaluation": configuration,
            "dataset_identity": identity,
            "data_quality": data_quality,
            "fold_results": fold_results,
            "artifacts": {
                "audit": "results/internal_evaluation_repair_audit.json",
                "group_folds": "results/internal_evaluation_repair_group_folds.csv",
                "oof_images": "results/internal_evaluation_repair_oof_images.csv",
                "oof_groups": "results/internal_evaluation_repair_oof_groups.csv",
            },
            "environment": {
                "python": "3.12",
                "torch": "test",
                "torchvision": "test",
                "numpy": "test",
                "pandas": "test",
                "scikit_learn": "test",
                "pillow": "test",
                "platform": "test",
            },
            "runtime_seconds": 1.0,
            "internal_evaluation": report,
        }

    def canonical_oof_bundle(self):
        """Return a deterministic, scientifically self-consistent 737/283 bundle."""

        from validation.evaluation_utils import (
            frame_metrics,
            make_outer_folds,
            outer_fold_auc_summary,
        )

        group_rows = []
        group_counts = (57, 57, 57, 56, 56)
        for namespace, group_count in enumerate(group_counts):
            for group_index in range(group_count):
                group_rows.append(
                    {
                        "evaluation_group": f"P{namespace:01d}{group_index:03d}",
                        "label": group_index % 2,
                        "image_name": f"seed_{namespace:01d}{group_index:03d}.jpg",
                    }
                )
        seed_metadata = pd.DataFrame(group_rows)
        canonical_folds = make_outer_folds(seed_metadata, 5, 20_260_711)
        fold_map = canonical_folds.set_index("evaluation_group")["outer_fold"]

        image_rows = []
        image_counts = (148, 148, 147, 147, 147)
        image_index = 0
        for fold, image_count in enumerate(image_counts):
            fold_groups = canonical_folds.loc[
                canonical_folds["outer_fold"] == fold
            ].reset_index(drop=True)
            positive_groups = fold_groups.loc[
                fold_groups["label"] == 1, "evaluation_group"
            ].tolist()
            negative_groups = fold_groups.loc[
                fold_groups["label"] == 0, "evaluation_group"
            ].tolist()
            group_count = len(fold_groups)
            extra_count = image_count - 2 * group_count
            desired_positive_images = image_count // 2
            positive_extra_count = desired_positive_images - 2 * len(positive_groups)
            extra_groups = set(positive_groups[:positive_extra_count]) | set(
                negative_groups[: extra_count - positive_extra_count]
            )
            for group in fold_groups.itertuples(index=False):
                evaluation_group = str(group.evaluation_group)
                label = int(group.label)
                for _ in range(2 + int(evaluation_group in extra_groups)):
                    image_name = f"image_{image_index:04d}.jpg"
                    probability = 0.9 if label else 0.1
                    image_rows.append(
                        {
                            "image_name": image_name,
                            "patient_id": f"patient_{evaluation_group}",
                            "evaluation_group": evaluation_group,
                            "sha256": hashlib.sha256(image_name.encode()).hexdigest(),
                            "label": label,
                            "quality_score": 6.0,
                            "outer_fold": int(fold_map[evaluation_group]),
                            "probability": probability,
                            "selected_threshold": 0.5,
                            "predicted_class": label,
                        }
                    )
                    image_index += 1
        images = pd.DataFrame(image_rows)
        groups = aggregate_groups(images)
        groups["predicted_class"] = (
            groups["probability"] >= groups["selected_threshold"]
        ).astype(int)
        group_folds = groups[
            ["evaluation_group", "label", "n_images", "outer_fold"]
        ].copy()
        group_metrics = frame_metrics(groups)
        image_metrics = frame_metrics(images)
        outer_summary = outer_fold_auc_summary(groups)
        perfect_ci = {
            "auroc_95ci": [1.0, 1.0],
            "sensitivity_95ci": [1.0, 1.0],
            "specificity_95ci": [1.0, 1.0],
        }
        report = build_primary_report(
            outer_fold_auc=outer_summary,
            fold_stratified_ci={
                **perfect_ci,
                "resampling_unit": "evaluation_group_within_outer_fold",
                "fold_weighting": "equal",
                "seed": 20_260_711 + 9_003,
                "n_bootstrap": 5_000,
                "valid_replicates": 5_000,
            },
            pooled_group_metrics=group_metrics,
            pooled_group_cluster_ci=perfect_ci,
            image_metrics=image_metrics,
            image_cluster_ci=perfect_ci,
        )
        image_summary = images.groupby("outer_fold", sort=True)["label"].agg(
            ["size", "sum"]
        )
        group_summary = groups.groupby("outer_fold", sort=True)["label"].agg(
            ["size", "sum"]
        )
        result = self.canonical_result(
            report,
            fold_image_counts=image_summary["size"].astype(int).tolist(),
            fold_group_counts=group_summary["size"].astype(int).tolist(),
            fold_image_positives=image_summary["sum"].astype(int).tolist(),
            fold_group_positives=group_summary["sum"].astype(int).tolist(),
        )
        return result, images, groups, group_folds

    def test_dataset_inventory_fingerprint_is_relocatable_and_order_independent(self):
        from validation.evaluation_utils import (
            build_dataset_identity,
            inventory_fingerprint,
        )

        frame = pd.DataFrame(
            {
                "sha256": ["b" * 64, "a" * 64, "a" * 64],
                "patient_id": ["2", "1", "3"],
                "label": [1, 0, 0],
                "image_path": ["/private/one.jpg", "/private/two.jpg", "/else/three.jpg"],
            }
        )
        first = inventory_fingerprint(frame)
        relocated = frame.sample(frac=1, random_state=4).copy()
        relocated["image_path"] = ["/new/a", "/new/b", "/new/c"]
        self.assertEqual(first, inventory_fingerprint(relocated))
        changed_patient = frame.copy()
        changed_patient.loc[0, "patient_id"] = "different"
        self.assertNotEqual(first, inventory_fingerprint(changed_patient))
        changed_label = frame.copy()
        changed_label.loc[0, "label"] = 0
        self.assertNotEqual(first, inventory_fingerprint(changed_label))
        removed_duplicate = frame.iloc[:2].copy()
        self.assertNotEqual(first, inventory_fingerprint(removed_duplicate))

        quality = {
            "input_images": 3,
            "unique_image_hashes": 2,
            "input_patient_ids": 3,
            "negative_rows": 2,
            "positive_rows": 1,
            "ordered_image_sha256_patient_id_label_inventory_sha256": first,
        }
        identity = build_dataset_identity(quality, labels_csv_sha256="c" * 64)
        self.assertFalse(identity["canonical_hygd_v1_1_0_match"])

    def test_pure_module_imports_when_torch_and_torchvision_are_unavailable(self):
        code = (
            "import builtins, sys; "
            "original = builtins.__import__; "
            "guard = lambda name, *a, **k: "
            "(_ for _ in ()).throw(AssertionError('heavy import: ' + name)) "
            "if name.split('.')[0] in {'torch', 'torchvision'} "
            "else original(name, *a, **k); "
            "builtins.__import__ = guard; "
            "import validation.evaluation_utils; "
            "assert 'torch' not in sys.modules and 'torchvision' not in sys.modules"
        )
        completed = subprocess.run(
            [sys.executable, "-c", code],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)

    def test_transitive_cross_patient_duplicates_are_linked_and_counted_once(self):
        temporary, metadata = self.make_metadata(
            [
                ("p1-a.jpg", "1", 1, b"shared-a"),
                ("p2-a.jpg", "2", 1, b"shared-a"),
                ("p2-b.jpg", "2", 1, b"shared-b"),
                ("p3-b.jpg", "3", 1, b"shared-b"),
                ("p4-c.jpg", "4", 0, b"unique-c"),
            ]
        )
        self.addCleanup(temporary.cleanup)

        deduplicated, quality = prepare_duplicate_aware_metadata(metadata)

        self.assertEqual(len(deduplicated), 3)
        self.assertEqual(quality["removed_exact_duplicate_rows"], 2)
        self.assertEqual(quality["duplicate_hash_groups"], 2)
        self.assertEqual(quality["cross_patient_duplicate_hash_groups"], 2)
        self.assertEqual(quality["independent_evaluation_groups"], 2)
        positive_groups = deduplicated.loc[
            deduplicated["label"] == 1, "evaluation_group"
        ].unique()
        self.assertEqual(positive_groups.tolist(), ["P1+2+3"])
        self.assertFalse(deduplicated["sha256"].duplicated().any())

    def test_conflicting_labels_on_an_exact_duplicate_fail_closed(self):
        temporary, metadata = self.make_metadata(
            [
                ("positive.jpg", "1", 1, b"same-file"),
                ("negative.jpg", "2", 0, b"same-file"),
            ]
        )
        self.addCleanup(temporary.cleanup)

        with self.assertRaisesRegex(ValueError, "conflicting labels"):
            prepare_duplicate_aware_metadata(metadata)

    def test_nonbinary_labels_fail_closed_without_integer_coercion(self):
        temporary, metadata = self.make_metadata(
            [
                ("fractional.jpg", "1", 0.5, b"fractional-label"),
                ("positive.jpg", "2", 1, b"positive-label"),
            ]
        )
        self.addCleanup(temporary.cleanup)

        with self.assertRaisesRegex(ValueError, "binary values 0 and 1"):
            prepare_duplicate_aware_metadata(metadata)

    def test_source_loader_rejects_unknown_labels_missing_patients_and_path_escape(self):
        from src.data_utils import load_dataset_metadata

        with tempfile.TemporaryDirectory() as temporary:
            raw = Path(temporary)
            (raw / "Images").mkdir()

            def write_labels(patient, label, image_name="image.jpg"):
                pd.DataFrame(
                    {
                        "Image Name": [image_name],
                        "Patient": [patient],
                        "Label": [label],
                        "Quality Score": [6.0],
                    }
                ).to_csv(raw / "Labels.csv", index=False)

            write_labels("1", "GON?")
            with self.assertRaisesRegex(ValueError, r"GON\+ or GON-"):
                load_dataset_metadata(raw)

            write_labels(np.nan, "GON+")
            with self.assertRaisesRegex(ValueError, "Supplied patient ID"):
                load_dataset_metadata(raw)

            write_labels("1", "GON+", "../outside.jpg")
            with self.assertRaisesRegex(ValueError, "plain filenames"):
                load_dataset_metadata(raw)

            write_labels("1", "GON+")
            labels_bytes = (raw / "Labels.csv").read_bytes()
            expected = hashlib.sha256(labels_bytes).hexdigest()
            loaded = load_dataset_metadata(raw, expected_labels_sha256=expected)
            self.assertEqual(loaded.attrs["labels_csv_sha256"], expected)
            (raw / "Labels.csv").write_bytes(labels_bytes + b"\n")
            with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
                load_dataset_metadata(raw, expected_labels_sha256=expected)

    def test_verified_file_reader_hashes_the_exact_bytes_it_returns(self):
        from validation.evaluation_utils import read_verified_file_bytes

        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "image.jpg"
            original = b"verified-original-image-bytes"
            altered = b"transiently-substituted-image-bytes"
            expected = hashlib.sha256(original).hexdigest()
            path.write_bytes(original)

            content, digest = read_verified_file_bytes(path, expected_sha256=expected)
            self.assertEqual(content, original)
            self.assertEqual(digest, expected)

            path.write_bytes(altered)
            with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
                read_verified_file_bytes(path, expected_sha256=expected)

            path.write_bytes(original)
            restored, restored_digest = read_verified_file_bytes(
                path, expected_sha256=expected
            )
            self.assertEqual(restored, original)
            self.assertEqual(restored_digest, expected)

    def test_all_verified_read_boundaries_reject_fifos_without_blocking(self):
        import os

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            results = root / "results"
            results.mkdir()
            fifo = results / "fifo"
            os.mkfifo(fifo)
            commands = (
                (
                    "from validation.evaluation_utils import read_verified_file_bytes; "
                    "import signal; signal.alarm(2); "
                    f"read_verified_file_bytes({str(fifo)!r})"
                ),
                (
                    "from validation.evaluation_utils import read_confined_json; "
                    "import signal; signal.alarm(2); "
                    f"read_confined_json({str(root)!r}, 'results/fifo')"
                ),
                (
                    "from validation.evaluation_utils import _read_terminal_artifact_bytes; "
                    "import signal; signal.alarm(2); "
                    f"_read_terminal_artifact_bytes({{'artifacts':{{'audit':'results/fifo'}},"
                    "'artifact_sha256':{'audit':'0'*64},"
                    f"'artifact_size_bytes':{{'audit':0}}}}, {str(root)!r})"
                ),
            )
            for index, command in enumerate(commands):
                with self.subTest(boundary=index):
                    completed = subprocess.run(
                        [sys.executable, "-B", "-c", command],
                        cwd=ROOT,
                        capture_output=True,
                        text=True,
                        timeout=15,
                        check=False,
                    )
                    self.assertNotEqual(completed.returncode, 0)
                    self.assertRegex(
                        completed.stderr,
                        "regular file|single-link",
                    )

    def test_canonical_image_decode_rejects_transient_valid_image_substitution(self):
        from PIL import Image

        from src.data_utils import load_verified_rgb_image

        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "image.png"
            Image.new("RGB", (2, 2), (255, 0, 0)).save(path)
            original = path.read_bytes()
            expected = hashlib.sha256(original).hexdigest()

            Image.new("RGB", (2, 2), (0, 0, 255)).save(path)
            with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
                load_verified_rgb_image(path, expected_sha256=expected)

            path.write_bytes(original)
            decoded = load_verified_rgb_image(path, expected_sha256=expected)
            self.assertEqual(decoded.getpixel((0, 0)), (255, 0, 0))

        experiment_source = (ROOT / "src" / "experiments.py").read_text()
        self.assertIn("load_verified_rgb_image(", experiment_source)
        self.assertIn("expected_sha256=expected_sha256", experiment_source)

    def test_outer_and_inner_splits_keep_evaluation_groups_disjoint(self):
        metadata = pd.DataFrame(
            [
                {
                    "evaluation_group": f"P{index:02d}",
                    "label": index % 2,
                    "image_name": f"{index}.jpg",
                }
                for index in range(40)
            ]
        )

        assignments = make_outer_folds(metadata, n_splits=5, seed=17)
        self.assertEqual(len(assignments), 40)
        self.assertEqual(set(assignments["outer_fold"]), set(range(5)))
        self.assertFalse(assignments["evaluation_group"].duplicated().any())

        fold_map = assignments.set_index("evaluation_group")["outer_fold"]
        metadata["outer_fold"] = metadata["evaluation_group"].map(fold_map)
        outer_train = metadata[metadata["outer_fold"] != 0]
        outer_test = metadata[metadata["outer_fold"] == 0]
        train_groups, validation_groups = inner_group_split(
            outer_train, validation_fraction=0.25, seed=23
        )

        outer_test_groups = set(outer_test["evaluation_group"])
        self.assertFalse(train_groups & validation_groups)
        self.assertFalse(train_groups & outer_test_groups)
        self.assertFalse(validation_groups & outer_test_groups)
        self.assertEqual(
            train_groups | validation_groups,
            set(outer_train["evaluation_group"]),
        )

    def test_group_aggregation_is_order_independent_and_checks_invariants(self):
        frame = pd.DataFrame(
            {
                "evaluation_group": ["b", "a", "a"],
                "label": [1, 0, 0],
                "probability": [0.8, 0.1, 0.3],
                "selected_threshold": [0.6, 0.4, 0.4],
                "outer_fold": [1, 0, 0],
                "image_name": ["b.jpg", "a1.jpg", "a2.jpg"],
            }
        )

        first = aggregate_groups(frame)
        second = aggregate_groups(frame.sample(frac=1, random_state=9))
        assert_frame_equal(first, second)
        self.assertAlmostEqual(
            first.loc[first["evaluation_group"] == "a", "probability"].item(),
            0.2,
        )

        conflicting = frame.copy()
        conflicting.loc[2, "label"] = 1
        with self.assertRaisesRegex(ValueError, "label"):
            aggregate_groups(conflicting)

        inconsistent_threshold = frame.copy()
        inconsistent_threshold.loc[2, "selected_threshold"] = 0.9
        with self.assertRaisesRegex(ValueError, "selected_threshold"):
            aggregate_groups(inconsistent_threshold)

    def test_threshold_is_selected_only_from_the_sensitivity_constraint(self):
        result = select_threshold(
            labels=np.array([1, 1, 1, 0, 0]),
            probabilities=np.array([0.9, 0.8, 0.4, 0.7, 0.1]),
            target_sensitivity=2 / 3,
        )
        self.assertAlmostEqual(result["threshold"], 0.8)
        self.assertAlmostEqual(result["inner_validation_sensitivity"], 2 / 3)
        self.assertEqual(result["inner_validation_specificity"], 1.0)

        with self.assertRaisesRegex(ValueError, "greater than 0 and at most 1"):
            select_threshold([0, 1], [0.1, 0.9], target_sensitivity=0)
        with self.assertRaisesRegex(ValueError, "greater than 0 and at most 1"):
            select_threshold([0, 1], [0.1, 0.9], target_sensitivity=1.1)
        with self.assertRaisesRegex(ValueError, "positive and negative"):
            select_threshold([0, 0], [0.1, 0.2], target_sensitivity=0.95)

    def test_threshold_tie_break_prefers_higher_sensitivity_before_threshold(self):
        result = select_threshold(
            labels=np.array([1, 1, 0]),
            probabilities=np.array([0.9, 0.8, 0.1]),
            target_sensitivity=0.5,
        )
        # Thresholds 0.8 and 0.9 both give specificity 1.0. The 0.8
        # threshold dominates because it retains sensitivity 1.0 rather than 0.5.
        self.assertAlmostEqual(result["threshold"], 0.8)
        self.assertEqual(result["inner_validation_sensitivity"], 1.0)

    def test_cluster_resampling_repeats_whole_clusters_not_individual_rows(self):
        frame = pd.DataFrame(
            {
                "row_id": ["a1", "a2", "b1", "c1", "c2", "c3"],
                "evaluation_group": ["a", "a", "b", "c", "c", "c"],
                "label": [0, 0, 1, 1, 1, 1],
                "probability": [0.1, 0.2, 0.9, 0.6, 0.7, 0.8],
                "selected_threshold": [0.5] * 6,
            }
        )

        sampled = resample_clusters(
            frame,
            cluster_column="evaluation_group",
            sampled_clusters=["a", "a", "c"],
        )
        counts = sampled["row_id"].value_counts().to_dict()
        self.assertEqual(
            counts,
            {"a1": 2, "a2": 2, "c1": 1, "c2": 1, "c3": 1},
        )
        self.assertNotIn("b1", counts)

    def test_cluster_bootstrap_is_reproducible(self):
        frame = pd.DataFrame(
            {
                "evaluation_group": ["a", "a", "b", "c", "d", "d"],
                "label": [0, 0, 1, 0, 1, 1],
                "probability": [0.1, 0.2, 0.9, 0.3, 0.8, 0.7],
                "selected_threshold": [0.5] * 6,
            }
        )
        first = cluster_bootstrap(frame, "evaluation_group", 200, seed=19)
        second = cluster_bootstrap(frame, "evaluation_group", 200, seed=19)
        self.assertEqual(first, second)
        self.assertEqual(
            set(first),
            {"auroc_95ci", "sensitivity_95ci", "specificity_95ci"},
        )

        one_class = frame.assign(label=1)
        with self.assertRaisesRegex(RuntimeError, "No valid bootstrap replicates"):
            cluster_bootstrap(one_class, "evaluation_group", 20, seed=19)

    def test_patient_cluster_auc_resamples_whole_patients_and_is_row_order_invariant(self):
        from validation.evaluation_utils import patient_cluster_auc_ci

        frame = pd.DataFrame(
            {
                "patient_id": ["p1", "p1", "p2", "p2", "p3", "p3", "p4", "p4"],
                "label": [0, 0, 0, 1, 1, 1, 0, 1],
                "probability": [0.05, 0.1, 0.2, 0.7, 0.8, 0.95, 0.3, 0.75],
            }
        )
        first = patient_cluster_auc_ci(
            frame["label"],
            frame["probability"],
            frame["patient_id"],
            n_bootstrap=500,
            seed=9,
        )
        shuffled = frame.sample(frac=1, random_state=4).reset_index(drop=True)
        second = patient_cluster_auc_ci(
            shuffled["label"],
            shuffled["probability"],
            shuffled["patient_id"],
            n_bootstrap=500,
            seed=9,
        )
        self.assertEqual(first, second)
        self.assertEqual(first["resampling_unit"], "patient")
        self.assertEqual(first["point_estimand_unit"], "eye_or_image_row")
        self.assertEqual(first["valid_replicates"], 500)
        self.assertFalse(first["canonical_or_confirmatory"])

        with self.assertRaisesRegex(ValueError, "identifiers"):
            patient_cluster_auc_ci([0, 1], [0.1, 0.9], ["", "p2"], 10, 1)
        with self.assertRaisesRegex(ValueError, "finite"):
            patient_cluster_auc_ci([0, 1], [0.1, np.nan], ["p1", "p2"], 10, 1)
        for invalid_ids in ([None, None, "b", "b"], ["p1\n", "p1\n", "p2", "p2"]):
            with self.subTest(invalid_ids=invalid_ids):
                with self.assertRaisesRegex(ValueError, "identifiers"):
                    patient_cluster_auc_ci(
                        [0, 0, 1, 1],
                        [0.1, 0.2, 0.8, 0.9],
                        invalid_ids,
                        10,
                        1,
                    )

    def test_metric_core_rejects_coerced_or_out_of_domain_vectors(self):
        from validation.evaluation_utils import (
            fixed_threshold_metrics,
            metrics_from_predictions,
        )

        invalid_cases = (
            ([0, 2], [0.1, 0.9], [0, 1]),
            ([0, 0.5], [0.1, 0.9], [0, 1]),
            ([0, 1], [0.1, 0.9], [0, 2]),
            ([0, 1], [0.1, 1.5], [0, 1]),
            (["0", "1"], [0.1, 0.9], [0, 1]),
            ([False, True], [0.1, 0.9], [0, 1]),
            ([0, 1], ["0.1", "0.9"], [0, 1]),
            ([0, 1], [0.1, 0.9], [False, True]),
            ([[0, 1]], [[0.1, 0.9]], [[0, 1]]),
        )
        for labels, probabilities, predictions in invalid_cases:
            with self.subTest(
                labels=labels,
                probabilities=probabilities,
                predictions=predictions,
            ):
                with self.assertRaisesRegex(ValueError, "binary|probabilities|one-dimensional"):
                    metrics_from_predictions(labels, probabilities, predictions)

        for threshold in (-0.1, 1.1, np.nan):
            with self.subTest(threshold=threshold):
                with self.assertRaisesRegex(ValueError, "threshold"):
                    fixed_threshold_metrics([0, 1], [0.1, 0.9], threshold)

    def test_stratified_group_holdout_is_disjoint_complete_and_deterministic(self):
        from validation.evaluation_utils import stratified_group_holdout

        rows = []
        for patient in range(12):
            pattern = patient % 3
            labels = (0, 0) if pattern == 0 else ((1, 1) if pattern == 1 else (0, 1))
            for eye, label in enumerate(labels):
                rows.append({"patient_id": f"p{patient}", "label": label, "eye": eye})
        frame = pd.DataFrame(rows)
        train, test = stratified_group_holdout(
            frame, "patient_id", "label", test_fraction=0.25, seed=7
        )
        train_again, test_again = stratified_group_holdout(
            frame, "patient_id", "label", test_fraction=0.25, seed=7
        )
        np.testing.assert_array_equal(train, train_again)
        np.testing.assert_array_equal(test, test_again)
        self.assertFalse(
            set(frame.iloc[train]["patient_id"]) & set(frame.iloc[test]["patient_id"])
        )
        self.assertEqual(set(train) | set(test), set(range(len(frame))))
        self.assertEqual(set(frame.iloc[train]["label"]), {0, 1})
        self.assertEqual(set(frame.iloc[test]["label"]), {0, 1})

    def test_hash_bound_subject_map_requires_exact_bytes_and_coverage(self):
        from validation.evaluation_utils import load_hash_bound_subject_map

        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "subjects.csv"
            pd.DataFrame(
                {"stem": ["a", "b", "c"], "subject_id": ["s1", "s1", "s2"]}
            ).to_csv(path, index=False)
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            self.assertEqual(
                load_hash_bound_subject_map(["c", "a", "b"], path, digest),
                ({"a": "s1", "b": "s1", "c": "s2"}, digest),
            )
            with self.assertRaisesRegex(ValueError, "exact expected stem set"):
                load_hash_bound_subject_map(["a", "b"], path, digest)
            with self.assertRaisesRegex(ValueError, "required"):
                load_hash_bound_subject_map(["a"], None, None)
            with self.assertRaisesRegex(ValueError, "mismatch"):
                load_hash_bound_subject_map(["a", "b", "c"], path, "0" * 64)

            symlink = path.with_name("subjects-link.csv")
            symlink.symlink_to(path)
            with self.assertRaisesRegex(ValueError, "no-follow"):
                load_hash_bound_subject_map(["a", "b", "c"], symlink, digest)

            hardlink = path.with_name("subjects-hardlink.csv")
            hardlink.hardlink_to(path)
            with self.assertRaisesRegex(ValueError, "single-link"):
                load_hash_bound_subject_map(["a", "b", "c"], path, digest)

    def test_outer_fold_macro_auc_is_invariant_to_fold_specific_score_scaling(self):
        frame = pd.DataFrame(
            {
                "evaluation_group": [f"g{i}" for i in range(8)],
                "outer_fold": [0, 0, 0, 0, 1, 1, 1, 1],
                "label": [0, 0, 1, 1, 0, 0, 1, 1],
                "probability": [0.1, 0.2, 0.8, 0.9, 0.15, 0.25, 0.75, 0.85],
                "selected_threshold": [0.5] * 8,
            }
        )
        original = outer_fold_auc_summary(frame)
        shifted = frame.copy()
        shifted.loc[shifted["outer_fold"] == 1, "probability"] += 10
        shifted_summary = outer_fold_auc_summary(shifted)

        self.assertEqual(original, shifted_summary)
        self.assertNotEqual(
            roc_auc_score(frame["label"], frame["probability"]),
            roc_auc_score(shifted["label"], shifted["probability"]),
        )

        first = fold_stratified_group_bootstrap(frame, 200, seed=29)
        second = fold_stratified_group_bootstrap(frame, 200, seed=29)
        self.assertEqual(first, second)
        self.assertEqual(
            first["resampling_unit"],
            "evaluation_group_within_outer_fold",
        )
        self.assertEqual(first["seed"], 29)

    def test_fold_stratified_bootstrap_is_invariant_to_input_row_order(self):
        frame = pd.DataFrame(
            {
                "evaluation_group": [f"g{i}" for i in range(12)],
                "outer_fold": [0] * 6 + [1] * 6,
                "label": [0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 1],
                "probability": [
                    0.01,
                    0.15,
                    0.32,
                    0.45,
                    0.54,
                    0.93,
                    0.05,
                    0.24,
                    0.39,
                    0.61,
                    0.72,
                    0.99,
                ],
                "selected_threshold": [0.5] * 12,
            }
        )
        original = fold_stratified_group_bootstrap(frame, 50, seed=1)
        shuffled = fold_stratified_group_bootstrap(
            frame.sample(frac=1, random_state=1).reset_index(drop=True),
            50,
            seed=1,
        )
        self.assertEqual(original, shuffled)

        duplicated = pd.concat([frame, frame.iloc[[0]]], ignore_index=True)
        with self.assertRaisesRegex(ValueError, "one row per evaluation group"):
            fold_stratified_group_bootstrap(duplicated, 10, seed=1)

        missing_group = frame.copy()
        missing_group.loc[0, "evaluation_group"] = None
        with self.assertRaisesRegex(ValueError, "non-null strings"):
            fold_stratified_group_bootstrap(missing_group, 10, seed=1)

    def test_historical_payload_policy_rejects_cross_validation_results(self):
        payload = finalize_historical_comparison(
            {
                "frozen_aug": {"test_auc": 0.99},
                "finetune_layer4_aug": {
                    "test_auc": 0.8,
                    "historical_image_level_bootstrap_95ci": {"auc": [0.8, 1.0]},
                }
            }
        )
        self.assertEqual(
            payload["_provenance"]["status"],
            "historical_development_comparison",
        )
        self.assertFalse(payload["_provenance"]["canonical_internal_estimate"])
        self.assertFalse(payload["_provenance"]["configuration_selection_permitted"])
        self.assertNotIn("selected_configuration", payload)
        self.assertEqual(payload["frozen_aug"]["test_auc"], 0.99)
        self.assertEqual(payload["finetune_layer4_aug"]["test_auc"], 0.8)

        with self.assertRaisesRegex(ValueError, "cross-validation"):
            finalize_historical_comparison(
                {
                    "finetune_layer4_aug": {"test_auc": 0.9},
                    "cv": {"fold_aucs": [0.8, 0.9]},
                }
            )
        with self.assertRaisesRegex(ValueError, "configuration selection"):
            finalize_historical_comparison(
                {
                    "frozen_aug": {"test_auc": 0.99},
                    "selected_configuration": "frozen_aug",
                }
            )

    def test_image_row_bootstrap_is_explicitly_historical_and_noncanonical(self):
        result = historical_image_bootstrap_ci(
            labels=[0, 0, 1, 1],
            probabilities=[0.1, 0.2, 0.8, 0.9],
            threshold=0.5,
            n_bootstrap=100,
            seed=7,
        )
        self.assertEqual(result["status"], "historical_image_level_only")
        self.assertEqual(result["sampling_unit"], "image_row")
        self.assertFalse(result["canonical_internal_estimate"])
        self.assertIn("auroc_95ci", result["intervals"])

    def test_primary_report_hard_separates_group_and_image_uncertainty(self):
        report = build_primary_report(
            outer_fold_auc={
                "fold_aurocs": [
                    {
                        "outer_fold": 0,
                        "n_groups": 10,
                        "positives": 5,
                        "negatives": 5,
                        "auroc": 0.8,
                    },
                    {
                        "outer_fold": 1,
                        "n_groups": 10,
                        "positives": 5,
                        "negatives": 5,
                        "auroc": 0.9,
                    },
                ],
                "unweighted_mean_auroc": 0.85,
                "group_count_weighted_mean_auroc": 0.85,
            },
            fold_stratified_ci={
                "auroc_95ci": [0.75, 0.94],
                "sensitivity_95ci": [0.7, 0.9],
                "specificity_95ci": [0.72, 0.91],
                "resampling_unit": "evaluation_group_within_outer_fold",
                "fold_weighting": "equal",
                "seed": 20_260_711 + 9_003,
                "n_bootstrap": 5_000,
                "valid_replicates": 5_000,
            },
            pooled_group_metrics={
                "auroc": 0.81,
                "n": 20,
                "positives": 10,
                "negatives": 10,
                "sensitivity": 0.8,
                "specificity": 0.8,
                "confusion_tn_fp_fn_tp": [8, 2, 2, 8],
            },
            pooled_group_cluster_ci={
                "auroc_95ci": [0.71, 0.91],
                "sensitivity_95ci": [0.7, 0.9],
                "specificity_95ci": [0.72, 0.91],
            },
            image_metrics={
                "auroc": 0.88,
                "n": 50,
                "positives": 25,
                "negatives": 25,
                "sensitivity": 0.8,
                "specificity": 0.8,
                "confusion_tn_fp_fn_tp": [20, 5, 5, 20],
            },
            image_cluster_ci={
                "auroc_95ci": [0.79, 0.95],
                "sensitivity_95ci": [0.7, 0.9],
                "specificity_95ci": [0.7, 0.9],
            },
        )
        self.assertEqual(
            report["preferred_internal_metric_source"],
            "duplicate_aware_group_macro_outer_fold_auc_with_fold_stratified_group_bootstrap",
        )
        self.assertEqual(report["primary"]["grain"], "evaluation_group")
        self.assertEqual(report["primary"]["auroc"], 0.85)
        self.assertEqual(report["primary"]["auroc_95ci"], [0.75, 0.94])
        self.assertEqual(report["pooled_oof_group_descriptive"]["auroc"], 0.81)
        self.assertEqual(report["secondary_image_level"]["auroc"], 0.88)
        self.assertEqual(
            report["secondary_image_level"]["uncertainty_source"],
            "evaluation_group_cluster_bootstrap",
        )
        self.assertEqual(
            report["pooled_oof_group_descriptive"]["uncertainty_source"],
            "global_group_cluster_bootstrap_over_pooled_oof_scores",
        )
        self.assertIn(
            "separately fitted fold models",
            report["secondary_image_level"]["warning"],
        )
        self.assertIn("primary percentile interval", report["uncertainty_scope"])
        self.assertNotIn("historical", repr(report).lower())
        validate_primary_report(report)

        wrong_source = dict(report)
        wrong_source["preferred_internal_metric_source"] = "historical_image_bootstrap"
        with self.assertRaisesRegex(ValueError, "preferred_internal_metric_source"):
            validate_primary_report(wrong_source)

        canonical_result = self.canonical_result(report)
        canonical_configuration = canonical_result[
            "configuration_fixed_before_outer_evaluation"
        ]
        validate_internal_result_payload(canonical_result)

        wrong_best_epoch = copy.deepcopy(canonical_result)
        wrong_best_epoch["fold_results"][0][
            "best_epoch_selected_on_inner_validation_loss"
        ] = 2
        with self.assertRaisesRegex(ValueError, "minimum inner-validation loss"):
            validate_internal_result_payload(wrong_best_epoch)

        missed_sensitivity_target = copy.deepcopy(canonical_result)
        missed_sensitivity_target["fold_results"][0][
            "selected_threshold_from_inner_group_validation"
        ]["inner_validation_sensitivity"] = 0.1
        with self.assertRaisesRegex(ValueError, "sensitivity target"):
            validate_internal_result_payload(missed_sensitivity_target)

        incomplete_partition_counts = copy.deepcopy(canonical_result)
        incomplete_partition_counts["fold_results"][0]["counts"][
            "inner_train_images"
        ] -= 1
        with self.assertRaisesRegex(ValueError, "partition counts"):
            validate_internal_result_payload(incomplete_partition_counts)

        for missing in (
            "protocol_version",
            "dataset_identity",
            "data_quality",
            "fold_results",
            "artifacts",
            "environment",
            "runtime_seconds",
        ):
            incomplete = copy.deepcopy(canonical_result)
            del incomplete[missing]
            with self.assertRaisesRegex(
                ValueError, "requires|invalid|protocol|exact scientific schema"
            ):
                validate_internal_result_payload(incomplete)

        wrong_dataset = copy.deepcopy(canonical_result)
        wrong_dataset["dataset_identity"]["labels_csv_sha256"] = "0" * 64
        wrong_dataset["dataset_identity"]["canonical_hygd_v1_1_0_match"] = False
        with self.assertRaisesRegex(ValueError, "HYGD v1.1.0"):
            validate_internal_result_payload(wrong_dataset)
        for path, bad_value in (
            (("dataset_identity", "inventory_rows"), 747.0),
            (("dataset_identity", "patient_ids"), 288.0),
            (("data_quality", "input_images"), 747.0),
            (("data_quality", "label_conflicts"), False),
        ):
            malformed = copy.deepcopy(canonical_result)
            malformed[path[0]][path[1]] = bad_value
            with self.assertRaisesRegex(ValueError, "HYGD v1.1.0"):
                validate_internal_result_payload(malformed)
        with self.assertRaisesRegex(ValueError, "locked canonical configuration"):
            validate_internal_result_payload(
                {
                    **copy.deepcopy(canonical_result),
                    "configuration_fixed_before_outer_evaluation": {
                        **canonical_configuration,
                        "epochs": 1,
                    },
                }
            )
        with self.assertRaisesRegex(ValueError, "historical/CV|exact scientific schema"):
            validate_internal_result_payload(
                {**copy.deepcopy(canonical_result), "selected_configuration": "best_test_auc_model"}
            )
        with self.assertRaisesRegex(ValueError, "historical/CV|exact scientific schema"):
            validate_internal_result_payload(
                {**copy.deepcopy(canonical_result), "historical": {"test_auc": 0.99}}
            )

        with self.assertRaisesRegex(ValueError, "status must be complete"):
            validate_internal_result_payload(
                {
                    **copy.deepcopy(canonical_result),
                    "status": "partial_smoke_run",
                }
            )

        nonfinite_runtime = copy.deepcopy(canonical_result)
        nonfinite_runtime["runtime_seconds"] = float("nan")
        with self.assertRaisesRegex(ValueError, "runtime_seconds"):
            validate_internal_result_payload(nonfinite_runtime)

        aliased = copy.deepcopy(canonical_result)
        aliased["artifacts"]["oof_groups"] = aliased["artifacts"]["oof_images"]
        with self.assertRaisesRegex(ValueError, "distinct"):
            validate_internal_result_payload(aliased)

        for field, bad_value in (
            ("completed_outer_folds", [0.9, 1.9, 2.9, 3.9, 4.9]),
        ):
            malformed = copy.deepcopy(canonical_result)
            malformed[field] = bad_value
            with self.assertRaisesRegex(ValueError, "outer-fold metadata"):
                validate_internal_result_payload(malformed)

        float_fold = copy.deepcopy(canonical_result)
        float_fold["fold_results"][0]["outer_fold"] = 0.0
        with self.assertRaisesRegex(ValueError, "ordered by outer_fold"):
            validate_internal_result_payload(float_fold)

        float_seed = copy.deepcopy(canonical_result)
        float_seed["fold_results"][0]["seed"] = float(
            float_seed["fold_results"][0]["seed"]
        )
        with self.assertRaisesRegex(ValueError, "seed is inconsistent"):
            validate_internal_result_payload(float_seed)

        malformed_commitments = copy.deepcopy(canonical_result)
        malformed_commitments["artifact_sha256"] = {
            role: "bad" for role in malformed_commitments["artifacts"]
        }
        malformed_commitments["artifact_size_bytes"] = {
            role: -1 for role in malformed_commitments["artifacts"]
        }
        with self.assertRaisesRegex(ValueError, "commitments|hashes"):
            validate_internal_result_payload(malformed_commitments)

        for section, key, value, message in (
            ("environment", "python", None, "environment provenance"),
            (
                "configuration_fixed_before_outer_evaluation",
                "device",
                False,
                "device|fixed-recipe metadata",
            ),
            (
                "configuration_fixed_before_outer_evaluation",
                "configuration_origin",
                "",
                "fixed-recipe metadata",
            ),
            (
                "configuration_fixed_before_outer_evaluation",
                "residual_post_selection_risk",
                "",
                "fixed-recipe metadata",
            ),
        ):
            malformed = copy.deepcopy(canonical_result)
            malformed[section][key] = value
            with self.assertRaisesRegex(ValueError, message):
                validate_internal_result_payload(malformed)

    def test_repair_outputs_are_confined_to_the_ignored_results_prefix(self):
        from validation.evaluation_utils import resolve_run_output_prefix

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            expected = root / "results" / "internal_evaluation_repair"
            self.assertEqual(
                resolve_repair_output_prefix(
                    root, "results/internal_evaluation_repair"
                ),
                expected,
            )
            with self.assertRaisesRegex(ValueError, "results/internal_evaluation_repair"):
                resolve_repair_output_prefix(root, "private/repair")
            with self.assertRaisesRegex(ValueError, "results/internal_evaluation_repair"):
                resolve_repair_output_prefix(root, "../escape")
            with self.assertRaisesRegex(ValueError, "results/internal_evaluation_repair"):
                resolve_repair_output_prefix(root, str(root.parent / "absolute-outside"))

            outside = root / "outside"
            outside.mkdir()
            results = root / "results"
            results.mkdir()
            (results / "internal_evaluation_repair_link").symlink_to(
                outside, target_is_directory=True
            )
            with self.assertRaisesRegex(ValueError, "results/internal_evaluation_repair"):
                resolve_repair_output_prefix(
                    root,
                    "results/internal_evaluation_repair_link/attempt",
                )

        with tempfile.TemporaryDirectory() as temporary, tempfile.TemporaryDirectory() as outside:
            root = Path(temporary).resolve()
            (root / "results").symlink_to(Path(outside), target_is_directory=True)
            with self.assertRaisesRegex(ValueError, "results directory cannot be a symlink"):
                resolve_repair_output_prefix(
                    root,
                    "results/internal_evaluation_repair",
                )

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            canonical = resolve_run_output_prefix(
                root,
                "results/internal_evaluation_repair",
                max_folds=None,
                total_folds=5,
                audit_only=False,
            )
            smoke = resolve_run_output_prefix(
                root,
                "results/internal_evaluation_repair",
                max_folds=1,
                total_folds=5,
                audit_only=False,
            )
            audit_only = resolve_run_output_prefix(
                root,
                "results/internal_evaluation_repair",
                max_folds=None,
                total_folds=5,
                audit_only=True,
            )
            self.assertEqual(canonical.name, "internal_evaluation_repair")
            self.assertEqual(smoke.name, "internal_evaluation_repair_smoke_1of5")
            self.assertEqual(
                audit_only.name,
                "internal_evaluation_repair_audit_only",
            )
            with self.assertRaisesRegex(ValueError, "max_folds must be positive"):
                resolve_run_output_prefix(
                    root,
                    "results/internal_evaluation_repair",
                    max_folds=0,
                    total_folds=5,
                    audit_only=False,
                )
            with self.assertRaisesRegex(ValueError, "proper subset"):
                resolve_run_output_prefix(
                    root,
                    "results/internal_evaluation_repair",
                    max_folds=5,
                    total_folds=5,
                    audit_only=False,
                )

    def test_atomic_text_write_replaces_symlinks_and_hardlinks_without_following(self):
        from validation.evaluation_utils import atomic_write_text

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            outside = root / "outside.txt"
            outside.write_text("private")

            symlink_target = root / "symlink-output.txt"
            symlink_target.symlink_to(outside)
            atomic_write_text(symlink_target, "safe-symlink", overwrite=True)
            self.assertEqual(outside.read_text(), "private")
            self.assertFalse(symlink_target.is_symlink())
            self.assertEqual(symlink_target.read_text(), "safe-symlink")

            hardlink_target = root / "hardlink-output.txt"
            hardlink_target.hardlink_to(outside)
            atomic_write_text(hardlink_target, "safe-hardlink", overwrite=True)
            self.assertEqual(outside.read_text(), "private")
            self.assertEqual(hardlink_target.read_text(), "safe-hardlink")

            fresh = root / "fresh.txt"
            atomic_write_text(fresh, "first")
            with self.assertRaises(FileExistsError):
                atomic_write_text(fresh, "second")

            linked_parent = root / "linked-parent"
            linked_parent.symlink_to(root, target_is_directory=True)
            with self.assertRaises(OSError):
                atomic_write_text(linked_parent / "escaped.txt", "blocked")

            outside_directory = root / "outside-directory"
            outside_directory.mkdir()
            nested = outside_directory / "nested"
            nested.mkdir()
            alias = root / "alias"
            alias.symlink_to(outside_directory, target_is_directory=True)
            escaped = alias / "nested" / "escaped.txt"
            with self.assertRaises(OSError):
                atomic_write_text(escaped, "blocked")
            self.assertFalse((nested / "escaped.txt").exists())

    def test_atomic_text_write_fsyncs_file_and_directory_after_publication(self):
        from validation.evaluation_utils import atomic_write_text

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for name, overwrite in (("new.txt", False), ("replace.txt", True)):
                target = root / name
                if overwrite:
                    target.write_text("old")
                with patch(
                    "validation.evaluation_utils.os.fsync",
                    wraps=__import__("os").fsync,
                ) as fsync:
                    atomic_write_text(target, "durable", overwrite=overwrite)
                self.assertGreaterEqual(
                    fsync.call_count,
                    2,
                    "publication must fsync both file data and the parent directory",
                )
                self.assertEqual(target.read_text(), "durable")

    def test_atomic_text_write_failure_does_not_publish_partial_target(self):
        from validation.evaluation_utils import atomic_write_text

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            target = root / "never-published.txt"
            with patch(
                "validation.evaluation_utils.os.fsync",
                side_effect=OSError("injected fsync failure"),
            ):
                with self.assertRaisesRegex(OSError, "injected fsync failure"):
                    atomic_write_text(target, "partial")
            self.assertFalse(target.exists())
            self.assertEqual(list(root.glob("*.tmp")), [])

    def test_output_bundle_refuses_any_preexisting_target(self):
        from validation.evaluation_utils import assert_fresh_output_bundle

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            paths = [root / "one.json", root / "two.csv"]
            assert_fresh_output_bundle(paths)
            paths[1].symlink_to(root / "outside.csv")
            with self.assertRaisesRegex(FileExistsError, "refusing to overwrite"):
                assert_fresh_output_bundle(paths)

            with self.assertRaisesRegex(ValueError, "unique"):
                assert_fresh_output_bundle(
                    [root / "alias.json", root / "nested" / ".." / "alias.json"]
                )

    def test_current_seed_aggregators_consume_patient_aware_outputs_and_refuse_overwrite(self):
        from validation import aggregate_seeds, aggregate_seeds_reverse

        status = {
            "status": "adaptive_development_evidence",
            "target_blind": False,
            "transportability_established": False,
            "license_compatibility": "needs-proof",
            "external_claim_permitted": False,
            "rimone_subject_identity": "operator_asserted_hash_bound_needs_independent_proof",
            "warning": "Target AUROC was displayed during development and target-domain anatomical resources influenced preprocessing.",
        }
        mapping = "a" * 64

        def uncertainty(auroc):
            return {
                "auroc": auroc,
                "auroc_95ci": [max(0.0, auroc - 0.05), min(1.0, auroc + 0.05)],
                "point_estimand_unit": "eye_or_image_row",
                "resampling_unit": "patient",
                "scope": "conditional_on_fixed_adaptive_predictions",
                "canonical_or_confirmatory": False,
                "seed": 42,
                "attempted_replicates": 2000,
                "valid_replicates": 2000,
            }

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            namespace = root / "results" / "patient_aware"
            namespace.mkdir(parents=True)
            for seed in range(5):
                forward_auroc = round(0.80 + seed / 100, 4)
                (namespace / f"attemptB_seed{seed}.json").write_text(
                    json.dumps(
                        {
                            "_evidence_status": status,
                            "rimone_subject_mapping_sha256": mapping,
                            "attempt": "B_multitask_VCDR",
                            "seed": seed,
                            "adaptive_papila_eye_level_auroc_TTA": forward_auroc,
                            "papila_patient_cluster_uncertainty": uncertainty(forward_auroc),
                            "vcdr_weight": 0.5,
                            "vcdr_supervised_on": "RIM-ONE only (dataset-provided mask-derived VCDR; historical auxiliary-label rule); HYGD masked out",
                            "attempt_A_was": 0.8251,
                        }
                    )
                )
                reverse_auroc = round(0.90 + seed / 100, 4)
                (namespace / f"reverse_seed{seed}.json").write_text(
                    json.dumps(
                        {
                            "_evidence_status": status,
                            "rimone_subject_mapping_sha256": mapping,
                            "attempt": "B_reverse_multitask_VCDR",
                            "seed": seed,
                            "held_out_dataset": "RIMONE",
                            "adaptive_rimone_image_level_auroc_TTA": reverse_auroc,
                            "rimone_patient_cluster_uncertainty": uncertainty(reverse_auroc),
                            "vcdr_weight": 0.5,
                            "vcdr_supervised_on": "PAPILA only (dataset-provided two-grader contour-derived VCDR; historical auxiliary-label rule); HYGD masked out",
                            "sources": ["HYGD", "PAPILA"],
                        }
                    )
                )
            with patch("builtins.print"):
                forward = aggregate_seeds.run([], root=root)
                reverse = aggregate_seeds_reverse.run([], root=root)
            self.assertEqual(forward["seeds"], list(range(5)))
            self.assertEqual(reverse["seeds"], list(range(5)))
            self.assertTrue((namespace / "seed_robustness_attemptB_patient_cluster_v2.json").is_file())
            self.assertTrue((namespace / "seed_robustness_reverse_patient_cluster_v2.json").is_file())
            with patch("builtins.print"), self.assertRaises(FileExistsError):
                aggregate_seeds.run([], root=root)
            with self.assertRaisesRegex(ValueError, "results/patient_aware"):
                aggregate_seeds.run(["--input-dir", "results/seed_runs"], root=root)

    def test_current_seed_aggregators_reject_schema_identity_and_claim_mutations(self):
        from validation import aggregate_seeds, aggregate_seeds_reverse
        from validation.seed_aggregation_utils import EXPECTED_EVIDENCE_STATUS

        def uncertainty(auroc):
            return {
                "auroc": auroc,
                "auroc_95ci": [auroc - 0.05, auroc + 0.05],
                "point_estimand_unit": "eye_or_image_row",
                "resampling_unit": "patient",
                "scope": "conditional_on_fixed_adaptive_predictions",
                "canonical_or_confirmatory": False,
                "seed": 42,
                "attempted_replicates": 2000,
                "valid_replicates": 2000,
            }

        def forward_payload(seed):
            auroc = round(0.80 + seed / 100, 4)
            return {
                "_evidence_status": dict(EXPECTED_EVIDENCE_STATUS),
                "rimone_subject_mapping_sha256": "a" * 64,
                "attempt": "B_multitask_VCDR",
                "seed": seed,
                "adaptive_papila_eye_level_auroc_TTA": auroc,
                "papila_patient_cluster_uncertainty": uncertainty(auroc),
                "vcdr_weight": 0.5,
                "vcdr_supervised_on": "RIM-ONE only (dataset-provided mask-derived VCDR; historical auxiliary-label rule); HYGD masked out",
                "attempt_A_was": 0.8251,
            }

        def reverse_payload(seed):
            auroc = round(0.90 + seed / 100, 4)
            return {
                "_evidence_status": dict(EXPECTED_EVIDENCE_STATUS),
                "rimone_subject_mapping_sha256": "a" * 64,
                "attempt": "B_reverse_multitask_VCDR",
                "seed": seed,
                "held_out_dataset": "RIMONE",
                "adaptive_rimone_image_level_auroc_TTA": auroc,
                "rimone_patient_cluster_uncertainty": uncertainty(auroc),
                "vcdr_weight": 0.5,
                "vcdr_supervised_on": "PAPILA only (dataset-provided two-grader contour-derived VCDR; historical auxiliary-label rule); HYGD masked out",
                "sources": ["HYGD", "PAPILA"],
            }

        def write_bundle(root, pattern, factory):
            paths = []
            for seed in range(5):
                path = root / pattern.format(seed=seed)
                path.write_text(json.dumps(factory(seed)))
                paths.append(path)
            return paths

        mutations = (
            (
                "extra contradictory claim",
                lambda payload: payload["_evidence_status"].update(
                    {"confirmatory_evidence": True}
                ),
                "adaptive evidence status",
            ),
            (
                "missing uncertainty",
                lambda payload: payload.pop("papila_patient_cluster_uncertainty"),
                "exact current schema",
            ),
            (
                "uppercase hash",
                lambda payload: payload.update(
                    {"rimone_subject_mapping_sha256": "A" * 64}
                ),
                "subject-map binding",
            ),
            (
                "contradictory nested claim",
                lambda payload: payload["papila_patient_cluster_uncertainty"].update(
                    {"canonical_or_confirmatory": True}
                ),
                "canonical_or_confirmatory",
            ),
        )
        for name, mutate, expected_error in mutations:
            with self.subTest(name=name), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                paths = write_bundle(root, "attemptB_seed{seed}.json", forward_payload)
                payload = json.loads(paths[0].read_text())
                mutate(payload)
                paths[0].write_text(json.dumps(payload))
                with self.assertRaisesRegex(ValueError, expected_error):
                    aggregate_seeds.summarize_runs(paths)

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            paths = write_bundle(root, "attemptB_seed{seed}.json", forward_payload)
            paths[0].rename(root / "attemptB_seed4_copy.json")
            paths[0] = root / "attemptB_seed4_copy.json"
            with self.assertRaisesRegex(ValueError, "filename"):
                aggregate_seeds.summarize_runs(paths)

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            paths = write_bundle(root, "reverse_seed{seed}.json", reverse_payload)
            payload = json.loads(paths[0].read_text())
            payload["rimone_patient_cluster_uncertainty"].pop("resampling_unit")
            paths[0].write_text(json.dumps(payload))
            with self.assertRaisesRegex(ValueError, "exact current schema"):
                aggregate_seeds_reverse.summarize_runs(paths)

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            paths = write_bundle(root, "reverse_seed{seed}.json", reverse_payload)
            payload = json.loads(paths[0].read_text())
            payload["sources"] = ["PAPILA", "HYGD"]
            paths[0].write_text(json.dumps(payload))
            with self.assertRaisesRegex(ValueError, "run-family"):
                aggregate_seeds_reverse.summarize_runs(paths)

        for pattern, factory, summarizer in (
            ("attemptB_seed{seed}.json", forward_payload, aggregate_seeds.summarize_runs),
            ("reverse_seed{seed}.json", reverse_payload, aggregate_seeds_reverse.summarize_runs),
        ):
            with tempfile.TemporaryDirectory() as temporary:
                paths = write_bundle(Path(temporary), pattern, factory)
                ordered = summarizer(paths)
                reversed_order = summarizer(list(reversed(paths)))
                self.assertEqual(
                    ordered["input_bundle_sha256"],
                    reversed_order["input_bundle_sha256"],
                )
                self.assertEqual(ordered["per_seed_auroc"], reversed_order["per_seed_auroc"])

    def test_private_outputs_are_confined_to_one_direct_namespace_file(self):
        from validation.evaluation_utils import resolve_private_output_path

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            expected = root / "results" / "patient_aware" / "summary.json"
            self.assertEqual(
                resolve_private_output_path(
                    root,
                    "results/patient_aware/summary.json",
                    "results/patient_aware",
                ),
                expected,
            )
            for requested in (
                "results/tracked.json",
                "results/patient_aware/nested/summary.json",
                "../escape.json",
            ):
                with self.subTest(requested=requested):
                    with self.assertRaisesRegex(ValueError, "namespace"):
                        resolve_private_output_path(
                            root, requested, "results/patient_aware"
                        )

            (root / "results").mkdir()
            outside = root / "outside"
            outside.mkdir()
            (root / "results" / "patient_aware").symlink_to(
                outside, target_is_directory=True
            )
            with self.assertRaisesRegex(ValueError, "symlinks"):
                resolve_private_output_path(
                    root,
                    "results/patient_aware/summary.json",
                    "results/patient_aware",
                )

    @unittest.skipUnless(TORCH_AVAILABLE, "PyTorch is an optional heavyweight dependency")
    def test_safe_checkpoint_round_trip_is_hash_bound_and_fresh(self):
        import torch

        from validation.evaluation_utils import (
            atomic_torch_save,
            load_torch_state_dict_safely,
        )

        with tempfile.TemporaryDirectory() as temporary:
            checkpoint = Path(temporary) / "weights.pt"
            original = {"weight": torch.tensor([1.0, 2.0])}
            digest = atomic_torch_save(checkpoint, original)
            loaded, observed = load_torch_state_dict_safely(
                checkpoint, expected_sha256=digest
            )
            self.assertEqual(observed, digest)
            self.assertTrue(torch.equal(loaded["weight"], original["weight"]))
            with self.assertRaises(FileExistsError):
                atomic_torch_save(checkpoint, original)
            with self.assertRaisesRegex(ValueError, "mismatch"):
                load_torch_state_dict_safely(
                    checkpoint, expected_sha256="0" * 64
                )

    @unittest.skipUnless(
        TORCH_AVAILABLE and TORCHVISION_AVAILABLE,
        "PyTorch and torchvision are optional heavyweight dependencies",
    )
    def test_full_checkpoint_architecture_never_requests_pretrained_weights(self):
        import torch.nn as nn

        from src.experiments import build_model

        dummy = nn.Module()
        dummy.layer4 = nn.Sequential(nn.Linear(4, 4))
        dummy.fc = nn.Linear(4, 1000)
        with patch("src.experiments.models.resnet18", return_value=dummy) as constructor:
            model = build_model(mode="finetune_layer4", pretrained=False)
        constructor.assert_called_once_with(weights=None)
        self.assertEqual(model.fc.out_features, 2)

    def test_external_label_frames_reject_fractional_labels_and_missing_groups(self):
        from validation.evaluation_utils import validate_external_label_frame

        valid = pd.DataFrame(
            {
                "image_path": ["one.png", "two.png"],
                "patient_id": ["p1", "p2"],
                "label": pd.Series([0, 1], dtype=object),
            }
        )
        normalized = validate_external_label_frame(valid, context="External")
        self.assertEqual(normalized["label"].tolist(), [0, 1])

        fractional = valid.copy()
        fractional.loc[0, "label"] = 0.5
        with self.assertRaisesRegex(ValueError, "binary integers"):
            validate_external_label_frame(fractional, context="External")

        missing_group = valid.copy()
        missing_group.loc[0, "patient_id"] = np.nan
        with self.assertRaisesRegex(ValueError, "patient_id"):
            validate_external_label_frame(missing_group, context="External")

        blank_image = valid.copy()
        blank_image.loc[0, "image_path"] = "  "
        with self.assertRaisesRegex(ValueError, "image_path"):
            validate_external_label_frame(blank_image, context="External")

    def test_terminal_result_claim_binds_scientific_content_without_a_hash_cycle(self):
        result = {
            "status": "complete",
            "artifacts": {"audit": "results/run_audit.json"},
            "internal_evaluation": {"primary": {"auroc": 0.9}},
            "artifact_sha256": {"audit": "a" * 64},
            "artifact_size_bytes": {"audit": 123},
        }
        baseline = terminal_result_claim_sha256(result)
        reordered = dict(reversed(list(result.items())))
        self.assertEqual(terminal_result_claim_sha256(reordered), baseline)

        scientific_mutation = copy.deepcopy(result)
        scientific_mutation["internal_evaluation"]["primary"]["auroc"] = 0.8
        self.assertNotEqual(
            terminal_result_claim_sha256(scientific_mutation), baseline
        )
        path_mutation = copy.deepcopy(result)
        path_mutation["artifacts"]["audit"] = "results/other_audit.json"
        self.assertNotEqual(terminal_result_claim_sha256(path_mutation), baseline)

        commitments_only = copy.deepcopy(result)
        commitments_only["artifact_sha256"]["audit"] = "b" * 64
        commitments_only["artifact_size_bytes"]["audit"] = 456
        self.assertEqual(terminal_result_claim_sha256(commitments_only), baseline)

    def test_oof_identity_fingerprint_binds_every_identity_field(self):
        from validation.evaluation_utils import oof_identity_fingerprint

        _, images, _, _ = self.canonical_oof_bundle()
        baseline = oof_identity_fingerprint(images)
        substitutions = {
            "image_name": "synthetic-substitute.jpg",
            "patient_id": "synthetic-substitute-patient",
            "evaluation_group": "synthetic-substitute-group",
            "sha256": "f" * 64,
            "label": 1 - int(images.loc[0, "label"]),
        }
        for column, replacement in substitutions.items():
            with self.subTest(column=column):
                mutated = images.copy()
                mutated.loc[0, column] = replacement
                self.assertNotEqual(baseline, oof_identity_fingerprint(mutated))

    def test_group_fold_fingerprint_is_order_independent_and_binds_every_field(self):
        from validation.evaluation_utils import group_fold_assignment_fingerprint

        _, _, _, group_folds = self.canonical_oof_bundle()
        baseline = group_fold_assignment_fingerprint(group_folds)
        self.assertEqual(
            baseline,
            group_fold_assignment_fingerprint(
                group_folds.sample(frac=1, random_state=17).reset_index(drop=True)
            ),
        )
        substitutions = {
            "evaluation_group": "synthetic-substitute-group",
            "label": 1 - int(group_folds.loc[0, "label"]),
            "n_images": int(group_folds.loc[0, "n_images"]) + 1,
            "outer_fold": (int(group_folds.loc[0, "outer_fold"]) + 1) % 5,
        }
        for column, replacement in substitutions.items():
            with self.subTest(column=column):
                mutated = group_folds.copy()
                mutated.loc[0, column] = replacement
                self.assertNotEqual(
                    baseline, group_fold_assignment_fingerprint(mutated)
                )

    def test_terminal_bundle_verifier_recomputes_every_artifact_hash(self):
        from validation.evaluation_utils import (
            group_fold_assignment_fingerprint,
            oof_identity_fingerprint,
            validate_terminal_bundle,
        )

        result, images, groups, group_folds = self.canonical_oof_bundle()
        identity_patcher = patch(
            "validation.evaluation_utils.EXPECTED_HYGD_DEDUPLICATED_OOF_IDENTITY_SHA256",
            oof_identity_fingerprint(images),
        )
        identity_patcher.start()
        self.addCleanup(identity_patcher.stop)
        fold_patcher = patch(
            "validation.evaluation_utils.EXPECTED_HYGD_CANONICAL_GROUP_FOLD_SHA256",
            group_fold_assignment_fingerprint(group_folds),
        )
        fold_patcher.start()
        self.addCleanup(fold_patcher.stop)

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for key, relative in result["artifacts"].items():
                path = root / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                if key == "audit":
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
                    path.write_text(
                        __import__("json").dumps(
                            {
                                "protocol_version": result["protocol_version"],
                                "status": "complete",
                                "dataset_identity": result["dataset_identity"],
                                "data_quality": result["data_quality"],
                                "split_summary": split_summary,
                                "group_fold_manifest": result["artifacts"][
                                    "group_folds"
                                ],
                                "completed_outer_folds": result["completed_outer_folds"],
                                "result_artifact": "results/internal_evaluation_repair.json",
                                "result_claim_sha256": terminal_result_claim_sha256(
                                    result
                                ),
                            }
                        )
                    )
                elif key == "group_folds":
                    group_folds.to_csv(path, index=False)
                elif key == "oof_images":
                    images.to_csv(path, index=False)
                elif key == "oof_groups":
                    groups.to_csv(path, index=False)
            result["artifact_sha256"] = {
                key: hashlib.sha256((root / relative).read_bytes()).hexdigest()
                for key, relative in result["artifacts"].items()
            }
            result["artifact_size_bytes"] = {
                key: (root / relative).stat().st_size
                for key, relative in result["artifacts"].items()
            }
            validate_terminal_bundle(result, root)

            audit_path = root / result["artifacts"]["audit"]
            original_audit = audit_path.read_bytes()
            original_audit_payload = __import__("json").loads(original_audit)
            invalid_audits = {
                "extra_key": {"unexpected_unbound_field": True},
                "protocol": {"protocol_version": "wrong"},
                "data_quality": {
                    "data_quality": {
                        **result["data_quality"],
                        "input_images": 746,
                    }
                },
                "split_summary": {
                    "split_summary": original_audit_payload["split_summary"][:-1]
                },
                "manifest_path": {
                    "group_fold_manifest": "results/wrong_group_folds.csv"
                },
                "result_path": {"result_artifact": "results/wrong.json"},
                "result_claim": {"result_claim_sha256": "0" * 64},
                "completed_folds": {"completed_outer_folds": [0, 1, 2, 3]},
            }
            for name, replacement in invalid_audits.items():
                with self.subTest(invalid_audit=name):
                    invalid_audit = copy.deepcopy(original_audit_payload)
                    invalid_audit.update(replacement)
                    audit_path.write_text(__import__("json").dumps(invalid_audit))
                    result["artifact_sha256"]["audit"] = hashlib.sha256(
                        audit_path.read_bytes()
                    ).hexdigest()
                    result["artifact_size_bytes"]["audit"] = audit_path.stat().st_size
                    with self.assertRaisesRegex(ValueError, "audit"):
                        validate_terminal_bundle(result, root)
            audit_path.write_bytes(original_audit)
            result["artifact_sha256"]["audit"] = hashlib.sha256(
                original_audit
            ).hexdigest()
            result["artifact_size_bytes"]["audit"] = len(original_audit)

            original_fold_artifacts = {}
            for role in ("oof_images", "oof_groups", "group_folds"):
                path = root / result["artifacts"][role]
                original_fold_artifacts[role] = path.read_bytes()
                reassigned = pd.read_csv(path)
                reassigned["outer_fold"] = reassigned["outer_fold"].replace(
                    {0: 1, 1: 0}
                )
                reassigned.to_csv(path, index=False)
                result["artifact_sha256"][role] = hashlib.sha256(
                    path.read_bytes()
                ).hexdigest()
                result["artifact_size_bytes"][role] = path.stat().st_size
            with self.assertRaisesRegex(ValueError, "canonical group-fold"):
                validate_terminal_bundle(result, root)
            for role, content in original_fold_artifacts.items():
                path = root / result["artifacts"][role]
                path.write_bytes(content)
                result["artifact_sha256"][role] = hashlib.sha256(content).hexdigest()
                result["artifact_size_bytes"][role] = len(content)

            image_path = root / result["artifacts"]["oof_images"]
            original_images = image_path.read_bytes()
            substituted_identity = pd.read_csv(image_path)
            substituted_identity.loc[0, "image_name"] = "substituted-image.jpg"
            substituted_identity.loc[0, "patient_id"] = "substituted-patient"
            substituted_identity.loc[0, "sha256"] = "f" * 64
            substituted_identity.to_csv(image_path, index=False)
            result["artifact_sha256"]["oof_images"] = hashlib.sha256(
                image_path.read_bytes()
            ).hexdigest()
            result["artifact_size_bytes"]["oof_images"] = image_path.stat().st_size
            with self.assertRaisesRegex(ValueError, "OOF identity"):
                validate_terminal_bundle(result, root)
            image_path.write_bytes(original_images)
            result["artifact_sha256"]["oof_images"] = hashlib.sha256(
                original_images
            ).hexdigest()
            result["artifact_size_bytes"]["oof_images"] = len(original_images)

            invalid_headline = copy.deepcopy(result)
            invalid_headline["internal_evaluation"]["primary"]["sensitivity"] = 0.5
            with self.assertRaisesRegex(ValueError, "metric|report|OOF|inconsistent"):
                validate_terminal_bundle(invalid_headline, root)

            mutated_images = pd.read_csv(image_path)
            mutated_images.loc[0, "probability"] = 0.95
            mutated_images.loc[0, "predicted_class"] = 1
            mutated_images.to_csv(image_path, index=False)
            result["artifact_sha256"]["oof_images"] = hashlib.sha256(
                image_path.read_bytes()
            ).hexdigest()
            result["artifact_size_bytes"]["oof_images"] = image_path.stat().st_size
            with self.assertRaisesRegex(ValueError, "metric|report|OOF|inconsistent"):
                validate_terminal_bundle(result, root)
            image_path.write_bytes(original_images)
            result["artifact_sha256"]["oof_images"] = hashlib.sha256(
                original_images
            ).hexdigest()
            result["artifact_size_bytes"]["oof_images"] = len(original_images)

            group_path = root / result["artifacts"]["oof_groups"]
            original_groups = group_path.read_bytes()
            group_path.write_text("tampered")
            with self.assertRaisesRegex(ValueError, "size mismatch|hash mismatch"):
                validate_terminal_bundle(result, root)
            group_path.write_bytes(original_groups)

            result["artifact_sha256"]["oof_groups"] = "not-a-sha256"
            with self.assertRaisesRegex(ValueError, "64-hex"):
                validate_terminal_bundle(result, root)
            result["artifact_sha256"]["oof_groups"] = hashlib.sha256(
                original_groups
            ).hexdigest()
            result["artifact_size_bytes"]["oof_groups"] = len(original_groups)

            linked_artifact = group_path
            outside = root / "outside.csv"
            linked_artifact.unlink()
            outside.write_bytes(original_groups)
            linked_artifact.hardlink_to(outside)
            result["artifact_sha256"]["oof_groups"] = hashlib.sha256(
                linked_artifact.read_bytes()
            ).hexdigest()
            result["artifact_size_bytes"]["oof_groups"] = linked_artifact.stat().st_size
            with self.assertRaisesRegex(ValueError, "single-link"):
                validate_terminal_bundle(result, root)

            linked_artifact.unlink()
            linked_artifact.symlink_to(outside)
            with self.assertRaisesRegex(ValueError, "missing or unsafe"):
                validate_terminal_bundle(result, root)

    def test_terminal_verifier_and_result_reader_reject_links(self):
        import json

        from validation.evaluation_utils import read_confined_json

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            results = root / "results"
            results.mkdir()
            outside = root / "outside.json"
            outside.write_text(json.dumps({"private": True}))

            symlink = results / "marker.json"
            symlink.symlink_to(outside)
            with self.assertRaisesRegex(ValueError, "missing or unsafe"):
                read_confined_json(root, "results/marker.json")

            symlink.unlink()
            hardlink = results / "marker.json"
            hardlink.hardlink_to(outside)
            with self.assertRaisesRegex(ValueError, "single-link"):
                read_confined_json(root, "results/marker.json")

            hardlink.unlink()
            outside.unlink()
            hardlink.write_text(json.dumps({"status": "ok"}))
            self.assertEqual(
                read_confined_json(root, "results/marker.json"),
                {"status": "ok"},
            )

    def test_console_data_quality_summary_redacts_hashes_names_and_patient_ids(self):
        from validation.evaluation_utils import console_data_quality_summary

        summary = console_data_quality_summary(
            {
                "input_images": 3,
                "unique_image_hashes": 2,
                "removed_exact_duplicate_rows": 1,
                "duplicate_hash_groups": 1,
                "cross_patient_duplicate_hash_groups": 1,
                "input_patient_ids": 2,
                "independent_evaluation_groups": 1,
                "label_conflicts": 0,
                "missing_images": 0,
                "duplicate_groups": [
                    {
                        "sha256": "secret-hash",
                        "images": ["patient-image.jpg"],
                        "patient_ids": ["private-patient"],
                    }
                ],
            }
        )
        serialized = repr(summary)
        self.assertEqual(summary["duplicate_hash_groups"], 1)
        self.assertNotIn("secret-hash", serialized)
        self.assertNotIn("patient-image.jpg", serialized)
        self.assertNotIn("private-patient", serialized)

    def test_canonical_runner_uses_checkpoint_not_model_selection_key(self):
        source = (ROOT / "validation" / "internal_evaluation_repair.py").read_text()
        self.assertIn('"checkpoint_selection": "minimum inner-validation loss"', source)
        self.assertNotIn('"model_selection": "minimum inner-validation loss"', source)
        self.assertIn("assert_fresh_output_bundle(bundle_paths)", source)
        self.assertIn("expected_labels_sha256=EXPECTED_HYGD_DATASET_IDENTITY", source)
        self.assertIn('source_metadata.attrs["labels_csv_sha256"]', source)
        self.assertIn("require_verified_sha256=True", source)
        self.assertNotIn('sha256_file(raw_dir / "Labels.csv")', source)
        self.assertLess(
            source.index("write_json(audit_path, audit, overwrite=True)"),
            source.index("write_json(result_path, result)"),
        )
        self.assertLess(
            source.index("validate_canonical_data_quality(data_quality)"),
            source.index("write_csv(fold_path, group_folds)"),
        )
        preflight = source[
            source.index("dataset_identity = build_dataset_identity") :
            source.index("group_folds = make_outer_folds")
        ]
        self.assertNotIn(
            "if complete:",
            preflight,
            "Dataset identity must fail closed for audit and smoke routes too",
        )
        self.assertLess(
            source.index("if not complete:", source.index("outer_probabilities =")),
            source.index('outer_test["probability"] = outer_probabilities'),
        )
        smoke_branch = source[
            source.index("if not complete:", source.index("outer_probabilities =")) :
            source.index('outer_test["probability"] = outer_probabilities')
        ]
        self.assertNotIn("frame_metrics", smoke_branch)
        self.assertNotIn("oof_frames.append", smoke_branch)
        self.assertLess(
            source.index("post_dataset_identity = build_dataset_identity"),
            source.index("write_csv(image_output_path"),
        )

    def test_per_fold_progress_withholds_outer_outcomes_until_terminal_publication(self):
        from validation.evaluation_utils import fold_completion_message

        message = fold_completion_message(completed_fold=0, total_folds=5).lower()
        for forbidden in (
            "auroc",
            "sensitivity",
            "specificity",
            "probability",
            "threshold",
            "confusion",
        ):
            self.assertNotIn(forbidden, message)
        self.assertIn("terminal bundle publication", message)

        source = (ROOT / "validation" / "internal_evaluation_repair.py").read_text()
        self.assertIn("fold_completion_message(int(fold), args.folds)", source)
        self.assertNotIn("finalized fold {fold + 1}: image AUROC", source)

    def test_partial_smoke_report_is_outcome_blind_and_exact_schema(self):
        from validation.evaluation_utils import (
            build_smoke_diagnostics,
            validate_smoke_prediction_execution,
            validate_smoke_result_payload,
        )

        execution = validate_smoke_prediction_execution(
            np.array([0.1, 0.9, 0.4]), expected_rows=3
        )
        diagnostic = build_smoke_diagnostics(
            fold_execution=[
                {
                    "outer_fold": 0,
                    "training_loop_completed": True,
                    "checkpoint_restore_completed": True,
                    "inner_selection_path_completed": True,
                    **execution,
                }
            ],
            completed_outer_folds=[0],
            total_outer_folds=5,
        )
        result = {
            "protocol_version": "2026-09-04-v5",
            "status": "partial_smoke_run",
            "canonical_internal_estimate": False,
            "completed_outer_folds": [0],
            "planned_outer_folds": 5,
            "smoke_diagnostics": diagnostic,
            "artifacts": {
                "audit": "results/internal_evaluation_repair_smoke_1of5_audit.json",
                "group_folds": (
                    "results/internal_evaluation_repair_smoke_1of5_group_folds.csv"
                ),
            },
            "runtime_seconds": 1.0,
        }
        validate_smoke_result_payload(result)
        custom_prefix = copy.deepcopy(result)
        custom_prefix["artifacts"] = {
            "audit": (
                "results/internal_evaluation_repair_20260831a_smoke_1of5_audit.json"
            ),
            "group_folds": (
                "results/internal_evaluation_repair_20260831a_smoke_1of5_group_folds.csv"
            ),
        }
        validate_smoke_result_payload(custom_prefix)
        self.assertFalse(diagnostic["canonical_internal_estimate"])
        self.assertFalse(diagnostic["outcomes_emitted"])
        serialized = repr(diagnostic).lower()
        for forbidden in (
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
        ):
            self.assertNotIn(forbidden, serialized)

        for forbidden in ("auroc", "probability", "threshold", "oof_predictions"):
            injected = copy.deepcopy(result)
            injected["smoke_diagnostics"][forbidden] = 0.9
            with self.assertRaisesRegex(ValueError, "outcome-blind"):
                validate_smoke_result_payload(injected)

        hidden_score = copy.deepcopy(result)
        hidden_score["smoke_diagnostics"]["fold_execution"][0]["score"] = 0.99
        with self.assertRaisesRegex(ValueError, "exact outcome-blind schema"):
            validate_smoke_result_payload(hidden_score)

        wrong_fold_type = copy.deepcopy(result)
        wrong_fold_type["planned_outer_folds"] = 5.0
        with self.assertRaisesRegex(ValueError, "integer fold metadata"):
            validate_smoke_result_payload(wrong_fold_type)

        nonfinite_runtime = copy.deepcopy(result)
        nonfinite_runtime["runtime_seconds"] = float("nan")
        with self.assertRaisesRegex(ValueError, "runtime_seconds"):
            validate_smoke_result_payload(nonfinite_runtime)

        for invalid in (
            (np.array([0.1, np.nan]), 2),
            (np.array([0.1, 1.1]), 2),
            (np.array([0.1]), 2),
        ):
            with self.assertRaisesRegex(ValueError, "smoke predictions"):
                validate_smoke_prediction_execution(invalid[0], invalid[1])

    def test_terminal_smoke_bundle_requires_and_verifies_committed_artifacts(self):
        import json

        from validation.evaluation_utils import (
            build_smoke_diagnostics,
            group_fold_assignment_fingerprint,
            validate_terminal_smoke_bundle,
        )

        diagnostic = build_smoke_diagnostics(
            fold_execution=[
                {
                    "outer_fold": 0,
                    "training_loop_completed": True,
                    "checkpoint_restore_completed": True,
                    "inner_selection_path_completed": True,
                    "outer_inference_path_completed": True,
                    "outer_inference_rows": 148,
                    "outer_inference_values_validated": True,
                }
            ],
            completed_outer_folds=[0],
            total_outer_folds=5,
        )
        result = {
            "protocol_version": "2026-09-04-v5",
            "status": "partial_smoke_run",
            "canonical_internal_estimate": False,
            "completed_outer_folds": [0],
            "planned_outer_folds": 5,
            "smoke_diagnostics": diagnostic,
            "artifacts": {
                "audit": "results/internal_evaluation_repair_smoke_1of5_audit.json",
                "group_folds": (
                    "results/internal_evaluation_repair_smoke_1of5_group_folds.csv"
                ),
            },
            "runtime_seconds": 1.0,
        }

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            results = root / "results"
            results.mkdir()
            _, _, _, group_frame = self.canonical_oof_bundle()
            fold_patcher = patch(
                "validation.evaluation_utils.EXPECTED_HYGD_CANONICAL_GROUP_FOLD_SHA256",
                group_fold_assignment_fingerprint(group_frame),
            )
            fold_patcher.start()
            self.addCleanup(fold_patcher.stop)
            group_path = root / result["artifacts"]["group_folds"]
            group_frame.to_csv(group_path, index=False)
            split_summary = [
                {
                    "outer_fold": int(fold),
                    "groups": int(len(rows)),
                    "images": int(rows["n_images"].sum()),
                    "positive_groups": int(rows["label"].sum()),
                    "negative_groups": int((1 - rows["label"]).sum()),
                }
                for fold, rows in group_frame.groupby("outer_fold")
            ]
            canonical = self.canonical_result(
                build_primary_report(
                    outer_fold_auc={
                        "fold_aurocs": [],
                        "unweighted_mean_auroc": 1.0,
                        "group_count_weighted_mean_auroc": 1.0,
                    },
                    fold_stratified_ci={
                        "auroc_95ci": [1.0, 1.0],
                        "sensitivity_95ci": [1.0, 1.0],
                        "specificity_95ci": [1.0, 1.0],
                        "resampling_unit": "evaluation_group_within_outer_fold",
                        "fold_weighting": "equal",
                        "seed": 20_269_714,
                        "n_bootstrap": 5_000,
                        "valid_replicates": 5_000,
                    },
                    pooled_group_metrics={
                        "n": 283,
                        "positives": int(group_frame["label"].sum()),
                        "negatives": int((1 - group_frame["label"]).sum()),
                        "auroc": 1.0,
                        "sensitivity": 1.0,
                        "specificity": 1.0,
                        "confusion_tn_fp_fn_tp": [
                            int((1 - group_frame["label"]).sum()),
                            0,
                            0,
                            int(group_frame["label"].sum()),
                        ],
                    },
                    pooled_group_cluster_ci={
                        "auroc_95ci": [1.0, 1.0],
                        "sensitivity_95ci": [1.0, 1.0],
                        "specificity_95ci": [1.0, 1.0],
                    },
                    image_metrics={
                        "n": 737,
                        "positives": 367,
                        "negatives": 370,
                        "auroc": 1.0,
                        "sensitivity": 1.0,
                        "specificity": 1.0,
                        "confusion_tn_fp_fn_tp": [370, 0, 0, 367],
                    },
                    image_cluster_ci={
                        "auroc_95ci": [1.0, 1.0],
                        "sensitivity_95ci": [1.0, 1.0],
                        "specificity_95ci": [1.0, 1.0],
                    },
                )
            )
            audit = {
                "protocol_version": "2026-09-04-v5",
                "status": "partial_smoke_run",
                "dataset_identity": canonical["dataset_identity"],
                "data_quality": canonical["data_quality"],
                "split_summary": split_summary,
                "group_fold_manifest": result["artifacts"]["group_folds"],
                "result_artifact": "results/internal_evaluation_repair_smoke_1of5.json",
                "result_claim_sha256": terminal_result_claim_sha256(result),
                "completed_outer_folds": [0],
            }
            audit_path = root / result["artifacts"]["audit"]
            audit_path.write_text(json.dumps(audit))

            with self.assertRaisesRegex(ValueError, "hash|size|commitment"):
                validate_terminal_smoke_bundle(result, root)

            result["artifact_sha256"] = {
                key: hashlib.sha256((root / relative).read_bytes()).hexdigest()
                for key, relative in result["artifacts"].items()
            }
            result["artifact_size_bytes"] = {
                key: (root / relative).stat().st_size
                for key, relative in result["artifacts"].items()
            }
            validate_terminal_smoke_bundle(result, root)

            original = group_path.read_bytes()
            group_path.write_bytes(original + b"\n")
            with self.assertRaisesRegex(ValueError, "size mismatch|hash mismatch"):
                validate_terminal_smoke_bundle(result, root)
            group_path.write_bytes(original)

            outside = root / "outside.csv"
            outside.write_bytes(original)
            group_path.unlink()
            group_path.hardlink_to(outside)
            with self.assertRaisesRegex(ValueError, "single-link"):
                validate_terminal_smoke_bundle(result, root)

            group_path.unlink()
            group_path.symlink_to(outside)
            with self.assertRaisesRegex(ValueError, "missing or unsafe"):
                validate_terminal_smoke_bundle(result, root)

    def test_only_locked_defaults_can_emit_a_complete_preferred_result(self):
        from validation.evaluation_utils import classify_run_mode

        defaults = {
            "folds": 5,
            "epochs": 10,
            "batch_size": 32,
            "seed": 20_260_711,
            "inner_val_fraction": 0.15,
            "target_sensitivity": 0.95,
            "bootstrap": 5_000,
            "max_folds": None,
            "audit_only": False,
        }
        self.assertEqual(classify_run_mode(**defaults), "complete")

        for changed in ({"folds": 2}, {"epochs": 1}, {"bootstrap": 10}):
            candidate = {**defaults, **changed}
            with self.assertRaisesRegex(ValueError, "configuration is locked"):
                classify_run_mode(**candidate)

        smoke = {
            **defaults,
            "epochs": 1,
            "bootstrap": 10,
            "max_folds": 1,
        }
        self.assertEqual(classify_run_mode(**smoke), "partial_smoke_run")
        with self.assertRaisesRegex(ValueError, "proper subset"):
            classify_run_mode(**{**defaults, "max_folds": 5})

        audit = {
            **defaults,
            "folds": 2,
            "epochs": 1,
            "bootstrap": 10,
            "audit_only": True,
        }
        self.assertEqual(classify_run_mode(**audit), "audit_only")
        with self.assertRaisesRegex(ValueError, "audit_only"):
            classify_run_mode(**{**audit, "max_folds": 1})

    def test_oof_validation_rejects_duplicate_hashes_and_incomplete_runs(self):
        metadata = pd.DataFrame(
            {
                "sha256": ["a", "b", "c"],
                "evaluation_group": ["g1", "g2", "g3"],
                "outer_fold": [0, 1, 2],
            }
        )
        valid = metadata.copy()
        validate_oof_predictions(valid, metadata, complete=True)

        duplicated = pd.concat([valid, valid.iloc[[0]]], ignore_index=True)
        with self.assertRaisesRegex(AssertionError, "duplicate hashes"):
            validate_oof_predictions(duplicated, metadata, complete=True)

        with self.assertRaisesRegex(AssertionError, "incomplete"):
            validate_oof_predictions(valid.iloc[:2], metadata, complete=True)

        wrong_group = valid.copy()
        wrong_group.loc[0, "evaluation_group"] = "leaked-group"
        with self.assertRaisesRegex(AssertionError, "metadata mismatch"):
            validate_oof_predictions(wrong_group, metadata, complete=True)

        wrong_fold = valid.copy()
        wrong_fold.loc[0, "outer_fold"] = 2
        with self.assertRaisesRegex(AssertionError, "metadata mismatch"):
            validate_oof_predictions(wrong_fold, metadata, complete=True)

    def test_successful_run_terminalizes_the_audit_record(self):
        running = {"protocol_version": "test", "status": "running"}
        finished = finalize_audit(
            running,
            result_status="complete",
            result_path="results/internal_evaluation_repair.json",
            completed_outer_folds=[0, 1, 2, 3, 4],
            result_claim_sha256="a" * 64,
        )
        self.assertEqual(running["status"], "running")
        self.assertEqual(finished["status"], "complete")
        self.assertEqual(finished["completed_outer_folds"], [0, 1, 2, 3, 4])
        self.assertEqual(
            finished["result_artifact"],
            "results/internal_evaluation_repair.json",
        )
        self.assertEqual(finished["result_claim_sha256"], "a" * 64)


if __name__ == "__main__":
    unittest.main()
