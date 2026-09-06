"""Hand-checkable tests for descriptive diagnostics and block randomization."""
import json
import unittest

import numpy as np
import pandas as pd


class CalibrationDiagnosticsTests(unittest.TestCase):
    def fixture(self, repeats=1):
        return pd.DataFrame({
            "evaluation_group": [f"private-{i}" for i in range(4 * repeats)],
            "label": [0, 0, 1, 1] * repeats,
            "probability": [0.1, 0.2, 0.8, 0.9] * repeats,
            "selected_threshold": [0.85] * (4 * repeats),
            "outer_fold": [0] * (4 * repeats),
        })

    def test_scores_and_operating_points_are_computed_from_group_rows(self):
        from validation.calibration_diagnostics import describe_group_predictions
        result = describe_group_predictions(self.fixture())
        self.assertAlmostEqual(result["brier_score"], 0.025)
        self.assertAlmostEqual(result["log_loss"], 0.164252033486018)
        self.assertEqual(result["observed_fraction"], 0.5)
        self.assertEqual(result["mean_predicted_risk"], 0.5)
        self.assertEqual(result["operating_points"]["recorded_inner_thresholds"]["confusion_matrix"], [[2, 0], [1, 1]])
        self.assertEqual(result["operating_points"]["fixed_0_5_descriptive"]["confusion_matrix"], [[2, 0], [0, 2]])

    def test_sparse_bins_suppress_outcomes_and_scores(self):
        from validation.calibration_diagnostics import describe_group_predictions
        result = describe_group_predictions(self.fixture())
        for cell in result["reliability_bins"]:
            self.assertTrue(cell["suppressed"])
            self.assertNotIn("observed_fraction", cell)
            self.assertNotIn("mean_predicted_risk", cell)
            self.assertNotIn("groups", cell)

    def test_threshold_equality_is_positive_and_inconsistent_fold_thresholds_fail(self):
        from validation.calibration_diagnostics import describe_group_predictions
        frame = self.fixture()
        frame["selected_threshold"] = 0.8
        result = describe_group_predictions(frame)
        self.assertEqual(result["operating_points"]["recorded_inner_thresholds"]["confusion_matrix"], [[2, 0], [0, 2]])
        frame.loc[0, "selected_threshold"] = 0.7
        with self.assertRaises(ValueError):
            describe_group_predictions(frame)

    def test_bins_include_probability_one_and_do_not_expose_identifiers(self):
        from validation.calibration_diagnostics import describe_group_predictions
        frame = self.fixture(10)
        frame.loc[frame.probability == 0.9, "probability"] = 1.0
        result = describe_group_predictions(frame)
        self.assertEqual(sum(cell.get("groups", 0) for cell in result["reliability_bins"]), 40)
        self.assertNotIn("private-", json.dumps(result))
        self.assertTrue(np.isfinite(result["log_loss"]))

    def test_invalid_rows_and_duplicate_evaluation_groups_fail(self):
        from validation.calibration_diagnostics import describe_group_predictions
        cases = [("label", 0.5), ("label", np.nan), ("probability", np.inf),
                 ("probability", -0.1), ("selected_threshold", 1.1),
                 ("outer_fold", 0.5), ("outer_fold", -1), ("evaluation_group", None)]
        for column, value in cases:
            frame = self.fixture().astype({column: object})
            frame.loc[0, column] = value
            with self.subTest(column=column, value=value), self.assertRaises(ValueError):
                describe_group_predictions(frame)
        frame = self.fixture()
        frame.loc[1, "evaluation_group"] = frame.loc[0, "evaluation_group"]
        with self.assertRaises(ValueError):
            describe_group_predictions(frame)

    def test_diagnostic_does_not_select_a_new_threshold_or_recalibrate(self):
        from validation.calibration_diagnostics import describe_group_predictions
        frame = self.fixture()
        before = frame.copy(deep=True)
        report = describe_group_predictions(frame)
        pd.testing.assert_frame_equal(frame, before)
        self.assertFalse(report["used_to_select_model_or_threshold"])
        self.assertEqual(report["status"], "post_development_descriptive_only")
        self.assertEqual(report["recorded_threshold_range"], [0.85, 0.85])


class BlockPermutationTests(unittest.TestCase):
    def fixture(self):
        # Paired labels differ by anatomical slot; a legal block move preserves
        # the complete ordered vector. Source B has a different label margin.
        return pd.DataFrame({
            "source": ["A"] * 8 + ["B"] * 4,
            "group": ["a", "a", "b", "b", "c", "c", "d", "d", "e", "e", "f", "f"],
            "slot": ["left", "right"] * 6,
            "label": [0, 0, 1, 1, 0, 1, 1, 0, 0, 0, 0, 0],
        })

    def test_complete_ordered_vectors_and_source_margins_are_preserved(self):
        from validation.permutation_controls import permute_label_blocks
        frame = self.fixture()
        labels = permute_label_blocks(frame, seed=42)
        shuffled = frame.assign(label=labels)
        for source in ["A", "B"]:
            original = frame[frame.source == source]
            permuted = shuffled[shuffled.source == source]
            vectors = lambda f: sorted(tuple(g.sort_values("slot").label) for _, g in f.groupby("group"))
            self.assertEqual(vectors(original), vectors(permuted))
            self.assertEqual(int(original.label.sum()), int(permuted.label.sum()))
        self.assertFalse(np.array_equal(labels[:8], frame.label.to_numpy()[:8]))

    def test_row_order_invariance_and_input_immutability(self):
        from validation.permutation_controls import permute_label_blocks
        frame = self.fixture()
        before = frame.copy(deep=True)
        original = permute_label_blocks(frame, seed=21)
        reordered = frame.sample(frac=1, random_state=8)
        restored = pd.Series(permute_label_blocks(reordered, seed=21), index=reordered.index).sort_index().to_numpy()
        np.testing.assert_array_equal(original, restored)
        pd.testing.assert_frame_equal(frame, before)

    def test_different_slot_signatures_never_exchange(self):
        from validation.permutation_controls import permute_label_blocks
        frame = pd.DataFrame({"source": ["A"] * 4, "group": ["a", "a", "b", "c"],
                              "slot": ["left", "right", "center", "center"], "label": [1, 1, 0, 0]})
        for seed in range(10):
            np.testing.assert_array_equal(permute_label_blocks(frame, seed=seed), [1, 1, 0, 0])

    def test_ambiguous_or_invalid_block_contract_fails(self):
        from validation.permutation_controls import permute_label_blocks
        for column, value in [("label", 0.5), ("source", None), ("group", ""), ("slot", None)]:
            frame = self.fixture().astype({column: object})
            frame.loc[0, column] = value
            with self.subTest(column=column), self.assertRaises(ValueError):
                permute_label_blocks(frame, seed=0)
        frame = self.fixture()
        frame.loc[1, "slot"] = "left"
        with self.assertRaises(ValueError):
            permute_label_blocks(frame, seed=0)
        frame = self.fixture()
        frame.loc[0, "source"] = "B"
        with self.assertRaises(ValueError):
            permute_label_blocks(frame, seed=0)

    def test_degenerate_no_exchangeable_block_fails(self):
        from validation.permutation_controls import permute_label_blocks
        frame = self.fixture().iloc[:2]
        with self.assertRaises(ValueError):
            permute_label_blocks(frame, seed=0)


if __name__ == "__main__":
    unittest.main()
