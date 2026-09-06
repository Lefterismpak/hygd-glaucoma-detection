"""Geometry diagnostics must not be presented as label-permutation inference."""
import unittest
import numpy as np


class RandomDirectionTests(unittest.TestCase):
    def test_constant_representation_has_only_chance_directions(self):
        from validation.random_direction_diagnostic import describe_random_directions
        result = describe_random_directions(np.ones((20, 4)), np.tile([0, 1], 10))
        self.assertEqual(result["auroc_quantiles"], [0.5] * 5)
        self.assertEqual(result["fraction_auroc_at_least_0_75"], 0)
        self.assertFalse(result["fractions_are_p_values"])

    def test_one_disease_axis_can_give_high_auc_without_label_fitting(self):
        from validation.random_direction_diagnostic import describe_random_directions
        y = np.tile([0, 1], 100)
        result = describe_random_directions((2 * y - 1).reshape(-1, 1), y)
        self.assertGreater(result["fraction_auroc_at_least_0_75"], 0.4)
        self.assertLess(result["fraction_auroc_at_least_0_75"], 0.6)
        self.assertEqual(result["auroc_quantiles"][0], 0)
        self.assertEqual(result["auroc_quantiles"][-1], 1)
        self.assertFalse(result["label_based_direction_selection"])

    def test_row_order_and_global_feature_offset_do_not_change_auc_distribution(self):
        from validation.random_direction_diagnostic import describe_random_directions
        rng = np.random.default_rng(13)
        x, y = rng.normal(size=(30, 5)), np.tile([0, 1], 15)
        result = describe_random_directions(x, y)
        reverse = describe_random_directions(x[::-1] + 5, y[::-1])
        self.assertEqual(result, reverse)

    def test_bad_shapes_nonfinite_and_nonbinary_values_fail(self):
        from validation.random_direction_diagnostic import describe_random_directions
        for x, y in [(np.ones((4, 2)), [0, 0, 0, 0]),
                     (np.ones((4, 2)), [0, 1, 0.5, 1]),
                     (np.ones((4, 2)), [0, 1]),
                     (np.full((4, 2), np.nan), [0, 0, 1, 1]),
                     (np.ones(4), [0, 0, 1, 1])]:
            with self.assertRaises(ValueError):
                describe_random_directions(x, y)


if __name__ == "__main__":
    unittest.main()
