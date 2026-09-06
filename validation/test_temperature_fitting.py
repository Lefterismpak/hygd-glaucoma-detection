"""Proper, positive temperature fitting on synthetic calibration data only."""
import unittest
import numpy as np


class TemperatureFitTests(unittest.TestCase):
    def test_known_legacy_clamp_failure_reaches_finite_likelihood_optimum(self):
        from validation.calibration_fitting import fit_positive_temperature, bernoulli_nll
        rng = np.random.default_rng(10)
        logits = rng.normal(size=20) * 3
        labels = (logits + rng.normal(size=20) * 2 > 0).astype(int)
        temperature = fit_positive_temperature(logits, labels)
        self.assertGreater(temperature, 0.42)
        self.assertLess(temperature, 0.44)
        self.assertLess(bernoulli_nll(logits, labels, temperature), 0.196)
        self.assertGreater(bernoulli_nll(logits, labels, 0.001), 45)

    def test_fit_is_positive_bounded_and_never_worse_than_unscaled_calibration_loss(self):
        from validation.calibration_fitting import fit_positive_temperature, bernoulli_nll
        for labels in [np.array([0, 0, 1, 1]), np.array([1, 1, 0, 0])]:
            logits = np.array([-3., -1., 1., 3.])
            temperature = fit_positive_temperature(logits, labels)
            self.assertTrue(0.001 <= temperature <= 1000)
            self.assertLessEqual(bernoulli_nll(logits, labels, temperature), bernoulli_nll(logits, labels, 1))

    def test_malformed_or_one_class_calibration_is_rejected(self):
        from validation.calibration_fitting import fit_positive_temperature
        for logits, labels in [([1., np.nan], [0, 1]), ([1, 2], [0, 0]),
                               ([1, 2], [0, .5]), ([1, 2], [0]), ([], [])]:
            with self.assertRaises(ValueError): fit_positive_temperature(logits, labels)


if __name__ == "__main__":
    unittest.main()
