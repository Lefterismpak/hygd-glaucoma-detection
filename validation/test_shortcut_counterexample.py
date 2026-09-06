"""A constructed counterexample must distinguish encoding from model reliance."""
import unittest


class ShortcutCounterexampleTests(unittest.TestCase):
    def test_identical_source_encoding_does_not_imply_identical_shift_behavior(self):
        from examples.shortcut_counterexample import run_counterexample
        result = run_counterexample()
        self.assertEqual(result["evidence_type"], "synthetic_counterexample_not_medical_evidence")
        for environment in ["development", "reversed_source_association"]:
            self.assertEqual(result[environment]["source_decoding_accuracy"], 1.0)
            self.assertGreater(result[environment]["disease_head_auroc"], 0.85)
        self.assertGreater(result["development"]["source_head_auroc"], 0.9)
        self.assertLess(result["reversed_source_association"]["source_head_auroc"], 0.1)

    def test_counterexample_is_reproducible_and_contains_no_private_input(self):
        from examples.shortcut_counterexample import run_counterexample
        self.assertEqual(run_counterexample(), run_counterexample())
        result = run_counterexample(seed=7)
        self.assertFalse(result["medical_data_used"])
        self.assertFalse(result["models_trained"])
        self.assertEqual(result["samples_per_environment"], 2000)


if __name__ == "__main__":
    unittest.main()
