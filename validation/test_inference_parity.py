"""Verify inference bytes and batching using synthetic images and a tiny model."""
import hashlib
import importlib.util
import tempfile
import unittest
from pathlib import Path


@unittest.skipUnless(importlib.util.find_spec("torch") is not None, "optional Torch inference tests")
class InferenceParityTests(unittest.TestCase):
    def model(self):
        import torch
        class MeanModel(torch.nn.Module):
            def forward(self, images):
                score = images.mean((1, 2, 3))
                return torch.stack((-score, score), dim=1)
        return MeanModel().eval()

    def image(self, root, color=(20, 90, 170)):
        from PIL import Image
        path = root / "synthetic.png"
        Image.new("RGB", (19, 13), color).save(path)
        return path

    def test_single_image_rejects_symlink_and_stale_expected_hash(self):
        from validation.predict import predict_prob
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); path = self.image(root); alias = root / "alias.png"; alias.symlink_to(path)
            with self.assertRaises(ValueError): predict_prob(alias, self.model())
            with self.assertRaises(ValueError):
                predict_prob(path, self.model(), expected_image_sha256="0" * 64)

    def test_single_and_batch_paths_match_on_the_same_verified_bytes(self):
        import pandas as pd
        from validation.verify_parity import check_parity
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); paths = []
            for i, color in enumerate([(0, 0, 0), (255, 255, 255), (20, 90, 170)]):
                path = self.image(root, color); target = root / f"sample-{i}.png"; path.rename(target); paths.append(target)
            frame = pd.DataFrame({"image_path": [str(p) for p in paths], "label": [0, 1, 0],
                "sha256": [hashlib.sha256(p.read_bytes()).hexdigest() for p in paths]})
            report = check_parity(self.model(), frame)
            self.assertEqual(report["status"], "passed")
            self.assertEqual(report["images"], 3)
            self.assertLess(report["max_absolute_difference"], 1e-4)

    def test_batch_hash_count_mismatch_fails_before_prediction(self):
        from validation.predict import predict_probs
        with self.assertRaises(ValueError):
            predict_probs(["missing.png"], self.model(), expected_image_sha256s=[])

    def test_training_mode_and_nonfinite_outputs_cannot_be_reported_as_probabilities(self):
        import torch
        from validation.predict import predict_prob
        with tempfile.TemporaryDirectory() as directory:
            path = self.image(Path(directory))
            with self.assertRaises(ValueError): predict_prob(path, self.model().train())
            class Nonfinite(torch.nn.Module):
                def forward(self, x): return torch.full((len(x), 2), float("nan"))
            with self.assertRaises(ValueError): predict_prob(path, Nonfinite().eval())


if __name__ == "__main__":
    unittest.main()
