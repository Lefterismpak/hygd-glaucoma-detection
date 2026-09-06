"""BCE segmentation targets must be binary probabilities, not PNG intensities."""
import unittest
import importlib.util
import numpy as np


class MaskTargetTests(unittest.TestCase):
    def test_released_png_intensities_become_zero_one_targets(self):
        from validation.mask_targets import binary_training_mask
        value = binary_training_mask(np.array([[0, 255], [255, 0]], dtype=np.uint8))
        np.testing.assert_array_equal(value, [[0., 1.], [1., 0.]])
        self.assertEqual(value.dtype, np.float32)

    def test_existing_binary_contours_are_preserved(self):
        from validation.mask_targets import binary_training_mask
        value = np.array([[0, 1], [1, 0]], dtype=np.float32)
        np.testing.assert_array_equal(binary_training_mask(value), value)

    def test_unknown_intensity_nan_and_wrong_shape_are_rejected(self):
        from validation.mask_targets import binary_training_mask
        for value in [np.array([[128]]), np.array([[-1]]), np.array([[np.nan]]),
                      np.zeros((2, 2, 3)), np.array([1, 0])]:
            with self.assertRaises(ValueError): binary_training_mask(value)

    @unittest.skipUnless(importlib.util.find_spec("torch") is not None and importlib.util.find_spec("cv2") is not None,
                         "optional Torch/OpenCV integration")
    def test_real_segmentation_dataset_normalizes_png_masks_before_bce(self):
        import torch
        import tempfile
        import types
        import sys
        from pathlib import Path
        from unittest.mock import patch
        from PIL import Image
        from validation import build_vcdr

        class Captured(Exception): pass
        class CheckTargets(torch.nn.Module):
            def forward(inner_self, logits, targets):
                self.assertTrue(bool(torch.all((targets == 0) | (targets == 1))))
                raise Captured()

        # Mock only heavy pretrained construction. Real image loading, resizing,
        # dataset target creation and DataLoader batching execute unchanged.
        fake_smp = types.SimpleNamespace(Unet=lambda *args, **kwargs: torch.nn.Conv2d(3, 2, 1))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "synthetic.png"; Image.new("RGB", (16, 16), (80, 90, 100)).save(path)
            mask = np.zeros((16, 16), dtype=np.uint8); mask[3:12, 3:12] = 255
            items = [(str(path), lambda shape: (mask, mask))] * 4
            # Restore only this one provider key. Restoring all sys.modules can
            # unload newly imported Torch modules while their native operators
            # remain registered, breaking a later cold torchvision import.
            absent = object()
            original = sys.modules.get("segmentation_models_pytorch", absent)
            sys.modules["segmentation_models_pytorch"] = fake_smp
            try:
                with patch.object(build_vcdr, "DEV", "cpu"), patch.object(torch.nn, "BCEWithLogitsLoss", CheckTargets):
                    with self.assertRaises(Captured): build_vcdr.train_cupdisc_unet(items, epochs=1, size=8)
            finally:
                if original is absent: sys.modules.pop("segmentation_models_pytorch", None)
                else: sys.modules["segmentation_models_pytorch"] = original

    @unittest.skipUnless(importlib.util.find_spec("torch") is not None and importlib.util.find_spec("cv2") is not None,
                         "optional historical preparation integration")
    def test_preparation_entrypoints_preserve_existing_frozen_outputs(self):
        import tempfile
        from pathlib import Path
        from unittest.mock import patch
        from validation import build_vcdr, build_disc_coords, build_crops
        for module, function, filename in [(build_vcdr, build_vcdr.main, "vcdr.csv"),
            (build_disc_coords, build_disc_coords.main, "disc_coords.csv"),
            (build_crops, build_crops.build, "crop_manifest.csv")]:
            with self.subTest(module=module.__name__), tempfile.TemporaryDirectory() as directory:
                root = Path(directory); file = root / filename; file.write_bytes(b"retained frozen artifact\n")
                with patch.object(module, "DATA", root), self.assertRaises(FileExistsError): function()
                self.assertEqual(file.read_bytes(), b"retained frozen artifact\n")


if __name__ == "__main__":
    unittest.main()
