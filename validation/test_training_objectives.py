"""Exercise real Torch loss aggregation without pretrained weights or data."""
import contextlib
import importlib.util
import io
import unittest

TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None


@unittest.skipUnless(TORCH_AVAILABLE, "optional Torch numeric tests")
class TrainingObjectiveTests(unittest.TestCase):
    def test_baseline_validation_loss_is_invariant_to_batch_partition(self):
        import torch
        from torch.utils.data import DataLoader, TensorDataset
        from src.train import train
        logits = torch.tensor([[2., 0.], [1., 0.], [0., 1.], [0., 2.]])
        labels = torch.tensor([0, 0, 1, 1])
        weights = torch.tensor([4., 1.])
        expected = torch.nn.functional.cross_entropy(logits, labels, weight=weights).item()
        for batch_size in [1, 2, 3, 4]:
            model = torch.nn.Linear(2, 2, bias=False)
            with torch.no_grad():
                model.weight.copy_(torch.eye(2))
            dataset = TensorDataset(logits, labels)
            with contextlib.redirect_stdout(io.StringIO()):
                _, history = train(model, DataLoader(dataset, batch_size=4),
                    DataLoader(dataset, batch_size=batch_size), epochs=1, lr=0, weights=weights)
            with self.subTest(batch_size=batch_size):
                self.assertAlmostEqual(history["val_loss"][0], expected, places=6)

    def test_global_weighted_reduction_can_reverse_checkpoint_ranking(self):
        from src.loss_utils import LossAccumulator
        # Two single-class batches: their CE means already cancel class weights.
        # The historical image-count mean selects A; true weighted CE selects B.
        a, b = LossAccumulator(), LossAccumulator()
        for loss, mass in [(0.8, 4), (0.05, 1)]:
            a.add(loss, mass)
        for loss, mass in [(0.5, 4), (0.6, 1)]:
            b.add(loss, mass)
        self.assertAlmostEqual(a.mean(), 0.65)
        self.assertAlmostEqual(b.mean(), 0.52)
        self.assertGreater(a.mean(), b.mean())
        self.assertLess((0.8 + 0.05) / 2, (0.5 + 0.6) / 2)

    def test_loss_accumulator_rejects_invalid_or_empty_data(self):
        from src.loss_utils import LossAccumulator
        for loss, mass in [(float("nan"), 1), (float("inf"), 1), (-1, 1), (1, 0), (1, -1)]:
            with self.assertRaises(ValueError):
                LossAccumulator().add(loss, mass)
        with self.assertRaises(ValueError):
            LossAccumulator().mean()

    def test_normalizer_matches_torch_class_weights_and_ignore_index(self):
        import torch
        from src.loss_utils import cross_entropy_normalizer
        labels = torch.tensor([0, 1, 1, -100])
        self.assertEqual(cross_entropy_normalizer(labels), 3)
        self.assertEqual(cross_entropy_normalizer(labels, torch.tensor([4., 1.])), 6)
        with self.assertRaises(ValueError):
            cross_entropy_normalizer(torch.tensor([-100]))

    def test_parameter_frozen_mode_still_adapts_batchnorm_buffers(self):
        import torch
        from src.experiments import build_model
        # Characterize the retained recipe accurately, without changing it.
        model = build_model(mode="frozen", pretrained=False)
        before_parameter = model.conv1.weight.detach().clone()
        before_buffer = model.bn1.running_mean.detach().clone()
        model.train()
        with torch.no_grad():
            model(torch.randn(2, 3, 32, 32) + 3)
        self.assertTrue(torch.equal(before_parameter, model.conv1.weight))
        self.assertFalse(torch.equal(before_buffer, model.bn1.running_mean))

    def test_v6_inner_loop_selects_global_ce_checkpoint_and_preserves_v5_replay(self):
        import torch
        import pandas as pd
        from unittest.mock import patch
        from validation import internal_evaluation_repair as runner

        class ToyDataset:
            def __init__(self, frame, *args, **kwargs): self.labels = frame.label.tolist()
            def __len__(self): return len(self.labels)
            def __getitem__(self, i):
                return torch.tensor([self.labels[i]], dtype=torch.float32), torch.tensor(self.labels[i])

        class TwoCheckpoints(torch.nn.Module):
            def __init__(self):
                super().__init__(); self.fc = torch.nn.Linear(1, 1, bias=False); self.epoch = 0
            def train(self, mode=True):
                if mode: self.epoch += 1
                return super().train(mode)
            def forward(self, x):
                index = x[:, 0].long()
                if self.training: logits = torch.zeros((len(x), 2))
                else:
                    losses = torch.tensor([.8, .05] if self.epoch == 1 else [.5, .6])
                    probability = torch.exp(-losses[index])
                    odds = torch.log(probability / (1-probability))
                    logits = torch.zeros((len(x), 2)); logits[torch.arange(len(x)), index] = odds
                return logits + self.fc.weight.sum() * 0

        train = pd.DataFrame({"label": [0, 1, 1, 1, 1]})  # class weight ratio 4:1
        valid = pd.DataFrame({"label": [0, 1]})
        for global_loss, expected_epoch in [(False, 1), (True, 2)]:
            with patch.object(runner, "build_model", side_effect=lambda **kwargs: TwoCheckpoints()), patch.object(runner, "TransformHYGDDataset", ToyDataset), patch.object(runner, "build_transforms", return_value=None):
                _, history, best = runner.train_with_inner_validation(train, valid, 2, 1, "cpu", 1,
                    verbose=False, global_weighted_loss=global_loss)
            self.assertEqual(best, expected_epoch)
            expected = [.65, .52] if global_loss else [.425, .55]
            for row, loss in zip(history, expected):
                self.assertAlmostEqual(row["inner_validation_loss"], loss, places=5)


if __name__ == "__main__":
    unittest.main()
