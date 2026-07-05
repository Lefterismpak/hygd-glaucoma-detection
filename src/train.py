"""Baseline model training (Phase 4): transfer learning on a pretrained ResNet18 backbone.

Class imbalance in HYGD is real (73% GON+ / 27% GON-) — handled here with a
class-weighted loss rather than oversampling, to keep the pipeline simple.
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torchvision import models

from src.data_utils import preprocess_image


class HYGDDataset(Dataset):
    """Wraps a metadata DataFrame (image_path, label) for PyTorch's DataLoader."""

    def __init__(self, metadata_df, target_size=(224, 224)):
        self.df = metadata_df.reset_index(drop=True)
        self.target_size = target_size

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image = preprocess_image(row["image_path"], self.target_size)
        label = int(row["label"])
        return torch.from_numpy(image), torch.tensor(label, dtype=torch.long)


def build_model(num_classes=2, freeze_backbone=True):
    """ResNet18 pretrained on ImageNet, with a new binary classification head.

    `freeze_backbone=True` trains only the new head first — a sane default for
    a small (747-image) dataset where fine-tuning the whole network risks
    overfitting. Unfreeze later (set to False) once the head-only baseline works.
    """
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False

    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)  # always trainable, even if backbone is frozen
    return model


def class_weights(labels):
    """Inverse-frequency class weights for CrossEntropyLoss, given a 1D array of 0/1 labels."""
    labels = np.asarray(labels)
    counts = np.bincount(labels, minlength=2)
    counts = np.where(counts == 0, 1, counts)  # avoid div-by-zero if a split is missing a class
    weights = counts.sum() / (len(counts) * counts)
    return torch.tensor(weights, dtype=torch.float32)


def train(model, train_loader, val_loader, epochs=10, lr=1e-3, device="cpu", weights=None):
    """Train loop. Returns the model and a history dict (train_loss, val_loss per epoch)."""
    model = model.to(device)
    criterion = nn.CrossEntropyLoss(weight=weights.to(device) if weights is not None else None)
    optimizer = torch.optim.Adam(
        (p for p in model.parameters() if p.requires_grad), lr=lr
    )

    history = {"train_loss": [], "val_loss": []}

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * images.size(0)
        train_loss = running_loss / len(train_loader.dataset)

        model.eval()
        val_running_loss = 0.0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_running_loss += loss.item() * images.size(0)
        val_loss = val_running_loss / len(val_loader.dataset)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        print(f"epoch {epoch + 1}/{epochs}  train_loss={train_loss:.4f}  val_loss={val_loss:.4f}")

    return model, history
