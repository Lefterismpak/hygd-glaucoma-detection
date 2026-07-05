"""Dataset loading, EDA helpers, and patient-level train/val/test splitting for HYGD.

Dataset: Hillel Yaffe Glaucoma Dataset (HYGD) v1.1.0, PhysioNet, DOI 10.13026/m92s-0z95.
Open Data Commons Attribution License v1.0. 747 fundus images from 288 patients.
Images named `{patient_id}_{image_number}.jpg`; labels in `Labels.csv` with columns
Image Name, Patient, Label (GON+/GON-), Quality Score (1-10).

Patients contribute multiple images each, so splitting must be done at the PATIENT
level, not the image level — otherwise the same patient's eye can leak across
train/val/test and inflate reported performance.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import GroupShuffleSplit

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406])
IMAGENET_STD = np.array([0.229, 0.224, 0.225])


def load_dataset_metadata(raw_dir):
    """Return a DataFrame with columns: image_path, patient_id, label (1=GON+, 0=GON-), quality_score.

    `raw_dir` should contain `Labels.csv` and an `Images/` folder (the HYGD layout).
    """
    raw_dir = Path(raw_dir)
    labels_path = raw_dir / "Labels.csv"
    if not labels_path.exists():
        raise FileNotFoundError(
            f"Labels.csv not found under {raw_dir}. Expected the HYGD zip to be "
            "extracted here (see README for the download source)."
        )

    df = pd.read_csv(labels_path)
    df.columns = [c.strip() for c in df.columns]

    images_dir = raw_dir / "Images"
    if not images_dir.exists():
        # Some PhysioNet zips flatten the top-level folder name; fall back to raw_dir itself.
        images_dir = raw_dir

    df["image_path"] = df["Image Name"].apply(lambda name: str(images_dir / name))
    df["patient_id"] = df["Patient"].astype(str)
    df["label"] = (df["Label"].str.strip() == "GON+").astype(int)
    df["quality_score"] = df["Quality Score"] if "Quality Score" in df.columns else np.nan

    return df[["image_path", "patient_id", "label", "quality_score"]]


def train_val_test_split(metadata, val_size=0.15, test_size=0.15, seed=42):
    """Patient-level split (GroupShuffleSplit) so no patient appears in more than one split.

    Returns (train_df, val_df, test_df).
    """
    groups = metadata["patient_id"]

    splitter1 = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    trainval_idx, test_idx = next(splitter1.split(metadata, groups=groups))
    trainval_df = metadata.iloc[trainval_idx].reset_index(drop=True)
    test_df = metadata.iloc[test_idx].reset_index(drop=True)

    relative_val_size = val_size / (1 - test_size)
    splitter2 = GroupShuffleSplit(n_splits=1, test_size=relative_val_size, random_state=seed)
    train_idx, val_idx = next(
        splitter2.split(trainval_df, groups=trainval_df["patient_id"])
    )
    train_df = trainval_df.iloc[train_idx].reset_index(drop=True)
    val_df = trainval_df.iloc[val_idx].reset_index(drop=True)

    return train_df, val_df, test_df


def preprocess_image(path, target_size=(224, 224)):
    """Load a fundus image, resize, and normalize to ImageNet stats (for the pretrained backbone).

    Returns a float32 array of shape (3, H, W), channel-first (PyTorch convention).
    """
    img = Image.open(path).convert("RGB").resize(target_size)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    arr = (arr - IMAGENET_MEAN) / IMAGENET_STD
    return arr.transpose(2, 0, 1).astype(np.float32)
