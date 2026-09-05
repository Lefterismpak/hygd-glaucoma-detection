"""Dataset loading, EDA helpers, and supplied-ID-grouped splitting for HYGD.

Dataset: Hillel Yaffe Glaucoma Dataset (HYGD) v1.1.0, PhysioNet, DOI 10.13026/m92s-0z95.
Open Data Commons Attribution License v1.0. 747 fundus images associated with
288 supplied patient IDs.
Images named `{patient_id}_{image_number}.jpg`; labels in `Labels.csv` with columns
Image Name, Patient, Label (GON+/GON-), Quality Score (1-10).

Supplied IDs can contribute multiple images, so the historical split grouped that
field rather than image rows. A later audit showed that exact duplicates can span
different supplied IDs, so the preferred evaluator additionally links those IDs.
"""

import io
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

from validation.evaluation_utils import read_verified_file_bytes

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406])
IMAGENET_STD = np.array([0.229, 0.224, 0.225])


def load_verified_rgb_image(path, *, expected_sha256=None):
    """Decode exactly the no-follow bytes whose SHA-256 was verified."""

    from PIL import Image

    content, _ = read_verified_file_bytes(
        path,
        expected_sha256=expected_sha256,
    )
    with Image.open(io.BytesIO(content)) as source:
        image = source.convert("RGB")
        image.load()
    return image


def load_dataset_metadata(raw_dir, *, expected_labels_sha256=None):
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

    labels_bytes, labels_sha256 = read_verified_file_bytes(
        labels_path,
        expected_sha256=expected_labels_sha256,
    )
    df = pd.read_csv(
        io.BytesIO(labels_bytes),
        dtype={"Image Name": "string", "Patient": "string", "Label": "string"},
    )
    df.columns = [c.strip() for c in df.columns]
    required = {"Image Name", "Patient", "Label"}
    missing_columns = sorted(required - set(df.columns))
    if missing_columns:
        raise ValueError(f"Labels.csv is missing required columns: {missing_columns}")
    if df.empty:
        raise ValueError("Labels.csv must contain at least one row")

    image_names = df["Image Name"]
    if image_names.isna().any():
        raise ValueError("Image Name cannot be missing")
    image_names = image_names.astype(str).str.strip()
    invalid_names = [
        name
        for name in image_names
        if not name or Path(name).name != name or name in {".", ".."}
    ]
    if invalid_names:
        raise ValueError("Image Name values must be plain filenames without path traversal")

    supplied_ids = df["Patient"]
    if supplied_ids.isna().any():
        raise ValueError("Supplied patient ID cannot be missing")
    patient_ids = supplied_ids.astype(str).str.strip()
    if patient_ids.eq("").any():
        raise ValueError("Supplied patient ID cannot be blank")

    labels = df["Label"]
    if labels.isna().any():
        raise ValueError("Label must be exactly GON+ or GON-")
    labels = labels.astype(str).str.strip()
    if not labels.isin(["GON+", "GON-"]).all():
        raise ValueError("Label must be exactly GON+ or GON-")

    images_dir = raw_dir / "Images"
    if not images_dir.exists():
        # Some PhysioNet zips flatten the top-level folder name; fall back to raw_dir itself.
        images_dir = raw_dir

    df["image_path"] = image_names.map(lambda name: str(images_dir / name))
    df["patient_id"] = patient_ids
    df["label"] = labels.map({"GON+": 1, "GON-": 0}).astype(int)
    df["quality_score"] = df["Quality Score"] if "Quality Score" in df.columns else np.nan

    metadata = df[["image_path", "patient_id", "label", "quality_score"]].copy()
    metadata.attrs["labels_csv_sha256"] = labels_sha256
    return metadata


def train_val_test_split(metadata, val_size=0.15, test_size=0.15, seed=42):
    """Group by supplied patient ID so no supplied ID spans historical splits.

    This does not guarantee biological-subject independence or link exact
    duplicates across different supplied IDs. Returns (train_df, val_df, test_df).
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
    img = load_verified_rgb_image(path).resize(target_size)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    arr = (arr - IMAGENET_MEAN) / IMAGENET_STD
    return arr.transpose(2, 0, 1).astype(np.float32)
