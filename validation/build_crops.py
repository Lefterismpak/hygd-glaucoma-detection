"""Historical adaptive crop/normalization candidate for three datasets.

The dataset-origin probe is a diagnostic of residual source separability, not a
causal shortcut test or a guarantee of downstream transfer. A high value weakens
source-invariance claims; a low value would not prove shortcut removal.

Crop: square of side k*disc_diameter centred on the disc (aspect-preserving) -> 224.
Norm: Shades-of-Gray colour constancy + CLAHE(L). A circular mask was tested and
rejected because it introduced a new dataset signature.
"""

import os
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from validation.evaluation_utils import assert_fresh_output_bundle, atomic_write_bytes, atomic_write_text
DATA = ROOT / "validation/data"
PROC = ROOT / "data/processed/crops"
K = 2.2
SIZE = 224


def shades_of_gray(img, p=6):
    f = img.astype(np.float32)
    illum = np.power(np.mean(np.power(f, p), axis=(0, 1)) + 1e-6, 1.0 / p)
    illum = illum / (np.sqrt((illum ** 2).sum()) + 1e-6)
    out = f / (illum[None, None, :] * np.sqrt(3) + 1e-6)
    return np.clip(out, 0, 255).astype(np.uint8)


def clahe_L(img):
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    lab[:, :, 0] = cv2.createCLAHE(2.0, (8, 8)).apply(lab[:, :, 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)


def circular_mask(img):
    h, w = img.shape[:2]
    m = np.zeros((h, w), np.uint8)
    cv2.circle(m, (w // 2, h // 2), int(0.5 * min(h, w)), 1, -1)
    return img * m[:, :, None]


def process_one(path, cx, cy, diam, k=K, size=SIZE):
    img = cv2.imread(path)
    if img is None:
        return None
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w = img.shape[:2]
    half = int(k * diam / 2)
    x0, y0, x1, y1 = int(cx - half), int(cy - half), int(cx + half), int(cy + half)
    pad = [max(0, -y0), max(0, y1 - h), max(0, -x0), max(0, x1 - w)]
    # REFLECT (not zero) padding: avoids the hard square/black edge that RIM-ONE's
    # already-tight crops otherwise get, which the dataset-probe latches onto.
    img = cv2.copyMakeBorder(img, *pad, cv2.BORDER_REFLECT)
    x0, x1, y0, y1 = x0 + pad[2], x1 + pad[2], y0 + pad[0], y1 + pad[0]
    crop = img[y0:y1, x0:x1]
    if crop.size == 0:
        return None
    crop = cv2.resize(crop, (size, size))
    crop = shades_of_gray(crop)
    crop = clahe_L(crop)
    # no circular mask — it re-introduced a square-in-circle dataset signature.
    return crop


def build():
    assert_fresh_output_bundle([DATA / "crop_manifest.csv"])
    coords = pd.read_csv(DATA / "disc_coords.csv")
    planned = [PROC / ds / (Path(row["image_path"]).stem + ".png")
               for ds in ["HYGD", "PAPILA", "RIMONE"]
               for _, row in coords[coords.dataset == ds].iterrows()]
    if len(planned) != len(set(planned)):
        raise ValueError("Crop rows must not map to duplicate output paths")
    assert_fresh_output_bundle([*planned, ROOT / "figures/dg_crops_contact_sheet.png"])
    manifest = []
    for ds in ["HYGD", "PAPILA", "RIMONE"]:
        (PROC / ds).mkdir(parents=True, exist_ok=True)
        sub = coords[coords.dataset == ds]
        n = 0
        for _, r in sub.iterrows():
            crop = process_one(r["image_path"], r["cx"], r["cy"], r["diameter"])
            if crop is None:
                continue
            stem = os.path.splitext(os.path.basename(r["image_path"]))[0]
            outp = PROC / ds / f"{stem}.png"
            success, encoded = cv2.imencode(".png", cv2.cvtColor(crop, cv2.COLOR_RGB2BGR))
            if not success:
                raise ValueError("Crop PNG encoding failed")
            atomic_write_bytes(outp, encoded.tobytes())
            manifest.append({"dataset": ds, "crop_path": str(outp), "orig": r["image_path"], "stem": stem})
            n += 1
        print(f"  {ds}: {n} crops")
    mf = pd.DataFrame(manifest)
    atomic_write_text(DATA / "crop_manifest.csv", mf.to_csv(index=False))

    # contact sheets
    ROOT.joinpath("figures").mkdir(exist_ok=True)
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, axes = plt.subplots(3, 8, figsize=(16, 6))
    for row, ds in enumerate(["HYGD", "PAPILA", "RIMONE"]):
        sub = mf[mf.dataset == ds].head(8)
        for ax, (_, r) in zip(axes[row], sub.iterrows()):
            ax.imshow(cv2.cvtColor(cv2.imread(r["crop_path"]), cv2.COLOR_BGR2RGB)); ax.axis("off")
        axes[row][0].set_ylabel(ds)
    fig.suptitle("Disc-standardized + colour-normalized crops (should look uniform across datasets)")
    plt.tight_layout(); plt.savefig(ROOT / "figures/dg_crops_contact_sheet.png", dpi=110); plt.close()
    print("  saved figures/dg_crops_contact_sheet.png")
    return mf


def dataset_probe(mf):
    """Describe dataset separability from downsized normalized-crop pixels.

    This historical probe splits image rows; related subjects may cross folds.
    Near-chance accuracy is compatible with this probe lacking source signal; it
    does not prove that shortcuts were removed or that transfer will succeed.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score
    X, y = [], []
    lab = {"HYGD": 0, "PAPILA": 1, "RIMONE": 2}
    for _, r in mf.iterrows():
        im = cv2.resize(cv2.imread(r["crop_path"]), (32, 32)).astype(np.float32).flatten() / 255.0
        X.append(im); y.append(lab[r["dataset"]])
    X, y = np.array(X), np.array(y)
    acc = cross_val_score(LogisticRegression(max_iter=500), X, y, cv=4).mean()
    chance = max(np.bincount(y)) / len(y)
    print(f"\n== HISTORICAL IMAGE-ROW DATASET PROBE (NONQUALIFYING) ==")
    print(f"  3-way dataset classification accuracy: {acc:.3f}  (chance/majority = {chance:.3f})")
    print("  This measures source decodability under image-row CV, not causal reliance or transportability.")
    return acc, chance


if __name__ == "__main__":
    mf = build()
    dataset_probe(mf)
