"""Build optic-disc centre+diameter for every image in HYGD / PAPILA / RIM-ONE,
so the preprocessing can crop a disc-size-standardized square (the load-bearing
step for cross-dataset transfer — see validation/FINDINGS DG plan).

- PAPILA: rasterize the expert disc contours (avg of exp1/exp2) -> GT centre/diam.
- RIM-ONE: read the shipped Disc-T PNG masks -> GT centre/diam.
- HYGD (no masks): train a small U-Net (smp, resnet18 encoder) on the PAPILA+RIM-ONE
  disc masks, then predict the disc on all HYGD images -> centre/diam.

Outputs validation/data/disc_coords.csv (dataset,image_path,cx,cy,diameter) and a
few sanity contact sheets in figures/. Images stay local (git-ignored).
"""

import glob
import os
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "validation/data"
PAP = DATA / "papila/PapilaDB-PAPILA-17f8fa7746adb20275b5b6a0d99dc9dfe3007e9f"
DEV = "mps" if torch.backends.mps.is_available() else "cpu"


def mask_to_centre_diam(mask):
    """Largest connected component -> (cx, cy, diameter) in pixels, or None."""
    m = (mask > 0).astype(np.uint8)
    n, lab, stats, cent = cv2.connectedComponentsWithStats(m)
    if n <= 1:
        return None
    i = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    cx, cy = cent[i]
    area = stats[i, cv2.CC_STAT_AREA]
    diam = 2 * np.sqrt(area / np.pi)  # equivalent-circle diameter
    # blend with bbox extent (robust to elongated masks)
    diam = 0.5 * diam + 0.5 * max(stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT])
    return float(cx), float(cy), float(diam)


# ---------- PAPILA GT from contours ----------
def papila_coords():
    rows = []
    lbl = pd.read_csv(DATA / "papila_labels.csv")
    for _, r in lbl.iterrows():
        stem = os.path.splitext(os.path.basename(r["image_path"]))[0]  # RET002OD
        w, h = Image.open(r["image_path"]).size
        polys = []
        for exp in (1, 2):
            f = PAP / "ExpertsSegmentations/Contours" / f"{stem}_disc_exp{exp}.txt"
            if f.exists():
                polys.append(np.loadtxt(f))
        if not polys:
            continue
        mask = np.zeros((h, w), np.uint8)
        for p in polys:
            cv2.fillPoly(mask, [p.astype(np.int32)], 1)
        cd = mask_to_centre_diam(mask)
        if cd:
            rows.append({"dataset": "PAPILA", "image_path": r["image_path"], "cx": cd[0], "cy": cd[1], "diameter": cd[2]})
    return pd.DataFrame(rows)


# ---------- RIM-ONE GT from PNG masks ----------
def rimone_coords():
    rows = []
    lbl = pd.read_csv(DATA / "rimone_labels.csv")
    segdir = DATA / "rimone_segs/RIM-ONE_DL_reference_segmentations"
    for _, r in lbl.iterrows():
        stem = os.path.splitext(os.path.basename(r["image_path"]))[0]  # r1_Im069
        cls = "glaucoma" if r["label"] == 1 else "normal"
        mp = segdir / cls / f"{stem}-1-Disc-T.png"
        if not mp.exists():
            # some masks may be under the other class folder or naming; search
            cand = glob.glob(str(segdir / "*" / f"{stem}-*-Disc-T.png"))
            if not cand:
                continue
            mp = Path(cand[0])
        mask = np.array(Image.open(mp).convert("L"))
        cd = mask_to_centre_diam(mask)
        if cd:
            rows.append({"dataset": "RIMONE", "image_path": r["image_path"], "cx": cd[0], "cy": cd[1], "diameter": cd[2]})
    return pd.DataFrame(rows)


# ---------- U-Net for HYGD ----------
def train_unet(train_items, epochs=14, size=256):
    import segmentation_models_pytorch as smp
    from torch.utils.data import Dataset, DataLoader

    class SegDS(Dataset):
        def __init__(self, items):
            self.items = items

        def __len__(self):
            return len(self.items)

        def __getitem__(self, i):
            path, maskfn = self.items[i]
            img = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
            m = maskfn(img.shape[:2])
            img = cv2.resize(img, (size, size)).astype(np.float32) / 255.0
            m = cv2.resize(m, (size, size), interpolation=cv2.INTER_NEAREST)
            x = torch.tensor(img).permute(2, 0, 1)
            return x, torch.tensor(m[None], dtype=torch.float32)

    tr, va = train_items[:int(0.85 * len(train_items))], train_items[int(0.85 * len(train_items)):]
    model = smp.Unet("resnet18", encoder_weights="imagenet", classes=1, activation=None).to(DEV)
    opt = torch.optim.Adam(model.parameters(), 1e-3)
    bce = torch.nn.BCEWithLogitsLoss()
    tl = DataLoader(SegDS(tr), batch_size=8, shuffle=True)
    vl = DataLoader(SegDS(va), batch_size=8)
    best, best_state = 0.0, None
    for ep in range(epochs):
        model.train()
        for x, y in tl:
            x, y = x.to(DEV), y.to(DEV)
            opt.zero_grad(); loss = bce(model(x), y); loss.backward(); opt.step()
        # val Dice
        model.eval(); dices = []
        with torch.no_grad():
            for x, y in vl:
                p = (torch.sigmoid(model(x.to(DEV))) > 0.5).float().cpu()
                inter = (p * y).sum((1, 2, 3)); d = (2 * inter / (p.sum((1, 2, 3)) + y.sum((1, 2, 3)) + 1e-6))
                dices.extend(d.tolist())
        dice = float(np.mean(dices))
        if dice > best:
            best, best_state = dice, {k: v.clone() for k, v in model.state_dict().items()}
        print(f"    unet epoch {ep+1}/{epochs}  val Dice {dice:.3f}", flush=True)
    model.load_state_dict(best_state)
    print(f"  U-Net best val Dice: {best:.3f}")
    return model, best


def predict_hygd(model, size=256):
    rows = []
    lbl = pd.read_csv(ROOT / "data/raw/Labels.csv")
    lbl.columns = [c.strip() for c in lbl.columns]
    model.eval()
    for name in lbl["Image Name"]:
        path = str(ROOT / "data/raw/Images" / name)
        if not os.path.exists(path):
            continue
        img = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
        h, w = img.shape[:2]
        x = torch.tensor(cv2.resize(img, (size, size)).astype(np.float32) / 255.0).permute(2, 0, 1)[None].to(DEV)
        with torch.no_grad():
            pm = (torch.sigmoid(model(x))[0, 0].cpu().numpy() > 0.5).astype(np.uint8)
        cd = mask_to_centre_diam(cv2.resize(pm, (w, h), interpolation=cv2.INTER_NEAREST))
        if cd:
            rows.append({"dataset": "HYGD", "image_path": path, "cx": cd[0], "cy": cd[1], "diameter": cd[2]})
    return pd.DataFrame(rows)


def main():
    print("== PAPILA disc from contours =="); pap = papila_coords(); print(f"  {len(pap)} coords")
    print("== RIM-ONE disc from PNG masks =="); rim = rimone_coords(); print(f"  {len(rim)} coords")

    # U-Net training items from PAPILA (rasterized) + RIM-ONE (png)
    items = []
    for _, r in pap.iterrows():
        stem = os.path.splitext(os.path.basename(r["image_path"]))[0]
        def mk(shape, s=stem):
            m = np.zeros(shape, np.uint8)
            for exp in (1, 2):
                f = PAP / "ExpertsSegmentations/Contours" / f"{s}_disc_exp{exp}.txt"
                if f.exists():
                    cv2.fillPoly(m, [np.loadtxt(f).astype(np.int32)], 1)
            return m.astype(np.float32)
        items.append((r["image_path"], mk))
    segdir = DATA / "rimone_segs/RIM-ONE_DL_reference_segmentations"
    rlbl = pd.read_csv(DATA / "rimone_labels.csv")
    for _, r in rlbl.iterrows():
        stem = os.path.splitext(os.path.basename(r["image_path"]))[0]
        cls = "glaucoma" if r["label"] == 1 else "normal"
        cand = glob.glob(str(segdir / "*" / f"{stem}-*-Disc-T.png"))
        if not cand:
            continue
        mpth = cand[0]
        def mk2(shape, mp=mpth):
            m = np.array(Image.open(mp).convert("L"))
            return (cv2.resize(m, (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST) > 0).astype(np.float32)
        items.append((r["image_path"], mk2))
    print(f"== train U-Net disc segmenter on {len(items)} PAPILA+RIM-ONE images (device={DEV}) ==")
    torch.manual_seed(0)
    model, dice = train_unet(items)

    print("== predict HYGD discs ==")
    hygd = predict_hygd(model)
    print(f"  {len(hygd)} HYGD coords")

    out = pd.concat([pap, rim, hygd], ignore_index=True)
    out.to_csv(DATA / "disc_coords.csv", index=False)
    print(f"saved {DATA/'disc_coords.csv'}  ({len(out)} rows; unet_val_dice={dice:.3f})")


if __name__ == "__main__":
    main()
