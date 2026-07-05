"""Compute vertical cup-to-disc ratio (VCDR) for every image, as a continuous,
camera-independent glaucoma target for the Attempt-B multi-task head.

VCDR = (vertical extent of optic cup) / (vertical extent of optic disc). Higher
VCDR = more glaucomatous cupping. Because it is a morphology ratio, it cannot be
satisfied by a dataset bezel/colour shortcut — which is why it improves transfer.

- PAPILA: rasterize disc AND cup expert contours -> vertical extents (held out from
  training; used only to sanity-check the VCDR head).
- RIM-ONE: read shipped Disc-T + Cup-T PNG masks.
- HYGD (no masks): a 2-channel (disc,cup) U-Net trained on PAPILA+RIM-ONE predicts both.

Outputs validation/data/vcdr.csv (dataset,stem,vcdr). Masks/images stay local.
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
SEG = DATA / "rimone_segs/RIM-ONE_DL_reference_segmentations"
DEV = "mps" if torch.backends.mps.is_available() else "cpu"


def vext(mask):
    """Vertical extent (rows) of the largest component; 0 if empty."""
    m = (mask > 0).astype(np.uint8)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(m)
    if n <= 1:
        return 0.0
    i = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return float(stats[i, cv2.CC_STAT_HEIGHT])


def vcdr_from_masks(disc, cup):
    d = vext(disc)
    return float(np.clip(vext(cup) / d, 0, 1)) if d > 0 else np.nan


def papila_masks(stem, shape):
    disc = np.zeros(shape, np.uint8); cup = np.zeros(shape, np.uint8)
    for exp in (1, 2):
        for kind, m in (("disc", disc), ("cup", cup)):
            f = PAP / "ExpertsSegmentations/Contours" / f"{stem}_{kind}_exp{exp}.txt"
            if f.exists():
                cv2.fillPoly(m, [np.loadtxt(f).astype(np.int32)], 1)
    return disc, cup


def rimone_masks(stem):
    dc = glob.glob(str(SEG / "*" / f"{stem}-*-Disc-T.png"))
    cc = glob.glob(str(SEG / "*" / f"{stem}-*-Cup-T.png"))
    if not dc or not cc:
        return None, None
    return np.array(Image.open(dc[0]).convert("L")), np.array(Image.open(cc[0]).convert("L"))


def train_cupdisc_unet(items, epochs=16, size=256):
    import segmentation_models_pytorch as smp
    from torch.utils.data import Dataset, DataLoader

    class DS(Dataset):
        def __init__(self, it): self.it = it
        def __len__(self): return len(self.it)
        def __getitem__(self, i):
            path, mkfn = self.it[i]
            img = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
            d, c = mkfn(img.shape[:2])
            img = cv2.resize(img, (size, size)).astype(np.float32) / 255.0
            d = cv2.resize(d, (size, size), interpolation=cv2.INTER_NEAREST)
            c = cv2.resize(c, (size, size), interpolation=cv2.INTER_NEAREST)
            x = torch.tensor(img).permute(2, 0, 1)
            y = torch.tensor(np.stack([d, c]), dtype=torch.float32)
            return x, y

    tr, va = items[:int(0.85 * len(items))], items[int(0.85 * len(items)):]
    model = smp.Unet("resnet18", encoder_weights="imagenet", classes=2, activation=None).to(DEV)
    opt = torch.optim.Adam(model.parameters(), 1e-3); bce = torch.nn.BCEWithLogitsLoss()
    tl = DataLoader(DS(tr), batch_size=8, shuffle=True); vl = DataLoader(DS(va), batch_size=8)
    best, best_state = 0.0, None
    for ep in range(epochs):
        model.train()
        for x, y in tl:
            opt.zero_grad(); loss = bce(model(x.to(DEV)), y.to(DEV)); loss.backward(); opt.step()
        model.eval(); dices = []
        with torch.no_grad():
            for x, y in vl:
                p = (torch.sigmoid(model(x.to(DEV))) > 0.5).float().cpu()
                inter = (p * y).sum((2, 3)); d = (2 * inter / (p.sum((2, 3)) + y.sum((2, 3)) + 1e-6))
                dices.append(d.mean().item())
        dice = float(np.mean(dices))
        if dice > best:
            best, best_state = dice, {k: v.clone() for k, v in model.state_dict().items()}
        print(f"    cupdisc-unet {ep+1}/{epochs} val Dice {dice:.3f}", flush=True)
    model.load_state_dict(best_state); print(f"  cup+disc U-Net best Dice {best:.3f}")
    return model


def main():
    rows = []
    # PAPILA (GT) — for validation of the head, not training
    pl = pd.read_csv(DATA / "papila_labels.csv")
    for _, r in pl.iterrows():
        stem = os.path.splitext(os.path.basename(r["image_path"]))[0]
        w, h = Image.open(r["image_path"]).size
        d, c = papila_masks(stem, (h, w))
        rows.append({"dataset": "PAPILA", "stem": stem, "vcdr": vcdr_from_masks(d, c)})
    # RIM-ONE (GT)
    rl = pd.read_csv(DATA / "rimone_labels.csv")
    for _, r in rl.iterrows():
        stem = os.path.splitext(os.path.basename(r["image_path"]))[0]
        d, c = rimone_masks(stem)
        if d is not None:
            rows.append({"dataset": "RIMONE", "stem": stem, "vcdr": vcdr_from_masks(d, c)})

    # U-Net for HYGD
    items = []
    for _, r in pl.iterrows():
        stem = os.path.splitext(os.path.basename(r["image_path"]))[0]
        items.append((r["image_path"], lambda shp, s=stem: tuple(m.astype(np.float32) for m in papila_masks(s, shp))))
    for _, r in rl.iterrows():
        stem = os.path.splitext(os.path.basename(r["image_path"]))[0]
        d, c = rimone_masks(stem)
        if d is None:
            continue
        items.append((r["image_path"], lambda shp, dd=d, cc=c: (
            cv2.resize(dd, (shp[1], shp[0]), interpolation=cv2.INTER_NEAREST).astype(np.float32),
            cv2.resize(cc, (shp[1], shp[0]), interpolation=cv2.INTER_NEAREST).astype(np.float32))))
    print(f"== train cup+disc U-Net on {len(items)} images (device={DEV}) ==")
    torch.manual_seed(0); model = train_cupdisc_unet(items)

    print("== predict HYGD VCDR ==")
    hl = pd.read_csv(ROOT / "data/raw/Labels.csv"); hl.columns = [c.strip() for c in hl.columns]
    model.eval()
    for name in hl["Image Name"]:
        path = str(ROOT / "data/raw/Images" / name)
        if not os.path.exists(path):
            continue
        img = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB); H, W = img.shape[:2]
        x = torch.tensor(cv2.resize(img, (256, 256)).astype(np.float32) / 255.0).permute(2, 0, 1)[None].to(DEV)
        with torch.no_grad():
            pm = (torch.sigmoid(model(x))[0].cpu().numpy() > 0.5).astype(np.uint8)
        d = cv2.resize(pm[0], (W, H), interpolation=cv2.INTER_NEAREST)
        c = cv2.resize(pm[1], (W, H), interpolation=cv2.INTER_NEAREST)
        rows.append({"dataset": "HYGD", "stem": os.path.splitext(name)[0], "vcdr": vcdr_from_masks(d, c)})

    df = pd.DataFrame(rows).dropna(subset=["vcdr"])
    df.to_csv(DATA / "vcdr.csv", index=False)
    for ds in ["HYGD", "RIMONE", "PAPILA"]:
        s = df[df.dataset == ds]["vcdr"]
        print(f"  {ds}: n={len(s)} mean VCDR {s.mean():.3f}")
    print(f"saved {DATA/'vcdr.csv'}")


if __name__ == "__main__":
    main()
