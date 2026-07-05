"""Attempt A — multi-source training for cross-dataset transfer.

Train ResNet-18 (ImageNet init, unfreeze layer3+layer4+fc) on disc-standardized,
colour-normalized crops from HYGD + RIM-ONE (two source domains), with heavy
colour/geometry augmentation (to defeat the residual camera-style gap) + domain-
balanced sampling + class-weighting. Evaluate on FULLY-HELD-OUT PAPILA crops.

Never touches PAPILA during training. Patient-grouped val split of the sources.
Usage: python validation/train_generalize.py [--epochs 20]
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupShuffleSplit
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import models, transforms

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "validation/data"
DEV = "mps" if torch.backends.mps.is_available() else "cpu"
IM_MEAN, IM_STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]


def labeled_crops():
    """Join cached crops to labels + patient ids for all 3 datasets."""
    mf = pd.read_csv(DATA / "crop_manifest.csv")
    # HYGD
    h = pd.read_csv(ROOT / "data/raw/Labels.csv"); h.columns = [c.strip() for c in h.columns]
    h["stem"] = h["Image Name"].str.replace(".jpg", "", regex=False)
    h["label"] = (h["Label"].str.strip() == "GON+").astype(int)
    h["patient_id"] = "H" + h["Patient"].astype(str)
    hmap = h.set_index("stem")[["label", "patient_id"]]
    # RIM-ONE
    r = pd.read_csv(DATA / "rimone_labels.csv")
    r["stem"] = r["image_path"].apply(lambda p: os.path.splitext(os.path.basename(p))[0])
    r["patient_id"] = "R" + r["stem"]
    rmap = r.set_index("stem")[["label", "patient_id"]]
    # PAPILA
    p = pd.read_csv(DATA / "papila_labels.csv")
    p["stem"] = p["image_path"].apply(lambda x: os.path.splitext(os.path.basename(x))[0])
    p["patient_id"] = "P" + p["patient_id"].astype(str)
    pmap = p.set_index("stem")[["label", "patient_id"]]

    rows = []
    for _, m in mf.iterrows():
        src = {"HYGD": hmap, "RIMONE": rmap, "PAPILA": pmap}[m["dataset"]]
        if m["stem"] in src.index:
            lab, pid = src.loc[m["stem"], "label"], src.loc[m["stem"], "patient_id"]
            rows.append({"crop_path": m["crop_path"], "dataset": m["dataset"], "label": int(lab), "patient_id": pid})
    return pd.DataFrame(rows)


class CropDS(Dataset):
    def __init__(self, df, tf):
        self.df = df.reset_index(drop=True); self.tf = tf

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        r = self.df.iloc[i]
        return self.tf(Image.open(r["crop_path"]).convert("RGB")), torch.tensor(int(r["label"]))


def tfs(train):
    if train:
        return transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(0.85, 1.0)),
            transforms.RandomHorizontalFlip(), transforms.RandomVerticalFlip(),
            transforms.RandomRotation(25),
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.06),
            transforms.RandomAdjustSharpness(2, p=0.3),
            transforms.RandomAutocontrast(p=0.3),
            transforms.ToTensor(), transforms.Normalize(IM_MEAN, IM_STD),
        ])
    return transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(), transforms.Normalize(IM_MEAN, IM_STD)])


@torch.no_grad()
def auc(model, df):
    model.eval(); tf = tfs(False); P, Y = [], []
    dl = DataLoader(CropDS(df, tf), batch_size=64)
    for x, y in dl:
        P.extend(torch.softmax(model(x.to(DEV)), 1)[:, 1].cpu().numpy()); Y.extend(y.numpy())
    return roc_auc_score(Y, P), np.array(P), np.array(Y)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--epochs", type=int, default=20); a = ap.parse_args()
    torch.manual_seed(0)
    df = labeled_crops()
    src = df[df.dataset.isin(["HYGD", "RIMONE"])].reset_index(drop=True)
    papila = df[df.dataset == "PAPILA"].reset_index(drop=True)
    print(f"sources: {len(src)} ({dict(src.dataset.value_counts())}) | held-out PAPILA: {len(papila)}")

    gss = GroupShuffleSplit(1, test_size=0.15, random_state=0)
    tri, vai = next(gss.split(src, groups=src.patient_id))
    tr, va = src.iloc[tri].reset_index(drop=True), src.iloc[vai].reset_index(drop=True)

    # domain+class balanced sampler: weight by 1/(dataset size) and 1/(class freq)
    wds = tr.groupby("dataset")["label"].transform("count")
    wcl = tr.groupby(["dataset", "label"])["label"].transform("count")
    weights = (1.0 / wds) * (1.0 / wcl)
    sampler = WeightedRandomSampler(torch.tensor(weights.values, dtype=torch.double), num_samples=len(tr), replacement=True)

    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    model.fc = nn.Linear(512, 2)
    for p in model.parameters():
        p.requires_grad = False
    for name, p in model.named_parameters():
        if name.startswith(("layer3", "layer4", "fc")):
            p.requires_grad = True
    model.to(DEV)

    crit = nn.CrossEntropyLoss()  # balancing handled by the sampler
    opt = torch.optim.Adam([
        {"params": model.fc.parameters(), "lr": 1e-3},
        {"params": model.layer4.parameters(), "lr": 3e-4},
        {"params": model.layer3.parameters(), "lr": 1e-4},
    ])
    tl = DataLoader(CropDS(tr, tfs(True)), batch_size=32, sampler=sampler)

    # SWA-style weight averaging over post-warmup epochs is TARGET-FREE (never looks
    # at PAPILA) and averages out the source-overfitting that makes the single best
    # source-val checkpoint transfer poorly.
    SWA_START = 4
    swa_sum, swa_n = None, 0
    for ep in range(a.epochs):
        model.train()
        for x, y in tl:
            opt.zero_grad(); loss = crit(model(x.to(DEV)), y.to(DEV)); loss.backward(); opt.step()
        va_auc, _, _ = auc(model, va)
        pap_auc, _, _ = auc(model, papila)  # printed for the record only; NOT used to select
        if ep >= SWA_START:
            sd = model.state_dict()
            if swa_sum is None:
                swa_sum = {k: v.detach().cpu().double() for k, v in sd.items()}
            else:
                for k in swa_sum:
                    swa_sum[k] += sd[k].detach().cpu().double()
            swa_n += 1
        print(f"    epoch {ep+1}/{a.epochs}  source-val AUROC {va_auc:.3f}   [PAPILA {pap_auc:.3f}]", flush=True)

    ref = model.state_dict()
    swa_state = {k: (v / swa_n).to(ref[k].dtype).to(ref[k].device) for k, v in swa_sum.items()}
    model.load_state_dict(swa_state)
    swa_val, _, _ = auc(model, va)
    final_pap, pP, pY = auc(model, papila)

    def bootstrap_ci(y, p, n=2000, seed=42):
        rng = np.random.default_rng(seed); v = []
        for _ in range(n):
            idx = rng.integers(0, len(y), len(y))
            if len(np.unique(y[idx])) > 1:
                v.append(roc_auc_score(y[idx], p[idx]))
        return [round(float(np.percentile(v, 2.5)), 4), round(float(np.percentile(v, 97.5)), 4)]

    torch.save(swa_state, ROOT / "results/generalize_multisource.pt")
    res = {"attempt": "A_multisource_disccrop_colornorm_SWA",
           "held_out_papila_auroc": round(float(final_pap), 4),
           "papila_auroc_95ci": bootstrap_ci(pY, pP),
           "model_selection": "SWA weight-average over epochs>=%d (target-free; PAPILA never used to select)" % (SWA_START + 1),
           "swa_source_val_auroc": round(float(swa_val), 4),
           "prior_zero_shot_papila": 0.5077, "prior_singlesource_finetune": 0.6834,
           "levers": "auto-disc-crop (U-Net Dice 0.958) + shades-of-gray+CLAHE colour-norm + 2-source (HYGD+RIM-ONE) + heavy colour aug + domain/class-balanced sampler"}
    (ROOT / "results/generalize_attemptA.json").write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
