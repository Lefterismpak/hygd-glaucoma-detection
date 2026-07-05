"""Attempt B — multi-source training + VCDR auxiliary regression head.

Same disc-crop / colour-norm / multi-source / SWA recipe as Attempt A, but the
backbone also predicts vertical cup-to-disc ratio (VCDR) as an auxiliary target.
VCDR is a continuous, camera-independent glaucoma morphology signal (VCDR alone
separates glaucoma at AUROC 0.96 on RIM-ONE, 0.81 on held-out PAPILA), so the aux
loss pulls the shared features onto cupping rather than dataset texture.

VCDR is supervised ONLY where it is reliable: RIM-ONE (expert masks). HYGD VCDR
(predicted on SLO-derived images) is noisy (VCDR->glaucoma only 0.61) so HYGD
samples are masked out of the VCDR loss — they still contribute the classification
loss. PAPILA stays fully held out.

Usage: python validation/train_generalize_vcdr.py [--epochs 20] [--vcdr_w 0.5]
"""

import argparse
import json
import os
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
RELIABLE_VCDR = {"RIMONE"}  # HYGD VCDR is U-Net-predicted on SLO images -> unreliable


def data_table():
    mf = pd.read_csv(DATA / "crop_manifest.csv")
    vcdr = pd.read_csv(DATA / "vcdr.csv").set_index(["dataset", "stem"])['vcdr']
    h = pd.read_csv(ROOT / "data/raw/Labels.csv"); h.columns = [c.strip() for c in h.columns]
    h["stem"] = h["Image Name"].str.replace(".jpg", "", regex=False)
    h["label"] = (h["Label"].str.strip() == "GON+").astype(int); h["patient_id"] = "H" + h["Patient"].astype(str)
    r = pd.read_csv(DATA / "rimone_labels.csv"); r["stem"] = r["image_path"].apply(lambda p: os.path.splitext(os.path.basename(p))[0]); r["patient_id"] = "R" + r["stem"]
    p = pd.read_csv(DATA / "papila_labels.csv"); p["stem"] = p["image_path"].apply(lambda x: os.path.splitext(os.path.basename(x))[0]); p["patient_id"] = "P" + p["patient_id"].astype(str)
    lab = {"HYGD": h.set_index("stem"), "RIMONE": r.set_index("stem"), "PAPILA": p.set_index("stem")}
    rows = []
    for _, m in mf.iterrows():
        src = lab[m["dataset"]]
        if m["stem"] not in src.index:
            continue
        vv = vcdr.get((m["dataset"], m["stem"]), np.nan)
        reliable = (m["dataset"] in RELIABLE_VCDR) and not np.isnan(vv)
        rows.append({"crop_path": m["crop_path"], "dataset": m["dataset"],
                     "label": int(src.loc[m["stem"], "label"]), "patient_id": src.loc[m["stem"], "patient_id"],
                     "vcdr": float(vv) if reliable else 0.0, "vcdr_w": 1.0 if reliable else 0.0})
    return pd.DataFrame(rows)


class DS(Dataset):
    def __init__(self, df, tf): self.df = df.reset_index(drop=True); self.tf = tf
    def __len__(self): return len(self.df)
    def __getitem__(self, i):
        r = self.df.iloc[i]
        return (self.tf(Image.open(r["crop_path"]).convert("RGB")),
                torch.tensor(int(r["label"])), torch.tensor(r["vcdr"], dtype=torch.float32),
                torch.tensor(r["vcdr_w"], dtype=torch.float32))


def tfs(train):
    if train:
        return transforms.Compose([transforms.RandomResizedCrop(224, scale=(0.85, 1.0)),
            transforms.RandomHorizontalFlip(), transforms.RandomVerticalFlip(), transforms.RandomRotation(25),
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.06), transforms.RandomAdjustSharpness(2, p=0.3),
            transforms.RandomAutocontrast(p=0.3), transforms.ToTensor(), transforms.Normalize(IM_MEAN, IM_STD)])
    return transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(), transforms.Normalize(IM_MEAN, IM_STD)])


class MultiTask(nn.Module):
    def __init__(self):
        super().__init__()
        bb = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        self.feat = nn.Sequential(*list(bb.children())[:-1])
        self.cls = nn.Linear(512, 2); self.vcdr = nn.Linear(512, 1)
        self._bb = bb  # keep names for param groups
    def forward(self, x):
        f = self.feat(x).flatten(1)
        return self.cls(f), self.vcdr(f).squeeze(1)


@torch.no_grad()
def papila_auc(model, df, tta=False):
    model.eval()
    import torchvision.transforms.functional as F
    norm = transforms.Normalize(IM_MEAN, IM_STD)
    def base(im): return transforms.functional.normalize(transforms.functional.to_tensor(im.resize((224, 224))), IM_MEAN, IM_STD)
    views = [lambda im: im]
    if tta:
        views += [lambda im: F.hflip(im), lambda im: F.vflip(im), lambda im: F.rotate(im, 10), lambda im: F.rotate(im, -10)]
    P, Y = [], []
    for _, r in df.iterrows():
        im = Image.open(r["crop_path"]).convert("RGB").resize((224, 224))
        ps = []
        for v in views:
            x = norm(transforms.functional.to_tensor(v(im))).unsqueeze(0).to(DEV)
            ps.append(torch.softmax(model(x)[0], 1)[0, 1].item())
        P.append(np.mean(ps)); Y.append(int(r["label"]))
    return roc_auc_score(Y, P), np.array(Y), np.array(P)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--epochs", type=int, default=20); ap.add_argument("--vcdr_w", type=float, default=0.5)
    a = ap.parse_args(); torch.manual_seed(0)
    df = data_table()
    src = df[df.dataset.isin(["HYGD", "RIMONE"])].reset_index(drop=True)
    pap = df[df.dataset == "PAPILA"].reset_index(drop=True)
    print(f"sources {len(src)} {dict(src.dataset.value_counts())} | VCDR-supervised {int(src.vcdr_w.sum())} | held-out PAPILA {len(pap)}")

    tri, vai = next(GroupShuffleSplit(1, test_size=0.15, random_state=0).split(src, groups=src.patient_id))
    tr, va = src.iloc[tri].reset_index(drop=True), src.iloc[vai].reset_index(drop=True)
    wds = tr.groupby("dataset")["label"].transform("count"); wcl = tr.groupby(["dataset", "label"])["label"].transform("count")
    sampler = WeightedRandomSampler(torch.tensor(((1.0 / wds) * (1.0 / wcl)).values, dtype=torch.double), len(tr), True)

    model = MultiTask().to(DEV)
    for name, p in model.named_parameters():
        p.requires_grad = any(k in name for k in ["feat.6", "feat.7", "cls", "vcdr"])  # layer3,layer4 = feat[6],feat[7]
    ce = nn.CrossEntropyLoss(); mse = nn.MSELoss(reduction="none")
    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=5e-4)
    tl = DataLoader(DS(tr, tfs(True)), batch_size=32, sampler=sampler)

    SWA_START = 4; swa_sum, swa_n = None, 0

    @torch.no_grad()
    def val_auc():
        model.eval(); P, Y = [], []
        for x, y, _, _ in DataLoader(DS(va, tfs(False)), batch_size=64):
            P.extend(torch.softmax(model(x.to(DEV))[0], 1)[:, 1].cpu().numpy()); Y.extend(y.numpy())
        return roc_auc_score(Y, P)

    for ep in range(a.epochs):
        model.train()
        for x, y, vc, vw in tl:
            x, y, vc, vw = x.to(DEV), y.to(DEV), vc.to(DEV), vw.to(DEV)
            opt.zero_grad()
            logit, vpred = model(x)
            lcls = ce(logit, y)
            lv = (mse(torch.sigmoid(vpred), vc) * vw).sum() / (vw.sum() + 1e-6)
            (lcls + a.vcdr_w * lv).backward(); opt.step()
        va_auc = val_auc(); pa, _, _ = papila_auc(model, pap)
        if ep >= SWA_START:
            sd = model.state_dict()
            swa_sum = {k: v.detach().cpu().double() for k, v in sd.items()} if swa_sum is None else {k: swa_sum[k] + sd[k].detach().cpu().double() for k in swa_sum}
            swa_n += 1
        print(f"    epoch {ep+1}/{a.epochs}  src-val {va_auc:.3f}  [PAPILA {pa:.3f}]", flush=True)

    ref = model.state_dict()
    model.load_state_dict({k: (v / swa_n).to(ref[k].dtype).to(ref[k].device) for k, v in swa_sum.items()})
    final, Y, P = papila_auc(model, pap, tta=True)
    rng = np.random.default_rng(42); v = [roc_auc_score(Y[i], P[i]) for i in (rng.integers(0, len(Y), len(Y)) for _ in range(2000)) if len(np.unique(Y[i])) > 1]
    res = {"attempt": "B_multitask_VCDR", "held_out_papila_auroc_TTA": round(float(final), 4),
           "papila_auroc_95ci": [round(float(np.percentile(v, 2.5)), 4), round(float(np.percentile(v, 97.5)), 4)],
           "vcdr_weight": a.vcdr_w, "vcdr_supervised_on": "RIM-ONE only (reliable); HYGD masked out",
           "attempt_A_was": 0.8251}
    (ROOT / "results/generalize_attemptB.json").write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
