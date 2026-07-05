"""Cross-dataset fine-tune (GlaucoGen Tier-3): can a short fine-tune build a model
that transfers to a genuinely held-out dataset?

Starts from the HYGD checkpoint, unfreezes layer4 + fc, fine-tunes on a SOURCE
external dataset (with an internal val split for early stopping), and evaluates on
a TARGET external dataset that is never seen during fine-tuning. Reports target
AUROC before (zero-shot) vs after — the honest test of transfer. Fine-tuning and
testing on the same dataset is deliberately avoided (proves nothing about transfer).

Usage:
  python validation/finetune_crossdataset.py --source RIMONE --target PAPILA
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.experiments import build_model, build_transforms  # noqa: E402

CSV = {"PAPILA": "validation/data/papila_labels.csv", "RIMONE": "validation/data/rimone_labels.csv"}


class CsvDataset(Dataset):
    def __init__(self, df, transform):
        self.df = df.reset_index(drop=True); self.tf = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        r = self.df.iloc[i]
        return self.tf(Image.open(r["image_path"]).convert("RGB")), torch.tensor(int(r["label"]))


@torch.no_grad()
def auroc_on(model, df):
    model.eval()
    tf = build_transforms(train=False)
    p, y = [], []
    for _, r in df.iterrows():
        x = tf(Image.open(r["image_path"]).convert("RGB")).unsqueeze(0)
        p.append(torch.softmax(model(x), 1)[0, 1].item()); y.append(int(r["label"]))
    return float(roc_auc_score(y, p)), np.array(p), np.array(y)


def main():
    import pandas as pd
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True, choices=list(CSV))
    ap.add_argument("--target", required=True, choices=list(CSV))
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    torch.manual_seed(args.seed)

    src = pd.read_csv(ROOT / CSV[args.source])
    tgt = pd.read_csv(ROOT / CSV[args.target])
    # internal train/val split of the SOURCE for early stopping (stratified by label)
    tr, va = train_test_split(src, test_size=0.15, stratify=src["label"], random_state=args.seed)

    model = build_model(mode="finetune_layer4")
    model.load_state_dict(torch.load(ROOT / "results/finetune_layer4_aug.pt", map_location="cpu"))

    tgt_before, _, _ = auroc_on(model, tgt)
    print(f"[{args.source}->{args.target}] target AUROC BEFORE (zero-shot): {tgt_before:.3f}", flush=True)

    counts = np.bincount(tr["label"].values, minlength=2)
    w = torch.tensor((counts.sum() / (2 * np.clip(counts, 1, None))), dtype=torch.float32)
    crit = nn.CrossEntropyLoss(weight=w)
    opt = torch.optim.Adam([
        {"params": model.fc.parameters(), "lr": 1e-3},
        {"params": model.layer4.parameters(), "lr": 1e-4},
    ])
    tl = DataLoader(CsvDataset(tr, build_transforms(train=True)), batch_size=32, shuffle=True)

    best_val, best_state = -1.0, None
    for ep in range(args.epochs):
        model.train()
        for x, y in tl:
            opt.zero_grad(); loss = crit(model(x), y); loss.backward(); opt.step()
        val_auc, _, _ = auroc_on(model, va)
        if val_auc > best_val:
            best_val, best_state = val_auc, {k: v.clone() for k, v in model.state_dict().items()}
        print(f"    epoch {ep+1}/{args.epochs}  source-val AUROC {val_auc:.3f}", flush=True)

    model.load_state_dict(best_state)
    tgt_after, _, _ = auroc_on(model, tgt)
    print(f"[{args.source}->{args.target}] target AUROC AFTER fine-tune: {tgt_after:.3f}  (was {tgt_before:.3f})", flush=True)

    res = {"source": args.source, "target": args.target,
           "target_auroc_zero_shot": round(tgt_before, 4),
           "target_auroc_after_finetune": round(tgt_after, 4),
           "best_source_val_auroc": round(best_val, 4),
           "delta": round(tgt_after - tgt_before, 4)}
    out = ROOT / "results" / f"finetune_{args.source.lower()}_to_{args.target.lower()}.json"
    out.write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
