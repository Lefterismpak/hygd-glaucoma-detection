"""Adaptive cross-dataset fine-tune diagnostic for authorized local research.

Starts from the HYGD checkpoint, unfreezes layer4 + fc, fine-tunes on a SOURCE
external dataset (with an internal val split for early stopping), and evaluates on
a TARGET external dataset excluded from gradient updates. Target outcomes were
visible in the wider development chronology, so the output is adaptive evidence,
not an untouched transfer or transportability test.

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
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.experiments import build_model, build_transforms  # noqa: E402
from validation.evaluation_utils import (  # noqa: E402
    assert_fresh_output_bundle,
    atomic_write_text,
    load_hash_bound_subject_map,
    load_torch_state_dict_safely,
    resolve_private_output_path,
    stratified_group_holdout,
    validate_external_label_frame,
)

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
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--checkpoint-sha256", required=True)
    ap.add_argument("--source-subject-map")
    ap.add_argument("--source-subject-map-sha256")
    ap.add_argument("--acknowledge-external-data-license-needs-proof", action="store_true")
    ap.add_argument("--out")
    args = ap.parse_args()
    if args.source == args.target:
        raise ValueError("source and target must be different datasets")
    if not args.acknowledge_external_data_license_needs_proof:
        raise ValueError(
            "External-data mixed-use compatibility is needs-proof; pass the explicit "
            "acknowledgement only for an authorized local research run."
        )
    if bool(args.source_subject_map) != bool(args.source_subject_map_sha256):
        raise ValueError(
            "--source-subject-map and --source-subject-map-sha256 are required together"
        )
    if args.source == "RIMONE" and not args.source_subject_map:
        raise ValueError("RIMONE source runs require a hash-bound subject map")
    if args.source != "RIMONE" and args.source_subject_map:
        raise ValueError("A source subject map is only accepted for RIMONE")
    if args.epochs <= 0:
        raise ValueError("--epochs must be positive")
    requested_out = (
        args.out
        or f"results/patient_aware/finetune_{args.source.lower()}_to_{args.target.lower()}_v2.json"
    )
    out = resolve_private_output_path(ROOT, requested_out, "results/patient_aware")
    assert_fresh_output_bundle([out])
    torch.manual_seed(args.seed)

    src = validate_external_label_frame(
        pd.read_csv(ROOT / CSV[args.source]), context=f"{args.source} source labels"
    )
    tgt = validate_external_label_frame(
        pd.read_csv(ROOT / CSV[args.target]), context=f"{args.target} target labels"
    )
    subject_mapping_sha256 = None
    if args.source == "RIMONE":
        source_stems = src["image_path"].map(lambda value: Path(value).stem)
        mapping, subject_mapping_sha256 = load_hash_bound_subject_map(
            source_stems,
            args.source_subject_map,
            args.source_subject_map_sha256,
        )
        src["patient_id"] = source_stems.map(mapping)
    train_indices, validation_indices = stratified_group_holdout(
        src, "patient_id", "label", test_fraction=0.15, seed=args.seed
    )
    tr = src.iloc[train_indices].reset_index(drop=True)
    va = src.iloc[validation_indices].reset_index(drop=True)
    if set(tr["patient_id"]) & set(va["patient_id"]):
        raise AssertionError("Source train and validation patients overlap")

    model = build_model(mode="finetune_layer4", pretrained=False)
    state, initial_checkpoint_sha256 = load_torch_state_dict_safely(
        args.checkpoint,
        map_location="cpu",
        expected_sha256=args.checkpoint_sha256,
    )
    model.load_state_dict(state)

    tgt_before, _, _ = auroc_on(model, tgt)

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
    res = {"_evidence_status": {
               "status": "adaptive_development_evidence",
               "target_blind": False,
               "transportability_established": False,
               "license_compatibility": "needs-proof",
               "external_claim_permitted": False,
               "source_patient_disjoint_split": True,
               "source_subject_identity": (
                   "operator_asserted_hash_bound_needs_independent_proof"
                   if subject_mapping_sha256
                   else "supplied_dataset_identifier_needs_proof"
               ),
           },
           "source": args.source, "target": args.target,
           "source_subject_mapping_sha256": subject_mapping_sha256,
           "initial_checkpoint_sha256": initial_checkpoint_sha256,
           "target_auroc_zero_shot": round(tgt_before, 4),
           "target_auroc_after_finetune": round(tgt_after, 4),
           "best_source_val_auroc": round(best_val, 4),
           "delta": round(tgt_after - tgt_before, 4)}
    atomic_write_text(out, json.dumps(res, indent=2) + "\n")
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
