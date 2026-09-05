"""Attempt A — multi-source training for cross-dataset transfer.

Retrospective status correction: this is adaptive development evidence, not an
untouched external validation. Target AUROC was displayed during development,
and target-domain anatomical resources influenced preprocessing.

Train ResNet-18 (ImageNet init, unfreeze layer3+layer4+fc) on disc-standardized,
colour-normalized crops from HYGD + RIM-ONE (two source domains), with heavy
colour/geometry augmentation + domain-balanced sampling + class-weighting.
PAPILA disease labels are excluded from the final classification loss, but target
data/resources and AUROC were visible in the adaptive chronology. This is not an
untouched external-validation or transportability experiment.

Future runs require an explicit hash-bound, operator-supplied RIM-ONE
stem-to-subject mapping. Its biological correctness remains ``needs-proof``.
They use patient-cluster PAPILA uncertainty. Historical row-bootstrap values
stay in the tracked legacy JSON with noncanonical provenance.
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
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import models, transforms

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from validation.evaluation_utils import (  # noqa: E402
    assert_fresh_output_bundle,
    atomic_torch_save,
    atomic_write_text,
    load_hash_bound_subject_map,
    patient_cluster_auc_ci,
    resolve_private_output_path,
    stratified_group_holdout,
)
DATA = ROOT / "validation/data"
DEV = "mps" if torch.backends.mps.is_available() else "cpu"
IM_MEAN, IM_STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]


def labeled_crops(rimone_subject_map, rimone_subject_map_sha256):
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
    rimone_mapping, mapping_sha256 = load_hash_bound_subject_map(
        r["stem"], rimone_subject_map, rimone_subject_map_sha256
    )
    r["patient_id"] = "R" + r["stem"].map(rimone_mapping)
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
    return pd.DataFrame(rows), mapping_sha256


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
    return (
        roc_auc_score(Y, P),
        np.array(P),
        np.array(Y),
        df["patient_id"].astype(str).to_numpy(),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--rimone-subject-map", required=True)
    ap.add_argument("--rimone-subject-map-sha256", required=True)
    ap.add_argument("--acknowledge-rimone-mixed-use-needs-proof", action="store_true")
    ap.add_argument(
        "--out",
        default="results/patient_aware/generalize_attemptA_patient_cluster_v2.json",
    )
    a = ap.parse_args()
    if not a.acknowledge_rimone_mixed_use_needs_proof:
        raise ValueError(
            "RIM-ONE mixed-source license compatibility is needs-proof; pass the "
            "explicit acknowledgement only for an authorized local research run."
        )
    if a.epochs <= 4:
        raise ValueError("--epochs must be greater than the fixed SWA start index 4")
    checkpoint_path = resolve_private_output_path(
        ROOT,
        "results/patient_aware/generalize_multisource_v2.pt",
        "results/patient_aware",
    )
    out_path = resolve_private_output_path(ROOT, a.out, "results/patient_aware")
    assert_fresh_output_bundle([checkpoint_path, out_path])
    torch.manual_seed(0)
    df, rimone_subject_map_sha256 = labeled_crops(
        a.rimone_subject_map, a.rimone_subject_map_sha256
    )
    src = df[df.dataset.isin(["HYGD", "RIMONE"])].reset_index(drop=True)
    papila = df[df.dataset == "PAPILA"].reset_index(drop=True)
    print(f"sources: {len(src)} ({dict(src.dataset.value_counts())}) | held-out PAPILA: {len(papila)}")

    tri, vai = stratified_group_holdout(
        src, "patient_id", "label", test_fraction=0.15, seed=0
    )
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

    # SWA uses no target label for its within-run weights, but PAPILA AUROC was
    # displayed throughout this adaptive development chronology. That history
    # prevents an untouched/target-blind external-validation claim.
    SWA_START = 4
    swa_sum, swa_n = None, 0
    for ep in range(a.epochs):
        model.train()
        for x, y in tl:
            opt.zero_grad(); loss = crit(model(x.to(DEV)), y.to(DEV)); loss.backward(); opt.step()
        va_auc, _, _, _ = auc(model, va)
        pap_auc, _, _, _ = auc(model, papila)
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
    swa_val, _, _, _ = auc(model, va)
    final_pap, pP, pY, pids = auc(model, papila)
    patient_ci = patient_cluster_auc_ci(pY, pP, pids, n_bootstrap=2000, seed=42)

    checkpoint_sha256 = atomic_torch_save(checkpoint_path, swa_state)
    res = {"_evidence_status": {
               "status": "adaptive_development_evidence",
               "target_blind": False,
               "transportability_established": False,
               "license_compatibility": "needs-proof",
               "external_claim_permitted": False,
               "rimone_subject_identity": "operator_asserted_hash_bound_needs_independent_proof",
               "warning": "Target AUROC was displayed during development and target-domain anatomical resources influenced preprocessing."
           },
           "rimone_subject_mapping_sha256": rimone_subject_map_sha256,
           "checkpoint_sha256": checkpoint_sha256,
           "attempt": "A_multisource_disccrop_colornorm_SWA",
           "adaptive_papila_eye_level_auroc": round(float(final_pap), 4),
           "papila_patient_cluster_uncertainty": patient_ci,
           "model_selection": "SWA weight-average over epochs>=%d using source-validation logic inside the final run; broader development was target-visible" % (SWA_START + 1),
           "swa_source_val_auroc": round(float(swa_val), 4),
           "prior_zero_shot_papila": 0.5077, "prior_singlesource_finetune": 0.6834,
           "levers": "auto-disc-crop (U-Net Dice 0.958) + shades-of-gray+CLAHE colour-norm + 2-source (HYGD+RIM-ONE) + heavy colour aug + domain/class-balanced sampler"}
    atomic_write_text(out_path, json.dumps(res, indent=2) + "\n")
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
