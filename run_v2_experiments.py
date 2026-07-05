"""v2 experiment runner. Produces results/v2_comparison.json and updated figures.

Run from the project root with the venv active:
    python run_v2_experiments.py
"""

import json
import sys
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, ".")
from src.data_utils import load_dataset_metadata, train_val_test_split
from src.experiments import (
    TransformHYGDDataset,
    bootstrap_ci,
    build_model,
    build_transforms,
    metrics_at_threshold,
    patient_kfold,
    threshold_for_target_sensitivity,
)
from src.train import class_weights

DEVICE = "cpu"
EPOCHS = 10
BATCH = 32
SEED = 42


def train_one(model, train_loader, val_loader, weights, epochs=EPOCHS, head_lr=1e-3, backbone_lr=1e-4):
    """Train with discriminative LRs: new head fast, unfrozen backbone params slow."""
    model.to(DEVICE)
    criterion = torch.nn.CrossEntropyLoss(weight=weights.to(DEVICE))

    head_params, backbone_params = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (head_params if name.startswith("fc.") else backbone_params).append(p)
    param_groups = [{"params": head_params, "lr": head_lr}]
    if backbone_params:
        param_groups.append({"params": backbone_params, "lr": backbone_lr})
    optimizer = torch.optim.Adam(param_groups)

    history = {"train_loss": [], "val_loss": []}
    best_state, best_val = None, float("inf")
    for epoch in range(epochs):
        model.train()
        run = 0.0
        for images, labels in train_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(images), labels)
            loss.backward()
            optimizer.step()
            run += loss.item() * images.size(0)
        tr = run / len(train_loader.dataset)

        model.eval()
        vrun = 0.0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                vrun += criterion(model(images), labels).item() * images.size(0)
        vl = vrun / len(val_loader.dataset)

        history["train_loss"].append(tr)
        history["val_loss"].append(vl)
        if vl < best_val:  # simple early-stopping: keep best-val checkpoint
            best_val, best_state = vl, {k: v.clone() for k, v in model.state_dict().items()}
        print(f"    epoch {epoch+1}/{epochs}  train={tr:.4f}  val={vl:.4f}", flush=True)

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, history


@torch.no_grad()
def predict(model, loader):
    model.eval()
    ys, ps = [], []
    for images, labels in loader:
        images = images.to(DEVICE)
        prob = torch.softmax(model(images), dim=1)[:, 1]
        ys.extend(labels.numpy().tolist())
        ps.extend(prob.cpu().numpy().tolist())
    return np.array(ys), np.array(ps)


def main():
    torch.manual_seed(SEED)
    df = load_dataset_metadata("data/raw")
    train_df, val_df, test_df = train_val_test_split(df, seed=SEED)
    weights = class_weights(train_df["label"].values)

    eval_tf = build_transforms(train=False)
    train_tf = build_transforms(train=True)

    val_loader = DataLoader(TransformHYGDDataset(val_df, eval_tf), batch_size=BATCH)
    test_loader = DataLoader(TransformHYGDDataset(test_df, eval_tf), batch_size=BATCH)

    configs = {
        "frozen_aug": "frozen",
        "finetune_layer4_aug": "finetune_layer4",
    }

    results = {}
    for name, mode in configs.items():
        print(f"\n=== training {name} (mode={mode}) ===", flush=True)
        t0 = time.time()
        train_loader = DataLoader(
            TransformHYGDDataset(train_df, train_tf), batch_size=BATCH, shuffle=True
        )
        model = build_model(mode=mode)
        model, history = train_one(model, train_loader, val_loader, weights)
        y_true, y_prob = predict(model, test_loader)
        m = metrics_at_threshold(y_true, y_prob, 0.5)
        ci = bootstrap_ci(y_true, y_prob, 0.5)
        thr = threshold_for_target_sensitivity(y_true, y_prob, 0.95)
        results[name] = {
            "test_auc": float(m["auc"]),
            "test_sensitivity": float(m["sensitivity"]),
            "test_specificity": float(m["specificity"]),
            "confusion_matrix_at_0.5": m["confusion_matrix"].tolist(),
            "bootstrap_95ci": ci,
            "screening_threshold_sens>=0.95": thr,
            "train_seconds": round(time.time() - t0, 1),
            "history": history,
        }
        torch.save(model.state_dict(), f"results/{name}.pt")
        print(f"  {name}: AUC={m['auc']:.3f} sens={m['sensitivity']:.3f} spec={m['specificity']:.3f} "
              f"({results[name]['train_seconds']}s)", flush=True)

    # Include the v1 frozen baseline number for reference (already computed earlier).
    try:
        with open("results/metrics.json") as f:
            v1 = json.load(f)
        results["frozen_noaug_v1"] = {
            "test_auc": v1["auc"],
            "test_sensitivity": v1["sensitivity"],
            "test_specificity": v1["specificity"],
            "note": "Phase-4 baseline, no augmentation, from results/metrics.json",
        }
    except FileNotFoundError:
        pass

    with open("results/v2_comparison.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nsaved results/v2_comparison.json", flush=True)

    # ---- 5-fold patient-level CV on the better of the two v2 configs ----
    best_name = max(configs, key=lambda n: results[n]["test_auc"])
    best_mode = configs[best_name]
    print(f"\n=== 5-fold patient-level CV on best config: {best_name} ===", flush=True)
    trainval_df = df  # use the whole dataset for CV (independent of the single split above)
    fold_aucs = []
    for i, (ftrain, fval) in enumerate(patient_kfold(trainval_df, n_splits=5, seed=SEED), 1):
        print(f"  fold {i}/5 ...", flush=True)
        w = class_weights(ftrain["label"].values)
        tl = DataLoader(TransformHYGDDataset(ftrain, train_tf), batch_size=BATCH, shuffle=True)
        vl = DataLoader(TransformHYGDDataset(fval, eval_tf), batch_size=BATCH)
        model = build_model(mode=best_mode)
        model, _ = train_one(model, tl, vl, w, epochs=EPOCHS)
        yt, yp = predict(model, vl)
        m = metrics_at_threshold(yt, yp, 0.5)
        fold_aucs.append(float(m["auc"]))
        print(f"    fold {i} AUC={m['auc']:.3f}", flush=True)

    cv = {
        "config": best_name,
        "fold_aucs": fold_aucs,
        "mean_auc": float(np.mean(fold_aucs)),
        "std_auc": float(np.std(fold_aucs)),
    }
    with open("results/cv_results.json", "w") as f:
        json.dump(cv, f, indent=2)
    print(f"\n5-fold CV AUC: {cv['mean_auc']:.3f} +/- {cv['std_auc']:.3f}", flush=True)
    print("saved results/cv_results.json", flush=True)


if __name__ == "__main__":
    main()
