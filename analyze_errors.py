"""Data-driven error analysis: are misclassifications concentrated in low-quality images?

This is the *quantitative* half of the Phase-5 clinical error analysis. It does NOT
replace the human/ophthalmology reflection (still a TODO in notebook 04) — it just
tests one concrete, checkable hypothesis: does the model fail more on images the
FundusQ-Net quality score already flags as poor? If yes, that's a real, honest
explanation for a chunk of the errors (and an argument for a quality gate at
inference time). If no, the errors are about something subtler than image quality.

Usage (venv active, from project root):
    python analyze_errors.py --model results/finetune_layer4_aug.pt --mode finetune_layer4
    python analyze_errors.py --model results/baseline_resnet18.pt --mode frozen   # v1 fallback
"""

import argparse
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.stats import mannwhitneyu
from torch.utils.data import DataLoader

from src.data_utils import load_dataset_metadata, train_val_test_split
from src.experiments import TransformHYGDDataset, build_model, build_transforms


@torch.no_grad()
def predict(model, loader):
    model.eval()
    ps = []
    for images, _ in loader:
        prob = torch.softmax(model(images), dim=1)[:, 1]
        ps.extend(prob.numpy().tolist())
    return np.array(ps)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="results/finetune_layer4_aug.pt")
    ap.add_argument("--mode", default="finetune_layer4")
    args = ap.parse_args()

    df = load_dataset_metadata("data/raw")
    _, _, test_df = train_val_test_split(df, seed=42)
    test_df = test_df.reset_index(drop=True)

    loader = DataLoader(TransformHYGDDataset(test_df, build_transforms(train=False)), batch_size=32)
    model = build_model(mode=args.mode)
    model.load_state_dict(torch.load(args.model, map_location="cpu"))

    y_prob = predict(model, loader)
    y_true = test_df["label"].values
    y_pred = (y_prob >= 0.5).astype(int)
    correct = y_pred == y_true

    q_correct = test_df.loc[correct, "quality_score"].values
    q_wrong = test_df.loc[~correct, "quality_score"].values

    result = {
        "model": args.model,
        "n_correct": int(correct.sum()),
        "n_wrong": int((~correct).sum()),
        "mean_quality_correct": float(np.mean(q_correct)) if len(q_correct) else None,
        "mean_quality_wrong": float(np.mean(q_wrong)) if len(q_wrong) else None,
    }

    # Mann-Whitney U: is the quality-score distribution of wrong predictions lower?
    if len(q_wrong) >= 2 and len(q_correct) >= 2:
        stat, p = mannwhitneyu(q_wrong, q_correct, alternative="less")
        result["mannwhitney_u"] = float(stat)
        result["p_value_wrong_lower_quality"] = float(p)

    # Error breakdown by type
    fn = int(((y_pred == 0) & (y_true == 1)).sum())  # missed glaucoma
    fp = int(((y_pred == 1) & (y_true == 0)).sum())  # false alarm
    result["false_negatives_missed_glaucoma"] = fn
    result["false_positives_false_alarm"] = fp

    with open("results/error_analysis.json", "w") as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result, indent=2))

    # Figure: quality score distribution, correct vs wrong
    fig, ax = plt.subplots(figsize=(6, 4))
    bins = np.linspace(test_df["quality_score"].min(), test_df["quality_score"].max(), 15)
    ax.hist(q_correct, bins=bins, alpha=0.6, label=f"correct (n={len(q_correct)})", color="#55A868")
    ax.hist(q_wrong, bins=bins, alpha=0.6, label=f"wrong (n={len(q_wrong)})", color="#C44E52")
    ax.axvline(np.mean(q_correct), color="#55A868", linestyle="--")
    ax.axvline(np.mean(q_wrong), color="#C44E52", linestyle="--")
    ax.set_xlabel("FundusQ-Net quality score")
    ax.set_ylabel("Count")
    ax.set_title("Image quality: correct vs misclassified test images")
    ax.legend()
    plt.tight_layout()
    plt.savefig("figures/10_error_vs_quality.png", dpi=150)
    plt.close()
    print("saved figures/10_error_vs_quality.png")


if __name__ == "__main__":
    main()
