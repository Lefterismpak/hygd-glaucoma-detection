"""Reproduce a historical post-hoc quality/error comparison locally.

This selected development-split analysis is underpowered, image-level despite
repeated-patient structure, and unadjusted for selection or multiplicity. It
provides no evidence for or against a quality/error association and cannot justify
a quality threshold, preprocessing rule, clinical mechanism, or deployment claim.

Usage (venv active, from project root):
    python analyze_errors.py --model results/finetune_layer4_aug.pt --mode finetune_layer4
    python analyze_errors.py --model results/baseline_resnet18.pt --mode frozen   # v1 fallback
"""

import argparse
import io
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.stats import mannwhitneyu
from torch.utils.data import DataLoader

from src.data_utils import load_dataset_metadata, train_val_test_split
from src.experiments import TransformHYGDDataset, build_model, build_transforms
from validation.evaluation_utils import (
    assert_fresh_output_bundle,
    atomic_write_bytes,
    atomic_write_text,
    load_torch_state_dict_safely,
    resolve_private_output_path,
)

ROOT = Path(__file__).resolve().parent


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
    ap.add_argument(
        "--out",
        default="results/historical_notebook_runs/error_analysis.json",
    )
    ap.add_argument(
        "--figure-out",
        default="figures/local/error_vs_quality.png",
    )
    args = ap.parse_args()

    output_path = resolve_private_output_path(
        ROOT, args.out, "results/historical_notebook_runs"
    )
    figure_path = resolve_private_output_path(
        ROOT, args.figure_out, "figures/local"
    )
    assert_fresh_output_bundle([output_path, figure_path])

    df = load_dataset_metadata(ROOT / "data/raw")
    _, _, test_df = train_val_test_split(df, seed=42)
    test_df = test_df.reset_index(drop=True)

    loader = DataLoader(TransformHYGDDataset(test_df, build_transforms(train=False)), batch_size=32)
    model = build_model(mode=args.mode, pretrained=False)
    checkpoint = Path(args.model)
    if not checkpoint.is_absolute():
        checkpoint = ROOT / checkpoint
    state, checkpoint_sha256 = load_torch_state_dict_safely(
        checkpoint, map_location="cpu"
    )
    model.load_state_dict(state)

    y_prob = predict(model, loader)
    y_true = test_df["label"].values
    y_pred = (y_prob >= 0.5).astype(int)
    correct = y_pred == y_true

    q_correct = test_df.loc[correct, "quality_score"].values
    q_wrong = test_df.loc[~correct, "quality_score"].values

    result = {
        "_provenance": {
            "status": "historical_single_split_post_hoc_artifact",
            "canonical_internal_estimate": False,
            "row_level_artifact": False,
            "warning": (
                "Underpowered image-level post-hoc comparison; no evidence for or "
                "against a quality/error association and no quality gate justified."
            ),
        },
        "checkpoint": "operator_supplied_local_weights_only_state_dict",
        "checkpoint_sha256": checkpoint_sha256,
        "n_correct": int(correct.sum()),
        "n_wrong": int((~correct).sum()),
        "mean_quality_correct": float(np.mean(q_correct)) if len(q_correct) else None,
        "mean_quality_wrong": float(np.mean(q_wrong)) if len(q_wrong) else None,
    }

    # Historical exploratory one-sided comparison; not a confirmatory hypothesis test.
    if len(q_wrong) >= 2 and len(q_correct) >= 2:
        stat, p = mannwhitneyu(q_wrong, q_correct, alternative="less")
        result["mannwhitney_u"] = float(stat)
        result["p_value_wrong_lower_quality"] = float(p)

    # Error breakdown by type
    fn = int(((y_pred == 0) & (y_true == 1)).sum())  # missed glaucoma
    fp = int(((y_pred == 1) & (y_true == 0)).sum())  # false alarm
    result["false_negatives_missed_glaucoma"] = fn
    result["false_positives_false_alarm"] = fp

    atomic_write_text(output_path, json.dumps(result, indent=2) + "\n")
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
    ax.set_title("HISTORICAL DEVELOPMENT ARTIFACT — NONCANONICAL")
    ax.legend()
    plt.tight_layout()
    figure_bytes = io.BytesIO()
    plt.savefig(figure_bytes, format="png", dpi=150)
    plt.close()
    atomic_write_bytes(figure_path, figure_bytes.getvalue())
    print(f"saved local-only figure to {figure_path}")


if __name__ == "__main__":
    main()
