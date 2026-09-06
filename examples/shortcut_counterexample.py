"""A dataset-free counterexample: encoding source does not prove reliance.

Two fixed prediction heads share the SAME representation. One reads the disease
coordinate; the other reads the source coordinate. Both representations permit
perfect source decoding. Only the source head collapses when the constructed
source/outcome association reverses. This illustrates a logical distinction; it
does not diagnose any real medical model or estimate deployment performance.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score


def run_counterexample(*, seed: int = 20260906) -> dict:
    rng = np.random.default_rng(seed)
    result = {
        "evidence_type": "synthetic_counterexample_not_medical_evidence",
        "seed": seed,
        "samples_per_environment": 2000,
        "medical_data_used": False,
        "models_trained": False,
        "representation": "same two coordinates for both heads: noisy disease and exact source",
        "intervention": "constructed reversal of source/outcome association; no change in disease mechanism",
    }
    for environment, agreement in [("development", 0.95), ("reversed_source_association", 0.05)]:
        labels = np.tile([0, 1], 1000)
        sources = np.where(rng.random(len(labels)) < agreement, labels, 1 - labels)
        disease_coordinate = (2 * labels - 1) + rng.normal(0, 0.8, len(labels))
        source_coordinate = 2 * sources - 1
        result[environment] = {
            "source_decoding_accuracy": float(np.mean((source_coordinate > 0) == sources)),
            "disease_head_auroc": float(roc_auc_score(labels, disease_coordinate)),
            "source_head_auroc": float(roc_auc_score(labels, source_coordinate)),
        }
    return result


def draw_figure(result: dict, output: Path) -> None:
    # Matplotlib is optional; the JSON example and all tests need no plot package.
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    with plt.rc_context({"font.family": "DejaVu Sans", "font.size": 11, "svg.fonttype": "none"}):
        figure, axes = plt.subplots(1, 2, figsize=(11, 5))
        figure.subplots_adjust(top=0.74, bottom=0.25, wspace=0.28)
        figure.suptitle("Source encoding is not the same as model reliance", fontsize=18, fontweight="bold", y=0.97)
        figure.text(0.5, 0.865, "SYNTHETIC COUNTEREXAMPLE  /  same representation, two fixed prediction heads", ha="center", fontsize=10, color="#46536a")
        colors = ["#157f82", "#c85b45"]
        for ax, environment, title in zip(axes, ["development", "reversed_source_association"], ["Source agrees with outcome (95%)", "Source association reverses (5%)"]):
            values = [result[environment]["disease_head_auroc"], result[environment]["source_head_auroc"]]
            bars = ax.bar([0, 1], values, color=colors, width=0.55)
            ax.set_xticks([0, 1], ["Disease coordinate", "Source coordinate"])
            ax.set_ylim(0, 1.13)
            ax.set_ylabel("AUROC")
            ax.set_title(title, fontsize=12, pad=15)
            ax.axhline(0.5, color="#677080", linestyle="--", linewidth=1)
            ax.spines[["top", "right"]].set_visible(False)
            for bar, value in zip(bars, values):
                ax.text(bar.get_x() + bar.get_width()/2, value + 0.025, f"{value:.3f}", ha="center", fontweight="bold")
        figure.text(0.5, 0.13, "Source decoding accuracy = 100% in both environments and for both heads.", ha="center", fontsize=11, fontweight="bold")
        figure.text(0.5, 0.067, "No medical data or trained model. This is a logical counterexample, not HYGD performance evidence.", ha="center", fontsize=9, color="#46536a")
        if output.exists():
            raise FileExistsError("Choose a new figure path; existing artifacts are preserved")
        figure.savefig(output, dpi=180, metadata={"Creator": "HYGD synthetic counterexample", "Date": None} if output.suffix == ".svg" else {"Software": "HYGD synthetic counterexample"})
        plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--figure", type=Path, help="Optional new PNG or SVG path (synthetic aggregate only)")
    args = parser.parse_args()
    result = run_counterexample()
    if args.figure:
        if args.figure.suffix not in {".png", ".svg"}:
            parser.error("Figure must end in .png or .svg")
        draw_figure(result, args.figure)
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
