"""Historical, noncanonical v2 development-comparison runner.

Produces a model-comparison artifact from one development test split. It does
not produce the primary internal estimate; run
``validation/internal_evaluation_repair.py`` for that evaluation.

Run with the project venv active (paths are anchored to this repository):
    python run_v2_experiments.py --run-historical-comparison
"""

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RESULTS_DIR = ROOT / "results"
HISTORICAL_NAMESPACE = "results/historical_notebook_runs"
sys.path.insert(0, str(ROOT))

DEVICE = "cpu"
EPOCHS = 10
BATCH = 32
SEED = 42
POLICY = {
    "status": "historical_development_only",
    "canonical_internal_estimate": False,
    "generates_cross_validation": False,
    "configuration_selection_permitted": False,
    "primary_command": "python validation/internal_evaluation_repair.py",
}


def load_training_stack():
    """Load heavyweight dependencies only after explicit historical opt-in."""

    global np, torch, DataLoader
    global load_dataset_metadata, train_val_test_split
    global TransformHYGDDataset, build_model, build_transforms
    global metrics_at_threshold, threshold_for_target_sensitivity, class_weights
    global finalize_historical_comparison, historical_image_bootstrap_ci
    global assert_fresh_output_bundle, atomic_torch_save, atomic_write_text
    global read_confined_json, resolve_private_output_path

    import numpy as numpy_module
    import torch as torch_module
    from torch.utils.data import DataLoader as data_loader

    from src.data_utils import load_dataset_metadata as load_metadata
    from src.data_utils import train_val_test_split as split_metadata
    from src.experiments import (
        TransformHYGDDataset as transformed_dataset,
        build_model as make_model,
        build_transforms as make_transforms,
        metrics_at_threshold as threshold_metrics,
        threshold_for_target_sensitivity as target_threshold,
    )
    from src.train import class_weights as compute_class_weights
    from validation.evaluation_utils import (
        assert_fresh_output_bundle as require_fresh_bundle,
        atomic_torch_save as safe_torch_save,
        atomic_write_text as safe_text_write,
        finalize_historical_comparison as finalize_comparison,
        historical_image_bootstrap_ci as historical_bootstrap,
        read_confined_json as safe_json_read,
        resolve_private_output_path as confine_output,
    )

    np = numpy_module
    torch = torch_module
    DataLoader = data_loader
    load_dataset_metadata = load_metadata
    train_val_test_split = split_metadata
    TransformHYGDDataset = transformed_dataset
    build_model = make_model
    build_transforms = make_transforms
    metrics_at_threshold = threshold_metrics
    threshold_for_target_sensitivity = target_threshold
    class_weights = compute_class_weights
    finalize_historical_comparison = finalize_comparison
    historical_image_bootstrap_ci = historical_bootstrap
    assert_fresh_output_bundle = require_fresh_bundle
    atomic_torch_save = safe_torch_save
    atomic_write_text = safe_text_write
    read_confined_json = safe_json_read
    resolve_private_output_path = confine_output


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


def predict(model, loader):
    model.eval()
    ys, ps = [], []
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(DEVICE)
            prob = torch.softmax(model(images), dim=1)[:, 1]
            ys.extend(labels.numpy().tolist())
            ps.extend(prob.cpu().numpy().tolist())
    return np.array(ys), np.array(ps)


def main(args):
    load_training_stack()
    output_prefix = resolve_private_output_path(
        ROOT, args.output_prefix, HISTORICAL_NAMESPACE
    )
    checkpoint_paths = {
        name: Path(f"{output_prefix}_{name}.pt")
        for name in ("frozen_aug", "finetune_layer4_aug")
    }
    result_path = Path(f"{output_prefix}.json")
    assert_fresh_output_bundle([*checkpoint_paths.values(), result_path])
    torch.manual_seed(SEED)
    df = load_dataset_metadata(ROOT / "data" / "raw")
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
        ci = historical_image_bootstrap_ci(y_true, y_prob, 0.5)
        thr = threshold_for_target_sensitivity(y_true, y_prob, 0.95)
        results[name] = {
            "test_auc": float(m["auc"]),
            "test_sensitivity": float(m["sensitivity"]),
            "test_specificity": float(m["specificity"]),
            "confusion_matrix_at_0.5": m["confusion_matrix"].tolist(),
            "historical_image_level_bootstrap": ci,
            "historical_test_derived_threshold_sens>=0.95_nondeployment": thr,
            "train_seconds": round(time.time() - t0, 1),
            "history": history,
        }
        results[name]["checkpoint_sha256"] = atomic_torch_save(
            checkpoint_paths[name], model.state_dict()
        )
        print(f"  {name}: AUC={m['auc']:.3f} sens={m['sensitivity']:.3f} spec={m['specificity']:.3f} "
              f"({results[name]['train_seconds']}s)", flush=True)

    # Include the v1 frozen baseline number for reference (already computed earlier).
    try:
        v1 = read_confined_json(ROOT, "results/metrics.json")
        results["frozen_noaug_v1"] = {
            "test_auc": v1["auc"],
            "test_sensitivity": v1["sensitivity"],
            "test_specificity": v1["specificity"],
            "note": "Phase-4 baseline, no augmentation, from results/metrics.json",
        }
    except FileNotFoundError:
        pass

    results = finalize_historical_comparison(results)
    atomic_write_text(result_path, json.dumps(results, indent=2) + "\n")
    print(f"\nsaved {result_path.relative_to(ROOT)}", flush=True)

    print(
        "\nHistorical CV generation is disabled: it selected the configuration on "
        "the development test set and reused each fold for checkpoint selection "
        "and scoring. Run validation/internal_evaluation_repair.py for the fixed, "
        "duplicate-aware nested evaluation.",
        flush=True,
    )


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group()
    action.add_argument(
        "--describe-policy",
        action="store_true",
        help="Print the noncanonical historical-runner policy without loading ML dependencies.",
    )
    action.add_argument(
        "--run-historical-comparison",
        action="store_true",
        help="Explicitly reproduce the historical single-split comparison; no CV is generated.",
    )
    parser.add_argument(
        "--output-prefix",
        default="results/historical_notebook_runs/v2_comparison_run",
        help="Fresh prefix directly under the ignored historical notebook namespace.",
    )
    args = parser.parse_args(argv)
    if not args.describe_policy and not args.run_historical_comparison:
        parser.error(
            "This runner is historical only. Use --run-historical-comparison "
            "to reproduce it, or run validation/internal_evaluation_repair.py "
            "for the canonical internal evaluation."
        )
    return args


if __name__ == "__main__":
    arguments = parse_args()
    if arguments.describe_policy:
        print(json.dumps(POLICY, indent=2))
    else:
        main(arguments)
