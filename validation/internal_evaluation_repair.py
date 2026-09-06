"""Leakage-resistant internal evaluation for the HYGD classifier.

This runner supersedes the historical 5-fold robustness estimate in
``run_v2_experiments.py``. It fixes four methodological problems:

1. Exact duplicate files are detected by SHA-256, linked across patient IDs,
   and counted once.
2. Outer test folds are separated from the inner validation split used for
   checkpoint and threshold selection.
3. The model configuration is fixed before outer-fold evaluation.
4. Results include image-level and duplicate-aware patient-group-level metrics,
   out-of-fold predictions, and cluster-bootstrap confidence intervals.

Residual limitation: the fixed configuration was chosen during earlier HYGD
development. This run repairs duplicate, checkpoint, threshold, and outer-fold
leakage, but it cannot make the overall model-development process prospectively
untouched. A new locked external target is required for confirmatory inference.

The outer test fold is evaluated exactly once, after training and all inner
validation decisions are complete.

Usage:
    python validation/internal_evaluation_repair.py --audit-only
    python validation/internal_evaluation_repair.py
"""

from __future__ import annotations

import argparse
import json
import platform
import random
import sys
import time
from importlib.metadata import version as package_version
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data_utils import load_dataset_metadata  # noqa: E402
from src.experiments import (  # noqa: E402
    TransformHYGDDataset,
    build_model,
    build_transforms,
)
from src.train import class_weights  # noqa: E402
from src.loss_utils import cross_entropy_normalizer  # noqa: E402
from validation.evaluation_utils import (  # noqa: E402
    DEFAULT_SEED,
    EXPECTED_HYGD_DATASET_IDENTITY,
    PROTOCOL_VERSION,
    aggregate_groups,
    assert_fresh_output_bundle,
    atomic_write_text,
    build_primary_report,
    build_dataset_identity,
    build_smoke_diagnostics,
    classify_run_mode,
    cluster_bootstrap,
    console_data_quality_summary,
    finalize_audit,
    fold_completion_message,
    fold_stratified_group_bootstrap,
    frame_metrics,
    inner_group_split,
    make_outer_folds,
    outer_fold_auc_summary,
    prepare_duplicate_aware_metadata,
    read_confined_json,
    resolve_run_output_prefix,
    select_threshold,
    sha256_file,
    terminal_result_claim_sha256,
    validate_internal_result_payload,
    validate_canonical_data_quality,
    validate_canonical_dataset_identity,
    validate_canonical_group_fold_assignment,
    validate_oof_predictions,
    validate_canonical_oof_identity,
    validate_smoke_result_payload,
    validate_smoke_prediction_execution,
    validate_terminal_bundle,
    validate_terminal_smoke_bundle,
)


def set_all_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(requested):
    if requested != "auto":
        return requested
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def predict_probabilities(model, metadata, transform, batch_size, device):
    loader = DataLoader(
        TransformHYGDDataset(
            metadata,
            transform,
            require_verified_sha256=True,
        ),
        batch_size=batch_size,
        shuffle=False,
    )
    labels, probabilities = [], []
    model.eval()
    with torch.no_grad():
        for images, batch_labels in loader:
            logits = model(images.to(device))
            probabilities.extend(torch.softmax(logits, dim=1)[:, 1].cpu().numpy().tolist())
            labels.extend(batch_labels.numpy().tolist())
    expected = metadata["label"].astype(int).tolist()
    if labels != expected:
        raise AssertionError("Prediction order no longer matches metadata order.")
    return np.asarray(probabilities, dtype=float)


def train_with_inner_validation(
    train_metadata,
    validation_metadata,
    epochs,
    batch_size,
    device,
    seed,
    *,
    verbose=True,
    global_weighted_loss=False,
):
    """Train the fixed configuration and select the epoch using inner val loss only."""

    set_all_seeds(seed)
    model = build_model(mode="finetune_layer4").to(device)
    weights = class_weights(train_metadata["label"].values).to(device)
    criterion = torch.nn.CrossEntropyLoss(weight=weights)

    head_parameters, backbone_parameters = [], []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        target = head_parameters if name.startswith("fc.") else backbone_parameters
        target.append(parameter)
    optimizer_groups = [{"params": head_parameters, "lr": 1e-3}]
    if backbone_parameters:
        optimizer_groups.append({"params": backbone_parameters, "lr": 1e-4})
    optimizer = torch.optim.Adam(optimizer_groups)

    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        TransformHYGDDataset(
            train_metadata,
            build_transforms(train=True),
            require_verified_sha256=True,
        ),
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
    )
    validation_loader = DataLoader(
        TransformHYGDDataset(
            validation_metadata,
            build_transforms(train=False),
            require_verified_sha256=True,
        ),
        batch_size=batch_size,
        shuffle=False,
    )

    history = []
    best_loss, best_epoch, best_state = float("inf"), None, None
    for epoch in range(1, epochs + 1):
        model.train()
        training_total = 0.0
        training_mass = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(images), labels)
            loss.backward()
            optimizer.step()
            mass = cross_entropy_normalizer(labels, criterion.weight) if global_weighted_loss else len(images)
            training_total += loss.item() * mass
            training_mass += mass

        model.eval()
        validation_total = 0.0
        validation_mass = 0.0
        with torch.no_grad():
            for images, labels in validation_loader:
                images, labels = images.to(device), labels.to(device)
                mass = cross_entropy_normalizer(labels, criterion.weight) if global_weighted_loss else len(images)
                validation_total += criterion(model(images), labels).item() * mass
                validation_mass += mass

        training_loss = training_total / training_mass
        validation_loss = validation_total / validation_mass
        history.append(
            {
                "epoch": epoch,
                "train_loss": float(training_loss),
                "inner_validation_loss": float(validation_loss),
            }
        )
        if validation_loss < best_loss:
            best_loss = validation_loss
            best_epoch = epoch
            best_state = {
                name: value.detach().cpu().clone()
                for name, value in model.state_dict().items()
            }
        if verbose:
            print(
                f"      epoch {epoch:02d}/{epochs} train={training_loss:.4f} "
                f"inner-val={validation_loss:.4f}",
                flush=True,
            )

    if best_state is None:
        raise RuntimeError("No inner-validation checkpoint was selected.")
    model.load_state_dict(best_state)
    model.to(device)
    return model, history, int(best_epoch)


def relative_path(path):
    return str(Path(path).resolve().relative_to(ROOT))


def write_json(path, payload, *, overwrite=False):
    atomic_write_text(
        path,
        json.dumps(payload, indent=2, allow_nan=False) + "\n",
        overwrite=overwrite,
    )


def write_csv(path, frame, *, overwrite=False):
    atomic_write_text(path, frame.to_csv(index=False), overwrite=overwrite)


def run(args):
    started = time.time()
    protocol = getattr(args, "protocol", "v5")
    if protocol not in {"v5", "v6"}:
        raise ValueError("Protocol must be explicitly v5 or v6")
    use_v6 = protocol == "v6"
    if use_v6 and args.max_folds is not None:
        raise ValueError("v6 supports complete runs or audit-only; partial diagnostics retain the explicit v5 route")
    from validation import evaluation_v6
    protocol_version = evaluation_v6.PROTOCOL_VERSION if use_v6 else PROTOCOL_VERSION
    result_validator = evaluation_v6.validate_result if use_v6 else validate_internal_result_payload
    terminal_validator = evaluation_v6.validate_terminal if use_v6 else validate_terminal_bundle
    run_mode = classify_run_mode(
        folds=args.folds,
        epochs=args.epochs,
        batch_size=args.batch_size,
        seed=args.seed,
        inner_val_fraction=args.inner_val_fraction,
        target_sensitivity=args.target_sensitivity,
        bootstrap=args.bootstrap,
        max_folds=args.max_folds,
        audit_only=args.audit_only,
    )
    complete = run_mode == "complete"
    device = resolve_device(args.device)
    prefix = resolve_run_output_prefix(
        ROOT,
        args.output_prefix,
        max_folds=args.max_folds,
        total_folds=args.folds,
        audit_only=args.audit_only,
    )
    fold_path = Path(str(prefix) + "_group_folds.csv")
    audit_path = Path(str(prefix) + "_audit.json")
    image_output_path = Path(str(prefix) + "_oof_images.csv")
    group_output_path = Path(str(prefix) + "_oof_groups.csv")
    result_path = Path(str(prefix) + ".json")
    bundle_paths = [fold_path, audit_path]
    if complete:
        bundle_paths.extend([image_output_path, group_output_path, result_path])
    elif run_mode == "partial_smoke_run":
        bundle_paths.append(result_path)
    assert_fresh_output_bundle(bundle_paths)
    raw_dir = (
        (ROOT / args.raw_dir).resolve()
        if not Path(args.raw_dir).is_absolute()
        else Path(args.raw_dir).resolve()
    )
    source_metadata = load_dataset_metadata(
        raw_dir,
        expected_labels_sha256=EXPECTED_HYGD_DATASET_IDENTITY[
            "labels_csv_sha256"
        ],
    )
    metadata, data_quality = prepare_duplicate_aware_metadata(source_metadata)
    dataset_identity = build_dataset_identity(
        data_quality,
        labels_csv_sha256=source_metadata.attrs["labels_csv_sha256"],
    )
    # Every route is bound to the exact official HYGD v1.1.0 identity. Audit and
    # smoke modes are noncanonical outputs, not permission to inspect a mutated
    # or look-alike dataset under an apparently successful HYGD receipt.
    validate_canonical_dataset_identity(dataset_identity)
    validate_canonical_data_quality(data_quality)
    validate_canonical_oof_identity(metadata)
    group_folds = make_outer_folds(metadata, args.folds, args.seed)
    validate_canonical_group_fold_assignment(group_folds)
    fold_map = group_folds.set_index("evaluation_group")["outer_fold"]
    metadata["outer_fold"] = metadata["evaluation_group"].map(fold_map).astype(int)

    write_csv(fold_path, group_folds)

    audit = {
        "protocol_version": protocol_version,
        "status": "audit_only_complete" if args.audit_only else "running",
        "dataset_identity": dataset_identity,
        "data_quality": data_quality,
        "split_summary": [
            {
                "outer_fold": int(fold),
                "groups": int(len(rows)),
                "images": int(rows["n_images"].sum()),
                "positive_groups": int(rows["label"].sum()),
                "negative_groups": int((1 - rows["label"]).sum()),
            }
            for fold, rows in group_folds.groupby("outer_fold")
        ],
        "group_fold_manifest": relative_path(fold_path),
    }
    write_json(audit_path, audit)
    print(
        json.dumps(
            {
                "protocol_version": protocol_version,
                "status": audit["status"],
                "data_quality_aggregate": console_data_quality_summary(data_quality),
                "audit_artifact": relative_path(audit_path),
                "group_fold_manifest": relative_path(fold_path),
            },
            indent=2,
        ),
        flush=True,
    )
    if args.audit_only:
        return audit

    eval_transform = build_transforms(train=False)
    oof_frames, fold_results, smoke_fold_execution = [], [], []
    folds_to_run = sorted(group_folds["outer_fold"].unique())
    if args.max_folds is not None:
        folds_to_run = folds_to_run[: args.max_folds]

    for fold in folds_to_run:
        fold_seed = args.seed + int(fold) * 101
        outer_test = metadata[metadata["outer_fold"] == fold].copy().reset_index(drop=True)
        outer_train = metadata[metadata["outer_fold"] != fold].copy().reset_index(drop=True)
        outer_train_groups = set(outer_train["evaluation_group"])
        outer_test_groups = set(outer_test["evaluation_group"])
        if outer_train_groups & outer_test_groups:
            raise AssertionError(f"Outer fold {fold} has group leakage.")

        train_groups, validation_groups = inner_group_split(
            outer_train, args.inner_val_fraction, fold_seed
        )
        inner_train = outer_train[
            outer_train["evaluation_group"].isin(train_groups)
        ].copy().reset_index(drop=True)
        inner_validation = outer_train[
            outer_train["evaluation_group"].isin(validation_groups)
        ].copy().reset_index(drop=True)
        if set(inner_train["evaluation_group"]) & set(inner_validation["evaluation_group"]):
            raise AssertionError(f"Inner fold {fold} has group leakage.")

        print(
            f"\n=== outer fold {fold + 1}/{args.folds}: "
            f"train {len(inner_train)} img/{len(train_groups)} groups, "
            f"inner-val {len(inner_validation)} img/{len(validation_groups)} groups, "
            f"outer-test {len(outer_test)} img/{len(outer_test_groups)} groups ===",
            flush=True,
        )
        model, history, best_epoch = train_with_inner_validation(
            inner_train,
            inner_validation,
            args.epochs,
            args.batch_size,
            device,
            fold_seed,
            verbose=complete,
            global_weighted_loss=use_v6,
        )

        inner_validation["probability"] = predict_probabilities(
            model, inner_validation, eval_transform, args.batch_size, device
        )
        inner_validation["selected_threshold"] = 0.5
        inner_validation["outer_fold"] = int(fold)
        validation_groups_frame = aggregate_groups(inner_validation)
        threshold_result = select_threshold(
            validation_groups_frame["label"],
            validation_groups_frame["probability"],
            args.target_sensitivity,
        )

        # The outer test fold is first evaluated here, after every training and
        # threshold decision for this fold has been finalized.
        outer_probabilities = predict_probabilities(
            model, outer_test, eval_transform, args.batch_size, device
        )
        if not complete:
            prediction_receipt = validate_smoke_prediction_execution(
                outer_probabilities,
                len(outer_test),
            )
            smoke_fold_execution.append(
                {
                    "outer_fold": int(fold),
                    "training_loop_completed": True,
                    "checkpoint_restore_completed": True,
                    "inner_selection_path_completed": True,
                    **prediction_receipt,
                }
            )
            del outer_probabilities
            print(fold_completion_message(int(fold), args.folds), flush=True)
            del model
            if device == "mps":
                torch.mps.empty_cache()
            elif device == "cuda":
                torch.cuda.empty_cache()
            continue

        outer_test["probability"] = outer_probabilities
        outer_test["selected_threshold"] = threshold_result["threshold"]
        outer_test["outer_fold"] = int(fold)
        outer_test["predicted_class"] = (
            outer_test["probability"] >= outer_test["selected_threshold"]
        ).astype(int)
        oof_frames.append(outer_test)

        outer_group_frame = aggregate_groups(outer_test)
        fold_result = {
            "outer_fold": int(fold),
            "seed": int(fold_seed),
            "best_epoch_selected_on_inner_validation_loss": best_epoch,
            "selected_threshold_from_inner_group_validation": threshold_result,
            "counts": {
                "inner_train_images": int(len(inner_train)),
                "inner_train_groups": int(len(train_groups)),
                "inner_validation_images": int(len(inner_validation)),
                "inner_validation_groups": int(len(validation_groups)),
                "outer_test_images": int(len(outer_test)),
                "outer_test_groups": int(len(outer_test_groups)),
            },
            "outer_test_image_level": frame_metrics(outer_test),
            "outer_test_group_level": frame_metrics(outer_group_frame),
            "training_history": history,
        }
        fold_results.append(fold_result)
        print(fold_completion_message(int(fold), args.folds), flush=True)
        del model
        if device == "mps":
            torch.mps.empty_cache()
        elif device == "cuda":
            torch.cuda.empty_cache()

    if complete and len(folds_to_run) != args.folds:
        raise AssertionError("A complete run must evaluate every outer fold")
    if complete:
        oof = pd.concat(oof_frames, ignore_index=True)
        validate_oof_predictions(oof, metadata, complete=True)
        oof = oof.sort_values(
            ["outer_fold", "evaluation_group", "image_name"]
        ).reset_index(drop=True)
        oof_group = aggregate_groups(oof)
        image_metrics = frame_metrics(oof)
        group_metrics = frame_metrics(oof_group)
        outer_fold_auc = outer_fold_auc_summary(oof_group)
        fold_stratified_ci = fold_stratified_group_bootstrap(
            oof_group, args.bootstrap, args.seed + 9_003
        )
        image_ci = cluster_bootstrap(
            oof, "evaluation_group", args.bootstrap, args.seed + 9_001
        )
        pooled_group_ci = cluster_bootstrap(
            oof_group, "evaluation_group", args.bootstrap, args.seed + 9_002
        )
        evaluation_report = build_primary_report(
            outer_fold_auc=outer_fold_auc,
            fold_stratified_ci=fold_stratified_ci,
            pooled_group_metrics=group_metrics,
            pooled_group_cluster_ci=pooled_group_ci,
            image_metrics=image_metrics,
            image_cluster_ci=image_ci,
        )
    else:
        evaluation_report = build_smoke_diagnostics(
            fold_execution=smoke_fold_execution,
            completed_outer_folds=folds_to_run,
            total_outer_folds=args.folds,
        )

    image_columns = [
        "image_name",
        "patient_id",
        "evaluation_group",
        "sha256",
        "label",
        "quality_score",
        "outer_fold",
        "probability",
        "selected_threshold",
        "predicted_class",
    ]
    if complete:
        oof_group["predicted_class"] = (
            oof_group["probability"] >= oof_group["selected_threshold"]
        ).astype(int)
        post_source_metadata = load_dataset_metadata(
            raw_dir,
            expected_labels_sha256=EXPECTED_HYGD_DATASET_IDENTITY[
                "labels_csv_sha256"
            ],
        )
        post_metadata, post_data_quality = prepare_duplicate_aware_metadata(
            post_source_metadata
        )
        post_dataset_identity = build_dataset_identity(
            post_data_quality,
            labels_csv_sha256=post_source_metadata.attrs["labels_csv_sha256"],
        )
        validate_canonical_dataset_identity(post_dataset_identity)
        validate_canonical_data_quality(post_data_quality)
        validate_canonical_oof_identity(post_metadata)
        post_group_folds = make_outer_folds(post_metadata, args.folds, args.seed)
        validate_canonical_group_fold_assignment(post_group_folds)
        if post_dataset_identity != dataset_identity:
            raise RuntimeError(
                "HYGD dataset identity changed during evaluation; no terminal result "
                "will be published."
            )
        if not post_group_folds.equals(group_folds):
            raise RuntimeError(
                "Canonical HYGD group-fold assignment changed during evaluation; "
                "no terminal result will be published."
            )

        result = {
            "protocol_version": protocol_version,
            "status": "complete",
            "completed_outer_folds": [int(fold) for fold in folds_to_run],
            "configuration_fixed_before_outer_evaluation": {
                "model": "ImageNet ResNet18, layer4 + head fine-tuned",
                "weights": "ResNet18_Weights.IMAGENET1K_V1",
                "trainable_parameters": "layer4 + fc",
                "optimizer": "Adam",
                "head_learning_rate": 1e-3,
                "backbone_learning_rate": 1e-4,
                "train_transform": (
                    "Resize(224x224), RandomHorizontalFlip(p=0.5), "
                    "RandomRotation(15deg), ColorJitter(brightness=0.1,contrast=0.1), "
                    "ImageNet normalization"
                ),
                "evaluation_transform": "Resize(224x224), ImageNet normalization",
                "configuration_origin": (
                    "Chosen during prior HYGD development, then frozen before this repair run"
                ),
                "residual_post_selection_risk": (
                    "Outer folds were hidden from this run, but the architecture and recipe "
                    "were historically informed by HYGD development/test results"
                ),
                "epochs": int(args.epochs),
                "batch_size": int(args.batch_size),
                "outer_folds": int(args.folds),
                "inner_validation_fraction": float(args.inner_val_fraction),
                "threshold_target_sensitivity": float(args.target_sensitivity),
                "bootstrap_samples": int(args.bootstrap),
                "seed": int(args.seed),
                "device": device,
                "checkpoint_selection": "minimum inner-validation loss",
                "threshold_selection": "inner-validation group-level predictions only",
                "outer_test_visibility_during_training": (
                    "withheld_until_terminal_bundle_publication"
                ),
            },
            "dataset_identity": dataset_identity,
            "data_quality": data_quality,
            "fold_results": fold_results,
            "artifacts": {
                "audit": relative_path(audit_path),
                "group_folds": relative_path(fold_path),
                "oof_images": relative_path(image_output_path),
                "oof_groups": relative_path(group_output_path),
            },
            "environment": {
                "python": platform.python_version(),
                "torch": torch.__version__,
                "torchvision": package_version("torchvision"),
                "numpy": np.__version__,
                "pandas": pd.__version__,
                "scikit_learn": package_version("scikit-learn"),
                "pillow": package_version("Pillow"),
                "platform": platform.platform(),
            },
            "runtime_seconds": round(time.time() - started, 1),
            "internal_evaluation": evaluation_report,
        }
    else:
        result = {
            "protocol_version": protocol_version,
            "status": "partial_smoke_run",
            "canonical_internal_estimate": False,
            "completed_outer_folds": [int(fold) for fold in folds_to_run],
            "planned_outer_folds": int(args.folds),
            "smoke_diagnostics": evaluation_report,
            "artifacts": {
                "audit": relative_path(audit_path),
                "group_folds": relative_path(fold_path),
            },
            "runtime_seconds": round(time.time() - started, 1),
        }
    if complete:
        if use_v6:
            result["configuration_fixed_before_outer_evaluation"].update(evaluation_v6.TRAINING_FIELDS)
        result_validator(result)
    else:
        validate_smoke_result_payload(result)
    if complete:
        write_csv(image_output_path, oof[image_columns])
        write_csv(group_output_path, oof_group)
    audit = finalize_audit(
        audit,
        result_status=result["status"],
        result_path=relative_path(result_path),
        completed_outer_folds=folds_to_run,
        result_claim_sha256=terminal_result_claim_sha256(result),
    )
    write_json(audit_path, audit, overwrite=True)
    result["artifact_sha256"] = {
        key: sha256_file(ROOT / relative)
        for key, relative in result["artifacts"].items()
    }
    result["artifact_size_bytes"] = {
        key: (ROOT / relative).stat().st_size
        for key, relative in result["artifacts"].items()
    }
    if complete:
        terminal_validator(result, ROOT)
    else:
        validate_terminal_smoke_bundle(result, ROOT)
    # The result is the bundle commit marker: it is published only after every
    # referenced artifact and the terminal audit exist and have been hashed.
    write_json(result_path, result)
    published_result = read_confined_json(ROOT, relative_path(result_path))
    if published_result != result:
        raise RuntimeError("Published result marker failed deterministic read-back")
    if complete:
        terminal_validator(published_result, ROOT)
    else:
        validate_terminal_smoke_bundle(published_result, ROOT)
    print(f"\nsaved {relative_path(result_path)}", flush=True)
    summary = {
        "status": result["status"],
        "runtime_seconds": result["runtime_seconds"],
    }
    if complete:
        summary.update(
            {
                "primary_group_level": result["internal_evaluation"]["primary"],
                "secondary_image_level": result["internal_evaluation"][
                    "secondary_image_level"
                ],
            }
        )
    else:
        summary["noncanonical_smoke_diagnostics"] = result["smoke_diagnostics"]
    print(json.dumps(summary, indent=2), flush=True)
    return result


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", choices=["v5", "v6"], default="v5",
                        help="v5 preserves historical batch-mean aggregation; v6 uses globally weighted CE")
    parser.add_argument("--raw-dir", default="data/raw")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--inner-val-fraction", type=float, default=0.15)
    parser.add_argument("--target-sensitivity", type=float, default=0.95)
    parser.add_argument("--bootstrap", type=int, default=5_000)
    parser.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"], default="auto")
    parser.add_argument("--audit-only", action="store_true")
    parser.add_argument(
        "--max-folds",
        type=int,
        default=None,
        help="Run only the first N outer folds for a smoke test. Such output is marked partial.",
    )
    parser.add_argument(
        "--output-prefix", default="results/internal_evaluation_repair"
    )
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
