#!/usr/bin/env python3
"""UR-Fall experiment using the official ICCV 2025 Hyper-GCN model.

This file does NOT reimplement Hyper-GCN's model. It imports
`model.hypergcn_base.Model` from the authors' official repository and only
adapts the experiment pipeline to the user's UR-Fall / BlazePose setup.

Expected project layout (run from fall_detection root):

    fall_detection/
    ├── data/
    │   ├── keypoints/                  # or keypoints_normalized/
    │   └── urfall-cam0-falls.csv
    ├── src/experiments/
    │   ├── hypergcn.py                 # this file
    │   ├── lstm.py
    │   └── experiment_mlp.py
    └── third_party/
        └── Hyper-GCN/
            ├── model/hypergcn_base.py  # official
            ├── graph/tools.py          # official
            └── graph/blazepose.py      # provided adapter

Example:
    uv run python src/experiments/hypergcn_b.py \
        --data-root data/keypoints \
        --result-root results/hypergcn_b \
        --hypergcn-root third_party/Hyper-GCN \
        --device cuda

The experiment preserves the same UR-Fall 30-frame windows, CSV-based window
labels and video-grouped 5-fold CV used by the existing ST-GCN experiment.
The model itself is the official Hyper-GCN base architecture.

Scheme B ablation:
- no class weights
- no DivergenceLoss
- plain nn.CrossEntropyLoss()
- fixed Fall decision threshold = 0.5
All other model/data/training settings are kept the same as the original run.
"""

from __future__ import annotations

import argparse
import importlib
import json
import re
import sys
import time
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import StratifiedGroupKFold
from torch import nn

# Reuse the project's existing data/window/metric implementation so Hyper-GCN
# receives exactly the same samples as RF/MLP/LSTM/ST-GCN.
from lstm import compute_metrics, load_all_windows, seed_everything
from mlp import make_loader


DEFAULT_ANNOTATION_CSV = Path("data/urfall-cam0-falls.csv")
NUM_POINT = 33
NUM_PERSON = 1
IN_CHANNELS = 2
NUM_CLASS = 2


def canonical_sequence_name(value) -> str | None:
    value = str(value).strip().lower()
    match = re.search(r"(fall|adl)-?(\d+)", value)
    if match is None:
        return None
    return f"{match.group(1)}-{int(match.group(2)):02d}"


def load_fall_frame_labels(csv_path: Path):
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"UR-Fall annotation CSV not found: {csv_path}")

    annotations = {}
    valid_rows = 0
    with csv_path.open("r", encoding="utf-8-sig") as f:
        for line_number, raw_line in enumerate(f, start=1):
            line = raw_line.strip()
            if not line:
                continue

            if "," in line:
                row = [x.strip() for x in line.split(",")]
            elif ";" in line:
                row = [x.strip() for x in line.split(";")]
            elif "\t" in line:
                row = [x.strip() for x in line.split("\t")]
            else:
                row = line.split()

            if len(row) < 3:
                continue

            sequence_name = canonical_sequence_name(row[0])
            try:
                frame_number = int(float(row[1]))
                posture_label = int(float(row[2]))
            except ValueError:
                # Header or malformed non-data row.
                continue

            if sequence_name is None or not sequence_name.startswith("fall-"):
                continue
            if posture_label not in (-1, 0, 1):
                raise ValueError(
                    f"Unknown posture label at {csv_path}:{line_number}: "
                    f"{posture_label}"
                )

            annotations.setdefault(sequence_name, {})[frame_number] = posture_label
            valid_rows += 1

    if valid_rows == 0:
        raise RuntimeError(f"No usable UR-Fall annotations read from {csv_path}")

    print(
        f"Loaded UR-Fall annotations: {len(annotations)} fall sequences, "
        f"{valid_rows} frames"
    )
    return annotations


def get_window_label_from_urfall(frame_labels: Sequence[int]) -> int | None:
    """Ignore posture=0, then majority vote -1 (ADL) versus +1 (Fall)."""
    labels = np.asarray(frame_labels, dtype=np.int8)
    labels = labels[labels != 0]
    if labels.size == 0:
        return None

    normal_count = int(np.sum(labels == -1))
    fall_count = int(np.sum(labels == 1))
    if fall_count > normal_count:
        return 1
    if normal_count > fall_count:
        return 0
    return None


def _window_frame_bounds(records: pd.DataFrame, i: int, window_size: int):
    row = records.iloc[i]
    if {"start_frame", "end_frame"}.issubset(records.columns):
        return int(row["start_frame"]), int(row["end_frame"])
    if {"window_start", "window_end"}.issubset(records.columns):
        return int(row["window_start"]) + 1, int(row["window_end"]) + 1
    if "window_start" in records.columns:
        start = int(row["window_start"]) + 1
        return start, start + window_size - 1
    raise RuntimeError(
        "records is missing frame bounds; expected start_frame/end_frame "
        "or window_start/window_end"
    )


def apply_urfall_window_labels(
    x: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    records: pd.DataFrame,
    window_size: int,
    annotations,
):
    if not isinstance(records, pd.DataFrame):
        records = pd.DataFrame(records)
    if len(x) != len(y) or len(x) != len(groups) or len(x) != len(records):
        raise RuntimeError(
            "x/y/groups/records length mismatch: "
            f"{len(x)}/{len(y)}/{len(groups)}/{len(records)}"
        )

    keep_indices = []
    labels = []
    skipped_transition_or_tie = 0
    skipped_no_annotation = 0

    for i, group in enumerate(groups):
        sequence_name = canonical_sequence_name(group)
        if sequence_name is None:
            raise RuntimeError(f"Cannot parse UR-Fall sequence from group: {group}")

        if sequence_name.startswith("adl-"):
            keep_indices.append(i)
            labels.append(0)
            continue

        seq_annotations = annotations.get(sequence_name)
        if seq_annotations is None:
            raise RuntimeError(f"No CSV annotations for {sequence_name}")

        start_frame, end_frame = _window_frame_bounds(records, i, window_size)
        frame_labels = [
            seq_annotations[frame]
            for frame in range(start_frame, end_frame + 1)
            if frame in seq_annotations
        ]
        if not frame_labels:
            skipped_no_annotation += 1
            continue

        label = get_window_label_from_urfall(frame_labels)
        if label is None:
            skipped_transition_or_tie += 1
            continue

        keep_indices.append(i)
        labels.append(label)

    keep_indices = np.asarray(keep_indices, dtype=np.int64)
    new_y = np.asarray(labels, dtype=np.int64)
    x = x[keep_indices]
    groups = groups[keep_indices]
    records = records.iloc[keep_indices].copy().reset_index(drop=True)
    records["window_label"] = new_y
    records["window_label_name"] = np.where(new_y == 1, "fall", "adl")

    print(
        "Window relabeling: "
        f"kept={len(new_y)}, ADL={int(np.sum(new_y == 0))}, "
        f"Fall={int(np.sum(new_y == 1))}, "
        f"skipped_transition_or_tie={skipped_transition_or_tie}, "
        f"skipped_no_annotation={skipped_no_annotation}"
    )

    if len(new_y) == 0 or len(np.unique(new_y)) < 2:
        raise RuntimeError("Window relabeling did not leave both classes")
    return x, new_y, groups, records


def video_type_labels(groups: np.ndarray) -> np.ndarray:
    labels = []
    for group in groups:
        seq = canonical_sequence_name(group)
        if seq is None:
            raise RuntimeError(f"Cannot parse video type from: {group}")
        labels.append(1 if seq.startswith("fall-") else 0)
    return np.asarray(labels, dtype=np.int64)


def inner_group_split(y, groups, seed):
    """Inner grouped validation split; stratify by original video type."""
    split_y = video_type_labels(groups)
    group_rows = (
        pd.DataFrame({"group": groups, "video_type": split_y})
        .drop_duplicates("group")
    )
    counts = np.bincount(
        group_rows["video_type"].to_numpy(dtype=np.int64), minlength=2
    )
    n_splits = int(min(5, counts.min()))
    if n_splits < 2:
        raise RuntimeError(
            "Inner validation needs at least two ADL and two fall videos; "
            f"got {counts.tolist()}"
        )

    splitter = StratifiedGroupKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=seed,
    )
    fit_idx, val_idx = next(splitter.split(np.zeros(len(y)), split_y, groups))
    return fit_idx, val_idx


def standardize(x: np.ndarray, fit_idx: np.ndarray):
    """Same train-only coordinate standardization as the current ST-GCN."""
    mean = x[fit_idx].mean(axis=(0, 1), keepdims=True)
    std = x[fit_idx].std(axis=(0, 1), keepdims=True)
    std[std < 1e-6] = 1.0
    return mean.astype(np.float32), std.astype(np.float32)


def load_official_hypergcn(hypergcn_root: Path):
    """Import the authors' official Hyper-GCN base Model class."""
    root = Path(hypergcn_root).expanduser().resolve()
    required = [
        root / "model" / "hypergcn_base.py",
        root / "graph" / "tools.py",
        root / "graph" / "blazepose.py",
    ]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        raise FileNotFoundError(
            "Hyper-GCN official repository / BlazePose adapter is incomplete.\n"
            "Missing:\n  - " + "\n  - ".join(missing) + "\n\n"
            "Clone https://github.com/6UOOON9/Hyper-GCN into "
            "third_party/Hyper-GCN and copy blazepose.py into its graph/ folder."
        )

    root_str = str(root)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)

    # Avoid accidentally reusing a different package named model/graph.
    importlib.invalidate_caches()
    module = importlib.import_module("model.hypergcn_base")
    return module.Model, root


class HyperGCNAdapter(nn.Module):
    """Only reshapes [B,T,66] into Hyper-GCN's [B,C,T,V,M] input."""

    def __init__(self, official_model_cls, dropout: float, hyper_joints: int):
        super().__init__()
        self.backbone = official_model_cls(
            num_class=NUM_CLASS,
            num_point=NUM_POINT,
            num_person=NUM_PERSON,
            graph="graph.blazepose.Graph",
            graph_args={"labeling_mode": "virtual_ensemble"},
            in_channels=IN_CHANNELS,
            hyper_joints=hyper_joints,
            drop_out=dropout,
        )

    def forward(self, x: torch.Tensor):
        if x.ndim != 3 or x.shape[-1] != NUM_POINT * IN_CHANNELS:
            raise ValueError(
                f"Expected [B,T,{NUM_POINT * IN_CHANNELS}], got {tuple(x.shape)}"
            )
        b, t, _ = x.shape
        x = x.reshape(b, t, NUM_POINT, IN_CHANNELS)
        x = x.permute(0, 3, 1, 2).contiguous().unsqueeze(-1)
        return self.backbone(x)


def unpack_logits(output):
    if isinstance(output, tuple):
        return output[0]
    return output


@torch.no_grad()
def predict_probabilities(model, loader, device):
    model.eval()
    probs = []
    for xb, _ in loader:
        xb = xb.to(device, non_blocking=True)
        logits = unpack_logits(model(xb))
        prob = torch.softmax(logits, dim=1)[:, 1]
        probs.append(prob.detach().cpu().numpy())
    if not probs:
        return np.empty((0,), dtype=np.float32)
    return np.concatenate(probs).astype(np.float32)


def set_official_lr(optimizer, base_lr, epoch_zero_based, warmup_epochs, steps, decay):
    """Official Hyper-GCN warm-up + step-decay schedule."""
    if warmup_epochs > 0 and epoch_zero_based < warmup_epochs:
        lr = base_lr * (epoch_zero_based + 1) / warmup_epochs
    else:
        decay_count = int(np.sum(epoch_zero_based >= np.asarray(steps)))
        lr = base_lr * (decay ** decay_count)
    for group in optimizer.param_groups:
        group["lr"] = lr
    return float(lr)


def train_fold(x, y, groups, args, device, model_cls, seed):
    fit_idx, val_idx = inner_group_split(y, groups, seed)
    mean, std = standardize(x, fit_idx)

    fit_x = ((x[fit_idx] - mean) / std).astype(np.float32)
    val_x = ((x[val_idx] - mean) / std).astype(np.float32)
    fit_loader = make_loader(fit_x, y[fit_idx], args.batch_size, True)
    val_loader = make_loader(val_x, y[val_idx], args.batch_size, False)

    seed_everything(seed)
    model = HyperGCNAdapter(
        model_cls,
        dropout=args.dropout,
        hyper_joints=args.hyper_joints,
    ).to(device)

    # Scheme B ablation:
    # - no class weights
    # - no label smoothing
    # - no DivergenceLoss
    # Use the same plain classification objective as a clean baseline.
    counts = np.bincount(y[fit_idx], minlength=2)
    if np.any(counts == 0):
        raise RuntimeError(f"A training split lost a class: counts={counts.tolist()}")
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=args.base_lr,
        momentum=0.9,
        nesterov=True,
        weight_decay=args.weight_decay,
    )

    best_f1 = -1.0
    best_loss = float("inf")
    best_epoch = 0
    best_state = None
    history = []
    started = time.perf_counter()

    for epoch in range(1, args.epochs + 1):
        lr = set_official_lr(
            optimizer,
            args.base_lr,
            epoch - 1,
            args.warmup_epochs,
            args.steps,
            args.lr_decay_rate,
        )

        model.train()
        total_loss_sum = 0.0
        correct = 0
        seen = 0

        for xb, yb in fit_loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            output = model(xb)
            logits = unpack_logits(output)
            loss = criterion(logits, yb)
            loss.backward()
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()

            n = len(yb)
            total_loss_sum += float(loss.detach()) * n
            correct += int((logits.argmax(1) == yb).sum().item())
            seen += n

        val_prob = predict_probabilities(model, val_loader, device)
        val_pred = (val_prob >= 0.5).astype(np.int64)
        val_metrics = compute_metrics(y[val_idx], val_pred, val_prob)
        train_loss = total_loss_sum / max(seen, 1)
        train_acc = correct / max(seen, 1)
        val_f1 = float(val_metrics["f1"])

        history.append(
            {
                "epoch": epoch,
                "lr": lr,
                "train_loss": train_loss,
                "train_accuracy": train_acc,
                "val_accuracy": val_metrics["accuracy"],
                "val_precision": val_metrics["precision"],
                "val_recall": val_metrics["recall"],
                "val_f1": val_f1,
            }
        )

        if (
            val_f1 > best_f1
            or (np.isclose(val_f1, best_f1) and train_loss < best_loss)
        ):
            best_f1 = val_f1
            best_loss = train_loss
            best_epoch = epoch
            best_state = {
                k: v.detach().cpu().clone()
                for k, v in model.state_dict().items()
            }

        if epoch == 1 or epoch % args.log_every == 0:
            print(
                f"    epoch={epoch:03d} lr={lr:.6f} "
                f"loss={train_loss:.4f} train_acc={train_acc:.4f} "
                f"val_f1={val_f1:.4f} val_recall={val_metrics['recall']:.4f}"
            )

        if epoch - best_epoch >= args.patience:
            print(
                f"    Early stopping at epoch {epoch}; "
                f"best epoch={best_epoch}, best val F1={best_f1:.4f}"
            )
            break

    if best_state is None:
        raise RuntimeError("Training finished without a valid checkpoint")

    model.load_state_dict(best_state)
    seconds = time.perf_counter() - started
    return (
        model,
        mean,
        std,
        best_epoch,
        best_f1,
        history,
        (fit_idx, val_idx),
        seconds,
    )


def fmt_metric(v):
    if isinstance(v, (float, np.floating)) and np.isnan(v):
        return "nan"
    if isinstance(v, (float, np.floating)):
        return f"{float(v):.6f}"
    return str(v)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Official Hyper-GCN on UR-Fall with video-grouped 5-fold CV"
    )
    parser.add_argument("--data-root", type=Path, default=Path("data/keypoints"))
    parser.add_argument(
        "--result-root",
        "--output-root",
        dest="result_root",
        type=Path,
        default=Path("results/hypergcn_b"),
    )
    parser.add_argument(
        "--hypergcn-root",
        type=Path,
        default=Path("third_party/Hyper-GCN"),
        help="Path to authors' official Hyper-GCN repository",
    )
    parser.add_argument(
        "--annotation-csv",
        type=Path,
        default=DEFAULT_ANNOTATION_CSV,
    )
    parser.add_argument("--window-size", type=int, default=30)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--missing-mode", choices=("zero", "interp"), default="interp")
    parser.add_argument("--visibility-threshold", type=float, default=0.3)

    # Model adaptation. Official NTU120 base config uses hyper_joints=3.
    parser.add_argument("--hyper-joints", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.3)

    # Official base config: SGD, lr=.05, wd=.0004, steps=110/120,
    # warm-up=5, 140 epochs, Nesterov=True.
    parser.add_argument("--epochs", type=int, default=140)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--base-lr", "--learning-rate", dest="base_lr", type=float, default=0.05)
    parser.add_argument("--weight-decay", type=float, default=0.0004)
    parser.add_argument("--steps", type=int, nargs="+", default=[110, 120])
    parser.add_argument("--lr-decay-rate", type=float, default=0.1)
    parser.add_argument("--warmup-epochs", type=int, default=5)
    parser.add_argument("--grad-clip", type=float, default=5.0)
    parser.add_argument("--log-every", type=int, default=5)

    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--device",
        default="cuda",
        help="cuda or cuda:N. Official Hyper-GCN contains CUDA-specific code.",
    )
    return parser.parse_args()


def resolve_cuda_device(requested: str):
    requested = str(requested).strip().lower()
    if requested == "cuda":
        requested = "cuda:0"
    if not requested.startswith("cuda"):
        raise ValueError(
            "Official Hyper-GCN uses CUDA-specific tensor placement; use --device cuda "
            "or --device cuda:N."
        )
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is not available to PyTorch. Check torch.cuda.is_available(), "
            "your NVIDIA driver, and your PyTorch CUDA build."
        )
    device = torch.device(requested)
    # Trigger early validation of the requested index.
    _ = torch.cuda.get_device_name(device)
    return device


def dataset_run_name(data_root: Path) -> str:
    name = data_root.resolve().name.lower()
    return "hypergcn_b_normalized" if "normalized" in name else "hypergcn_b"


def main():
    args = parse_args()
    if min(args.window_size, args.stride, args.epochs, args.batch_size) < 1:
        raise ValueError("window-size/stride/epochs/batch-size must be positive")
    if args.folds < 2 or args.patience < 1:
        raise ValueError("folds >= 2 and patience >= 1 are required")
    if args.hyper_joints < 1:
        raise ValueError("--hyper-joints must be >= 1")
    if not 0 <= args.dropout < 1:
        raise ValueError("dropout must be in [0,1)")

    seed_everything(args.seed)
    device = resolve_cuda_device(args.device)
    model_cls, official_root = load_official_hypergcn(args.hypergcn_root)

    # Keep user-specified result-root literal. If they leave the default and run
    # normalized data, separate it automatically.
    if args.result_root == Path("results/hypergcn_b"):
        output = Path("results") / dataset_run_name(args.data_root)
    else:
        output = args.result_root
    output.mkdir(parents=True, exist_ok=True)
    (output / "models").mkdir(parents=True, exist_ok=True)
    (output / "histories").mkdir(parents=True, exist_ok=True)

    print(f"Device: {device} ({torch.cuda.get_device_name(device)})")
    print(f"Official Hyper-GCN root: {official_root}")
    print(f"Data root: {args.data_root}")
    print(f"Output dir: {output}")
    print(
        f"Input: window={args.window_size}, 33 BlazePose joints, XY only, "
        f"hyper_joints={args.hyper_joints}"
    )

    x, y, groups, records = load_all_windows(
        args.data_root,
        args.window_size,
        args.stride,
        args.visibility_threshold,
        args.missing_mode,
        "xy66",
        None,
    )
    annotations = load_fall_frame_labels(args.annotation_csv)
    x, y, groups, records = apply_urfall_window_labels(
        x,
        y,
        groups,
        records,
        args.window_size,
        annotations,
    )

    split_y = video_type_labels(groups)
    group_df = (
        pd.DataFrame({"group": groups, "video_type": split_y})
        .drop_duplicates("group")
    )
    class_groups = np.bincount(
        group_df["video_type"].to_numpy(dtype=np.int64), minlength=2
    )
    if class_groups.min() < args.folds:
        raise ValueError(
            f"Need at least {args.folds} videos per video type; "
            f"found ADL/Fall={class_groups.tolist()}"
        )

    config = vars(args).copy()
    for key, value in list(config.items()):
        if isinstance(value, Path):
            config[key] = str(value)
    config.update(
        {
            "model_type": "Hyper-GCN (official ICCV 2025 base model)",
            "official_model": "model.hypergcn_base.Model",
            "official_repo": "https://github.com/6UOOON9/Hyper-GCN",
            "graph": "graph.blazepose.Graph (UR-Fall adapter)",
            "num_class": NUM_CLASS,
            "num_point": NUM_POINT,
            "num_person": NUM_PERSON,
            "in_channels": IN_CHANNELS,
            "feature_mode": "xy66",
            "pose_extractor": "BlazePose",
            "label_level": "window",
            "window_label_rule": "ignore_0_then_majority_vote_-1_vs_1",
            "optimizer": "SGD(momentum=0.9,nesterov=True)",
            "experiment": "Scheme B ablation",
            "classification_loss": "plain CrossEntropyLoss",
            "class_weight": None,
            "label_smoothing": 0.0,
            "hypergraph_regularizer": None,
            "classification_threshold": 0.5,
        }
    )
    (output / "config.json").write_text(
        json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    splitter = StratifiedGroupKFold(
        n_splits=args.folds,
        shuffle=True,
        random_state=args.seed,
    )

    all_true, all_pred, all_prob, all_test_indices = [], [], [], []
    fold_rows = []
    metrics_lines = [
        "=" * 80,
        "MODEL: Hyper-GCN (official ICCV 2025 base implementation) - Scheme B",
        "=" * 80,
        f"official_root: {official_root}",
        f"data_root: {args.data_root}",
        f"feature_mode: xy66",
        f"input_shape_per_window: [{args.window_size}, 33, 2]",
        f"hyper_joints: {args.hyper_joints}",
        "graph: BlazePose 33 joints + Hyper-GCN virtual_ensemble",
        "num_class: 2",
        f"missing_mode: {args.missing_mode}",
        f"visibility_threshold: {args.visibility_threshold}",
        f"epochs: {args.epochs}",
        f"patience: {args.patience}",
        f"batch_size: {args.batch_size}",
        f"base_lr: {args.base_lr}",
        f"steps: {args.steps}",
        f"warmup_epochs: {args.warmup_epochs}",
        f"weight_decay: {args.weight_decay}",
        "experiment: Scheme B ablation",
        "classification_loss: plain CrossEntropyLoss",
        "class_weight: None",
        "label_smoothing: 0.0",
        "divergence_loss: False",
        "classification_threshold: 0.5",
        "optimizer: SGD(momentum=0.9,nesterov=True)",
        f"folds: {args.folds}",
        f"seed: {args.seed}",
        f"device: {device}",
        f"annotation_csv: {args.annotation_csv}",
        "label_level: window",
        "window_label_rule: ignore posture 0; majority vote -1(normal) vs 1(fall)",
        "",
    ]

    models_dir = output / "models"
    histories_dir = output / "histories"

    for fold, (train_idx, test_idx) in enumerate(
        splitter.split(x, split_y, groups), start=1
    ):
        train_groups = set(groups[train_idx])
        test_groups = set(groups[test_idx])
        overlap = train_groups & test_groups
        if overlap:
            raise RuntimeError(f"Video leakage detected: {sorted(overlap)[:5]}")

        print("\n" + "=" * 76)
        print(
            f"Hyper-GCN | Fold {fold}/{args.folds} | "
            f"train_windows={len(train_idx)} | test_windows={len(test_idx)}"
        )

        (
            model,
            mean,
            std,
            best_epoch,
            val_f1,
            history,
            _,
            seconds,
        ) = train_fold(
            x[train_idx],
            y[train_idx],
            groups[train_idx],
            args,
            device,
            model_cls,
            args.seed + fold,
        )

        test_x = ((x[test_idx] - mean) / std).astype(np.float32)
        test_loader = make_loader(
            test_x,
            y[test_idx],
            args.batch_size,
            False,
        )
        prob = predict_probabilities(model, test_loader, device)
        pred = (prob >= 0.5).astype(np.int64)
        fold_metrics = compute_metrics(y[test_idx], pred, prob)

        row = dict(fold_metrics)
        row.update(
            {
                "fold": fold,
                "train_videos": len(train_groups),
                "test_videos": len(test_groups),
                "train_windows": len(train_idx),
                "test_windows": len(test_idx),
                "best_epoch": best_epoch,
                "val_f1": val_f1,
                "train_seconds": seconds,
            }
        )
        fold_rows.append(row)

        print(
            "    "
            f"Accuracy={fold_metrics['accuracy']:.4f} | "
            f"Precision={fold_metrics['precision']:.4f} | "
            f"Recall={fold_metrics['recall']:.4f} | "
            f"F1={fold_metrics['f1']:.4f} | "
            f"ROC-AUC={fmt_metric(fold_metrics['roc_auc'])}"
        )
        print(
            f"    TN={fold_metrics['tn']} FP={fold_metrics['fp']} "
            f"FN={fold_metrics['fn']} TP={fold_metrics['tp']} | "
            f"best_epoch={best_epoch} val_f1={val_f1:.4f}"
        )

        metrics_lines.extend(
            [
                f"[Fold {fold}/{args.folds}]",
                f"train_videos: {len(train_groups)}",
                f"test_videos: {len(test_groups)}",
                f"train_windows: {len(train_idx)}",
                f"test_windows: {len(test_idx)}",
                f"best_epoch: {best_epoch}",
                f"val_f1: {fmt_metric(val_f1)}",
                f"train_seconds: {seconds:.2f}",
            ]
        )
        for key in [
            "accuracy", "precision", "recall", "f1", "roc_auc",
            "tn", "fp", "fn", "tp",
        ]:
            metrics_lines.append(f"{key}: {fmt_metric(fold_metrics[key])}")
        metrics_lines.append("")

        checkpoint_args = vars(args).copy()
        for key, value in list(checkpoint_args.items()):
            if isinstance(value, Path):
                checkpoint_args[key] = str(value)

        torch.save(
            {
                "model_type": "hypergcn_official_base_urfall_scheme_b",
                "official_model": "model.hypergcn_base.Model",
                "num_class": NUM_CLASS,
                "num_point": NUM_POINT,
                "num_person": NUM_PERSON,
                "in_channels": IN_CHANNELS,
                "hyper_joints": args.hyper_joints,
                "state_dict": model.state_dict(),
                "feature_mean": mean,
                "feature_std": std,
                "best_epoch": best_epoch,
                "val_f1": val_f1,
                "pose_extractor": "BlazePose",
                "args": checkpoint_args,
            },
            models_dir / f"fold_{fold}.pt",
        )
        pd.DataFrame(history).to_csv(
            histories_dir / f"fold_{fold}_history.csv", index=False
        )

        all_true.extend(y[test_idx].tolist())
        all_pred.extend(pred.tolist())
        all_prob.extend(prob.tolist())
        all_test_indices.extend(test_idx.tolist())

        # Release fold GPU memory before the next model is instantiated.
        del model
        torch.cuda.empty_cache()

    y_true_all = np.asarray(all_true, dtype=np.int64)
    y_pred_all = np.asarray(all_pred, dtype=np.int64)
    y_prob_all = np.asarray(all_prob, dtype=np.float32)

    overall = compute_metrics(y_true_all, y_pred_all, y_prob_all)
    cm = confusion_matrix(y_true_all, y_pred_all, labels=[0, 1])
    report = classification_report(
        y_true_all,
        y_pred_all,
        labels=[0, 1],
        target_names=["ADL", "Fall"],
        digits=4,
        zero_division=0,
    )

    metrics_lines.extend(
        [
            "=" * 80,
            "OVERALL OUT-OF-FOLD METRICS",
            "=" * 80,
        ]
    )
    for key in [
        "accuracy", "precision", "recall", "f1", "roc_auc",
        "tn", "fp", "fn", "tp",
    ]:
        metrics_lines.append(f"{key}: {fmt_metric(overall[key])}")
    metrics_lines.extend(
        [
            "",
            "Classification report:",
            report,
            "Confusion matrix [[TN, FP], [FN, TP]]:",
            np.array2string(cm),
        ]
    )

    fold_df = pd.DataFrame(fold_rows)
    metrics_lines.extend(["", "Fold mean ± std:"])
    for key in ["accuracy", "precision", "recall", "f1", "roc_auc"]:
        metrics_lines.append(
            f"{key}: {fold_df[key].mean():.6f} ± "
            f"{fold_df[key].std(ddof=1):.6f}"
        )

    metrics_path = output / "metrics.txt"
    metrics_path.write_text("\n".join(metrics_lines), encoding="utf-8")
    fold_df.to_csv(output / "fold_metrics.csv", index=False)
    pd.DataFrame(
        cm,
        index=["true_ADL", "true_Fall"],
        columns=["pred_ADL", "pred_Fall"],
    ).to_csv(output / "confusion_matrix.csv")

    pred_df = records.iloc[all_test_indices].copy().reset_index(drop=True)
    pred_df["y_true"] = y_true_all
    pred_df["y_pred"] = y_pred_all
    pred_df["fall_probability"] = y_prob_all
    pred_df.to_csv(output / "predictions.csv", index=False)

    print("\n" + "=" * 76)
    print("Hyper-GCN Scheme B OVERALL")
    print(
        f"Accuracy={overall['accuracy']:.4f} | "
        f"Precision={overall['precision']:.4f} | "
        f"Recall={overall['recall']:.4f} | "
        f"F1={overall['f1']:.4f} | "
        f"ROC-AUC={fmt_metric(overall['roc_auc'])}"
    )
    print(report)
    print(f"Saved: {metrics_path}")


if __name__ == "__main__":
    main()
