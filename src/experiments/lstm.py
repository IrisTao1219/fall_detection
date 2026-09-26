#!/usr/bin/env python3
"""
Fall detection experiment with LSTM on BlazePose keypoint sequences.

Compatible with the user's existing NPZ structure:
    keypoints:   [T, 33, 4] -> x, y, z, visibility
    valid_mask:  [T]
    fps:         scalar
    label:       scalar video type ("adl"/"fall" or 0/1)
    video_id:    scalar
    frame_indices: optional [T]; if absent, 1..T is used for CSV matching

Default experiment:
- feature: all BlazePose x/y coordinates -> 33 * 2 = 66 dims per frame
- window length: 30 frames
- stride: 1 frame
- missing joint handling: interpolate within each window (matching RF)
- discard windows with no detected pose frames (matching RF)
- LSTM: 10 recurrent layers, 80 hidden units
- BatchNorm after input and recurrent outputs
- window labels: UR-Fall official CSV, ignore 0 then majority vote -1 vs 1
- video-grouped 5-fold CV to prevent windows from the same video leaking
  across train/test folds
- metrics: Accuracy, Precision, Recall, F1, ROC-AUC,
  classification report, confusion matrix, per-fold results

Examples:
    uv run python src/lstm/experiment.py

Raw keypoints:
    uv run python src/lstm/experiment.py \
        --data-root data/keypoints \

Normalized keypoints:
    uv run python src/lstm/experiment.py \
        --data-root data/keypoints_normalized \

If you later confirm the paper's exact 16-D posture vector as 8 x/y joints:
    uv run python src/lstm/experiment.py \
        --feature-mode joints16 \
        --joint-indices 11,12,23,24,25,26,27,28 \

NOTE:
The 8-joint example above is only an explicit user-selected 16-D proxy.
It is NOT claimed to be the paper's undocumented exact 16-D construction.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import re
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedGroupKFold
from torch.utils.data import DataLoader, Dataset

try:
    from . import common as experiment_common
    from .common import (
        choose_device,
        compute_metrics,
        fmt_metric,
        load_sequence_windows,
        seed_everything,
        video_type_labels,
    )
except ImportError:
    import common as experiment_common
    from common import (
        choose_device,
        compute_metrics,
        fmt_metric,
        load_sequence_windows,
        seed_everything,
        video_type_labels,
    )


# -----------------------------
# Constants
# -----------------------------

LABEL_MAP = {
    "adl": 0,
    "normal": 0,
    "nonfall": 0,
    "non_fall": 0,
    "non-fall": 0,
    "0": 0,
    "fall": 1,
    "falling": 1,
    "1": 1,
}

CLASS_NAMES = ["ADL", "Fall"]

# UR-Fall official frame-level posture annotations (fall sequences only).
# First 3 columns: sequence name, frame number, posture label
# posture label: -1=not lying, 0=falling transition, 1=lying
FALL_ANNOTATION_CSV = Path("data/urfall-cam0-falls.csv")


# -----------------------------
# Reproducibility / device
# -----------------------------

def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def choose_device(requested: str) -> torch.device:
    requested = requested.lower()

    if requested != "auto":
        return torch.device(requested)

    if torch.cuda.is_available():
        return torch.device("cuda")

    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")

    return torch.device("cpu")


# -----------------------------
# Label utilities
# -----------------------------

def normalize_label(value) -> int:
    """Convert scalar/string label to {0,1}."""
    if isinstance(value, np.ndarray):
        if value.ndim == 0:
            value = value.item()
        elif value.size == 1:
            value = value.reshape(-1)[0].item()

    if isinstance(value, bytes):
        value = value.decode("utf-8")

    if isinstance(value, str):
        key = value.strip().lower()
        if key in LABEL_MAP:
            return LABEL_MAP[key]
        raise ValueError(f"Unknown string label: {value!r}")

    value = int(value)
    if value not in (0, 1):
        raise ValueError(f"Expected binary label 0/1, got {value}")
    return value


def normalize_label_array(values: np.ndarray) -> np.ndarray:
    return np.asarray([normalize_label(v) for v in values], dtype=np.int64)


def canonical_sequence_name(value) -> Optional[str]:
    """Normalize IDs like fall-01-cam0-d / fall-01-cam0-rgb to fall-01."""
    value = str(value).strip().lower()
    match = re.search(r"(fall|adl)-?(\d+)", value)
    if match is None:
        return None
    prefix = match.group(1)
    number = int(match.group(2))
    return f"{prefix}-{number:02d}"


def load_fall_frame_labels(csv_path: Path) -> Dict[str, Dict[int, int]]:
    """Load UR-Fall official per-frame posture labels from the first 3 CSV columns."""
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"UR-Fall annotation CSV not found: {csv_path}")

    annotations: Dict[str, Dict[int, int]] = {}
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
                # Header or malformed row.
                continue

            if sequence_name is None or not sequence_name.startswith("fall-"):
                continue
            if posture_label not in (-1, 0, 1):
                raise ValueError(
                    f"{csv_path} line {line_number}: unknown posture label {posture_label}"
                )

            annotations.setdefault(sequence_name, {})[frame_number] = posture_label
            valid_rows += 1

    if valid_rows == 0:
        raise RuntimeError(f"No valid UR-Fall annotations read from {csv_path}")

    print(
        f"Loaded UR-Fall annotations: {len(annotations)} fall sequences | "
        f"{valid_rows} annotated frames"
    )
    return annotations


def get_window_label_from_urfall(frame_labels: Sequence[int]) -> Optional[int]:
    """
    Convert UR-Fall posture labels to one binary window label.

    Same rule as the RF/ST-GCN/MLP experiments:
      - ignore posture label 0 (falling transition)
      - -1 -> ADL/non-fall (0)
      -  1 -> Fall (1)
      - majority vote over the remaining labels
      - no remaining labels or an exact tie -> None (drop the window)
    """
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


def video_type_label(value) -> int:
    """Return the original UR-Fall video class for CV stratification."""
    sequence_name = canonical_sequence_name(value)
    if sequence_name is None:
        raise ValueError(f"Cannot infer UR-Fall video type from: {value!r}")
    return 1 if sequence_name.startswith("fall-") else 0


def video_type_labels(groups: np.ndarray) -> np.ndarray:
    return np.asarray([video_type_label(g) for g in groups], dtype=np.int64)


# -----------------------------
# Missing-value preprocessing
# -----------------------------


# -----------------------------
# Standardization
# -----------------------------

@dataclass
class SequenceStandardizer:
    mean: np.ndarray
    std: np.ndarray

    @classmethod
    def fit(cls, X: np.ndarray) -> "SequenceStandardizer":
        # X: [N, T, D], fit only on training fold.
        flat = X.reshape(-1, X.shape[-1])
        mean = flat.mean(axis=0, keepdims=True).astype(np.float32)
        std = flat.std(axis=0, keepdims=True).astype(np.float32)
        std[std < 1e-6] = 1.0
        return cls(mean=mean, std=std)

    def transform(self, X: np.ndarray) -> np.ndarray:
        return ((X - self.mean[None, :, :]) / self.std[None, :, :]).astype(
            np.float32
        )


# -----------------------------
# Torch dataset
# -----------------------------

class WindowDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.from_numpy(X.astype(np.float32))
        self.y = torch.from_numpy(y.astype(np.int64))

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int):
        return self.X[idx], self.y[idx]


# -----------------------------
# Models
# -----------------------------

class RecurrentClassifier(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_size: int,
        num_layers: int,
        dropout: float,
        num_classes: int = 2,
    ):
        super().__init__()

        self.input_bn = nn.BatchNorm1d(input_dim)

        recurrent_dropout = dropout if num_layers > 1 else 0.0

        self.recurrent = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=recurrent_dropout,
        )

        self.recurrent_bn = nn.BatchNorm1d(hidden_size)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, D]
        B, T, D = x.shape

        # BatchNorm after input.
        x = self.input_bn(x.reshape(B * T, D)).reshape(B, T, D)

        # recurrent output: [B, T, H]
        out, _ = self.recurrent(x)

        B2, T2, H = out.shape

        # BatchNorm after recurrent layer(s).
        out = self.recurrent_bn(out.reshape(B2 * T2, H)).reshape(B2, T2, H)

        # Sequence classification from last timestep.
        last = out[:, -1, :]
        logits = self.fc(last)
        return logits


# -----------------------------
# Training / inference
# -----------------------------

def train_one_model(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    epochs: int,
    learning_rate: float,
    weight_decay: float,
    grad_clip: float,
) -> List[float]:
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )

    model.to(device)
    history: List[float] = []

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        seen = 0

        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)

            optimizer.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()

            if grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

            optimizer.step()

            batch_n = yb.size(0)
            running_loss += loss.item() * batch_n
            seen += batch_n

        epoch_loss = running_loss / max(seen, 1)
        history.append(epoch_loss)

        if epoch == 1 or epoch % 5 == 0 or epoch == epochs:
            print(f"    epoch {epoch:03d}/{epochs} | train_loss={epoch_loss:.6f}")

    return history


@torch.no_grad()
def predict(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray]:
    model.eval()

    probs: List[np.ndarray] = []
    preds: List[np.ndarray] = []

    for xb, _ in loader:
        xb = xb.to(device)
        logits = model(xb)
        p = torch.softmax(logits, dim=1)[:, 1]
        yhat = torch.argmax(logits, dim=1)

        probs.append(p.detach().cpu().numpy())
        preds.append(yhat.detach().cpu().numpy())

    return np.concatenate(preds), np.concatenate(probs)


# -----------------------------
# Metrics
# -----------------------------

def safe_roc_auc(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(roc_auc_score(y_true, y_prob))


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
) -> Dict[str, float]:
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(
            precision_score(y_true, y_pred, pos_label=1, zero_division=0)
        ),
        "recall": float(
            recall_score(y_true, y_pred, pos_label=1, zero_division=0)
        ),
        "f1": float(f1_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "roc_auc": safe_roc_auc(y_true, y_prob),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def fmt_metric(v: float) -> str:
    if isinstance(v, float) and math.isnan(v):
        return "nan"
    return f"{v:.6f}" if isinstance(v, float) else str(v)


seed_everything = experiment_common.seed_everything
choose_device = experiment_common.choose_device
video_type_labels = experiment_common.video_type_labels
compute_metrics = experiment_common.compute_metrics
fmt_metric = experiment_common.fmt_metric


# -----------------------------
# Cross-validation experiment
# -----------------------------

def run_model_cv(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    records_df: pd.DataFrame,
    args,
    device: torch.device,
) -> None:
    model_dir = args.experiment_output_dir
    model_dir.mkdir(parents=True, exist_ok=True)

    models_dir = model_dir / "models"
    histories_dir = model_dir / "histories"

    models_dir.mkdir(parents=True, exist_ok=True)
    histories_dir.mkdir(parents=True, exist_ok=True)

    unique_groups = np.unique(groups)
    n_splits = min(args.folds, len(unique_groups))

    # Stratify by the ORIGINAL video type (adl/fall), not by the majority
    # of window labels. A fall video can legitimately contain many ADL windows.
    split_y = video_type_labels(groups)
    unique_group_types = (
        pd.DataFrame({"group": groups, "video_type": split_y})
        .drop_duplicates("group")
    )
    class_group_counts = np.bincount(
        unique_group_types["video_type"].to_numpy(dtype=np.int64),
        minlength=2,
    )
    max_valid_splits = int(class_group_counts.min())

    if max_valid_splits < 2:
        raise RuntimeError(
            "Need at least 2 videos/groups in each class for grouped CV. "
            f"Group counts by class: ADL={class_group_counts[0]}, "
            f"Fall={class_group_counts[1]}"
        )

    n_splits = min(n_splits, max_valid_splits)

    splitter = StratifiedGroupKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=args.seed,
    )

    all_true: List[int] = []
    all_pred: List[int] = []
    all_prob: List[float] = []
    all_test_indices: List[int] = []
    fold_rows: List[Dict[str, float]] = []

    metrics_header = [
        f"data_root: {args.data_root}",
        f"annotation_csv: {FALL_ANNOTATION_CSV}",
        "label_level: window",
        "urfall_window_label_rule: ignore_0_then_majority_vote_-1_vs_1",
        f"feature_mode: {args.feature_mode}",
        f"input_dim: {X.shape[-1]}",
        f"window_size: {args.window_size}",
        f"stride: {args.stride}",
        f"missing_mode: {args.missing_mode}",
        f"min_valid_frames: {args.min_valid_frames}",
        f"visibility_threshold: {args.visibility_threshold}",
        f"hidden_size: {args.hidden_size}",
        f"num_layers: {args.num_layers}",
        f"dropout: {args.dropout}",
        f"epochs: {args.epochs}",
        f"batch_size: {args.batch_size}",
        f"learning_rate: {args.learning_rate}",
        f"weight_decay: {args.weight_decay}",
        f"folds: {n_splits}",
        f"seed: {args.seed}",
        f"device: {device}",
    ]

    for fold, (train_idx, test_idx) in enumerate(
        splitter.split(X, split_y, groups=groups),
        start=1,
    ):
        print("\n" + "=" * 72)
        print(
            f"LSTM | Fold {fold}/{n_splits} | "
            f"train_windows={len(train_idx)} | test_windows={len(test_idx)}"
        )

        train_groups = set(groups[train_idx])
        test_groups = set(groups[test_idx])
        overlap = train_groups & test_groups
        if overlap:
            raise RuntimeError(f"Video leakage detected: {sorted(overlap)[:5]}")

        # Fit standardizer ONLY on this fold's training data.
        scaler = SequenceStandardizer.fit(X[train_idx])
        X_train = scaler.transform(X[train_idx])
        X_test = scaler.transform(X[test_idx])

        y_train = y[train_idx]
        y_test = y[test_idx]

        train_loader = DataLoader(
            WindowDataset(X_train, y_train),
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=0,
            drop_last=False,
        )
        test_loader = DataLoader(
            WindowDataset(X_test, y_test),
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=0,
            drop_last=False,
        )

        seed_everything(args.seed + fold)

        model = RecurrentClassifier(
            input_dim=X.shape[-1],
            hidden_size=args.hidden_size,
            num_layers=args.num_layers,
            dropout=args.dropout,
            num_classes=2,
        )

        history = train_one_model(
            model=model,
            loader=train_loader,
            device=device,
            epochs=args.epochs,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            grad_clip=args.grad_clip,
        )

        y_pred, y_prob = predict(model, test_loader, device)
        fold_metrics = compute_metrics(y_test, y_pred, y_prob)
        fold_metrics["fold"] = fold
        fold_metrics["train_videos"] = len(train_groups)
        fold_metrics["test_videos"] = len(test_groups)
        fold_metrics["train_windows"] = len(train_idx)
        fold_metrics["test_windows"] = len(test_idx)
        fold_rows.append(fold_metrics)

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
            f"FN={fold_metrics['fn']} TP={fold_metrics['tp']}"
        )

        # Save fold checkpoint + training history.
        checkpoint = {
            "model_type": "lstm",
            "input_dim": X.shape[-1],
            "hidden_size": args.hidden_size,
            "num_layers": args.num_layers,
            "dropout": args.dropout,
            "state_dict": model.state_dict(),
            "feature_mean": scaler.mean,
            "feature_std": scaler.std,
            "args": vars(args).copy(),
        }
        # Convert Path values so checkpoint metadata is serializable/readable.
        for k, v in list(checkpoint["args"].items()):
            if isinstance(v, Path):
                checkpoint["args"][k] = str(v)

        torch.save(checkpoint, models_dir / f"fold_{fold}.pt")
        pd.DataFrame(
            {
                "epoch": np.arange(1, len(history) + 1),
                "train_loss": history,
            }
        ).to_csv(histories_dir / f"fold_{fold}_history.csv", index=False)

        all_true.extend(y_test.tolist())
        all_pred.extend(y_pred.tolist())
        all_prob.extend(y_prob.tolist())
        all_test_indices.extend(test_idx.tolist())

    # Overall out-of-fold metrics
    y_true_all = np.asarray(all_true, dtype=np.int64)
    y_pred_all = np.asarray(all_pred, dtype=np.int64)
    y_prob_all = np.asarray(all_prob, dtype=np.float32)

    overall = compute_metrics(y_true_all, y_pred_all, y_prob_all)
    cm = confusion_matrix(y_true_all, y_pred_all, labels=[0, 1])
    report = classification_report(
        y_true_all,
        y_pred_all,
        labels=[0, 1],
        target_names=CLASS_NAMES,
        digits=4,
        zero_division=0,
    )

    fold_df = pd.DataFrame(fold_rows)
    experiment_common.save_experiment_metrics_text(
        output=model_dir,
        model="lstm",
        header=metrics_header,
        fold_rows=fold_df,
        overall=overall,
        y_true=y_true_all,
        y_pred=y_pred_all,
        confusion=cm,
    )
    experiment_common.save_experiment_metrics_json(
        output=model_dir,
        model="lstm",
        data_root=args.data_root,
        overall=overall,
        fold_rows=fold_df,
        device=device,
    )

    fold_df.to_csv(model_dir / "fold_metrics.csv", index=False)

    pd.DataFrame(
        cm,
        index=["true_ADL", "true_Fall"],
        columns=["pred_ADL", "pred_Fall"],
    ).to_csv(model_dir / "confusion_matrix.csv")

    # OOF predictions aligned to original window metadata.
    pred_df = records_df.iloc[all_test_indices].copy().reset_index(drop=True)
    pred_df["y_true"] = y_true_all
    pred_df["y_pred"] = y_pred_all
    pred_df["fall_probability"] = y_prob_all
    pred_df.to_csv(model_dir / "predictions.csv", index=False)

    print("\n" + "=" * 72)
    print("LSTM OVERALL")
    print(
        f"Accuracy={overall['accuracy']:.4f} | "
        f"Precision={overall['precision']:.4f} | "
        f"Recall={overall['recall']:.4f} | "
        f"F1={overall['f1']:.4f} | "
        f"ROC-AUC={fmt_metric(overall['roc_auc'])}"
    )
    print(report)
    print(f"Saved: {metrics_path}")


# -----------------------------
# CLI
# -----------------------------

def parse_joint_indices(text: Optional[str]) -> Optional[List[int]]:
    if text is None:
        return None
    values = [v.strip() for v in text.split(",") if v.strip()]
    return [int(v) for v in values]


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Fall detection with grouped-CV LSTM on BlazePose sequences."
    )

    p.add_argument(
        "--data-root",
        type=Path,
        default=Path("data/keypoints_normalized"),
        help="Root containing label subfolders and .npz files.",
    )
    p.add_argument("--windows-cache", type=Path, required=True)
    p.add_argument(
        "--output-root",
        type=Path,
        default=Path("results"),
        help=(
            "Base output directory. A dataset-specific subdirectory is "
            "created automatically from --data-root, e.g. "
            "results/lstm_normalized/."
        ),
    )
    p.add_argument(
        "--feature-mode",
        choices=["xy66", "joints16"],
        default="xy66",
        help="xy66 = 33*(x,y); joints16 = exactly 8 selected joints*(x,y).",
    )
    p.add_argument(
        "--joint-indices",
        type=str,
        default=None,
        help="Comma-separated 8 BlazePose indices for joints16 mode.",
    )

    p.add_argument("--window-size", type=int, default=30)
    p.add_argument("--stride", type=int, default=1)

    p.add_argument(
        "--missing-mode",
        choices=["zero", "interp"],
        default="interp",
        help="interp fills missing joints within each window like RF; zero fills them with 0.",
    )
    p.add_argument("--min-valid-frames", type=int, default=1,
                   help="Minimum detected-pose frames required per window (RF default: 1).")
    p.add_argument("--visibility-threshold", type=float, default=0.3)

    p.add_argument("--hidden-size", type=int, default=80)
    p.add_argument("--num-layers", type=int, default=10)
    p.add_argument("--dropout", type=float, default=0.0)

    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--learning-rate", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--grad-clip", type=float, default=5.0)

    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--device",
        type=str,
        default="auto",
        help="auto / cpu / cuda / mps",
    )

    return p


def dataset_run_name(data_root: Path) -> str:
    """Map raw and normalized inputs to the standard LSTM result folders."""
    name = data_root.resolve().name.lower()
    return "lstm_normalized" if "normalized" in name else "lstm"


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    args.joint_indices = parse_joint_indices(args.joint_indices)
    if args.min_valid_frames < 0 or args.min_valid_frames > args.window_size:
        parser.error("--min-valid-frames must be between 0 and --window-size")

    # One run = one data root. Store each dataset in its own output folder,
    # so raw and normalized experiments are kept completely separate.
    run_name = dataset_run_name(args.data_root)
    args.experiment_output_dir = args.output_root / run_name
    args.experiment_output_dir.mkdir(parents=True, exist_ok=True)

    seed_everything(args.seed)
    device = choose_device(args.device)

    print(f"Device: {device}")
    print(f"Data root: {args.data_root}")
    print(f"Output dir: {args.experiment_output_dir}")
    print(
        f"Window={args.window_size}, stride={args.stride}, "
        f"feature_mode={args.feature_mode}, missing_mode={args.missing_mode}"
    )

    X, y, groups, records_df, cache_config = experiment_common.load_windows_cache(
        args.windows_cache
    )

    # Save experiment configuration.
    config = vars(args).copy()
    for k, v in list(config.items()):
        if isinstance(v, Path):
            config[k] = str(v)
    config.update(
        {
            "annotation_csv": str(FALL_ANNOTATION_CSV),
            "label_level": "window",
            "urfall_window_label_rule": "ignore_0_then_majority_vote_-1_vs_1",
        }
    )
    (args.experiment_output_dir / "config.json").write_text(
        json.dumps(config, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    run_model_cv(
        X=X,
        y=y,
        groups=groups,
        records_df=records_df,
        args=args,
        device=device,
    )


if __name__ == "__main__":
    main()
