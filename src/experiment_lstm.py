#!/usr/bin/env python3
"""
Fall detection experiment with LSTM on BlazePose keypoint sequences.

Compatible with the user's existing NPZ structure:
    keypoints:   [T, 33, 4] -> x, y, z, visibility
    valid_mask:  [T]
    fps:         scalar
    label:       scalar ("adl"/"fall" or 0/1), OR optional frame-level labels [T]
    video_id:    scalar

Default experiment:
- feature: all BlazePose x/y coordinates -> 33 * 2 = 66 dims per frame
- window length: 30 frames
- stride: 1 frame
- missing joint handling: interpolate within each window (matching RF)
- discard windows with no detected pose frames (matching RF)
- LSTM: 10 recurrent layers, 80 hidden units
- BatchNorm after input and recurrent outputs
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


# -----------------------------
# Missing-value preprocessing
# -----------------------------

def interpolate_1d(values: np.ndarray) -> np.ndarray:
    """
    Linear interpolation along time.
    If all values are missing, returns zeros.
    Edge NaNs are filled with the nearest valid value.
    """
    out = values.astype(np.float32, copy=True)
    idx = np.arange(len(out))
    valid = np.isfinite(out)

    if not np.any(valid):
        return np.zeros_like(out, dtype=np.float32)

    out[~valid] = np.interp(idx[~valid], idx[valid], out[valid])
    return out


def preprocess_keypoints(
    keypoints: np.ndarray,
    valid_mask: Optional[np.ndarray],
    visibility_threshold: float,
    missing_mode: str,
) -> np.ndarray:
    """
    Return cleaned x/y tensor with shape [T, 33, 2].

    A frame/joint is considered missing when:
      - x or y is NaN/inf
      - frame valid_mask is False
      - visibility exists and is below threshold

    missing_mode:
      mask   -> leave missing coordinates as NaN for window-level processing
      zero   -> unidentified joints become [0, 0]
      interp -> temporal interpolation over the full video (legacy behavior)
    """
    if keypoints.ndim != 3 or keypoints.shape[1] != 33 or keypoints.shape[2] < 2:
        raise ValueError(
            f"Expected keypoints [T,33,C>=2], got shape {keypoints.shape}"
        )

    xy = keypoints[..., :2].astype(np.float32, copy=True)
    T = xy.shape[0]

    joint_valid = np.isfinite(xy).all(axis=-1)

    if keypoints.shape[2] >= 4:
        visibility = keypoints[..., 3]
        joint_valid &= np.isfinite(visibility) & (visibility >= visibility_threshold)

    if valid_mask is not None:
        valid_mask = np.asarray(valid_mask).reshape(-1).astype(bool)
        if len(valid_mask) != T:
            raise ValueError(
                f"valid_mask length {len(valid_mask)} != keypoints frames {T}"
            )
        joint_valid &= valid_mask[:, None]

    xy[~joint_valid] = np.nan

    if missing_mode == "mask":
        pass
    elif missing_mode == "zero":
        xy = np.nan_to_num(xy, nan=0.0, posinf=0.0, neginf=0.0)

    elif missing_mode == "interp":
        for joint_idx in range(xy.shape[1]):
            for coord_idx in range(2):
                xy[:, joint_idx, coord_idx] = interpolate_1d(
                    xy[:, joint_idx, coord_idx]
                )
        xy = np.nan_to_num(xy, nan=0.0, posinf=0.0, neginf=0.0)

    else:
        raise ValueError(f"Unknown missing_mode: {missing_mode}")

    return xy.astype(np.float32)


# -----------------------------
# Feature construction
# -----------------------------

def build_frame_features(
    xy: np.ndarray,
    feature_mode: str,
    joint_indices: Optional[Sequence[int]],
) -> np.ndarray:
    """
    xy66:
        33 joints * (x,y) = 66-D per frame.

    joints16:
        exactly 8 user-selected joints * (x,y) = 16-D per frame.
        This mode is intentionally explicit because the paper does not
        document which 16 scalar dimensions were used.
    """
    if feature_mode == "xy66":
        return xy.reshape(xy.shape[0], -1).astype(np.float32)

    if feature_mode == "joints16":
        if joint_indices is None or len(joint_indices) != 8:
            raise ValueError(
                "--feature-mode joints16 requires exactly 8 indices via "
                "--joint-indices, e.g. 11,12,23,24,25,26,27,28"
            )
        idx = np.asarray(joint_indices, dtype=np.int64)
        if np.any(idx < 0) or np.any(idx >= 33):
            raise ValueError("All joint indices must be in [0, 32]")
        return xy[:, idx, :].reshape(xy.shape[0], 16).astype(np.float32)

    raise ValueError(f"Unknown feature_mode: {feature_mode}")


# -----------------------------
# Window generation
# -----------------------------

@dataclass
class WindowRecord:
    video_id: str
    source_file: str
    start_frame: int
    end_frame: int
    label: int


def make_windows(
    features: np.ndarray,
    label_data,
    video_id: str,
    source_file: str,
    window_size: int,
    stride: int,
    valid_mask: Optional[np.ndarray] = None,
    min_valid_frames: int = 1,
    missing_mode: str = "interp",
) -> Tuple[List[np.ndarray], List[int], List[WindowRecord]]:
    T = features.shape[0]

    if T < window_size:
        return [], [], []

    # Scalar video-level label OR frame-level labels.
    label_arr = np.asarray(label_data)
    is_frame_level = label_arr.ndim > 0 and label_arr.size == T

    if is_frame_level:
        frame_labels = normalize_label_array(label_arr.reshape(-1))
        scalar_label = None
    else:
        scalar_label = normalize_label(label_data)
        frame_labels = None

    X_list: List[np.ndarray] = []
    y_list: List[int] = []
    records: List[WindowRecord] = []

    for start in range(0, T - window_size + 1, stride):
        end = start + window_size
        if valid_mask is not None and int(np.count_nonzero(valid_mask[start:end])) < min_valid_frames:
            continue
        x_win = features[start:end].copy()
        if missing_mode == "interp":
            for feature_idx in range(x_win.shape[1]):
                x_win[:, feature_idx] = interpolate_1d(x_win[:, feature_idx])
        elif missing_mode == "zero":
            x_win = np.nan_to_num(x_win, nan=0.0, posinf=0.0, neginf=0.0)
        else:
            raise ValueError(f"Unknown missing_mode: {missing_mode}")

        if frame_labels is not None:
            values, counts = np.unique(frame_labels[start:end], return_counts=True)
            # Majority label. On a tie, choose Fall (1) to avoid silently
            # suppressing a fall-containing window.
            max_count = counts.max()
            candidates = values[counts == max_count]
            y_win = int(candidates.max())
        else:
            y_win = int(scalar_label)

        X_list.append(x_win.astype(np.float32))
        y_list.append(y_win)
        records.append(
            WindowRecord(
                video_id=video_id,
                source_file=source_file,
                start_frame=start,
                end_frame=end - 1,
                label=y_win,
            )
        )

    return X_list, y_list, records


# -----------------------------
# Dataset loading
# -----------------------------

def discover_npz_files(root: Path) -> List[Path]:
    files = sorted(root.rglob("*.npz"))
    if not files:
        raise FileNotFoundError(f"No .npz files found under: {root}")
    return files


def load_all_windows(
    data_root: Path,
    window_size: int,
    stride: int,
    visibility_threshold: float,
    missing_mode: str,
    feature_mode: str,
    joint_indices: Optional[Sequence[int]],
    min_valid_frames: int = 1,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, pd.DataFrame]:
    X_all: List[np.ndarray] = []
    y_all: List[int] = []
    groups_all: List[str] = []
    records_all: List[WindowRecord] = []

    files = discover_npz_files(data_root)
    skipped_short = 0

    for path in files:
        with np.load(path, allow_pickle=True) as d:
            if "keypoints" not in d:
                raise KeyError(f"{path}: missing 'keypoints'")

            keypoints = d["keypoints"]
            valid_mask = d["valid_mask"] if "valid_mask" in d else None

            if "label" in d:
                label_data = d["label"]
            else:
                # Fallback to parent directory: .../<label>/<video>.npz
                label_data = path.parent.name

            if "video_id" in d:
                raw_video_id = d["video_id"]
                if isinstance(raw_video_id, np.ndarray) and raw_video_id.ndim == 0:
                    raw_video_id = raw_video_id.item()
                if isinstance(raw_video_id, bytes):
                    raw_video_id = raw_video_id.decode("utf-8")
                video_id = str(raw_video_id)
            else:
                video_id = path.stem

        xy = preprocess_keypoints(
            keypoints=keypoints,
            valid_mask=valid_mask,
            visibility_threshold=visibility_threshold,
            missing_mode="mask",
        )

        features = build_frame_features(
            xy=xy,
            feature_mode=feature_mode,
            joint_indices=joint_indices,
        )

        X_list, y_list, records = make_windows(
            features=features,
            label_data=label_data,
            video_id=video_id,
            source_file=str(path),
            window_size=window_size,
            stride=stride,
            valid_mask=valid_mask,
            min_valid_frames=min_valid_frames,
            missing_mode=missing_mode,
        )

        if not X_list:
            skipped_short += 1
            continue

        X_all.extend(X_list)
        y_all.extend(y_list)
        groups_all.extend([video_id] * len(X_list))
        records_all.extend(records)

    if not X_all:
        raise RuntimeError("No valid windows were generated.")

    X = np.stack(X_all).astype(np.float32)
    y = np.asarray(y_all, dtype=np.int64)
    groups = np.asarray(groups_all)

    records_df = pd.DataFrame([asdict(r) for r in records_all])

    print(
        f"Loaded {len(files)} NPZ videos | "
        f"windows={len(X)} | shape={X.shape} | "
        f"ADL={int((y == 0).sum())} | Fall={int((y == 1).sum())} | "
        f"short_videos_skipped={skipped_short}"
    )

    return X, y, groups, records_df


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

    # StratifiedGroupKFold also needs enough groups for both classes.
    group_labels = []
    for g in unique_groups:
        vals = y[groups == g]
        values, counts = np.unique(vals, return_counts=True)
        group_labels.append(int(values[np.argmax(counts)]))

    class_group_counts = np.bincount(group_labels, minlength=2)
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

    metrics_lines: List[str] = []
    metrics_lines.append("=" * 80)
    metrics_lines.append("MODEL: LSTM")
    metrics_lines.append("=" * 80)
    metrics_lines.append(f"data_root: {args.data_root}")
    metrics_lines.append(f"feature_mode: {args.feature_mode}")
    metrics_lines.append(f"input_dim: {X.shape[-1]}")
    metrics_lines.append(f"window_size: {args.window_size}")
    metrics_lines.append(f"stride: {args.stride}")
    metrics_lines.append(f"missing_mode: {args.missing_mode}")
    metrics_lines.append(f"min_valid_frames: {args.min_valid_frames}")
    metrics_lines.append(f"visibility_threshold: {args.visibility_threshold}")
    metrics_lines.append(f"hidden_size: {args.hidden_size}")
    metrics_lines.append(f"num_layers: {args.num_layers}")
    metrics_lines.append(f"dropout: {args.dropout}")
    metrics_lines.append(f"epochs: {args.epochs}")
    metrics_lines.append(f"batch_size: {args.batch_size}")
    metrics_lines.append(f"learning_rate: {args.learning_rate}")
    metrics_lines.append(f"weight_decay: {args.weight_decay}")
    metrics_lines.append(f"folds: {n_splits}")
    metrics_lines.append(f"seed: {args.seed}")
    metrics_lines.append(f"device: {device}")
    metrics_lines.append("")

    for fold, (train_idx, test_idx) in enumerate(
        splitter.split(X, y, groups=groups),
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

        metrics_lines.append(f"[Fold {fold}/{n_splits}]")
        metrics_lines.append(f"train_videos: {len(train_groups)}")
        metrics_lines.append(f"test_videos: {len(test_groups)}")
        metrics_lines.append(f"train_windows: {len(train_idx)}")
        metrics_lines.append(f"test_windows: {len(test_idx)}")
        for key in [
            "accuracy",
            "precision",
            "recall",
            "f1",
            "roc_auc",
            "tn",
            "fp",
            "fn",
            "tp",
        ]:
            metrics_lines.append(f"{key}: {fmt_metric(fold_metrics[key])}")
        metrics_lines.append("")

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

    metrics_lines.append("=" * 80)
    metrics_lines.append("OVERALL OUT-OF-FOLD METRICS")
    metrics_lines.append("=" * 80)
    for key in [
        "accuracy",
        "precision",
        "recall",
        "f1",
        "roc_auc",
        "tn",
        "fp",
        "fn",
        "tp",
    ]:
        metrics_lines.append(f"{key}: {fmt_metric(overall[key])}")

    metrics_lines.append("")
    metrics_lines.append("Classification report:")
    metrics_lines.append(report)
    metrics_lines.append("Confusion matrix [[TN, FP], [FN, TP]]:")
    metrics_lines.append(np.array2string(cm))

    # Mean/std of fold metrics (useful for model comparison)
    fold_df = pd.DataFrame(fold_rows)
    metrics_lines.append("")
    metrics_lines.append("Fold mean ± std:")
    for key in ["accuracy", "precision", "recall", "f1", "roc_auc"]:
        mean_v = fold_df[key].mean()
        std_v = fold_df[key].std(ddof=1)
        metrics_lines.append(f"{key}: {mean_v:.6f} ± {std_v:.6f}")

    metrics_path = model_dir / "metrics.txt"
    metrics_path.write_text("\n".join(metrics_lines), encoding="utf-8")

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
    """Keep RF-aligned reruns separate from the previously saved LSTM results."""
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

    X, y, groups, records_df = load_all_windows(
        data_root=args.data_root,
        window_size=args.window_size,
        stride=args.stride,
        visibility_threshold=args.visibility_threshold,
        missing_mode=args.missing_mode,
        feature_mode=args.feature_mode,
        joint_indices=args.joint_indices,
        min_valid_frames=args.min_valid_frames,
    )

    # Save experiment configuration.
    config = vars(args).copy()
    for k, v in list(config.items()):
        if isinstance(v, Path):
            config[k] = str(v)
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
