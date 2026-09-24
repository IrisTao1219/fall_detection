#!/usr/bin/env python3
"""Standalone MLP baseline for UR-Fall window-based fall detection.

This file is fully self-contained and does not import experiment_lstm.
Its data loading, frame-index mapping, sliding-window construction, UR-Fall
CSV labeling, device selection, reproducibility helpers, and metrics are
implemented directly here and aligned with the current LSTM experiment.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import re
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
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
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

try:
    from . import common as experiment_common
    from .common import (
        compute_metrics,
        choose_device,
        fmt_metric,
        inner_group_split_by_video_type,
        load_sequence_windows,
        make_loader,
        probabilities,
        seed_everything,
        video_type_labels,
    )
except ImportError:
    import common as experiment_common
    from common import (
        compute_metrics,
        choose_device,
        fmt_metric,
        inner_group_split_by_video_type,
        load_sequence_windows,
        make_loader,
        probabilities,
        seed_everything,
        video_type_labels,
    )


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
    video_level_label: int,
    video_id: str,
    source_file: str,
    window_size: int,
    stride: int,
    frame_indices: np.ndarray,
    sequence_annotations: Optional[Dict[int, int]],
    valid_mask: Optional[np.ndarray] = None,
    min_valid_frames: int = 1,
    missing_mode: str = "interp",
) -> Tuple[List[np.ndarray], List[int], List[WindowRecord]]:
    T = features.shape[0]

    if T < window_size:
        return [], [], []

    frame_indices = np.asarray(frame_indices).reshape(-1)
    if len(frame_indices) != T:
        raise ValueError(
            f"frame_indices length {len(frame_indices)} != feature frames {T}"
        )

    X_list: List[np.ndarray] = []
    y_list: List[int] = []
    records: List[WindowRecord] = []

    for start in range(0, T - window_size + 1, stride):
        end = start + window_size

        if (
            valid_mask is not None
            and int(np.count_nonzero(valid_mask[start:end])) < min_valid_frames
        ):
            continue

        # Window label: exactly the same UR-Fall CSV rule as RF/ST-GCN/MLP.
        if video_level_label == 0:
            # ADL videos are normal for the whole sequence.
            y_win = 0
        else:
            if sequence_annotations is None:
                raise RuntimeError(
                    f"Missing frame annotations for fall video {video_id}"
                )

            posture_labels: List[int] = []
            for frame_number in frame_indices[start:end]:
                posture_label = sequence_annotations.get(int(frame_number))
                if posture_label is not None:
                    posture_labels.append(posture_label)

            y_from_csv = get_window_label_from_urfall(posture_labels)
            if y_from_csv is None:
                # All annotated frames are posture 0, no annotations are present,
                # or -1/1 are tied after ignoring 0.
                continue
            y_win = int(y_from_csv)

        x_win = features[start:end].copy()
        if missing_mode == "interp":
            for feature_idx in range(x_win.shape[1]):
                x_win[:, feature_idx] = interpolate_1d(x_win[:, feature_idx])
        elif missing_mode == "zero":
            x_win = np.nan_to_num(x_win, nan=0.0, posinf=0.0, neginf=0.0)
        else:
            raise ValueError(f"Unknown missing_mode: {missing_mode}")

        X_list.append(x_win.astype(np.float32))
        y_list.append(y_win)
        records.append(
            WindowRecord(
                video_id=video_id,
                source_file=source_file,
                start_frame=int(frame_indices[start]),
                end_frame=int(frame_indices[end - 1]),
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
    return experiment_common.load_sequence_windows(
        data_root=data_root,
        window_size=window_size,
        stride=stride,
        visibility_threshold=visibility_threshold,
        missing_mode=missing_mode,
        feature_mode=feature_mode,
        joint_indices=joint_indices,
        min_valid_frames=min_valid_frames,
        annotation_csv=FALL_ANNOTATION_CSV,
    )



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


class MLP(nn.Module):
    def __init__(self, input_dim: int, hidden: tuple[int, ...], dropout: float):
        super().__init__()
        layers: list[nn.Module] = []
        width = input_dim
        for next_width in hidden:
            layers.extend((nn.Linear(width, next_width), nn.ReLU(), nn.Dropout(dropout)))
            width = next_width
        layers.append(nn.Linear(width, 2))
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x.flatten(start_dim=1))


def make_loader(x: np.ndarray, y: np.ndarray, batch_size: int, shuffle: bool) -> DataLoader:
    dataset = TensorDataset(torch.from_numpy(x), torch.from_numpy(y.astype(np.int64)))
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


@torch.no_grad()
def probabilities(model: nn.Module, loader: DataLoader, device: torch.device) -> np.ndarray:
    model.eval()
    pieces = []
    for xb, _ in loader:
        pieces.append(torch.softmax(model(xb.to(device)), dim=1)[:, 1].cpu().numpy())
    return np.concatenate(pieces)


def inner_split(y: np.ndarray, groups: np.ndarray, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """
    内层 train/validation 仍按视频分组，但按原始视频类型做 stratify。
    同一个 fall 视频可以同时包含 normal/fall 窗口。
    """
    return experiment_common.inner_group_split_by_video_type(y, groups, seed)


seed_everything = experiment_common.seed_everything
choose_device = experiment_common.choose_device
video_type_labels = experiment_common.video_type_labels
make_loader = experiment_common.make_loader
probabilities = experiment_common.probabilities
compute_metrics = experiment_common.compute_metrics
fmt_metric = experiment_common.fmt_metric


def train_fold(x: np.ndarray, y: np.ndarray, groups: np.ndarray, args, device, seed: int):
    fit_idx, val_idx = inner_split(y, groups, seed)
    # Each feature's statistics come only from the inner training videos.
    mean = x[fit_idx].mean(axis=0, keepdims=True)
    std = x[fit_idx].std(axis=0, keepdims=True)
    std[std < 1e-6] = 1.0
    x_fit = ((x[fit_idx] - mean) / std).astype(np.float32)
    x_val = ((x[val_idx] - mean) / std).astype(np.float32)
    fit_loader = make_loader(x_fit, y[fit_idx], args.batch_size, True)
    val_loader = make_loader(x_val, y[val_idx], args.batch_size, False)

    seed_everything(seed)
    model = MLP(x.shape[1], args.hidden, args.dropout).to(device)
    counts = np.bincount(y[fit_idx], minlength=2)
    weights = torch.tensor(len(fit_idx) / (2 * counts), dtype=torch.float32, device=device)
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    best_f1, best_loss, best_state, best_epoch = -1.0, float("inf"), None, 0
    history = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        for xb, yb in fit_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(yb)
        val_prob = probabilities(model, val_loader, device)
        val_pred = (val_prob >= 0.5).astype(np.int64)
        val_f1 = compute_metrics(y[val_idx], val_pred, val_prob)["f1"]
        train_loss = total_loss / len(fit_idx)
        history.append({"epoch": epoch, "train_loss": train_loss, "val_f1": val_f1})
        if val_f1 > best_f1 or (val_f1 == best_f1 and train_loss < best_loss):
            best_f1, best_loss, best_epoch = val_f1, train_loss, epoch
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        if epoch - best_epoch >= args.patience:
            break

    model.load_state_dict(best_state)
    return model, mean, std, best_epoch, best_f1, history, (fit_idx, val_idx)


def parse_args():
    parser = argparse.ArgumentParser(description="Video-grouped MLP fall detection experiment")
    parser.add_argument("--data-root", type=Path, default=Path("data/keypoints_normalized"))
    parser.add_argument("--output-root", type=Path, default=Path("results"))
    parser.add_argument("--window-size", type=int, default=30)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--missing-mode", choices=("zero", "interp"), default="interp")
    parser.add_argument("--visibility-threshold", type=float, default=0.3)
    parser.add_argument("--hidden", type=int, nargs="+", default=[256, 64])
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.window_size < 1 or args.stride < 1 or args.folds < 2 or args.patience < 1:
        raise ValueError("window-size, stride, patience must be positive; folds must be at least 2")
    if not args.hidden or any(width < 1 for width in args.hidden):
        raise ValueError("--hidden requires positive layer widths")
    seed_everything(args.seed)
    device = choose_device(args.device)
    run_name = "mlp_normalized" if "normalized" in args.data_root.resolve().name.lower() else "mlp"
    output = args.output_root / run_name
    output.mkdir(parents=True, exist_ok=True)
    models_dir = output / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    history_dir = output / "histories"
    history_dir.mkdir(parents=True, exist_ok=True)

    # 本文件内置的数据加载逻辑与当前 LSTM 版本保持一致：
    # 1. 优先使用 NPZ 中的 frame_indices；没有时使用 1..T。
    # 2. ADL 视频的所有窗口标为 0。
    # 3. Fall 视频直接根据 UR-Fall CSV 对当前窗口逐帧取标签。
    # 4. 忽略 posture=0，对 -1/1 多数投票；无有效标签或平票则丢弃窗口。
    x, y, groups, records = load_all_windows(
        data_root=args.data_root,
        window_size=args.window_size,
        stride=args.stride,
        visibility_threshold=args.visibility_threshold,
        missing_mode=args.missing_mode,
        feature_mode="xy66",
        joint_indices=None,
        min_valid_frames=1,
    )

    x = x.reshape(len(x), -1)

    # 同一个 fall 视频现在可同时包含 normal/fall 窗口，
    # 所以 fold 的 stratification 使用视频名本身的 fall/adl 类型，
    # 模型真正训练与指标计算仍使用窗口级 y。
    split_y = video_type_labels(groups)
    unique_group_types = (
        pd.DataFrame({"group": groups, "video_type": split_y})
        .drop_duplicates("group")
    )
    class_groups = np.bincount(
        unique_group_types["video_type"].to_numpy(dtype=np.int64),
        minlength=2,
    )
    if class_groups.min() < args.folds:
        raise ValueError(
            f"Need at least {args.folds} videos per class; "
            f"found {class_groups.tolist()}"
        )

    splitter = StratifiedGroupKFold(
        n_splits=args.folds,
        shuffle=True,
        random_state=args.seed,
    )
    predictions, fold_rows = [], []

    for fold, (train_idx, test_idx) in enumerate(splitter.split(x, split_y, groups), 1):
        if set(groups[train_idx]) & set(groups[test_idx]):
            raise RuntimeError("Training and test videos overlap")
        model, mean, std, best_epoch, val_f1, history, (fit_idx, val_idx) = train_fold(
            x[train_idx], y[train_idx], groups[train_idx], args, device, args.seed + fold
        )
        test_x = ((x[test_idx] - mean) / std).astype(np.float32)
        prob = probabilities(model, make_loader(test_x, y[test_idx], args.batch_size, False), device)
        pred = (prob >= 0.5).astype(np.int64)
        metrics = compute_metrics(y[test_idx], pred, prob)
        fold_rows.append({
            **metrics, "fold": fold,
            "train_videos": len(set(groups[train_idx])),
            "test_videos": len(set(groups[test_idx])),
            "train_windows": len(train_idx),
            "test_windows": len(test_idx),
            "best_epoch": best_epoch, "val_f1": val_f1,
        })
        frame = records.iloc[test_idx].copy()
        frame["y_true"] = y[test_idx]
        frame["y_pred"] = pred
        frame["fall_probability"] = prob
        predictions.append(frame)
        torch.save({
            "model_type": "mlp",
            "state_dict": model.state_dict(), "input_dim": x.shape[1],
            "hidden": args.hidden, "dropout": args.dropout,
            "feature_mean": mean, "feature_std": std,
            "best_epoch": best_epoch,
            "args": {key: str(value) if isinstance(value, Path) else value
                     for key, value in vars(args).items()},
        }, models_dir / f"fold_{fold}.pt")
        pd.DataFrame(history).to_csv(history_dir / f"fold_{fold}_history.csv", index=False)
        print(f"Fold {fold}: test F1={metrics['f1']:.4f}, recall={metrics['recall']:.4f}; best epoch={best_epoch}")

    pred_df = pd.concat(predictions, ignore_index=True)
    pred_df.to_csv(output / "predictions.csv", index=False)
    fold_df = pd.DataFrame(fold_rows)
    fold_df.to_csv(output / "fold_metrics.csv", index=False)

    y_true = pred_df["y_true"].to_numpy()
    y_pred = pred_df["y_pred"].to_numpy()
    y_prob = pred_df["fall_probability"].to_numpy()
    overall = compute_metrics(y_true, y_pred, y_prob)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    pd.DataFrame(cm, index=["true_ADL", "true_Fall"],
                 columns=["pred_ADL", "pred_Fall"]).to_csv(output / "confusion_matrix.csv")
    report = classification_report(y_true, y_pred, labels=[0, 1],
                                   target_names=["ADL", "Fall"], digits=4,
                                   zero_division=0)
    lines = ["=" * 80, "MODEL: MLP", "=" * 80,
             f"data_root: {args.data_root}", "feature_mode: xy66",
             f"input_dim: {x.shape[1]}", f"window_size: {args.window_size}",
             f"stride: {args.stride}", f"missing_mode: {args.missing_mode}",
             "min_valid_frames: 1", f"visibility_threshold: {args.visibility_threshold}",
             f"hidden: {args.hidden}", f"dropout: {args.dropout}",
             f"epochs: {args.epochs}", f"patience: {args.patience}",
             f"batch_size: {args.batch_size}", f"learning_rate: {args.learning_rate}",
             f"weight_decay: {args.weight_decay}", f"folds: {args.folds}",
             f"seed: {args.seed}", f"device: {device}",
             f"annotation_csv: {FALL_ANNOTATION_CSV}",
             "label_level: window",
             "window_label_rule: ignore posture 0; majority vote -1(normal) vs 1(fall)",
             ""]
    metric_keys = ("accuracy", "precision", "recall", "f1", "roc_auc", "tn", "fp", "fn", "tp")
    for row in fold_rows:
        fold = row["fold"]
        lines.extend([f"[Fold {fold}/{args.folds}]",
                      f"train_videos: {row['train_videos']}",
                      f"test_videos: {row['test_videos']}",
                      f"train_windows: {row['train_windows']}",
                      f"test_windows: {row['test_windows']}",
                      f"best_epoch: {row['best_epoch']}",
                      f"val_f1: {fmt_metric(row['val_f1'])}"])
        lines.extend(f"{key}: {fmt_metric(row[key])}" for key in metric_keys)
        lines.append("")
    lines.extend(["=" * 80, "OVERALL OUT-OF-FOLD METRICS", "=" * 80])
    lines.extend(f"{key}: {fmt_metric(overall[key])}" for key in metric_keys)
    lines.extend(["", "Classification report:", report,
                  "Confusion matrix [[TN, FP], [FN, TP]]:", np.array2string(cm),
                  "", "Fold mean ± std:"])
    for key in ("accuracy", "precision", "recall", "f1", "roc_auc"):
        lines.append(f"{key}: {fold_df[key].mean():.6f} ± {fold_df[key].std(ddof=1):.6f}")
    (output / "metrics.txt").write_text("\n".join(lines), encoding="utf-8")
    config = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    config["experiment_output_dir"] = str(output)
    config["annotation_csv"] = str(FALL_ANNOTATION_CSV)
    config["label_level"] = "window"
    config["urfall_window_label_rule"] = "ignore_0_then_majority_vote_-1_vs_1"
    (output / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    print(f"Overall F1={overall['f1']:.4f}, recall={overall['recall']:.4f}")
    print(f"Results: {output}")


if __name__ == "__main__":
    main()
