#!/usr/bin/env python3
"""Temporal Transformer experiment for pose-keypoint fall detection.

The loader accepts keypoints from BlazePose, TransPose, ViTPose, or another
pose estimator as long as every NPZ file follows the contract documented in
REPRODUCE_4_4.md. Joint counts may differ between separate runs, but must be
consistent within one data root.

Example:
    uv run python src/experiment_transformer.py \
        --data-root data/keypoints \
        --run-name transformer_blazepose_urfall
"""

from __future__ import annotations

import argparse
import json
import re
import time
from dataclasses import asdict
from pathlib import Path
from typing import Dict, Optional, Sequence

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import StratifiedGroupKFold
from torch import nn

from lstm import (
    WindowRecord,
    choose_device,
    compute_metrics,
    discover_npz_files,
    seed_everything,
)
from mlp import make_loader, probabilities


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


def normalize_label(value) -> int:
    """Convert a scalar/string video label to {0, 1}."""
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


def canonical_sequence_name(value) -> Optional[str]:
    """Normalize IDs such as fall-01-cam0-rgb to fall-01."""
    value = str(value).strip().lower()
    match = re.search(r"(fall|adl)-?(\d+)", value)
    if match is None:
        return None
    return f"{match.group(1)}-{int(match.group(2)):02d}"


def load_fall_frame_labels(csv_path: Path) -> Dict[str, Dict[int, int]]:
    """Load UR-Fall frame-level posture labels from the first three CSV columns."""
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"UR-Fall annotation CSV not found: {csv_path}")

    annotations: Dict[str, Dict[int, int]] = {}
    valid_rows = 0
    with csv_path.open("r", encoding="utf-8-sig") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            if "," in line:
                row = [item.strip() for item in line.split(",")]
            elif ";" in line:
                row = [item.strip() for item in line.split(";")]
            elif "\t" in line:
                row = [item.strip() for item in line.split("\t")]
            else:
                row = line.split()
            if len(row) < 3:
                continue

            sequence_name = canonical_sequence_name(row[0])
            try:
                frame_number = int(float(row[1]))
                posture_label = int(float(row[2]))
            except ValueError:
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
    """Apply the same UR-Fall window-label rule as the RF/MLP/LSTM/ST-GCN runs."""
    labels = np.asarray(frame_labels, dtype=np.int8)
    labels = labels[labels != 0]  # ignore falling-transition frames
    if labels.size == 0:
        return None

    normal_count = int(np.sum(labels == -1))
    fall_count = int(np.sum(labels == 1))
    if fall_count > normal_count:
        return 1
    if normal_count > fall_count:
        return 0
    return None  # exact tie: drop this window


def video_type_label(value) -> int:
    """Original UR-Fall video type used only for grouped stratification."""
    sequence_name = canonical_sequence_name(value)
    if sequence_name is None:
        raise ValueError(f"Cannot infer UR-Fall video type from: {value!r}")
    return 1 if sequence_name.startswith("fall-") else 0


def video_type_labels(groups: np.ndarray) -> np.ndarray:
    return np.asarray([video_type_label(group) for group in groups], dtype=np.int64)


def interpolate_1d(values: np.ndarray) -> np.ndarray:
    """Linear interpolation in time; edge NaNs use the nearest valid value."""
    out = values.astype(np.float32, copy=True)
    idx = np.arange(len(out))
    valid = np.isfinite(out)
    if not np.any(valid):
        return np.zeros_like(out, dtype=np.float32)
    out[~valid] = np.interp(idx[~valid], idx[valid], out[valid])
    return out


def make_urfall_windows(
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
):
    """Create windows using the exact current UR-Fall CSV labeling rule."""
    frame_count = features.shape[0]
    if frame_count < window_size:
        return [], [], []

    frame_indices = np.asarray(frame_indices).reshape(-1)
    if len(frame_indices) != frame_count:
        raise ValueError(
            f"frame_indices length {len(frame_indices)} != feature frames {frame_count}"
        )

    x_list, y_list, records = [], [], []
    for start in range(0, frame_count - window_size + 1, stride):
        end = start + window_size
        if valid_mask is not None:
            valid_count = int(np.count_nonzero(valid_mask[start:end]))
            if valid_count < min_valid_frames:
                continue

        if video_level_label == 0:
            y_win = 0
        else:
            if sequence_annotations is None:
                raise RuntimeError(f"Missing frame annotations for fall video {video_id}")
            posture_labels = []
            for frame_number in frame_indices[start:end]:
                posture_label = sequence_annotations.get(int(frame_number))
                if posture_label is not None:
                    posture_labels.append(posture_label)
            y_from_csv = get_window_label_from_urfall(posture_labels)
            if y_from_csv is None:
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

        x_list.append(x_win.astype(np.float32))
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
    return x_list, y_list, records


def urfall_inner_split(y: np.ndarray, groups: np.ndarray, seed: int):
    """Video-grouped validation split, stratified by original ADL/fall video type."""
    split_y = video_type_labels(groups)
    unique_groups = np.unique(groups)
    group_type = {}
    for group in unique_groups:
        values = np.unique(split_y[groups == group])
        if len(values) != 1:
            raise RuntimeError(f"Video {group} has multiple video types")
        group_type[group] = int(values[0])

    class_group_counts = np.bincount(
        np.asarray(list(group_type.values()), dtype=np.int64), minlength=2
    )
    n_splits = int(min(5, class_group_counts.min()))
    if n_splits < 2:
        raise RuntimeError(
            "Inner validation needs at least two videos of each original video type; "
            f"found {class_group_counts.tolist()}"
        )

    splitter = StratifiedGroupKFold(
        n_splits=n_splits, shuffle=True, random_state=seed
    )
    for fit_idx, val_idx in splitter.split(np.zeros(len(y)), split_y, groups):
        if len(np.unique(y[fit_idx])) == 2 and len(np.unique(y[val_idx])) == 2:
            return fit_idx, val_idx
    raise ValueError(
        "Could not create a video-grouped validation split with both window classes"
    )


class TemporalTransformer(nn.Module):
    """Lightweight Transformer encoder over a fixed-length pose sequence."""

    def __init__(
        self,
        input_dim: int,
        window_size: int,
        d_model: int = 128,
        nhead: int = 4,
        layers: int = 2,
        feedforward_dim: int = 256,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        if d_model % nhead:
            raise ValueError("d_model must be divisible by nhead")
        self.input_projection = nn.Linear(input_dim, d_model)
        self.position = nn.Parameter(torch.zeros(1, window_size, d_model))
        nn.init.trunc_normal_(self.position, std=0.02)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=feedforward_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=False,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=layers,
            norm=nn.LayerNorm(d_model),
        )
        self.classifier = nn.Sequential(nn.Dropout(dropout), nn.Linear(d_model, 2))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[1] > self.position.shape[1]:
            raise ValueError("Input sequence is longer than the configured window size")
        encoded = self.input_projection(x) + self.position[:, : x.shape[1]]
        encoded = self.encoder(encoded)
        return self.classifier(encoded.mean(dim=1))


def _scalar(value):
    if isinstance(value, np.ndarray) and value.ndim == 0:
        return value.item()
    return value


def _video_id(data, path: Path) -> str:
    if "video_id" not in data:
        return path.stem
    value = _scalar(data["video_id"])
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    return str(value)


def _pose_features(
    keypoints: np.ndarray,
    scores: np.ndarray | None,
    valid_mask: np.ndarray | None,
    visibility_threshold: float,
    feature_mode: str,
    confidence_index: int | None,
) -> tuple[np.ndarray, np.ndarray, int]:
    if keypoints.ndim != 3 or keypoints.shape[2] < 2:
        raise ValueError(f"Expected keypoints [T,J,C>=2], got {keypoints.shape}")
    frames, joints, channels = keypoints.shape
    xy = keypoints[..., :2].astype(np.float32, copy=True)
    joint_valid = np.isfinite(xy).all(axis=-1)

    confidence = None
    if scores is not None:
        confidence = np.asarray(scores, dtype=np.float32)
        if confidence.shape != (frames, joints):
            raise ValueError(
                f"scores must have shape {(frames, joints)}, got {confidence.shape}"
            )
    elif confidence_index is not None:
        index = confidence_index if confidence_index >= 0 else channels + confidence_index
        if index < 0 or index >= channels:
            raise ValueError(f"confidence index {confidence_index} is invalid for C={channels}")
        confidence = keypoints[..., index].astype(np.float32, copy=False)
    elif channels >= 3:
        confidence = keypoints[..., -1].astype(np.float32, copy=False)

    if confidence is not None:
        joint_valid &= np.isfinite(confidence) & (confidence >= visibility_threshold)
    if valid_mask is not None:
        supplied_frame_valid = np.asarray(valid_mask).reshape(-1).astype(bool)
        if len(supplied_frame_valid) != frames:
            raise ValueError(
                f"valid_mask length {len(supplied_frame_valid)} != frame count {frames}"
            )
        joint_valid &= supplied_frame_valid[:, None]
    frame_valid = joint_valid.any(axis=1)

    xy[~joint_valid] = np.nan
    parts = [xy]
    if feature_mode == "xyc":
        if confidence is None:
            confidence = joint_valid.astype(np.float32)
        confidence = confidence.astype(np.float32, copy=True)
        confidence[~joint_valid] = np.nan
        parts.append(confidence[..., None])
    elif feature_mode != "xy":
        raise ValueError(f"Unknown feature mode: {feature_mode}")

    return np.concatenate(parts, axis=-1).reshape(frames, -1), frame_valid, joints


def load_pose_windows(args) -> tuple[np.ndarray, np.ndarray, np.ndarray, pd.DataFrame, int]:
    windows: list[np.ndarray] = []
    labels: list[int] = []
    groups: list[str] = []
    records: list[WindowRecord] = []
    joint_count: int | None = None
    files = discover_npz_files(args.data_root)
    fall_annotations = load_fall_frame_labels(args.annotation_csv)
    skipped_short = 0
    skipped_no_valid_windows = 0

    for path in files:
        with np.load(path, allow_pickle=True) as data:
            if "keypoints" not in data:
                raise KeyError(f"{path}: missing keypoints")
            keypoints = data["keypoints"]
            scores = data["scores"] if "scores" in data else None
            valid_mask = data["valid_mask"] if "valid_mask" in data else None
            label = data["label"] if "label" in data else path.parent.name
            video_id = _video_id(data, path)
            frame_count = keypoints.shape[0]
            if "frame_indices" in data:
                frame_indices = np.asarray(data["frame_indices"]).reshape(-1)
            else:
                # UR-Fall official frame numbers are 1-based.
                frame_indices = np.arange(1, frame_count + 1, dtype=np.int64)

        features, frame_valid, joints = _pose_features(
            keypoints,
            scores,
            valid_mask,
            args.visibility_threshold,
            args.feature_mode,
            args.confidence_index,
        )
        if joint_count is None:
            joint_count = joints
        elif joints != joint_count:
            raise ValueError(
                f"All files in one run must use the same joint count: "
                f"expected {joint_count}, found {joints} in {path}"
            )

        sequence_name = canonical_sequence_name(video_id)
        if sequence_name is None:
            sequence_name = canonical_sequence_name(path.stem)
        if sequence_name is not None:
            video_level_label = 1 if sequence_name.startswith("fall-") else 0
        else:
            video_level_label = normalize_label(label)

        sequence_annotations: Optional[Dict[int, int]] = None
        if video_level_label == 1:
            if sequence_name is None:
                raise RuntimeError(
                    f"Cannot parse UR-Fall sequence name from {video_id} / {path.name}"
                )
            sequence_annotations = fall_annotations.get(sequence_name)
            if sequence_annotations is None:
                raise RuntimeError(
                    f"{video_id} -> {sequence_name} has no frame annotations in "
                    f"{args.annotation_csv}"
                )

        # Match the current RF/MLP/LSTM/ST-GCN runs: window validity uses the
        # NPZ frame-level valid_mask when available. If absent, fall back to
        # the validity inferred from finite/confident joints.
        window_valid_mask = (
            np.asarray(valid_mask).reshape(-1).astype(bool)
            if valid_mask is not None
            else frame_valid
        )
        x_list, y_list, recs = make_urfall_windows(
            features=features,
            video_level_label=video_level_label,
            video_id=video_id,
            source_file=str(path),
            window_size=args.window_size,
            stride=args.stride,
            frame_indices=frame_indices,
            sequence_annotations=sequence_annotations,
            valid_mask=window_valid_mask,
            min_valid_frames=args.min_valid_frames,
            missing_mode=args.missing_mode,
        )
        if frame_count < args.window_size:
            skipped_short += 1
            continue
        if not x_list:
            skipped_no_valid_windows += 1
            continue
        windows.extend(x_list)
        labels.extend(y_list)
        groups.extend([video_id] * len(x_list))
        records.extend(recs)

    if not windows or joint_count is None:
        raise RuntimeError("No valid windows were generated")
    x = np.stack(windows).astype(np.float32)
    y = np.asarray(labels, dtype=np.int64)
    group_array = np.asarray(groups)
    record_frame = pd.DataFrame([asdict(record) for record in records])
    print(
        f"Loaded {len(files)} videos | windows={len(x)} | shape={x.shape} | "
        f"joints={joint_count} | ADL={(y == 0).sum()} | Fall={(y == 1).sum()} | "
        f"short_videos_skipped={skipped_short} | "
        f"no_valid_windows_skipped={skipped_no_valid_windows}"
    )
    print(
        "UR-Fall reference for the current 30-frame/stride-1 setup: "
        "8963 windows = 8080 ADL + 883 Fall."
    )
    if args.window_size == 30 and args.stride == 1:
        if len(x) != 8963 or int((y == 0).sum()) != 8080 or int((y == 1).sum()) != 883:
            print(
                "WARNING: window counts differ from the current reference experiment. "
                "Check frame_indices, NPZ files, valid_mask, and annotation CSV before "
                "comparing metrics directly."
            )
    return x, y, group_array, record_frame, joint_count


def fit_scaler(
    x: np.ndarray, fit_idx: np.ndarray, mode: str
) -> tuple[np.ndarray, np.ndarray]:
    fit_x = x[fit_idx]
    if mode == "zscore":
        offset = fit_x.mean(axis=(0, 1), keepdims=True)
        scale = fit_x.std(axis=(0, 1), keepdims=True)
    elif mode == "minmax":
        offset = fit_x.min(axis=(0, 1), keepdims=True)
        scale = fit_x.max(axis=(0, 1), keepdims=True) - offset
    elif mode == "none":
        offset = np.zeros((1, 1, x.shape[-1]), dtype=np.float32)
        scale = np.ones((1, 1, x.shape[-1]), dtype=np.float32)
    else:
        raise ValueError(f"Unknown normalization mode: {mode}")
    scale[scale < 1e-6] = 1.0
    return offset.astype(np.float32), scale.astype(np.float32)


def train_fold(x, y, groups, args, device, seed):
    fit_idx, val_idx = urfall_inner_split(y, groups, seed)
    offset, scale = fit_scaler(x, fit_idx, args.normalization)
    fit_x = ((x[fit_idx] - offset) / scale).astype(np.float32)
    val_x = ((x[val_idx] - offset) / scale).astype(np.float32)
    fit_loader = make_loader(fit_x, y[fit_idx], args.batch_size, True)
    val_loader = make_loader(val_x, y[val_idx], args.batch_size, False)

    seed_everything(seed)
    model = TemporalTransformer(
        input_dim=x.shape[-1],
        window_size=x.shape[1],
        d_model=args.d_model,
        nhead=args.heads,
        layers=args.layers,
        feedforward_dim=args.feedforward_dim,
        dropout=args.dropout,
    ).to(device)
    counts = np.bincount(y[fit_idx], minlength=2)
    if np.any(counts == 0):
        raise ValueError("Inner training split must contain both classes")
    weights = torch.tensor(len(fit_idx) / (2 * counts), dtype=torch.float32, device=device)
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=max(2, args.patience // 3)
    )
    best_f1, best_loss, best_epoch, best_state = -1.0, float("inf"), 0, None
    history = []
    started = time.perf_counter()

    for epoch in range(1, args.epochs + 1):
        model.train()
        loss_sum = 0.0
        for xb, yb in fit_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), args.gradient_clip)
            optimizer.step()
            loss_sum += loss.item() * len(yb)
        train_loss = loss_sum / len(fit_idx)
        val_prob = probabilities(model, val_loader, device)
        val_pred = (val_prob >= args.threshold).astype(np.int64)
        val_f1 = compute_metrics(y[val_idx], val_pred, val_prob)["f1"]
        scheduler.step(val_f1)
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_f1": val_f1,
                "learning_rate": optimizer.param_groups[0]["lr"],
            }
        )
        if val_f1 > best_f1 or (val_f1 == best_f1 and train_loss < best_loss):
            best_f1, best_loss, best_epoch = val_f1, train_loss, epoch
            best_state = {
                key: value.detach().cpu().clone() for key, value in model.state_dict().items()
            }
        if epoch - best_epoch >= args.patience:
            break

    if best_state is None:
        raise RuntimeError("Training did not produce a checkpoint")
    model.load_state_dict(best_state)
    return (
        model,
        offset,
        scale,
        best_epoch,
        best_f1,
        history,
        (fit_idx, val_idx),
        time.perf_counter() - started,
    )


def _metrics_frame(df: pd.DataFrame) -> dict[str, float]:
    return compute_metrics(
        df["label"].to_numpy(),
        df["y_pred"].to_numpy(),
        df["fall_probability"].to_numpy(),
    )


def _json_args(args) -> dict:
    return {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description="Video-grouped temporal Transformer fall detection experiment"
    )
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument(
        "--annotation-csv",
        type=Path,
        default=Path("data/urfall-cam0-falls.csv"),
        help="UR-Fall official per-frame posture annotation CSV",
    )
    parser.add_argument("--output-root", type=Path, default=Path("results"))
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--pose-estimator", default="unspecified")
    parser.add_argument("--dataset-name", default="unspecified")
    parser.add_argument("--window-size", type=int, default=30)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--min-valid-frames", type=int, default=1)
    parser.add_argument("--missing-mode", choices=("zero", "interp"), default="interp")
    parser.add_argument("--visibility-threshold", type=float, default=0.3)
    parser.add_argument("--confidence-index", type=int, default=None)
    parser.add_argument("--feature-mode", choices=("xy", "xyc"), default="xy")
    parser.add_argument(
        "--normalization", choices=("minmax", "zscore", "none"), default="none"
    )
    parser.add_argument("--d-model", type=int, default=512)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--feedforward-dim", type=int, default=2048)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--gradient-clip", type=float, default=1.0)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42])
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def main():
    args = parse_args()
    positive = (
        args.window_size,
        args.stride,
        args.min_valid_frames,
        args.d_model,
        args.heads,
        args.layers,
        args.feedforward_dim,
        args.epochs,
        args.patience,
        args.batch_size,
    )
    if min(positive) < 1 or args.folds < 2:
        raise ValueError("Positive sizes are required and folds must be at least 2")
    if not args.seeds:
        raise ValueError("At least one training seed is required")
    if args.d_model % args.heads:
        raise ValueError("d-model must be divisible by heads")

    device = choose_device(args.device)
    seed_everything(args.split_seed)
    x, y, groups, records, joint_count = load_pose_windows(args)
    # A fall video can now contain both ADL and Fall windows, so stratify folds
    # by the original UR-Fall video type while training/evaluating on window y.
    split_y = video_type_labels(groups)
    unique_group_types = (
        pd.DataFrame({"group": groups, "video_type": split_y})
        .drop_duplicates("group")
    )
    class_groups = np.bincount(
        unique_group_types["video_type"].to_numpy(dtype=np.int64), minlength=2
    )
    if class_groups.min() < args.folds:
        raise ValueError(
            f"Need at least {args.folds} videos per class; found {class_groups.tolist()}"
        )
    splitter = StratifiedGroupKFold(
        n_splits=args.folds, shuffle=True, random_state=args.split_seed
    )
    split_indices = list(splitter.split(x, split_y, groups))
    run_name = args.run_name or f"transformer_{args.data_root.resolve().name.lower()}"
    output = args.output_root / run_name
    output.mkdir(parents=True, exist_ok=True)

    split_rows = []
    for fold, (train_idx, test_idx) in enumerate(split_indices, 1):
        if set(groups[train_idx]) & set(groups[test_idx]):
            raise RuntimeError("Training and test videos overlap")
        for role, indices in (("outer_train", train_idx), ("outer_test", test_idx)):
            for video_id in np.unique(groups[indices]):
                split_rows.append({"fold": fold, "video_id": video_id, "role": role})
    pd.DataFrame(split_rows).to_csv(output / "splits.csv", index=False)

    all_predictions = []
    all_video_predictions = []
    all_fold_rows = []
    seed_rows = []
    parameter_count = None

    for seed in args.seeds:
        seed_dir = output / f"seed_{seed}"
        model_dir = seed_dir / "models"
        history_dir = seed_dir / "histories"
        model_dir.mkdir(parents=True, exist_ok=True)
        history_dir.mkdir(parents=True, exist_ok=True)
        seed_predictions = []
        for fold, (train_idx, test_idx) in enumerate(split_indices, 1):
            model, offset, scale, best_epoch, val_f1, history, _, seconds = train_fold(
                x[train_idx],
                y[train_idx],
                groups[train_idx],
                args,
                device,
                seed + fold,
            )
            parameter_count = sum(parameter.numel() for parameter in model.parameters())
            test_x = ((x[test_idx] - offset) / scale).astype(np.float32)
            test_loader = make_loader(test_x, y[test_idx], args.batch_size, False)
            prob = probabilities(model, test_loader, device)
            pred = (prob >= args.threshold).astype(np.int64)
            metrics = compute_metrics(y[test_idx], pred, prob)
            all_fold_rows.append(
                {
                    "seed": seed,
                    "fold": fold,
                    "best_epoch": best_epoch,
                    "val_f1": val_f1,
                    "train_seconds": seconds,
                    **metrics,
                }
            )
            frame = records.iloc[test_idx].copy()
            frame["seed"] = seed
            frame["fold"] = fold
            frame["fall_probability"] = prob
            frame["y_pred"] = pred
            seed_predictions.append(frame)
            pd.DataFrame(history).to_csv(
                history_dir / f"fold_{fold}_history.csv", index=False
            )
            torch.save(
                {
                    "model_type": "temporal_transformer",
                    "state_dict": model.state_dict(),
                    "feature_offset": offset,
                    "feature_scale": scale,
                    "normalization": args.normalization,
                    "input_dim": x.shape[-1],
                    "window_size": x.shape[1],
                    "joint_count": joint_count,
                    "parameter_count": parameter_count,
                    "best_epoch": best_epoch,
                    "args": _json_args(args),
                },
                model_dir / f"fold_{fold}.pt",
            )
            print(
                f"Seed {seed} fold {fold}: F1={metrics['f1']:.4f}, "
                f"recall={metrics['recall']:.4f}, best_epoch={best_epoch}"
            )

        seed_frame = pd.concat(seed_predictions, ignore_index=True)
        seed_frame.to_csv(seed_dir / "predictions.csv", index=False)
        all_predictions.append(seed_frame)
        video_frame = seed_frame.groupby("video_id", as_index=False).agg(
            seed=("seed", "first"),
            fold=("fold", "first"),
            label=("label", "first"),
            fall_probability=("fall_probability", "mean"),
            windows=("label", "size"),
        )
        video_frame["y_pred"] = (
            video_frame["fall_probability"] >= args.threshold
        ).astype(int)
        video_frame.to_csv(seed_dir / "video_predictions.csv", index=False)
        all_video_predictions.append(video_frame)
        window_metrics = _metrics_frame(seed_frame)
        video_metrics = _metrics_frame(video_frame)
        seed_rows.append(
            {
                "seed": seed,
                **{f"window_{key}": value for key, value in window_metrics.items()},
                **{f"video_{key}": value for key, value in video_metrics.items()},
            }
        )
        cm = confusion_matrix(seed_frame.label, seed_frame.y_pred, labels=[0, 1])
        pd.DataFrame(
            cm,
            index=["true_ADL", "true_Fall"],
            columns=["pred_ADL", "pred_Fall"],
        ).to_csv(seed_dir / "confusion_matrix.csv")
        report = classification_report(
            seed_frame.label,
            seed_frame.y_pred,
            labels=[0, 1],
            target_names=["ADL", "Fall"],
            digits=4,
            zero_division=0,
        )
        (seed_dir / "metrics.txt").write_text(
            "WINDOW METRICS\n"
            + "\n".join(f"{key}: {value}" for key, value in window_metrics.items())
            + "\n\nVIDEO METRICS\n"
            + "\n".join(f"{key}: {value}" for key, value in video_metrics.items())
            + "\n\nClassification report:\n"
            + report,
            encoding="utf-8",
        )

    prediction_frame = pd.concat(all_predictions, ignore_index=True)
    prediction_frame.to_csv(output / "predictions.csv", index=False)
    pd.concat(all_video_predictions, ignore_index=True).to_csv(
        output / "video_predictions.csv", index=False
    )
    fold_frame = pd.DataFrame(all_fold_rows)
    fold_frame.to_csv(output / "fold_metrics.csv", index=False)
    seed_frame = pd.DataFrame(seed_rows)
    seed_frame.to_csv(output / "seed_metrics.csv", index=False)
    metric_only = seed_frame.drop(columns=["seed"])

    summary = {
        "pose_estimator": args.pose_estimator,
        "dataset": args.dataset_name,
        "joint_count": joint_count,
        "input_dim": int(x.shape[-1]),
        "window_size": int(x.shape[1]),
        "parameter_count": parameter_count,
        "device": str(device),
        "seeds": args.seeds,
        "split_seed": args.split_seed,
        "seed_metric_mean": metric_only.mean(numeric_only=True).to_dict(),
        "seed_metric_std": metric_only.std(numeric_only=True, ddof=1).fillna(0).to_dict(),
    }
    (output / "metrics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    config = _json_args(args)
    config["output_dir"] = str(output)
    (output / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    print(f"Results: {output}")


if __name__ == "__main__":
    main()
