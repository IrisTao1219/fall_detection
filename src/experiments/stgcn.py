#!/usr/bin/env python3
"""Ten-block ST-GCN fall-detection experiment using BlazePose windows.

Run from the fall_detection directory with
``uv run python src/experiment_stgcn.py --data-root data/keypoints_normalized``.
Device selection, data loading, window creation, metrics, and PyTorch loaders
are implemented directly in this file.

The network uses four 64-channel, three 128-channel, and three 256-channel
spatiotemporal graph blocks. Each block has a temporal kernel of size 9,
a residual path, and dropout. Softmax is applied when obtaining probabilities;
training uses logits with CrossEntropyLoss.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import time
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
        choose_device,
        compute_metrics,
        fmt_metric,
        inner_group_split_by_video_type,
        load_sequence_windows,
        make_loader,
        probabilities,
        seed_everything,
        standardize,
        video_type_labels,
    )
except ImportError:
    import common as experiment_common
    from common import (
        choose_device,
        compute_metrics,
        fmt_metric,
        inner_group_split_by_video_type,
        load_sequence_windows,
        make_loader,
        probabilities,
        seed_everything,
        standardize,
        video_type_labels,
    )


# MediaPipe Pose's 33-landmark skeleton. Undirected edges and self loops are
# added in make_adjacency. The temporal convolution links each joint over time.
POSE_EDGES = (
    (0, 1), (1, 2), (2, 3), (3, 7), (0, 4), (4, 5), (5, 6), (6, 8),
    (9, 10), (11, 12), (11, 13), (13, 15), (15, 17), (15, 19),
    (15, 21), (17, 19), (12, 14), (14, 16), (16, 18), (16, 20),
    (16, 22), (18, 20), (11, 23), (12, 24), (23, 24), (23, 25),
    (24, 26), (25, 27), (26, 28), (27, 29), (28, 30), (29, 31),
    (30, 32), (27, 31), (28, 32),
)
STGCN_CHANNELS = (64,) * 4 + (128,) * 3 + (256,) * 3
TEMPORAL_KERNEL_SIZE = 9


def choose_device(requested: str = "auto") -> torch.device:
    """
    在当前文件内选择运行设备，不依赖 external helper module。

    requested:
        auto  -> 优先 CUDA，其次 MPS，最后 CPU
        cuda  -> 强制使用 CUDA；不可用时报错
        mps   -> 强制使用 Apple Silicon MPS；不可用时报错
        cpu   -> 强制使用 CPU
    """
    requested = str(requested).strip().lower()

    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")

        mps_backend = getattr(torch.backends, "mps", None)
        if mps_backend is not None and mps_backend.is_available():
            return torch.device("mps")

        return torch.device("cpu")

    if requested == "cuda" or requested.startswith("cuda:"):
        if not torch.cuda.is_available():
            raise RuntimeError(
                f"--device {requested} specified, but CUDA is not available."
            )

        if requested == "cuda":
            index = 0
        else:
            index = int(requested.split(":", 1)[1])

        count = torch.cuda.device_count()

        if index < 0 or index >= count:
            raise ValueError(
                f"CUDA device {index} does not exist. "
                f"PyTorch sees {count} CUDA devices."
            )

        # 关键：设置当前默认 CUDA 设备
        # 即使第三方代码内部调用 .cuda()，也会使用这张卡
        torch.cuda.set_device(index)

        return torch.device(f"cuda:{index}")

    if requested == "mps":
        mps_backend = getattr(torch.backends, "mps", None)
        if mps_backend is None or not mps_backend.is_available():
            raise RuntimeError(
                "--device mps 已指定，但当前环境没有可用的 Apple MPS。"
            )
        return torch.device("mps")

    if requested == "cpu":
        return torch.device("cpu")

    raise ValueError(
        f"未知 device={requested!r}，请选择 auto / cpu / cuda / mps"
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


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def normalize_label(value) -> int:
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


def interpolate_1d(values: np.ndarray) -> np.ndarray:
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
    if keypoints.ndim != 3 or keypoints.shape[1] != 33 or keypoints.shape[2] < 2:
        raise ValueError(f"Expected keypoints [T,33,C>=2], got shape {keypoints.shape}")

    xy = keypoints[..., :2].astype(np.float32, copy=True)
    frame_count = xy.shape[0]
    joint_valid = np.isfinite(xy).all(axis=-1)

    if keypoints.shape[2] >= 4:
        visibility = keypoints[..., 3]
        joint_valid &= np.isfinite(visibility) & (visibility >= visibility_threshold)

    if valid_mask is not None:
        valid_mask = np.asarray(valid_mask).reshape(-1).astype(bool)
        if len(valid_mask) != frame_count:
            raise ValueError(
                f"valid_mask length {len(valid_mask)} != keypoints frames {frame_count}"
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


def build_frame_features(
    xy: np.ndarray,
    feature_mode: str,
    joint_indices: Optional[Sequence[int]],
) -> np.ndarray:
    if feature_mode == "xy66":
        return xy.reshape(xy.shape[0], -1).astype(np.float32)
    if feature_mode == "joints16":
        if joint_indices is None or len(joint_indices) != 8:
            raise ValueError("joints16 requires exactly 8 joint indices")
        idx = np.asarray(joint_indices, dtype=np.int64)
        if np.any(idx < 0) or np.any(idx >= 33):
            raise ValueError("All joint indices must be in [0, 32]")
        return xy[:, idx, :].reshape(xy.shape[0], 16).astype(np.float32)
    raise ValueError(f"Unknown feature_mode: {feature_mode}")


@dataclass
class WindowRecord:
    video_id: str
    source_file: str
    window_start: int
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
    frame_indices: np.ndarray,
    valid_mask: Optional[np.ndarray] = None,
    min_valid_frames: int = 1,
    missing_mode: str = "interp",
) -> Tuple[List[np.ndarray], List[int], List[WindowRecord]]:
    frame_count = features.shape[0]
    if frame_count < window_size:
        return [], [], []
    frame_indices = np.asarray(frame_indices).reshape(-1)
    if len(frame_indices) != frame_count:
        raise ValueError(
            f"frame_indices length {len(frame_indices)} != feature frames {frame_count}"
        )

    label_arr = np.asarray(label_data)
    is_frame_level = label_arr.ndim > 0 and label_arr.size == frame_count
    if is_frame_level:
        frame_labels = normalize_label_array(label_arr.reshape(-1))
        scalar_label = None
    else:
        scalar_label = normalize_label(label_data)
        frame_labels = None

    x_list: List[np.ndarray] = []
    y_list: List[int] = []
    records: List[WindowRecord] = []

    for start_frame in range(0, frame_count - window_size + 1, stride):
        end_exclusive = start_frame + window_size
        if valid_mask is not None:
            valid_count = int(np.count_nonzero(valid_mask[start_frame:end_exclusive]))
            if valid_count < min_valid_frames:
                continue

        window = features[start_frame:end_exclusive].copy()
        if missing_mode == "interp":
            for feature_idx in range(window.shape[1]):
                window[:, feature_idx] = interpolate_1d(window[:, feature_idx])
        elif missing_mode == "zero":
            window = np.nan_to_num(window, nan=0.0, posinf=0.0, neginf=0.0)
        else:
            raise ValueError(f"Unknown missing_mode: {missing_mode}")

        if frame_labels is not None:
            values, counts = np.unique(
                frame_labels[start_frame:end_exclusive], return_counts=True
            )
            candidates = values[counts == counts.max()]
            window_label = int(candidates.max())
        else:
            window_label = int(scalar_label)

        x_list.append(window.astype(np.float32))
        y_list.append(window_label)
        records.append(
            WindowRecord(
                video_id=video_id,
                source_file=source_file,
                window_start=start_frame,
                start_frame=int(frame_indices[start_frame]),
                end_frame=int(frame_indices[end_exclusive - 1]),
                label=window_label,
            )
        )
    return x_list, y_list, records


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
    x, y, groups, records = experiment_common.load_sequence_windows(
        data_root=data_root,
        window_size=window_size,
        stride=stride,
        visibility_threshold=visibility_threshold,
        missing_mode=missing_mode,
        feature_mode=feature_mode,
        joint_indices=joint_indices,
        min_valid_frames=min_valid_frames,
    )
    return x.reshape(x.shape[0], x.shape[1], 33, 2), y, groups, records



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


def safe_roc_auc(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(roc_auc_score(y_true, y_prob))


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray) -> Dict[str, float]:
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "roc_auc": safe_roc_auc(y_true, y_prob),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }

# UR-Fall 官方逐帧姿态标注。路径按当前项目根目录写死。
FALL_ANNOTATION_CSV = Path("data/urfall-cam0-falls.csv")


def canonical_sequence_name(value) -> str | None:
    """把 video/group 名统一成 fall-01 / adl-01 形式。"""
    value = str(value).strip().lower()
    match = re.search(r"(fall|adl)-?(\d+)", value)
    if match is None:
        return None
    return f"{match.group(1)}-{int(match.group(2)):02d}"


def load_fall_frame_labels(csv_path: Path):
    """
    读取 UR-Fall 官方 fall CSV 的前三列：
        sequence name, frame number, posture label

    posture label:
        -1 = not lying
         0 = temporary falling pose
         1 = lying on the ground
    """
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"找不到 UR-Fall 标注 CSV：{csv_path}")

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
                # 兼容可能存在的表头。
                continue

            if sequence_name is None or not sequence_name.startswith("fall-"):
                continue

            if posture_label not in (-1, 0, 1):
                raise ValueError(
                    f"{csv_path} 第 {line_number} 行出现未知姿态标签：{posture_label}"
                )

            annotations.setdefault(sequence_name, {})[frame_number] = posture_label
            valid_rows += 1

    if valid_rows == 0:
        raise RuntimeError(f"{csv_path} 没有读取到有效标注，请检查 CSV 格式")

    print(
        f"Loaded UR-Fall annotations: "
        f"{len(annotations)} fall sequences, {valid_rows} frames"
    )
    return annotations


def get_window_label_from_urfall(frame_labels) -> int | None:
    """
    与 RF 版本保持一致：
      - 忽略 posture label=0
      - -1 -> normal/ADL (0)
      -  1 -> fall (1)
      - 对剩余帧多数投票
      - 全为 0 或 -1/1 平票时丢弃窗口
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


def apply_urfall_window_labels(
    x: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    records: pd.DataFrame,
    window_size: int,
    annotations,
):
    """
    将共享 loader 产生的视频级标签改成 UR-Fall 窗口级标签。

    ADL:
        所有窗口 -> 0

    Fall:
        当前窗口帧号 -> CSV posture labels
        -> 忽略 0 -> -1/1 多数投票
        -> 0(normal) / 1(fall)

    无有效标注或平票的窗口直接从数据集中移除。
    """
    if not isinstance(records, pd.DataFrame):
        records = pd.DataFrame(records)

    if len(x) != len(y) or len(x) != len(groups) or len(x) != len(records):
        raise RuntimeError(
            "x/y/groups/records 数量不一致："
            f"{len(x)}/{len(y)}/{len(groups)}/{len(records)}"
        )

    keep_indices = []
    new_labels = []
    skipped_transition_or_tie = 0
    skipped_no_annotation = 0
    frame_indices_by_source = {}

    for i, group in enumerate(groups):
        sequence_name = canonical_sequence_name(group)
        if sequence_name is None:
            raise RuntimeError(f"无法从 group/video_id 解析 UR-Fall 序列名：{group}")

        if sequence_name.startswith("adl-"):
            keep_indices.append(i)
            new_labels.append(0)
            continue

        sequence_annotations = annotations.get(sequence_name)
        if sequence_annotations is None:
            raise RuntimeError(
                f"CSV 中找不到 {sequence_name} 的逐帧标注；"
                f"当前 group={group}"
            )

        row = records.iloc[i]
        source_file = str(row["source_file"])
        if source_file not in frame_indices_by_source:
            with np.load(source_file, allow_pickle=True) as data:
                frame_indices_by_source[source_file] = (
                    np.asarray(data["frame_indices"]).reshape(-1)
                    if "frame_indices" in data
                    else np.arange(1, len(data["keypoints"]) + 1, dtype=np.int64)
                )
        start = int(row["window_start"])
        window_frame_indices = frame_indices_by_source[source_file][
            start : start + window_size
        ]
        if len(window_frame_indices) != window_size:
            raise RuntimeError(f"{source_file}: 窗口帧数不足，起始位置 {start}")

        frame_labels = [
            sequence_annotations[int(frame_number)]
            for frame_number in window_frame_indices
            if int(frame_number) in sequence_annotations
        ]

        if not frame_labels:
            skipped_no_annotation += 1
            continue

        window_label = get_window_label_from_urfall(frame_labels)
        if window_label is None:
            skipped_transition_or_tie += 1
            continue

        keep_indices.append(i)
        new_labels.append(window_label)

    keep_indices = np.asarray(keep_indices, dtype=np.int64)
    new_y = np.asarray(new_labels, dtype=np.int64)

    x = x[keep_indices]
    groups = groups[keep_indices]
    records = records.iloc[keep_indices].copy().reset_index(drop=True)

    # 记录最终窗口标签，便于 predictions.csv 追踪。
    records["window_label"] = new_y
    records["window_label_name"] = np.where(new_y == 1, "fall", "adl")

    print(
        "Window relabeling: "
        f"kept={len(new_y)}, "
        f"ADL={int(np.sum(new_y == 0))}, "
        f"Fall={int(np.sum(new_y == 1))}, "
        f"skipped_transition_or_tie={skipped_transition_or_tie}, "
        f"skipped_no_annotation={skipped_no_annotation}"
    )

    if len(new_y) == 0:
        raise RuntimeError("窗口级重新标注后没有剩余样本")

    if len(np.unique(new_y)) < 2:
        raise RuntimeError(
            "窗口级重新标注后只剩一个类别，请检查 CSV 与帧号是否对齐"
        )

    return x, new_y, groups, records


def video_type_labels(groups: np.ndarray) -> np.ndarray:
    """
    用原始视频类型做 split stratification：
        adl-*  -> 0
        fall-* -> 1

    注意：这只用于保证各 fold 都包含两类视频；
    模型真正训练/评估仍使用窗口级 y。
    """
    labels = []
    for group in groups:
        sequence_name = canonical_sequence_name(group)
        if sequence_name is None:
            raise RuntimeError(f"无法解析视频类型：{group}")
        labels.append(1 if sequence_name.startswith("fall-") else 0)
    return np.asarray(labels, dtype=np.int64)


def inner_group_split_window_labels(y, groups, seed):
    """
    内层 train/validation 也按视频分组，并按原始视频类型 stratify。
    避免一个 fall 视频同时含 normal/fall 窗口后破坏原 inner_split 假设。
    """
    return experiment_common.inner_group_split_by_video_type(y, groups, seed)


def make_adjacency() -> torch.Tensor:
    adjacency = np.eye(33, dtype=np.float32)
    for a, b in POSE_EDGES:
        adjacency[a, b] = adjacency[b, a] = 1.0
    degree = adjacency.sum(axis=1)
    adjacency /= np.sqrt(degree[:, None] * degree[None, :])
    return torch.from_numpy(adjacency)


class STGCNBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int, dropout: float):
        super().__init__()
        self.spatial = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.temporal = nn.Conv2d(
            out_channels, out_channels, kernel_size=(TEMPORAL_KERNEL_SIZE, 1),
            stride=(stride, 1), padding=(TEMPORAL_KERNEL_SIZE // 2, 0),
        )
        self.norm = nn.BatchNorm2d(out_channels)
        self.dropout = nn.Dropout2d(dropout)
        self.residual = (
            nn.Identity() if in_channels == out_channels and stride == 1
            else nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=(stride, 1)),
                nn.BatchNorm2d(out_channels),
            )
        )
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
        spatial = torch.einsum("bctv,vw->bctw", x, adjacency)
        features = self.dropout(self.norm(self.temporal(self.spatial(spatial))))
        return self.activation(features + self.residual(x))


class STGCN(nn.Module):
    def __init__(self, dropout: float = 0.3):
        super().__init__()
        self.register_buffer("adjacency", make_adjacency())
        blocks = []
        in_channels = 2
        for layer, out_channels in enumerate(STGCN_CHANNELS):
            # Reduce temporal resolution when entering a wider channel stage.
            stride = 2 if layer in (4, 7) else 1
            blocks.append(STGCNBlock(in_channels, out_channels, stride, dropout))
            in_channels = out_channels
        self.blocks = nn.ModuleList(blocks)
        self.classifier = nn.Linear(STGCN_CHANNELS[-1], 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Loader supplies [B,T,66]; graph layers use [B,C,T,V].
        x = x.reshape(x.shape[0], x.shape[1], 33, 2).permute(0, 3, 1, 2)
        for block in self.blocks:
            x = block(x, self.adjacency)
        # Keep logits for CrossEntropyLoss. Prediction uses Softmax in probabilities().
        return self.classifier(x.mean(dim=(2, 3)))


def standardize(x: np.ndarray, fit_idx: np.ndarray):
    # One mean/std for each coordinate of each joint, using fit videos only.
    mean = x[fit_idx].mean(axis=(0, 1), keepdims=True)
    std = x[fit_idx].std(axis=(0, 1), keepdims=True)
    std[std < 1e-6] = 1.0
    return mean.astype(np.float32), std.astype(np.float32)


def train_fold(x, y, groups, args, device, seed):
    fit_idx, val_idx = inner_group_split_window_labels(y, groups, seed)
    mean, std = standardize(x, fit_idx)
    fit_loader = make_loader(((x[fit_idx] - mean) / std).astype(np.float32), y[fit_idx], args.batch_size, True)
    val_loader = make_loader(((x[val_idx] - mean) / std).astype(np.float32), y[val_idx], args.batch_size, False)
    seed_everything(seed)
    model = STGCN(args.dropout).to(device)
    counts = np.bincount(y[fit_idx], minlength=2)
    weights = torch.tensor(len(fit_idx) / (2 * counts), dtype=torch.float32, device=device)
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
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
            optimizer.step()
            loss_sum += loss.item() * len(yb)
        val_prob = probabilities(model, val_loader, device)
        val_f1 = compute_metrics(y[val_idx], (val_prob >= 0.5).astype(int), val_prob)["f1"]
        train_loss = loss_sum / len(fit_idx)
        history.append({"epoch": epoch, "train_loss": train_loss, "val_f1": val_f1})
        if val_f1 > best_f1 or (val_f1 == best_f1 and train_loss < best_loss):
            best_f1, best_loss, best_epoch = val_f1, train_loss, epoch
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        if epoch - best_epoch >= args.patience:
            break
    model.load_state_dict(best_state)
    return model, mean, std, best_epoch, best_f1, history, (fit_idx, val_idx), time.perf_counter() - started


def fmt_metric(v):
    if isinstance(v, (float, np.floating)) and np.isnan(v):
        return "nan"
    return f"{v:.6f}" if isinstance(v, (float, np.floating)) else str(v)


seed_everything = experiment_common.seed_everything
choose_device = experiment_common.choose_device
video_type_labels = experiment_common.video_type_labels
make_loader = experiment_common.make_loader
probabilities = experiment_common.probabilities
compute_metrics = experiment_common.compute_metrics
fmt_metric = experiment_common.fmt_metric


def parse_args():
    parser = argparse.ArgumentParser(description="Video-grouped ST-GCN fall detection experiment")
    parser.add_argument("--data-root", type=Path, default=Path("data/keypoints"))
    parser.add_argument("--output-root", type=Path, default=Path("results"))
    parser.add_argument("--window-size", type=int, default=30)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--missing-mode", choices=("zero", "interp"), default="interp")
    parser.add_argument("--visibility-threshold", type=float, default=0.3)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto", help="auto / cpu / cuda / mps")
    return parser.parse_args()


def dataset_run_name(data_root: Path) -> str:
    """Keep raw/normalized output folders consistent with the sequence baseline experiment."""
    name = data_root.resolve().name.lower()
    return "stgcn_normalized" if "normalized" in name else "stgcn"


def main():
    args = parse_args()
    if min(args.window_size, args.stride, args.patience, args.epochs, args.batch_size) < 1 or args.folds < 2:
        raise ValueError("Window size, stride, patience, epochs, batch size must be positive; folds >= 2")
    if not 0 <= args.dropout < 1:
        raise ValueError("dropout must be in [0, 1)")

    # Match sequence baseline output layout: one dataset root -> one experiment directory.
    run_name = dataset_run_name(args.data_root)
    args.experiment_output_dir = args.output_root / run_name
    output = args.experiment_output_dir
    output.mkdir(parents=True, exist_ok=True)

    seed_everything(args.seed)
    device = choose_device(args.device)

    print(f"Device: {device}")
    print(f"Data root: {args.data_root}")
    print(f"Output dir: {output}")
    print(
        f"Window={args.window_size}, stride={args.stride}, "
        f"feature_mode=xy66, missing_mode={args.missing_mode}"
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

    # 一个 fall 视频现在可以同时含 normal/fall 窗口，所以不能再用窗口 y 的 mode
    # 判断“这个视频属于哪一类”。这里按视频名本身的 fall/adl 类型统计。
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

    models_dir = output / "models"
    histories_dir = output / "histories"
    models_dir.mkdir(parents=True, exist_ok=True)
    histories_dir.mkdir(parents=True, exist_ok=True)

    # Save config in the same place/style as sequence baseline.
    config = vars(args).copy()
    for key, value in list(config.items()):
        if isinstance(value, Path):
            config[key] = str(value)
    config.update(
        {
            "model_type": "stgcn",
            "feature_mode": "xy66",
            "channels": list(STGCN_CHANNELS),
            "temporal_kernel_size": TEMPORAL_KERNEL_SIZE,
            "pose_extractor": "BlazePose",
            "annotation_csv": str(FALL_ANNOTATION_CSV),
            "label_level": "window",
            "urfall_window_label_rule": "ignore_0_then_majority_vote_-1_vs_1",
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

    all_true = []
    all_pred = []
    all_prob = []
    all_test_indices = []
    fold_rows = []

    metrics_lines = []
    metrics_lines.append("=" * 80)
    metrics_lines.append("MODEL: ST-GCN")
    metrics_lines.append("=" * 80)
    metrics_lines.append(f"data_root: {args.data_root}")
    metrics_lines.append("feature_mode: xy66")
    metrics_lines.append(f"input_dim: {x.shape[-1]}")
    metrics_lines.append(f"window_size: {args.window_size}")
    metrics_lines.append(f"stride: {args.stride}")
    metrics_lines.append(f"missing_mode: {args.missing_mode}")
    metrics_lines.append(f"visibility_threshold: {args.visibility_threshold}")
    metrics_lines.append(f"channels: {STGCN_CHANNELS}")
    metrics_lines.append(f"temporal_kernel_size: {TEMPORAL_KERNEL_SIZE}")
    metrics_lines.append(f"dropout: {args.dropout}")
    metrics_lines.append(f"epochs: {args.epochs}")
    metrics_lines.append(f"patience: {args.patience}")
    metrics_lines.append(f"batch_size: {args.batch_size}")
    metrics_lines.append(f"learning_rate: {args.learning_rate}")
    metrics_lines.append(f"weight_decay: {args.weight_decay}")
    metrics_lines.append(f"folds: {args.folds}")
    metrics_lines.append(f"seed: {args.seed}")
    metrics_lines.append(f"device: {device}")
    metrics_lines.append(f"annotation_csv: {FALL_ANNOTATION_CSV}")
    metrics_lines.append("label_level: window")
    metrics_lines.append("window_label_rule: ignore posture 0; majority vote -1(normal) vs 1(fall)")
    metrics_lines.append("")

    for fold, (train_idx, test_idx) in enumerate(splitter.split(x, split_y, groups), 1):
        train_groups = set(groups[train_idx])
        test_groups = set(groups[test_idx])
        overlap = train_groups & test_groups
        if overlap:
            raise RuntimeError(f"Video leakage detected: {sorted(overlap)[:5]}")

        print("\n" + "=" * 72)
        print(
            f"ST-GCN | Fold {fold}/{args.folds} | "
            f"train_windows={len(train_idx)} | test_windows={len(test_idx)}"
        )

        model, mean, std, best_epoch, val_f1, history, _, seconds = train_fold(
            x[train_idx],
            y[train_idx],
            groups[train_idx],
            args,
            device,
            args.seed + fold,
        )

        test_x = ((x[test_idx] - mean) / std).astype(np.float32)
        prob = probabilities(
            model,
            make_loader(test_x, y[test_idx], args.batch_size, False),
            device,
        )
        pred = (prob >= 0.5).astype(np.int64)
        fold_metrics = compute_metrics(y[test_idx], pred, prob)

        # Keep the same core fold_metrics.csv columns as sequence baseline.
        row = dict(fold_metrics)
        row["fold"] = fold
        row["train_videos"] = len(train_groups)
        row["test_videos"] = len(test_groups)
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
            f"FN={fold_metrics['fn']} TP={fold_metrics['tp']}"
        )
        metrics_lines.append(f"[Fold {fold}/{args.folds}]")
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

        checkpoint = {
            "model_type": "stgcn",
            "input_dim": x.shape[-1],
            "dropout": args.dropout,
            "channels": STGCN_CHANNELS,
            "temporal_kernel_size": TEMPORAL_KERNEL_SIZE,
            "state_dict": model.state_dict(),
            "feature_mean": mean,
            "feature_std": std,
            "best_epoch": best_epoch,
            "val_f1": val_f1,
            "pose_extractor": "BlazePose",
            "args": vars(args).copy(),
        }
        for key, value in list(checkpoint["args"].items()):
            if isinstance(value, Path):
                checkpoint["args"][key] = str(value)
        torch.save(checkpoint, models_dir / f"fold_{fold}.pt")
        # Match sequence baseline history CSV structure: epoch + train_loss.
        pd.DataFrame(history)[["epoch", "train_loss"]].to_csv(
            histories_dir / f"fold_{fold}_history.csv", index=False
        )

        all_true.extend(y[test_idx].tolist())
        all_pred.extend(pred.tolist())
        all_prob.extend(prob.tolist())
        all_test_indices.extend(test_idx.tolist())

    # Overall out-of-fold metrics: same structure as sequence baseline.
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

    fold_df = pd.DataFrame(fold_rows)
    metrics_lines.append("")
    metrics_lines.append("Fold mean ± std:")
    for key in ["accuracy", "precision", "recall", "f1", "roc_auc"]:
        mean_v = fold_df[key].mean()
        std_v = fold_df[key].std(ddof=1)
        metrics_lines.append(f"{key}: {mean_v:.6f} ± {std_v:.6f}")

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

    print("\n" + "=" * 72)
    print("ST-GCN OVERALL")
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
