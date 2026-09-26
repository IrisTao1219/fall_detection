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
COCO17_EDGES = (
    (0, 1), (0, 2), (1, 3), (2, 4),
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
    (5, 11), (6, 12), (11, 12),
    (11, 13), (13, 15), (12, 14), (14, 16),
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


def skeleton_edges(joint_count: int):
    if joint_count == 33:
        return POSE_EDGES, "BlazePose 33-joint physical graph"
    if joint_count == 17:
        return COCO17_EDGES, "COCO/VitPose 17-joint physical graph"
    raise ValueError(f"ST-GCN only supports 33-joint BlazePose or 17-joint COCO/VitPose, got {joint_count}")


def make_adjacency(joint_count: int) -> torch.Tensor:
    edges, _ = skeleton_edges(joint_count)
    adjacency = np.eye(joint_count, dtype=np.float32)
    for a, b in edges:
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
    def __init__(self, joint_count: int, dropout: float = 0.3):
        super().__init__()
        self.joint_count = joint_count
        self.register_buffer("adjacency", make_adjacency(joint_count))
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
        # Loader supplies [B,T,V*2]; graph layers use [B,C,T,V].
        x = x.reshape(x.shape[0], x.shape[1], self.joint_count, 2).permute(0, 3, 1, 2)
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


def train_fold(x, y, groups, args, device, seed, joint_count: int):
    fit_idx, val_idx = inner_group_split_window_labels(y, groups, seed)
    mean, std = standardize(x, fit_idx)
    fit_loader = make_loader(((x[fit_idx] - mean) / std).astype(np.float32), y[fit_idx], args.batch_size, True)
    val_loader = make_loader(((x[val_idx] - mean) / std).astype(np.float32), y[val_idx], args.batch_size, False)
    seed_everything(seed)
    model = STGCN(joint_count, args.dropout).to(device)
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
    parser.add_argument("--windows-cache", type=Path, required=True)
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

    x, y, groups, records, cache_config = experiment_common.load_windows_cache(
        args.windows_cache
    )
    joint_count = int(cache_config.get("joint_count", x.shape[-1] // 2))
    if x.shape[-1] != joint_count * 2:
        raise ValueError(
            f"ST-GCN expects XY features with D=joint_count*2, got D={x.shape[-1]} "
            f"and joint_count={joint_count}"
        )
    _, graph_description = skeleton_edges(joint_count)
    pose_extractor = "ViTPose" if joint_count == 17 else "BlazePose"
    feature_mode = str(cache_config.get("feature_mode", f"xy{joint_count * 2}"))
    x = x.reshape(x.shape[0], x.shape[1], joint_count, 2)

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
            "feature_mode": feature_mode,
            "channels": list(STGCN_CHANNELS),
            "temporal_kernel_size": TEMPORAL_KERNEL_SIZE,
            "pose_extractor": pose_extractor,
            "joint_count": joint_count,
            "graph": graph_description,
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

    metrics_header = [
        f"data_root: {args.data_root}",
        f"feature_mode: {feature_mode}",
        f"input_shape_per_window: [{args.window_size}, {joint_count}, 2]",
        f"graph: {graph_description}",
        f"window_size: {args.window_size}",
        f"stride: {args.stride}",
        f"missing_mode: {args.missing_mode}",
        f"visibility_threshold: {args.visibility_threshold}",
        f"channels: {STGCN_CHANNELS}",
        f"temporal_kernel_size: {TEMPORAL_KERNEL_SIZE}",
        f"dropout: {args.dropout}",
        f"epochs: {args.epochs}",
        f"patience: {args.patience}",
        f"batch_size: {args.batch_size}",
        f"learning_rate: {args.learning_rate}",
        f"weight_decay: {args.weight_decay}",
        f"folds: {args.folds}",
        f"seed: {args.seed}",
        f"device: {device}",
        f"annotation_csv: {FALL_ANNOTATION_CSV}",
        "label_level: window",
        "window_label_rule: ignore posture 0; majority vote -1(normal) vs 1(fall)",
    ]

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
            joint_count,
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
        row["train_windows"] = len(train_idx)
        row["test_windows"] = len(test_idx)
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
        checkpoint = {
            "model_type": "stgcn",
            "input_dim": x.shape[-1],
            "dropout": args.dropout,
            "channels": STGCN_CHANNELS,
            "temporal_kernel_size": TEMPORAL_KERNEL_SIZE,
            "state_dict": model.state_dict(),
            "feature_mean": mean,
            "feature_std": std,
            "joint_count": joint_count,
            "graph": graph_description,
            "best_epoch": best_epoch,
            "val_f1": val_f1,
            "pose_extractor": pose_extractor,
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

    fold_df = pd.DataFrame(fold_rows)
    experiment_common.save_experiment_metrics_text(
        output=output,
        model="stgcn",
        header=metrics_header,
        fold_rows=fold_df,
        overall=overall,
        y_true=y_true_all,
        y_pred=y_pred_all,
        confusion=cm,
    )
    experiment_common.save_experiment_metrics_json(
        output=output,
        model="stgcn",
        data_root=args.data_root,
        overall=overall,
        fold_rows=fold_df,
        device=device,
    )

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
    print(f"Saved: {output / 'metrics.txt'}")


if __name__ == "__main__":
    main()
