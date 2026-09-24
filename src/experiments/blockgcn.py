#!/usr/bin/env python3
"""Official BlockGCN fall-detection experiment adapted to BlazePose/UR-Fall.

This file reuses the authors' official CVPR 2024 BlockGCN model from
``third_party/BlockGCN`` while keeping this project's established pipeline:
BlazePose 33 joints (XY), 30-frame sliding windows, UR-Fall frame-label
relabeling, video-grouped 5-fold CV, and the same output/metric structure.

Important adaptation: the official BlockGCN TopoTrans.forward() repeats
topological features twice for NTU's two-person input. UR-Fall contains one
person, so this experiment patches only that person-count-specific repeat.
The BlockGCN graph/temporal/topological model itself remains the official code.
"""

from __future__ import annotations

BLOCKGCN_SCRIPT_VERSION = "2026-09-21-topology-padding-fix-v4"

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
        video_type_labels,
    )


# MediaPipe / BlazePose 33-landmark physical skeleton used by graph.blazepose.
POSE_EDGES = (
    (0, 1), (1, 2), (2, 3), (3, 7), (0, 4), (4, 5), (5, 6), (6, 8),
    (9, 10), (11, 12), (11, 13), (13, 15), (15, 17), (15, 19),
    (15, 21), (17, 19), (12, 14), (14, 16), (16, 18), (16, 20),
    (16, 22), (18, 20), (11, 23), (12, 24), (23, 24), (23, 25),
    (24, 26), (25, 27), (26, 28), (27, 29), (28, 30), (29, 31),
    (30, 32), (27, 31), (28, 32),
)

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
    frame_count = features.shape[0]
    if frame_count < window_size:
        return [], [], []

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
                start_frame=start_frame,
                end_frame=end_exclusive - 1,
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
    annotation_csv: Path = Path("data/urfall-cam0-falls.csv"),
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
        annotation_csv=annotation_csv,
    )
    return x.reshape(x.shape[0], x.shape[1], 33, 2), y, groups, records



def make_loader(x: np.ndarray, y: np.ndarray, batch_size: int, shuffle: bool) -> DataLoader:
    dataset = TensorDataset(torch.from_numpy(x), torch.from_numpy(y.astype(np.int64)))
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


@torch.no_grad()
def probabilities(model: nn.Module, loader: DataLoader, device: torch.device) -> np.ndarray:
    model.eval()
    pieces = []
    for batch_index, (xb, _) in enumerate(loader):
        xb = xb.to(device)
        if not torch.isfinite(xb).all():
            bad = int((~torch.isfinite(xb)).sum().item())
            raise FloatingPointError(
                f"Evaluation input contains {bad} non-finite values in batch {batch_index}."
            )

        logits = model(xb)
        finite_logits = torch.isfinite(logits)
        if not finite_logits.all():
            bad = int((~finite_logits).sum().item())
            finite_vals = logits[finite_logits]
            lo = float(finite_vals.min().item()) if finite_vals.numel() else float("nan")
            hi = float(finite_vals.max().item()) if finite_vals.numel() else float("nan")
            raise FloatingPointError(
                f"BlockGCN produced {bad}/{logits.numel()} non-finite validation logits "
                f"in batch {batch_index}; finite range=[{lo:.6g}, {hi:.6g}]. "
                "The model parameters likely became non-finite during optimizer.step(). "
                "Use the updated script diagnostics below; if this occurs at epoch 1, "
                "run with --base-lr 0.01 (warm-up starts at 0.002)."
            )

        probs = torch.softmax(logits.float(), dim=1)[:, 1]
        if not torch.isfinite(probs).all():
            bad = int((~torch.isfinite(probs)).sum().item())
            raise FloatingPointError(
                f"Softmax produced {bad}/{probs.numel()} non-finite probabilities "
                f"in evaluation batch {batch_index}, although logits were finite."
            )
        pieces.append(probs.cpu().numpy())

    out = np.concatenate(pieces).astype(np.float64, copy=False)
    if not np.isfinite(out).all():
        bad = int((~np.isfinite(out)).sum())
        raise FloatingPointError(
            f"Concatenated validation probabilities contain {bad}/{out.size} non-finite values."
        )
    return out


def safe_roc_auc(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    y_prob = np.asarray(y_prob, dtype=np.float64)
    if not np.isfinite(y_prob).all():
        bad = int((~np.isfinite(y_prob)).sum())
        raise FloatingPointError(
            f"ROC-AUC received {bad}/{y_prob.size} non-finite probabilities. "
            "This is a model numerical-stability problem, not a sklearn metric problem."
        )
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


def _window_frame_bounds(records: pd.DataFrame, i: int, window_size: int):
    """
    从共享 loader 的 records 中取得真实帧范围。
    优先使用 start_frame/end_frame；若没有，则使用零基 window_start/window_end。
    """
    row = records.iloc[i]

    if {"start_frame", "end_frame"}.issubset(records.columns):
        return int(row["start_frame"]), int(row["end_frame"])

    if {"window_start", "window_end"}.issubset(records.columns):
        # shared loader 通常把窗口位置记为 0-based。
        return int(row["window_start"]) + 1, int(row["window_end"]) + 1

    if "window_start" in records.columns:
        start_frame = int(row["window_start"]) + 1
        return start_frame, start_frame + window_size - 1

    raise RuntimeError(
        "load_all_windows() 返回的 records 中缺少帧范围字段；"
        "需要 start_frame/end_frame 或 window_start/window_end"
    )


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

        start_frame, end_frame = _window_frame_bounds(records, i, window_size)

        frame_labels = [
            sequence_annotations[frame_number]
            for frame_number in range(start_frame, end_frame + 1)
            if frame_number in sequence_annotations
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


def standardize(x: np.ndarray, fit_idx: np.ndarray):
    """Fit coordinate-wise standardization using training videos only."""
    mean = x[fit_idx].mean(axis=(0, 1), keepdims=True)
    std = x[fit_idx].std(axis=(0, 1), keepdims=True)
    std[std < 1e-6] = 1.0
    return mean.astype(np.float32), std.astype(np.float32)


def _patch_topotrans_for_single_person(blockgcn_module) -> None:
    """Remove the official NTU-only x.repeat(2, 1) from TopoTrans.forward()."""
    def forward_single_person(self, x):
        x = self.mlp(x)
        x = self.bn(x)
        x = self.relu(x)
        return x.unsqueeze(2).unsqueeze(3)

    blockgcn_module.TopoTrans.forward = forward_single_person


def _patch_structure_element_layer_for_safe_padding(blockgcn_module) -> None:
    """
    Make torch_topological.StructureElementLayer backward-safe.

    torch_topological.make_tensor() pads ragged persistence diagrams with NaN.
    The upstream StructureElementLayer performs arithmetic with those NaNs and
    only removes them later with torch.nansum(). Forward values can therefore
    look finite while gradients of ``centres`` and ``sharpness`` become NaN.

    This implementation keeps the same Gaussian structure-element computation
    for every finite persistence point, but masks padded / non-finite rows
    *before* they can participate in parameter arithmetic. Infinite essential
    bars are ignored as well, which is the standard practical treatment when
    feeding finite persistence coordinates to this feature layer.
    """

    def forward_safe(self, x):
        if x.ndim != 3 or x.shape[-1] != 3:
            raise ValueError(
                f"StructureElementLayer expected [B,N,3], got {tuple(x.shape)}"
            )

        B, N, _ = x.shape
        # A row is usable only when birth, death and dimension are all finite.
        # make_tensor() uses all-NaN rows for padding; essential bars may also
        # carry inf as their death coordinate. Neither should enter arithmetic
        # involving trainable centres/sharpness.
        valid = torch.isfinite(x).all(dim=-1)  # [B,N]
        clean = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

        # Same layout/computation as torch_topological StructureElementLayer.
        batch = torch.cat([clean] * self.n_elements, dim=1)

        centres = torch.cat([self.centres] * N, dim=1)
        centres = centres.view(-1, self.dim)
        centres = torch.stack([centres] * B, dim=0)
        centres = torch.cat(
            (centres, 2.0 * batch[..., -1].unsqueeze(-1)), dim=2
        )

        sharpness = torch.pow(self.sharpness, 2)
        sharpness = torch.cat([sharpness] * N, dim=1)
        sharpness = sharpness.view(-1, self.dim)
        sharpness = torch.stack([sharpness] * B, dim=0)
        sharpness = torch.cat(
            (sharpness, torch.ones_like(batch[..., -1].unsqueeze(-1))), dim=2
        )

        scores = (centres - batch).pow(2)
        scores = torch.mul(scores, sharpness).sum(dim=2)
        scores = torch.exp(-scores)
        scores = scores.view(B, self.n_elements, N)

        # Padding/non-finite persistence points contribute exactly zero.
        scores = scores * valid.unsqueeze(1).to(dtype=scores.dtype)
        return scores.sum(dim=2)

    blockgcn_module.StructureElementLayer.forward = forward_safe


def _patch_topo_for_numerical_stability(blockgcn_module) -> None:
    """
    Make the official Topo.forward() safe for degenerate UR-Fall windows.

    The upstream implementation uses
        (x - x.min()) / (x.max() - x.min())
    with no epsilon. A batch whose pairwise-distance matrix has zero range
    therefore produces 0/0 -> NaN, which later contaminates logits/softmax.

    We keep the official computation unchanged except for clamping that
    denominator to a small positive value.
    """
    def forward_stable(self, x):
        x = x.mean(1)
        x = x.unsqueeze(-1) - x.unsqueeze(-2)
        x = x.mean(-3)
        x = self.L2_norm(x)

        x_min = torch.min(x)
        x_max = torch.max(x)
        denom = (x_max - x_min).clamp_min(1e-6)
        x = (x - x_min) / denom

        # Defensive guard before persistent-homology operators. This should be
        # a no-op for normal batches, but prevents one malformed window from
        # poisoning an entire fold.
        x = torch.nan_to_num(x, nan=0.0, posinf=1.0, neginf=0.0)
        x = self.vr(x)
        x = blockgcn_module.make_tensor(x)
        x = self.pl(x)
        return torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

    blockgcn_module.Topo.forward = forward_stable


def load_official_blockgcn(blockgcn_root: Path):
    """Load the authors' official model and apply the UR-Fall one-person patch."""
    import importlib
    import sys

    root = Path(blockgcn_root).resolve()
    model_file = root / "model" / "BlockGCN.py"
    graph_file = root / "graph" / "blazepose.py"
    if not model_file.exists():
        raise FileNotFoundError(
            f"Official BlockGCN model not found: {model_file}\n"
            "Clone https://github.com/ZhouYuxuanYX/BlockGCN.git to "
            "third_party/BlockGCN (or pass --blockgcn-root)."
        )
    if not graph_file.exists():
        raise FileNotFoundError(
            f"BlazePose graph adapter not found: {graph_file}\n"
            "Copy the supplied blazepose.py to third_party/BlockGCN/graph/blazepose.py."
        )

    root_str = str(root)
    # Put the official repository first. More importantly, remove any already
    # imported package named ``graph`` (for example from Hyper-GCN or another
    # dependency). Python caches modules in sys.modules, so merely changing
    # sys.path is not enough when a different ``graph`` package was imported
    # earlier in the same interpreter.
    sys.path = [p for p in sys.path if p != root_str]
    sys.path.insert(0, root_str)
    for name in list(sys.modules):
        if name == "graph" or name.startswith("graph."):
            del sys.modules[name]
        if name == "model.BlockGCN":
            del sys.modules[name]

    importlib.invalidate_caches()
    graph_submodule = importlib.import_module("graph.blazepose")
    graph_package = importlib.import_module("graph")
    # Be explicit because the authors' import_class() walks attributes with
    # getattr(graph, "blazepose") rather than importing the dotted module.
    setattr(graph_package, "blazepose", graph_submodule)

    try:
        module = importlib.import_module("model.BlockGCN")
    except ModuleNotFoundError as exc:
        if exc.name in {"einops", "torch_topological"} or str(exc.name).startswith("torch_topological"):
            raise ModuleNotFoundError(
                f"Missing BlockGCN dependency: {exc.name}. "
                "Install with: uv pip install einops torch-topological"
            ) from exc
        raise

    # Replace the repository's fragile attribute-walking importer with a
    # standard dotted-module importer. This keeps the official model itself
    # unchanged while making graph.blazepose.Graph resolve reliably.
    def _import_class_dotted(name: str):
        module_name, attr_name = name.rsplit(".", 1)
        imported = importlib.import_module(module_name)
        return getattr(imported, attr_name)

    module.import_class = _import_class_dotted

    # Fail early with a useful diagnostic instead of the opaque getattr error.
    resolved_graph = module.import_class("graph.blazepose.Graph")
    if resolved_graph is None:
        raise RuntimeError("Failed to resolve graph.blazepose.Graph")

    _patch_structure_element_layer_for_safe_padding(module)
    _patch_topotrans_for_single_person(module)
    _patch_topo_for_numerical_stability(module)
    return module.Model, root


class BlockGCNAdapter(nn.Module):
    """Convert this project's [B,T,66] windows to official [N,C,T,V,M]."""
    def __init__(self, model_cls, window_size: int, dropout: float):
        super().__init__()
        self.model = model_cls(
            num_class=2,
            num_point=33,
            num_person=1,
            graph="graph.blazepose.Graph",
            graph_args={"labeling_mode": "spatial"},
            in_channels=2,
            drop_out=dropout,
            adaptive=True,
            window_size=window_size,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # [B,T,66] -> [B,2,T,33,1]
        x5 = x.reshape(x.shape[0], x.shape[1], 33, 2)
        x5 = x5.permute(0, 3, 1, 2).contiguous().unsqueeze(-1)
        dummy_y = torch.zeros(x5.shape[0], dtype=torch.long, device=x5.device)
        logits, _ = self.model(x5, dummy_y, x5)
        return logits


def set_sgd_lr(optimizer, base_lr: float, epoch_index: int, warmup_epochs: int,
               steps: Sequence[int], decay_rate: float) -> float:
    """CTR/BlockGCN-style warm-up + step schedule."""
    if warmup_epochs > 0 and epoch_index < warmup_epochs:
        lr = base_lr * float(epoch_index + 1) / float(warmup_epochs)
    else:
        lr = base_lr
        for step in steps:
            if epoch_index >= step:
                lr *= decay_rate
    for group in optimizer.param_groups:
        group["lr"] = lr
    return lr


def train_fold(x, y, groups, args, device, model_cls, seed):
    fit_idx, val_idx = inner_group_split_window_labels(y, groups, seed)
    mean, std = standardize(x, fit_idx)
    fit_x = ((x[fit_idx] - mean) / std).astype(np.float32)
    val_x = ((x[val_idx] - mean) / std).astype(np.float32)
    fit_loader = make_loader(fit_x, y[fit_idx], args.batch_size, True)
    val_loader = make_loader(val_x, y[val_idx], args.batch_size, False)

    seed_everything(seed)
    model = BlockGCNAdapter(model_cls, args.window_size, args.dropout).to(device)

    counts = np.bincount(y[fit_idx], minlength=2)
    if np.any(counts == 0):
        raise RuntimeError(f"A training split lost a class: counts={counts.tolist()}")
    if args.balanced_loss:
        weights = torch.tensor(
            len(fit_idx) / (2.0 * counts), dtype=torch.float32, device=device
        )
        criterion = nn.CrossEntropyLoss(weight=weights)
    else:
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
        lr = set_sgd_lr(
            optimizer,
            args.base_lr,
            epoch - 1,
            args.warmup_epochs,
            args.steps,
            args.lr_decay_rate,
        )
        model.train()
        loss_sum = 0.0
        correct = 0
        seen = 0

        for xb, yb in fit_loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            if not torch.isfinite(xb).all():
                raise FloatingPointError("Non-finite values found in a training input batch.")
            logits = model(xb)
            if not torch.isfinite(logits).all():
                bad = int((~torch.isfinite(logits)).sum().item())
                raise FloatingPointError(
                    f"BlockGCN produced {bad} non-finite training logits at epoch {epoch}. "
                    "The topology normalization has already been stabilized; if this "
                    "still occurs, retry with --base-lr 0.01."
                )
            loss = criterion(logits, yb)
            if not torch.isfinite(loss):
                raise FloatingPointError(
                    f"Non-finite training loss at epoch {epoch}; retry with --base-lr 0.01."
                )
            loss.backward()

            # Detect the first numerical failure BEFORE it contaminates validation.
            bad_grad_names = []
            for name, param in model.named_parameters():
                if param.grad is not None and not torch.isfinite(param.grad).all():
                    bad_grad_names.append(name)
                    if len(bad_grad_names) >= 5:
                        break
            if bad_grad_names:
                raise FloatingPointError(
                    f"Non-finite gradients at epoch {epoch}: {bad_grad_names}. "
                    "The safe persistence-padding patch is active in this script; "
                    "please report this exact parameter list if it still occurs."
                )

            if args.grad_clip > 0:
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                if not torch.isfinite(torch.as_tensor(grad_norm)):
                    raise FloatingPointError(
                        f"Non-finite gradient norm at epoch {epoch}. "
                        "Retry with --base-lr 0.01."
                    )
            optimizer.step()

            bad_param_names = []
            for name, param in model.named_parameters():
                if not torch.isfinite(param).all():
                    bad_param_names.append(name)
                    if len(bad_param_names) >= 5:
                        break
            if bad_param_names:
                raise FloatingPointError(
                    f"Optimizer step created non-finite parameters at epoch {epoch}: "
                    f"{bad_param_names}. Retry with --base-lr 0.01."
                )
            n = len(yb)
            loss_sum += float(loss.detach()) * n
            correct += int((logits.argmax(1) == yb).sum().item())
            seen += n

        train_loss = loss_sum / max(seen, 1)
        train_acc = correct / max(seen, 1)
        val_prob = probabilities(model, val_loader, device)
        val_pred = (val_prob >= args.threshold).astype(np.int64)
        val_metrics = compute_metrics(y[val_idx], val_pred, val_prob)
        val_f1 = float(val_metrics["f1"])

        history.append({
            "epoch": epoch,
            "lr": lr,
            "train_loss": train_loss,
            "train_accuracy": train_acc,
            "val_accuracy": val_metrics["accuracy"],
            "val_precision": val_metrics["precision"],
            "val_recall": val_metrics["recall"],
            "val_f1": val_f1,
            "val_roc_auc": val_metrics["roc_auc"],
        })

        if val_f1 > best_f1 or (np.isclose(val_f1, best_f1) and train_loss < best_loss):
            best_f1 = val_f1
            best_loss = train_loss
            best_epoch = epoch
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }

        if epoch == 1 or epoch % args.log_every == 0:
            print(
                f"    epoch={epoch:03d} lr={lr:.6f} loss={train_loss:.4f} "
                f"train_acc={train_acc:.4f} val_f1={val_f1:.4f} "
                f"val_recall={val_metrics['recall']:.4f}"
            )

        if epoch - best_epoch >= args.patience:
            print(
                f"    Early stopping at epoch {epoch}; best_epoch={best_epoch}, "
                f"best_val_f1={best_f1:.4f}"
            )
            break

    if best_state is None:
        raise RuntimeError("Training finished without a valid checkpoint")
    model.load_state_dict(best_state)
    return (
        model, mean, std, best_epoch, best_f1, history,
        (fit_idx, val_idx), time.perf_counter() - started,
    )


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
    parser = argparse.ArgumentParser(
        description="Official BlockGCN on UR-Fall with BlazePose and video-grouped CV"
    )
    parser.add_argument("--data-root", type=Path, default=Path("data/keypoints"))
    parser.add_argument(
        "--result-root", "--output-root", dest="result_root", type=Path,
        default=Path("results/blockgcn"),
    )
    parser.add_argument(
        "--blockgcn-root", type=Path, default=Path("third_party/BlockGCN"),
        help="Path to the authors' official BlockGCN repository",
    )
    parser.add_argument(
        "--annotation-csv", type=Path, default=Path("data/urfall-cam0-falls.csv")
    )
    parser.add_argument("--window-size", type=int, default=30)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--missing-mode", choices=("zero", "interp"), default="interp")
    parser.add_argument("--visibility-threshold", type=float, default=0.3)
    parser.add_argument("--dropout", type=float, default=0.3)

    # BlockGCN paper/official-training-style defaults.
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
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument(
        "--balanced-loss", action="store_true",
        help="Optional inverse-frequency class weighting; default is plain CE like the paper baseline.",
    )
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto", help="auto / cpu / cuda / mps")
    return parser.parse_args()


def dataset_run_name(data_root: Path) -> str:
    name = data_root.resolve().name.lower()
    return "blockgcn_normalized" if "normalized" in name else "blockgcn"


def main():
    args = parse_args()
    print(f"BlockGCN experiment script: {BLOCKGCN_SCRIPT_VERSION}")
    if min(args.window_size, args.stride, args.patience, args.epochs, args.batch_size) < 1:
        raise ValueError("window-size/stride/patience/epochs/batch-size must be positive")
    if args.folds < 2:
        raise ValueError("folds must be at least 2")
    if not 0 <= args.dropout < 1:
        raise ValueError("dropout must be in [0,1)")
    if not 0 < args.threshold < 1:
        raise ValueError("threshold must be in (0,1)")

    seed_everything(args.seed)
    device = choose_device(args.device)
    model_cls, official_root = load_official_blockgcn(args.blockgcn_root)

    if args.result_root == Path("results/blockgcn"):
        output = Path("results") / dataset_run_name(args.data_root)
    else:
        output = args.result_root
    output.mkdir(parents=True, exist_ok=True)
    models_dir = output / "models"
    histories_dir = output / "histories"
    models_dir.mkdir(parents=True, exist_ok=True)
    histories_dir.mkdir(parents=True, exist_ok=True)

    print(f"Device: {device}")
    print(f"Official BlockGCN root: {official_root}")
    print(f"Data root: {args.data_root}")
    print(f"Output dir: {output}")
    print(
        f"Input: window={args.window_size}, stride={args.stride}, "
        "33 BlazePose joints, XY only, one person"
    )

    x, y, groups, records = load_all_windows(
        args.data_root,
        args.window_size,
        args.stride,
        args.visibility_threshold,
        args.missing_mode,
        "xy66",
        None,
        annotation_csv=args.annotation_csv,
    )
    split_y = video_type_labels(groups)
    group_df = pd.DataFrame({"group": groups, "video_type": split_y}).drop_duplicates("group")
    class_groups = np.bincount(group_df["video_type"].to_numpy(dtype=np.int64), minlength=2)
    if class_groups.min() < args.folds:
        raise ValueError(
            f"Need at least {args.folds} videos per video type; "
            f"found ADL/Fall={class_groups.tolist()}"
        )

    config = vars(args).copy()
    for key, value in list(config.items()):
        if isinstance(value, Path):
            config[key] = str(value)
    config.update({
        "model_type": "BlockGCN (official CVPR 2024 implementation, UR-Fall adapter)",
        "official_model": "model.BlockGCN.Model",
        "official_repo": "https://github.com/ZhouYuxuanYX/BlockGCN",
        "graph": "graph.blazepose.Graph",
        "num_class": 2,
        "num_point": 33,
        "num_person": 1,
        "in_channels": 2,
        "feature_mode": "xy66",
        "pose_extractor": "BlazePose",
        "topotrans_patch": "remove NTU two-person repeat(2,1)",
        "label_level": "window",
        "window_label_rule": "ignore_0_then_majority_vote_-1_vs_1",
        "optimizer": "SGD(momentum=0.9,nesterov=True)",
        "classification_loss": "balanced CrossEntropyLoss" if args.balanced_loss else "plain CrossEntropyLoss",
    })
    (output / "config.json").write_text(
        json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    splitter = StratifiedGroupKFold(
        n_splits=args.folds, shuffle=True, random_state=args.seed
    )
    all_true, all_pred, all_prob, all_test_indices = [], [], [], []
    fold_rows = []
    metrics_lines = [
        "=" * 80,
        "MODEL: BlockGCN (official CVPR 2024 implementation, UR-Fall adapter)",
        "=" * 80,
        f"official_root: {official_root}",
        f"data_root: {args.data_root}",
        "feature_mode: xy66",
        f"input_shape_per_window: [{args.window_size}, 33, 2]",
        "num_person: 1",
        "graph: BlazePose 33-joint physical graph",
        "topotrans_patch: remove official NTU two-person repeat(2,1)",
        f"missing_mode: {args.missing_mode}",
        f"visibility_threshold: {args.visibility_threshold}",
        f"dropout: {args.dropout}",
        f"epochs: {args.epochs}",
        f"patience: {args.patience}",
        f"batch_size: {args.batch_size}",
        f"base_lr: {args.base_lr}",
        f"steps: {args.steps}",
        f"warmup_epochs: {args.warmup_epochs}",
        f"lr_decay_rate: {args.lr_decay_rate}",
        f"weight_decay: {args.weight_decay}",
        f"balanced_loss: {args.balanced_loss}",
        f"classification_threshold: {args.threshold}",
        "optimizer: SGD(momentum=0.9,nesterov=True)",
        f"folds: {args.folds}",
        f"seed: {args.seed}",
        f"device: {device}",
        f"annotation_csv: {args.annotation_csv}",
        "label_level: window",
        "window_label_rule: ignore posture 0; majority vote -1(normal) vs 1(fall)",
        "",
    ]

    for fold, (train_idx, test_idx) in enumerate(splitter.split(x, split_y, groups), 1):
        train_groups = set(groups[train_idx])
        test_groups = set(groups[test_idx])
        overlap = train_groups & test_groups
        if overlap:
            raise RuntimeError(f"Video leakage detected: {sorted(overlap)[:5]}")

        print("\n" + "=" * 76)
        print(
            f"BlockGCN | Fold {fold}/{args.folds} | "
            f"train_windows={len(train_idx)} | test_windows={len(test_idx)}"
        )
        (
            model, mean, std, best_epoch, val_f1, history, _, seconds
        ) = train_fold(
            x[train_idx], y[train_idx], groups[train_idx],
            args, device, model_cls, args.seed + fold,
        )

        test_x = ((x[test_idx] - mean) / std).astype(np.float32)
        prob = probabilities(
            model,
            make_loader(test_x, y[test_idx], args.batch_size, False),
            device,
        )
        pred = (prob >= args.threshold).astype(np.int64)
        fold_metrics = compute_metrics(y[test_idx], pred, prob)
        row = dict(fold_metrics)
        row.update({
            "fold": fold,
            "train_videos": len(train_groups),
            "test_videos": len(test_groups),
            "train_windows": len(train_idx),
            "test_windows": len(test_idx),
            "best_epoch": best_epoch,
            "val_f1": val_f1,
            "train_seconds": seconds,
        })
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

        metrics_lines.extend([
            f"[Fold {fold}/{args.folds}]",
            f"train_videos: {len(train_groups)}",
            f"test_videos: {len(test_groups)}",
            f"train_windows: {len(train_idx)}",
            f"test_windows: {len(test_idx)}",
            f"best_epoch: {best_epoch}",
            f"val_f1: {fmt_metric(val_f1)}",
            f"train_seconds: {seconds:.2f}",
        ])
        for key in ["accuracy", "precision", "recall", "f1", "roc_auc", "tn", "fp", "fn", "tp"]:
            metrics_lines.append(f"{key}: {fmt_metric(fold_metrics[key])}")
        metrics_lines.append("")

        checkpoint_args = vars(args).copy()
        for key, value in list(checkpoint_args.items()):
            if isinstance(value, Path):
                checkpoint_args[key] = str(value)
        torch.save({
            "model_type": "blockgcn_official_urfall",
            "official_model": "model.BlockGCN.Model",
            "num_class": 2,
            "num_point": 33,
            "num_person": 1,
            "in_channels": 2,
            "state_dict": model.state_dict(),
            "feature_mean": mean,
            "feature_std": std,
            "best_epoch": best_epoch,
            "val_f1": val_f1,
            "pose_extractor": "BlazePose",
            "args": checkpoint_args,
        }, models_dir / f"fold_{fold}.pt")
        pd.DataFrame(history).to_csv(
            histories_dir / f"fold_{fold}_history.csv", index=False
        )

        all_true.extend(y[test_idx].tolist())
        all_pred.extend(pred.tolist())
        all_prob.extend(prob.tolist())
        all_test_indices.extend(test_idx.tolist())

        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    y_true_all = np.asarray(all_true, dtype=np.int64)
    y_pred_all = np.asarray(all_pred, dtype=np.int64)
    y_prob_all = np.asarray(all_prob, dtype=np.float32)
    overall = compute_metrics(y_true_all, y_pred_all, y_prob_all)
    cm = confusion_matrix(y_true_all, y_pred_all, labels=[0, 1])
    report = classification_report(
        y_true_all, y_pred_all, labels=[0, 1],
        target_names=["ADL", "Fall"], digits=4, zero_division=0,
    )

    metrics_lines.extend([
        "=" * 80,
        "OVERALL OUT-OF-FOLD METRICS",
        "=" * 80,
    ])
    for key in ["accuracy", "precision", "recall", "f1", "roc_auc", "tn", "fp", "fn", "tp"]:
        metrics_lines.append(f"{key}: {fmt_metric(overall[key])}")
    metrics_lines.extend([
        "", "Classification report:", report,
        "Confusion matrix [[TN, FP], [FN, TP]]:", np.array2string(cm),
    ])

    fold_df = pd.DataFrame(fold_rows)
    metrics_lines.extend(["", "Fold mean ± std:"])
    for key in ["accuracy", "precision", "recall", "f1", "roc_auc"]:
        metrics_lines.append(
            f"{key}: {fold_df[key].mean():.6f} ± {fold_df[key].std(ddof=1):.6f}"
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
    print("BlockGCN OVERALL")
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
