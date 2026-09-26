#!/usr/bin/env python3
"""跌倒检测实验脚本的公共工具函数。

这个模块尽量保持轻量、少副作用。它先收集各实验脚本中重复的工具函数，
但现有实验脚本暂时还没有切换到这里，便于先审核公共接口再做大范围重构。
"""

from __future__ import annotations
import math
import random
import re
import json
from io import StringIO
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple
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

FALL_ANNOTATION_CSV = Path("data/urfall-cam0-falls.csv")


@dataclass


class WindowRecord:
    video_id: str
    source_file: str
    start_frame: int
    end_frame: int
    label: int


@dataclass(frozen=True)


class KeypointAdapter:
    name: str
    default_confidence_index: Optional[int]
    expected_joints: Optional[int]
    confidence_fields: Tuple[str, ...] = ("scores", "keypoint_scores")
KEYPOINT_ADAPTERS: Dict[str, KeypointAdapter] = {
    "auto": KeypointAdapter("auto", default_confidence_index=None, expected_joints=None),
    "generic": KeypointAdapter("generic", default_confidence_index=None, expected_joints=None),
    "blazepose": KeypointAdapter("blazepose", default_confidence_index=3, expected_joints=33),
    "vitpose": KeypointAdapter("vitpose", default_confidence_index=-1, expected_joints=17),
    "openpose": KeypointAdapter("openpose", default_confidence_index=2, expected_joints=25),
}


def get_keypoint_adapter(name: str) -> KeypointAdapter:
    key = name.strip().lower()
    if key not in KEYPOINT_ADAPTERS:
        available = ", ".join(sorted(KEYPOINT_ADAPTERS))
        raise ValueError(f"未知 keypoint adapter: {name!r}; 可选值：{available}")
    return KEYPOINT_ADAPTERS[key]


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def choose_device(requested: str = "auto") -> torch.device:
    requested = requested.lower()
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def normalize_label(value: Any) -> int:
    """把标量或字符串标签转换成二分类类别编号。"""
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
        raise ValueError(f"未知字符串标签：{value!r}")
    value = int(value)
    if value not in (0, 1):
        raise ValueError(f"期望二分类标签 0/1，实际得到 {value}")
    return value


def normalize_label_array(values: np.ndarray) -> np.ndarray:
    return np.asarray([normalize_label(value) for value in values], dtype=np.int64)


def canonical_sequence_name(value: Any) -> Optional[str]:
    """把 fall-01-cam0-rgb 这类 UR-Fall ID 统一成 fall-01。"""
    value = str(value).strip().lower()
    match = re.search(r"(fall|adl)-?(\d+)", value)
    if match is None:
        return None
    return f"{match.group(1)}-{int(match.group(2)):02d}"


def video_type_label(value: Any) -> int:
    """返回原始视频级 UR-Fall 类型，用于分组和分层划分。"""
    sequence_name = canonical_sequence_name(value)
    if sequence_name is None:
        raise ValueError(f"无法从以下值推断 UR-Fall 视频类型：{value!r}")
    return 1 if sequence_name.startswith("fall-") else 0


def video_type_labels(groups: np.ndarray) -> np.ndarray:
    return np.asarray([video_type_label(group) for group in groups], dtype=np.int64)


def group_stratification_labels(y: np.ndarray, groups: np.ndarray) -> np.ndarray:
    """Return per-window labels used only for grouped stratification.

    UR-Fall groups encode the original video type in their names, so we keep
    using that stable video-level type. Datasets such as Le2i have mixed
    videos; for those, stratify each group by whether it contains at least one
    positive window.
    """
    try:
        labels = video_type_labels(groups)
    except ValueError:
        y = np.asarray(y, dtype=np.int64)
        groups = np.asarray(groups).astype(str)
        group_labels: Dict[str, int] = {}
        for group in np.unique(groups):
            group_y = y[groups == group]
            group_labels[str(group)] = int(np.any(group_y == 1))
        labels = np.asarray([group_labels[str(group)] for group in groups], dtype=np.int64)
    if len(np.unique(labels)) >= 2:
        return labels

    # Some event datasets, including typical Le2i exports, can have a fall
    # interval in every video. StratifiedGroupKFold still needs at least two
    # strata, so create a stable auxiliary split label while keeping groups
    # intact and leaving the real y labels untouched.
    groups = np.asarray(groups).astype(str)
    group_to_aux = {
        str(group): index % 2
        for index, group in enumerate(sorted(np.unique(groups)))
    }
    return np.asarray([group_to_aux[str(group)] for group in groups], dtype=np.int64)


def load_fall_frame_labels(csv_path: Path) -> Dict[str, Dict[int, int]]:
    """从前三列读取 UR-Fall 官方逐帧姿态标签。"""
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"找不到 UR-Fall 标注 CSV：{csv_path}")
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
                    f"{csv_path} 第 {line_number} 行出现未知姿态标签：{posture_label}"
                )
            annotations.setdefault(sequence_name, {})[frame_number] = posture_label
            valid_rows += 1
    if valid_rows == 0:
        raise RuntimeError(f"{csv_path} 没有读取到有效 UR-Fall 标注")
    print(
        f"已读取 UR-Fall 标注：{len(annotations)} 个 fall 序列 | "
        f"{valid_rows} 个已标注帧"
    )
    return annotations


def get_window_label_from_urfall(
    frame_labels: Sequence[Optional[int]],
) -> Optional[int]:
    """按“跌倒事件”而不是“倒地姿态”给 UR-Fall 窗口标注。

    UR-Fall 原始逐帧姿态标签：
      - -1：正常/非躺倒状态
      -  0：跌倒过渡过程（新的 Fall Event 正类）
      -  1：跌倒后的躺地状态

    新窗口规则：
      - 窗口中心位于原始标签 0 的跌倒过渡阶段 -> Fall = 1
      - 整个窗口完全不包含原始标签 0              -> Non-fall = 0
      - 窗口包含原始标签 0，但中心不在过渡阶段    -> None（忽略）
      - 整个窗口都没有有效逐帧标注                  -> None（忽略）

    偶数长度窗口的中心位于两个中间帧之间；只有两个中间帧都为 0 时，
    才认为中心明确位于跌倒过渡阶段。
    """
    labels = list(frame_labels)
    if not labels:
        return None
    valid_labels = [int(label) for label in labels if label is not None]
    if not valid_labels:
        return None

    # 完全没有跌倒过渡帧 0：Non-fall。
    # 因此正常阶段 -1 和跌倒后的躺地阶段 1 都不再直接作为 Fall 正类。
    if 0 not in valid_labels:
        return 0
    window_size = len(labels)
    if window_size % 2 == 1:
        center_labels = [labels[window_size // 2]]
    else:
        center_labels = [
            labels[window_size // 2 - 1],
            labels[window_size // 2],
        ]
    if all(label == 0 for label in center_labels):
        return 1

    # 窗口碰到跌倒事件，但中心不在事件内部：边界窗口，忽略。
    return None


def interpolate_1d(values: np.ndarray) -> np.ndarray:
    """沿时间轴做线性插值，边缘 NaN 使用最近的有效值填充。"""
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
    expected_joints: Optional[int] = 33,
    scores: Optional[np.ndarray] = None,
    confidence_index: Optional[int] = None,
) -> np.ndarray:
    """返回清洗后的 x/y 关键点，形状为 [T,J,2]。"""
    if keypoints.ndim != 3 or keypoints.shape[2] < 2:
        raise ValueError(f"期望 keypoints 形状为 [T,J,C>=2]，实际为 {keypoints.shape}")
    if expected_joints is not None and keypoints.shape[1] != expected_joints:
        raise ValueError(
            f"期望 {expected_joints} 个关节，实际为 {keypoints.shape[1]}"
        )
    xy = keypoints[..., :2].astype(np.float32, copy=True)
    joint_valid = np.isfinite(xy).all(axis=-1)
    if scores is not None:
        visibility = np.asarray(scores, dtype=np.float32)
        if visibility.shape != keypoints.shape[:2]:
            raise ValueError(
                f"scores 形状应为 {keypoints.shape[:2]}，实际为 {visibility.shape}"
            )
    elif confidence_index is not None:
        channels = keypoints.shape[2]
        index = confidence_index if confidence_index >= 0 else channels + confidence_index
        if index < 0 or index >= channels:
            raise ValueError(f"confidence_index={confidence_index} 超出 C={channels}")
        visibility = keypoints[..., index]
    elif keypoints.shape[2] >= 4:
        visibility = keypoints[..., 3]
    elif keypoints.shape[2] >= 3:
        visibility = keypoints[..., -1]
    else:
        visibility = None
    if visibility is not None:
        joint_valid &= np.isfinite(visibility) & (visibility >= visibility_threshold)
    if valid_mask is not None:
        supplied_valid = np.asarray(valid_mask).astype(bool)
        if supplied_valid.shape == xy.shape[:2]:
            joint_valid &= supplied_valid
        else:
            supplied_flat = supplied_valid.reshape(-1)
            if supplied_flat.size == xy.shape[0] * xy.shape[1]:
                joint_valid &= supplied_flat.reshape(xy.shape[:2])
            elif supplied_flat.size == xy.shape[0]:
                joint_valid &= supplied_flat[:, None]
            else:
                raise ValueError(
                    f"valid_mask 长度 {supplied_flat.size} 与帧数/关节数 "
                    f"{xy.shape[:2]} 不一致"
                )
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
        raise ValueError(f"未知 missing_mode：{missing_mode}")
    return xy.astype(np.float32)


def frame_valid_mask_from_npz_mask(
    valid_mask: Optional[np.ndarray],
    frame_count: int,
    joint_count: int,
) -> Optional[np.ndarray]:
    """把 NPZ 中可能的帧级/关节级 valid_mask 统一成帧级 [T]。"""
    if valid_mask is None:
        return None
    supplied_valid = np.asarray(valid_mask).astype(bool)
    if supplied_valid.shape == (frame_count, joint_count):
        return supplied_valid.any(axis=1)
    supplied_flat = supplied_valid.reshape(-1)
    if supplied_flat.size == frame_count * joint_count:
        return supplied_flat.reshape(frame_count, joint_count).any(axis=1)
    if supplied_flat.size == frame_count:
        return supplied_flat
    raise ValueError(
        f"valid_mask 长度 {supplied_flat.size} 与帧数/关节数 "
        f"({frame_count}, {joint_count}) 不一致"
    )


def adapt_keypoints_from_npz(
    data: Any,
    adapter: str = "auto",
    visibility_threshold: float = 0.3,
    missing_mode: str = "mask",
    confidence_index: Optional[int] = None,
) -> Tuple[np.ndarray, Optional[np.ndarray], int, str]:
    """读取一个 NPZ 的关键点字段，并按指定姿态估计器转换为 [T,J,2]。



    支持的最低约定是存在 keypoints=[T,J,C>=2]。VitPose 一类输出可以把置信度

    放在 keypoints 最后一维，也可以额外保存 scores/keypoint_scores=[T,J]。

    """
    if "keypoints" not in data:
        raise KeyError("missing 'keypoints'")
    spec = get_keypoint_adapter(adapter)
    keypoints = np.asarray(data["keypoints"])
    valid_mask = data["valid_mask"] if "valid_mask" in data else None
    scores = None
    for field in spec.confidence_fields:
        if field in data:
            scores = np.asarray(data[field])
            break
    selected_confidence_index = (
        confidence_index
        if confidence_index is not None
        else spec.default_confidence_index
    )
    expected_joints = spec.expected_joints if spec.name != "auto" else None
    xy = preprocess_keypoints(
        keypoints=keypoints,
        valid_mask=valid_mask,
        visibility_threshold=visibility_threshold,
        missing_mode=missing_mode,
        expected_joints=expected_joints,
        scores=scores,
        confidence_index=selected_confidence_index,
    )
    frame_valid_mask = frame_valid_mask_from_npz_mask(
        valid_mask,
        frame_count=int(xy.shape[0]),
        joint_count=int(xy.shape[1]),
    )
    return xy, frame_valid_mask, int(xy.shape[1]), spec.name


def fill_missing_temporally(
    values: np.ndarray,
    missing_mode: str = "interp",
) -> np.ndarray:
    """对一个窗口内的缺失值做统一处理，支持 [T,D] 或 [T,J,C]。"""
    out = values.astype(np.float32, copy=True)
    if missing_mode in ("interp", "interpolate"):
        if out.ndim < 2:
            return interpolate_1d(out)
        flat = out.reshape(out.shape[0], -1)
        for feature_idx in range(flat.shape[1]):
            flat[:, feature_idx] = interpolate_1d(flat[:, feature_idx])
        out = flat.reshape(out.shape)
        return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    if missing_mode == "zero":
        return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    raise ValueError(f"未知 missing_mode：{missing_mode}")


def build_frame_features(
    xy: np.ndarray,
    feature_mode: str,
    joint_indices: Optional[Sequence[int]] = None,
) -> np.ndarray:
    """从 [T,J,2] 的 x/y 关键点构建逐帧模型特征。"""
    if xy.ndim != 3 or xy.shape[2] != 2:
        raise ValueError(f"期望 xy 形状为 [T,J,2]，实际为 {xy.shape}")
    if feature_mode in ("xy", "xy66", "xy_flat"):
        return xy.reshape(xy.shape[0], -1).astype(np.float32)
    if feature_mode in ("joints", "joints16"):
        if joint_indices is None:
            raise ValueError(f"{feature_mode} 需要 --joint-indices")
        if feature_mode == "joints16" and len(joint_indices) != 8:
            raise ValueError("joints16 需要正好 8 个关节索引")
        idx = np.asarray(joint_indices, dtype=np.int64)
        if np.any(idx < 0) or np.any(idx >= xy.shape[1]):
            raise ValueError(f"所有关节索引必须位于 [0, {xy.shape[1] - 1}]")
        return xy[:, idx, :].reshape(xy.shape[0], len(idx) * 2).astype(np.float32)
    raise ValueError(f"未知 feature_mode：{feature_mode}")


def discover_npz_files(root: Path) -> List[Path]:
    files = sorted(Path(root).rglob("*.npz"))
    if not files:
        raise FileNotFoundError(f"在以下目录没有找到 .npz 文件：{root}")
    return files


def load_sequence_windows(
    data_root: Path,
    window_size: int,
    stride: int,
    visibility_threshold: float,
    missing_mode: str,
    feature_mode: str = "xy66",
    joint_indices: Optional[Sequence[int]] = None,
    min_valid_frames: int = 1,
    annotation_csv: Path = FALL_ANNOTATION_CSV,
    keypoint_adapter: str = "blazepose",
    confidence_index: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, pd.DataFrame, Dict[str, Any]]:
    """从 NPZ 目录加载统一的 [N,T,D] 滑动窗口数据。

    UR-Fall 的 Fall 视频使用事件中心标注：原始标签 0 视为跌倒过程，
    窗口中心位于该过程时为正类；完全不接触该过程时为负类；边界窗口忽略。



    该函数对应 LSTM/MLP/Transformer 一类时序实验的公共数据管线。

    图卷积实验可在此基础上 reshape 成 [N,C,T,V]。

    """
    x_all: List[np.ndarray] = []
    y_all: List[int] = []
    groups_all: List[str] = []
    records_all: List[WindowRecord] = []
    files = discover_npz_files(data_root)
    fall_annotations: Optional[Dict[str, Dict[int, int]]] = None
    detected_label_source: Optional[str] = None
    skipped_short = 0
    skipped_no_valid_windows = 0
    for path in files:
        with np.load(path, allow_pickle=True) as data:
            label_data = data["label"] if "label" in data else path.parent.name
            if "video_id" in data:
                raw_video_id = data["video_id"]
                if isinstance(raw_video_id, np.ndarray) and raw_video_id.ndim == 0:
                    raw_video_id = raw_video_id.item()
                if isinstance(raw_video_id, bytes):
                    raw_video_id = raw_video_id.decode("utf-8")
                video_id = str(raw_video_id)
            else:
                video_id = path.stem
            xy, valid_mask, joint_count, resolved_adapter = adapt_keypoints_from_npz(
                data,
                adapter=keypoint_adapter,
                visibility_threshold=visibility_threshold,
                missing_mode="mask",
                confidence_index=confidence_index,
            )
            frame_count = xy.shape[0]
            if "frame_indices" in data:
                frame_indices = np.asarray(data["frame_indices"]).reshape(-1)
            else:
                frame_indices = np.arange(1, frame_count + 1, dtype=np.int64)
            frame_labels = (
                np.asarray(data["frame_labels"]).reshape(-1)
                if "frame_labels" in data
                else None
            )
        features = build_frame_features(
            xy=xy,
            feature_mode=feature_mode,
            joint_indices=joint_indices,
        )
        if frame_labels is not None:
            current_label_source = "frame_labels"
            if detected_label_source is None:
                detected_label_source = current_label_source
            elif detected_label_source != current_label_source:
                raise RuntimeError("同一窗口缓存中混用了 frame_labels 和 UR-Fall CSV 标签源")
            x_list, y_list, records = make_frame_label_windows(
                features=features,
                frame_labels=frame_labels,
                video_id=video_id,
                source_file=str(path),
                window_size=window_size,
                stride=stride,
                frame_indices=frame_indices,
                valid_mask=valid_mask,
                min_valid_frames=min_valid_frames,
                missing_mode=missing_mode,
            )
        else:
            current_label_source = "urfall_csv"
            if detected_label_source is None:
                detected_label_source = current_label_source
            elif detected_label_source != current_label_source:
                raise RuntimeError("同一窗口缓存中混用了 frame_labels 和 UR-Fall CSV 标签源")
            if fall_annotations is None:
                fall_annotations = load_fall_frame_labels(annotation_csv)
            sequence_name = canonical_sequence_name(video_id)
            if sequence_name is None:
                sequence_name = canonical_sequence_name(path.stem)
            if sequence_name is not None:
                video_level_label = 1 if sequence_name.startswith("fall-") else 0
            else:
                video_level_label = normalize_label(label_data)
            sequence_annotations: Optional[Dict[int, int]] = None
            if video_level_label == 1:
                if sequence_name is None:
                    raise RuntimeError(
                        f"无法从 {video_id} / {path.name} 解析 UR-Fall 序列名"
                    )
                sequence_annotations = fall_annotations.get(sequence_name)
                if sequence_annotations is None:
                    raise RuntimeError(
                        f"{video_id} -> {sequence_name} 在 {annotation_csv} 中没有逐帧标注"
                    )
            x_list, y_list, records = make_urfall_windows(
                features=features,
                video_level_label=video_level_label,
                video_id=video_id,
                source_file=str(path),
                window_size=window_size,
                stride=stride,
                frame_indices=frame_indices,
                sequence_annotations=sequence_annotations,
                valid_mask=valid_mask,
                min_valid_frames=min_valid_frames,
                missing_mode=missing_mode,
            )
        if frame_count < window_size:
            skipped_short += 1
            continue
        if not x_list:
            skipped_no_valid_windows += 1
            continue
        x_all.extend(x_list)
        y_all.extend(y_list)
        groups_all.extend([video_id] * len(x_list))
        records_all.extend(records)
    if not x_all:
        raise RuntimeError("没有生成有效窗口")
    x = np.stack(x_all).astype(np.float32)
    y = np.asarray(y_all, dtype=np.int64)
    groups = np.asarray(groups_all)
    records = pd.DataFrame([asdict(record) for record in records_all])
    print(
        f"已加载 {len(files)} 个 NPZ 视频 | windows={len(x)} | shape={x.shape} | "
        f"ADL={int((y == 0).sum())} | Fall={int((y == 1).sum())} | "
        f"short_videos_skipped={skipped_short} | "
        f"no_valid_windows_skipped={skipped_no_valid_windows}"
    )
    if feature_mode in ("xy", "xy66", "xy_flat"):
        output_joint_count = int(x.shape[-1] // 2)
    elif joint_indices is not None:
        output_joint_count = len(joint_indices)
    else:
        output_joint_count = int(joint_count)
    metadata = {
        "label_source": detected_label_source or "unknown",
        "keypoint_adapter": resolved_adapter,
        "joint_count": output_joint_count,
        "feature_dim": int(x.shape[-1]),
    }
    return x, y, groups, records, metadata


def save_windows_cache(
    output: Path,
    x: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    records: pd.DataFrame,
    config: Optional[Dict[str, Any]] = None,
) -> None:
    """保存统一滑动窗口缓存，供所有实验复用。"""
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        x=x.astype(np.float32),
        y=y.astype(np.int64),
        groups=np.asarray(groups).astype(str),
        records_json=records.to_json(orient="records", force_ascii=False),
        config_json=json.dumps(config or {}, ensure_ascii=False),
    )


def load_windows_cache(cache_path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray, pd.DataFrame, Dict[str, Any]]:
    """读取 prepare_windows.py 生成的统一滑动窗口缓存。"""
    cache_path = Path(cache_path)
    if not cache_path.exists():
        raise FileNotFoundError(f"找不到窗口缓存：{cache_path}")
    with np.load(cache_path, allow_pickle=False) as data:
        x = data["x"].astype(np.float32)
        y = data["y"].astype(np.int64)
        groups = data["groups"].astype(str)
        records_json = str(data["records_json"].item())
        config_json = str(data["config_json"].item())
    records = pd.read_json(StringIO(records_json), orient="records")
    config = json.loads(config_json) if config_json else {}
    return x, y, groups, records, config


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
) -> Tuple[List[np.ndarray], List[int], List[WindowRecord]]:
    """按新的 UR-Fall Fall Event 规则生成滑动窗口。

    ADL 视频的有效窗口直接标记为 0。
    Fall 视频依据官方逐帧 -1/0/1 标注：中心位于 0 阶段时标记为 1；
    完全不包含 0 的窗口标记为 0；碰到 0 但中心不在 0 阶段的边界窗口忽略。
    """
    frame_count = features.shape[0]
    if frame_count < window_size:
        return [], [], []
    frame_indices = np.asarray(frame_indices).reshape(-1)
    if len(frame_indices) != frame_count:
        raise ValueError(
            f"frame_indices 长度 {len(frame_indices)} 与特征帧数 {frame_count} 不一致"
        )
    x_list: List[np.ndarray] = []
    y_list: List[int] = []
    records: List[WindowRecord] = []
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
                raise RuntimeError(f"fall 视频 {video_id} 缺少逐帧标注")

            # 保持窗口内逐帧位置与标注一一对应；缺失标注保留为 None。
            # 这样才能按窗口中心判断，而不是过滤后做多数投票。
            posture_labels = [
                sequence_annotations.get(int(frame_number))
                for frame_number in frame_indices[start:end]
            ]
            y_from_csv = get_window_label_from_urfall(posture_labels)
            if y_from_csv is None:
                continue
            y_win = int(y_from_csv)
        x_win = fill_missing_temporally(features[start:end], missing_mode)
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


def make_frame_label_windows(
    features: np.ndarray,
    frame_labels: np.ndarray,
    video_id: str,
    source_file: str,
    window_size: int,
    stride: int,
    frame_indices: np.ndarray,
    valid_mask: Optional[np.ndarray] = None,
    min_valid_frames: int = 1,
    missing_mode: str = "interp",
) -> Tuple[List[np.ndarray], List[int], List[WindowRecord]]:
    """Generate windows from binary frame labels.

    The window label is the center frame label. Boundary or unknown labels
    (<0) are skipped. This matches Le2i NPZ files produced by
    blazepose_le2i.py, where videos can contain both classes.
    """
    frame_count = features.shape[0]
    if frame_count < window_size:
        return [], [], []
    frame_indices = np.asarray(frame_indices).reshape(-1)
    labels = np.asarray(frame_labels).reshape(-1)
    if len(frame_indices) != frame_count:
        raise ValueError(
            f"frame_indices 长度 {len(frame_indices)} 与特征帧数 {frame_count} 不一致"
        )
    if len(labels) != frame_count:
        raise ValueError(
            f"frame_labels 长度 {len(labels)} 与特征帧数 {frame_count} 不一致"
        )
    x_list: List[np.ndarray] = []
    y_list: List[int] = []
    records: List[WindowRecord] = []
    for start in range(0, frame_count - window_size + 1, stride):
        end = start + window_size
        if valid_mask is not None:
            valid_count = int(np.count_nonzero(valid_mask[start:end]))
            if valid_count < min_valid_frames:
                continue
        center = start + window_size // 2
        label = int(labels[center])
        if label not in (0, 1):
            continue
        x_win = fill_missing_temporally(features[start:end], missing_mode)
        x_list.append(x_win.astype(np.float32))
        y_list.append(label)
        records.append(
            WindowRecord(
                video_id=video_id,
                source_file=source_file,
                start_frame=int(frame_indices[start]),
                end_frame=int(frame_indices[end - 1]),
                label=label,
            )
        )
    return x_list, y_list, records


def fit_scaler(
    x: np.ndarray,
    fit_idx: np.ndarray,
    mode: str,
) -> Tuple[np.ndarray, np.ndarray]:
    """仅用指定索引拟合序列特征归一化参数。"""
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
        raise ValueError(f"未知归一化模式：{mode}")
    scale[scale < 1e-6] = 1.0
    return offset.astype(np.float32), scale.astype(np.float32)


def standardize(x: np.ndarray, fit_idx: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """为 [N,C,T,V] 或类似张量计算 ST-GCN 风格的 z-score 参数。"""
    mean = x[fit_idx].mean(axis=(0, 2), keepdims=True)
    std = x[fit_idx].std(axis=(0, 2), keepdims=True)
    std[std < 1e-6] = 1.0
    return mean.astype(np.float32), std.astype(np.float32)


def make_loader(
    x: np.ndarray,
    y: np.ndarray,
    batch_size: int,
    shuffle: bool,
) -> DataLoader:
    dataset = TensorDataset(torch.from_numpy(x), torch.from_numpy(y.astype(np.int64)))
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


@torch.no_grad()


def probabilities(model: nn.Module, loader: DataLoader, device: torch.device) -> np.ndarray:
    model.eval()
    pieces = []
    for xb, _ in loader:
        logits = model(xb.to(device))
        pieces.append(torch.softmax(logits, dim=1)[:, 1].cpu().numpy())
    return np.concatenate(pieces)


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
        "recall": float(recall_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "roc_auc": safe_roc_auc(y_true, y_prob),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def fmt_metric(value: Any) -> str:
    if isinstance(value, float) and math.isnan(value):
        return "nan"
    return f"{value:.6f}" if isinstance(value, float) else str(value)


def inner_group_split_by_video_type(
    y: np.ndarray,
    groups: np.ndarray,
    seed: int,
    max_splits: int = 5,
) -> Tuple[np.ndarray, np.ndarray]:
    """按视频分组，并按原始 ADL/fall 视频类型分层划分验证集。"""
    split_y = group_stratification_labels(y, groups)
    unique_groups = np.unique(groups)
    group_type: Dict[Any, int] = {}
    for group in unique_groups:
        values = np.unique(split_y[groups == group])
        if len(values) != 1:
            raise RuntimeError(f"视频 {group} 出现多个视频类型")
        group_type[group] = int(values[0])
    class_group_counts = np.bincount(
        np.asarray(list(group_type.values()), dtype=np.int64), minlength=2
    )
    n_splits = int(min(max_splits, class_group_counts.min()))
    if n_splits < 2:
        raise RuntimeError(
            "内层验证至少需要每种原始视频类型各 2 个视频；"
            f"当前为 {class_group_counts.tolist()}"
        )
    splitter = StratifiedGroupKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=seed,
    )
    for fit_idx, val_idx in splitter.split(np.zeros(len(y)), split_y, groups):
        if len(np.unique(y[fit_idx])) == 2 and len(np.unique(y[val_idx])) == 2:
            return fit_idx, val_idx
    raise ValueError(
        "无法创建同时包含两个窗口类别的视频分组验证划分"
    )


def tune_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    objective: str = "accuracy",
    min_recall: float = 0.0,
) -> Tuple[float, Dict[str, float]]:
    """为类别不平衡的二分类实验选择验证集阈值。"""
    if objective not in {"accuracy", "precision", "recall", "f1"}:
        raise ValueError(f"不支持的阈值优化目标：{objective}")
    if not 0.0 <= min_recall <= 1.0:
        raise ValueError("min_recall 必须位于 0 到 1 之间")
    candidates = np.unique(np.r_[0.05, np.linspace(0.1, 0.9, 81), 0.95, y_prob])
    best_threshold = 0.5
    best_metrics = compute_metrics(y_true, (y_prob >= 0.5).astype(np.int64), y_prob)
    best_score = -float("inf")
    for threshold in candidates:
        pred = (y_prob >= threshold).astype(np.int64)
        metrics = compute_metrics(y_true, pred, y_prob)
        if metrics["recall"] + 1e-12 < min_recall:
            continue
        score = metrics[objective]
        if score > best_score or (
            score == best_score and float(threshold) > best_threshold
        ):
            best_score = score
            best_threshold = float(threshold)
            best_metrics = metrics
    if best_score == -float("inf"):
        for threshold in candidates:
            pred = (y_prob >= threshold).astype(np.int64)
            metrics = compute_metrics(y_true, pred, y_prob)
            score = metrics[objective]
            if score > best_score or (
                score == best_score and float(threshold) > best_threshold
            ):
                best_score = score
                best_threshold = float(threshold)
                best_metrics = metrics
    return best_threshold, best_metrics


def aggregate_video_predictions(
    prediction_frame: pd.DataFrame,
    mode: str = "mean",
    topk: int = 5,
) -> pd.DataFrame:
    """把窗口级预测聚合成每个视频一个概率。"""
    if topk < 1:
        raise ValueError("topk 必须为正数")
    rows = []
    for video_id, group in prediction_frame.groupby("video_id", sort=False):
        probs = group["fall_probability"].to_numpy()
        if mode == "mean":
            video_probability = float(probs.mean())
        elif mode == "max":
            video_probability = float(probs.max())
        elif mode == "topk_mean":
            k = min(topk, len(probs))
            video_probability = float(np.sort(probs)[-k:].mean())
        else:
            raise ValueError(f"未知视频级聚合方式：{mode}")
        row = {
            "video_id": video_id,
            "label": video_type_label(video_id),
            "fall_probability": video_probability,
            "windows": int(len(group)),
        }
        if "seed" in group:
            row["seed"] = int(group["seed"].iloc[0])
        if "fold" in group:
            row["fold"] = int(group["fold"].iloc[0])
        if "threshold" in group:
            row["threshold"] = float(group["threshold"].iloc[0])
        rows.append(row)
    return pd.DataFrame(rows)


def make_seed_output_dirs(
    output: Path,
    seed: int,
) -> Tuple[Path, Path, Path]:
    """创建并返回 seed 级输出目录、模型目录和训练历史目录。"""
    seed_dir = Path(output) / f"seed_{seed}"
    model_dir = seed_dir / "models"
    history_dir = seed_dir / "histories"
    model_dir.mkdir(parents=True, exist_ok=True)
    history_dir.mkdir(parents=True, exist_ok=True)
    return seed_dir, model_dir, history_dir


def json_ready_args(args: Any) -> Dict[str, Any]:
    """把 argparse Namespace 等对象转换成可写入 JSON 的字典。"""
    values = vars(args) if hasattr(args, "__dict__") else dict(args)
    return {
        key: str(value) if isinstance(value, Path) else value
        for key, value in values.items()
    }


def save_experiment_config(
    output: Path,
    args: Any,
    extra: Optional[Dict[str, Any]] = None,
    filename: str = "config.json",
) -> Dict[str, Any]:
    """保存实验配置，并返回最终写出的配置字典。"""
    config = json_ready_args(args)
    if extra:
        config.update(extra)
    Path(output).mkdir(parents=True, exist_ok=True)
    (Path(output) / filename).write_text(
        json.dumps(config, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return config


def save_metrics_json(
    output: Path,
    summary: Dict[str, Any],
    filename: str = "metrics.json",
) -> None:
    """保存实验指标汇总 JSON。"""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    (output / filename).write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def save_experiment_metrics_json(
    output: Path,
    model: str,
    data_root: Path,
    overall: Dict[str, Any],
    fold_rows: Sequence[Dict[str, Any]] | pd.DataFrame,
    device: Optional[Any] = None,
    extra: Optional[Dict[str, Any]] = None,
    metric_columns: Sequence[str] = (
        "accuracy",
        "precision",
        "recall",
        "f1",
        "roc_auc",
        "tn",
        "fp",
        "fn",
        "tp",
    ),
    filename: str = "metrics.json",
) -> Dict[str, Any]:
    """保存各实验统一结构的 JSON 指标汇总。"""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    fold_frame = (
        fold_rows.copy()
        if isinstance(fold_rows, pd.DataFrame)
        else pd.DataFrame(list(fold_rows))
    )
    available_metrics = [column for column in metric_columns if column in fold_frame]
    summary: Dict[str, Any] = {
        "model": model,
        "data_root": str(data_root),
        "output_dir": str(output),
        "overall": dict(overall),
        "fold_mean": fold_frame[available_metrics].mean(numeric_only=True).to_dict(),
        "fold_std": fold_frame[available_metrics]
        .std(numeric_only=True, ddof=1)
        .fillna(0)
        .to_dict(),
        "folds": fold_frame.to_dict(orient="records"),
    }
    if device is not None:
        summary["device"] = str(device)
    if extra:
        summary.update(extra)
    save_metrics_json(output, summary, filename=filename)
    return summary


def metrics_from_prediction_frame(
    prediction_frame: pd.DataFrame,
    label_column: str = "label",
    pred_column: str = "y_pred",
    prob_column: str = "fall_probability",
) -> Dict[str, float]:
    """从预测表中计算二分类指标。"""
    return compute_metrics(
        prediction_frame[label_column].to_numpy(),
        prediction_frame[pred_column].to_numpy(),
        prediction_frame[prob_column].to_numpy(),
    )


def classification_report_text(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> str:
    """生成统一格式的二分类 classification report。"""
    return classification_report(
        y_true,
        y_pred,
        labels=[0, 1],
        target_names=CLASS_NAMES,
        digits=4,
        zero_division=0,
    )


def save_confusion_matrix_csv(
    output: Path,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    filename: str = "confusion_matrix.csv",
) -> np.ndarray:
    """保存二分类混淆矩阵 CSV，并返回矩阵本身。"""
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    pd.DataFrame(
        cm,
        index=["true_ADL", "true_Fall"],
        columns=["pred_ADL", "pred_Fall"],
    ).to_csv(Path(output) / filename)
    return cm


def metrics_text_block(
    title: str,
    metrics: Dict[str, Any],
) -> str:
    """把一组指标转换成 metrics.txt 中的文本块。"""
    lines = [title]
    lines.extend(f"{key}: {value}" for key, value in metrics.items())
    return "\n".join(lines)


def save_experiment_metrics_text(
    output: Path,
    model: str,
    header: Sequence[str],
    fold_rows: Sequence[Dict[str, Any]] | pd.DataFrame,
    overall: Dict[str, Any],
    y_true: np.ndarray,
    y_pred: np.ndarray,
    confusion: Optional[np.ndarray] = None,
    filename: str = "metrics.txt",
    metric_keys: Sequence[str] = (
        "accuracy",
        "precision",
        "recall",
        "f1",
        "roc_auc",
        "tn",
        "fp",
        "fn",
        "tp",
    ),
    fold_extra_keys: Sequence[str] = (
        "train_videos",
        "test_videos",
        "train_windows",
        "test_windows",
        "best_epoch",
        "val_f1",
    ),
) -> str:
    """保存统一格式的 metrics.txt，并返回写入的文本。"""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    fold_frame = (
        fold_rows.copy()
        if isinstance(fold_rows, pd.DataFrame)
        else pd.DataFrame(list(fold_rows))
    )
    cm = confusion if confusion is not None else confusion_matrix(y_true, y_pred, labels=[0, 1])
    report = classification_report_text(y_true, y_pred)
    lines = ["=" * 80, f"MODEL: {model.upper()}", "=" * 80]
    lines.extend(str(item) for item in header)
    lines.append("")
    for _, row in fold_frame.iterrows():
        fold = row.get("fold", "?")
        total_folds = len(fold_frame)
        lines.append(f"[Fold {fold}/{total_folds}]")
        for key in fold_extra_keys:
            if key in fold_frame.columns:
                lines.append(f"{key}: {fmt_metric(row[key])}")
        for key in metric_keys:
            if key in fold_frame.columns:
                lines.append(f"{key}: {fmt_metric(row[key])}")
        lines.append("")
    lines.extend(["=" * 80, "OVERALL OUT-OF-FOLD METRICS", "=" * 80])
    lines.extend(f"{key}: {fmt_metric(overall[key])}" for key in metric_keys if key in overall)
    lines.extend(
        [
            "",
            "Classification report:",
            report,
            "Confusion matrix [[TN, FP], [FN, TP]]:",
            np.array2string(cm),
            "",
            "Fold mean +/- std:",
        ]
    )
    for key in ("accuracy", "precision", "recall", "f1", "roc_auc"):
        if key in fold_frame.columns:
            lines.append(
                f"{key}: {fold_frame[key].mean():.6f} +/- "
                f"{fold_frame[key].std(ddof=1):.6f}"
            )
    text = "\n".join(lines)
    (output / filename).write_text(text, encoding="utf-8")
    return text


def save_summary_metrics_text(
    output: Path,
    title: str,
    summary: Dict[str, Any],
    filename: str = "metrics.txt",
) -> str:
    """保存嵌套 summary 字典为简洁文本指标文件。"""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    lines = [title, "=" * 60]
    for key, value in summary.items():
        if isinstance(value, dict):
            lines.extend(["", str(key)])
            lines.extend(
                f"{inner_key}: {inner_value}"
                for inner_key, inner_value in value.items()
            )
        else:
            lines.append(f"{key}: {value}")
    text = "\n".join(lines) + "\n"
    (output / filename).write_text(text, encoding="utf-8")
    return text


def save_binary_classification_outputs(
    output: Path,
    prediction_frame: pd.DataFrame,
    label_column: str = "label",
    pred_column: str = "y_pred",
    prob_column: str = "fall_probability",
    prediction_filename: str = "predictions.csv",
    metrics_filename: str = "metrics.txt",
    confusion_filename: str = "confusion_matrix.csv",
    title: str = "WINDOW METRICS",
    extra_blocks: Optional[Sequence[Tuple[str, Dict[str, Any]]]] = None,
) -> Dict[str, float]:
    """保存预测表、混淆矩阵和指标文本，返回主指标。"""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    prediction_frame.to_csv(output / prediction_filename, index=False)
    y_true = prediction_frame[label_column].to_numpy()
    y_pred = prediction_frame[pred_column].to_numpy()
    y_prob = prediction_frame[prob_column].to_numpy()
    metrics = compute_metrics(y_true, y_pred, y_prob)
    save_confusion_matrix_csv(output, y_true, y_pred, confusion_filename)
    report = classification_report_text(y_true, y_pred)
    text_parts = [metrics_text_block(title, metrics)]
    if extra_blocks:
        for block_title, block_metrics in extra_blocks:
            text_parts.append(metrics_text_block(block_title, block_metrics))
    text_parts.append("Classification report:\n" + report)
    (output / metrics_filename).write_text(
        "\n\n".join(text_parts),
        encoding="utf-8",
    )
    return metrics


def add_binary_predictions(
    records: pd.DataFrame,
    indices: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    seed: Optional[int] = None,
    fold: Optional[int] = None,
    threshold: Optional[float] = None,
    label_column: str = "label",
) -> pd.DataFrame:
    """基于窗口记录和预测数组构建统一预测表。"""
    frame = records.iloc[indices].copy().reset_index(drop=True)
    frame[label_column] = y_true
    frame["y_pred"] = y_pred
    frame["fall_probability"] = y_prob
    if seed is not None:
        frame["seed"] = seed
    if fold is not None:
        frame["fold"] = fold
    if threshold is not None:
        frame["threshold"] = threshold
    return frame


def summarize_seed_metrics(seed_rows: Sequence[Dict[str, Any]]) -> pd.DataFrame:
    """把多 seed 指标整理成表格。"""
    if not seed_rows:
        raise ValueError("seed_rows 不能为空")
    return pd.DataFrame(seed_rows)


def seed_metric_summary(seed_frame: pd.DataFrame) -> Dict[str, Dict[str, float]]:
    """计算多 seed 指标均值和标准差。"""
    metric_only = seed_frame.drop(columns=["seed"], errors="ignore")
    return {
        "seed_metric_mean": metric_only.mean(numeric_only=True).to_dict(),
        "seed_metric_std": metric_only.std(numeric_only=True, ddof=1)
        .fillna(0)
        .to_dict(),
    }


def save_seed_summary_outputs(
    output: Path,
    prediction_frames: Sequence[pd.DataFrame],
    fold_rows: Sequence[Dict[str, Any]],
    seed_rows: Sequence[Dict[str, Any]],
    video_frames: Optional[Sequence[pd.DataFrame]] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """保存多 seed 实验的根目录汇总文件。"""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    prediction_frame = pd.concat(prediction_frames, ignore_index=True)
    prediction_frame.to_csv(output / "predictions.csv", index=False)
    fold_frame = pd.DataFrame(fold_rows)
    fold_frame.to_csv(output / "fold_metrics.csv", index=False)
    seed_frame = summarize_seed_metrics(seed_rows)
    seed_frame.to_csv(output / "seed_metrics.csv", index=False)
    if video_frames:
        pd.concat(video_frames, ignore_index=True).to_csv(
            output / "video_predictions.csv",
            index=False,
        )
    return prediction_frame, fold_frame, seed_frame
