#!/usr/bin/env python3
"""MLP baseline for window-based fall detection from BlazePose NPZ files.

Run from the fall_detection directory:
    uv run python src/experiment_mlp.py --data-root data/keypoints_normalized

Each 30-frame window of 33 x/y joints is flattened to 1980 inputs. Outer
cross-validation and the inner early-stopping split are both grouped by video.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import StratifiedGroupKFold
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from experiment_lstm import (
    choose_device,
    compute_metrics,
    fmt_metric,
    load_all_windows,
    seed_everything,
)


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
    窗口标签规则：
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

    records["window_label"] = new_y
    records["window_label_name"] = np.where(new_y == 1, "fall", "adl")

    print(
        "Final window labels: "
        f"windows={len(new_y)} | "
        f"ADL={int(np.sum(new_y == 0))} | "
        f"Fall={int(np.sum(new_y == 1))} | "
        f"skipped_transition_or_tie={skipped_transition_or_tie} | "
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
    仅用于 split stratification：
        adl-*  -> 0
        fall-* -> 1

    模型训练/评估仍使用窗口级 y。
    """
    labels = []
    for group in groups:
        sequence_name = canonical_sequence_name(group)
        if sequence_name is None:
            raise RuntimeError(f"无法解析视频类型：{group}")
        labels.append(1 if sequence_name.startswith("fall-") else 0)
    return np.asarray(labels, dtype=np.int64)


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
    split_y = video_type_labels(groups)

    unique_groups = np.unique(groups)
    group_type = {}
    for group in unique_groups:
        values = np.unique(split_y[groups == group])
        if len(values) != 1:
            raise RuntimeError(f"视频 {group} 出现多个视频类型")
        group_type[group] = int(values[0])

    class_group_counts = np.bincount(
        np.asarray(list(group_type.values()), dtype=np.int64),
        minlength=2,
    )
    n_splits = int(min(5, class_group_counts.min()))
    if n_splits < 2:
        raise RuntimeError(
            "内层验证至少需要每种视频类型各 2 个视频；"
            f"当前为 {class_group_counts.tolist()}"
        )

    splitter = StratifiedGroupKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=seed,
    )

    for fit_idx, val_idx in splitter.split(np.zeros(len(y)), split_y, groups):
        # early stopping 的 F1 需要 fit / val 中都有最终窗口类别
        if len(np.unique(y[fit_idx])) == 2 and len(np.unique(y[val_idx])) == 2:
            return fit_idx, val_idx

    raise ValueError(
        "Could not create a video-grouped validation split with both window classes"
    )


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

    x, y, groups, records = load_all_windows(
        args.data_root, args.window_size, args.stride,
        args.visibility_threshold, args.missing_mode, "xy66", None,
    )

    # 共享 loader 仍负责关键点过滤、插值和滑动窗口生成。
    # 它最先打印的 ADL/Fall 数量仍是“继承视频标签”的旧统计；
    # 下面重新按 UR-Fall 官方逐帧 CSV 生成最终窗口级标签。
    annotations = load_fall_frame_labels(FALL_ANNOTATION_CSV)
    x, y, groups, records = apply_urfall_window_labels(
        x,
        y,
        groups,
        records,
        args.window_size,
        annotations,
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
