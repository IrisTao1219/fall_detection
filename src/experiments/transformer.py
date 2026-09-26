#!/usr/bin/env python3
"""基于姿态关键点序列的 Temporal Transformer 跌倒检测实验。

数据加载器可接收 BlazePose、TransPose、ViTPose 或其他姿态估计器输出的
关键点，只要每个 NPZ 文件遵守项目约定即可。不同实验可以使用不同关节数，
但同一次运行的数据目录内必须保持一致。

Example:
    uv run python src/experiment_transformer.py \
        --data-root data/keypoints \
        --run-name transformer_blazepose_urfall
"""

from __future__ import annotations

import argparse
import time
from dataclasses import asdict
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import StratifiedGroupKFold
from torch import nn

import common as experiment_common
from common import (
    WindowRecord,
    aggregate_video_predictions,
    canonical_sequence_name,
    choose_device,
    compute_metrics,
    discover_npz_files,
    fit_scaler,
    inner_group_split_by_video_type,
    load_fall_frame_labels,
    make_loader,
    make_seed_output_dirs,
    make_urfall_windows,
    normalize_label,
    probabilities,
    seed_everything,
    save_binary_classification_outputs,
    save_experiment_config,
    save_metrics_json,
    save_seed_summary_outputs,
    save_summary_metrics_text,
    seed_metric_summary,
    tune_threshold,
    video_type_labels,
)


class TemporalTransformer(nn.Module):
    """用于固定长度姿态序列的轻量 Transformer 编码器。"""

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



def train_fold(x, y, groups, args, device, seed):
    fit_idx, val_idx = inner_group_split_by_video_type(y, groups, seed)
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
    best_score, best_loss, best_epoch, best_state = -1.0, float("inf"), 0, None
    best_threshold, best_val_metrics = args.threshold, {}
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
        val_threshold, val_metrics = tune_threshold(
            y[val_idx], val_prob, args.metric_objective, args.min_recall
        )
        val_score = val_metrics[args.metric_objective]
        scheduler.step(val_score)
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_threshold": val_threshold,
                **{f"val_{key}": value for key, value in val_metrics.items()},
                "learning_rate": optimizer.param_groups[0]["lr"],
            }
        )
        if val_score > best_score or (val_score == best_score and train_loss < best_loss):
            best_score, best_loss, best_epoch = val_score, train_loss, epoch
            best_threshold = val_threshold
            best_val_metrics = val_metrics
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
        best_score,
        best_threshold,
        best_val_metrics,
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
    parser.add_argument("--windows-cache", type=Path, required=True)
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
        "--normalization", choices=("minmax", "zscore", "none"), default="zscore"
    )
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--feedforward-dim", type=int, default=512)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--gradient-clip", type=float, default=1.0)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument(
        "--metric-objective",
        choices=("accuracy", "f1", "precision", "recall"),
        default="accuracy",
        help="Validation metric used for early stopping and threshold tuning.",
    )
    parser.add_argument(
        "--min-recall",
        type=float,
        default=0.70,
        help="Minimum validation recall required during threshold tuning.",
    )
    parser.add_argument(
        "--video-aggregation",
        choices=("mean", "max", "topk_mean"),
        default="topk_mean",
    )
    parser.add_argument("--video-topk", type=int, default=5)
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
    if not 0.0 <= args.min_recall <= 1.0:
        raise ValueError("min-recall must be between 0 and 1")
    if args.video_topk < 1:
        raise ValueError("video-topk must be positive")

    device = choose_device(args.device)
    seed_everything(args.split_seed)
    x, y, groups, records, cache_config = experiment_common.load_windows_cache(
        args.windows_cache
    )
    joint_count = int(cache_config.get("joint_count", x.shape[-1] // 2))
    # 一个 fall 视频现在可能同时包含 ADL/Fall 窗口，因此外层 fold 按原始
    # UR-Fall 视频类型分层，训练和评估仍使用窗口级 y。
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
        seed_dir, model_dir, history_dir = make_seed_output_dirs(output, seed)
        seed_predictions = []
        for fold, (train_idx, test_idx) in enumerate(split_indices, 1):
            (
                model,
                offset,
                scale,
                best_epoch,
                val_score,
                tuned_threshold,
                best_val_metrics,
                history,
                _,
                seconds,
            ) = train_fold(
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
            pred = (prob >= tuned_threshold).astype(np.int64)
            metrics = compute_metrics(y[test_idx], pred, prob)
            all_fold_rows.append(
                {
                    "seed": seed,
                    "fold": fold,
                    "best_epoch": best_epoch,
                    "val_threshold": tuned_threshold,
                    f"val_{args.metric_objective}": val_score,
                    **{f"best_val_{key}": value for key, value in best_val_metrics.items()},
                    "train_seconds": seconds,
                    **metrics,
                }
            )
            frame = records.iloc[test_idx].copy()
            frame["seed"] = seed
            frame["fold"] = fold
            frame["fall_probability"] = prob
            frame["threshold"] = tuned_threshold
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
                    "threshold": tuned_threshold,
                    "args": _json_args(args),
                },
                model_dir / f"fold_{fold}.pt",
            )
            print(
                f"Seed {seed} fold {fold}: F1={metrics['f1']:.4f}, "
                f"accuracy={metrics['accuracy']:.4f}, recall={metrics['recall']:.4f}, "
                f"threshold={tuned_threshold:.3f}, best_epoch={best_epoch}"
            )

        seed_frame = pd.concat(seed_predictions, ignore_index=True)
        all_predictions.append(seed_frame)
        video_frame = aggregate_video_predictions(
            seed_frame, args.video_aggregation, args.video_topk
        )
        video_frame["y_pred"] = (
            video_frame["fall_probability"] >= video_frame["threshold"]
        ).astype(int)
        video_frame.to_csv(seed_dir / "video_predictions.csv", index=False)
        all_video_predictions.append(video_frame)
        window_metrics = _metrics_frame(seed_frame)
        video_metrics = _metrics_frame(video_frame)
        save_binary_classification_outputs(
            seed_dir,
            seed_frame,
            extra_blocks=[("VIDEO METRICS", video_metrics)],
        )
        seed_rows.append(
            {
                "seed": seed,
                **{f"window_{key}": value for key, value in window_metrics.items()},
                **{f"video_{key}": value for key, value in video_metrics.items()},
            }
        )

    _, _, seed_frame = save_seed_summary_outputs(
        output,
        all_predictions,
        all_fold_rows,
        seed_rows,
        all_video_predictions,
    )

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
        **seed_metric_summary(seed_frame),
    }
    save_metrics_json(output, summary)
    save_summary_metrics_text(output, "TRANSFORMER SUMMARY", summary)
    save_experiment_config(output, args, {"output_dir": str(output)})
    print(f"Results: {output}")


if __name__ == "__main__":
    main()
