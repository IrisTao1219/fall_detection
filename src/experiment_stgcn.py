#!/usr/bin/env python3
"""ST-GCN experiment on the same BlazePose windows as the LSTM baseline.

Run from the fall_detection directory with
``uv run python src/experiment_stgcn.py --data-root data/keypoints_normalized``.
The shared loader keeps window extraction and filtering identical to the
current XY LSTM and MLP experiments.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import StratifiedGroupKFold
from torch import nn

from experiment_lstm import choose_device, compute_metrics, load_all_windows, seed_everything
from experiment_mlp import inner_split, make_loader, probabilities


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


def make_adjacency() -> torch.Tensor:
    adjacency = np.eye(33, dtype=np.float32)
    for a, b in POSE_EDGES:
        adjacency[a, b] = adjacency[b, a] = 1.0
    degree = adjacency.sum(axis=1)
    adjacency /= np.sqrt(degree[:, None] * degree[None, :])
    return torch.from_numpy(adjacency)


class STGCNBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int):
        super().__init__()
        self.spatial = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.temporal = nn.Conv2d(
            out_channels, out_channels, kernel_size=(9, 1),
            stride=(stride, 1), padding=(4, 0),
        )
        self.norm = nn.BatchNorm2d(out_channels)
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
        return self.activation(self.norm(self.temporal(self.spatial(spatial))) + self.residual(x))


class STGCN(nn.Module):
    def __init__(self, dropout: float = 0.3):
        super().__init__()
        self.register_buffer("adjacency", make_adjacency())
        self.blocks = nn.ModuleList((
            STGCNBlock(2, 32, 1), STGCNBlock(32, 64, 2),
            STGCNBlock(64, 128, 2),
        ))
        self.classifier = nn.Sequential(nn.Dropout(dropout), nn.Linear(128, 2))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Loader supplies [B,T,66]; graph layers use [B,C,T,V].
        x = x.reshape(x.shape[0], x.shape[1], 33, 2).permute(0, 3, 1, 2)
        for block in self.blocks:
            x = block(x, self.adjacency)
        return self.classifier(x.mean(dim=(2, 3)))


def standardize(x: np.ndarray, fit_idx: np.ndarray):
    # One mean/std for each coordinate of each joint, using fit videos only.
    mean = x[fit_idx].mean(axis=(0, 1), keepdims=True)
    std = x[fit_idx].std(axis=(0, 1), keepdims=True)
    std[std < 1e-6] = 1.0
    return mean.astype(np.float32), std.astype(np.float32)


def train_fold(x, y, groups, args, device, seed):
    fit_idx, val_idx = inner_split(y, groups, seed)
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


def parse_args():
    parser = argparse.ArgumentParser(description="Video-grouped ST-GCN fall detection experiment")
    parser.add_argument("--data-root", type=Path, default=Path("data/keypoints_normalized"))
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
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def main():
    args = parse_args()
    if min(args.window_size, args.stride, args.patience, args.epochs, args.batch_size) < 1 or args.folds < 2:
        raise ValueError("Window size, stride, patience, epochs, batch size must be positive; folds >= 2")
    seed_everything(args.seed)
    device = choose_device(args.device)
    x, y, groups, records = load_all_windows(
        args.data_root, args.window_size, args.stride,
        args.visibility_threshold, args.missing_mode, "xy66", None,
    )
    group_labels = pd.DataFrame({"group": groups, "label": y}).groupby("group")["label"].agg(lambda s: s.mode().iloc[0])
    class_groups = np.bincount(group_labels.to_numpy(), minlength=2)
    if class_groups.min() < args.folds:
        raise ValueError(f"Need at least {args.folds} videos per class; found {class_groups.tolist()}")
    run_name = "stgcn_normalized" if "normalized" in args.data_root.resolve().name.lower() else "stgcn"
    output = args.output_root / run_name
    output.mkdir(parents=True, exist_ok=True)
    models_dir = output / "models"
    histories_dir = output / "histories"
    models_dir.mkdir(parents=True, exist_ok=True)
    histories_dir.mkdir(parents=True, exist_ok=True)
    splitter = StratifiedGroupKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
    predictions, fold_rows, split_rows = [], [], []
    for fold, (train_idx, test_idx) in enumerate(splitter.split(x, y, groups), 1):
        if set(groups[train_idx]) & set(groups[test_idx]):
            raise RuntimeError("Training and test videos overlap")
        model, mean, std, best_epoch, val_f1, history, (fit_idx, val_idx), seconds = train_fold(
            x[train_idx], y[train_idx], groups[train_idx], args, device, args.seed + fold,
        )
        test_x = ((x[test_idx] - mean) / std).astype(np.float32)
        prob = probabilities(model, make_loader(test_x, y[test_idx], args.batch_size, False), device)
        pred = (prob >= 0.5).astype(np.int64)
        metrics = compute_metrics(y[test_idx], pred, prob)
        fold_rows.append({"fold": fold, "best_epoch": best_epoch, "val_f1": val_f1,
                          "train_seconds": seconds, **metrics})
        frame = records.iloc[test_idx].copy()
        frame["fold"], frame["fall_probability"], frame["y_pred"] = fold, prob, pred
        predictions.append(frame)
        for role, indices in (("train", train_idx[fit_idx]), ("val", train_idx[val_idx]), ("outer_test", test_idx)):
            for video_id in np.unique(groups[indices]):
                split_rows.append({"fold": fold, "video_id": video_id, "role": role})
        torch.save({"state_dict": model.state_dict(), "feature_mean": mean, "feature_std": std,
                    "best_epoch": best_epoch, "dropout": args.dropout}, models_dir / f"fold_{fold}.pt")
        pd.DataFrame(history).to_csv(histories_dir / f"fold_{fold}_history.csv", index=False)
        print(f"Fold {fold}: F1={metrics['f1']:.4f}, recall={metrics['recall']:.4f}, best epoch={best_epoch}")

    window_df = pd.concat(predictions, ignore_index=True)
    window_df.to_csv(output / "predictions.csv", index=False)
    video_df = window_df.groupby("video_id", as_index=False).agg(
        fold=("fold", "first"), label=("label", "first"),
        fall_probability=("fall_probability", "mean"), windows=("label", "size"),
    )
    video_df["y_pred"] = (video_df.fall_probability >= 0.5).astype(int)
    video_df.to_csv(output / "video_predictions.csv", index=False)
    pd.DataFrame(fold_rows).to_csv(output / "fold_metrics.csv", index=False)
    pd.DataFrame(split_rows).to_csv(output / "splits.csv", index=False)
    video_fold_rows = [
        {"fold": fold, **compute_metrics(part.label.to_numpy(), part.y_pred.to_numpy(), part.fall_probability.to_numpy())}
        for fold, part in video_df.groupby("fold")
    ]
    pd.DataFrame(video_fold_rows).to_csv(output / "video_fold_metrics.csv", index=False)
    window_metrics = compute_metrics(window_df.label.to_numpy(), window_df.y_pred.to_numpy(), window_df.fall_probability.to_numpy())
    video_metrics = compute_metrics(video_df.label.to_numpy(), video_df.y_pred.to_numpy(), video_df.fall_probability.to_numpy())
    summary = {"video": video_metrics, "window": window_metrics,
               "video_report": classification_report(video_df.label, video_df.y_pred, labels=[0, 1],
                                                       target_names=["ADL", "Fall"], zero_division=0),
               "video_confusion_matrix": confusion_matrix(video_df.label, video_df.y_pred, labels=[0, 1]).tolist(),
               "parameter_count": sum(p.numel() for p in model.parameters()), "device": str(device)}
    (output / "metrics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    pd.DataFrame(
        confusion_matrix(window_df.label, window_df.y_pred, labels=[0, 1]),
        index=["true_ADL", "true_Fall"],
        columns=["pred_ADL", "pred_Fall"],
    ).to_csv(output / "confusion_matrix.csv")
    report = classification_report(
        window_df.label, window_df.y_pred, labels=[0, 1],
        target_names=["ADL", "Fall"], zero_division=0,
    )
    metric_lines = ["OVERALL OUT-OF-FOLD WINDOW METRICS"]
    metric_lines.extend(f"{key}: {value}" for key, value in window_metrics.items())
    metric_lines.extend(["", "Classification report:", report])
    (output / "metrics.txt").write_text("\n".join(metric_lines), encoding="utf-8")
    config = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    (output / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    print(f"Video-level F1={video_metrics['f1']:.4f}, recall={video_metrics['recall']:.4f}")
    print(f"Results: {output}")


if __name__ == "__main__":
    main()
