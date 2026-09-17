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
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import StratifiedGroupKFold, GroupShuffleSplit
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from experiment_lstm import (
    choose_device,
    compute_metrics,
    load_all_windows,
    seed_everything,
)


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
    # A group shuffle can occasionally leave a class out of validation on
    # small datasets. Try reproducible alternatives before failing clearly.
    splitter = GroupShuffleSplit(n_splits=100, test_size=0.2, random_state=seed)
    for fit_idx, val_idx in splitter.split(np.zeros(len(y)), y, groups):
        if len(np.unique(y[fit_idx])) == 2 and len(np.unique(y[val_idx])) == 2:
            return fit_idx, val_idx
    raise ValueError("Could not create a video-grouped validation split with both classes")


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

    x, y, groups, records = load_all_windows(
        args.data_root, args.window_size, args.stride,
        args.visibility_threshold, args.missing_mode, "xy66", None,
    )
    x = x.reshape(len(x), -1)
    group_labels = pd.DataFrame({"group": groups, "label": y}).groupby("group")["label"].agg(lambda s: s.mode().iloc[0])
    class_groups = np.bincount(group_labels.to_numpy(), minlength=2)
    if class_groups.min() < args.folds:
        raise ValueError(f"Need at least {args.folds} videos per class; found {class_groups.tolist()}")
    splitter = StratifiedGroupKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
    predictions, fold_rows, video_rows = [], [], []

    for fold, (train_idx, test_idx) in enumerate(splitter.split(x, y, groups), 1):
        if set(groups[train_idx]) & set(groups[test_idx]):
            raise RuntimeError("Training and test videos overlap")
        model, mean, std, best_epoch, val_f1, history, (fit_idx, val_idx) = train_fold(
            x[train_idx], y[train_idx], groups[train_idx], args, device, args.seed + fold
        )
        test_x = ((x[test_idx] - mean) / std).astype(np.float32)
        prob = probabilities(model, make_loader(test_x, y[test_idx], args.batch_size, False), device)
        pred = (prob >= 0.5).astype(np.int64)
        metrics = compute_metrics(y[test_idx], pred, prob)
        fold_rows.append({"fold": fold, "best_epoch": best_epoch, "val_f1": val_f1, **metrics})
        frame = records.iloc[test_idx].copy()
        frame["fold"], frame["fall_probability"], frame["y_pred"] = fold, prob, pred
        predictions.append(frame)
        torch.save({
            "state_dict": model.state_dict(), "input_dim": x.shape[1],
            "hidden": args.hidden, "dropout": args.dropout,
            "feature_mean": mean, "feature_std": std,
            "best_epoch": best_epoch,
        }, output / f"fold_{fold}.pt")
        pd.DataFrame(history).to_csv(output / f"fold_{fold}_history.csv", index=False)
        print(f"Fold {fold}: test F1={metrics['f1']:.4f}, recall={metrics['recall']:.4f}; best epoch={best_epoch}")

    window_df = pd.concat(predictions, ignore_index=True)
    window_df.to_csv(output / "window_predictions.csv", index=False)
    # One score per video avoids weighting long videos more heavily in the main result.
    video_df = window_df.groupby("video_id", as_index=False).agg(
        fold=("fold", "first"), label=("label", "first"),
        fall_probability=("fall_probability", "mean"), windows=("label", "size"),
    )
    video_df["y_pred"] = (video_df["fall_probability"] >= 0.5).astype(int)
    video_df.to_csv(output / "video_predictions.csv", index=False)
    for fold, part in video_df.groupby("fold"):
        video_rows.append({"fold": fold, **compute_metrics(part.label.to_numpy(), part.y_pred.to_numpy(), part.fall_probability.to_numpy())})
    pd.DataFrame(fold_rows).to_csv(output / "fold_metrics.csv", index=False)
    pd.DataFrame(video_rows).to_csv(output / "video_fold_metrics.csv", index=False)

    window_metrics = compute_metrics(window_df.label.to_numpy(), window_df.y_pred.to_numpy(), window_df.fall_probability.to_numpy())
    video_metrics = compute_metrics(video_df.label.to_numpy(), video_df.y_pred.to_numpy(), video_df.fall_probability.to_numpy())
    report = classification_report(video_df.label, video_df.y_pred, labels=[0, 1], target_names=["ADL", "Fall"], zero_division=0)
    summary = {"video": video_metrics, "window": window_metrics, "video_report": report,
               "video_confusion_matrix": confusion_matrix(video_df.label, video_df.y_pred, labels=[0, 1]).tolist()}
    (output / "metrics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    config = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    (output / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    print(f"Video-level F1={video_metrics['f1']:.4f}, recall={video_metrics['recall']:.4f}")
    print(f"Results: {output}")


if __name__ == "__main__":
    main()
