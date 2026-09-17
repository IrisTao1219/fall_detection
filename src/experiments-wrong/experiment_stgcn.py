#!/usr/bin/env python3
"""Ten-block ST-GCN experiment on the LSTM baseline's BlazePose windows.

Run from the fall_detection directory with
``uv run python src/experiment_stgcn.py --data-root data/keypoints_normalized``.
The shared loader keeps window extraction and filtering identical to the
current XY LSTM and MLP experiments.

The network uses four 64-channel, three 128-channel, and three 256-channel
spatiotemporal graph blocks. Each block has a temporal kernel of size 9,
a residual path, and dropout. Softmax is applied when obtaining probabilities;
training uses logits with CrossEntropyLoss.
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
STGCN_CHANNELS = (64,) * 4 + (128,) * 3 + (256,) * 3
TEMPORAL_KERNEL_SIZE = 9


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


def fmt_metric(v):
    if isinstance(v, (float, np.floating)) and np.isnan(v):
        return "nan"
    return f"{v:.6f}" if isinstance(v, (float, np.floating)) else str(v)


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
    parser.add_argument("--device", default="auto", help="auto / cpu / cuda / mps")
    return parser.parse_args()


def dataset_run_name(data_root: Path) -> str:
    """Keep raw/normalized output folders consistent with the LSTM experiment."""
    name = data_root.resolve().name.lower()
    return "stgcn_normalized" if "normalized" in name else "stgcn"


def main():
    args = parse_args()
    if min(args.window_size, args.stride, args.patience, args.epochs, args.batch_size) < 1 or args.folds < 2:
        raise ValueError("Window size, stride, patience, epochs, batch size must be positive; folds >= 2")
    if not 0 <= args.dropout < 1:
        raise ValueError("dropout must be in [0, 1)")

    # Match LSTM output layout: one dataset root -> one experiment directory.
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

    group_labels = (
        pd.DataFrame({"group": groups, "label": y})
        .groupby("group")["label"]
        .agg(lambda s: s.mode().iloc[0])
    )
    class_groups = np.bincount(group_labels.to_numpy(), minlength=2)
    if class_groups.min() < args.folds:
        raise ValueError(f"Need at least {args.folds} videos per class; found {class_groups.tolist()}")

    models_dir = output / "models"
    histories_dir = output / "histories"
    models_dir.mkdir(parents=True, exist_ok=True)
    histories_dir.mkdir(parents=True, exist_ok=True)

    # Save config in the same place/style as LSTM.
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
    metrics_lines.append("")

    for fold, (train_idx, test_idx) in enumerate(splitter.split(x, y, groups), 1):
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

        # Keep the same core fold_metrics.csv columns as LSTM.
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
        # Match LSTM history CSV structure: epoch + train_loss.
        pd.DataFrame(history)[["epoch", "train_loss"]].to_csv(
            histories_dir / f"fold_{fold}_history.csv", index=False
        )

        all_true.extend(y[test_idx].tolist())
        all_pred.extend(pred.tolist())
        all_prob.extend(prob.tolist())
        all_test_indices.extend(test_idx.tolist())

    # Overall out-of-fold metrics: same structure as LSTM.
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
