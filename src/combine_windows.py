#!/usr/bin/env python3
"""Combine multiple prepared window caches into one training cache."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from experiments import common as experiment_common


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Combine prepared sliding-window NPZ caches."
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--input",
        action="append",
        nargs=2,
        metavar=("NAME", "CACHE"),
        required=True,
        help="Dataset/profile name and its prepared windows cache. Repeatable.",
    )
    return parser.parse_args()


def _jsonable_config(config: dict[str, Any]) -> dict[str, Any]:
    return json.loads(json.dumps(config, ensure_ascii=False, default=str))


def resample_time_axis(x: np.ndarray, target_frames: int) -> np.ndarray:
    """Linearly resample [N, T, D] windows to a shared temporal length."""
    if x.ndim != 3:
        raise ValueError(f"Expected window tensor [N, T, D], got {x.shape}")
    source_frames = int(x.shape[1])
    if source_frames == target_frames:
        return x.astype(np.float32, copy=False)
    source_grid = np.linspace(0.0, 1.0, source_frames)
    target_grid = np.linspace(0.0, 1.0, target_frames)
    # Move time to the last axis before flattening so every interpolation row
    # represents one feature from one window. Reshaping [N, T, D] directly to
    # [-1, T] would mix adjacent features instead of following them over time.
    features_by_time = x.transpose(0, 2, 1).reshape(-1, source_frames)
    resampled = np.empty(
        (features_by_time.shape[0], target_frames), dtype=np.float32
    )
    for row_index, feature in enumerate(features_by_time):
        resampled[row_index] = np.interp(target_grid, source_grid, feature)
    return resampled.reshape(
        x.shape[0], x.shape[2], target_frames
    ).transpose(0, 2, 1)


def main() -> None:
    args = parse_args()
    loaded = []
    for name, cache_path in args.input:
        loaded.append((name, *experiment_common.load_windows_cache(cache_path)))

    target_frames = max(int(x.shape[1]) for _, x, *_ in loaded)
    xs = []
    ys = []
    groups = []
    records = []
    configs = {}
    reference_joint_count = None
    reference_feature_dim = None

    for name, x, y, group, record, config in loaded:
        if reference_joint_count is None:
            reference_joint_count = config.get("joint_count")
            reference_feature_dim = config.get("feature_dim", x.shape[-1])
        elif x.shape[-1] != reference_feature_dim:
            raise ValueError(
                f"Cannot combine {name}: feature_dim={x.shape[-1]} does not "
                f"match {reference_feature_dim}."
            )
        if config.get("joint_count") != reference_joint_count:
            raise ValueError(
                f"Cannot combine {name}: joint_count={config.get('joint_count')} "
                f"does not match {reference_joint_count}."
            )
        if config.get("feature_dim", x.shape[-1]) != reference_feature_dim:
            raise ValueError(
                f"Cannot combine {name}: feature_dim={config.get('feature_dim')} "
                f"does not match {reference_feature_dim}."
            )
        original_frames = int(x.shape[1])
        x = resample_time_axis(x, target_frames)

        prefix = str(name)
        prefixed_groups = np.asarray([f"{prefix}::{item}" for item in group])
        record = record.copy()
        record.insert(0, "dataset", prefix)
        record["video_id"] = [f"{prefix}::{item}" for item in record["video_id"].astype(str)]

        xs.append(x)
        ys.append(y)
        groups.append(prefixed_groups)
        records.append(record)
        config = _jsonable_config(config)
        config["combined_original_window_frames"] = original_frames
        config["combined_resampled_window_frames"] = target_frames
        configs[prefix] = config

    combined_x = np.concatenate(xs, axis=0)
    combined_y = np.concatenate(ys, axis=0)
    combined_groups = np.concatenate(groups, axis=0)
    combined_records = pd.concat(records, ignore_index=True)
    config = {
        "combined": True,
        "datasets": list(configs),
        "source_configs": configs,
        "joint_count": reference_joint_count,
        "feature_dim": reference_feature_dim,
        "window_resampling": "linear_time_axis_to_max_source_window_frames",
        "target_window_frames": target_frames,
        "shape": list(combined_x.shape),
        "windows": int(len(combined_x)),
        "adl_windows": int((combined_y == 0).sum()),
        "fall_windows": int((combined_y == 1).sum()),
        "split_policy": (
            "Combined cache prefixes every group/video_id with dataset name. "
            "Cross-validation must keep these prefixed original videos disjoint."
        ),
    }
    experiment_common.save_windows_cache(
        args.output,
        combined_x,
        combined_y,
        combined_groups,
        combined_records,
        config,
    )
    print(
        f"Saved combined windows cache: {args.output} | "
        f"windows={len(combined_x)} | shape={combined_x.shape} | "
        f"ADL={int((combined_y == 0).sum())} | Fall={int((combined_y == 1).sum())}"
    )


if __name__ == "__main__":
    main()
