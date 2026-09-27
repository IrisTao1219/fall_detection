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


def main() -> None:
    args = parse_args()
    xs = []
    ys = []
    groups = []
    records = []
    configs = {}
    reference_shape = None
    reference_joint_count = None
    reference_feature_dim = None

    for name, cache_path in args.input:
        x, y, group, record, config = experiment_common.load_windows_cache(cache_path)
        if reference_shape is None:
            reference_shape = x.shape[1:]
            reference_joint_count = config.get("joint_count")
            reference_feature_dim = config.get("feature_dim", x.shape[-1])
        elif x.shape[1:] != reference_shape:
            raise ValueError(
                f"Cannot combine {name}: window shape {x.shape[1:]} does not "
                f"match {reference_shape}. Use a common --window-size and "
                "--feature-mode before combining."
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

        prefix = str(name)
        prefixed_groups = np.asarray([f"{prefix}::{item}" for item in group])
        record = record.copy()
        record.insert(0, "dataset", prefix)
        record["video_id"] = [f"{prefix}::{item}" for item in record["video_id"].astype(str)]

        xs.append(x)
        ys.append(y)
        groups.append(prefixed_groups)
        records.append(record)
        configs[prefix] = _jsonable_config(config)

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
