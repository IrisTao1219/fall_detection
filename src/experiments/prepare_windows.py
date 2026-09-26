#!/usr/bin/env python3
"""Generate the shared sliding-window dataset used by all experiments."""

from __future__ import annotations

import argparse
from pathlib import Path

try:
    from .experiments import common as experiment_common
except ImportError:
    import common as experiment_common


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare shared UR-Fall sliding-window cache."
    )
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--annotation-csv", type=Path, default=Path("data/urfall-cam0-falls.csv"))
    parser.add_argument("--window-size", type=int, default=30)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--missing-mode", choices=("zero", "interp"), default="interp")
    parser.add_argument("--visibility-threshold", type=float, default=0.3)
    parser.add_argument(
        "--feature-mode",
        choices=("xy", "xy66", "xy_flat", "joints", "joints16"),
        default="xy66",
    )
    parser.add_argument("--joint-indices", default=None)
    parser.add_argument("--min-valid-frames", type=int, default=1)
    parser.add_argument(
        "--keypoint-adapter",
        choices=sorted(experiment_common.KEYPOINT_ADAPTERS),
        default="blazepose",
        help="How to interpret keypoints/confidence fields in each NPZ.",
    )
    parser.add_argument(
        "--confidence-index",
        type=int,
        default=None,
        help="Override adapter confidence channel index; negative values count from the end.",
    )
    return parser.parse_args()


def parse_joint_indices(text: str | None) -> list[int] | None:
    if text is None or not text.strip():
        return None
    return [int(item.strip()) for item in text.split(",") if item.strip()]


def main() -> None:
    args = parse_args()
    joint_indices = parse_joint_indices(args.joint_indices)
    x, y, groups, records, metadata = experiment_common.load_sequence_windows(
        data_root=args.data_root,
        window_size=args.window_size,
        stride=args.stride,
        visibility_threshold=args.visibility_threshold,
        missing_mode=args.missing_mode,
        feature_mode=args.feature_mode,
        joint_indices=joint_indices,
        min_valid_frames=args.min_valid_frames,
        annotation_csv=args.annotation_csv,
        keypoint_adapter=args.keypoint_adapter,
        confidence_index=args.confidence_index,
    )
    config = {
        "data_root": str(args.data_root),
        "annotation_csv": str(args.annotation_csv),
        "keypoint_adapter": args.keypoint_adapter,
        "resolved_keypoint_adapter": metadata["keypoint_adapter"],
        "confidence_index": args.confidence_index,
        "window_size": args.window_size,
        "stride": args.stride,
        "missing_mode": args.missing_mode,
        "visibility_threshold": args.visibility_threshold,
        "feature_mode": args.feature_mode,
        "joint_indices": joint_indices,
        "min_valid_frames": args.min_valid_frames,
        "joint_count": metadata["joint_count"],
        "feature_dim": metadata["feature_dim"],
        "shape": list(x.shape),
        "windows": int(len(x)),
        "adl_windows": int((y == 0).sum()),
        "fall_windows": int((y == 1).sum()),
    }
    experiment_common.save_windows_cache(args.output, x, y, groups, records, config)
    print(f"Saved windows cache: {args.output}")
    print(
        f"windows={len(x)} | shape={x.shape} | "
        f"ADL={config['adl_windows']} | Fall={config['fall_windows']}"
    )


if __name__ == "__main__":
    main()
