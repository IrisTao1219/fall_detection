#!/usr/bin/env python3
"""Generate the shared sliding-window dataset used by all experiments."""

from __future__ import annotations

import argparse
from pathlib import Path


try:
    from .experiments import common as experiment_common
except ImportError:
    import common as experiment_common


# ============================================================
# UR-Fall 时间参数
# ============================================================

# UR-Fall RGB 视频帧率
URFALL_FPS = 30.0

# 一个窗口观察 2 秒
WINDOW_SECONDS = 2.0

# 每隔 0.5 秒产生一个新窗口
STRIDE_SECONDS = 0.5

# 30 FPS × 2 s = 60 frames
DEFAULT_WINDOW_SIZE = int(
    round(URFALL_FPS * WINDOW_SECONDS)
)

# 30 FPS × 0.5 s = 15 frames
DEFAULT_STRIDE = int(
    round(URFALL_FPS * STRIDE_SECONDS)
)


# ============================================================
# 命令行参数
# ============================================================

def parse_args() -> argparse.Namespace:

    parser = argparse.ArgumentParser(
        description=(
            "Prepare shared UR-Fall sliding-window cache "
            "for Fall Event Detection."
        )
    )

    parser.add_argument(
        "--data-root",
        type=Path,
        required=True,
        help="Directory containing UR-Fall keypoint NPZ files.",
    )

    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output sliding-window cache NPZ.",
    )

    parser.add_argument(
        "--annotation-csv",
        type=Path,
        default=Path(
            "data/urfall-cam0-falls.csv"
        ),
        help="UR-Fall official frame-level annotation CSV.",
    )

    # ========================================================
    # 新窗口设置
    #
    # UR-Fall = 30 FPS
    #
    # 60 frames = 2.0 seconds
    # 15 frames = 0.5 seconds
    # ========================================================

    parser.add_argument(
        "--window-size",
        type=int,
        default=DEFAULT_WINDOW_SIZE,
        help=(
            "Sliding-window length in frames. "
            f"Default: {DEFAULT_WINDOW_SIZE} "
            f"({WINDOW_SECONDS:.1f}s at {URFALL_FPS:.0f} FPS)."
        ),
    )

    parser.add_argument(
        "--stride",
        type=int,
        default=DEFAULT_STRIDE,
        help=(
            "Sliding-window stride in frames. "
            f"Default: {DEFAULT_STRIDE} "
            f"({STRIDE_SECONDS:.1f}s at {URFALL_FPS:.0f} FPS)."
        ),
    )

    parser.add_argument(
        "--missing-mode",
        choices=(
            "zero",
            "interp",
        ),
        default="interp",
    )

    parser.add_argument(
        "--visibility-threshold",
        type=float,
        default=0.3,
    )

    parser.add_argument(
        "--feature-mode",
        choices=(
            "xy",
            "xy66",
            "xy_flat",
            "joints",
            "joints16",
        ),
        default="xy66",
    )

    parser.add_argument(
        "--joint-indices",
        default=None,
    )

    parser.add_argument(
        "--min-valid-frames",
        type=int,
        default=1,
    )

    parser.add_argument(
        "--keypoint-adapter",
        choices=sorted(
            experiment_common.KEYPOINT_ADAPTERS
        ),
        default="blazepose",
        help=(
            "How to interpret keypoints/confidence "
            "fields in each NPZ."
        ),
    )

    parser.add_argument(
        "--confidence-index",
        type=int,
        default=None,
        help=(
            "Override adapter confidence channel index; "
            "negative values count from the end."
        ),
    )

    return parser.parse_args()


# ============================================================
# 解析关节索引
# ============================================================

def parse_joint_indices(
    text: str | None
) -> list[int] | None:

    if text is None or not text.strip():
        return None

    return [
        int(
            item.strip()
        )
        for item in text.split(",")
        if item.strip()
    ]


# ============================================================
# main
# ============================================================

def main() -> None:

    args = parse_args()

    # ========================================================
    # 参数检查
    # ========================================================

    if args.window_size <= 0:

        raise ValueError(
            "--window-size 必须大于 0"
        )

    if args.stride <= 0:

        raise ValueError(
            "--stride 必须大于 0"
        )

    if (
        args.min_valid_frames
        > args.window_size
    ):

        raise ValueError(
            "--min-valid-frames "
            "不能大于 --window-size"
        )

    joint_indices = (
        parse_joint_indices(
            args.joint_indices
        )
    )

    # ========================================================
    # 生成滑动窗口
    #
    # 真正的窗口标签逻辑在 common.py：
    #
    # UR-Fall 官方标签：
    #
    # -1 = normal
    #  0 = falling transition / Fall Event
    #  1 = post-fall / lying
    #
    # 新逻辑：
    #
    # 窗口中心处于 label=0
    #       -> Fall
    #
    # 窗口完全没有 label=0
    #       -> Non-fall
    #
    # 窗口与 label=0 有交集，
    # 但中心不处于 label=0
    #       -> Ignore
    #
    # 不再进行多数投票。
    # ========================================================

    (
        x,
        y,
        groups,
        records,
        metadata,
    ) = experiment_common.load_sequence_windows(

        data_root=args.data_root,

        window_size=args.window_size,

        stride=args.stride,

        visibility_threshold=(
            args.visibility_threshold
        ),

        missing_mode=(
            args.missing_mode
        ),

        feature_mode=(
            args.feature_mode
        ),

        joint_indices=(
            joint_indices
        ),

        min_valid_frames=(
            args.min_valid_frames
        ),

        annotation_csv=(
            args.annotation_csv
        ),

        keypoint_adapter=(
            args.keypoint_adapter
        ),

        confidence_index=(
            args.confidence_index
        ),
    )

    # ========================================================
    # 保存配置
    # ========================================================

    config = {

        "dataset":
            "UR-Fall",

        "data_root":
            str(
                args.data_root
            ),

        "annotation_csv":
            str(
                args.annotation_csv
            ),

        # ----------------------------------------------------
        # 时间信息
        # ----------------------------------------------------

        "fps":
            URFALL_FPS,

        "window_seconds":
            args.window_size
            / URFALL_FPS,

        "stride_seconds":
            args.stride
            / URFALL_FPS,

        # ----------------------------------------------------
        # 窗口参数
        # ----------------------------------------------------

        "window_size":
            args.window_size,

        "stride":
            args.stride,

        # ----------------------------------------------------
        # 标签规则
        # ----------------------------------------------------

        "label_rule":
            (
                "Fall if window center is inside "
                "UR-Fall transition label 0; "
                "Non-fall if window has no transition "
                "frames; ambiguous boundary windows ignored."
            ),

        # ----------------------------------------------------
        # 关键点设置
        # ----------------------------------------------------

        "keypoint_adapter":
            args.keypoint_adapter,

        "resolved_keypoint_adapter":
            metadata[
                "keypoint_adapter"
            ],

        "confidence_index":
            args.confidence_index,

        "missing_mode":
            args.missing_mode,

        "visibility_threshold":
            args.visibility_threshold,

        "feature_mode":
            args.feature_mode,

        "joint_indices":
            joint_indices,

        "min_valid_frames":
            args.min_valid_frames,

        "joint_count":
            metadata[
                "joint_count"
            ],

        "feature_dim":
            metadata[
                "feature_dim"
            ],

        # ----------------------------------------------------
        # 数据统计
        # ----------------------------------------------------

        "shape":
            list(
                x.shape
            ),

        "windows":
            int(
                len(x)
            ),

        "adl_windows":
            int(
                (
                    y == 0
                ).sum()
            ),

        "fall_windows":
            int(
                (
                    y == 1
                ).sum()
            ),
    }

    # ========================================================
    # 保存窗口缓存
    # ========================================================

    experiment_common.save_windows_cache(

        args.output,

        x,

        y,

        groups,

        records,

        config,
    )

    # ========================================================
    # 输出
    # ========================================================

    print()
    print("=" * 70)

    print(
        f"Saved windows cache: "
        f"{args.output}"
    )

    print(
        f"Window: "
        f"{args.window_size} frames "
        f"= "
        f"{args.window_size / URFALL_FPS:.2f}s"
    )

    print(
        f"Stride: "
        f"{args.stride} frames "
        f"= "
        f"{args.stride / URFALL_FPS:.2f}s"
    )

    print(
        f"windows={len(x)} | "
        f"shape={x.shape} | "
        f"Non-fall={config['adl_windows']} | "
        f"Fall={config['fall_windows']}"
    )

    print("=" * 70)


# ============================================================
# 程序入口
# ============================================================

if __name__ == "__main__":
    main()