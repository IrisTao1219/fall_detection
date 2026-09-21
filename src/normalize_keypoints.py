from pathlib import Path

import numpy as np
from tqdm import tqdm


# ============================================================
# 1. 配置
# ============================================================

# 原始 keypoints
INPUT_ROOT = Path("data/keypoints")

# 归一化后的 keypoints
# 文件结构和 INPUT_ROOT 完全一致
OUTPUT_ROOT = Path("data/keypoints_normalized")

# BlazePose visibility 阈值：
# 仅用于选择“可靠的身体锚点/尺度”，不再用于把普通关键点额外置 NaN。
VISIBILITY_THRESHOLD = 0.3

# 数值稳定性
EPS = 1e-6
MIN_SCALE = 1e-3

# 极端坐标裁剪范围
CLIP_MIN = -5.0
CLIP_MAX = 5.0

# 计算视频级稳定尺度时，至少希望有这么多可靠帧。
# 少于该值仍可计算中位数，但会在统计中体现。
MIN_SCALE_FRAMES = 3


# ============================================================
# 2. BlazePose 关键点编号
# ============================================================

LEFT_SHOULDER = 11
RIGHT_SHOULDER = 12
LEFT_HIP = 23
RIGHT_HIP = 24


# ============================================================
# 3. 基础工具
# ============================================================

def is_xy_finite(frame, index):
    """关键点 x/y 是否为有限值。"""
    return bool(np.all(np.isfinite(frame[index, :2])))


def is_landmark_reliable(frame, index):
    """
    判断关键点是否足够可靠，可用于计算髋中心或人体尺度。

    注意：这个判断只用于“锚点/尺度估计”，不会把低 visibility 的
    普通关键点自动删除，从而保证 raw 与 normalized 的比较尽量公平。
    """
    if not is_xy_finite(frame, index):
        return False

    visibility = frame[index, 3]
    return bool(
        np.isfinite(visibility)
        and visibility >= VISIBILITY_THRESHOLD
    )


def midpoint_xy(frame, index1, index2):
    """返回两个关键点的二维中点。"""
    return (
        frame[index1, :2].astype(np.float64)
        + frame[index2, :2].astype(np.float64)
    ) / 2.0


def robust_median(values):
    """
    对尺度序列做稳健中位数估计。

    先用 MAD 去除明显离群值，再取中位数；如果 MAD 退化，直接取中位数。
    """
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    values = values[values >= MIN_SCALE]

    if values.size == 0:
        return None, 0, 0

    median = float(np.median(values))
    mad = float(np.median(np.abs(values - median)))

    if mad <= EPS:
        return median, int(values.size), int(values.size)

    # 1.4826 * MAD 约等于高斯分布下的标准差估计。
    robust_sigma = 1.4826 * mad
    keep = np.abs(values - median) <= 3.0 * robust_sigma
    filtered = values[keep]

    if filtered.size == 0:
        filtered = values

    return (
        float(np.median(filtered)),
        int(values.size),
        int(filtered.size),
    )


# ============================================================
# 4. 为整段视频估计髋中心轨迹
# ============================================================

def estimate_hip_centers(keypoints, valid_mask):
    """
    返回：
        hip_centers: [T, 2]
        center_available: [T]
        center_interpolated: [T]
        center_source: str

    策略：
    1. 优先使用“左右髋都可靠”的帧计算髋中心；
    2. 对中间缺失帧做时间线性插值；
    3. 对视频开头/结尾缺失帧使用最近可靠中心；
    4. 如果整段视频没有任何高 visibility 的双髋帧，
       再退化为“左右髋 x/y 有限”即可计算中心；
    5. 原始 valid_mask=False 的帧最终仍保持无效。

    这样不会因为某一帧一个髋 visibility 短暂下降，就直接把整帧丢掉。
    """
    num_frames = keypoints.shape[0]
    hip_centers = np.full((num_frames, 2), np.nan, dtype=np.float64)
    reliable = np.zeros(num_frames, dtype=bool)

    # --------------------------------------------------------
    # 第一遍：高可靠双髋
    # --------------------------------------------------------
    for t in range(num_frames):
        if not valid_mask[t]:
            continue

        frame = keypoints[t]
        if (
            is_landmark_reliable(frame, LEFT_HIP)
            and is_landmark_reliable(frame, RIGHT_HIP)
        ):
            hip_centers[t] = midpoint_xy(frame, LEFT_HIP, RIGHT_HIP)
            reliable[t] = True

    center_source = "reliable_hips"

    # --------------------------------------------------------
    # 如果一个高可靠双髋帧都没有，退化为只要求 x/y 有限
    # --------------------------------------------------------
    if not np.any(reliable):
        for t in range(num_frames):
            if not valid_mask[t]:
                continue

            frame = keypoints[t]
            if (
                is_xy_finite(frame, LEFT_HIP)
                and is_xy_finite(frame, RIGHT_HIP)
            ):
                hip_centers[t] = midpoint_xy(frame, LEFT_HIP, RIGHT_HIP)
                reliable[t] = True

        center_source = "finite_hips_fallback"

    center_available = np.zeros(num_frames, dtype=bool)
    center_interpolated = np.zeros(num_frames, dtype=bool)

    valid_indices = np.flatnonzero(reliable)
    if valid_indices.size == 0:
        return (
            hip_centers,
            center_available,
            center_interpolated,
            "unavailable",
        )

    # np.interp：中间做线性插值，首尾使用最近有效值。
    all_indices = np.arange(num_frames)
    for dim in range(2):
        hip_centers[:, dim] = np.interp(
            all_indices,
            valid_indices,
            hip_centers[valid_indices, dim],
        )

    center_available = valid_mask.astype(bool).copy()
    center_interpolated = center_available & (~reliable)

    # 原始无效帧不提供归一化中心。
    hip_centers[~valid_mask] = np.nan

    return (
        hip_centers,
        center_available,
        center_interpolated,
        center_source,
    )


# ============================================================
# 5. 为整段视频估计一个稳定尺度
# ============================================================

def estimate_video_scale(keypoints, valid_mask, hip_centers):
    """
    整段视频只使用一个稳定尺度，不再逐帧改变缩放比例。

    优先：可靠左右肩的肩宽中位数。
    回退：可靠左右肩的肩中心 -> 髋中心距离中位数。

    两种尺度都会先用 MAD 去除明显离群值。
    """
    shoulder_widths = []
    torso_lengths = []

    num_frames = keypoints.shape[0]

    for t in range(num_frames):
        if not valid_mask[t]:
            continue

        frame = keypoints[t]
        if not (
            is_landmark_reliable(frame, LEFT_SHOULDER)
            and is_landmark_reliable(frame, RIGHT_SHOULDER)
        ):
            continue

        left_shoulder = frame[LEFT_SHOULDER, :2].astype(np.float64)
        right_shoulder = frame[RIGHT_SHOULDER, :2].astype(np.float64)

        shoulder_width = float(
            np.linalg.norm(left_shoulder - right_shoulder)
        )
        if np.isfinite(shoulder_width) and shoulder_width >= MIN_SCALE:
            shoulder_widths.append(shoulder_width)

        if np.all(np.isfinite(hip_centers[t])):
            shoulder_center = (left_shoulder + right_shoulder) / 2.0
            torso_length = float(
                np.linalg.norm(shoulder_center - hip_centers[t])
            )
            if np.isfinite(torso_length) and torso_length >= MIN_SCALE:
                torso_lengths.append(torso_length)

    shoulder_scale, shoulder_count, shoulder_kept = robust_median(
        shoulder_widths
    )

    if shoulder_scale is not None:
        return {
            "scale": shoulder_scale,
            "source": "median_shoulder_width",
            "candidate_frames": shoulder_count,
            "kept_frames": shoulder_kept,
            "enough_frames": shoulder_kept >= MIN_SCALE_FRAMES,
        }

    torso_scale, torso_count, torso_kept = robust_median(torso_lengths)

    if torso_scale is not None:
        return {
            "scale": torso_scale,
            "source": "median_torso_length",
            "candidate_frames": torso_count,
            "kept_frames": torso_kept,
            "enough_frames": torso_kept >= MIN_SCALE_FRAMES,
        }

    return {
        "scale": None,
        "source": "unavailable",
        "candidate_frames": 0,
        "kept_frames": 0,
        "enough_frames": False,
    }


# ============================================================
# 6. 归一化一段视频
# ============================================================

def normalize_keypoints(keypoints, valid_mask):
    """
    输入：
        keypoints: [T, 33, 4]
        valid_mask: [T]

    输出：
        normalized_keypoints: [T, 33, 4]
        normalized_valid_mask: [T]
        stats: dict

    只修改 x、y；z 和 visibility 原样保留。

    核心变化：
    - 髋中心轨迹允许插值，减少额外丢帧；
    - 整段视频使用一个稳定尺度，避免逐帧缩放抖动；
    - 不再仅因为普通关键点 visibility < 0.3 就额外置 NaN；
    - 统计 clipping 比例，便于判断归一化是否异常。
    """
    normalized = keypoints.astype(np.float32).copy()
    normalized_valid_mask = valid_mask.astype(bool).copy()

    num_frames = keypoints.shape[0]

    (
        hip_centers,
        center_available,
        center_interpolated,
        center_source,
    ) = estimate_hip_centers(
        keypoints,
        normalized_valid_mask,
    )

    scale_info = estimate_video_scale(
        keypoints,
        normalized_valid_mask,
        hip_centers,
    )
    scale = scale_info["scale"]

    stats = {
        "frames_total": int(num_frames),
        "raw_valid_frames": int(np.sum(valid_mask)),
        "normalized_valid_frames": 0,
        "interpolated_center_frames": int(np.sum(center_interpolated)),
        "center_source": center_source,
        "scale": float(scale) if scale is not None else np.nan,
        "scale_source": scale_info["source"],
        "scale_candidate_frames": int(scale_info["candidate_frames"]),
        "scale_kept_frames": int(scale_info["kept_frames"]),
        "scale_enough_frames": bool(scale_info["enough_frames"]),
        "finite_xy_values": 0,
        "clipped_xy_values": 0,
    }

    # 整段视频都无法得到中心或稳定尺度时，不能可靠归一化。
    if (
        scale is None
        or not np.isfinite(scale)
        or scale < MIN_SCALE
        or not np.any(center_available)
    ):
        normalized[:, :, 0:2] = np.nan
        normalized_valid_mask[:] = False
        stats["normalized_valid_frames"] = 0
        return normalized, normalized_valid_mask, stats

    for t in range(num_frames):
        # 原始就没有有效人体，保持无效。
        if not valid_mask[t]:
            normalized[t, :, 0:2] = np.nan
            normalized_valid_mask[t] = False
            continue

        # 理论上只要视频中存在可靠双髋，valid 帧都会得到插值中心。
        if not center_available[t] or not np.all(np.isfinite(hip_centers[t])):
            normalized[t, :, 0:2] = np.nan
            normalized_valid_mask[t] = False
            continue

        xy = keypoints[t, :, 0:2].astype(np.float64)
        finite_xy = np.all(np.isfinite(xy), axis=1)

        # 只对原本有限的坐标做变换；原本 NaN 的点继续保持 NaN。
        normalized_xy = np.full_like(xy, np.nan, dtype=np.float64)
        normalized_xy[finite_xy] = (
            xy[finite_xy] - hip_centers[t]
        ) / (scale + EPS)

        finite_values = np.isfinite(normalized_xy)
        stats["finite_xy_values"] += int(np.sum(finite_values))

        # 在真正 clip 之前统计越界数量。
        out_of_range = finite_values & (
            (normalized_xy < CLIP_MIN)
            | (normalized_xy > CLIP_MAX)
        )
        stats["clipped_xy_values"] += int(np.sum(out_of_range))

        normalized_xy = np.clip(
            normalized_xy,
            CLIP_MIN,
            CLIP_MAX,
        )

        normalized[t, :, 0:2] = normalized_xy.astype(np.float32)
        normalized_valid_mask[t] = True

    stats["normalized_valid_frames"] = int(
        np.sum(normalized_valid_mask)
    )

    return normalized, normalized_valid_mask, stats


# ============================================================
# 7. 处理一个 NPZ 文件
# ============================================================

def process_npz(input_path, output_path):
    with np.load(input_path, allow_pickle=False) as data:
        if "keypoints" not in data.files:
            raise KeyError(f"{input_path} 没有 keypoints")

        if "valid_mask" not in data.files:
            raise KeyError(f"{input_path} 没有 valid_mask")

        keypoints = data["keypoints"]
        valid_mask = data["valid_mask"].astype(bool)

        if (
            keypoints.ndim != 3
            or keypoints.shape[1] != 33
            or keypoints.shape[2] != 4
        ):
            raise ValueError(
                f"{input_path} keypoints shape 异常：{keypoints.shape}"
            )

        if valid_mask.ndim != 1 or valid_mask.shape[0] != keypoints.shape[0]:
            raise ValueError(
                f"{input_path} valid_mask shape 异常：{valid_mask.shape}"
            )

        (
            normalized_keypoints,
            normalized_valid_mask,
            stats,
        ) = normalize_keypoints(
            keypoints,
            valid_mask,
        )

        # 保持原文件所有字段一致，只覆盖 keypoints / valid_mask。
        save_data = {
            key: data[key]
            for key in data.files
        }

    save_data["keypoints"] = normalized_keypoints
    save_data["valid_mask"] = normalized_valid_mask

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, **save_data)

    return stats


# ============================================================
# 8. 批量处理 + 汇总诊断
# ============================================================

def main():
    if not INPUT_ROOT.exists():
        raise FileNotFoundError(f"输入目录不存在：{INPUT_ROOT}")

    npz_files = sorted(INPUT_ROOT.rglob("*.npz"))
    if not npz_files:
        raise RuntimeError(f"{INPUT_ROOT} 没有找到 npz 文件")

    print(f"发现 {len(npz_files)} 个 keypoints 文件")

    total_raw_valid = 0
    total_normalized_valid = 0
    total_interpolated_centers = 0
    total_finite_xy = 0
    total_clipped_xy = 0
    scale_failures = 0
    low_scale_frame_videos = 0

    for input_path in tqdm(npz_files, desc="Normalizing"):
        relative_path = input_path.relative_to(INPUT_ROOT)
        output_path = OUTPUT_ROOT / relative_path

        stats = process_npz(input_path, output_path)

        total_raw_valid += stats["raw_valid_frames"]
        total_normalized_valid += stats["normalized_valid_frames"]
        total_interpolated_centers += stats["interpolated_center_frames"]
        total_finite_xy += stats["finite_xy_values"]
        total_clipped_xy += stats["clipped_xy_values"]

        if stats["scale_source"] == "unavailable":
            scale_failures += 1

        if (
            stats["scale_source"] != "unavailable"
            and not stats["scale_enough_frames"]
        ):
            low_scale_frame_videos += 1

    clip_ratio = (
        total_clipped_xy / total_finite_xy
        if total_finite_xy > 0
        else 0.0
    )

    added_invalid_frames = total_raw_valid - total_normalized_valid

    print()
    print("归一化完成")
    print(f"输入：{INPUT_ROOT}")
    print(f"输出：{OUTPUT_ROOT}")
    print()
    print("========== 诊断统计 ==========")
    print(f"原始有效帧：{total_raw_valid}")
    print(f"归一化后有效帧：{total_normalized_valid}")
    print(f"额外变无效帧：{added_invalid_frames}")
    print(f"使用插值髋中心的帧：{total_interpolated_centers}")
    print(f"无法获得视频级尺度的视频：{scale_failures}")
    print(
        "稳定尺度可靠帧少于 "
        f"{MIN_SCALE_FRAMES} 的视频：{low_scale_frame_videos}"
    )
    print(f"有限 x/y 数值总数：{total_finite_xy}")
    print(f"触发 clipping 的 x/y 数值：{total_clipped_xy}")
    print(f"clipping 比例：{clip_ratio:.6%}")


if __name__ == "__main__":
    main()
