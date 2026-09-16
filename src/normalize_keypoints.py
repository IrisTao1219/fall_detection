from pathlib import Path

import numpy as np
from tqdm import tqdm


# ============================================================
# 1. 配置
# ============================================================

# 原始 keypoints
INPUT_ROOT = Path("data/keypoints")

# 归一化后的 keypoints
#
# 文件结构和 INPUT_ROOT 完全一致
OUTPUT_ROOT = Path("data/keypoints_normalized")

# BlazePose visibility 阈值
VISIBILITY_THRESHOLD = 0.3

# 防止除 0
EPS = 1e-6

# 极端坐标裁剪范围
CLIP_MIN = -5.0
CLIP_MAX = 5.0


# ============================================================
# 2. BlazePose 关键点编号
# ============================================================

LEFT_SHOULDER = 11
RIGHT_SHOULDER = 12

LEFT_HIP = 23
RIGHT_HIP = 24


# ============================================================
# 3. 判断关键点是否有效
# ============================================================

def is_landmark_valid(
    frame,
    index
):
    """
    frame:
        [33, 4]

    每个关键点：
        x, y, z, visibility
    """

    x = frame[index, 0]
    y = frame[index, 1]
    visibility = frame[index, 3]

    return (
        np.isfinite(x)
        and np.isfinite(y)
        and np.isfinite(visibility)
        and visibility >= VISIBILITY_THRESHOLD
    )


# ============================================================
# 4. 两个关键点中点
# ============================================================

def midpoint(
    frame,
    index1,
    index2
):
    """
    返回二维中点：
        [x, y]
    """

    p1 = frame[index1, :2]
    p2 = frame[index2, :2]

    return (p1 + p2) / 2.0


# ============================================================
# 5. 髋中心
# ============================================================

def get_hip_center(frame):
    """
    身体中心：

        左髋 + 右髋
             ↓
           中点

    如果左右髋不可用，返回 None。
    """

    left_valid = is_landmark_valid(
        frame,
        LEFT_HIP
    )

    right_valid = is_landmark_valid(
        frame,
        RIGHT_HIP
    )

    if not (
        left_valid
        and right_valid
    ):
        return None

    return midpoint(
        frame,
        LEFT_HIP,
        RIGHT_HIP
    )


# ============================================================
# 6. 当前帧尺度
# ============================================================

def get_current_scale(
    frame,
    hip_center
):
    """
    尺度选择：

    1. 优先肩宽
    2. 肩宽不可用时：
       使用肩中心到髋中心的躯干长度
    3. 均不可用：
       返回 None

    最近有效尺度的沿用在主循环中实现。
    """

    left_valid = is_landmark_valid(
        frame,
        LEFT_SHOULDER
    )

    right_valid = is_landmark_valid(
        frame,
        RIGHT_SHOULDER
    )

    # ========================================================
    # 左右肩都可靠
    # ========================================================

    if left_valid and right_valid:

        left_shoulder = frame[
            LEFT_SHOULDER,
            :2
        ]

        right_shoulder = frame[
            RIGHT_SHOULDER,
            :2
        ]

        # ----------------------------------------------------
        # 方案 1：肩宽
        # ----------------------------------------------------

        shoulder_width = np.linalg.norm(
            left_shoulder
            - right_shoulder
        )

        if (
            np.isfinite(shoulder_width)
            and shoulder_width > EPS
        ):

            return float(
                shoulder_width
            )

        # ----------------------------------------------------
        # 方案 2：肩中心 -> 髋中心
        # ----------------------------------------------------

        shoulder_center = (
            left_shoulder
            + right_shoulder
        ) / 2.0

        torso_length = np.linalg.norm(
            shoulder_center
            - hip_center
        )

        if (
            np.isfinite(torso_length)
            and torso_length > EPS
        ):

            return float(
                torso_length
            )

    return None


# ============================================================
# 7. 归一化一段视频
# ============================================================

def normalize_keypoints(
    keypoints,
    valid_mask
):
    """
    输入：

        keypoints:
            [T, 33, 4]

        valid_mask:
            [T]

    输出：

        normalized_keypoints:
            [T, 33, 4]

        normalized_valid_mask:
            [T]

    保持 keypoints 原有 shape 不变。

    只修改：
        x, y

    保留：
        z, visibility
    """

    # 直接复制原数据
    normalized = (
        keypoints.astype(
            np.float32
        ).copy()
    )

    # 输出的 valid_mask
    normalized_valid_mask = (
        valid_mask.astype(
            bool
        ).copy()
    )

    num_frames = (
        keypoints.shape[0]
    )

    # 最近一次有效尺度
    last_valid_scale = None

    # ========================================================
    # 遍历每帧
    # ========================================================

    for t in range(
        num_frames
    ):

        # ----------------------------------------------------
        # 原始 BlazePose 就没有人体关键点
        # ----------------------------------------------------

        if not valid_mask[t]:

            normalized[
                t,
                :,
                0:2
            ] = np.nan

            normalized_valid_mask[
                t
            ] = False

            continue

        frame = keypoints[t]

        # ====================================================
        # 1. 当前帧髋中心
        # ====================================================

        hip_center = (
            get_hip_center(
                frame
            )
        )

        # 无法得到当前身体中心
        # 这一帧不能可靠归一化
        if hip_center is None:

            normalized[
                t,
                :,
                0:2
            ] = np.nan

            normalized_valid_mask[
                t
            ] = False

            continue

        # ====================================================
        # 2. 当前帧尺度
        # ====================================================

        current_scale = (
            get_current_scale(
                frame,
                hip_center
            )
        )

        # ----------------------------------------------------
        # 当前尺度有效
        # ----------------------------------------------------

        if (
            current_scale is not None
            and current_scale > EPS
        ):

            scale = current_scale

            last_valid_scale = (
                current_scale
            )

        # ----------------------------------------------------
        # 当前尺度不可用
        #
        # 沿用最近有效尺度
        # ----------------------------------------------------

        elif last_valid_scale is not None:

            scale = (
                last_valid_scale
            )

        # ----------------------------------------------------
        # 视频开头还没有任何有效尺度
        # ----------------------------------------------------

        else:

            normalized[
                t,
                :,
                0:2
            ] = np.nan

            normalized_valid_mask[
                t
            ] = False

            continue

        # ====================================================
        # 3. 取原始 x,y
        # ====================================================

        xy = frame[
            :,
            0:2
        ].astype(
            np.float64
        )

        # ====================================================
        # 4. 平移 + 尺度归一化
        #
        # x' = (x - xc) / (s + eps)
        # y' = (y - yc) / (s + eps)
        # ====================================================

        normalized_xy = (
            xy - hip_center
        ) / (
            scale + EPS
        )

        # ====================================================
        # 5. 不可靠关键点设为 NaN
        # ====================================================

        visibility = frame[
            :,
            3
        ]

        invalid_landmarks = (
            ~np.isfinite(
                visibility
            )
            |
            (
                visibility
                <
                VISIBILITY_THRESHOLD
            )
        )

        normalized_xy[
            invalid_landmarks
        ] = np.nan

        # ====================================================
        # 6. 裁剪极端值
        # ====================================================

        normalized_xy = np.clip(
            normalized_xy,
            CLIP_MIN,
            CLIP_MAX
        )

        # ====================================================
        # 7. 覆盖原来的 x,y
        # ====================================================

        normalized[
            t,
            :,
            0:2
        ] = normalized_xy.astype(
            np.float32
        )

        normalized_valid_mask[
            t
        ] = True

    return (
        normalized,
        normalized_valid_mask
    )


# ============================================================
# 8. 处理一个 NPZ 文件
# ============================================================

def process_npz(
    input_path,
    output_path
):

    with np.load(
        input_path,
        allow_pickle=False
    ) as data:

        if "keypoints" not in data.files:

            raise KeyError(
                f"{input_path} "
                f"没有 keypoints"
            )

        if "valid_mask" not in data.files:

            raise KeyError(
                f"{input_path} "
                f"没有 valid_mask"
            )

        keypoints = (
            data["keypoints"]
        )

        valid_mask = (
            data["valid_mask"]
            .astype(bool)
        )

        # ----------------------------------------------------
        # 检查 keypoints shape
        # ----------------------------------------------------

        if (
            keypoints.ndim != 3
            or keypoints.shape[1] != 33
            or keypoints.shape[2] != 4
        ):

            raise ValueError(
                f"{input_path} "
                f"keypoints shape 异常："
                f"{keypoints.shape}"
            )

        # ====================================================
        # 归一化
        # ====================================================

        (
            normalized_keypoints,
            normalized_valid_mask
        ) = normalize_keypoints(
            keypoints,
            valid_mask
        )

        # ====================================================
        # 保持原文件所有字段完全一致
        # ====================================================

        save_data = {
            key: data[key]
            for key in data.files
        }

    # --------------------------------------------------------
    # 只覆盖两个字段的值
    #
    # 字段名称不变
    # --------------------------------------------------------

    save_data[
        "keypoints"
    ] = normalized_keypoints

    save_data[
        "valid_mask"
    ] = normalized_valid_mask

    # ========================================================
    # 保存
    # ========================================================

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    np.savez_compressed(
        output_path,
        **save_data
    )


# ============================================================
# 9. 批量处理
# ============================================================

def main():

    if not INPUT_ROOT.exists():

        raise FileNotFoundError(
            f"输入目录不存在："
            f"{INPUT_ROOT}"
        )

    npz_files = sorted(
        INPUT_ROOT.rglob(
            "*.npz"
        )
    )

    if not npz_files:

        raise RuntimeError(
            f"{INPUT_ROOT} "
            f"没有找到 npz 文件"
        )

    print(
        f"发现 {len(npz_files)} "
        f"个 keypoints 文件"
    )

    for input_path in tqdm(
        npz_files,
        desc="Normalizing"
    ):

        relative_path = (
            input_path.relative_to(
                INPUT_ROOT
            )
        )

        output_path = (
            OUTPUT_ROOT
            / relative_path
        )

        process_npz(
            input_path,
            output_path
        )

    print()
    print(
        "归一化完成"
    )

    print(
        f"输入：{INPUT_ROOT}"
    )

    print(
        f"输出：{OUTPUT_ROOT}"
    )


if __name__ == "__main__":
    main()