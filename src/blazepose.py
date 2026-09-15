import re
import cv2
import numpy as np
import mediapipe as mp

from pathlib import Path
from tqdm import tqdm


# ============================================================
# 配置
# ============================================================

DATA_ROOT = Path("data/raw")
OUTPUT_ROOT = Path("data/pose")

# 所有视频均为 30 FPS
FPS = 30.0

# 目前处理 cam0 RGB
CAMERA_SUFFIX = "-cam0-rgb"

# 是否保存少量关键点可视化图片
SAVE_VIS = True

# 每段视频保存多少张可视化图片
NUM_VIS_SAMPLES = 5


# ============================================================
# BlazePose
# ============================================================

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils


# ============================================================
# 获取帧编号
# ============================================================

def get_frame_index(frame_path: Path) -> int:
    """
    例如：

    fall-01-cam0-rgb-001.png -> 1
    fall-01-cam0-rgb-002.png -> 2
    """

    match = re.search(
        r"-(\d+)$",
        frame_path.stem
    )

    if match is None:
        raise ValueError(
            f"无法解析帧号: {frame_path.name}"
        )

    return int(match.group(1))


# ============================================================
# 单个视频处理
# ============================================================

def process_video(
    frame_dir: Path,
    label: str,
    video_id: str
):

    # --------------------------------------------------------
    # 获取所有图片
    # --------------------------------------------------------

    frame_paths = sorted(
        frame_dir.glob("*.png"),
        key=get_frame_index
    )

    num_frames = len(frame_paths)

    if num_frames == 0:
        print(
            f"[WARN] {video_id}: 没找到图片"
        )
        return

    print("\n" + "=" * 60)
    print(f"video_id : {video_id}")
    print(f"label    : {label}")
    print(f"frames   : {num_frames}")
    print("=" * 60)

    # ========================================================
    # 保存数组
    # ========================================================

    # --------------------------------------------------------
    # keypoints
    #
    # shape = [T, 33, 4]
    #
    # 33 = BlazePose 的 33 个关键点
    #
    # 最后一维：
    #
    # 0 = x
    # 1 = y
    # 2 = z
    # 3 = visibility
    #
    # 如果这一帧没检测到人体，则为 NaN
    # --------------------------------------------------------

    keypoints = np.full(
        (num_frames, 33, 4),
        np.nan,
        dtype=np.float32
    )

    # --------------------------------------------------------
    # 是否检测成功
    # --------------------------------------------------------

    valid_mask = np.zeros(
        num_frames,
        dtype=np.bool_
    )

    # --------------------------------------------------------
    # 原始帧编号
    # --------------------------------------------------------

    frame_indices = np.zeros(
        num_frames,
        dtype=np.int32
    )

    # --------------------------------------------------------
    # 时间戳
    #
    # 单位：秒
    #
    # 001 -> 0
    # 002 -> 1/30
    # 003 -> 2/30
    # --------------------------------------------------------

    timestamps = np.zeros(
        num_frames,
        dtype=np.float64
    )

    # ========================================================
    # 可视化帧
    # ========================================================

    if SAVE_VIS:

        vis_indices = set(
            np.linspace(
                0,
                num_frames - 1,
                min(
                    NUM_VIS_SAMPLES,
                    num_frames
                ),
                dtype=int
            )
        )

        vis_dir = (
            OUTPUT_ROOT
            / "visualization"
            / label
            / video_id
        )

        vis_dir.mkdir(
            parents=True,
            exist_ok=True
        )

    else:

        vis_indices = set()
        vis_dir = None

    # ========================================================
    # BlazePose
    #
    # static_image_mode=True
    #
    # 非常重要：
    # 每张图片独立检测
    # 不使用前一帧
    # 不使用 tracking
    # 不做视频时序处理
    # ========================================================

    with mp_pose.Pose(

        static_image_mode=True,

        # BlazePose Heavy
        model_complexity=2,

        # 不需要人体分割
        enable_segmentation=False,

        # 人体检测阈值
        min_detection_confidence=0.5

    ) as pose:

        for i, frame_path in enumerate(
            tqdm(
                frame_paths,
                desc=video_id
            )
        ):

            # =================================================
            # 帧编号
            # =================================================

            frame_index = get_frame_index(
                frame_path
            )

            frame_indices[i] = frame_index

            # =================================================
            # 时间戳
            # =================================================

            timestamps[i] = (
                (frame_index - 1)
                / FPS
            )

            # =================================================
            # 读取图片
            # =================================================

            image = cv2.imread(
                str(frame_path)
            )

            if image is None:

                print(
                    f"\n[WARN] 无法读取图片: "
                    f"{frame_path}"
                )

                continue

            # OpenCV:
            # BGR
            #
            # MediaPipe:
            # RGB

            image_rgb = cv2.cvtColor(
                image,
                cv2.COLOR_BGR2RGB
            )

            # =================================================
            # BlazePose 推理
            # =================================================

            result = pose.process(
                image_rgb
            )

            # =================================================
            # 检测成功
            # =================================================

            if result.pose_landmarks:

                valid_mask[i] = True

                landmarks = (
                    result
                    .pose_landmarks
                    .landmark
                )

                # ---------------------------------------------
                # 保存 33 个关键点
                # ---------------------------------------------

                for j, landmark in enumerate(
                    landmarks
                ):

                    keypoints[i, j, 0] = (
                        landmark.x
                    )

                    keypoints[i, j, 1] = (
                        landmark.y
                    )

                    keypoints[i, j, 2] = (
                        landmark.z
                    )

                    keypoints[i, j, 3] = (
                        landmark.visibility
                    )

                # ---------------------------------------------
                # 保存少量可视化图片
                # ---------------------------------------------

                if (
                    SAVE_VIS
                    and i in vis_indices
                ):

                    vis_image = image.copy()

                    mp_drawing.draw_landmarks(
                        vis_image,
                        result.pose_landmarks,
                        mp_pose.POSE_CONNECTIONS
                    )

                    cv2.imwrite(
                        str(
                            vis_dir
                            / frame_path.name
                        ),
                        vis_image
                    )

            # =================================================
            # 检测失败
            # =================================================

            else:

                valid_mask[i] = False

                # keypoints 保持 NaN

                if (
                    SAVE_VIS
                    and i in vis_indices
                ):

                    cv2.imwrite(
                        str(
                            vis_dir
                            / (
                                frame_path.stem
                                + "_NO_PERSON.png"
                            )
                        ),
                        image
                    )

    # ========================================================
    # 保存 NPZ
    # ========================================================

    output_dir = (
        OUTPUT_ROOT
        / label
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    output_path = (
        output_dir
        / f"{video_id}.npz"
    )

    np.savez_compressed(

        output_path,

        # [T, 33, 4]
        keypoints=keypoints,

        # [T]
        valid_mask=valid_mask,

        # [T]
        frame_indices=frame_indices,

        # [T]，单位秒
        timestamps=timestamps,

        # 30 FPS
        fps=np.float32(FPS),

        # fall
        label=np.array(label),

        # fall-01-cam0-rgb
        video_id=np.array(video_id)
    )

    # ========================================================
    # 输出统计
    # ========================================================

    valid_count = int(
        valid_mask.sum()
    )

    print()
    print(
        f"保存完成: {output_path}"
    )

    print(
        f"检测成功: "
        f"{valid_count}/{num_frames}"
    )

    print(
        f"检测成功率: "
        f"{valid_count / num_frames:.2%}"
    )


# ============================================================
# 批量处理所有视频
# ============================================================

def main():

    if not DATA_ROOT.exists():

        raise FileNotFoundError(
            f"数据目录不存在: {DATA_ROOT}"
        )

    OUTPUT_ROOT.mkdir(
        parents=True,
        exist_ok=True
    )

    # ========================================================
    # 第一层：
    #
    # data/raw/fall
    # data/raw/adl
    #
    # 文件夹名直接作为 label
    # ========================================================

    label_dirs = sorted(
        [
            p
            for p in DATA_ROOT.iterdir()
            if p.is_dir()
        ]
    )

    for label_dir in label_dirs:

        label = label_dir.name

        print("\n" + "#" * 60)
        print(f"Label: {label}")
        print("#" * 60)

        # ====================================================
        # 第二层
        #
        # fall-01-cam0-rgb
        # fall-02-cam0-rgb
        # ...
        # ====================================================

        video_dirs = sorted(
            [
                p
                for p in label_dir.iterdir()
                if (
                    p.is_dir()
                    and p.name.endswith(
                        CAMERA_SUFFIX
                    )
                )
            ]
        )

        for outer_video_dir in video_dirs:

            video_id = (
                outer_video_dir.name
            )

            # ------------------------------------------------
            # 你的目录有两层同名：
            #
            # fall-01-cam0-rgb/
            #     fall-01-cam0-rgb/
            #         xxx.png
            # ------------------------------------------------

            inner_video_dir = (
                outer_video_dir
                / video_id
            )

            if inner_video_dir.is_dir():

                frame_dir = (
                    inner_video_dir
                )

            else:

                # 顺便兼容没有第二层目录的情况
                frame_dir = (
                    outer_video_dir
                )

            process_video(
                frame_dir=frame_dir,
                label=label,
                video_id=video_id
            )


# ============================================================
# 程序入口
# ============================================================

if __name__ == "__main__":
    main()