import csv
import cv2
import numpy as np
import mediapipe as mp

from pathlib import Path
from tqdm import tqdm


# ============================================================
# 配置
# ============================================================

# Le2i 已经抽帧后的目录：
#
# data/raw_le2i/pngs/
# ├── Coffee_room_01_video_1/
# │   ├── 000001.png
# │   ├── 000002.png
# │   ├── ...
# │   └── labels.csv
# ├── Coffee_room_01_video_2/
# └── ...
#
DATA_ROOT = Path("data/raw_le2i/pngs")

OUTPUT_ROOT = Path("data")

# Le2i 固定为 25 FPS
FPS = 25.0

# 是否保存少量关键点可视化图片
SAVE_VIS = True

# 每段视频保存多少张可视化图片
NUM_VIS_SAMPLES = 5

# 如果对应 NPZ 已存在，是否跳过
SKIP_EXISTING = True


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

    000001.png -> 1
    000002.png -> 2
    000048.png -> 48
    """

    try:
        return int(frame_path.stem)

    except ValueError:
        raise ValueError(
            f"无法解析帧号: {frame_path.name}"
        )


# ============================================================
# 读取 labels.csv
# ============================================================

def load_labels(csv_path: Path):
    """
    返回：

    frame_label_dict:
        {
            1: 0,
            2: 0,
            ...
            48: 1,
            ...
        }

    bbox_dict:
        {
            1: [x1, y1, x2, y2],
            ...
        }

    bbox_valid_dict:
        {
            1: False,
            ...
        }

    fall_start
    fall_end
    """

    if not csv_path.exists():
        raise FileNotFoundError(
            f"labels.csv 不存在: {csv_path}"
        )

    frame_label_dict = {}
    bbox_dict = {}
    bbox_valid_dict = {}

    fall_start = None
    fall_end = None

    with csv_path.open(
        "r",
        encoding="utf-8"
    ) as f:

        reader = csv.DictReader(f)

        for row in reader:

            frame_id = int(row["frame_id"])
            label = int(row["label"])

            frame_label_dict[frame_id] = label

            # ------------------------------------------------
            # 保存 bbox
            # 后续如果 ViTPose 需要，可以直接使用
            # ------------------------------------------------

            x1 = int(row.get("x1", 0))
            y1 = int(row.get("y1", 0))
            x2 = int(row.get("x2", 0))
            y2 = int(row.get("y2", 0))

            bbox_dict[frame_id] = [
                x1,
                y1,
                x2,
                y2
            ]

            bbox_valid_dict[frame_id] = (
                int(
                    row.get(
                        "bbox_valid",
                        0
                    )
                )
                == 1
            )

            if fall_start is None:

                if "fall_start" in row:
                    fall_start = int(
                        row["fall_start"]
                    )

                if "fall_end" in row:
                    fall_end = int(
                        row["fall_end"]
                    )

    return (
        frame_label_dict,
        bbox_dict,
        bbox_valid_dict,
        fall_start,
        fall_end,
    )


# ============================================================
# 单个视频处理
# ============================================================

def process_video(
    frame_dir: Path,
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
            f"[WARN] {video_id}: 没找到 PNG"
        )

        return

    # --------------------------------------------------------
    # labels.csv
    # --------------------------------------------------------

    csv_path = frame_dir / "labels.csv"

    (
        frame_label_dict,
        bbox_dict,
        bbox_valid_dict,
        fall_start,
        fall_end,
    ) = load_labels(csv_path)

    print("\n" + "=" * 60)

    print(f"video_id   : {video_id}")
    print(f"frames     : {num_frames}")
    print(f"fall_start : {fall_start}")
    print(f"fall_end   : {fall_end}")

    print("=" * 60)

    # ========================================================
    # 保存数组
    # ========================================================

    # --------------------------------------------------------
    # keypoints
    #
    # shape = [T, 33, 4]
    #
    # 33 = BlazePose 33 个关键点
    #
    # 最后一维：
    #
    # 0 = x
    # 1 = y
    # 2 = z
    # 3 = visibility
    #
    # 如果这一帧没有检测到人体，则保持 NaN
    # --------------------------------------------------------

    keypoints = np.full(
        (num_frames, 33, 4),
        np.nan,
        dtype=np.float32
    )

    # --------------------------------------------------------
    # BlazePose 是否检测成功
    # --------------------------------------------------------

    valid_mask = np.zeros(
        num_frames,
        dtype=np.bool_
    )

    # --------------------------------------------------------
    # 原始帧编号
    #
    # 000001.png -> 1
    # 000002.png -> 2
    # --------------------------------------------------------

    frame_indices = np.zeros(
        num_frames,
        dtype=np.int32
    )

    # --------------------------------------------------------
    # 时间戳
    #
    # Le2i = 25 FPS
    #
    # frame 1 -> 0 秒
    # frame 2 -> 1/25 秒
    # --------------------------------------------------------

    timestamps = np.zeros(
        num_frames,
        dtype=np.float64
    )

    # --------------------------------------------------------
    # Le2i 逐帧标签
    #
    # 0 = non-fall
    # 1 = fall
    # --------------------------------------------------------

    frame_labels = np.zeros(
        num_frames,
        dtype=np.int8
    )

    # --------------------------------------------------------
    # bbox
    #
    # shape = [T, 4]
    #
    # x1, y1, x2, y2
    # --------------------------------------------------------

    bboxes = np.zeros(
        (num_frames, 4),
        dtype=np.int32
    )

    bbox_valid_mask = np.zeros(
        num_frames,
        dtype=np.bool_
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
            / "pose_le2i"
            / "visualization"
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
    # 与之前 UR-Fall 保持一致：
    #
    # static_image_mode=True
    #
    # 每张图片独立检测
    # 不使用 tracking
    #
    # model_complexity=2
    # BlazePose Heavy
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
            # 对应 Le2i 标签
            # =================================================

            if frame_index not in frame_label_dict:

                print(
                    f"\n[WARN] {video_id}: "
                    f"frame {frame_index} "
                    f"在 labels.csv 中不存在"
                )

                frame_labels[i] = -1

            else:

                frame_labels[i] = (
                    frame_label_dict[
                        frame_index
                    ]
                )

            # =================================================
            # bbox
            # =================================================

            if frame_index in bbox_dict:

                bboxes[i] = np.array(
                    bbox_dict[
                        frame_index
                    ],
                    dtype=np.int32
                )

                bbox_valid_mask[i] = (
                    bbox_valid_dict.get(
                        frame_index,
                        False
                    )
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

                    # -----------------------------------------
                    # 左上角显示当前标签
                    # -----------------------------------------

                    label_text = (
                        "FALL"
                        if frame_labels[i] == 1
                        else "NON-FALL"
                    )

                    cv2.putText(
                        vis_image,
                        label_text,
                        (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 0, 255)
                        if frame_labels[i] == 1
                        else (0, 255, 0),
                        2
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
        / "keypoints_le2i"
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

        # 25 FPS
        fps=np.float32(FPS),

        # ----------------------------------------------------
        # 为了尽量与之前 UR-Fall 的 NPZ 保持兼容，
        # 仍然保留 label 字段。
        #
        # 但是 Le2i 一个视频中同时存在 non-fall 和 fall，
        # 因此这里不能再用单一的 fall / adl 视频级标签。
        # ----------------------------------------------------

        label=np.array("mixed"),

        # ----------------------------------------------------
        # Le2i 真正需要使用的是逐帧标签
        #
        # [T]
        # 0 = non-fall
        # 1 = fall
        # ----------------------------------------------------

        frame_labels=frame_labels,

        # [T, 4]
        # x1, y1, x2, y2
        bboxes=bboxes,

        # [T]
        bbox_valid_mask=bbox_valid_mask,

        # 跌倒开始帧
        fall_start=np.int32(
            fall_start
            if fall_start is not None
            else -1
        ),

        # 跌倒结束帧
        fall_end=np.int32(
            fall_end
            if fall_end is not None
            else -1
        ),

        # Coffee_room_01_video_1
        video_id=np.array(video_id)
    )

    # ========================================================
    # 输出统计
    # ========================================================

    valid_count = int(
        valid_mask.sum()
    )

    fall_count = int(
        (frame_labels == 1).sum()
    )

    nonfall_count = int(
        (frame_labels == 0).sum()
    )

    invalid_label_count = int(
        (frame_labels < 0).sum()
    )

    print()

    print(
        f"保存完成: {output_path}"
    )

    print(
        f"检测到人体: "
        f"{valid_count}/{num_frames} "
        f"({valid_count / num_frames * 100:.2f}%)"
    )

    print(
        f"Fall frames    : {fall_count}"
    )

    print(
        f"Non-fall frames: {nonfall_count}"
    )

    if invalid_label_count > 0:

        print(
            f"[WARN] 未匹配标签帧: "
            f"{invalid_label_count}"
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
    # Le2i 当前目录：
    #
    # data/raw_le2i/pngs/
    #
    # ├── Coffee_room_01_video_1/
    # ├── Coffee_room_01_video_2/
    # ├── Coffee_room_02_video_1/
    # ├── Home_01_video_1/
    # └── ...
    #
    # 每个目录就是一个原始视频
    # ========================================================

    video_dirs = sorted(
        [
            p
            for p in DATA_ROOT.iterdir()
            if p.is_dir()
        ]
    )

    print(
        f"Found {len(video_dirs)} videos"
    )

    for index, frame_dir in enumerate(
        video_dirs,
        start=1
    ):

        video_id = frame_dir.name

        print("\n" + "#" * 60)

        print(
            f"[{index}/{len(video_dirs)}] "
            f"{video_id}"
        )

        print("#" * 60)

        labels_path = (
            frame_dir
            / "labels.csv"
        )

        if not labels_path.exists():

            print(
                f"[WARN] 跳过，没有 labels.csv: "
                f"{frame_dir}"
            )

            continue

        # ========================================================
        # 检查 NPZ 是否已经存在
        # ========================================================

        output_path = (
            OUTPUT_ROOT
            / "keypoints_le2i"
            / f"{video_id}.npz"
        )

        if (
            SKIP_EXISTING
            and output_path.exists()
        ):

            print(
                f"[SKIP] NPZ 已存在: "
                f"{output_path}"
            )

            continue

    # ========================================================
    # 不存在才开始跑 BlazePose
    # ========================================================

        process_video(
            frame_dir=frame_dir,
            video_id=video_id
        )


# ============================================================
# 程序入口
# ============================================================

if __name__ == "__main__":
    main()