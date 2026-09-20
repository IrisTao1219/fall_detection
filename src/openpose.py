# src/openpose/extract_keypoints.py

import argparse
import json
import subprocess
import tempfile
from pathlib import Path

import numpy as np


IMAGE_EXTENSIONS = {
    ".png",
    ".jpg",
    ".jpeg",
    ".bmp",
}


def get_frame_files(sequence_dir):
    """
    获取一个序列中的所有图片，并按文件名排序。
    """
    frames = [
        p for p in sequence_dir.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    ]

    return sorted(frames)


def find_sequences(data_root):
    """
    查找：

    data/raw/
    ├── adl/
    │   ├── adl-01-cam0-rgb/
    │   └── ...
    └── fall/
        ├── fall-01-cam0-rgb/
        └── ...
    """

    sequences = []

    for category in ["adl", "fall"]:

        category_dir = data_root / category

        if not category_dir.exists():
            print(f"[WARNING] 目录不存在: {category_dir}")
            continue

        for sequence_dir in sorted(category_dir.iterdir()):

            if sequence_dir.is_dir():
                sequences.append(
                    (category, sequence_dir)
                )

    return sequences


def select_person(people):
    """
    OpenPose 可能一帧检测到多个人。

    这里选择 BODY_25 平均置信度最高的人。

    返回：
        shape = (25, 3)
        每个点：[x, y, confidence]

    如果没有检测到人：
        返回 None
    """

    if not people:
        return None

    best_keypoints = None
    best_score = -1.0

    for person in people:

        pose = person.get(
            "pose_keypoints_2d",
            []
        )

        if len(pose) != 25 * 3:
            continue

        keypoints = np.asarray(
            pose,
            dtype=np.float32
        ).reshape(25, 3)

        # 第三列为 confidence
        confidence = keypoints[:, 2]

        score = float(
            np.mean(confidence)
        )

        if score > best_score:
            best_score = score
            best_keypoints = keypoints

    return best_keypoints


def load_openpose_json(
    json_path
):
    """
    读取单帧 OpenPose JSON。
    """

    if not json_path.exists():
        return None

    try:
        with open(
            json_path,
            "r",
            encoding="utf-8"
        ) as f:
            data = json.load(f)

    except (
        json.JSONDecodeError,
        OSError
    ):
        return None

    people = data.get(
        "people",
        []
    )

    return select_person(people)


def run_openpose(
    openpose_bin,
    model_folder,
    sequence_dir,
    json_output_dir,
):
    """
    调用 OpenPose CLI。

    OpenPose：
        PNG
         ↓
        BODY_25
         ↓
        JSON
    """

    command = [
        str(openpose_bin),

        "--image_dir",
        str(sequence_dir),

        "--model_folder",
        str(model_folder),

        "--model_pose",
        "BODY_25",

        "--write_json",
        str(json_output_dir),

        # 使用原始图像尺度坐标
        "--keypoint_scale",
        "0",

        # 服务器不打开 GUI
        "--display",
        "0",

        # 不生成骨架可视化图
        "--render_pose",
        "0",
    ]

    print("Running OpenPose...")

    result = subprocess.run(
        command,
        check=False,
    )

    if result.returncode != 0:
        raise RuntimeError(
            f"OpenPose 执行失败，"
            f"return code = {result.returncode}"
        )


def extract_sequence(
    category,
    sequence_dir,
    output_path,
    openpose_bin,
    model_folder,
):
    """
    提取一个完整图片序列。
    """

    frame_files = get_frame_files(
        sequence_dir
    )

    if not frame_files:
        print(
            f"[WARNING] 没有发现图片: "
            f"{sequence_dir}"
        )
        return False

    print(
        f"Frames : {len(frame_files)}"
    )

    # 使用临时目录存储 OpenPose JSON
    # 转成 NPZ 后会自动删除
    with tempfile.TemporaryDirectory(
        prefix="openpose_"
    ) as temp_dir:

        json_dir = Path(temp_dir)

        # -------------------------
        # 调用 OpenPose
        # -------------------------

        run_openpose(
            openpose_bin=openpose_bin,
            model_folder=model_folder,
            sequence_dir=sequence_dir,
            json_output_dir=json_dir,
        )

        # -------------------------
        # 读取 JSON
        # -------------------------

        keypoints_list = []
        valid_mask = []
        frame_names = []

        for frame_path in frame_files:

            # OpenPose 输出格式：
            #
            # 原图片：
            # fall-01-cam0-rgb-001.png
            #
            # JSON：
            # fall-01-cam0-rgb-001_keypoints.json

            json_name = (
                frame_path.stem
                + "_keypoints.json"
            )

            json_path = (
                json_dir / json_name
            )

            keypoints = (
                load_openpose_json(
                    json_path
                )
            )

            if keypoints is None:

                keypoints = np.zeros(
                    (25, 3),
                    dtype=np.float32,
                )

                valid = False

            else:
                valid = True

            keypoints_list.append(
                keypoints
            )

            valid_mask.append(
                valid
            )

            frame_names.append(
                frame_path.name
            )

    # -------------------------
    # 转成 numpy
    # -------------------------

    keypoints = np.stack(
        keypoints_list
    ).astype(np.float32)

    valid_mask = np.asarray(
        valid_mask,
        dtype=bool,
    )

    frame_names = np.asarray(
        frame_names
    )

    # -------------------------
    # 保存 NPZ
    # -------------------------

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    np.savez_compressed(
        output_path,

        # [T, 25, 3]
        keypoints=keypoints,

        # [T]
        valid_mask=valid_mask,

        # 原始 PNG 文件名
        frame_names=frame_names,

        # adl / fall
        category=category,

        # 保持和原实验格式一致
        label=category,

        # fall-01-cam0-rgb
        video_id=sequence_dir.name,
    )

    valid_count = int(
        valid_mask.sum()
    )

    total = len(
        valid_mask
    )

    detection_rate = (
        valid_count / total * 100
    )

    print(
        f"[OK] {sequence_dir.name}\n"
        f"     category  : {category}\n"
        f"     frames    : {total}\n"
        f"     keypoints : {keypoints.shape}\n"
        f"     valid     : "
        f"{valid_count}/{total} "
        f"({detection_rate:.2f}%)\n"
        f"     saved     : {output_path}"
    )

    return True


def main():

    parser = argparse.ArgumentParser(
        description=(
            "Extract OpenPose BODY_25 "
            "keypoints from UR Fall images"
        )
    )

    parser.add_argument(
        "--data-root",
        type=str,
        default="data/raw",
        help="原始图片目录",
    )

    parser.add_argument(
        "--output-root",
        type=str,
        default="data/keypoints_openpose",
        help="关键点输出目录",
    )

    parser.add_argument(
        "--model-folder",
        type=str,
        default="models",
        help="OpenPose 模型目录",
    )

    parser.add_argument(
        "--openpose-bin",
        type=str,
        required=True,
        help=(
            "OpenPose 可执行文件路径，例如 "
            "/home/user/openpose/"
            "build/examples/openpose/openpose.bin"
        ),
    )

    args = parser.parse_args()

    data_root = Path(
        args.data_root
    ).resolve()

    output_root = Path(
        args.output_root
    ).resolve()

    model_folder = Path(
        args.model_folder
    ).resolve()

    openpose_bin = Path(
        args.openpose_bin
    ).resolve()

    # -------------------------
    # 检查 OpenPose
    # -------------------------

    if not openpose_bin.exists():
        raise FileNotFoundError(
            f"找不到 OpenPose："
            f"{openpose_bin}"
        )

    if not model_folder.exists():
        raise FileNotFoundError(
            f"找不到模型目录："
            f"{model_folder}"
        )

    # -------------------------
    # 查找序列
    # -------------------------

    sequences = find_sequences(
        data_root
    )

    print(
        f"Found {len(sequences)} sequences."
    )

    success = 0
    failed = 0

    for index, (
        category,
        sequence_dir,
    ) in enumerate(
        sequences,
        start=1,
    ):

        print()
        print("=" * 70)

        print(
            f"[{index}/{len(sequences)}]"
        )

        print(
            f"Sequence : "
            f"{sequence_dir.name}"
        )

        print(
            f"Category : {category}"
        )

        output_path = (
            output_root
            / category
            / f"{sequence_dir.name}.npz"
        )

        try:

            ok = extract_sequence(
                category=category,
                sequence_dir=sequence_dir,
                output_path=output_path,
                openpose_bin=openpose_bin,
                model_folder=model_folder,
            )

            if ok:
                success += 1
            else:
                failed += 1

        except Exception as e:

            failed += 1

            print(
                f"[ERROR] "
                f"{sequence_dir.name}: "
                f"{e}"
            )

    print()
    print("=" * 70)

    print(
        "OpenPose extraction finished."
    )

    print(
        f"Success : {success}"
    )

    print(
        f"Failed  : {failed}"
    )


if __name__ == "__main__":
    main()