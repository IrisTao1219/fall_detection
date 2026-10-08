from pathlib import Path
import cv2
import csv
import re


# ============================================================
# 配置
# ============================================================

INPUT_ROOT = Path("data/raw_le2i")

# 输出：
# data/raw_le2i/pngs/Coffee_room_01_video_1/
OUTPUT_ROOT = INPUT_ROOT / "pngs"

IMAGE_EXT = ".png"


# ============================================================
# 从 video (1).avi 中提取序号 1
# ============================================================

def get_video_number(video_path: Path):
    """
    Examples:
        video (1).avi -> 1
        video (12).avi -> 12
        video1.avi -> 1
    """

    numbers = re.findall(r"\d+", video_path.stem)

    if numbers:
        return int(numbers[-1])

    # 实在找不到数字时，保留原文件名
    return video_path.stem.replace(" ", "_")


# ============================================================
# 读取 Le2i annotation
# ============================================================

def parse_annotation(annotation_path: Path):

    lines = [
        line.strip()
        for line in annotation_path.read_text(
            encoding="utf-8",
            errors="ignore"
        ).splitlines()
        if line.strip()
    ]

    if len(lines) < 2:
        raise ValueError(
            f"Annotation too short: {annotation_path}"
        )

    # 第一行：跌倒开始帧
    # 第二行：跌倒结束帧
    try:
        fall_start = int(lines[0])
        fall_end = int(lines[1])

    except ValueError:
        raise ValueError(
            f"Invalid fall start/end: {annotation_path}\n"
            f"First two lines: {lines[:2]}"
        )

    bbox_dict = {}

    # 后面的格式：
    # frame_id,target_id,x1,y1,x2,y2
    for line in lines[2:]:

        parts = [x.strip() for x in line.split(",")]

        if len(parts) != 6:
            print(
                f"[WARNING] Invalid annotation line:\n"
                f"  {annotation_path}\n"
                f"  {line}"
            )
            continue

        try:
            frame_id = int(parts[0])
            target_id = int(parts[1])
            x1 = int(parts[2])
            y1 = int(parts[3])
            x2 = int(parts[4])
            y2 = int(parts[5])

        except ValueError:
            print(
                f"[WARNING] Cannot parse line:\n"
                f"  {line}"
            )
            continue

        bbox_dict[frame_id] = {
            "target_id": target_id,
            "x1": x1,
            "y1": y1,
            "x2": x2,
            "y2": y2,
        }

    return fall_start, fall_end, bbox_dict


# ============================================================
# 抽帧 + CSV
# ============================================================

def process_video(
    video_path: Path,
    annotation_path: Path,
    scene_name: str,
):

    video_number = get_video_number(video_path)

    output_name = f"{scene_name}_video_{video_number}"

    output_dir = OUTPUT_ROOT / output_name
    output_dir.mkdir(parents=True, exist_ok=True)

    # --------------------------------------------------------
    # annotation
    # --------------------------------------------------------

    fall_start, fall_end, bbox_dict = parse_annotation(
        annotation_path
    )

    # --------------------------------------------------------
    # video
    # --------------------------------------------------------

    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        print(f"[ERROR] Cannot open video: {video_path}")
        return False

    fps = cap.get(cv2.CAP_PROP_FPS)
    expected_frames = int(
        cap.get(cv2.CAP_PROP_FRAME_COUNT)
    )

    width = int(
        cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    )

    height = int(
        cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    )

    print("\n============================================")
    print(f"Scene      : {scene_name}")
    print(f"Video      : {video_path.name}")
    print(f"Annotation : {annotation_path.name}")
    print(f"Output     : {output_dir}")
    print(
        f"FPS={fps:.2f}, "
        f"frames={expected_frames}, "
        f"size={width}x{height}"
    )
    print(
        f"Fall interval: "
        f"{fall_start} ~ {fall_end}"
    )

    # --------------------------------------------------------
    # CSV
    # --------------------------------------------------------

    csv_path = output_dir / "labels.csv"

    csv_file = csv_path.open(
        "w",
        newline="",
        encoding="utf-8"
    )

    writer = csv.DictWriter(
        csv_file,
        fieldnames=[
            "frame_id",
            "image",
            "label",
            "fall_start",
            "fall_end",
            "bbox_valid",
            "target_id",
            "x1",
            "y1",
            "x2",
            "y2",
        ],
    )

    writer.writeheader()

    # --------------------------------------------------------
    # 开始抽帧
    # --------------------------------------------------------

    frame_id = 1
    saved_count = 0

    while True:

        ret, frame = cap.read()

        if not ret:
            break

        # 与 Le2i 标注保持一致，从 1 开始
        image_name = f"{frame_id:06d}{IMAGE_EXT}"

        image_path = output_dir / image_name

        success = cv2.imwrite(
            str(image_path),
            frame
        )

        if not success:
            print(
                f"[WARNING] Cannot save: "
                f"{image_path}"
            )

        # ----------------------------------------------------
        # Fall label
        # ----------------------------------------------------

        if fall_start <= frame_id <= fall_end:
            label = 1
        else:
            label = 0

        # ----------------------------------------------------
        # bbox
        # ----------------------------------------------------

        bbox = bbox_dict.get(frame_id)

        if bbox is None:

            target_id = -1
            x1 = y1 = x2 = y2 = 0
            bbox_valid = 0

        else:

            target_id = bbox["target_id"]
            x1 = bbox["x1"]
            y1 = bbox["y1"]
            x2 = bbox["x2"]
            y2 = bbox["y2"]

            bbox_valid = int(
                not (
                    x1 == 0
                    and y1 == 0
                    and x2 == 0
                    and y2 == 0
                )
            )

        writer.writerow({
            "frame_id": frame_id,
            "image": image_name,
            "label": label,
            "fall_start": fall_start,
            "fall_end": fall_end,
            "bbox_valid": bbox_valid,
            "target_id": target_id,
            "x1": x1,
            "y1": y1,
            "x2": x2,
            "y2": y2,
        })

        frame_id += 1
        saved_count += 1

    cap.release()
    csv_file.close()

    print(
        f"[DONE] saved={saved_count}, "
        f"csv={csv_path}"
    )

    # 检查 OpenCV 报告的帧数与实际读取帧数
    if (
        expected_frames > 0
        and expected_frames != saved_count
    ):
        print(
            f"[WARNING] Frame count mismatch: "
            f"OpenCV reports {expected_frames}, "
            f"actually read {saved_count}"
        )

    return True


# ============================================================
# main
# ============================================================

def main():

    OUTPUT_ROOT.mkdir(
        parents=True,
        exist_ok=True
    )

    # 找所有 Videos 文件夹里的 avi
    videos = sorted(
        INPUT_ROOT.rglob("Videos/*.avi")
    )

    print(f"Found {len(videos)} AVI videos.")

    success = 0
    failed = 0

    for index, video_path in enumerate(
        videos,
        start=1
    ):

        # 例如：
        #
        # raw_le2i/
        #   Coffee_room_01/
        #     Coffee_room_01/
        #       Videos/
        #         video (1).avi
        #
        # video_path.parent        -> Videos
        # video_path.parent.parent -> 内层 Coffee_room_01

        scene_name = video_path.parent.parent.name

        # 对应 annotation
        annotation_dir = (
            video_path.parent.parent
            / "Annotation_files"
        )

        annotation_path = (
            annotation_dir
            / f"{video_path.stem}.txt"
        )

        print(
            f"\n[{index}/{len(videos)}] "
            f"{scene_name} / {video_path.name}"
        )

        if not annotation_path.exists():

            print(
                "[ERROR] Annotation not found:\n"
                f"  Video: {video_path}\n"
                f"  Expected: {annotation_path}"
            )

            failed += 1
            continue

        try:

            ok = process_video(
                video_path,
                annotation_path,
                scene_name,
            )

            if ok:
                success += 1
            else:
                failed += 1

        except Exception as e:

            print(
                f"[ERROR] Failed processing "
                f"{video_path}: {e}"
            )

            failed += 1

    print("\n============================================")
    print("Finished")
    print(f"Success : {success}")
    print(f"Failed  : {failed}")
    print(f"Output  : {OUTPUT_ROOT}")
    print("============================================")


if __name__ == "__main__":
    main()