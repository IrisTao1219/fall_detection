from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
from tqdm import tqdm

from mmpose.apis import MMPoseInferencer


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp"}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Extract UR-Fall keypoints with ViTPose."
    )

    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("data/raw"),
        help="Root directory of extracted UR-Fall frames.",
    )

    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("data/keypoints_vitpose"),
        help="Directory for output NPZ files.",
    )

    parser.add_argument(
        "--model",
        type=str,
        default="vitpose-b",
        choices=[
            "vitpose",
            "vitpose-s",
            "vitpose-b",
            "vitpose-l",
            "vitpose-h",
        ],
        help="ViTPose model variant.",
    )

    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Inference device, e.g. cpu or cuda:0.",
    )

    parser.add_argument(
        "--fps",
        type=float,
        default=30.0,
        help="Video FPS.",
    )

    parser.add_argument(
        "--score-thr",
        type=float,
        default=0.3,
        help="Keypoint confidence threshold used for valid_mask.",
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing NPZ files.",
    )

    return parser.parse_args()


def frame_number(path: Path) -> int:
    """
    Extract the last integer from filename.

    Examples:
        adl-01-cam0-rgb-001.png -> 1
        fall-02-cam0-rgb-145.png -> 145
    """
    numbers = re.findall(r"\d+", path.stem)

    if not numbers:
        raise ValueError(f"Cannot parse frame number from: {path}")

    return int(numbers[-1])


def get_frame_paths(video_dir: Path) -> list[Path]:
    images = [
        p
        for p in video_dir.rglob("*")
        if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES
    ]

    return sorted(images, key=frame_number)


def get_person_area(person: dict) -> float:
    """
    Calculate person bounding-box area.

    If bbox is not available, estimate area using keypoints.
    """

    bbox = person.get("bbox")

    if bbox is not None:
        bbox = np.asarray(bbox, dtype=np.float32).reshape(-1)

        if len(bbox) >= 4:
            x1, y1, x2, y2 = bbox[:4]

            # Usually xyxy.
            width = max(0.0, x2 - x1)
            height = max(0.0, y2 - y1)

            if width > 0 and height > 0:
                return float(width * height)

    keypoints = np.asarray(person.get("keypoints", []), dtype=np.float32)

    if keypoints.ndim != 2 or keypoints.shape[0] == 0:
        return 0.0

    xs = keypoints[:, 0]
    ys = keypoints[:, 1]

    valid = np.isfinite(xs) & np.isfinite(ys)

    if not valid.any():
        return 0.0

    width = float(np.max(xs[valid]) - np.min(xs[valid]))
    height = float(np.max(ys[valid]) - np.min(ys[valid]))

    return max(width, 0.0) * max(height, 0.0)


def choose_main_person(persons: list[dict]) -> dict | None:
    """
    UR-Fall contains one main subject.
    If multiple people are detected, select the largest person.
    """

    if not persons:
        return None

    return max(persons, key=get_person_area)


def infer_frame(
    inferencer: MMPoseInferencer,
    image_path: Path,
    score_thr: float,
):
    """
    Run ViTPose on one frame.

    Returns
    -------
    keypoints:
        shape = (17, 3)
        each point = (x, y, score)

    valid_mask:
        shape = (17,)
    """

    result_generator = inferencer(
        str(image_path),
        return_vis=False,
        show=False,
    )

    result = next(result_generator)

    # batch size is 1:
    # predictions = [[person1, person2, ...]]
    predictions = result.get("predictions", [])

    if not predictions:
        return (
            np.full((17, 3), np.nan, dtype=np.float32),
            np.zeros(17, dtype=bool),
        )

    persons = predictions[0]

    person = choose_main_person(persons)

    if person is None:
        return (
            np.full((17, 3), np.nan, dtype=np.float32),
            np.zeros(17, dtype=bool),
        )

    xy = np.asarray(
        person["keypoints"],
        dtype=np.float32,
    )

    scores = np.asarray(
        person["keypoint_scores"],
        dtype=np.float32,
    ).reshape(-1)

    if xy.shape != (17, 2):
        raise RuntimeError(
            f"Unexpected ViTPose keypoint shape: {xy.shape}"
        )

    if scores.shape != (17,):
        raise RuntimeError(
            f"Unexpected keypoint score shape: {scores.shape}"
        )

    keypoints = np.concatenate(
        [
            xy,
            scores[:, None],
        ],
        axis=1,
    )

    valid_mask = (
        np.isfinite(xy[:, 0])
        & np.isfinite(xy[:, 1])
        & (scores >= score_thr)
    )

    return keypoints, valid_mask


def process_video(
    inferencer: MMPoseInferencer,
    video_dir: Path,
    output_path: Path,
    label: int,
    fps: float,
    score_thr: float,
):
    frames = get_frame_paths(video_dir)

    if len(frames) == 0:
        print(f"[SKIP] no frames: {video_dir}")
        return

    video_id = video_dir.name

    all_keypoints = []
    all_valid_masks = []
    frame_indices = []

    detected_frames = 0

    for image_path in tqdm(
        frames,
        desc=video_id,
        leave=False,
    ):
        keypoints, valid_mask = infer_frame(
            inferencer=inferencer,
            image_path=image_path,
            score_thr=score_thr,
        )

        all_keypoints.append(keypoints)
        all_valid_masks.append(valid_mask)

        idx = frame_number(image_path)
        frame_indices.append(idx)

        if valid_mask.any():
            detected_frames += 1

    keypoints = np.asarray(
        all_keypoints,
        dtype=np.float32,
    )

    valid_mask = np.asarray(
        all_valid_masks,
        dtype=bool,
    )

    frame_indices = np.asarray(
        frame_indices,
        dtype=np.int32,
    )

    # Relative timestamp in seconds.
    timestamps = np.arange(
        len(frames),
        dtype=np.float32,
    ) / fps

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    np.savez_compressed(
        output_path,

        # T x 17 x 3
        keypoints=keypoints,

        # T x 17
        valid_mask=valid_mask,

        # T
        frame_indices=frame_indices,

        # T
        timestamps=timestamps,

        fps=np.float32(fps),

        label=np.int64(label),

        video_id=np.asarray(video_id),
    )

    detection_rate = detected_frames / len(frames)

    print(
        f"[OK] {video_id:<25} "
        f"frames={len(frames):4d} | "
        f"detected={detected_frames:4d} | "
        f"rate={detection_rate:.2%} | "
        f"shape={keypoints.shape}"
    )


def find_video_dirs(data_root: Path):
    """
    Expected structure:

    data/raw/
    ├── adl/
    │   ├── adl-01-cam0-rgb/
    │   ├── adl-02-cam0-rgb/
    │   └── ...
    └── fall/
        ├── fall-01-cam0-rgb/
        ├── fall-02-cam0-rgb/
        └── ...
    """

    items = []

    for class_name, label in [
        ("adl", 0),
        ("fall", 1),
    ]:
        class_dir = data_root / class_name

        if not class_dir.exists():
            print(f"[WARN] missing directory: {class_dir}")
            continue

        for video_dir in sorted(class_dir.iterdir()):
            if video_dir.is_dir():
                items.append(
                    (
                        video_dir,
                        label,
                    )
                )

    return items


def main():
    args = parse_args()

    print("=" * 60)
    print("ViTPose Keypoint Extraction")
    print("=" * 60)
    print(f"data_root   : {args.data_root}")
    print(f"output_root : {args.output_root}")
    print(f"model       : {args.model}")
    print(f"device      : {args.device}")
    print(f"fps         : {args.fps}")
    print(f"score_thr   : {args.score_thr}")
    print("=" * 60)

    # Model is created only once.
    inferencer = MMPoseInferencer(
        pose2d=args.model,
        device=args.device,
    )

    videos = find_video_dirs(args.data_root)

    print(f"Found {len(videos)} videos")

    success = 0
    skipped = 0

    for video_dir, label in videos:

        output_path = (
            args.output_root
            / f"{video_dir.name}.npz"
        )

        if output_path.exists() and not args.overwrite:
            print(f"[SKIP] exists: {output_path}")
            skipped += 1
            continue

        process_video(
            inferencer=inferencer,
            video_dir=video_dir,
            output_path=output_path,
            label=label,
            fps=args.fps,
            score_thr=args.score_thr,
        )

        success += 1

    print("=" * 60)
    print("Finished")
    print(f"processed : {success}")
    print(f"skipped   : {skipped}")
    print("=" * 60)


if __name__ == "__main__":
    main()