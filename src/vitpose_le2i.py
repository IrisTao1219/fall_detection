from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
from tqdm import tqdm
from mmpose.apis import MMPoseInferencer

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp"}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", type=Path, default=Path("data/raw_le2i/pngs"))
    p.add_argument("--output-root", type=Path, default=Path("data/keypoints_le2i_vitpose"))
    p.add_argument("--model", default="vitpose-b")
    p.add_argument("--device", default="cuda:1")
    p.add_argument("--fps", type=float, default=25.0)
    p.add_argument("--score-thr", type=float, default=0.3)
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def frame_index(path: Path) -> int:
    return int(path.stem)


def load_labels(path: Path):
    frame_labels = {}
    bboxes = {}
    bbox_valid = {}

    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            idx = int(row["frame_id"])
            frame_labels[idx] = int(row["label"])
            bboxes[idx] = [
                int(row.get("x1", 0)),
                int(row.get("y1", 0)),
                int(row.get("x2", 0)),
                int(row.get("y2", 0)),
            ]
            bbox_valid[idx] = int(row.get("bbox_valid", 0)) == 1

    return frame_labels, bboxes, bbox_valid


def person_area(person: dict) -> float:
    bbox = person.get("bbox")
    if bbox is not None:
        box = np.asarray(bbox, dtype=np.float32).reshape(-1)
        if len(box) >= 4:
            x1, y1, x2, y2 = box[:4]
            return max(0.0, x2 - x1) * max(0.0, y2 - y1)
    kpts = np.asarray(person.get("keypoints", []), dtype=np.float32)
    if kpts.ndim != 2 or len(kpts) == 0:
        return 0.0
    valid = np.isfinite(kpts[:, 0]) & np.isfinite(kpts[:, 1])
    if not valid.any():
        return 0.0
    return float((kpts[valid, 0].max() - kpts[valid, 0].min()) * (kpts[valid, 1].max() - kpts[valid, 1].min()))


def infer_frame(inferencer, image_path: Path, score_thr: float):
    result = next(inferencer(str(image_path), return_vis=False, show=False))
    predictions = result.get("predictions", [])

    if not predictions or not predictions[0]:
        return np.full((17, 3), np.nan, dtype=np.float32), np.zeros(17, dtype=bool)

    person = max(predictions[0], key=person_area)

    xy = np.asarray(person["keypoints"], dtype=np.float32)
    scores = np.asarray(person["keypoint_scores"], dtype=np.float32).reshape(-1)

    keypoints = np.concatenate([xy, scores[:, None]], axis=1)
    valid_mask = np.isfinite(xy).all(axis=1) & (scores >= score_thr)

    return keypoints.astype(np.float32), valid_mask.astype(bool)


def process_video(inferencer, video_dir: Path, output_path: Path, fps: float, score_thr: float):
    label_csv = video_dir / "labels.csv"
    if not label_csv.exists():
        print(f"[SKIP] missing labels.csv: {video_dir}")
        return

    frame_paths = sorted(
        [p for p in video_dir.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES],
        key=frame_index,
    )

    if not frame_paths:
        print(f"[SKIP] no frames: {video_dir}")
        return

    labels, bboxes, bbox_valid = load_labels(label_csv)

    keypoints = []
    valid_masks = []
    frame_indices = []
    frame_labels = []
    bbox_array = []
    bbox_valid_array = []

    for frame_path in tqdm(frame_paths, desc=video_dir.name):
        idx = frame_index(frame_path)
        kpts, mask = infer_frame(inferencer, frame_path, score_thr)

        keypoints.append(kpts)
        valid_masks.append(mask)
        frame_indices.append(idx)
        frame_labels.append(labels.get(idx, -1))
        bbox_array.append(bboxes.get(idx, [0, 0, 0, 0]))
        bbox_valid_array.append(bbox_valid.get(idx, False))

    frame_indices = np.asarray(frame_indices, dtype=np.int32)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        keypoints=np.asarray(keypoints, dtype=np.float32),
        valid_mask=np.asarray(valid_masks, dtype=bool),
        frame_indices=frame_indices,
        timestamps=(frame_indices - 1).astype(np.float32) / fps,
        frame_labels=np.asarray(frame_labels, dtype=np.int8),
        bboxes=np.asarray(bbox_array, dtype=np.int32),
        bbox_valid_mask=np.asarray(bbox_valid_array, dtype=bool),
        fps=np.float32(fps),
        video_id=np.asarray(video_dir.name),
    )

    print(f"[OK] {video_dir.name} -> {output_path}")


def main():
    args = parse_args()
    inferencer = MMPoseInferencer(pose2d=args.model, device=args.device)

    videos = sorted([p for p in args.data_root.iterdir() if p.is_dir()])
    print(f"Found {len(videos)} Le2i videos")

    for video_dir in videos:
        out = args.output_root / f"{video_dir.name}.npz"
        if out.exists() and not args.overwrite:
            print(f"[SKIP] exists: {out}")
            continue
        process_video(inferencer, video_dir, out, args.fps, args.score_thr)


if __name__ == "__main__":
    main()