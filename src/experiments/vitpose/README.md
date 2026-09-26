# VitPose Experiments

This directory contains the VitPose entry point for the shared UR-Fall
experiment pipeline.

Expected NPZ input:

- `keypoints`: `[T,17,2]` or `[T,17,3]`
- optional `scores` or `keypoint_scores`: `[T,17]`
- optional `valid_mask`, `frame_indices`, `video_id`, `label`

Run:

```bash
bash src/experiments/vitpose/run_all.sh --data-root data/keypoints_vitpose --device auto
```

The script builds one shared cache with:

```bash
uv run python src/prepare_windows_ur.py \
  --data-root data/keypoints_vitpose \
  --output results/vitpose/windows/vitpose_windows.npz \
  --keypoint-adapter vitpose \
  --feature-mode xy
```

Then it runs the skeleton-agnostic experiments:

- `stgcn`
- `rf`
- `mlp`
- `lstm`
- `transformer`
- `blockgcn`

For VitPose, `stgcn` and `blockgcn` use a COCO-17 physical skeleton graph.
`blockgcn` still requires the official repository at `third_party/BlockGCN`
or a custom path passed to `src/experiments/blockgcn.py --blockgcn-root`.
