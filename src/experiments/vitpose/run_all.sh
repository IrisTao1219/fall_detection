#!/bin/bash

# Run all skeleton-agnostic experiments on VitPose keypoint NPZ files.
#
# Expected VitPose NPZ layout:
#   keypoints: [T,17,2] or [T,17,3]
#   optional scores/keypoint_scores: [T,17]
#   optional valid_mask, frame_indices, video_id, label

DATA_ROOT="data/keypoints_vitpose"
OUTPUT_ROOT="results/vitpose"
WINDOWS_CACHE="${OUTPUT_ROOT}/windows/vitpose_windows.npz"
DEVICE="cuda:1"
VISIBILITY_THRESHOLD="0.3"
FEATURE_MODE="xy"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --data-root)
      DATA_ROOT="$2"
      shift 2
      ;;
    --data-root=*)
      DATA_ROOT="${1#*=}"
      shift
      ;;
    --output-root)
      OUTPUT_ROOT="$2"
      shift 2
      ;;
    --output-root=*)
      OUTPUT_ROOT="${1#*=}"
      shift
      ;;
    --device)
      DEVICE="$2"
      shift 2
      ;;
    --device=*)
      DEVICE="${1#*=}"
      shift
      ;;
    --visibility-threshold)
      VISIBILITY_THRESHOLD="$2"
      shift 2
      ;;
    --visibility-threshold=*)
      VISIBILITY_THRESHOLD="${1#*=}"
      shift
      ;;
    *)
      echo "Unknown argument: $1" >&2
      exit 1
      ;;
  esac
done

WINDOWS_CACHE="${OUTPUT_ROOT}/windows/vitpose_windows.npz"

EXPERIMENTS=(
  "rf"
  "mlp"
  "lstm"
  "stgcn"
  "transformer"
)

echo "========================================"
echo "Preparing VitPose shared windows cache"
echo "Data root: $DATA_ROOT"
echo "Output: $WINDOWS_CACHE"
echo "========================================"

uv run python "src/experiments/prepare_windows.py" \
  --data-root "$DATA_ROOT" \
  --output "$WINDOWS_CACHE" \
  --keypoint-adapter vitpose \
  --feature-mode "$FEATURE_MODE" \
  --visibility-threshold "$VISIBILITY_THRESHOLD"

for EXP in "${EXPERIMENTS[@]}"; do
  echo "========================================"
  echo "Running VitPose experiment: $EXP"
  echo "Device: $DEVICE"
  echo "========================================"

  if [[ "$EXP" == "rf" ]]; then
    uv run python "src/experiments/${EXP}.py" \
      --data-root "$DATA_ROOT" \
      --windows-cache "$WINDOWS_CACHE" \
      --output-root "$OUTPUT_ROOT"
  elif [[ "$EXP" == "blockgcn" ]]; then
    uv run python "src/experiments/${EXP}.py" \
      --data-root "$DATA_ROOT" \
      --windows-cache "$WINDOWS_CACHE" \
      --output-root "$OUTPUT_ROOT" \
      --device "$DEVICE"
  elif [[ "$EXP" == "transformer" ]]; then
    uv run python "src/experiments/${EXP}.py" \
      --data-root "$DATA_ROOT" \
      --windows-cache "$WINDOWS_CACHE" \
      --output-root "$OUTPUT_ROOT" \
      --run-name "transformer_vitpose" \
      --pose-estimator "ViTPose" \
      --dataset-name "UR-Fall" \
      --device "$DEVICE"
  else
    uv run python "src/experiments/${EXP}.py" \
      --data-root "$DATA_ROOT" \
      --windows-cache "$WINDOWS_CACHE" \
      --output-root "$OUTPUT_ROOT" \
      --device "$DEVICE"
  fi

  echo "$EXP finished."
  echo
done

echo "========================================"
echo "VitPose experiments finished!"
echo "========================================"
