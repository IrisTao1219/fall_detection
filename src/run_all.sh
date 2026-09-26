#!/bin/bash

set -euo pipefail

# Default profile matches the current results layout:
#   results/ur/{rf,mlp,lstm,stgcn}
#
# Other built-in profiles:
#   --keypoints ur-origin            -> results/ur_origin/{rf,mlp,lstm,stgcn}
#   --keypoints blazepose-normalized -> results/blazepose_normalized/{model}/{model_normalized}
#   --keypoints le2i-blazepose       -> results/le2i_blazepose/{rf,mlp,lstm,stgcn}
KEYPOINTS_PROFILE="ur"
DATA_ROOT="data/keypoints_ur"
OUTPUT_ROOT="results/ur"
WINDOWS_CACHE="data/windows/keypoints_ur_windows.npz"
DEVICE="cuda:1"
WINDOW_SIZE="45"
STRIDE="15"
NEST_OUTPUT_BY_MODEL="0"

usage() {
  cat <<'EOF'
Usage: bash src/run_all.sh [options]

Options:
  --keypoints NAME        Built-in profile: ur, ur-origin, blazepose-normalized, le2i-blazepose
  --data-root PATH        Override keypoint NPZ root
  --output-root PATH      Override output root
  --windows-cache PATH    Override shared windows cache path
  --device DEVICE         Torch device, e.g. cuda:1, cuda:0, cpu
  --window-size N         Window length passed to prepare_windows_ur.py
  --stride N              Window stride passed to prepare_windows_ur.py
  --nested-output         Pass results root as OUTPUT_ROOT/model for each experiment
  -h, --help              Show this help

Current default:
  data/keypoints_ur -> results/ur/{rf,mlp,lstm,stgcn}
EOF
}

apply_keypoints_profile() {
  case "$KEYPOINTS_PROFILE" in
    ur)
      DATA_ROOT="data/keypoints_ur"
      OUTPUT_ROOT="results/ur"
      WINDOWS_CACHE="data/windows/keypoints_ur_windows.npz"
      WINDOW_SIZE="45"
      STRIDE="15"
      NEST_OUTPUT_BY_MODEL="0"
      ;;
    ur-origin)
      DATA_ROOT="data/keypoints"
      OUTPUT_ROOT="results/ur_origin"
      WINDOWS_CACHE="data/windows/keypoints_windows.npz"
      WINDOW_SIZE="30"
      STRIDE="1"
      NEST_OUTPUT_BY_MODEL="0"
      ;;
    blazepose-normalized)
      DATA_ROOT="data/keypoints_ur_normalized"
      OUTPUT_ROOT="results/blazepose_normalized"
      WINDOWS_CACHE="data/windows/keypoints_ur_normalized_windows.npz"
      WINDOW_SIZE="30"
      STRIDE="1"
      NEST_OUTPUT_BY_MODEL="1"
      ;;
    le2i|le2i-blazepose|blazepose-le2i)
      DATA_ROOT="data/keypoints_le2i"
      OUTPUT_ROOT="results/le2i_blazepose"
      WINDOWS_CACHE="data/windows/keypoints_le2i_windows.npz"
      WINDOW_SIZE="25"
      STRIDE="5"
      NEST_OUTPUT_BY_MODEL="0"
      ;;
    *)
      echo "Unknown keypoints profile: $KEYPOINTS_PROFILE" >&2
      usage >&2
      exit 1
      ;;
  esac
}

apply_keypoints_profile

while [[ $# -gt 0 ]]; do
  case "$1" in
    --keypoints)
      KEYPOINTS_PROFILE="$2"
      apply_keypoints_profile
      shift 2
      ;;
    --keypoints=*)
      KEYPOINTS_PROFILE="${1#*=}"
      apply_keypoints_profile
      shift
      ;;
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
    --windows-cache)
      WINDOWS_CACHE="$2"
      shift 2
      ;;
    --windows-cache=*)
      WINDOWS_CACHE="${1#*=}"
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
    --window-size)
      WINDOW_SIZE="$2"
      shift 2
      ;;
    --window-size=*)
      WINDOW_SIZE="${1#*=}"
      shift
      ;;
    --stride)
      STRIDE="$2"
      shift 2
      ;;
    --stride=*)
      STRIDE="${1#*=}"
      shift
      ;;
    --nested-output)
      NEST_OUTPUT_BY_MODEL="1"
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

EXPERIMENTS=(
  "rf"
  "mlp"
  "lstm"
  "stgcn"
)

echo "========================================"
echo "Preparing shared windows cache"
echo "Keypoints profile: $KEYPOINTS_PROFILE"
echo "Data root: $DATA_ROOT"
echo "Output: $WINDOWS_CACHE"
echo "Window size: $WINDOW_SIZE"
echo "Stride: $STRIDE"
echo "========================================"

uv run python "src/prepare_windows_ur.py" \
  --data-root "$DATA_ROOT" \
  --output "$WINDOWS_CACHE" \
  --window-size "$WINDOW_SIZE" \
  --stride "$STRIDE"

for EXP in "${EXPERIMENTS[@]}"; do
  EXP_OUTPUT_ROOT="$OUTPUT_ROOT"
  if [[ "$NEST_OUTPUT_BY_MODEL" == "1" ]]; then
    EXP_OUTPUT_ROOT="$OUTPUT_ROOT/$EXP"
  fi

  echo "========================================"
  echo "Running experiment: $EXP"
  echo "Device: $DEVICE"
  echo "Output root: $EXP_OUTPUT_ROOT"
  echo "========================================"

  if [[ "$EXP" == "rf" ]]; then
    uv run python "src/experiments/${EXP}.py" \
      --data-root "$DATA_ROOT" \
      --windows-cache "$WINDOWS_CACHE" \
      --output-root "$EXP_OUTPUT_ROOT"
  else
    uv run python "src/experiments/${EXP}.py" \
      --data-root "$DATA_ROOT" \
      --windows-cache "$WINDOWS_CACHE" \
      --output-root "$EXP_OUTPUT_ROOT" \
      --device "$DEVICE"
  fi

  echo "$EXP finished."
  echo
done

echo "========================================"
echo "All experiments finished!"
echo "========================================"
