#!/bin/bash

# set -e

DATA_ROOT="data/keypoints_ur"
OUTPUT_ROOT="results_ur"
WINDOWS_CACHE="data/windows/keypoints_ur_windows.npz"

# DATA_ROOT="data/keypoints_normalized"
# OUTPUT_ROOT="results_normalized"

# DATA_ROOT="data/keypoints"
# OUTPUT_ROOT="results"

DEVICE="cuda:1"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --device)
      DEVICE="$2"
      shift 2
      ;;
    --device=*)
      DEVICE="${1#*=}"
      shift
      ;;
    *)
      echo "Unknown argument: $1" >&2
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
echo "Output: $WINDOWS_CACHE"
echo "========================================"

uv run python "src/prepare_windows_ur.py" \
  --data-root "$DATA_ROOT" \
  --output "$WINDOWS_CACHE"

for EXP in "${EXPERIMENTS[@]}"; do
  echo "========================================"
  echo "Running experiment: $EXP"
  echo "Device: $DEVICE"
  echo "========================================"

  if [[ "$EXP" == "rf" ]]; then
    uv run python "src/experiments/${EXP}.py" \
      --data-root "$DATA_ROOT" \
      --windows-cache "$WINDOWS_CACHE" \
      --output-root "$OUTPUT_ROOT"
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
echo "All experiments finished!"
echo "========================================"
