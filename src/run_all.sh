#!/bin/bash

# set -e

DATA_ROOT="data/keypoints_vitpose"
OUTPUT_ROOT="results_vitpose"

# DATA_ROOT="data/keypoints_normalized"
# OUTPUT_ROOT="results_normalized"

# DATA_ROOT="data/keypoints"
# OUTPUT_ROOT="results"

DEVICE="cuda:1"

EXPERIMENTS=(
  "rf"
  "mlp"
  "lstm"
  "stgcn"
)

for EXP in "${EXPERIMENTS[@]}"; do
  echo "========================================"
  echo "Running experiment: $EXP"
  echo "========================================"

  if [[ "$EXP" == "rf" ]]; then
    uv run python "src/experiments/${EXP}.py" \
      --data-root "$DATA_ROOT" \
      --output-root "$OUTPUT_ROOT"
  else
    uv run python "src/experiments/${EXP}.py" \
      --data-root "$DATA_ROOT" \
      --output-root "$OUTPUT_ROOT" \
      --device "$DEVICE"
  fi

  echo "$EXP finished."
  echo
done

echo "========================================"
echo "All experiments finished!"
echo "========================================"
