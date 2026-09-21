#!/bin/bash

# set -e

DATA_ROOT="data/keypoints_normalized"
OUTPUT_ROOT="results_normalized"

EXPERIMENTS=(
  "stgcn"
  "rf"
  "mlp"
  "lstm"
)

for EXP in "${EXPERIMENTS[@]}"; do
  echo "========================================"
  echo "Running experiment: $EXP"
  echo "========================================"

  uv run python "src/experiments/${EXP}.py" \
    --data-root "$DATA_ROOT" \
    --output-root "${OUTPUT_ROOT}/${EXP}"

  echo "$EXP finished."
  echo
done

echo "========================================"
echo "All experiments finished!"
echo "========================================"