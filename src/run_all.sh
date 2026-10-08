#!/bin/bash

set -euo pipefail

# Default profile matches the current results layout:
#   results/ur/{rf,mlp,lstm,stgcn}
#
# Other built-in profiles:
#   --keypoints ur-origin            -> results/ur_origin/{rf,mlp,lstm,stgcn}
#   --keypoints blazepose-normalized -> results/blazepose_normalized/{model}/{model_normalized}
#   --keypoints le2i-blazepose       -> results/le2i_blazepose/{rf,mlp,lstm,stgcn}
#   --keypoints le2i-blazepose-normalized -> results/le2i_blazepose_normalized/{rf,mlp,lstm,stgcn}
#   --keypoints ur-vitpose           -> results/ur_vitpose/{rf,mlp,lstm,stgcn,transformer}
#   --keypoints le2i-vitpose         -> results/le2i_vitpose/{rf,mlp,lstm,stgcn,transformer}
KEYPOINTS_PROFILES=("ur")
DATA_ROOT="data/keypoints_ur_blazepose"
OUTPUT_ROOT="results/ur"
WINDOWS_CACHE="data/windows/keypoints_ur_windows.npz"
DEVICE="cuda:1"
WINDOW_SIZE="30"
STRIDE="6"
FPS="30"
WINDOW_SECONDS="1"
STRIDE_SECONDS="0.2"
NEST_OUTPUT_BY_MODEL="0"
KEYPOINT_ADAPTER="blazepose"
FEATURE_MODE="xy66"
FEATURE_MODE_OVERRIDE=""
VISIBILITY_THRESHOLD="0.3"
VARIANT=""
STGCN_ARCHITECTURE="baseline"
STGCN_GRAPH_PARTITION="uniform"
DATA_ROOT_OVERRIDE=""
OUTPUT_ROOT_OVERRIDE=""
WINDOWS_CACHE_OVERRIDE=""
WINDOW_SIZE_OVERRIDE=""
STRIDE_OVERRIDE=""
WINDOW_SECONDS_OVERRIDE=""
STRIDE_SECONDS_OVERRIDE=""
NEST_OUTPUT_BY_MODEL_OVERRIDE=""
COMBINE_DATASETS="0"

usage() {
  cat <<'EOF'
Usage: bash src/run_all.sh [options]

Options:
  --keypoints NAME[,NAME] Built-in profile(s): ur-blazepose, ur, ur-origin, blazepose-normalized, le2i-blazepose, le2i-blazepose-normalized, ur-vitpose, le2i-vitpose
  --data-root PATH        Override keypoint NPZ root
  --output-root PATH      Override output root
  --windows-cache PATH    Override shared windows cache path
  --device DEVICE         Torch device, e.g. cuda:1, cuda:0, cpu
  --window-size N         Window length passed to prepare_windows.py
  --stride N              Window stride passed to prepare_windows.py
  --window-seconds S      Window length in seconds for combined dataset preparation
  --stride-seconds S      Window stride in seconds for combined dataset preparation
  --feature-mode MODE     Override prepared features: xy, xy66, or xyc
  --variant NAME          Add a suffix to cache/result paths, e.g. xyc
  --stgcn-architecture A  ST-GCN architecture: baseline, lite, coco17-lite
  --stgcn-graph-partition P
                          ST-GCN graph: uniform or spatial
  --combine-datasets      Combine multiple --keypoints profiles into one training/evaluation cache
  --nested-output         Pass results root as OUTPUT_ROOT/model for each experiment
  -h, --help              Show this help

Current default:
  data/keypoints_ur_blazepose -> results/ur/{rf,mlp,lstm,stgcn}

Examples:
  bash src/run_all.sh --keypoints ur-blazepose,le2i-blazepose --combine-datasets
  bash src/run_all.sh --keypoints ur-blazepose,le2i-blazepose --combine-datasets --feature-mode xyc --variant xyc --stgcn-graph-partition spatial
  bash src/run_all.sh --keypoints ur-vitpose,le2i-vitpose --combine-datasets
  bash src/run_all.sh --keypoints ur --device cuda:0
EOF
}

apply_keypoints_profile() {
  local profile="$1"
  case "$profile" in
    ur)
      DATA_ROOT="data/keypoints_ur_blazepose"
      OUTPUT_ROOT="results/ur"
      WINDOWS_CACHE="data/windows/keypoints_ur_windows.npz"
      WINDOW_SIZE="30"
      STRIDE="6"
      FPS="30"
      NEST_OUTPUT_BY_MODEL="0"
      KEYPOINT_ADAPTER="blazepose"
      FEATURE_MODE="xy66"
      VISIBILITY_THRESHOLD="0.3"
      ;;
    ur-blazepose|blazepose-ur)
      DATA_ROOT="data/keypoints_ur_blazepose"
      OUTPUT_ROOT="results/ur_blazepose"
      WINDOWS_CACHE="data/windows/keypoints_ur_blazepose_windows.npz"
      WINDOW_SIZE="30"
      STRIDE="6"
      FPS="30"
      NEST_OUTPUT_BY_MODEL="0"
      KEYPOINT_ADAPTER="blazepose"
      FEATURE_MODE="xy66"
      VISIBILITY_THRESHOLD="0.3"
      ;;
    ur-origin)
      DATA_ROOT="data/keypoints"
      OUTPUT_ROOT="results/ur_origin"
      WINDOWS_CACHE="data/windows/keypoints_windows.npz"
      WINDOW_SIZE="30"
      STRIDE="6"
      FPS="30"
      NEST_OUTPUT_BY_MODEL="0"
      KEYPOINT_ADAPTER="blazepose"
      FEATURE_MODE="xy66"
      VISIBILITY_THRESHOLD="0.3"
      ;;
    blazepose-normalized)
      DATA_ROOT="data/keypoints_ur_normalized"
      OUTPUT_ROOT="results/blazepose_normalized"
      WINDOWS_CACHE="data/windows/keypoints_ur_normalized_windows.npz"
      WINDOW_SIZE="30"
      STRIDE="6"
      FPS="30"
      NEST_OUTPUT_BY_MODEL="1"
      KEYPOINT_ADAPTER="blazepose"
      FEATURE_MODE="xy66"
      VISIBILITY_THRESHOLD="0.3"
      ;;
    ur-vitpose|vitpose-ur)
      DATA_ROOT="data/keypoints_ur_vitpose"
      OUTPUT_ROOT="results/ur_vitpose"
      WINDOWS_CACHE="data/windows/keypoints_ur_vitpose_windows.npz"
      WINDOW_SIZE="30"
      STRIDE="6"
      FPS="30"
      NEST_OUTPUT_BY_MODEL="0"
      KEYPOINT_ADAPTER="vitpose"
      FEATURE_MODE="xy"
      VISIBILITY_THRESHOLD="0.3"
      ;;
    le2i|le2i-blazepose|blazepose-le2i)
      DATA_ROOT="data/keypoints_le2i_blazepose"
      OUTPUT_ROOT="results/le2i_blazepose"
      WINDOWS_CACHE="data/windows/keypoints_le2i_windows.npz"
      WINDOW_SIZE="25"
      STRIDE="5"
      FPS="25"
      NEST_OUTPUT_BY_MODEL="0"
      KEYPOINT_ADAPTER="blazepose"
      FEATURE_MODE="xy66"
      VISIBILITY_THRESHOLD="0.3"
      ;;
    le2i-normalized|le2i-blazepose-normalized|blazepose-le2i-normalized)
      DATA_ROOT="data/keypoints_le2i_normalized"
      OUTPUT_ROOT="results/le2i_blazepose_normalized"
      WINDOWS_CACHE="data/windows/keypoints_le2i_normalized_windows.npz"
      WINDOW_SIZE="25"
      STRIDE="5"
      FPS="25"
      NEST_OUTPUT_BY_MODEL="0"
      KEYPOINT_ADAPTER="blazepose"
      FEATURE_MODE="xy66"
      VISIBILITY_THRESHOLD="0.3"
      ;;
    le2i-vitpose|vitpose-le2i)
      DATA_ROOT="data/keypoints_le2i_vitpose"
      OUTPUT_ROOT="results/le2i_vitpose"
      WINDOWS_CACHE="data/windows/keypoints_le2i_vitpose_windows.npz"
      WINDOW_SIZE="25"
      STRIDE="5"
      FPS="25"
      NEST_OUTPUT_BY_MODEL="0"
      KEYPOINT_ADAPTER="vitpose"
      FEATURE_MODE="xy"
      VISIBILITY_THRESHOLD="0.3"
      ;;
    *)
      echo "Unknown keypoints profile: $profile" >&2
      usage >&2
      exit 1
      ;;
  esac
}

set_keypoints_profiles() {
  local raw="$1"
  IFS=',' read -r -a KEYPOINTS_PROFILES <<< "$raw"
  local profile
  for profile in "${KEYPOINTS_PROFILES[@]}"; do
    if [[ -z "$profile" ]]; then
      echo "--keypoints contains an empty profile: $raw" >&2
      exit 1
    fi
  done
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --keypoints)
      set_keypoints_profiles "$2"
      shift 2
      ;;
    --keypoints=*)
      set_keypoints_profiles "${1#*=}"
      shift
      ;;
    --data-root)
      DATA_ROOT_OVERRIDE="$2"
      shift 2
      ;;
    --data-root=*)
      DATA_ROOT_OVERRIDE="${1#*=}"
      shift
      ;;
    --output-root)
      OUTPUT_ROOT_OVERRIDE="$2"
      shift 2
      ;;
    --output-root=*)
      OUTPUT_ROOT_OVERRIDE="${1#*=}"
      shift
      ;;
    --windows-cache)
      WINDOWS_CACHE_OVERRIDE="$2"
      shift 2
      ;;
    --windows-cache=*)
      WINDOWS_CACHE_OVERRIDE="${1#*=}"
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
      WINDOW_SIZE_OVERRIDE="$2"
      shift 2
      ;;
    --window-size=*)
      WINDOW_SIZE_OVERRIDE="${1#*=}"
      shift
      ;;
    --stride)
      STRIDE_OVERRIDE="$2"
      shift 2
      ;;
    --stride=*)
      STRIDE_OVERRIDE="${1#*=}"
      shift
      ;;
    --window-seconds)
      WINDOW_SECONDS_OVERRIDE="$2"
      shift 2
      ;;
    --window-seconds=*)
      WINDOW_SECONDS_OVERRIDE="${1#*=}"
      shift
      ;;
    --stride-seconds)
      STRIDE_SECONDS_OVERRIDE="$2"
      shift 2
      ;;
    --stride-seconds=*)
      STRIDE_SECONDS_OVERRIDE="${1#*=}"
      shift
      ;;
    --feature-mode)
      FEATURE_MODE_OVERRIDE="$2"
      shift 2
      ;;
    --feature-mode=*)
      FEATURE_MODE_OVERRIDE="${1#*=}"
      shift
      ;;
    --variant)
      VARIANT="$2"
      shift 2
      ;;
    --variant=*)
      VARIANT="${1#*=}"
      shift
      ;;
    --stgcn-architecture)
      STGCN_ARCHITECTURE="$2"
      shift 2
      ;;
    --stgcn-architecture=*)
      STGCN_ARCHITECTURE="${1#*=}"
      shift
      ;;
    --stgcn-graph-partition)
      STGCN_GRAPH_PARTITION="$2"
      shift 2
      ;;
    --stgcn-graph-partition=*)
      STGCN_GRAPH_PARTITION="${1#*=}"
      shift
      ;;
    --nested-output)
      NEST_OUTPUT_BY_MODEL_OVERRIDE="1"
      shift
      ;;
    --combine-datasets)
      COMBINE_DATASETS="1"
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

case "$FEATURE_MODE_OVERRIDE" in
  ""|xy|xy66|xy_flat|xyc|xyc_flat) ;;
  *) echo "Unsupported --feature-mode: $FEATURE_MODE_OVERRIDE" >&2; exit 1 ;;
esac
case "$STGCN_ARCHITECTURE" in
  baseline|lite|coco17-lite) ;;
  *) echo "Unsupported --stgcn-architecture: $STGCN_ARCHITECTURE" >&2; exit 1 ;;
esac
case "$STGCN_GRAPH_PARTITION" in
  uniform|spatial) ;;
  *) echo "Unsupported --stgcn-graph-partition: $STGCN_GRAPH_PARTITION" >&2; exit 1 ;;
esac
if [[ -n "$VARIANT" && ! "$VARIANT" =~ ^[A-Za-z0-9_-]+$ ]]; then
  echo "--variant may contain only letters, digits, underscore, and hyphen." >&2
  exit 1
fi

if [[ "${#KEYPOINTS_PROFILES[@]}" -gt 1 ]]; then
  if [[ -n "$DATA_ROOT_OVERRIDE" || -n "$OUTPUT_ROOT_OVERRIDE" || -n "$WINDOWS_CACHE_OVERRIDE" ]]; then
    echo "--data-root, --output-root and --windows-cache are only supported with one --keypoints profile." >&2
    echo "Use built-in profiles for multi-dataset runs, e.g. --keypoints ur-blazepose,le2i-blazepose --combine-datasets." >&2
    exit 1
  fi
fi

EXPERIMENTS=(
  "rf"
  "mlp"
  "lstm"
  "stgcn"
  "transformer"
)

run_profile() {
  local profile="$1"
  apply_keypoints_profile "$profile"

  if [[ -n "$DATA_ROOT_OVERRIDE" ]]; then
    DATA_ROOT="$DATA_ROOT_OVERRIDE"
  fi
  if [[ -n "$OUTPUT_ROOT_OVERRIDE" ]]; then
    OUTPUT_ROOT="$OUTPUT_ROOT_OVERRIDE"
  fi
  if [[ -n "$WINDOWS_CACHE_OVERRIDE" ]]; then
    WINDOWS_CACHE="$WINDOWS_CACHE_OVERRIDE"
  fi
  if [[ -n "$WINDOW_SIZE_OVERRIDE" ]]; then
    WINDOW_SIZE="$WINDOW_SIZE_OVERRIDE"
  fi
  if [[ -n "$STRIDE_OVERRIDE" ]]; then
    STRIDE="$STRIDE_OVERRIDE"
  fi
  if [[ -n "$NEST_OUTPUT_BY_MODEL_OVERRIDE" ]]; then
    NEST_OUTPUT_BY_MODEL="$NEST_OUTPUT_BY_MODEL_OVERRIDE"
  fi
  if [[ -n "$FEATURE_MODE_OVERRIDE" ]]; then
    FEATURE_MODE="$FEATURE_MODE_OVERRIDE"
  fi
  if [[ -n "$VARIANT" ]]; then
    local variant_slug
    variant_slug="${VARIANT//-/_}"
    OUTPUT_ROOT="${OUTPUT_ROOT}_${variant_slug}"
    WINDOWS_CACHE="${WINDOWS_CACHE%.npz}_${variant_slug}.npz"
  fi

  echo "========================================"
  echo "Preparing shared windows cache"
  echo "Keypoints profile: $profile"
  echo "Data root: $DATA_ROOT"
  echo "Output: $WINDOWS_CACHE"
  echo "Window size: $WINDOW_SIZE"
  echo "Stride: $STRIDE"
  echo "Keypoint adapter: $KEYPOINT_ADAPTER"
  echo "Feature mode: $FEATURE_MODE"
  echo "========================================"

  uv run python "src/prepare_windows.py" \
    --data-root "$DATA_ROOT" \
    --output "$WINDOWS_CACHE" \
    --window-size "$WINDOW_SIZE" \
    --stride "$STRIDE" \
    --keypoint-adapter "$KEYPOINT_ADAPTER" \
    --feature-mode "$FEATURE_MODE" \
    --visibility-threshold "$VISIBILITY_THRESHOLD"

  for EXP in "${EXPERIMENTS[@]}"; do
    EXP_OUTPUT_ROOT="$OUTPUT_ROOT"
    if [[ "$NEST_OUTPUT_BY_MODEL" == "1" ]]; then
      EXP_OUTPUT_ROOT="$OUTPUT_ROOT/$EXP"
    fi

    echo "========================================"
    echo "Running experiment: $EXP"
    echo "Keypoints profile: $profile"
    echo "Device: $DEVICE"
    echo "Output root: $EXP_OUTPUT_ROOT"
    echo "========================================"

    if [[ "$EXP" == "rf" ]]; then
      uv run python "src/experiments/${EXP}.py" \
        --data-root "$DATA_ROOT" \
        --windows-cache "$WINDOWS_CACHE" \
        --output-root "$EXP_OUTPUT_ROOT" \
        --window-size "$WINDOW_SIZE" \
        --stride "$STRIDE"
    elif [[ "$EXP" == "stgcn" ]]; then
      uv run python "src/experiments/${EXP}.py" \
        --data-root "$DATA_ROOT" \
        --windows-cache "$WINDOWS_CACHE" \
        --output-root "$EXP_OUTPUT_ROOT" \
        --window-size "$WINDOW_SIZE" \
        --stride "$STRIDE" \
        --architecture "$STGCN_ARCHITECTURE" \
        --graph-partition "$STGCN_GRAPH_PARTITION" \
        --device "$DEVICE"
    elif [[ "$EXP" == "transformer" ]]; then
      uv run python "src/experiments/${EXP}.py" \
        --data-root "$DATA_ROOT" \
        --windows-cache "$WINDOWS_CACHE" \
        --output-root "$EXP_OUTPUT_ROOT" \
        --window-size "$WINDOW_SIZE" \
        --stride "$STRIDE" \
        --feature-mode "$(transformer_feature_mode "$FEATURE_MODE")" \
        --device "$DEVICE"
    else
      uv run python "src/experiments/${EXP}.py" \
        --data-root "$DATA_ROOT" \
        --windows-cache "$WINDOWS_CACHE" \
        --output-root "$EXP_OUTPUT_ROOT" \
        --window-size "$WINDOW_SIZE" \
        --stride "$STRIDE" \
        --device "$DEVICE"
    fi

    echo "$EXP finished for $profile."
    echo
  done
}

profile_slug() {
  local text="$1"
  text="${text//,/_}"
  text="${text//-/_}"
  echo "$text"
}

transformer_feature_mode() {
  case "$1" in
    xyc|xyc_flat) echo "xyc" ;;
    *) echo "xy" ;;
  esac
}

run_experiments_once() {
  local profile="$1"
  local data_root="$2"
  local windows_cache="$3"
  local output_root="$4"
  local nest_output_by_model="$5"
  local window_size="$6"
  local stride="$7"

  for EXP in "${EXPERIMENTS[@]}"; do
    EXP_OUTPUT_ROOT="$output_root"
    if [[ "$nest_output_by_model" == "1" ]]; then
      EXP_OUTPUT_ROOT="$output_root/$EXP"
    fi

    echo "========================================"
    echo "Running experiment: $EXP"
    echo "Keypoints profile: $profile"
    echo "Device: $DEVICE"
    echo "Output root: $EXP_OUTPUT_ROOT"
    echo "========================================"

    if [[ "$EXP" == "rf" ]]; then
      uv run python "src/experiments/${EXP}.py" \
        --data-root "$data_root" \
        --windows-cache "$windows_cache" \
        --output-root "$EXP_OUTPUT_ROOT" \
        --window-size "$window_size" \
        --stride "$stride"
    elif [[ "$EXP" == "stgcn" ]]; then
      uv run python "src/experiments/${EXP}.py" \
        --data-root "$data_root" \
        --windows-cache "$windows_cache" \
        --output-root "$EXP_OUTPUT_ROOT" \
        --window-size "$window_size" \
        --stride "$stride" \
        --architecture "$STGCN_ARCHITECTURE" \
        --graph-partition "$STGCN_GRAPH_PARTITION" \
        --device "$DEVICE"
    elif [[ "$EXP" == "transformer" ]]; then
      uv run python "src/experiments/${EXP}.py" \
        --data-root "$data_root" \
        --windows-cache "$windows_cache" \
        --output-root "$EXP_OUTPUT_ROOT" \
        --window-size "$window_size" \
        --stride "$stride" \
        --feature-mode "$(transformer_feature_mode "$FEATURE_MODE")" \
        --device "$DEVICE"
    else
      uv run python "src/experiments/${EXP}.py" \
        --data-root "$data_root" \
        --windows-cache "$windows_cache" \
        --output-root "$EXP_OUTPUT_ROOT" \
        --window-size "$window_size" \
        --stride "$stride" \
        --device "$DEVICE"
    fi

    echo "$EXP finished for $profile."
    echo
  done
}

run_combined_profiles() {
  if [[ "${#KEYPOINTS_PROFILES[@]}" -lt 2 ]]; then
    echo "--combine-datasets requires at least two --keypoints profiles." >&2
    exit 1
  fi

  local joined_profiles
  joined_profiles="$(IFS=,; echo "${KEYPOINTS_PROFILES[*]}")"
  local slug
  slug="$(profile_slug "$joined_profiles")"
  if [[ -n "$VARIANT" ]]; then
    slug="${slug}_$(profile_slug "$VARIANT")"
  fi
  local common_window_seconds="${WINDOW_SECONDS_OVERRIDE:-$WINDOW_SECONDS}"
  local common_stride_seconds="${STRIDE_SECONDS_OVERRIDE:-$STRIDE_SECONDS}"
  local combined_cache="data/windows/combined_${slug}_windows.npz"
  local combined_output_root="results/combined_${slug}"
  local combined_data_root="data/combined_${slug}"
  local combine_args=()
  local combined_window_size=0
  local combined_stride=0

  echo "========================================"
  echo "Preparing combined dataset windows"
  echo "Profiles: $joined_profiles"
  echo "Common window seconds: $common_window_seconds"
  echo "Common stride seconds: $common_stride_seconds"
  echo "Combined cache: $combined_cache"
  echo "Combined output root: $combined_output_root"
  echo "========================================"

  for PROFILE in "${KEYPOINTS_PROFILES[@]}"; do
    apply_keypoints_profile "$PROFILE"
    if [[ -n "$FEATURE_MODE_OVERRIDE" ]]; then
      FEATURE_MODE="$FEATURE_MODE_OVERRIDE"
    fi
    WINDOW_SIZE="$(python3 -c "print(max(1, round(float('$common_window_seconds') * float('$FPS'))))")"
    STRIDE="$(python3 -c "print(max(1, round(float('$common_stride_seconds') * float('$FPS'))))")"
    if [[ -n "$WINDOW_SIZE_OVERRIDE" ]]; then
      WINDOW_SIZE="$WINDOW_SIZE_OVERRIDE"
    fi
    if [[ -n "$STRIDE_OVERRIDE" ]]; then
      STRIDE="$STRIDE_OVERRIDE"
    fi
    if (( WINDOW_SIZE > combined_window_size )); then
      combined_window_size="$WINDOW_SIZE"
    fi
    if (( STRIDE > combined_stride )); then
      combined_stride="$STRIDE"
    fi
    WINDOWS_CACHE="data/windows/${PROFILE}_for_${slug}_windows.npz"

    echo "----------------------------------------"
    echo "Preparing profile for combined cache: $PROFILE"
    echo "Data root: $DATA_ROOT"
    echo "Output: $WINDOWS_CACHE"
    echo "Window size: $WINDOW_SIZE"
    echo "Stride: $STRIDE"
    echo "FPS: $FPS"
    echo "Keypoint adapter: $KEYPOINT_ADAPTER"
    echo "Feature mode: $FEATURE_MODE"
    echo "----------------------------------------"

    uv run python "src/prepare_windows.py" \
      --data-root "$DATA_ROOT" \
      --output "$WINDOWS_CACHE" \
      --window-size "$WINDOW_SIZE" \
      --stride "$STRIDE" \
      --keypoint-adapter "$KEYPOINT_ADAPTER" \
      --feature-mode "$FEATURE_MODE" \
      --visibility-threshold "$VISIBILITY_THRESHOLD"

    combine_args+=(--input "$PROFILE" "$WINDOWS_CACHE")
  done

  uv run python "src/combine_windows.py" \
    --output "$combined_cache" \
    "${combine_args[@]}"

  run_experiments_once \
    "combined:${joined_profiles}" \
    "$combined_data_root" \
    "$combined_cache" \
    "$combined_output_root" \
    "0" \
    "$combined_window_size" \
    "$combined_stride"
}

if [[ "$COMBINE_DATASETS" == "1" ]]; then
  run_combined_profiles
else
  for PROFILE in "${KEYPOINTS_PROFILES[@]}"; do
    run_profile "$PROFILE"
  done
fi

echo "========================================"
echo "All experiments finished!"
echo "========================================"
