#!/usr/bin/env bash
set -euo pipefail

# Launch Robometer AI2 export jobs on Beaker (no eval server dependency).
#
# Usage:
#   bash scripts/run_ai2_robometer_beaker.sh [dataset_name ...]
#
# Examples:
#   bash scripts/run_ai2_robometer_beaker.sh lerobot_annotations
#   bash scripts/run_ai2_robometer_beaker.sh lerobot_annotations lerobot_failed
#
# Supported dataset names:
#   lerobot_annotations lerobot_failed franka_data single_YAM bimanual_YAM_annotations

if [[ $# -gt 0 ]]; then
  DATASETS=("$@")
else
  DATASETS=("lerobot_annotations")
fi

BEAKER_IMAGE="${BEAKER_IMAGE:-shiruic/shirui-torch2.8.0_cuda12.8}"
WORKSPACE="${WORKSPACE:-ai2/molmo-act}"
CLUSTER="${CLUSTER:-ai2/titan}"
WEKA_MOUNT="${WEKA_MOUNT:-oe-training-default:/mount/weka}"
PRIORITY="${PRIORITY:-urgent}"
GPUS="${GPUS:-1}"

MODEL_PATH="${MODEL_PATH:-aliangdw/Robometer-4B}"
DATASETS_ROOT="${DATASETS_ROOT:-/mount/weka/shiruic/instruction_gvl/datasets}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/mount/weka/shiruic/robometer/baseline_eval_output}"
MAX_INPUT_FRAMES="${MAX_INPUT_FRAMES:-20}"
REQUEST_RETRIES="${REQUEST_RETRIES:-2}"
TIMEOUT_S="${TIMEOUT_S:-240}"

USE_FULL_VIDEO="${USE_FULL_VIDEO:-1}"
RESUME="${RESUME:-1}"
FSYNC_EACH_RECORD="${FSYNC_EACH_RECORD:-1}"
RUN_ANALYSIS="${RUN_ANALYSIS:-1}"

HF_TOKEN_SECRET="${HF_TOKEN_SECRET:-}"

for dataset in "${DATASETS[@]}"; do
  timestamp="$(date +%Y%m%d_%H%M%S)"
  run_name="robometer_ai2_${dataset}_${timestamp}"
  beaker_name="${run_name:0:123}"
  output_dir="${OUTPUT_ROOT}/${run_name}"

  echo "Submitting ${run_name}"
  echo "  dataset: ${dataset}"
  echo "  output:  ${output_dir}"

  eval_cmd="uv run python scripts/eval_ai2_robometer_export.py"
  eval_cmd+=" --inference-mode local"
  eval_cmd+=" --model-path \"${MODEL_PATH}\""
  eval_cmd+=" --datasets-root \"${DATASETS_ROOT}\""
  eval_cmd+=" --dataset-names \"${dataset}\""
  eval_cmd+=" --max-input-frames \"${MAX_INPUT_FRAMES}\""
  eval_cmd+=" --request-retries \"${REQUEST_RETRIES}\""
  eval_cmd+=" --timeout-s \"${TIMEOUT_S}\""
  eval_cmd+=" --output-dir \"${output_dir}\""

  if [[ "${USE_FULL_VIDEO}" == "1" ]]; then
    eval_cmd+=" --use-full-video"
  fi
  if [[ "${RESUME}" == "1" ]]; then
    eval_cmd+=" --resume"
  fi
  if [[ "${FSYNC_EACH_RECORD}" == "1" ]]; then
    eval_cmd+=" --fsync-each-record"
  fi

  remote_cmd=(
    "set -euo pipefail"
    "uv sync --extra robometer"
    "echo \"Running dataset: ${dataset}\""
    "echo \"Output dir: ${output_dir}\""
    "mkdir -p \"${output_dir}\""
    "${eval_cmd}"
  )

  if [[ "${RUN_ANALYSIS}" == "1" ]]; then
    remote_cmd+=(
      "uv run python scripts/analyze_ai2_robometer_export.py --input-dir \"${output_dir}\""
    )
  fi

  gantry_args=(
    gantry run
    --allow-dirty
    --beaker-image "${BEAKER_IMAGE}"
    --gpus "${GPUS}"
    --cluster "${CLUSTER}"
    --workspace "${WORKSPACE}"
    --weka "${WEKA_MOUNT}"
    --name "${beaker_name}"
    --task-name "${beaker_name}"
    --description "${run_name}"
    --priority "${PRIORITY}"
    --no-python
  )

  if [[ -n "${HF_TOKEN_SECRET}" ]]; then
    gantry_args+=(--env-secret "HF_TOKEN=${HF_TOKEN_SECRET}")
  fi

  gantry_args+=(
    --
    bash -lc "$(printf '%s\n' "${remote_cmd[@]}")"
  )

  "${gantry_args[@]}"
done

