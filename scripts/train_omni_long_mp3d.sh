#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

RUN_TYPE="${RUN_TYPE:-train}"
MODEL_DIR="${MODEL_DIR:-data/models/omni_long_mp3d}"
EXP_CONFIG="${EXP_CONFIG:-ss_baselines/omni_long/config/omni_long/mp3d/ppo_spectrogram_pointgoal_rgb.yaml}"
TRAIN_SPLIT="${TRAIN_SPLIT:-train}"
EVAL_SPLIT="${EVAL_SPLIT:-val}"
GOAL_ORDER_MODE="${GOAL_ORDER_MODE:-ordered}"
NUM_PROCESSES="${NUM_PROCESSES:-4}"
NUM_UPDATES="${NUM_UPDATES:-20000}"
CHECKPOINT_INTERVAL="${CHECKPOINT_INTERVAL:-50}"
LOG_INTERVAL="${LOG_INTERVAL:-10}"
TORCH_GPU_ID="${TORCH_GPU_ID:-0}"
SIMULATOR_GPU_ID="${SIMULATOR_GPU_ID:-0}"
WANDB_ENABLED="${WANDB_ENABLED:-False}"
USE_LAST_CKPT="${USE_LAST_CKPT:-False}"
EVAL_CKPT_PATH_DIR="${EVAL_CKPT_PATH_DIR:-}"

cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

CMD=(
  python ss_baselines/omni_long/run.py
  --run-type "${RUN_TYPE}"
  --exp-config "${EXP_CONFIG}"
  --model-dir "${MODEL_DIR}"
  NUM_PROCESSES "${NUM_PROCESSES}"
  NUM_UPDATES "${NUM_UPDATES}"
  CHECKPOINT_INTERVAL "${CHECKPOINT_INTERVAL}"
  LOG_INTERVAL "${LOG_INTERVAL}"
  TORCH_GPU_ID "${TORCH_GPU_ID}"
  SIMULATOR_GPU_ID "${SIMULATOR_GPU_ID}"
  USE_LAST_CKPT "${USE_LAST_CKPT}"
  TASK_CONFIG.DATASET.SPLIT "${TRAIN_SPLIT}"
  TASK_CONFIG.TASK.GOAL_ORDER_MODE "${GOAL_ORDER_MODE}"
  EVAL.SPLIT "${EVAL_SPLIT}"
  WANDB.ENABLED "${WANDB_ENABLED}"
)

if [[ -n "${EVAL_CKPT_PATH_DIR}" ]]; then
  CMD+=(EVAL_CKPT_PATH_DIR "${EVAL_CKPT_PATH_DIR}")
fi

CMD+=("$@")

echo "============================================================"
echo "[omni-long] run_type=${RUN_TYPE}"
echo "[omni-long] model_dir=${MODEL_DIR}"
echo "[omni-long] exp_config=${EXP_CONFIG}"
echo "[omni-long] train_split=${TRAIN_SPLIT} eval_split=${EVAL_SPLIT}"
echo "[omni-long] goal_order_mode=${GOAL_ORDER_MODE}"
echo "[omni-long] num_processes=${NUM_PROCESSES} num_updates=${NUM_UPDATES}"
echo "[omni-long] torch_gpu_id=${TORCH_GPU_ID} simulator_gpu_id=${SIMULATOR_GPU_ID}"
echo "============================================================"

"${CMD[@]}"
