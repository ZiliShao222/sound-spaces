#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

RUN_TYPE="${RUN_TYPE:-train}"
MODEL_DIR="${MODEL_DIR:-data/models/omni_long_mp3d_ddppo}"
EXP_CONFIG="${EXP_CONFIG:-ss_baselines/omni_long/config/omni_long/mp3d/ppo_spectrogram_pointgoal_rgb.yaml}"
TRAIN_SPLIT="${TRAIN_SPLIT:-train}"
EVAL_SPLIT="${EVAL_SPLIT:-val}"
GOAL_ORDER_MODE="${GOAL_ORDER_MODE:-ordered}"
NUM_GPUS="${NUM_GPUS:-2}"
NUM_PROCESSES_PER_GPU="${NUM_PROCESSES_PER_GPU:-8}"
NUM_UPDATES="${NUM_UPDATES:-20000}"
CHECKPOINT_INTERVAL="${CHECKPOINT_INTERVAL:-50}"
LOG_INTERVAL="${LOG_INTERVAL:-10}"
WANDB_ENABLED="${WANDB_ENABLED:-False}"
MASTER_PORT="${MASTER_PORT:-29500}"

cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}:${REPO_ROOT}/habitat-lab:${PYTHONPATH:-}"

CMD=(
  torchrun
  --nproc_per_node "${NUM_GPUS}"
  --master_port "${MASTER_PORT}"
  ss_baselines/omni_long/run.py
  --run-type "${RUN_TYPE}"
  --exp-config "${EXP_CONFIG}"
  --model-dir "${MODEL_DIR}"
  TRAINER_NAME "OmniLongDDPPOTrainer"
  NUM_PROCESSES "${NUM_PROCESSES_PER_GPU}"
  NUM_UPDATES "${NUM_UPDATES}"
  CHECKPOINT_INTERVAL "${CHECKPOINT_INTERVAL}"
  LOG_INTERVAL "${LOG_INTERVAL}"
  TASK_CONFIG.DATASET.SPLIT "${TRAIN_SPLIT}"
  TASK_CONFIG.TASK.GOAL_ORDER_MODE "${GOAL_ORDER_MODE}"
  EVAL.SPLIT "${EVAL_SPLIT}"
  WANDB.ENABLED "${WANDB_ENABLED}"
  RL.DDPPO.distrib_backend "nccl"
)

CMD+=("$@")

echo "============================================================"
echo "[omni-long-ddppo] run_type=${RUN_TYPE}"
echo "[omni-long-ddppo] model_dir=${MODEL_DIR}"
echo "[omni-long-ddppo] exp_config=${EXP_CONFIG}"
echo "[omni-long-ddppo] train_split=${TRAIN_SPLIT} eval_split=${EVAL_SPLIT}"
echo "[omni-long-ddppo] goal_order_mode=${GOAL_ORDER_MODE}"
echo "[omni-long-ddppo] num_gpus=${NUM_GPUS} num_processes_per_gpu=${NUM_PROCESSES_PER_GPU}"
echo "[omni-long-ddppo] total_envs=$((NUM_GPUS * NUM_PROCESSES_PER_GPU))"
echo "============================================================"

"${CMD[@]}"
