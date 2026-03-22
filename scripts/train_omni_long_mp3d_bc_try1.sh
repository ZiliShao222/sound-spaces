#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

RUN_TYPE="${RUN_TYPE:-train}"
MODEL_DIR="${MODEL_DIR:-data/models/omni_long_mp3d_bc_no_audio}"  #规定模型存放位置
EXP_CONFIG="${EXP_CONFIG:-ss_baselines/omni_long/config/omni_long/mp3d/bc_no_audio_pointgoal_rgb.yaml}"  #规定配置文件位置
TRAIN_SPLIT="${TRAIN_SPLIT:-train}" #规定训练集分割
EVAL_SPLIT="${EVAL_SPLIT:-val}" #规定验证集分割
GOAL_ORDER_MODE="${GOAL_ORDER_MODE:-ordered}" #规定目标排序模式
NUM_PROCESSES="${NUM_PROCESSES:-4}" #规定训练时使用的进程数
NUM_UPDATES="${NUM_UPDATES:-20000}" # 规定训练时使用的更新次数
CHECKPOINT_INTERVAL="${CHECKPOINT_INTERVAL:-50}" # 规定训练时使用的检查点间隔
LOG_INTERVAL="${LOG_INTERVAL:-10}" # 规定训练时使用的日志间隔
TORCH_GPU_ID="${TORCH_GPU_ID:-0}" # 规定训练时使用的pytorch GPU ID
SIMULATOR_GPU_ID="${SIMULATOR_GPU_ID:-0}" # 规定训练时使用的simulator GPU ID
WANDB_ENABLED="${WANDB_ENABLED:-False}" # 规定训练时使用的wandb是否开启
USE_LAST_CKPT="${USE_LAST_CKPT:-False}" # 规定训练时使用的最后一个检查点是否使用，从头开始训练,初始化随机权重，True意思是从最后一个检查点开始训练
EVAL_CKPT_PATH_DIR="${EVAL_CKPT_PATH_DIR:-}" # 规定训练时使用的检查点路径,不使用的话，默认使用模型目录下的最后一个检查点，只在eval时使用

cd "${REPO_ROOT}" # 进入仓库根目录
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}" # 添加仓库根目录到PYTHONPATH环境变量,和当时srtod每次手敲的一样

CMD=(   # 构建训练命令
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
fi   # 如果EVAL_CKPT_PATH_DIR不为空，则添加EVAL_CKPT_PATH_DIR到CMD

CMD+=("$@") # 添加所有参数到CMD

echo "============================================================"
echo "[omni-long-no-audio] run_type=${RUN_TYPE}"
echo "[omni-long-no-audio] model_dir=${MODEL_DIR}"
echo "[omni-long-no-audio] exp_config=${EXP_CONFIG}"
echo "[omni-long-no-audio] train_split=${TRAIN_SPLIT} eval_split=${EVAL_SPLIT}"
echo "[omni-long-no-audio] goal_order_mode=${GOAL_ORDER_MODE}"
echo "[omni-long-no-audio] num_processes=${NUM_PROCESSES} num_updates=${NUM_UPDATES}"
echo "[omni-long-no-audio] torch_gpu_id=${TORCH_GPU_ID} simulator_gpu_id=${SIMULATOR_GPU_ID}"
echo "[omni-long-no-audio] NOTE: This baseline uses RGB + Depth only, NO audio"
echo "============================================================"

"${CMD[@]}"