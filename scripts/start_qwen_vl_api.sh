#!/usr/bin/env bash
set -euo pipefail

if ! command -v vllm >/dev/null 2>&1; then
  echo "vllm is not installed. Install it first, for example: pip install \"vllm>=0.8.5\"" >&2
  exit 1
fi

DEFAULT_QWEN_REMOTE_MODEL="Qwen/Qwen2.5-VL-7B-Instruct"
DEFAULT_QWEN_DATA_ROOT="${QWEN_DATA_ROOT:-/Data}"
DEFAULT_QWEN_DATA_MODEL="${DEFAULT_QWEN_DATA_ROOT}/Qwen2.5-VL-7B-Instruct"
DEFAULT_QWEN_CACHE_ROOT="${HOME}/.cache/huggingface/hub/models--Qwen--Qwen2.5-VL-7B-Instruct"
DEFAULT_QWEN_LOCAL_MODEL="${QWEN_LOCAL_MODEL:-}"

if [[ -z "$DEFAULT_QWEN_LOCAL_MODEL" && -d "$DEFAULT_QWEN_DATA_MODEL" ]]; then
  DEFAULT_QWEN_LOCAL_MODEL="$DEFAULT_QWEN_DATA_MODEL"
fi

if [[ -z "$DEFAULT_QWEN_LOCAL_MODEL" && -d "$DEFAULT_QWEN_CACHE_ROOT/snapshots" ]]; then
  if [[ -f "$DEFAULT_QWEN_CACHE_ROOT/refs/main" ]]; then
    snapshot_ref="$(<"$DEFAULT_QWEN_CACHE_ROOT/refs/main")"
    if [[ -n "$snapshot_ref" && -d "$DEFAULT_QWEN_CACHE_ROOT/snapshots/$snapshot_ref" ]]; then
      DEFAULT_QWEN_LOCAL_MODEL="$DEFAULT_QWEN_CACHE_ROOT/snapshots/$snapshot_ref"
    fi
  fi
  if [[ -z "$DEFAULT_QWEN_LOCAL_MODEL" ]]; then
    DEFAULT_QWEN_LOCAL_MODEL="$(find "$DEFAULT_QWEN_CACHE_ROOT/snapshots" -mindepth 1 -maxdepth 1 -type d 2>/dev/null | head -n 1)"
  fi
fi

QWEN_MODEL="${QWEN_MODEL:-${DEFAULT_QWEN_LOCAL_MODEL:-$DEFAULT_QWEN_REMOTE_MODEL}}"
QWEN_SERVED_MODEL_NAME="${QWEN_SERVED_MODEL_NAME:-}"
QWEN_API_HOST="${QWEN_API_HOST:-127.0.0.1}"
QWEN_API_PORT="${QWEN_API_PORT:-8000}"
QWEN_API_KEY="${QWEN_API_KEY:-EMPTY}"
QWEN_TENSOR_PARALLEL_SIZE="${QWEN_TENSOR_PARALLEL_SIZE:-1}"
QWEN_GPU_MEMORY_UTILIZATION="${QWEN_GPU_MEMORY_UTILIZATION:-0.90}"
QWEN_MAX_MODEL_LEN="${QWEN_MAX_MODEL_LEN:-8192}"
QWEN_MAX_NUM_SEQS="${QWEN_MAX_NUM_SEQS:-4}"
QWEN_LIMIT_MM_PER_PROMPT="${QWEN_LIMIT_MM_PER_PROMPT:-{\"image\":4}}"
QWEN_ENABLE_MULTIMODAL="${QWEN_ENABLE_MULTIMODAL:-auto}"

if [[ -z "$QWEN_SERVED_MODEL_NAME" ]]; then
  if [[ "$QWEN_MODEL" == *"/snapshots/"* ]]; then
    cache_root="${QWEN_MODEL%/snapshots/*}"
    cache_name="$(basename "$cache_root")"
    QWEN_SERVED_MODEL_NAME="${cache_name#models--}"
    QWEN_SERVED_MODEL_NAME="${QWEN_SERVED_MODEL_NAME#*--}"
  else
    QWEN_SERVED_MODEL_NAME="$(basename "$QWEN_MODEL")"
  fi
fi

model_lower="${QWEN_MODEL,,}"
enable_multimodal=0
if [[ "$QWEN_ENABLE_MULTIMODAL" == "1" ]]; then
  enable_multimodal=1
elif [[ "$QWEN_ENABLE_MULTIMODAL" == "auto" ]]; then
  if [[ "$model_lower" == *"-vl-"* ]] || [[ "$model_lower" == *"/qwen3-vl"* ]] || [[ "$model_lower" == *"/qwen2-vl"* ]]; then
    enable_multimodal=1
  fi
fi

cmd=(
  vllm serve "$QWEN_MODEL"
  --host "$QWEN_API_HOST"
  --port "$QWEN_API_PORT"
  --served-model-name "$QWEN_SERVED_MODEL_NAME"
  --api-key "$QWEN_API_KEY"
  --tensor-parallel-size "$QWEN_TENSOR_PARALLEL_SIZE"
  --gpu-memory-utilization "$QWEN_GPU_MEMORY_UTILIZATION"
  --max-model-len "$QWEN_MAX_MODEL_LEN"
  --max-num-seqs "$QWEN_MAX_NUM_SEQS"
)

if [[ "$enable_multimodal" == "1" ]]; then
  cmd+=(--limit-mm-per-prompt "$QWEN_LIMIT_MM_PER_PROMPT")
fi

if [[ "${QWEN_TRUST_REMOTE_CODE:-0}" == "1" ]]; then
  cmd+=(--trust-remote-code)
fi

if [[ -n "${QWEN_DOWNLOAD_DIR:-}" ]]; then
  cmd+=(--download-dir "$QWEN_DOWNLOAD_DIR")
fi

if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  echo "Using CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
fi

echo "Starting Qwen API on http://${QWEN_API_HOST}:${QWEN_API_PORT}/v1"
echo "Model: ${QWEN_MODEL}"
echo "Served model name: ${QWEN_SERVED_MODEL_NAME}"
if [[ "$QWEN_MODEL" != "$DEFAULT_QWEN_REMOTE_MODEL" ]]; then
  echo "Model source: local"
else
  echo "Model source: remote"
fi
echo "Multimodal enabled: ${enable_multimodal}"

cmd+=("$@")
exec "${cmd[@]}"
