# Local Qwen VL API

This repository can use a self-hosted OpenAI-compatible API for Qwen instead of Ollama.

## Recommended stack

- Server: `vLLM`
- Model: `Qwen/Qwen2.5-VL-7B-Instruct`
- API style: OpenAI-compatible `/v1/chat/completions`

## Why this setup

- `Qwen2.5-VL-7B-Instruct` stays within the requested `<=8B` size class.
- `vLLM` exposes an OpenAI-compatible API and is much faster than Ollama for this workload.
- The existing repository scripts already know how to talk to an OpenAI-compatible VLM endpoint.

## Install

Use a dedicated environment if possible.

```bash
pip install "vllm>=0.8.5"
```

## Start the server

```bash
bash scripts/start_qwen_vl_api.sh
```

By default, the launcher now prefers the local model at `/Data/Qwen2.5-VL-7B-Instruct`. If that directory does not exist, it falls back to the Hugging Face cache, and then to the remote model ID.

Useful environment variables:

```bash
export QWEN_LOCAL_MODEL=/Data/Qwen2.5-VL-7B-Instruct
export QWEN_SERVED_MODEL_NAME=qwen2.5-vl-7b-instruct
export QWEN_API_HOST=127.0.0.1
export QWEN_API_PORT=8000
export QWEN_API_KEY=EMPTY
export QWEN_TENSOR_PARALLEL_SIZE=1
export QWEN_GPU_MEMORY_UTILIZATION=0.90
export QWEN_MAX_MODEL_LEN=8192
export QWEN_MAX_NUM_SEQS=4
export QWEN_LIMIT_MM_PER_PROMPT=image=4
```

Multi-GPU example:

```bash
export CUDA_VISIBLE_DEVICES=0,1
export QWEN_TENSOR_PARALLEL_SIZE=2
bash scripts/start_qwen_vl_api.sh
```

## Smoke test

Text-only:

```bash
python scripts/check_qwen_vl_api.py \
  --api-base http://127.0.0.1:8000/v1 \
  --api-key EMPTY \
  --model qwen2.5-vl-7b-instruct \
  --prompt "Extract a JSON object with primary_goal and context_nodes from: the white sink is next to a window and under wooden cabinets"
```

Image + text:

```bash
python scripts/check_qwen_vl_api.py \
  --api-base http://127.0.0.1:8000/v1 \
  --api-key EMPTY \
  --model qwen2.5-vl-7b-instruct \
  --image path/to/example.png \
  --prompt "Return JSON with primary_goal, context_nodes, and relations."
```

## Using it with existing repository scripts

Example with the existing description-generation script:

```bash
python scripts/generate_instance_descriptions_qwen.py \
  --api-base http://127.0.0.1:8000/v1 \
  --api-key EMPTY \
  --model qwen2.5-vl-7b-instruct
```

## Notes

- Keep `QWEN_MAX_NUM_SEQS` low for a 7B VLM if GPU memory is tight.
- Increase `QWEN_LIMIT_MM_PER_PROMPT` only if you really need more images per prompt.
- If you later add a goal-graph parser, point it to the same local API base URL.
