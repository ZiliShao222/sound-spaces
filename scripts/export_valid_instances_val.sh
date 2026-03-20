#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
DATASET_SPLIT="${DATASET_SPLIT:-val}"
MAX_JOBS="${MAX_JOBS:-65}"
NUM_GPUS="${NUM_GPUS:-8}"
OUTPUT_ROOT="${OUTPUT_ROOT:-output_val}"
SKIP_EXPORT_VALID_INSTANCES="${SKIP_EXPORT_VALID_INSTANCES:-1}"

VAL_SCENES=(
  "2azQ1b91cZZ"
  "x8F5xyUWy9e"
  "QUCTc6BB5sX"
  "EU6Fwq7SyZv"
  "Z6MFQCViBuw"
  "pLe4wQe7qrG"
  "oLBMNvg9in8"
  "X7HyMhZNoso"
  "zsNo4HB9uLZ"
  "TbHJrupSAjP"
  "8194nk5LbLH"
)

TRAIN_SCENES=(
  # "sT4fr6TAbpF"
  # "E9uDoFAP3SH"
  # "VzqfbhrpDEA"
  # "kEZ7cmS4wCh"
  # "29hnd4uzFmX"
  # "ac26ZMwG7aT"
  # "i5noydFURQK"
  # "s8pcmisQ38h"
  # "rPc6DW4iMge"
  # "EDJbREhghzL"
  # "mJXqzFtmKg4"
  # "B6ByNegPMKs"
  # "JeFG25nYj2p"
  # "82sE5b5pLXE"
  # "D7N2EKCX4Sj"
  # "7y3sRwLe3Va"
  # "HxpKQynjfin"
  # "5LpN3gDmAk7"
  # "gTV8FGcVJC9"
  # "ur6pFq6Qu1A"
  # "qoiz87JEwZ2"
  # "PuKPg4mmafe"
  # "VLzqgDo317F"
  # "aayBHfsNo7d"
  # "JmbYfDe2QKZ"
  # "XcA2TqTSSAj"
  # "8WUmhLawc2A"
  # "sKLMLpTHeUy"
  # "r47D5H71a5s"
  # "Uxmj2M2itWa"
  # "Pm6F8kyY3z2"
  # "p5wJjkQkbXX"
  # "759xd9YjKW5"
  # "JF19kD82Mey"
  # "V2XKFyX4ASd"
  # "1LXtFkjw3qL"
  # "17DRP5sb8fy"
  # "5q7pvUzZiYa"
  # "VVfe2KiqLaN"
  # "Vvot9Ly1tCj"
  # "ULsKaCPVFJR"
  # "D7G3Y4RVNrH"
  # "uNb9QFRL6hY"
  # "ZMojNkEp431"
  # "2n8kARJN3HM"
  # "vyrNrziPKCB"
  # "e9zR4mvMWw7"
  # "r1Q1Z4BcV1o"
  # "PX4nDJXEHrG"
  # "YmJkqBEsHnH"
  # "b8cTxDM8gDG"
  # "GdvgFV5R1Z5"
  # "pRbA3pwrgk9"
  # "jh4fc5c5qoQ"
  # "dhjEzFoUFzH"
  "gZ6f7yhEvPG"
  # "1pXnuDYAj8r"
  # "S9hNv5qa7GM"
  # "VFuaQ6m2Qom"
  # "cV4RVeZvu5T"
  # "SN83YJsR3w2"
)

TEST_SCENES=(
  "pa4otMbVnkk"
  "yqstnuAEVhm"
  "5ZKStnWn8Zo"
  "Vt2qJdWjCF2"
  "wc2JMjhGNzB"
  "WYY7iVyf5p8"
  "fzynW3qQPVF"
  "UwV83HsGsw3"
  "q9vSo1VnCiC"
  "ARNzJeq3xxb"
  "rqfALeAoiTq"
  "gYvKGZ5eRqb"
  "YFuZgdQ5vWj"
  "jtcxE69GiFV"
  "gxdoqLR6rwA"
  "2t7WUuJeko7"
  "RPmz2sHmrrY"
  "YVUC4YcDtcY"
)

case "${DATASET_SPLIT}" in
  val)
    SCENES=("${VAL_SCENES[@]}")
    DEFAULT_NUM_TRAJECTORIES=20
    ;;
  train)
    SCENES=("${TRAIN_SCENES[@]}")
    DEFAULT_NUM_TRAJECTORIES=10000
    ;;
  test)
    SCENES=("${TEST_SCENES[@]}")
    DEFAULT_NUM_TRAJECTORIES=20
    ;;
  *)
    echo "DATASET_SPLIT must be one of: val, train, test; got ${DATASET_SPLIT}" >&2
    exit 1
    ;;
esac

NUM_TRAJECTORIES="${NUM_TRAJECTORIES:-${DEFAULT_NUM_TRAJECTORIES}}"
DISTANCE_MAX_ATTEMPTS="${DISTANCE_MAX_ATTEMPTS:-200}"
START_DISTANCE_RETRIES="${START_DISTANCE_RETRIES:-8}"


EXPORT_ARGS=(
  --generate-descriptions
  # --no-save-images
  # --no-filter-instances
  # --description-max-images 4
  # --no-generate-descriptions
)

TRAJ_ARGS=(
  --num-trajectories "${NUM_TRAJECTORIES}"
  --min-goals 2
  --max-goals 2
  --seed 0
  --sim-start
  --start-min-clearance 0.01
  --start-distance-retries "${START_DISTANCE_RETRIES}"
  --distance-max-attempts "${DISTANCE_MAX_ATTEMPTS}"
  --min-goal-distance 2
)

cd "${REPO_ROOT}"

cleanup_jobs() {
  local job_pids
  job_pids="$(jobs -pr)"
  if [[ -n "${job_pids}" ]]; then
    kill ${job_pids} 2>/dev/null || true
  fi
}

run_scene() {
  local scene="$1"
  local gpu_id="$2"
  shift 2

  local out="${OUTPUT_ROOT}/${scene}"
  local log_file="${out}/pipeline.log"
  local render_input="${out}/instances.json"
  mkdir -p "${out}"

  echo "============================================================"
  echo "[launch] scene=${scene} gpu=${gpu_id} log= ${log_file}"

  (
    set -euo pipefail
    export CUDA_VISIBLE_DEVICES="${gpu_id}"

    echo "[pipeline] scene=${scene} gpu=${gpu_id}"


    # echo "[1/3] export instances"
    # python scripts/export_instances.py \
    #   "${scene}" \
    #   --yolo-device cuda:0 \
    #   --output-root "${OUTPUT_ROOT}" \
    #   "${EXPORT_ARGS[@]}" \
    #   "$@"

    echo "[2/3] build trajectory dataset"
    python scripts/build_trajectories_from_valid_instances.py \
      --input "${out}/instances.json" \
      --output "${out}/${scene}.json" \
      --gpu-device-id 0 \
      "${TRAJ_ARGS[@]}"

    echo "[3/3] pack gz"
    python scripts/pack_semantic_audionav_json_gz.py \
      --input "${out}/${scene}.json" \
      --output "${out}/${scene}.json.gz"

    echo "add new glb file"
    if [[ ! -f "${render_input}" ]]; then
      render_input="${out}/valid_instances.json"
    fi
    python scripts/render_valid_instances_glb.py \
      --input "${render_input}" \
      --scene-name "${scene}" \
      --output "${out}/valid_instances_bbox.glb"
  ) >"${log_file}" 2>&1

  echo "[done] scene=${scene} gpu=${gpu_id} log= ${log_file}"
}

trap cleanup_jobs EXIT INT TERM

if (( MAX_JOBS < 1 )); then
  echo "MAX_JOBS must be >= 1, got ${MAX_JOBS}" >&2
  exit 1
fi

if (( NUM_GPUS < 1 )); then
  echo "NUM_GPUS must be >= 1, got ${NUM_GPUS}" >&2
  exit 1
fi

if (( NUM_TRAJECTORIES < 1 )); then
  echo "NUM_TRAJECTORIES must be >= 1, got ${NUM_TRAJECTORIES}" >&2
  exit 1
fi

echo "============================================================"
echo "[config] split=${DATASET_SPLIT} scenes=${#SCENES[@]} output_root=${OUTPUT_ROOT} num_trajectories=${NUM_TRAJECTORIES} max_jobs=${MAX_JOBS} num_gpus=${NUM_GPUS} skip_export_valid_instances=${SKIP_EXPORT_VALID_INSTANCES}"

scene_index=0
for scene in "${SCENES[@]}"; do
  gpu_id=$(( scene_index % NUM_GPUS ))
  run_scene "${scene}" "${gpu_id}" "$@" &
  ((scene_index += 1))

  while (( $(jobs -pr | wc -l) >= MAX_JOBS )); do
    wait -n
  done
done

wait

echo "============================================================"
echo "[pipeline] done: ${#SCENES[@]} ${DATASET_SPLIT} scenes (max_jobs=${MAX_JOBS}, num_gpus=${NUM_GPUS}, num_trajectories=${NUM_TRAJECTORIES})"
