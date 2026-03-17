#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export DATASET_SPLIT="${DATASET_SPLIT:-train}"
export NUM_TRAJECTORIES="${NUM_TRAJECTORIES:-10000}"
export OUTPUT_ROOT="${OUTPUT_ROOT:-output_train}"

exec bash "${SCRIPT_DIR}/export_valid_instances_val.sh" "$@"
