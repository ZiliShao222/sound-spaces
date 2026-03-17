#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export DATASET_SPLIT="${DATASET_SPLIT:-test}"
export NUM_TRAJECTORIES="${NUM_TRAJECTORIES:-20}"
export OUTPUT_ROOT="${OUTPUT_ROOT:-output_test}"
export SKIP_EXPORT_VALID_INSTANCES="${SKIP_EXPORT_VALID_INSTANCES:-0}"

exec bash "${SCRIPT_DIR}/export_valid_instances_val.sh" "$@"
