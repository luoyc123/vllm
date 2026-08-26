#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG_FILE="${HXINFER_CONFIG:-${PROJECT_ROOT}/configs/experiment.env}"

load_config() {
  if [[ ! -f "${CONFIG_FILE}" ]]; then
    echo "Missing ${CONFIG_FILE}; copy configs/experiment.env.example first." >&2
    return 2
  fi
  set -a
  # shellcheck disable=SC1090
  source "${CONFIG_FILE}"
  set +a
  : "${MODEL:?MODEL must be set in ${CONFIG_FILE}}"
}

activate_vllm_env() {
  if [[ -n "${VLLM_ENV:-}" ]]; then
    if [[ ! -f "${VLLM_ENV}/bin/activate" ]]; then
      echo "VLLM_ENV does not contain bin/activate: ${VLLM_ENV}" >&2
      return 2
    fi
    # shellcheck disable=SC1090
    source "${VLLM_ENV}/bin/activate"
  fi
}

new_run_dir() {
  local case_name="$1"
  local run_id
  run_id="$(date -u +%Y%m%dT%H%M%SZ)"
  RUN_DIR="${PROJECT_ROOT}/results/${case_name}/${run_id}"
  mkdir -p "${RUN_DIR}"
  export RUN_DIR
}
