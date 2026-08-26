#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
source "${SCRIPT_DIR}/common.sh"
load_config
activate_vllm_env

failures=0
for command_name in python nvidia-smi curl; do
  if ! command -v "${command_name}" >/dev/null 2>&1; then
    echo "MISSING command: ${command_name}" >&2
    failures=$((failures + 1))
  fi
done

if ! python "${SCRIPT_DIR}/check_runtime.py"; then
  failures=$((failures + 1))
fi

if command -v nvidia-smi >/dev/null 2>&1; then
  gpu_count="$(nvidia-smi --query-gpu=index --format=csv,noheader 2>/dev/null | wc -l)"
  if [[ "${gpu_count}" -lt 2 ]]; then
    echo "Need at least 2 visible NVIDIA GPUs; found ${gpu_count}." >&2
    failures=$((failures + 1))
  fi
fi

if [[ "${MODEL}" == /* && ! -e "${MODEL}" ]]; then
  echo "MODEL path does not exist: ${MODEL}" >&2
  failures=$((failures + 1))
fi

if [[ "${PIPELINE_PARALLEL_SIZE:-2}" != "2" || "${TENSOR_PARALLEL_SIZE:-1}" != "1" ]]; then
  echo "Baseline must use PP=2 and TP=1." >&2
  failures=$((failures + 1))
fi

if [[ "${failures}" -ne 0 ]]; then
  echo "Preflight failed with ${failures} problem(s). Do not run the baseline." >&2
  exit 1
fi
echo "Preflight passed."
