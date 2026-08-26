#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
source "${SCRIPT_DIR}/common.sh"
load_config
activate_vllm_env

PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}" python -m external_pp.controller
new_run_dir external_socket
bash "${SCRIPT_DIR}/collect_env.sh" "${RUN_DIR}"

export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
run_id="$(basename "${RUN_DIR}")"
export HXINFER_SOCKET_PATH="${HXINFER_SOCKET_PATH:-/tmp/hxinfer-${run_id}.sock}"
export HXINFER_LOG_DIR="${RUN_DIR}"
export VLLM_USE_FLASHINFER_SAMPLER="${VLLM_USE_FLASHINFER_SAMPLER:-0}"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}" python "${SCRIPT_DIR}/smoke_vllm.py" \
  --model "${MODEL}" \
  --pipeline-parallel-size 2 \
  --tensor-parallel-size 1 \
  --worker-cls external_pp.vllm_worker.HXinferWorker \
  --max-model-len "${MAX_MODEL_LEN:-512}" \
  --max-tokens "${SMOKE_MAX_TOKENS:-16}" \
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION:-0.70}" \
  --output "${RUN_DIR}/generation.json" \
  2>&1 | tee "${RUN_DIR}/run.log"

echo "External two-process PP run complete: ${RUN_DIR}"
