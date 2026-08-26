#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
source "${SCRIPT_DIR}/common.sh"
load_config
activate_vllm_env

PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
RUN_DIR="${RUN_DIR:-${PROJECT_ROOT}/results/remote_api/$(date +%Y%m%d-%H%M%S)}"

: "${MODEL:?Set MODEL in configs/experiment.env or the environment}"
: "${HXINFER_WORKER_ENDPOINTS:?Set two control endpoints}"
: "${HXINFER_DIST_INIT:?Set the shared Gloo rendezvous endpoint}"

mkdir -p "${RUN_DIR}"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
export HXINFER_LOG_DIR="${RUN_DIR}"
export VLLM_USE_FLASHINFER_SAMPLER="${VLLM_USE_FLASHINFER_SAMPLER:-0}"

api_args=(
  python -m external_pp.api_server
  --model "${MODEL}"
  --served-model-name "${SERVED_MODEL_NAME:-$(basename "${MODEL}")}"
  --host "${API_HOST:-0.0.0.0}"
  --port "${API_PORT:-8000}"
  --worker-cls external_pp.remote.worker.HXinferRemoteStageWorker
  --distributed-executor-backend external_pp.remote.executor.HXinferRemoteExecutor
  --max-model-len "${MAX_MODEL_LEN:-2048}"
  --max-tokens "${MAX_TOKENS:-128}"
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION:-0.80}"
)
if [[ -n "${MAX_NUM_BATCHED_TOKENS:-}" ]]; then
  api_args+=(--max-num-batched-tokens "${MAX_NUM_BATCHED_TOKENS}")
fi
if [[ -n "${MAX_NUM_SEQS:-}" ]]; then
  api_args+=(--max-num-seqs "${MAX_NUM_SEQS}")
fi
if [[ -n "${ATTENTION_BACKEND:-}" ]]; then
  api_args+=(--attention-backend "${ATTENTION_BACKEND}")
fi
if [[ "${LANGUAGE_MODEL_ONLY:-0}" == "1" ]]; then
  api_args+=(--language-model-only)
fi
if [[ -n "${HXINFER_API_KEY:-}" ]]; then
  api_args+=(--api-key "${HXINFER_API_KEY}")
fi

"${api_args[@]}" 2>&1 | tee "${RUN_DIR}/api-server.log"
