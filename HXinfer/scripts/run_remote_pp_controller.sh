#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
source "${SCRIPT_DIR}/common.sh"
load_config
activate_vllm_env

PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
RUN_DIR="${RUN_DIR:-${PROJECT_ROOT}/results/remote_workers/$(date +%Y%m%d-%H%M%S)}"
PROMPTS_FILE="${PROMPTS_FILE:-${PROJECT_ROOT}/configs/qa_prompts.json}"

: "${MODEL:?Set MODEL in configs/experiment.env or the environment}"
: "${HXINFER_WORKER_ENDPOINTS:?Set two control endpoints, e.g. 10.0.0.1:29600,10.0.0.2:29600}"
: "${HXINFER_DIST_INIT:?Set the shared Gloo rendezvous endpoint, e.g. tcp://10.0.0.1:29610}"

mkdir -p "${RUN_DIR}"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
export HXINFER_LOG_DIR="${RUN_DIR}"
export VLLM_USE_FLASHINFER_SAMPLER="${VLLM_USE_FLASHINFER_SAMPLER:-0}"

controller_args=(
  python "${SCRIPT_DIR}/qa_vllm.py"
  --model "${MODEL}" \
  --prompts-file "${PROMPTS_FILE}" \
  --worker-cls external_pp.remote.worker.HXinferRemoteStageWorker \
  --distributed-executor-backend external_pp.remote.executor.HXinferRemoteExecutor \
  --max-model-len "${MAX_MODEL_LEN:-2048}" \
  --max-tokens "${MAX_TOKENS:-128}" \
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION:-0.80}" \
  --output "${RUN_DIR}/generation.json"
)
if [[ "${LANGUAGE_MODEL_ONLY:-0}" == "1" ]]; then
  controller_args+=(--language-model-only)
fi
if [[ -n "${MAX_NUM_BATCHED_TOKENS:-}" ]]; then
  controller_args+=(--max-num-batched-tokens "${MAX_NUM_BATCHED_TOKENS}")
fi
if [[ -n "${MAX_NUM_SEQS:-}" ]]; then
  controller_args+=(--max-num-seqs "${MAX_NUM_SEQS}")
fi
"${controller_args[@]}" 2>&1 | tee "${RUN_DIR}/controller.log"
