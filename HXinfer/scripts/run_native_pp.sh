#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
source "${SCRIPT_DIR}/common.sh"
load_config
activate_vllm_env
bash "${SCRIPT_DIR}/preflight.sh"
new_run_dir native_pp
bash "${SCRIPT_DIR}/collect_env.sh" "${RUN_DIR}"

server_pid=""
monitor_pid=""
cleanup() {
  if [[ -n "${server_pid}" ]]; then
    kill "${server_pid}" 2>/dev/null || true
    wait "${server_pid}" 2>/dev/null || true
  fi
  if [[ -n "${monitor_pid}" ]]; then
    kill "${monitor_pid}" 2>/dev/null || true
    wait "${monitor_pid}" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi dmon -s pucvmet -d 1 -o DT > "${RUN_DIR}/nvidia_dmon.log" 2>&1 &
  monitor_pid=$!
fi

HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8000}"
PIPELINE_PARALLEL_SIZE="${PIPELINE_PARALLEL_SIZE:-2}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-1}"

partition_args=(--model "${MODEL}" --pp-size "${PIPELINE_PARALLEL_SIZE}" --output "${RUN_DIR}/partition.json")
if [[ " ${EXTRA_SERVE_ARGS:-} " == *" --trust-remote-code "* ]]; then
  partition_args+=(--trust-remote-code)
fi
python "${SCRIPT_DIR}/inspect_partition.py" "${partition_args[@]}"

serve_cmd=(vllm serve "${MODEL}"
  --host "${HOST}"
  --port "${PORT}"
  --pipeline-parallel-size "${PIPELINE_PARALLEL_SIZE}"
  --tensor-parallel-size "${TENSOR_PARALLEL_SIZE}"
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION:-0.90}"
  --dtype "${DTYPE:-auto}"
  --max-model-len "${MAX_MODEL_LEN:-4096}")
if [[ -n "${EXTRA_SERVE_ARGS:-}" ]]; then
  # Intentional word splitting: this field is a shell argument fragment owned by the operator.
  # shellcheck disable=SC2206
  extra_args=(${EXTRA_SERVE_ARGS})
  serve_cmd+=("${extra_args[@]}")
fi
printf '%q ' "${serve_cmd[@]}" > "${RUN_DIR}/serve_command.txt"
printf '\n' >> "${RUN_DIR}/serve_command.txt"

NCCL_DEBUG="${NCCL_DEBUG:-INFO}" VLLM_LOGGING_LEVEL="${VLLM_LOGGING_LEVEL:-INFO}" \
  "${serve_cmd[@]}" > "${RUN_DIR}/server.log" 2>&1 &
server_pid=$!

ready=0
for _ in $(seq 1 180); do
  if ! kill -0 "${server_pid}" 2>/dev/null; then
    echo "vLLM server exited before readiness; inspect ${RUN_DIR}/server.log" >&2
    exit 1
  fi
  if curl -fsS "http://${HOST}:${PORT}/health" >/dev/null; then
    ready=1
    break
  fi
  sleep 2
done
if [[ "${ready}" -ne 1 ]]; then
  echo "vLLM server was not ready after 360 seconds." >&2
  exit 1
fi

vllm bench serve \
  --backend vllm \
  --base-url "http://${HOST}:${PORT}" \
  --model "${MODEL}" \
  --dataset-name random \
  --num-prompts "${NUM_PROMPTS:-32}" \
  --random-input-len "${RANDOM_INPUT_LEN:-512}" \
  --random-output-len "${RANDOM_OUTPUT_LEN:-128}" \
  --request-rate "${REQUEST_RATE:-inf}" \
  --seed "${SEED:-20260823}" \
  --save-result \
  --result-dir "${RUN_DIR}" \
  --result-filename benchmark.json \
  2>&1 | tee "${RUN_DIR}/benchmark.log"

python "${SCRIPT_DIR}/summarize_native.py" "${RUN_DIR}"

echo "Native PP run complete: ${RUN_DIR}"
echo "Review logs and metrics; create ${PROJECT_ROOT}/results/native_pp/PASS manually only after acceptance."
