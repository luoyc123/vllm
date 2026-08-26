#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
run_id="$(date +%Y%m%d-%H%M%S)"
export RUN_DIR="${RUN_DIR:-${PROJECT_ROOT}/results/remote_workers/${run_id}}"
mkdir -p "${RUN_DIR}"

base_port="${HXINFER_BASE_PORT:-29600}"
export HXINFER_WORKER_ENDPOINTS="127.0.0.1:${base_port},127.0.0.1:$((base_port + 1))"
export HXINFER_DIST_INIT="tcp://127.0.0.1:$((base_port + 10))"
export HXINFER_ACTIVATION_HOST="127.0.0.1"
export HXINFER_ACTIVATION_PORT="$((base_port + 20))"
export HXINFER_LOG_DIR="${RUN_DIR}"

worker0_pid=""
worker1_pid=""
cleanup() {
  for worker_pid in "${worker0_pid}" "${worker1_pid}"; do
    if [[ -n "${worker_pid}" ]]; then
      kill "${worker_pid}" 2>/dev/null || true
      wait "${worker_pid}" 2>/dev/null || true
    fi
  done
}
trap cleanup EXIT INT TERM

"${SCRIPT_DIR}/start_remote_worker.sh" 0 "127.0.0.1:${base_port}" 0 \
  > "${RUN_DIR}/worker0.log" 2>&1 &
worker0_pid=$!
"${SCRIPT_DIR}/start_remote_worker.sh" 1 "127.0.0.1:$((base_port + 1))" 1 \
  > "${RUN_DIR}/worker1.log" 2>&1 &
worker1_pid=$!

ready=0
for _ in $(seq 1 120); do
  if grep -q HXINFER_WORKER_READY "${RUN_DIR}/worker0.log" 2>/dev/null && \
     grep -q HXINFER_WORKER_READY "${RUN_DIR}/worker1.log" 2>/dev/null; then
    ready=1
    break
  fi
  if ! kill -0 "${worker0_pid}" 2>/dev/null || ! kill -0 "${worker1_pid}" 2>/dev/null; then
    echo "A worker exited before readiness; inspect ${RUN_DIR}/worker*.log" >&2
    exit 1
  fi
  sleep 1
done
if [[ "${ready}" -ne 1 ]]; then
  echo "Workers were not ready after 120 seconds" >&2
  exit 1
fi

bash "${SCRIPT_DIR}/run_remote_pp_controller.sh"
