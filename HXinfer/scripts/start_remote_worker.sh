#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 || $# -gt 3 ]]; then
  echo "usage: $0 RANK LISTEN_HOST:PORT [VISIBLE_DEVICE]" >&2
  exit 2
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
source "${SCRIPT_DIR}/common.sh"
if [[ -f "${CONFIG_FILE}" ]]; then
  load_config
fi
activate_vllm_env

PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
rank="$1"
listen="$2"
visible_device="${3:-0}"
visible_device_env="${HXINFER_VISIBLE_DEVICE_ENV:-CUDA_VISIBLE_DEVICES}"

if [[ "${rank}" != "0" && "${rank}" != "1" ]]; then
  echo "RANK must be 0 or 1" >&2
  exit 2
fi

case "${visible_device_env}" in
  CUDA_VISIBLE_DEVICES|HIP_VISIBLE_DEVICES|ROCR_VISIBLE_DEVICES)
    export "${visible_device_env}=${visible_device}"
    ;;
  *)
    echo "Unsupported HXINFER_VISIBLE_DEVICE_ENV=${visible_device_env}" >&2
    exit 2
    ;;
esac
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
worker_args=(
  python -m external_pp.remote.worker_service
  --rank "${rank}"
  --listen "${listen}"
)
if [[ -n "${VLLM_PP_LAYER_PARTITION:-}" ]]; then
  worker_args+=(--layer-partition "${VLLM_PP_LAYER_PARTITION}")
fi
exec "${worker_args[@]}"
