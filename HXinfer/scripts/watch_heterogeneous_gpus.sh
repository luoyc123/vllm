#!/usr/bin/env bash
set -euo pipefail

interval="${1:-1}"
api_container="${HXINFER_API_CONTAINER:-hxinfer-api}"
cuda_container="${HXINFER_CUDA_CONTAINER:-hxinfer-worker-cuda}"
rocm_container="${HXINFER_ROCM_CONTAINER:-hxinfer-worker-rocm}"

while true; do
  printf '\033[H\033[2J'
  date '+%F %T'
  echo
  echo '[containers]'
  docker ps \
    --filter "name=^/${api_container}$" \
    --filter "name=^/${cuda_container}$" \
    --filter "name=^/${rocm_container}$" \
    --format 'table {{.Names}}\t{{.Status}}'
  echo
  echo '[NVIDIA]'
  nvidia-smi \
    --query-gpu=name,utilization.gpu,memory.used,memory.total \
    --format=csv,noheader
  nvidia-smi \
    --query-compute-apps=pid,process_name,used_memory \
    --format=csv,noheader 2>/dev/null || true
  echo
  echo '[AMD]'
  /opt/rocm/bin/rocm-smi --showuse --showmemuse 2>/dev/null \
    | sed -n '/GPU use (%)/p;/VRAM%/p'
  /opt/rocm/bin/rocm-smi --showpids 2>/dev/null \
    | sed -n '/PID.*PROCESS NAME/p;/python/p'
  echo
  echo "refresh interval: ${interval}s (Ctrl-C to stop)"
  sleep "${interval}"
done
