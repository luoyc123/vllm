#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "usage: $0 RESULT_DIR" >&2
  exit 2
fi
result_dir="$1"
mkdir -p "${result_dir}"

python - <<'PY' > "${result_dir}/python_packages.txt"
import importlib.metadata
for name in ("vllm", "torch", "transformers", "ray"):
    try:
        print(f"{name}=={importlib.metadata.version(name)}")
    except importlib.metadata.PackageNotFoundError:
        print(f"{name}=NOT_INSTALLED")
PY

uname -a > "${result_dir}/uname.txt"
python --version > "${result_dir}/python_version.txt" 2>&1
python "$(dirname "$0")/audit_vllm_source.py" "${result_dir}/vllm_source_audit.json"
{
  for name in CUDA_VISIBLE_DEVICES HIP_VISIBLE_DEVICES ROCR_VISIBLE_DEVICES \
    NCCL_DEBUG NCCL_SOCKET_IFNAME VLLM_WORKER_MULTIPROC_METHOD \
    VLLM_PP_LAYER_PARTITION HF_HOME TRANSFORMERS_CACHE; do
    printf '%s=%s\n' "${name}" "${!name-}"
  done
} > "${result_dir}/environment.txt"

if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi -q > "${result_dir}/nvidia_smi_q.txt" 2>&1 || true
  nvidia-smi topo -m > "${result_dir}/nvidia_topology.txt" 2>&1 || true
  nvidia-smi --query-gpu=index,name,uuid,driver_version,memory.total,memory.used,pci.bus_id --format=csv \
    > "${result_dir}/gpu_inventory.csv" 2>&1 || true
fi

if command -v nvcc >/dev/null 2>&1; then
  nvcc --version > "${result_dir}/nvcc_version.txt" 2>&1 || true
fi
