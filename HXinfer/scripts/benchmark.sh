#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ ! -f "${PROJECT_ROOT}/results/native_pp/PASS" ]]; then
  echo "Native baseline gate is closed. Run scripts/run_native_pp.sh and review it first." >&2
  exit 1
fi
echo "Native gate is open. External IPC/socket benchmark remains gated on split-forward correctness."
exit 3
