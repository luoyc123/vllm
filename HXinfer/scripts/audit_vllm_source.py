#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import importlib.metadata
import importlib.util
import json
import re
import sys
from pathlib import Path

TARGETS = {
    "sequence.py": ("IntermediateTensors",),
    "distributed/parallel_state.py": (
        "get_pp_group",
        "send_tensor_dict",
        "recv_tensor_dict",
        "isend_tensor_dict",
        "irecv_tensor_dict",
    ),
    "model_executor/models/utils.py": ("make_layers",),
    "v1/worker/gpu_worker.py": ("get_pp_group", "IntermediateTensors"),
    "v1/worker/gpu_model_runner.py": ("IntermediateTensors", "send_tensor_dict"),
    "v1/executor/multiproc_executor.py": ("MultiprocExecutor",),
    "v1/executor/ray_executor.py": ("RayDistributedExecutor",),
    "platforms/cuda.py": ("dist_backend", "get_device_communicator_cls"),
    "platforms/rocm.py": ("dist_backend", "get_device_communicator_cls"),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def find_lines(path: Path, symbols: tuple[str, ...]) -> dict[str, list[int]]:
    found = {symbol: [] for symbol in symbols}
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        for symbol in symbols:
            if re.search(rf"\b{re.escape(symbol)}\b", line):
                found[symbol].append(number)
    return found


def main() -> int:
    if len(sys.argv) != 2:
        print(f"usage: {sys.argv[0]} OUTPUT_JSON", file=sys.stderr)
        return 2
    spec = importlib.util.find_spec("vllm")
    if spec is None or spec.origin is None:
        print("vllm source not found", file=sys.stderr)
        return 1
    root = Path(spec.origin).resolve().parent
    files: dict[str, object] = {}
    for relative, symbols in TARGETS.items():
        path = root / relative
        if not path.is_file():
            files[relative] = {"present": False}
            continue
        files[relative] = {
            "present": True,
            "sha256": sha256(path),
            "symbols": find_lines(path, symbols),
        }
    value = {
        "vllm_version": importlib.metadata.version("vllm"),
        "source_root": str(root),
        "files": files,
    }
    Path(sys.argv[1]).write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
