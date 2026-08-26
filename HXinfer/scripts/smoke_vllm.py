#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import platform
import time
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="vLLM generation smoke test")
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--prompt", default="Briefly describe pipeline parallelism.")
    parser.add_argument("--max-tokens", type=int, default=16)
    parser.add_argument("--max-model-len", type=int, default=512)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.65)
    parser.add_argument("--pipeline-parallel-size", type=int, default=1)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--worker-cls")
    parser.add_argument(
        "--use-v2-model-runner",
        action="store_true",
        help="Keep vLLM V2 runner on WSL; it requires UVA and normally fails there.",
    )
    args = parser.parse_args()

    is_wsl = "microsoft" in platform.release().lower()
    if is_wsl and not args.use_v2_model_runner:
        os.environ.setdefault("VLLM_USE_V2_MODEL_RUNNER", "0")

    # Import inside main so multiprocessing spawn can safely re-import this file.
    from vllm import LLM, SamplingParams

    started = time.perf_counter()
    llm_kwargs = {
        "model": args.model,
        "tensor_parallel_size": args.tensor_parallel_size,
        "pipeline_parallel_size": args.pipeline_parallel_size,
        "dtype": "float16",
        "max_model_len": args.max_model_len,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "enforce_eager": True,
    }
    if args.worker_cls:
        llm_kwargs["worker_cls"] = args.worker_cls
    llm = LLM(**llm_kwargs)
    load_seconds = time.perf_counter() - started
    sampling = SamplingParams(temperature=0, max_tokens=args.max_tokens)
    started = time.perf_counter()
    outputs = llm.generate([args.prompt], sampling)
    generate_seconds = time.perf_counter() - started
    first = outputs[0]
    result = {
        "model": args.model,
        "v2_model_runner": os.environ.get("VLLM_USE_V2_MODEL_RUNNER", "default"),
        "pipeline_parallel_size": args.pipeline_parallel_size,
        "tensor_parallel_size": args.tensor_parallel_size,
        "worker_cls": args.worker_cls or "auto",
        "load_seconds": load_seconds,
        "generate_seconds": generate_seconds,
        "prompt_tokens": len(first.prompt_token_ids),
        "output_tokens": len(first.outputs[0].token_ids),
        "output_token_ids": list(first.outputs[0].token_ids),
        "text": first.outputs[0].text,
    }
    rendered = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()
