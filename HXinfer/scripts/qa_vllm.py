#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib
import json
import time
from pathlib import Path


def resolve_executor_backend(value: str):
    if value in {"mp", "ray", "uni", "external_launcher"}:
        return value
    module_name, separator, class_name = value.rpartition(".")
    if not separator:
        return value
    return getattr(importlib.import_module(module_name), class_name)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Deterministic multi-question vLLM test"
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--prompts-file", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pipeline-parallel-size", type=int, default=2)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--worker-cls")
    parser.add_argument("--distributed-executor-backend")
    parser.add_argument("--attention-backend")
    parser.add_argument(
        "--language-model-only",
        action="store_true",
        help="Skip multimodal components for text-only validation.",
    )
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--max-model-len", type=int, default=2048)
    parser.add_argument("--max-num-batched-tokens", type=int)
    parser.add_argument("--max-num-seqs", type=int)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    args = parser.parse_args()

    prompts = json.loads(args.prompts_file.read_text(encoding="utf-8"))
    if (
        not isinstance(prompts, list)
        or not prompts
        or not all(isinstance(prompt, str) and prompt for prompt in prompts)
    ):
        raise ValueError("prompts file must contain a non-empty JSON string list")

    from vllm import LLM, SamplingParams

    kwargs = {
        "model": args.model,
        "tensor_parallel_size": args.tensor_parallel_size,
        "pipeline_parallel_size": args.pipeline_parallel_size,
        "dtype": "bfloat16",
        "max_model_len": args.max_model_len,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "enforce_eager": True,
        "language_model_only": args.language_model_only,
    }
    if args.worker_cls:
        kwargs["worker_cls"] = args.worker_cls
    if args.distributed_executor_backend:
        kwargs["distributed_executor_backend"] = resolve_executor_backend(
            args.distributed_executor_backend
        )
    if args.attention_backend:
        kwargs["attention_backend"] = args.attention_backend
    if args.max_num_batched_tokens is not None:
        kwargs["max_num_batched_tokens"] = args.max_num_batched_tokens
    if args.max_num_seqs is not None:
        kwargs["max_num_seqs"] = args.max_num_seqs

    started = time.perf_counter()
    llm = LLM(**kwargs)
    load_seconds = time.perf_counter() - started
    conversations = [[{"role": "user", "content": prompt}] for prompt in prompts]
    sampling = SamplingParams(temperature=0, max_tokens=args.max_tokens)
    started = time.perf_counter()
    outputs = llm.chat(conversations, sampling_params=sampling)
    generate_seconds = time.perf_counter() - started

    result = {
        "model": args.model,
        "pipeline_parallel_size": args.pipeline_parallel_size,
        "tensor_parallel_size": args.tensor_parallel_size,
        "worker_cls": args.worker_cls or "auto",
        "load_seconds": load_seconds,
        "generate_seconds": generate_seconds,
        "questions": [
            {
                "prompt": prompt,
                "prompt_token_ids": list(output.prompt_token_ids),
                "output_token_ids": list(output.outputs[0].token_ids),
                "text": output.outputs[0].text,
                "finish_reason": output.outputs[0].finish_reason,
            }
            for prompt, output in zip(prompts, outputs)
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
