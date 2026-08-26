#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib
import json
import time
from pathlib import Path


def normalize(text: str) -> str:
    return "".join(text.strip().rstrip("。.!！").split()).lower()


def resolve_executor_backend(value: str):
    if value in {"mp", "ray", "uni", "external_launcher"}:
        return value
    module_name, separator, class_name = value.rpartition(".")
    if not separator:
        return value
    return getattr(importlib.import_module(module_name), class_name)


def main() -> None:
    parser = argparse.ArgumentParser(description="Short deterministic QA test")
    parser.add_argument("--model", required=True)
    parser.add_argument("--cases-file", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--worker-cls")
    parser.add_argument("--distributed-executor-backend")
    parser.add_argument("--attention-backend", default="TRITON_ATTN")
    parser.add_argument("--max-tokens", type=int, default=32)
    parser.add_argument("--max-num-batched-tokens", type=int)
    parser.add_argument("--max-num-seqs", type=int)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.80)
    args = parser.parse_args()

    cases = json.loads(args.cases_file.read_text(encoding="utf-8"))
    from vllm import LLM, SamplingParams

    kwargs = {
        "model": args.model,
        "tensor_parallel_size": 1,
        "pipeline_parallel_size": 2,
        "dtype": "bfloat16",
        "max_model_len": 512,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "enforce_eager": True,
        "attention_backend": args.attention_backend,
    }
    if args.worker_cls:
        kwargs["worker_cls"] = args.worker_cls
    if args.distributed_executor_backend:
        kwargs["distributed_executor_backend"] = resolve_executor_backend(
            args.distributed_executor_backend
        )
    if args.max_num_batched_tokens is not None:
        kwargs["max_num_batched_tokens"] = args.max_num_batched_tokens
    if args.max_num_seqs is not None:
        kwargs["max_num_seqs"] = args.max_num_seqs
    started = time.perf_counter()
    llm = LLM(**kwargs)
    load_seconds = time.perf_counter() - started
    conversations = [[{"role": "user", "content": case["prompt"]}] for case in cases]
    started = time.perf_counter()
    outputs = llm.chat(
        conversations,
        sampling_params=SamplingParams(temperature=0, max_tokens=args.max_tokens),
    )
    generate_seconds = time.perf_counter() - started

    answers = []
    for case, output in zip(cases, outputs):
        generated = output.outputs[0]
        actual = generated.text
        expected = case["expected"]
        answers.append(
            {
                **case,
                "actual": actual,
                "correct": normalize(actual) == normalize(expected),
                "output_token_ids": list(generated.token_ids),
                "finish_reason": generated.finish_reason,
            }
        )
    result = {
        "model": args.model,
        "worker_cls": args.worker_cls or "auto",
        "attention_backend": args.attention_backend,
        "load_seconds": load_seconds,
        "generate_seconds": generate_seconds,
        "passed": all(answer["correct"] for answer in answers),
        "answers": answers,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    rendered = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
