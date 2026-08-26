from __future__ import annotations

import argparse
import asyncio
import importlib
import time
import uuid
from contextlib import asynccontextmanager
from threading import Lock
from typing import Any


def resolve_executor_backend(value: str):
    if value in {"mp", "ray", "uni", "external_launcher"}:
        return value
    module_name, separator, class_name = value.rpartition(".")
    if not separator:
        return value
    return getattr(importlib.import_module(module_name), class_name)


def _messages_from_request(body: dict[str, Any]) -> list[dict[str, str]]:
    messages = body.get("messages")
    if not isinstance(messages, list) or not messages:
        raise ValueError("messages must be a non-empty list")
    normalized: list[dict[str, str]] = []
    for message in messages:
        if not isinstance(message, dict):
            raise ValueError("each message must be an object")  # noqa: TRY004
        role = message.get("role")
        content = message.get("content")
        if role not in {"system", "user", "assistant", "tool"}:
            raise ValueError(f"unsupported message role: {role!r}")
        if not isinstance(content, str):
            raise ValueError(  # noqa: TRY004
                "this prototype currently requires string message content"
            )
        normalized.append({"role": role, "content": content})
    return normalized


def _sampling_params(body: dict[str, Any], default_max_tokens: int):
    from vllm import SamplingParams

    max_tokens = body.get("max_tokens", default_max_tokens)
    temperature = body.get("temperature", 1.0)
    top_p = body.get("top_p", 1.0)
    seed = body.get("seed")
    stop = body.get("stop")
    if not isinstance(max_tokens, int) or max_tokens <= 0:
        raise ValueError("max_tokens must be a positive integer")
    if not isinstance(temperature, (int, float)) or temperature < 0:
        raise ValueError("temperature must be a non-negative number")
    if not isinstance(top_p, (int, float)) or not 0 < top_p <= 1:
        raise ValueError("top_p must be in (0, 1]")
    if seed is not None and not isinstance(seed, int):
        raise ValueError("seed must be an integer")
    if stop is not None and not isinstance(stop, (str, list)):
        raise ValueError("stop must be a string or list of strings")
    return SamplingParams(
        temperature=float(temperature),
        top_p=float(top_p),
        max_tokens=max_tokens,
        seed=seed,
        stop=stop,
    )


def build_app(
    llm,
    served_model_name: str,
    default_max_tokens: int,
    api_key: str | None,
):
    from fastapi import FastAPI, Header, HTTPException

    inference_lock = Lock()

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        yield
        engine = getattr(llm, "llm_engine", None)
        shutdown = getattr(engine, "shutdown", None)
        if shutdown is not None:
            shutdown()

    app = FastAPI(title="HXinfer OpenAI-compatible API", lifespan=lifespan)

    def authorize(authorization: str | None) -> None:
        if api_key is None:
            return
        if authorization != f"Bearer {api_key}":
            raise HTTPException(status_code=401, detail="invalid API key")

    def reject_stream(body: dict[str, Any]) -> None:
        if body.get("stream", False):
            raise HTTPException(
                status_code=400,
                detail="stream=true is not implemented by the HXinfer prototype",
            )

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/v1/models")
    async def models(
        authorization: str | None = Header(default=None),
    ) -> dict[str, Any]:
        authorize(authorization)
        return {
            "object": "list",
            "data": [
                {
                    "id": served_model_name,
                    "object": "model",
                    "created": int(time.time()),
                    "owned_by": "hxinfer",
                }
            ],
        }

    def run_chat(body: dict[str, Any]) -> dict[str, Any]:
        messages = _messages_from_request(body)
        sampling_params = _sampling_params(body, default_max_tokens)
        with inference_lock:
            output = llm.chat(
                [messages], sampling_params=sampling_params, use_tqdm=False
            )[0]
        generated = output.outputs[0]
        prompt_tokens = len(output.prompt_token_ids)
        completion_tokens = len(generated.token_ids)
        return {
            "id": f"chatcmpl-{uuid.uuid4().hex}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": served_model_name,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": generated.text},
                    "logprobs": None,
                    "finish_reason": generated.finish_reason,
                }
            ],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
            },
        }

    @app.post("/v1/chat/completions")
    async def chat_completions(
        body: dict[str, Any],
        authorization: str | None = Header(default=None),
    ) -> dict[str, Any]:
        authorize(authorization)
        reject_stream(body)
        requested_model = body.get("model")
        if requested_model not in (None, served_model_name):
            raise HTTPException(status_code=404, detail="model not found")
        try:
            return await asyncio.to_thread(run_chat, body)
        except ValueError as error:
            raise HTTPException(status_code=400, detail=str(error)) from error

    def run_completion(body: dict[str, Any]) -> dict[str, Any]:
        prompt = body.get("prompt")
        if not isinstance(prompt, str) or not prompt:
            raise ValueError("prompt must be a non-empty string")
        sampling_params = _sampling_params(body, default_max_tokens)
        with inference_lock:
            output = llm.generate(
                [prompt], sampling_params=sampling_params, use_tqdm=False
            )[0]
        generated = output.outputs[0]
        prompt_tokens = len(output.prompt_token_ids)
        completion_tokens = len(generated.token_ids)
        return {
            "id": f"cmpl-{uuid.uuid4().hex}",
            "object": "text_completion",
            "created": int(time.time()),
            "model": served_model_name,
            "choices": [
                {
                    "index": 0,
                    "text": generated.text,
                    "logprobs": None,
                    "finish_reason": generated.finish_reason,
                }
            ],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
            },
        }

    @app.post("/v1/completions")
    async def completions(
        body: dict[str, Any],
        authorization: str | None = Header(default=None),
    ) -> dict[str, Any]:
        authorize(authorization)
        reject_stream(body)
        requested_model = body.get("model")
        if requested_model not in (None, served_model_name):
            raise HTTPException(status_code=404, detail="model not found")
        try:
            return await asyncio.to_thread(run_completion, body)
        except ValueError as error:
            raise HTTPException(status_code=400, detail=str(error)) from error

    return app


def main() -> None:
    parser = argparse.ArgumentParser(description="HXinfer OpenAI-compatible API")
    parser.add_argument("--model", required=True)
    parser.add_argument("--served-model-name")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--api-key")
    parser.add_argument("--pipeline-parallel-size", type=int, default=2)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--max-model-len", type=int, default=2048)
    parser.add_argument("--max-num-batched-tokens", type=int)
    parser.add_argument("--max-num-seqs", type=int)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.80)
    parser.add_argument("--worker-cls", required=True)
    parser.add_argument("--distributed-executor-backend", required=True)
    parser.add_argument("--attention-backend")
    parser.add_argument("--language-model-only", action="store_true")
    args = parser.parse_args()

    from vllm import LLM

    kwargs: dict[str, Any] = {
        "model": args.model,
        "tensor_parallel_size": args.tensor_parallel_size,
        "pipeline_parallel_size": args.pipeline_parallel_size,
        "dtype": "bfloat16",
        "max_model_len": args.max_model_len,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "enforce_eager": True,
        "worker_cls": args.worker_cls,
        "distributed_executor_backend": resolve_executor_backend(
            args.distributed_executor_backend
        ),
        "language_model_only": args.language_model_only,
    }
    if args.max_num_batched_tokens is not None:
        kwargs["max_num_batched_tokens"] = args.max_num_batched_tokens
    if args.max_num_seqs is not None:
        kwargs["max_num_seqs"] = args.max_num_seqs
    if args.attention_backend:
        kwargs["attention_backend"] = args.attention_backend

    llm = LLM(**kwargs)
    served_model_name = (
        args.served_model_name or args.model.rstrip("/").rsplit("/", 1)[-1]
    )
    app = build_app(llm, served_model_name, args.max_tokens, args.api_key)

    import uvicorn

    print(
        f"HXINFER_API_READY_PENDING host={args.host} port={args.port} "
        f"model={served_model_name}",
        flush=True,
    )
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
