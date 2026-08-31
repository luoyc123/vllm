from __future__ import annotations

import argparse
import asyncio
import importlib
import json
import time
import uuid
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from typing import Any


def resolve_executor_backend(value: str):
    if value in {"mp", "ray", "uni", "external_launcher"}:
        return value
    module_name, separator, class_name = value.rpartition(".")
    if not separator:
        return value
    return getattr(importlib.import_module(module_name), class_name)


def _message_content_as_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for part in content:
            if (
                not isinstance(part, dict)
                or part.get("type") != "text"
                or not isinstance(part.get("text"), str)
            ):
                raise ValueError(
                    "this prototype only supports text message content parts"
                )
            parts.append(part["text"])
        if parts:
            return "".join(parts)
    raise ValueError("message content must be text or a non-empty text-part list")


def _messages_from_request(body: dict[str, Any]) -> list[dict[str, str]]:
    messages = body.get("messages")
    if not isinstance(messages, list) or not messages:
        raise ValueError("messages must be a non-empty list")
    normalized: list[dict[str, str]] = []
    for message in messages:
        if not isinstance(message, dict):
            raise ValueError("each message must be an object")  # noqa: TRY004
        role = message.get("role")
        if role not in {"system", "user", "assistant", "tool"}:
            raise ValueError(f"unsupported message role: {role!r}")
        content = _message_content_as_text(message.get("content"))
        normalized.append({"role": role, "content": content})
    return normalized


def _sampling_params(
    body: dict[str, Any],
    default_max_tokens: int,
    *,
    stream: bool = False,
):
    from vllm import SamplingParams
    from vllm.sampling_params import RequestOutputKind

    max_tokens = body.get(
        "max_completion_tokens", body.get("max_tokens", default_max_tokens)
    )
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
        output_kind=(
            RequestOutputKind.DELTA
            if stream
            else RequestOutputKind.FINAL_ONLY
        ),
    )


def _sse(payload: dict[str, Any] | str) -> str:
    data = (
        payload
        if isinstance(payload, str)
        else json.dumps(payload, ensure_ascii=False)
    )
    return f"data: {data}\n\n"


def _usage(prompt_tokens: int, completion_tokens: int) -> dict[str, int]:
    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": prompt_tokens + completion_tokens,
    }


async def _chat_prompt(engine, messages: list[dict[str, str]]) -> str:
    tokenizer = engine.get_tokenizer()
    prompt = await asyncio.to_thread(
        tokenizer.apply_chat_template,
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    if not isinstance(prompt, str):
        raise ValueError("chat template did not produce a text prompt")
    return prompt


async def _collect_final(engine, prompt: str, sampling_params, request_id: str):
    final_output = None
    async for output in engine.generate(prompt, sampling_params, request_id):
        final_output = output
    if final_output is None:
        raise RuntimeError("engine completed without a response")
    return final_output


async def _stream_chat(
    engine,
    prompt: str,
    sampling_params,
    request_id: str,
    created: int,
    model: str,
    include_usage: bool,
) -> AsyncGenerator[str, None]:
    prompt_tokens = 0
    completion_tokens = 0
    first_output = True
    async for output in engine.generate(prompt, sampling_params, request_id):
        if output.prompt_token_ids is not None:
            prompt_tokens = len(output.prompt_token_ids)
        generated = output.outputs[0]
        if first_output:
            first_output = False
            yield _sse(
                {
                    "id": request_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": model,
                    "choices": [
                        {
                            "index": 0,
                            "delta": {"role": "assistant", "content": ""},
                            "logprobs": None,
                            "finish_reason": None,
                        }
                    ],
                }
            )
        completion_tokens += len(generated.token_ids)
        yield _sse(
            {
                "id": request_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": model,
                "choices": [
                    {
                        "index": 0,
                        "delta": {"content": generated.text},
                        "logprobs": None,
                        "finish_reason": generated.finish_reason,
                    }
                ],
            }
        )
    if include_usage:
        yield _sse(
            {
                "id": request_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": model,
                "choices": [],
                "usage": _usage(prompt_tokens, completion_tokens),
            }
        )
    yield _sse("[DONE]")


async def _stream_completion(
    engine,
    prompt: str,
    sampling_params,
    request_id: str,
    created: int,
    model: str,
    include_usage: bool,
) -> AsyncGenerator[str, None]:
    prompt_tokens = 0
    completion_tokens = 0
    async for output in engine.generate(prompt, sampling_params, request_id):
        if output.prompt_token_ids is not None:
            prompt_tokens = len(output.prompt_token_ids)
        generated = output.outputs[0]
        completion_tokens += len(generated.token_ids)
        yield _sse(
            {
                "id": request_id,
                "object": "text_completion",
                "created": created,
                "model": model,
                "choices": [
                    {
                        "index": 0,
                        "text": generated.text,
                        "logprobs": None,
                        "finish_reason": generated.finish_reason,
                    }
                ],
            }
        )
    if include_usage:
        yield _sse(
            {
                "id": request_id,
                "object": "text_completion",
                "created": created,
                "model": model,
                "choices": [],
                "usage": _usage(prompt_tokens, completion_tokens),
            }
        )
    yield _sse("[DONE]")


def build_app(
    llm,
    served_model_name: str,
    default_max_tokens: int,
    api_key: str | None,
):
    from fastapi import FastAPI, Header, HTTPException

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        yield
        shutdown = getattr(llm, "shutdown", None)
        if shutdown is not None:
            shutdown()

    app = FastAPI(title="HXinfer OpenAI-compatible API", lifespan=lifespan)

    def authorize(authorization: str | None) -> None:
        if api_key is None:
            return
        if authorization != f"Bearer {api_key}":
            raise HTTPException(status_code=401, detail="invalid API key")

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

    async def run_chat(body: dict[str, Any]) -> dict[str, Any]:
        messages = _messages_from_request(body)
        sampling_params = _sampling_params(body, default_max_tokens)
        prompt = await _chat_prompt(llm, messages)
        request_id = f"chatcmpl-{uuid.uuid4().hex}"
        output = await _collect_final(llm, prompt, sampling_params, request_id)
        generated = output.outputs[0]
        prompt_tokens = len(output.prompt_token_ids)
        completion_tokens = len(generated.token_ids)
        return {
            "id": request_id,
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
            "usage": _usage(prompt_tokens, completion_tokens),
        }

    @app.post("/v1/chat/completions")
    async def chat_completions(
        body: dict[str, Any],
        authorization: str | None = Header(default=None),
    ):
        authorize(authorization)
        requested_model = body.get("model")
        if requested_model not in (None, served_model_name):
            raise HTTPException(status_code=404, detail="model not found")
        try:
            if body.get("stream", False):
                from fastapi.responses import StreamingResponse

                messages = _messages_from_request(body)
                sampling_params = _sampling_params(
                    body, default_max_tokens, stream=True
                )
                prompt = await _chat_prompt(llm, messages)
                request_id = f"chatcmpl-{uuid.uuid4().hex}"
                include_usage = bool(
                    (body.get("stream_options") or {}).get("include_usage", False)
                )
                return StreamingResponse(
                    _stream_chat(
                        llm,
                        prompt,
                        sampling_params,
                        request_id,
                        int(time.time()),
                        served_model_name,
                        include_usage,
                    ),
                    media_type="text/event-stream",
                )
            return await run_chat(body)
        except ValueError as error:
            raise HTTPException(status_code=400, detail=str(error)) from error

    async def run_completion(body: dict[str, Any]) -> dict[str, Any]:
        prompt = body.get("prompt")
        if not isinstance(prompt, str) or not prompt:
            raise ValueError("prompt must be a non-empty string")
        sampling_params = _sampling_params(body, default_max_tokens)
        request_id = f"cmpl-{uuid.uuid4().hex}"
        output = await _collect_final(llm, prompt, sampling_params, request_id)
        generated = output.outputs[0]
        prompt_tokens = len(output.prompt_token_ids)
        completion_tokens = len(generated.token_ids)
        return {
            "id": request_id,
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
            "usage": _usage(prompt_tokens, completion_tokens),
        }

    @app.post("/v1/completions")
    async def completions(
        body: dict[str, Any],
        authorization: str | None = Header(default=None),
    ):
        authorize(authorization)
        requested_model = body.get("model")
        if requested_model not in (None, served_model_name):
            raise HTTPException(status_code=404, detail="model not found")
        try:
            if body.get("stream", False):
                from fastapi.responses import StreamingResponse

                prompt = body.get("prompt")
                if not isinstance(prompt, str) or not prompt:
                    raise ValueError("prompt must be a non-empty string")
                sampling_params = _sampling_params(
                    body, default_max_tokens, stream=True
                )
                request_id = f"cmpl-{uuid.uuid4().hex}"
                include_usage = bool(
                    (body.get("stream_options") or {}).get("include_usage", False)
                )
                return StreamingResponse(
                    _stream_completion(
                        llm,
                        prompt,
                        sampling_params,
                        request_id,
                        int(time.time()),
                        served_model_name,
                        include_usage,
                    ),
                    media_type="text/event-stream",
                )
            return await run_completion(body)
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

    from vllm.engine.arg_utils import AsyncEngineArgs
    from vllm.v1.engine.async_llm import AsyncLLM

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

    served_model_name = (
        args.served_model_name or args.model.rstrip("/").rsplit("/", 1)[-1]
    )

    import uvicorn

    async def serve() -> None:
        llm = AsyncLLM.from_engine_args(AsyncEngineArgs(**kwargs))
        app = build_app(llm, served_model_name, args.max_tokens, args.api_key)
        print(
            f"HXINFER_API_READY_PENDING host={args.host} port={args.port} "
            f"model={served_model_name}",
            flush=True,
        )
        server = uvicorn.Server(
            uvicorn.Config(
                app,
                host=args.host,
                port=args.port,
                log_level="info",
            )
        )
        await server.serve()

    asyncio.run(serve())


if __name__ == "__main__":
    main()
