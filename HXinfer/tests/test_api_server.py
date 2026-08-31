import asyncio
import json
from types import SimpleNamespace

import pytest

from external_pp.api_server import (
    _message_content_as_text,
    _messages_from_request,
    _sse,
    _stream_chat,
    _stream_completion,
    _usage,
    resolve_executor_backend,
)


def test_messages_from_request_normalizes_supported_roles():
    body = {
        "messages": [
            {"role": "system", "content": "Be concise."},
            {"role": "user", "content": "2+2?"},
        ]
    }

    assert _messages_from_request(body) == body["messages"]


def test_messages_from_request_accepts_openai_text_parts():
    body = {
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "hello "},
                    {"type": "text", "text": "world"},
                ],
            }
        ]
    }

    assert _messages_from_request(body) == [
        {"role": "user", "content": "hello world"}
    ]


@pytest.mark.parametrize(
    "content",
    [None, [], [{"type": "text"}], [{"type": "image_url", "image_url": {}}]],
)
def test_message_content_rejects_unsupported_parts(content):
    with pytest.raises(ValueError):
        _message_content_as_text(content)


@pytest.mark.parametrize(
    "body",
    [
        {},
        {"messages": []},
        {"messages": ["not an object"]},
        {"messages": [{"role": "invalid", "content": "x"}]},
        {"messages": [{"role": "user", "content": [{"type": "text"}]}]},
    ],
)
def test_messages_from_request_rejects_invalid_payload(body):
    with pytest.raises(ValueError):
        _messages_from_request(body)


@pytest.mark.parametrize("backend", ["mp", "ray", "uni", "external_launcher"])
def test_resolve_executor_backend_keeps_builtin_names(backend):
    assert resolve_executor_backend(backend) == backend


def test_resolve_executor_backend_imports_class():
    resolved = resolve_executor_backend("pathlib.Path")

    assert resolved.__name__ == "Path"


def _output(text, token_ids, *, prompt_token_ids=None, finish_reason=None):
    return SimpleNamespace(
        prompt_token_ids=prompt_token_ids,
        outputs=[
            SimpleNamespace(
                text=text,
                token_ids=token_ids,
                finish_reason=finish_reason,
            )
        ],
    )


class FakeAsyncEngine:
    def __init__(self, outputs):
        self.outputs = outputs
        self.requests = []

    async def generate(self, prompt, sampling_params, request_id):
        self.requests.append((prompt, sampling_params, request_id))
        for output in self.outputs:
            yield output


async def _collect(generator):
    return [item async for item in generator]


def _decode_sse(item):
    assert item.startswith("data: ") and item.endswith("\n\n")
    return json.loads(item[6:-2])


def test_sse_and_usage_helpers():
    assert _sse("[DONE]") == "data: [DONE]\n\n"
    assert _decode_sse(_sse({"text": "你好"})) == {"text": "你好"}
    assert _usage(3, 2) == {
        "prompt_tokens": 3,
        "completion_tokens": 2,
        "total_tokens": 5,
    }


def test_chat_stream_emits_openai_chunks_usage_and_done():
    engine = FakeAsyncEngine(
        [
            _output("你", [10], prompt_token_ids=[1, 2, 3]),
            _output("好", [11], finish_reason="length"),
        ]
    )

    events = asyncio.run(
        _collect(
            _stream_chat(
                engine,
                "prompt",
                object(),
                "chatcmpl-test",
                123,
                "test-model",
                True,
            )
        )
    )

    role, first, second, usage = map(_decode_sse, events[:-1])
    assert role["choices"][0]["delta"] == {
        "role": "assistant",
        "content": "",
    }
    assert first["choices"][0]["delta"]["content"] == "你"
    assert second["choices"][0]["delta"]["content"] == "好"
    assert second["choices"][0]["finish_reason"] == "length"
    assert usage["choices"] == []
    assert usage["usage"] == _usage(3, 2)
    assert events[-1] == "data: [DONE]\n\n"


def test_completion_stream_emits_text_chunks_without_optional_usage():
    engine = FakeAsyncEngine(
        [_output("answer", [20], prompt_token_ids=[1], finish_reason="stop")]
    )

    events = asyncio.run(
        _collect(
            _stream_completion(
                engine,
                "prompt",
                object(),
                "cmpl-test",
                123,
                "test-model",
                False,
            )
        )
    )

    chunk = _decode_sse(events[0])
    assert chunk["choices"][0]["text"] == "answer"
    assert chunk["choices"][0]["finish_reason"] == "stop"
    assert events[-1] == "data: [DONE]\n\n"
