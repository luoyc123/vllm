import pytest

from external_pp.api_server import _messages_from_request, resolve_executor_backend


def test_messages_from_request_normalizes_supported_roles():
    body = {
        "messages": [
            {"role": "system", "content": "Be concise."},
            {"role": "user", "content": "2+2?"},
        ]
    }

    assert _messages_from_request(body) == body["messages"]


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
