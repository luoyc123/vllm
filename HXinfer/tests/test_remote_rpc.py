from __future__ import annotations

import socket
import threading

import pytest

from external_pp.remote.rpc import (
    RpcClient,
    parse_endpoint,
    recv_message,
    send_message,
)


def test_parse_endpoint() -> None:
    assert parse_endpoint("127.0.0.1:1234") == ("127.0.0.1", 1234)
    with pytest.raises(ValueError):
        parse_endpoint("missing-port")


def test_rpc_client_round_trip() -> None:
    try:
        listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        listener.bind(("127.0.0.1", 0))
    except PermissionError:
        if "listener" in locals():
            listener.close()
        pytest.skip("the local sandbox blocks TCP sockets")
    listener.listen(1)
    endpoint = f"127.0.0.1:{listener.getsockname()[1]}"

    def serve() -> None:
        connection, _ = listener.accept()
        request = recv_message(connection)
        send_message(
            connection,
            {
                "id": request["id"],
                "ok": True,
                "result": sum(request["args"]),
            },
        )
        connection.close()
        listener.close()

    thread = threading.Thread(target=serve)
    thread.start()
    client = RpcClient(endpoint)
    assert client.call("sum", args=(2, 3)) == 5
    client.close()
    thread.join(timeout=5)
    assert not thread.is_alive()
