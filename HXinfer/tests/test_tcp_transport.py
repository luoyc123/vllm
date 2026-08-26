from __future__ import annotations

import socket
import threading

import pytest
import torch

from external_pp.transport.tcp_socket import TcpSocketPPTransport


def _free_port() -> int:
    probe = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    probe.bind(("127.0.0.1", 0))
    port = probe.getsockname()[1]
    probe.close()
    return port


def test_tcp_transport_round_trip() -> None:
    try:
        port = _free_port()
    except PermissionError:
        pytest.skip("the local sandbox blocks TCP sockets")
    sender = TcpSocketPPTransport(0, "127.0.0.1", port)
    receiver = TcpSocketPPTransport(1, "127.0.0.1", port)
    expected = {
        "hidden_states": torch.arange(12, dtype=torch.float16).reshape(3, 4),
        "request": "r0",
    }
    received: dict = {}
    thread = threading.Thread(
        target=lambda: received.update(receiver.recv(torch.device("cpu")))
    )
    thread.start()
    sender.send(expected)
    thread.join(timeout=5)
    sender.close()
    receiver.close()

    assert not thread.is_alive()
    torch.testing.assert_close(received["hidden_states"], expected["hidden_states"])
    assert received["request"] == "r0"
