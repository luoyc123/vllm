from __future__ import annotations

import socket
import threading

import pytest
import torch

from external_pp.transport.unix_socket import UnixSocketPPTransport


def test_unix_socket_round_trip(tmp_path) -> None:
    socket_path = str(tmp_path / "pp.sock")
    probe = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    try:
        probe.bind(str(tmp_path / "probe.sock"))
    except PermissionError:
        pytest.skip("the local sandbox blocks Unix-domain sockets")
    finally:
        probe.close()
    sender = UnixSocketPPTransport(0, socket_path)
    receiver = UnixSocketPPTransport(1, socket_path)
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

    assert not thread.is_alive()
    torch.testing.assert_close(received["hidden_states"], expected["hidden_states"])
    assert received["request"] == "r0"
