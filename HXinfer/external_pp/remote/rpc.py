from __future__ import annotations

import pickle
import socket
import struct
import threading
import time
from typing import Any

_LENGTH = struct.Struct("!Q")
_MAX_FRAME_BYTES = 1 << 30


def parse_endpoint(value: str) -> tuple[str, int]:
    host, separator, port = value.rpartition(":")
    if not separator or not host or not port.isdigit():
        raise ValueError(f"invalid endpoint {value!r}; expected HOST:PORT")
    return host, int(port)


def recv_exact(connection: socket.socket, length: int) -> bytes:
    chunks: list[bytes] = []
    remaining = length
    while remaining:
        chunk = connection.recv(remaining)
        if not chunk:
            raise ConnectionError("RPC peer closed the connection")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def send_message(connection: socket.socket, value: Any) -> None:
    payload = pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL)
    if len(payload) > _MAX_FRAME_BYTES:
        raise ValueError(f"RPC frame is too large: {len(payload)} bytes")
    connection.sendall(_LENGTH.pack(len(payload)))
    connection.sendall(payload)


def recv_message(connection: socket.socket) -> Any:
    length = _LENGTH.unpack(recv_exact(connection, _LENGTH.size))[0]
    if length > _MAX_FRAME_BYTES:
        raise ValueError(f"RPC frame is too large: {length} bytes")
    return pickle.loads(recv_exact(connection, length))


class RpcClient:
    def __init__(self, endpoint: str, connect_timeout: float = 120.0) -> None:
        self.endpoint = endpoint
        self.host, self.port = parse_endpoint(endpoint)
        self.connect_timeout = connect_timeout
        self._connection: socket.socket | None = None
        self._lock = threading.Lock()
        self._next_id = 0

    def connect(self) -> None:
        if self._connection is not None:
            return
        deadline = time.monotonic() + self.connect_timeout
        while True:
            connection = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            connection.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            try:
                connection.connect((self.host, self.port))
                self._connection = connection
                return
            except OSError:
                connection.close()
                if time.monotonic() >= deadline:
                    raise TimeoutError(
                        f"timed out connecting to worker {self.endpoint}"
                    )
                time.sleep(0.1)

    def send_call(self, method: str | bytes, args: tuple, kwargs: dict) -> int:
        self.connect()
        assert self._connection is not None
        request_id = self._next_id
        self._next_id += 1
        send_message(
            self._connection,
            {
                "id": request_id,
                "method": method,
                "args": args,
                "kwargs": kwargs,
            },
        )
        return request_id

    def recv_result(self, request_id: int) -> Any:
        assert self._connection is not None
        response = recv_message(self._connection)
        if response.get("id") != request_id:
            raise RuntimeError(
                f"RPC response id mismatch from {self.endpoint}: "
                f"expected {request_id}, got {response.get('id')}"
            )
        if not response.get("ok"):
            raise RuntimeError(
                f"remote worker {self.endpoint} failed: {response.get('error')}\n"
                f"{response.get('traceback', '')}"
            )
        return response.get("result")

    def call(
        self, method: str | bytes, args: tuple = (), kwargs: dict | None = None
    ) -> Any:
        with self._lock:
            request_id = self.send_call(method, args, kwargs or {})
            return self.recv_result(request_id)

    def close(self) -> None:
        if self._connection is not None:
            try:
                self._connection.close()
            except OSError:
                pass
            self._connection = None
