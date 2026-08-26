from __future__ import annotations

import io
import json
import os
import socket
import struct
import time
from pathlib import Path
from typing import Any

import torch

_LENGTH = struct.Struct("!Q")


class TcpSocketPPTransport:
    """Synchronous stage-0 -> stage-1 transport over a persistent TCP socket."""

    def __init__(
        self,
        rank: int,
        host: str,
        port: int,
        log_dir: str | None = None,
    ) -> None:
        if rank not in (0, 1):
            raise ValueError("TcpSocketPPTransport currently supports PP=2 only")
        self.rank = rank
        self.host = host
        self.port = port
        self.log_path = (
            Path(log_dir) / f"transport-rank{rank}.jsonl" if log_dir else None
        )
        self._listener: socket.socket | None = None
        self._connection: socket.socket | None = None
        self._sequence = 0

    @staticmethod
    def _recv_exact(connection: socket.socket, length: int) -> bytes:
        chunks: list[bytes] = []
        remaining = length
        while remaining:
            chunk = connection.recv(remaining)
            if not chunk:
                raise ConnectionError("peer closed the PP TCP connection")
            chunks.append(chunk)
            remaining -= len(chunk)
        return b"".join(chunks)

    def _connect_sender(self) -> socket.socket:
        if self._connection is not None:
            return self._connection
        listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        listener.bind((self.host, self.port))
        listener.listen(1)
        self._listener = listener
        self._connection, _ = listener.accept()
        self._connection.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        return self._connection

    def _connect_receiver(self) -> socket.socket:
        if self._connection is not None:
            return self._connection
        deadline = time.monotonic() + float(os.getenv("HXINFER_CONNECT_TIMEOUT", "120"))
        while True:
            connection = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            connection.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            try:
                connection.connect((self.host, self.port))
                self._connection = connection
                return connection
            except (ConnectionRefusedError, TimeoutError, OSError):
                connection.close()
                if time.monotonic() >= deadline:
                    raise TimeoutError(
                        f"timed out connecting to PP transport {self.host}:{self.port}"
                    )
                time.sleep(0.05)

    def _log(self, event: str, **fields: Any) -> None:
        if self.log_path is None:
            return
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        record = {
            "event": event,
            "rank": self.rank,
            "sequence": self._sequence,
            "time_ns": time.time_ns(),
            **fields,
        }
        with self.log_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, separators=(",", ":")) + "\n")

    def send(self, tensor_dict: dict[str, torch.Tensor | Any]) -> None:
        if self.rank != 0:
            raise RuntimeError("only PP rank 0 may send in the PP=2 prototype")
        started = time.perf_counter_ns()
        staged: dict[str, Any] = {}
        d2h_started = time.perf_counter_ns()
        has_cuda = False
        for key, value in tensor_dict.items():
            if isinstance(value, torch.Tensor) and value.is_cuda:
                has_cuda = True
                source = value.detach().contiguous()
                host_tensor = torch.empty(
                    source.shape,
                    dtype=source.dtype,
                    device="cpu",
                    pin_memory=True,
                )
                host_tensor.copy_(source, non_blocking=True)
                staged[key] = host_tensor
            elif isinstance(value, torch.Tensor):
                staged[key] = value.detach().contiguous()
            else:
                staged[key] = value
        if has_cuda:
            torch.cuda.current_stream().synchronize()
        d2h_ns = time.perf_counter_ns() - d2h_started

        buffer = io.BytesIO()
        torch.save(staged, buffer)
        payload = buffer.getvalue()
        connection = self._connect_sender()
        socket_started = time.perf_counter_ns()
        connection.sendall(_LENGTH.pack(len(payload)))
        connection.sendall(payload)
        socket_ns = time.perf_counter_ns() - socket_started
        self._log(
            "send",
            bytes=len(payload),
            d2h_ns=d2h_ns,
            socket_ns=socket_ns,
            total_ns=time.perf_counter_ns() - started,
            keys=list(tensor_dict),
        )
        self._sequence += 1

    def recv(self, device: torch.device) -> dict[str, torch.Tensor | Any]:
        if self.rank != 1:
            raise RuntimeError("only PP rank 1 may receive in the PP=2 prototype")
        started = time.perf_counter_ns()
        connection = self._connect_receiver()
        socket_started = time.perf_counter_ns()
        length = _LENGTH.unpack(self._recv_exact(connection, _LENGTH.size))[0]
        payload = self._recv_exact(connection, length)
        socket_ns = time.perf_counter_ns() - socket_started
        staged = torch.load(io.BytesIO(payload), map_location="cpu", weights_only=False)

        h2d_started = time.perf_counter_ns()
        result: dict[str, Any] = {}
        for key, value in staged.items():
            if not isinstance(value, torch.Tensor):
                result[key] = value
                continue
            value = value.contiguous()
            if device.type == "cpu":
                result[key] = value
                continue
            pinned = torch.empty(
                value.shape,
                dtype=value.dtype,
                device="cpu",
                pin_memory=True,
            )
            pinned.copy_(value)
            result[key] = pinned.to(device=device, non_blocking=True)
        if device.type == "cuda" and any(
            isinstance(value, torch.Tensor) for value in staged.values()
        ):
            torch.cuda.current_stream(device).synchronize()
        h2d_ns = time.perf_counter_ns() - h2d_started
        self._log(
            "recv",
            bytes=length,
            socket_ns=socket_ns,
            h2d_ns=h2d_ns,
            total_ns=time.perf_counter_ns() - started,
            keys=list(result),
        )
        self._sequence += 1
        return result

    def close(self) -> None:
        for connection in (self._connection, self._listener):
            if connection is not None:
                try:
                    connection.close()
                except OSError:
                    pass
        self._connection = None
        self._listener = None
