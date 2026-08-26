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


class UnixSocketPPTransport:
    """A synchronous stage-0 -> stage-1 transport for a two-stage PP run.

    Tensor payloads leave the vLLM/NCCL pipeline group: CUDA tensors are staged
    through pinned host memory and sent over one persistent Unix-domain socket.
    The first version intentionally favors a small, auditable integration over
    overlap; its timings make the copies and socket transfer visible.
    """

    def __init__(self, rank: int, socket_path: str, log_dir: str | None = None):
        if rank not in (0, 1):
            raise ValueError("UnixSocketPPTransport currently supports PP=2 only")
        self.rank = rank
        self.socket_path = socket_path
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
                raise ConnectionError("peer closed the PP socket")
            chunks.append(chunk)
            remaining -= len(chunk)
        return b"".join(chunks)

    def _connect_sender(self) -> socket.socket:
        if self._connection is not None:
            return self._connection
        path = Path(self.socket_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            path.unlink()
        except FileNotFoundError:
            pass
        listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        listener.bind(self.socket_path)
        listener.listen(1)
        self._listener = listener
        self._connection, _ = listener.accept()
        return self._connection

    def _connect_receiver(self) -> socket.socket:
        if self._connection is not None:
            return self._connection
        deadline = time.monotonic() + float(os.getenv("HXINFER_CONNECT_TIMEOUT", "120"))
        while True:
            connection = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            try:
                connection.connect(self.socket_path)
                self._connection = connection
                return connection
            except (FileNotFoundError, ConnectionRefusedError):
                connection.close()
                if time.monotonic() >= deadline:
                    raise TimeoutError(f"timed out connecting to {self.socket_path}")
                time.sleep(0.01)

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
            raise RuntimeError("only PP rank 0 may send in the two-stage prototype")
        started = time.perf_counter_ns()
        staged: dict[str, Any] = {}
        d2h_started = time.perf_counter_ns()
        for key, value in tensor_dict.items():
            if isinstance(value, torch.Tensor) and value.is_cuda:
                # vLLM's native receiver allocates from TensorMetadata.size and
                # therefore always reconstructs a contiguous tensor. Match that
                # contract instead of preserving a view's source strides.
                source = value.detach().contiguous()
                host = torch.empty(
                    source.shape,
                    dtype=source.dtype,
                    device="cpu",
                    pin_memory=True,
                )
                host.copy_(source, non_blocking=True)
                staged[key] = host
            elif isinstance(value, torch.Tensor):
                staged[key] = value.detach().contiguous()
            else:
                staged[key] = value
        if any(
            isinstance(value, torch.Tensor) and value.is_cuda
            for value in tensor_dict.values()
        ):
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
            raise RuntimeError("only PP rank 1 may receive in the two-stage prototype")
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
            if isinstance(value, torch.Tensor):
                value = value.contiguous()
                if device.type == "cpu":
                    result[key] = value
                else:
                    pinned = torch.empty(
                        value.shape,
                        dtype=value.dtype,
                        device="cpu",
                        pin_memory=True,
                    )
                    pinned.copy_(value)
                    result[key] = pinned.to(device=device, non_blocking=True)
            else:
                result[key] = value
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
