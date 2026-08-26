#!/usr/bin/env python3
from __future__ import annotations

import argparse
import multiprocessing as mp
import socket
import struct
import time
from statistics import median

_LENGTH = struct.Struct("!Q")
_ACK = b"OK"


def recv_exact_into(connection: socket.socket, buffer: bytearray, length: int) -> None:
    view = memoryview(buffer)[:length]
    received = 0
    while received < length:
        count = connection.recv_into(view[received:])
        if count == 0:
            raise ConnectionError("peer closed the connection")
        received += count


def recv_exact(connection: socket.socket, length: int) -> bytes:
    chunks: list[bytes] = []
    remaining = length
    while remaining:
        chunk = connection.recv(remaining)
        if not chunk:
            raise ConnectionError("peer closed the connection")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def server_main(address_pipe) -> None:
    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    listener.bind(("127.0.0.1", 0))
    listener.listen(1)
    address_pipe.send(listener.getsockname())
    address_pipe.close()

    connection, _ = listener.accept()
    connection.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    buffer = bytearray()
    try:
        while True:
            length = _LENGTH.unpack(recv_exact(connection, _LENGTH.size))[0]
            if length == 0:
                connection.sendall(_ACK)
                return
            if len(buffer) < length:
                buffer = bytearray(length)
            recv_exact_into(connection, buffer, length)
            connection.sendall(_ACK)
    finally:
        connection.close()
        listener.close()


def percentile(values: list[int], fraction: float) -> int:
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, int(len(ordered) * fraction + 0.999) - 1))
    return ordered[index]


def run_size(
    connection: socket.socket,
    size_bytes: int,
    iterations: int,
    warmups: int,
) -> None:
    payload = bytes(size_bytes)
    header = _LENGTH.pack(size_bytes)

    for _ in range(warmups):
        connection.sendall(header)
        connection.sendall(payload)
        if recv_exact(connection, len(_ACK)) != _ACK:
            raise RuntimeError("invalid warmup acknowledgment")

    elapsed_values: list[int] = []
    started_all = time.perf_counter_ns()
    for _ in range(iterations):
        started = time.perf_counter_ns()
        connection.sendall(header)
        connection.sendall(payload)
        if recv_exact(connection, len(_ACK)) != _ACK:
            raise RuntimeError("invalid acknowledgment")
        elapsed_values.append(time.perf_counter_ns() - started)
    elapsed_all = time.perf_counter_ns() - started_all

    total_bytes = size_bytes * iterations
    aggregate_gbps = total_bytes / elapsed_all
    p50_ns = int(median(elapsed_values))
    p95_ns = percentile(elapsed_values, 0.95)
    p50_gbps = size_bytes / p50_ns
    p95_gbps = size_bytes / p95_ns
    print(
        f"size_MiB={size_bytes / (1 << 20):g} iterations={iterations} "
        f"total_GiB={total_bytes / (1 << 30):.3f} "
        f"aggregate_GBps={aggregate_gbps:.3f} "
        f"p50_ms={p50_ns / 1e6:.3f} p50_GBps={p50_gbps:.3f} "
        f"p95_ms={p95_ns / 1e6:.3f} p95_GBps={p95_gbps:.3f}",
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Measure full-message TCP loopback delivery between two processes; "
            "the client waits for an ACK sent after the server receives every byte"
        )
    )
    parser.add_argument(
        "--case",
        action="append",
        default=[],
        metavar="MIB:ITERATIONS",
        help="Message size and measured iterations; repeat for multiple cases",
    )
    parser.add_argument("--warmups", type=int, default=2)
    args = parser.parse_args()

    cases = args.case or ["1:64", "16:16", "64:8"]
    parsed_cases: list[tuple[int, int]] = []
    for value in cases:
        size_text, separator, iterations_text = value.partition(":")
        if not separator:
            parser.error(f"invalid --case {value!r}; expected MIB:ITERATIONS")
        try:
            size_mib = int(size_text)
            iterations = int(iterations_text)
        except ValueError:
            parser.error(f"invalid --case {value!r}; values must be integers")
        if size_mib <= 0 or iterations <= 0:
            parser.error("message size and iterations must be positive")
        parsed_cases.append((size_mib << 20, iterations))
    if args.warmups < 0:
        parser.error("--warmups must not be negative")

    context = mp.get_context("spawn")
    parent_pipe, child_pipe = context.Pipe(duplex=False)
    server = context.Process(target=server_main, args=(child_pipe,), daemon=True)
    server.start()
    host, port = parent_pipe.recv()
    parent_pipe.close()

    connection = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    connection.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    connection.connect((host, port))
    try:
        print(
            "mode=two-process TCP-loopback acknowledgment=after-full-receive "
            f"endpoint={host}:{port}",
            flush=True,
        )
        for size_bytes, iterations in parsed_cases:
            run_size(connection, size_bytes, iterations, args.warmups)
        connection.sendall(_LENGTH.pack(0))
        if recv_exact(connection, len(_ACK)) != _ACK:
            raise RuntimeError("invalid shutdown acknowledgment")
    finally:
        connection.close()
        server.join(timeout=5)
        if server.is_alive():
            server.terminate()
            server.join()
        if server.exitcode != 0:
            raise SystemExit(f"server process exited with code {server.exitcode}")


if __name__ == "__main__":
    main()
