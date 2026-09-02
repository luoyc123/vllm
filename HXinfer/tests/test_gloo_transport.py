from __future__ import annotations

import multiprocessing as mp
import os
import queue
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
from external_pp.transport.gloo import (
    GlooPPTransport,
    _dtype_from_name,
    _dtype_name,
)


def _run_gloo_rank(
    rank: int,
    rendezvous_file: str,
    log_dir: str,
    output_queue,
) -> None:
    os.environ.setdefault("GLOO_SOCKET_IFNAME", "lo")
    dist.init_process_group(
        backend="gloo",
        init_method=f"file://{rendezvous_file}",
        rank=rank,
        world_size=2,
    )
    try:
        transport = GlooPPTransport(
            rank=rank,
            group=dist.group.WORLD,
            ranks=(0, 1),
            log_dir=log_dir,
        )
        if rank == 0:
            transport.send(
                {
                    "hidden_states": torch.arange(12, dtype=torch.float32).reshape(
                        3, 4
                    ),
                    "residual": torch.tensor([1.5, -2.0], dtype=torch.float16),
                    "empty": torch.empty((0, 4), dtype=torch.bfloat16),
                    "request": "r0",
                }
            )
        else:
            received = transport.recv(torch.device("cpu"))
            output_queue.put(
                {
                    "hidden_states": received["hidden_states"].tolist(),
                    "hidden_dtype": str(received["hidden_states"].dtype),
                    "residual": received["residual"].float().tolist(),
                    "empty_shape": list(received["empty"].shape),
                    "empty_dtype": str(received["empty"].dtype),
                    "request": received["request"],
                }
            )
        transport.close()
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_dtype_round_trip(dtype: torch.dtype) -> None:
    assert _dtype_from_name(_dtype_name(dtype)) == dtype


def test_dtype_rejects_unknown_value() -> None:
    with pytest.raises(TypeError, match="unsupported tensor dtype"):
        _dtype_from_name("not_a_dtype")


def test_gloo_transport_real_two_process_round_trip(tmp_path: Path) -> None:
    if not dist.is_available() or not dist.is_gloo_available():
        pytest.skip("PyTorch Gloo is unavailable")

    context = mp.get_context("spawn")
    output_queue = context.Queue()
    rendezvous_file = str(tmp_path / "gloo-rendezvous")
    processes = [
        context.Process(
            target=_run_gloo_rank,
            args=(rank, rendezvous_file, str(tmp_path), output_queue),
        )
        for rank in range(2)
    ]
    for process in processes:
        process.start()
    for process in processes:
        process.join(timeout=20)
        if process.is_alive():
            process.terminate()
            process.join(timeout=5)
            pytest.fail("Gloo transport subprocess timed out")
        assert process.exitcode == 0

    try:
        received = output_queue.get(timeout=2)
    except queue.Empty:
        pytest.fail("receiver did not return a Gloo payload")

    assert received == {
        "hidden_states": [
            [0.0, 1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0, 7.0],
            [8.0, 9.0, 10.0, 11.0],
        ],
        "hidden_dtype": "torch.float32",
        "residual": [1.5, -2.0],
        "empty_shape": [0, 4],
        "empty_dtype": "torch.bfloat16",
        "request": "r0",
    }

    sender_log = (tmp_path / "transport-rank0.jsonl").read_text()
    receiver_log = (tmp_path / "transport-rank1.jsonl").read_text()
    assert '"backend":"gloo"' in sender_log
    assert '"event":"send"' in sender_log
    assert '"event":"recv"' in receiver_log
    assert '"bytes":52' in sender_log
    assert '"bytes":52' in receiver_log
