from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist


def _dtype_name(dtype: torch.dtype) -> str:
    return str(dtype).removeprefix("torch.")


def _dtype_from_name(name: str) -> torch.dtype:
    dtype = getattr(torch, name, None)
    if not isinstance(dtype, torch.dtype):
        raise TypeError(f"unsupported tensor dtype received from Gloo: {name!r}")
    return dtype


class GlooPPTransport:
    """Synchronous PP=2 activation transport over an existing CPU/Gloo group.

    GPU tensors are explicitly staged to CPU before Gloo send/recv and copied
    to the destination accelerator after receipt. The transport does not own
    the process group and therefore does not destroy it in ``close``.
    """

    def __init__(
        self,
        rank: int,
        group: dist.ProcessGroup,
        ranks: list[int] | tuple[int, ...],
        log_dir: str | None = None,
    ) -> None:
        if rank not in (0, 1):
            raise ValueError("GlooPPTransport currently supports PP=2 only")
        if len(ranks) != 2:
            raise ValueError("GlooPPTransport requires exactly two global ranks")
        backend = str(dist.get_backend(group)).lower()
        if "gloo" not in backend:
            raise ValueError(f"GlooPPTransport requires a Gloo group, got {backend}")
        self.rank = rank
        self.group = group
        self.ranks = tuple(int(item) for item in ranks)
        self.log_path = (
            Path(log_dir) / f"transport-rank{rank}.jsonl" if log_dir else None
        )
        self._sequence = 0

    def _log(self, event: str, **fields: Any) -> None:
        if self.log_path is None:
            return
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        record = {
            "backend": "gloo",
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
        staged: list[torch.Tensor] = []
        entries: list[dict[str, Any]] = []
        d2h_started = time.perf_counter_ns()
        has_accelerator_tensor = False
        payload_bytes = 0
        for key, value in tensor_dict.items():
            if not isinstance(value, torch.Tensor):
                entries.append({"key": key, "kind": "object", "value": value})
                continue

            source = value.detach().contiguous()
            if source.device.type != "cpu":
                has_accelerator_tensor = True
                host_tensor = torch.empty(
                    source.shape,
                    dtype=source.dtype,
                    device="cpu",
                    pin_memory=True,
                )
                host_tensor.copy_(source, non_blocking=True)
            else:
                host_tensor = source
            staged.append(host_tensor)
            payload_bytes += host_tensor.numel() * host_tensor.element_size()
            entries.append(
                {
                    "key": key,
                    "kind": "tensor",
                    "shape": list(host_tensor.shape),
                    "dtype": _dtype_name(host_tensor.dtype),
                }
            )
        if has_accelerator_tensor:
            torch.cuda.current_stream().synchronize()
        d2h_ns = time.perf_counter_ns() - d2h_started

        gloo_started = time.perf_counter_ns()
        destination = self.ranks[1]
        dist.send_object_list([entries], dst=destination, group=self.group)
        for tensor in staged:
            if tensor.numel():
                dist.send(tensor, dst=destination, group=self.group)
        gloo_ns = time.perf_counter_ns() - gloo_started
        self._log(
            "send",
            bytes=payload_bytes,
            d2h_ns=d2h_ns,
            gloo_ns=gloo_ns,
            total_ns=time.perf_counter_ns() - started,
            keys=list(tensor_dict),
        )
        self._sequence += 1

    def recv(self, device: torch.device) -> dict[str, torch.Tensor | Any]:
        if self.rank != 1:
            raise RuntimeError("only PP rank 1 may receive in the PP=2 prototype")

        started = time.perf_counter_ns()
        metadata: list[Any] = [None]
        source = self.ranks[0]
        gloo_started = time.perf_counter_ns()
        dist.recv_object_list(metadata, src=source, group=self.group)
        entries = metadata[0]
        if not isinstance(entries, list):
            raise TypeError("invalid Gloo PP metadata")

        result: dict[str, Any] = {}
        received_tensors: list[tuple[str, torch.Tensor]] = []
        payload_bytes = 0
        for entry in entries:
            if not isinstance(entry, dict) or not isinstance(entry.get("key"), str):
                raise TypeError("invalid Gloo PP metadata entry")
            key = entry["key"]
            if entry.get("kind") == "object":
                result[key] = entry.get("value")
                continue
            if entry.get("kind") != "tensor":
                raise ValueError(f"invalid Gloo PP entry kind for {key!r}")
            shape = tuple(int(item) for item in entry["shape"])
            if any(item < 0 for item in shape):
                raise ValueError(f"invalid tensor shape for {key!r}: {shape!r}")
            host_tensor = torch.empty(
                shape,
                dtype=_dtype_from_name(entry["dtype"]),
                device="cpu",
                pin_memory=device.type != "cpu",
            )
            if host_tensor.numel():
                dist.recv(host_tensor, src=source, group=self.group)
            payload_bytes += host_tensor.numel() * host_tensor.element_size()
            received_tensors.append((key, host_tensor))
        gloo_ns = time.perf_counter_ns() - gloo_started

        h2d_started = time.perf_counter_ns()
        for key, host_tensor in received_tensors:
            result[key] = (
                host_tensor
                if device.type == "cpu"
                else host_tensor.to(device=device, non_blocking=True)
            )
        if device.type == "cuda" and received_tensors:
            torch.cuda.current_stream(device).synchronize()
        h2d_ns = time.perf_counter_ns() - h2d_started
        self._log(
            "recv",
            bytes=payload_bytes,
            gloo_ns=gloo_ns,
            h2d_ns=h2d_ns,
            total_ns=time.perf_counter_ns() - started,
            keys=list(result),
        )
        self._sequence += 1
        return result

    def close(self) -> None:
        # The vLLM GroupCoordinator owns this process group.
        return None
