from __future__ import annotations

import os
from types import MethodType
from typing import Any

import torch
from vllm.distributed.parallel_state import get_pp_group
from vllm.v1.worker.gpu_worker import Worker

from external_pp.transport.unix_socket import UnixSocketPPTransport


class HXinferWorker(Worker):
    """vLLM worker that replaces only PP activation send/receive."""

    def init_device(self) -> None:
        super().init_device()
        if self.parallel_config.pipeline_parallel_size != 2:
            raise ValueError(
                "HXinfer external PP currently requires pipeline_parallel_size=2"
            )
        if self.parallel_config.tensor_parallel_size != 1:
            raise ValueError("HXinfer external PP prototype currently requires TP=1")

        pp_group = get_pp_group()
        transport = UnixSocketPPTransport(
            rank=pp_group.rank_in_group,
            socket_path=os.environ.get("HXINFER_SOCKET_PATH", "/tmp/hxinfer-pp.sock"),
            log_dir=os.environ.get("HXINFER_LOG_DIR"),
        )
        self.hxinfer_transport = transport

        def external_send(
            _group: Any,
            tensor_dict: dict[str, torch.Tensor | Any],
            dst: int | None = None,
            all_gather_group: Any = None,
            all_gather_tensors: dict[str, bool] | None = None,
        ) -> list[Any]:
            del dst, all_gather_group, all_gather_tensors
            transport.send(tensor_dict)
            return []

        def external_recv(
            _group: Any,
            src: int | None = None,
            all_gather_group: Any = None,
            all_gather_tensors: dict[str, bool] | None = None,
        ) -> tuple[dict[str, torch.Tensor | Any], list[Any], list[Any]]:
            del src, all_gather_group, all_gather_tensors
            return transport.recv(self.device), [], []

        pp_group.isend_tensor_dict = MethodType(external_send, pp_group)
        pp_group.irecv_tensor_dict = MethodType(external_recv, pp_group)
