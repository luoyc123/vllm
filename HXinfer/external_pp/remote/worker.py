from __future__ import annotations

import os
from types import MethodType
from typing import Any

import torch
from vllm.v1.worker.gpu_worker import Worker

from external_pp.transport.tcp_socket import TcpSocketPPTransport


class HXinferRemoteStageWorker(Worker):
    """A PP worker whose cross-stage group is CPU-only and data plane is TCP."""

    def init_device(self) -> None:
        if self.parallel_config.pipeline_parallel_size != 2:
            raise ValueError("remote HXinfer prototype requires PP=2")
        if self.parallel_config.tensor_parallel_size != 1:
            raise ValueError("remote HXinfer prototype requires TP=1")

        from vllm.distributed import parallel_state
        from vllm.v1.worker import gpu_worker

        original_init_group = parallel_state.init_model_parallel_group
        original_init_environment = gpu_worker.init_worker_distributed_environment

        def init_group_without_pp_device_communicator(
            group_ranks,
            local_rank,
            backend,
            use_message_queue_broadcaster=False,
            group_name=None,
            use_device_communicator=True,
            use_all2all=False,
        ):
            if group_name == "pp":
                use_device_communicator = False
            return original_init_group(
                group_ranks,
                local_rank,
                backend,
                use_message_queue_broadcaster=use_message_queue_broadcaster,
                group_name=group_name,
                use_device_communicator=use_device_communicator,
                use_all2all=use_all2all,
            )

        def init_cpu_only_environment(
            vllm_config,
            rank,
            distributed_init_method=None,
            local_rank=-1,
            backend="nccl",
        ):
            del backend
            return original_init_environment(
                vllm_config,
                rank,
                distributed_init_method,
                local_rank,
                "gloo",
            )

        parallel_state.init_model_parallel_group = (
            init_group_without_pp_device_communicator
        )
        gpu_worker.init_worker_distributed_environment = init_cpu_only_environment
        self.parallel_config.disable_custom_all_reduce = True
        try:
            super().init_device()
        finally:
            parallel_state.init_model_parallel_group = original_init_group
            gpu_worker.init_worker_distributed_environment = original_init_environment

        pp_group = parallel_state.get_pp_group()
        host = os.environ.get("HXINFER_ACTIVATION_HOST", "127.0.0.1")
        port = int(os.environ.get("HXINFER_ACTIVATION_PORT", "29620"))
        transport = TcpSocketPPTransport(
            rank=pp_group.rank_in_group,
            host=host,
            port=port,
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
            assert self.device is not None
            return transport.recv(self.device), [], []

        pp_group.isend_tensor_dict = MethodType(external_send, pp_group)
        pp_group.irecv_tensor_dict = MethodType(external_recv, pp_group)

    def shutdown(self) -> None:
        transport = getattr(self, "hxinfer_transport", None)
        if transport is not None:
            transport.close()
        super().shutdown()
