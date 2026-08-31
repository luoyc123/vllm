from __future__ import annotations

import os
import pickle
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import suppress
from typing import Any

import cloudpickle

from external_pp.partition import (
    normalize_layer_partition,
    validate_worker_partitions,
)
from external_pp.remote.rpc import RpcClient


def _endpoints_from_env(expected: int) -> list[str]:
    value = os.environ.get("HXINFER_WORKER_ENDPOINTS", "")
    endpoints = [item.strip() for item in value.split(",") if item.strip()]
    if len(endpoints) != expected:
        raise ValueError(
            f"HXINFER_WORKER_ENDPOINTS must contain {expected} endpoints; "
            f"got {endpoints!r}"
        )
    return endpoints


from vllm.platforms import current_platform
from vllm.v1.core.sched.output import GrammarOutput, SchedulerOutput
from vllm.v1.executor.abstract import Executor
from vllm.v1.outputs import DraftTokenIds, ModelRunnerOutput


class HXinferRemoteExecutor(Executor):
    """One vLLM engine controlling independently launched stage workers."""

    supports_pp = True
    uses_ray = False

    def _init_executor(self) -> None:
        world_size = self.parallel_config.world_size
        if world_size != 2 or self.parallel_config.tensor_parallel_size != 1:
            raise ValueError("HXinferRemoteExecutor currently requires PP=2 and TP=1")
        self.output_rank = 1
        self._pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="hxinfer-rpc")
        self._clients = [RpcClient(endpoint) for endpoint in _endpoints_from_env(2)]
        for client in self._clients:
            client.connect()

        controller_partition = normalize_layer_partition(
            os.getenv("VLLM_PP_LAYER_PARTITION")
        )
        worker_statuses = [client.call("__ping__") for client in self._clients]
        validate_worker_partitions(controller_partition, worker_statuses)
        print(
            "HXINFER_PARTITION_CONFIG "
            f"partition={controller_partition or 'vllm-default'}",
            flush=True,
        )

        distributed_init_method = os.environ.get(
            "HXINFER_DIST_INIT", "tcp://127.0.0.1:29610"
        )
        all_kwargs = [
            {
                "vllm_config": self.vllm_config,
                "local_rank": 0,
                "rank": rank,
                "distributed_init_method": distributed_init_method,
                "is_driver_worker": True,
            }
            for rank in range(world_size)
        ]
        self.collective_rpc("init_worker", args=(all_kwargs,))
        self.collective_rpc("init_device")
        self.collective_rpc("load_model")
        partition_info = self.collective_rpc("get_hxinfer_partition_info")
        print(f"HXINFER_PARTITION_ACTIVE stages={partition_info!r}", flush=True)
        current_platform.update_block_size_for_backend(self.vllm_config)

    def _collective_rpc_sync(
        self,
        method: str | Callable,
        args: tuple,
        kwargs: dict,
        unique_reply_rank: int | None,
    ) -> Any:
        wire_method: str | bytes = (
            method
            if isinstance(method, str)
            else cloudpickle.dumps(method, protocol=pickle.HIGHEST_PROTOCOL)
        )
        request_ids = [
            client.send_call(wire_method, args, kwargs) for client in self._clients
        ]
        results = [
            client.recv_result(request_id)
            for client, request_id in zip(self._clients, request_ids)
        ]
        return results[unique_reply_rank] if unique_reply_rank is not None else results

    def collective_rpc(  # type: ignore[override]
        self,
        method: str | Callable,
        timeout: float | None = None,
        args: tuple = (),
        kwargs: dict | None = None,
        non_block: bool = False,
        unique_reply_rank: int | None = None,
    ) -> Any:
        del timeout
        call = lambda: self._collective_rpc_sync(
            method, args, kwargs or {}, unique_reply_rank
        )
        if non_block:
            return self._pool.submit(call)
        return call()

    def execute_model(
        self, scheduler_output: SchedulerOutput, non_block: bool = False
    ) -> ModelRunnerOutput | None | Future[ModelRunnerOutput | None]:
        return self.collective_rpc(
            "execute_model",
            args=(scheduler_output,),
            non_block=non_block,
            unique_reply_rank=self.output_rank,
        )

    def sample_tokens(
        self, grammar_output: GrammarOutput | None, non_block: bool = False
    ) -> ModelRunnerOutput | Future[ModelRunnerOutput]:
        return self.collective_rpc(
            "sample_tokens",
            args=(grammar_output,),
            non_block=non_block,
            unique_reply_rank=self.output_rank,
        )

    def take_draft_token_ids(self) -> DraftTokenIds | None:
        return self.collective_rpc(
            "take_draft_token_ids", unique_reply_rank=self.output_rank
        )

    def check_health(self) -> None:
        self.collective_rpc("check_health")

    def shutdown(self) -> None:
        clients = getattr(self, "_clients", [])
        for client in clients:
            with suppress(Exception):
                client.call("__shutdown_service__")
            client.close()
        pool = getattr(self, "_pool", None)
        if pool is not None:
            pool.shutdown(wait=False, cancel_futures=True)

    @classmethod
    def supports_async_scheduling(cls) -> bool:
        return False
