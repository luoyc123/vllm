# vLLM 0.27.1 Pipeline Parallel 源码分析

## 分析基线与结论边界

分析对象是项目隔离环境中的最新稳定 wheel：

```text
vLLM: 0.27.1+cu129
source: /home/arden/HXinfer/.venv/lib/python3.13/site-packages/vllm
PyTorch: 2.13.0+cu129
transformers: 5.15.1
```

`pip check`、vLLM CLI/API 和单卡 Qwen3-0.6B 推理均已通过。当前 WSL 主机只有一张 RTX 4060 Laptop 8GB，因此不能执行 PP=2。WSL 不提供 V2 Model Runner 所需的 UVA，本地 smoke 使用 `VLLM_USE_V2_MODEL_RUNNER=0`；租机上的原生 Linux 双卡 baseline 应保留默认 V2 runner 并重新验证本文路径。正式租机若 checkout/安装了其他 vLLM commit，应重新核对本文行号和结论。

核心结论：vLLM 原生 PP 是一个逻辑 Engine 管理多个 Worker；Worker 管理由 Multiprocessing 或 Ray executor 负责，而 PP activation 的数据面是 `GroupCoordinator` 上的 tensor-dict 点对点通信。两者不是同一概念。

## 1. 一个逻辑 vLLM 如何创建 PP Worker / PP Rank

调用链概括如下：

```text
LLMEngine / AsyncLLM
  -> Executor.get_class(vllm_config)
  -> MultiprocExecutor 或 RayDistributedExecutor
  -> 每个 global rank 初始化一个 Worker
  -> init_distributed_environment
  -> initialize_model_parallel
  -> 构造 TP / PP / DP GroupCoordinator
  -> Worker.load_model
```

`vllm/v1/executor/multiproc_executor.py:103-190` 中，`world_size` 被检查为 TP、PP 和 PCP 的乘积，然后逐 local rank 调用 `WorkerProc.make_worker_process(...)`。因此在 `TP=1, PP=2` 时，一个 executor 启动两个 worker process，每个 process 对应一个 global rank/GPU。

Ray 路径在 `vllm/v1/executor/ray_executor.py:330-380` 中把 placement group 内 worker 排序后逐一传入 `rank`、`local_rank` 和同一个 `distributed_init_method`，再统一执行 `init_worker`、`init_device`、`load_model`。`pp_tp_workers` 只是 executor 对 worker 的逻辑编组，不是 activation transport。

模型并行 group 的 rank layout 位于 `vllm/distributed/parallel_state.py:1770-1900`：

```python
# layout: ExternalDP x DP x PP x PCP x TP
all_ranks = torch.arange(world_size).reshape(
    -1, data_parallel_size, pipeline_model_parallel_size,
    prefill_context_model_parallel_size, tensor_model_parallel_size,
)

group_ranks = (
    all_ranks.transpose(2, 4)
    .reshape(-1, pipeline_model_parallel_size)
    .unbind(0)
)
_PP = init_model_parallel_group(
    [x.tolist() for x in group_ranks],
    get_world_group().local_rank,
    backend,
    group_name="pp",
)
```

`get_pp_group()` 本身只是返回已经初始化的全局 `_PP: GroupCoordinator`（`parallel_state.py:1405-1412`）。

## 2. 模型如何按 PP Rank 切 Layers

公共切层入口是 `vllm/model_executor/models/utils.py:798` 的 `make_layers()`：

```python
start_layer, end_layer = get_pp_indices(
    num_hidden_layers,
    get_pp_group().rank_in_group,
    get_pp_group().world_size,
)
modules = torch.nn.ModuleList(
    [PPMissingLayer() for _ in range(start_layer)]
    + [layer_fn(prefix=f"{prefix}.{idx}")
       for idx in range(start_layer, end_layer)]
    + [PPMissingLayer() for _ in range(end_layer, num_hidden_layers)]
)
```

即每个 worker 构造相同的模型外壳，只实例化自己 `[start_layer, end_layer)` 内的真实 decoder layers，其他位置放 `PPMissingLayer`。embedding 通常仅在 first PP rank，final norm/lm head 通常仅在 last PP rank。具体边界仍由各模型实现决定，不能假设所有模型完全相同。

以 Qwen2 为例，`vllm/model_executor/models/qwen2.py:365-433`：first rank 创建 embedding；所有 rank 经 `make_layers` 得到自己的层；last rank 创建 final norm。脚本 `inspect_partition.py` 可以提前计算预期边界，但最终需要和 worker 实际日志/模型对象核对。

## 3. Stage 间传递哪些 IntermediateTensors

容器类型是 `vllm/sequence.py:12` 的 `IntermediateTensors`，核心字段为 `tensors: dict[str, torch.Tensor]`。实际 key 由模型定义，不是固定单 tensor。

Qwen2 在非末 rank 返回：

```python
return IntermediateTensors({
    "hidden_states": hidden_states,
    "residual": residual,
})
```

下一 rank 从同名 key 读取。因此本项目 transport 必须传整个 mapping，同时传递 shape、dtype、request/step 顺序信息，不能只为 `hidden_states` 特化。不同模型可能产生其他 key；上机选择模型后必须从该模型的 `make_empty_intermediate_tensors` 和 `forward` 实际确认。

V1 常规执行路径在 `vllm/v1/worker/gpu_worker.py:1064-1105`。0.27.1 已将 PP activation 收发改为可异步重叠：非首 rank 用 `irecv_tensor_dict` 得到 handles，并包装为按需等待的 `AsyncIntermediateTensors`；非末 rank 用 `isend_tensor_dict` 启动发送。

```python
if forward_pass and not get_pp_group().is_first_rank:
    tensor_dict, comm_handles, comm_postprocess = (
        get_pp_group().irecv_tensor_dict(...)
    )
    intermediate_tensors = AsyncIntermediateTensors(
        tensor_dict, comm_handles=comm_handles,
        comm_postprocess=comm_postprocess,
    )

output = self.model_runner.execute_model(scheduler_output, intermediate_tensors)

self._pp_send_work = get_pp_group().isend_tensor_dict(output.tensors, ...)
```

由此可见，scheduler/executor 控制 worker 执行；真正的 stage boundary payload 是 model runner 返回的 `IntermediateTensors.tensors`。

## 4. PP Tensor 使用什么通信 backend

`GroupCoordinator.send_tensor_dict()` / `isend_tensor_dict()` / `recv_tensor_dict()` / `irecv_tensor_dict()` 位于 `vllm/distributed/parallel_state.py:981-1160`。同步方法现在是异步方法加逐 handle `wait()` 的包装。流程为：

1. 用 CPU group 发送包含 key、shape、dtype、device 等内容的 metadata；
2. 每个 CPU tensor 经 CPU/Gloo group 发送；
3. 每个 GPU tensor 经 `device_group` 调用 `torch.distributed.isend/irecv`；
4. 接收端先按 metadata 在原 device 分配 tensor，再接收 payload；
5. TP>1 时可选择只 send slice 后 all-gather；本项目 baseline TP=1，不触发有效切片优化。

关键数据路径为：

```python
# isend_tensor_dict
self.send_object(metadata_list, dst=dst)  # cpu_group
handle = torch.distributed.isend(
    tensor, dst=self.ranks[dst], group=self.device_group
)

# irecv_tensor_dict
recv_metadata_list = self.recv_object(src=src)
tensor = torch.empty(value.size, dtype=value.dtype, device=value.device)
handle = torch.distributed.irecv(
    tensor, src=self.ranks[src], group=self.device_group
)
```

CUDA platform 的默认 distributed backend 是 `nccl`（`vllm/platforms/cuda.py:214`）；ROCm platform 同样向 PyTorch 传 backend 名称 `nccl`（`vllm/platforms/rocm.py:486`），但 ROCm 构建中的实际 GPU collective library 是 RCCL 的 NCCL-compatible API。两者仍复用 CUDA 风格 device contract。因此“源码字符串是 nccl”不能被误解为 ROCm 使用 NVIDIA 的二进制 NCCL。

`CudaCommunicator.send/recv`（`device_communicators/cuda_communicator.py:532-557`）是单 tensor device communicator 路径。不过 tensor-dict 的默认分支直接调用 `torch.distributed.isend/irecv`；只有 `use_cpu_custom_send_recv` 路径才把整个 dict 委托给 device communicator，且该自定义路径仍是同步的。

原计划点名的 `isend_tensor_dict()` 与 `irecv_tensor_dict()` 在 0.27.1 中均已存在，并且已进入 V1 `gpu_worker.py` 的常规 PP activation 路径。External PP baseline 对比时必须计入这种通信/计算重叠，不能只拿串行拷贝时间与 Native PP 对比。

## 单机与跨机：管理面和数据面分离

| 场景 | Worker 管理（executor backend） | PP Tensor 数据面 |
|---|---|---|
| 单机，默认多 GPU | `MultiprocExecutor`，每 GPU 一个 worker process | `GroupCoordinator` -> `torch.distributed.send/recv` -> CUDA 上通常为 NCCL |
| 跨机 | 常见为 `RayDistributedExecutor`；具体取决于启动配置 | 仍为跨 rank 的 `GroupCoordinator`/device process group；CUDA 上通常为 NCCL，经网络路径由 NCCL/系统拓扑决定 |

Ray Compiled Graph 是另一个执行分支，并可使用 Ray NCCL channel；使用时要单独记录，不能把这个特例泛化为所有 Ray PP。Baseline 应先使用最朴素且可解释的配置，并保存完整启动参数。

## 对 External PP 的最小侵入点

第一版不应重写 HF 模型切层。最值得复用的是：

- `get_pp_indices` / `make_layers` 和各模型已有的 first/last rank 分支；
- 模型已有的 `IntermediateTensors` schema；
- vLLM weight loader 对 `PPMissingLayer` 的处理；
- ModelRunner 的输入准备、KV cache 和 stage-local forward。

最小替换边界位于 `gpu_worker.py` 中 `recv_tensor_dict -> IntermediateTensors -> execute_model -> send_tensor_dict` 这一圈。先通过 Native baseline 和单次 split-forward 证明可控，再决定采用 wrapper/subclass、窄 patch，还是上游可扩展 hook。
