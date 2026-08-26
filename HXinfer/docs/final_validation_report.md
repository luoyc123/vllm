# HXinfer：vLLM Pipeline Parallel、通信后端、原型实现与异构迁移完整报告

> 版本：v3.1（收尾版）
> 日期：2026-08-24
> 源码与实验基线：vLLM 0.27.1
> 状态：独立远程 worker PP 已在双 4090 跑通；短输出 correctness 通过；CUDA/ROCm 待目标机验证

## 0. 执行摘要

本报告覆盖五个方面：vLLM 单机与跨机 PP 管理机制；Ray 与 NCCL/RCCL 的职责边界；NVLink、PCIe、InfiniBand/RDMA、RoCE、TCP 的区别；HXinfer 的实现；以及从双 NVIDIA 验证迁移至同机 NVIDIA + AMD 和双 VM 的方案。

核心结论如下：

1. vLLM 的 executor backend 和 tensor communication backend 是两层概念。Multiprocessing/Ray 负责创建、放置和调用 worker；`GroupCoordinator` 与 PyTorch ProcessGroup 才负责 PP tensor 数据面。
2. 单机默认通常是 multiprocessing，每 GPU 一个 vLLM worker process；CUDA GPU tensor 默认经 NCCL，metadata/CPU tensor经 CPU group（通常为 Gloo）。NCCL 再根据拓扑选择 NVLink、PCIe P2P、共享内存或网络路径。
3. 跨机 Ray 模式下，各节点先组成 Ray 集群，然后只需在一个节点执行一次 `vllm serve`。其他节点只运行 Ray runtime，vLLM driver 会创建远程 GPU worker actors；它们不另开 API server。
4. 新版 vLLM 也支持跨机 multiprocessing：每个节点都运行 `vllm serve`，只有 node-rank 0 正常暴露 API，其他节点必须使用 `--headless`。
5. HXinfer 现在支持一个 vLLM EngineCore 管理两个预先独立启动的 vLLM worker service。Controller 与 worker 通过 TCP RPC 通信，PP activation 通过另一条 TCP 通道传输；worker 不再由 `MultiprocExecutor` fork，也不各自暴露 API。这正对应未来“一台 VM 一个 worker、一个统一 server”的边界。
6. 双 RTX 4090 实验确认了独立 worker 的真实 PP 分层、kernel warm-up、prefill 和连续 decode。Qwen2.5-3B 的 8 道简单题 8/8 正确且与 Native token 完全一致；6 个正常问题累计生成 690 token，完成 261 对 activation 消息，无丢包或死锁。长输出仍存在确定性 token 分叉，因此严格长序列 correctness Gate 仍为 FAIL。
7. 当前 4090 拓扑没有 NVLink且 CUDA P2P 为 `NS`。HXinfer 显式执行 GPU0→pinned host→TCP→pinned host→GPU1，所以 GPU 两侧拷贝均走 PCIe，但不是 PCIe GPU-direct P2P。该 TCP/host-staging 边界可迁移到双 VM；CUDA+ROCm 本身仍需目标机验证。
8. HXinfer 运行两个独立的 vLLM Worker OS 进程，但不采用两套独立 `vllm serve`。两个完整 server 各有自己的 Engine、scheduler、请求状态和 KV cache，默认不会共同完成一个 PP 请求，因此不适合作为本项目的基本边界。

## 1. vLLM PP 基础知识

### 1.1 PP、TP 和 worker rank

Pipeline Parallelism（PP）按层切模型：stage0 计算前半段层并产生 activation，stage1 接收 activation 后继续计算后半段层。Tensor Parallelism（TP）则把同一层内部的矩阵切到多个设备。

忽略 DP/PCP 等额外维度时：

```text
world_size = pipeline_parallel_size × tensor_parallel_size
```

例如 `PP=2, TP=1` 需要两个 worker；`PP=2, TP=8` 需要 16 个 worker。一个 worker 通常对应一个 global rank 和一张 GPU。

PP 的优点是每个 stage 只加载部分层，适合单卡放不下模型、跨节点带宽有限或设备不均衡的情况；代价是 stage 间必须传 activation，并存在 pipeline bubble。对逐 token decode 来说，每一步都跨 PP boundary，通信延迟比只做一次大 prefill 更敏感。

### 1.2 必须分开的三个层次

| 层次 | 解决的问题 | 典型实现 |
|---|---|---|
| 服务/控制面 | HTTP、请求排队、调度、采样结果返回 | API server、EngineCore、scheduler |
| Worker 管理面 | worker 在哪里创建、如何 RPC、如何发现 GPU | multiprocessing、Ray executor |
| Tensor 数据面 | rank 间 tensor 如何真正搬运 | ProcessGroup NCCL/RCCL/Gloo、Ray NCCL channel、HXinfer transport |

所以“Ray 是 PP 通信后端”并不准确。通常 Ray 是 worker 管理和调用框架，PP GPU tensor 仍由 NCCL/RCCL 数据面发送；只有启用 Ray Compiled Graph 的特定路径时，Ray channel 才直接承载中间 tensor。

### 1.3 单机多卡：进程和数据流

vLLM 官方说明，单机分布式默认 runtime 是 native Python multiprocessing；可以通过 `--distributed-executor-backend mp|ray` 显式覆盖。[vLLM Parallelism and Scaling](https://docs.vllm.ai/en/stable/serving/parallelism_scaling/)

以 `PP=2, TP=1` 为例：

```text
API / LLM caller
       │
       ▼
EngineCore + scheduler
       │ scheduler output / control RPC
       ├──────────────────────────┐
       ▼                          ▼
Worker rank0 / PP0 / GPU0      Worker rank1 / PP1 / GPU1
embedding + layers 0..K-1      layers K..N-1 + norm + lm_head
       │                          ▲
       └── IntermediateTensors ───┘
             GPU data: NCCL
             metadata: CPU group
```

`MultiprocExecutor` 先检查 world size 与 TP×PP×PCP 相等，然后按 local rank 调用 `WorkerProc.make_worker_process`。源码证据见 [multiproc_executor.py（v0.27.1）](https://github.com/vllm-project/vllm/blob/v0.27.1/vllm/v1/executor/multiproc_executor.py)。每个 worker 依次执行 `init_worker`、`init_device`、`load_model`，拥有自己的 CUDA context、模型分片、KV cache 和 ModelRunner。

#### 模型如何切层

公共入口 `make_layers()` 根据 PP rank 调用 `get_pp_indices()`，只实例化 `[start_layer, end_layer)` 内的真实层，其余位置放 `PPMissingLayer`：[models/utils.py（v0.27.1）](https://github.com/vllm-project/vllm/blob/v0.27.1/vllm/model_executor/models/utils.py#L798-L832)。

简化后的源码逻辑为：

```python
start_layer, end_layer = get_pp_indices(
    num_hidden_layers, get_pp_group().rank_in_group,
    get_pp_group().world_size,
)
modules = [PPMissingLayer() ...] + local_layers + [PPMissingLayer() ...]
```

模型实现还决定 embedding、final norm 和 lm head 放在哪个 stage。以 Qwen2 为例，first rank 持有 embedding，last rank 持有 final norm；非末 rank 返回 `hidden_states` 与 `residual`。

#### Stage 间传什么

传输对象不是固定的单个 hidden state，而是 `IntermediateTensors.tensors` 的完整 mapping。Qwen2 的典型 boundary 是：

```python
IntermediateTensors({
    "hidden_states": hidden_states,
    "residual": residual,
})
```

不同模型可能增加其他 key，因此外置 transport 必须支持通用 tensor mapping，而不能写死一个 tensor。

#### PP tensor 如何发送

vLLM 0.27.1 的 V1 worker 在非首 rank 调用 `irecv_tensor_dict()`，执行本 stage 模型后，非末 rank 调用 `isend_tensor_dict()`：[gpu_worker.py（v0.27.1）](https://github.com/vllm-project/vllm/blob/v0.27.1/vllm/v1/worker/gpu_worker.py#L1058-L1107)。接收结果包装成 `AsyncIntermediateTensors`，允许把等待与计算准备尽量重叠。

`GroupCoordinator` 的 tensor-dict 路径先通过 CPU group 发送 key/shape/dtype/device metadata，再对每个 tensor 调用 `torch.distributed.isend/irecv`：[parallel_state.py（v0.27.1）](https://github.com/vllm-project/vllm/blob/v0.27.1/vllm/distributed/parallel_state.py#L1019-L1194)。

```python
self.send_object(metadata_list, dst=dst)        # CPU group
handle = torch.distributed.isend(
    tensor, dst=self.ranks[dst], group=comm_group
)
```

CUDA platform 默认 `dist_backend="nccl"`。ROCm platform 在 PyTorch 接口上也使用字符串 `"nccl"`，但 ROCm wheel 中实际由 RCCL 提供 NCCL-compatible 实现，不代表 AMD 在加载 NVIDIA NCCL 二进制。

### 1.4 单机通信路径不是固定的一条线

在 CUDA 原生 PP 中，vLLM 提交的是 NCCL P2P send/recv。NCCL 会按机器拓扑选择实际 transport：

- GPU 之间存在 NVLink：优先走 NVLink P2P；
- 只有 PCIe 且 CUDA P2P 可用：走 PCIe peer access；
- P2P 不可用：可能退化到共享 host memory staging；
- 跨 CPU socket或被配置禁用 SHM：还可能走网络 transport。

NVIDIA 官方说明 NCCL 在本机优化 NVLink、PCIe 与 shared memory，跨机使用 sockets 或 InfiniBand verbs；GPU Direct 是否生效取决于实际拓扑。[NCCL release documentation](https://docs.nvidia.com/deeplearning/nccl/archives/nccl_2243/release-notes/rel_2.0.5.html)

因此启动参数中的 `backend=nccl` 只说明使用哪一个通信库，不能单凭它断言物理数据一定经过 NVLink、PCIe P2P 或 RDMA。必须同时保存：

```bash
nvidia-smi topo -m
nvidia-smi topo -p2p p
NCCL_DEBUG=INFO
```

### 1.5 跨机多卡：Ray 模式

官方推荐流程是先让各节点加入一个 Ray cluster，再在其中一个节点运行一次 `vllm serve`。[vLLM 多节点部署文档](https://docs.vllm.ai/en/stable/serving/parallelism_scaling/) 和 [官方 multi-node-serving 示例](https://docs.vllm.ai/en/stable/examples/ray_serving/multi-node-serving/)

Ray 模式的部署语义如下：

- 集群只需执行一次 `vllm serve` 并提供一个对外 API endpoint；
- 整个集群并非只有一个 OS process，Ray 会在各节点创建多个 vLLM GPU worker actor/process；
- worker 节点不需要再启动独立 HTTP API，但必须先启动 Ray runtime，具备相同 vLLM/PyTorch/模型环境和可访问的模型路径。

典型两节点启动方式：

```bash
# node1：启动 Ray head
ray start --head --node-ip-address=<NODE1_IP> --port=6379

# node2：加入集群；该进程需要持续运行
ray start --address=<NODE1_IP>:6379 --node-ip-address=<NODE2_IP> --block

# Ray 集群 ready 后，仅执行一次服务命令
vllm serve /models/large-model \
  --tensor-parallel-size 8 \
  --pipeline-parallel-size 2 \
  --distributed-executor-backend ray
```

实际生产更推荐使用官方 `run_cluster.sh` 容器脚本或 KubeRay。每个节点需要唯一的 `VLLM_HOST_IP`，节点间网络必须互通；官方明确警告集群内部流量未加密，应只放在可信私网。

#### worker 节点上的“基础进程”包括什么

从职责看，worker 节点至少包含：

1. Ray node runtime：负责节点注册、资源上报、actor 生命周期和进程拉起；
2. Ray worker/actor wrapper：接收 driver 的远程调用；
3. vLLM GPU Worker：初始化 process group，绑定 local GPU；
4. ModelRunner：加载本 rank 的模型层并执行 forward；
5. 本 rank 的 KV cache 与 CUDA context；
6. NCCL communicator 与 CPU metadata group。

它不包含第二套对外 OpenAI API server。模型、tokenizer、Python package 和环境变量必须在各节点一致；共享存储不是强制，但模型必须能以相同配置被每个节点读取。

Ray executor 的源码会为各 actor 分配 `rank`、`local_rank` 和同一个 `distributed_init_method`，再统一调用 `init_worker`、`init_device`、`load_model`：[ray_executor.py（v0.27.1）](https://github.com/vllm-project/vllm/blob/v0.27.1/vllm/v1/executor/ray_executor.py)。`pp_tp_workers` 只是逻辑 rank 编组，不等于数据面 transport。

#### 16 worker 完整示例

假设 node1 与 node2 各 8 张 GPU，配置 `TP=8, PP=2`，并确保 placement 把一个 TP group 放在一台机器内：

| 节点 | global rank | PP rank | TP rank | 职责 |
|---|---:|---:|---:|---|
| node1 | worker0–7 | 0 | 0–7 | embedding + 前半模型层 |
| node2 | worker8–15 | 1 | 0–7 | 后半模型层 + norm/lm head |

逻辑编组为：

```text
PP0 TP group = [worker0, worker1, ..., worker7]
PP1 TP group = [worker8, worker9, ..., worker15]

PP 边：worker0 -> worker8
       worker1 -> worker9
       ...
       worker7 -> worker15
```

在 `TP>1` 时，vLLM 的 tensor-dict 路径可让源 TP rank 各发送一个 slice，然后在目标 TP group 内 all-gather 重建完整 tensor；具体 tensor 是否启用该优化由 shape 与 `all_gather_tensors` 决定。网络上同时存在节点内 TP collective 和节点间 PP P2P。官方推荐“TP size = 每节点 GPU 数、PP size = 节点数”，目的正是让高通信量 TP 尽量留在节点内，而只让 PP activation 跨机。

rank 到节点的映射必须用 Ray placement group、日志和 `ray list actors` 实际核对，不能只按表格想当然。如果启用 Ray Compiled Graph，channel 可为 `auto`、`nccl` 或 `shm`，其 activation 路径与普通 `GroupCoordinator` 路径不同，应单独记录。

### 1.6 跨机多卡：multiprocessing/headless 模式

当前官方文档还提供不使用 Ray 的多机 multiprocessing 方案。此时不是只运行一次命令，而是每个节点都执行 `vllm serve`：

```bash
# node1
vllm serve /models/large-model \
  --tensor-parallel-size 8 --pipeline-parallel-size 2 \
  --nnodes 2 --node-rank 0 \
  --master-addr <NODE1_IP>

# node2
vllm serve /models/large-model \
  --tensor-parallel-size 8 --pipeline-parallel-size 2 \
  --nnodes 2 --node-rank 1 \
  --master-addr <NODE1_IP> --headless
```

node2 的 headless 进程参与 Engine/worker 初始化与执行，但不暴露 API。两边必须使用相同模型、TP、PP、node count、master address/port 和软件环境。该模式的 worker 管理由每个节点本地 MultiprocExecutor 完成，节点间通过共同 distributed init 组成全局 rank。

## 2. 通信后端、协议与物理链路

### 2.1 先纠正术语层级

“NVLink、PCIe、RDMA、IB、TCP、NCCL”不能当成同一级选项：

```text
应用：vLLM IntermediateTensors
  ↓
通信 API/库：torch.distributed / NCCL / RCCL / Gloo
  ↓
传输机制：P2P DMA / shared memory / RDMA / sockets
  ↓
物理链路：NVLink / PCIe / InfiniBand / Ethernet
```

“IBM”如果是指厂商 IBM，它不是这里的 vLLM/NCCL 通信后端；如果原意是“IB”，则是 InfiniBand。

### 2.2 对照表

| 名称 | 所在层 | 单机 | 跨机 | 常见实现与特点 |
|---|---|---|---|---|
| NVLink | GPU 互连 | 是 | 普通服务器否；专用 MNNVL 系统例外 | 高带宽、低延迟，NCCL 自动利用；消费级 RTX 4090 无 NVLink |
| PCIe | 本机 I/O 总线 | 是 | 否 | GPU P2P 可用时直接 DMA；否则经 host memory；受 PCIe switch/root complex/IOMMU/ACS 影响 |
| Shared Memory | 本机 host transport | 是 | 否 | P2P 不可用时常见 fallback；需要 D2H/H2D，CPU/NUMA 带宽参与 |
| InfiniBand（IB） | 跨机 fabric | 本机 NIC 也可见 | 是 | 使用 IB Verbs；可承载 RDMA，配合 GPUDirect RDMA 可由 NIC 直接访问 GPU memory |
| RoCE | Ethernet 上的 RDMA | 本机 NIC 也可见 | 是 | RDMA over Converged Ethernet；需要无损/拥塞控制等网络配置 |
| RDMA | 远程内存访问机制 | 可用于设备/NIC P2P | 是 | 绕过传统 socket 数据拷贝；它不是一种线缆，可运行在 IB 或 RoCE 上 |
| TCP/IP sockets | 网络传输机制 | 是 | 是 | 通用、易调试，无 RDMA 依赖；通常经过内核协议栈与 host memory，延迟/CPU 开销较高 |
| NCCL | NVIDIA 通信库 | 是 | 是 | 为 NVIDIA GPU 提供 collective/P2P；自动选择 NVLink/PCIe/SHM/IB/socket 等 transport |
| RCCL | AMD 通信库 | 是 | 是 | NCCL-compatible API；AMD GPU 上利用 PCIe/xGMI，跨机可用 IB、RoCE、TCP/IP |
| Gloo | CPU ProcessGroup | 是 | 是 | 常用于 metadata、bootstrap、CPU tensor；不是 CUDA activation 的首选高性能路径 |
| Ray | worker 管理/执行框架 | 是 | 是 | 管理 actor、placement、RPC；默认情况下不等于 PP tensor 的物理通信协议 |

AMD 官方说明 RCCL 支持 GPU P2P send/recv，节点内利用 PCIe/xGMI，节点间支持 InfiniBand、RoCE 与 TCP/IP。[AMD RCCL 文档](https://rocm.docs.amd.com/_/downloads/rccl/en/docs-6.4.2/pdf/)

### 2.3 PCIe P2P 与 host staging

两张 GPU 都插在 PCIe 插槽中，并不代表它们一定能直接访问彼此显存。需要 GPU、驱动、主板拓扑、PCIe root complex、IOMMU/ACS 与虚拟化配置共同支持。NVIDIA 建议用 `nvidia-smi topo -p2p p` 验证；NCCL 只会在 CUDA 报告 peer access 可用时优先使用 GPU P2P。[NCCL GPU troubleshooting](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/troubleshooting/gpu_troubleshooting.html)

P2P 不可用时，可靠的公共路径是：

```text
GPU A HBM -> pinned host buffer -> IPC/network -> pinned host buffer -> GPU B HBM
```

这正是 HXinfer 当前验证的路径。它牺牲零拷贝性能，换取跨厂商、跨进程和跨 VM 可迁移性。

### 2.4 RDMA、IB、RoCE 与 GPUDirect RDMA

RDMA 的含义是 NIC 能直接 DMA 到已注册内存，减少 CPU 和内核协议栈参与。IB 和 RoCE 是其常见承载网络。若再启用 GPUDirect RDMA，NIC 可直接访问支持的 GPU memory，进一步去掉 host bounce buffer。NVIDIA 官方要求 GPU/NIC 拓扑、DMA-BUF 或 `nvidia-peermem` 等条件，并建议 GPU 与 NIC 位于合适的 PCIe root complex。[NVIDIA GPUDirect RDMA](https://docs.nvidia.com/datacenter/cloud-native/gpu-operator/latest/gpu-operator-rdma.html)

AMD ROCm 也可通过 PeerDirect 让 RDMA-capable NIC 直接访问 AMD GPU memory。[ROCm GPU-aware MPI](https://rocm.docs.amd.com/en/docs-6.2.4/how-to/gpu-enabled-mpi.html)

但“双方各自支持 GPU Direct/RDMA”不自动推出“NVIDIA HBM 可以与 AMD HBM 跨厂商直连”。内存注册、peer-memory module、runtime 和通信库必须形成同一条受支持的数据路径。对 HXinfer 的第一版，跨厂商 host staging 应被视为设计基线，跨厂商 direct DMA 只能作为目标硬件上的后续优化实验。

### 2.5 TCP 的定位

TCP 支持跨机、跨容器和跨 VM，是功能验证最稳妥的共同分母。它可以验证消息协议、顺序、shape/dtype、数值正确性、取消和故障处理，但不能代表 RDMA 性能。正式实现应支持：

- 固定长度 frame header；
- payload length 上限；
- request/step/sequence；
- model/layout fingerprint；
- checksum；
- backpressure、timeout、断线与重连语义；
- TLS 或可信隔离网络。

## 3. HXinfer 的实现思路与当前边界

### 3.1 双独立 vLLM Server 方案评估

“vLLM 双进程”有两种含义，必须区分：

| 含义 | HXinfer 是否采用 | 原因 |
|---|---|---|
| 两个独立 vLLM Worker OS process | 是 | 每个 stage 各有模型分片、GPU context、ModelRunner 和 KV cache |
| 两套完整 `vllm serve` / 两个 Engine | 否 | 两套 scheduler 和请求状态彼此独立，默认不会把同一个请求按前后层协作执行 |

如果直接启动两个完整 server，每个 server 都认为自己负责一次完整推理。Server A 不会自动把本轮的 `IntermediateTensors`、KV block 生命周期和采样状态交给 Server B；仅在两个 HTTP 接口之间转发文本也不是 Pipeline Parallel。要把两套 server 改造成 PP，仍需额外实现统一调度、step 对齐、activation 协议、KV cache 生命周期和故障处理，相当于绕开后再重写 vLLM 已有 EngineCore 能力。

因此 HXinfer 选择一个统一 EngineCore 和两个独立 Worker：

- EngineCore 保留 vLLM scheduler、请求状态、tokenizer 和采样控制；
- 每个 Worker 只加载并运行自己的 PP stage；
- Controller 通过 TCP RPC 下发相同 scheduler step；
- stage0 通过独立 TCP activation channel 把中间 tensor 交给 stage1；
- 目标服务形态只需要一个统一 API 入口，两个 Worker 不暴露 HTTP API；当前原型使用离线 `LLM` 入口验证，在线 API 封装仍待完成。

这不是为了减少进程数量。本次运行至少包含 Controller/EngineCore 与两个预先启动的 vLLM Worker 进程；选择的核心是“一个推理控制面、两个独立执行面”，而不是“两套互不相干的推理控制面”。

vLLM 原生 Ray 或 multi-node headless PP 对同构集群很合适，但它们通常假设各 worker 能加入兼容的 PyTorch/vLLM distributed 环境，GPU 数据面使用同一种 NCCL-compatible backend。目标场景一侧是 CUDA/NCCL、另一侧是 ROCm/RCCL，并可能处于两个 VM；所以本项目把 GPU activation 数据面降到双方都能实现的 TCP + host staging，同时尽量保留原生 EngineCore 和 Worker 接口。

### 3.2 目标

HXinfer 采用“先验证、再最小改造”的路线：保留 vLLM 的模型切层、weight loading、scheduler、KV cache、ModelRunner 与 sampling，只替换 PP boundary transport；不 fork vLLM，不重新实现一份 Hugging Face 切层模型。

```text
一个 vLLM LLM / EngineCore / scheduler
       │ TCP control RPC
       ├──────────────────────┐
       ▼                      ▼
独立 Worker Service 0      独立 Worker Service 1
PP0 / GPU0                PP1 / GPU1
layers 0..K-1             layers K..N-1 + sampling
       │                      ▲
       └── TCP activation ────┘
           pinned host staging
```

两个 worker 是在 Controller 之前分别启动的独立 OS process，不是 `MultiprocExecutor` 创建的子进程。它们各自拥有 vLLM `Worker`、ModelRunner、GPU context、模型分片和 KV cache，但由同一个 EngineCore 调度；它们不是两个完整 vLLM server，也不暴露两个 API。

### 3.3 Controller 与最小注入点

`external_pp.remote.executor.HXinferRemoteExecutor` 实现 vLLM `Executor`，连接环境变量 `HXINFER_WORKER_ENDPOINTS` 指定的两个 worker。初始化、KV cache profile/config、warm-up、每一步 `execute_model` / `sample_tokens` 和退出都通过持久 TCP RPC 广播，最后一个 PP rank 返回采样结果。

`external_pp.remote.worker.HXinferRemoteStageWorker` 继承 vLLM `Worker`。执行原生 `init_device()` 后，它获得 `get_pp_group()`，只替换：

- `pp_group.isend_tensor_dict`：调用 HXinfer `transport.send()`；
- `pp_group.irecv_tensor_dict`：调用 HXinfer `transport.recv(device)`。

这意味着模型层、KV cache、scheduler 和输出采样均仍由 vLLM 执行。两个 worker 的公共 process group 使用 CPU/Gloo，PP group 禁止创建 device communicator；实验日志没有 NCCL communicator，PP activation 不经过 NCCL/RCCL。

### 3.4 当前 transport

`TcpSocketPPTransport` 是 correctness-first 实现：

1. 接收完整 `dict[str, Tensor|Any]`；
2. CUDA tensor 先 `detach().contiguous()`，匹配 Native receiver 按 shape 新建 contiguous tensor 的语义；
3. 分配 pinned CPU tensor并执行 D2H；
4. 同步当前 CUDA stream；
5. `torch.save` 序列化；
6. 通过带 8-byte network-order length header 的持久 TCP socket 发送；
7. 对端反序列化，经 pinned tensor H2D；
8. 记录 sequence、bytes、keys、D2H/socket/H2D/total timing。

当前限制：

- 仅 PP=2、TP=1；
- activation 仅实现 stage0 → stage1；
- 同步传输，没有 compute/communication overlap；
- steady state 仍分配 host tensor与序列化 buffer；
- control RPC 使用 pickle/cloudpickle，activation 使用 `torch.save`，均只适合可信隔离网络；
- 两个 worker 仍需要同版本 vLLM，并暂时通过 CPU/Gloo 建立公共 rank；CUDA/ROCm 两种 PyTorch build 的 Gloo wire compatibility 尚未实测。

### 3.5 为什么这个原型仍有价值

它已经验证四件关键事情：

1. vLLM 的 PP model partition 和连续 decode 可以保留；
2. 完整 `IntermediateTensors` mapping 可以脱离原生 NCCL PP 数据面传输；
3. 一个 EngineCore 可以管理两个独立启动的远程 vLLM worker；
4. transport 可以作为窄接口替换，而不是重构 vLLM scheduler 和模型实现。

这使后续 AMD/VM 迁移主要集中在 ROCm 单 stage 兼容性、跨 build 控制 ABI 和 transport 优化，而不是重新解耦 worker 生命周期或推倒重写推理栈。完整实现与本次新实验见 `docs/remote_worker_validation_report.md`。

## 4. 双 RTX 4090 实验报告

### 4.1 环境

| 项目 | 实测值 |
|---|---|
| OS | Ubuntu 22.04.5 LTS，Linux 5.15.0-97-generic |
| CPU / RAM | 32 cores / 240 GB |
| GPU | 2 × NVIDIA GeForce RTX 4090，24,564 MiB |
| Driver | 580.76.05 |
| CUDA | 13.0 |
| Python | 3.12.3 |
| PyTorch | 2.13.0+cu130 |
| vLLM | 0.27.1 |
| Transformers | 5.15.1 |
| NCCL | vLLM 日志报告 2.29.7 |

`nvidia-smi topo -m` 中 GPU0↔GPU1 为 `NODE`：路径经过 PCIe 和同一 NUMA node 内不同 PCIe host bridge；不存在 NVLink。`nvidia-smi topo -p2p p` 两个方向均为 `NS`，说明 PCIe P2P 不支持。两卡均位于 NUMA node 1。测试结束后两卡显存为 0 MiB，未发现残留 compute process。

### 4.2 模型和参数

| 项目 | 小模型 smoke | 大模型验证 |
|---|---|---|
| 模型 | Qwen3-0.6B | Qwen2.5-3B-Instruct |
| dtype | FP16 | BF16 |
| parallel | PP=2, TP=1 | PP=2, TP=1 |
| worker | 两个独立 OS worker process | 两个独立 OS worker process |
| decoding | greedy，temperature=0 | greedy，temperature=0 |
| correctness backend | eager | eager + TRITON_ATTN |

严格对照还设置 `PYTHONHASHSEED=0`、`CUBLAS_WORKSPACE_CONFIG=:4096:8`，并关闭 FlashInfer sampler。长 token 同样是 greedy，不存在温度采样随机性。

Qwen3 每个 stage 加载约 0.71 GiB；Qwen2.5-3B 两个 worker 各报告加载 3.19 GiB，而非各加载完整模型，确认实际执行了 PP layer partition。

### 4.3 总结果

| 验证项 | 结果 | 解释 |
|---|---|---|
| Native PP=2 初始化与生成 | PASS | 两个 PP worker 正常运行 |
| External PP activation 通路 | PASS | trace 中 send/recv sequence 对齐 |
| 独立 worker service + 单 Controller | PASS | 两个预启动 worker 经 TCP RPC 完成完整 Engine 生命周期 |
| Qwen3-0.6B 16-token smoke | PASS | Native/External 16/16 token IDs 相同 |
| Qwen2.5-3B 8 道简单题 | PASS | 两边 8/8 正确，8/8 token 序列相同 |
| Remote worker 690-token 连续 decode | PASS（功能） | 261 对 TCP activation，无丢消息和死锁 |
| External 512-token 连续 decode | PASS（功能） | 516 对消息，无丢消息和死锁 |
| 3B 长输出严格 token 等价 | FAIL | Native/External 6/6 输出发生确定性分叉 |
| 性能排名 | 未放行 | correctness 未全过，测量也不是正式 benchmark |

### 4.4 简单问答

| 题目摘要 | 期望 | Native | External | token 一致 |
|---|---|---|---|---|
| 2 + 3 | 5 | 5 | 5 | 是 |
| 中国首都 | 北京 | 北京 | 北京 | 是 |
| 12 - 5 | 7 | 7 | 7 | 是 |
| apple 中文 | 苹果 | 苹果 | 苹果 | 是 |
| `len("hello")` | 5 | 5 | 5 | 是 |
| 2, 4, 6, 8 的下一项 | 10 | 10 | 10 | 是 |
| 水的冰点 | 0 | 0 | 0 | 是 |
| 地球绕哪颗恒星 | 太阳 | 太阳 | 太阳 | 是 |

Native engine load 15.781 s、8 题批量生成 0.155 s；External load 15.839 s、生成 0.200 s。External 产生 13 次 send 和 13 次 recv，累计 payload 8,240,353 bytes（7.859 MiB），单条最大 4,196,133 bytes。sender 累计 D2H、socket send 和 total 分别为 42.162 ms、17.473 ms 和 72.327 ms。

这些值只证明功能和可观测性。receiver `total` 包含等待上游计算的阻塞，不能当作纯通信延迟；单次运行也不能用于性能排名。

### 4.5 正常问答和长 decode

6 问覆盖 PP 概念、火车算术、Python、无 NVLink 通信、KV cache 与英文总结。External 512-token 运行完成 516 次 send/recv，累计约 24.35 MiB payload，最大序列化 payload 4,196,133 bytes；没有消息丢失或 decode 死锁。算术题正确得到 200 米，Python 题给出 O(n) 时间/空间方案。模型对部分系统概念的解释不严谨属于 3B 模型能力，不是 PP 执行失败，原始输出未人工改写。

### 4.6 长输出为什么仍判 FAIL

最初 FlashAttention Native 跨重启本身不能逐 token 复现，因此严格 A/B 改用 Triton Attention。之后得到：

- Native repeat：6/6 token 序列一致；
- External repeat：6/6 token 序列一致；
- Native vs External：6/6 最终 token 序列不一致。

保留的 256-token 对照中，6 题首个分叉的零起始索引分别为 60、45、60、40、218、34，即自然计数第 61、46、61、41、219、35 个生成 token。由于是 `temperature=0` greedy 且两边各自可重复，分叉不能归因于随机采样。

greedy 只保证每一步选当前最大 logit，并不保证不同计算/通信路径位级相同。微小数值差异如果发生在 top-1/top-2 很接近的位置，可能改变 argmax，并在自回归生成中持续放大。

External 已把 boundary tensor 规范化为 contiguous，但未消除差异。下一步需要判断差异在 stage0 boundary 产生前、D2H/序列化/H2D 过程中，还是 stage1 某一层或 logits 中。

### 4.7 现有计时及不可得出的结论

| 路径 | engine load | 6 问 generation | stage 消息 |
|---|---:|---:|---:|
| Native Triton PP，256 max tokens | 19.118 s | 7.418 s | Native NCCL |
| External socket PP，256 max tokens | 15.714 s | 4.448 s | 261 对 |
| External socket PP，512 max tokens | 16.516 s | 8.635 s | 516 对 |

不能据此得出 External 更快：两边输出序列不同，初始化/JIT/cache 状态不同，只运行了少量次数，当前 timer 还混入 CUDA stream 上游工作。保留实验没有形成标准在线 serving 的 TTFT、TPOT、request throughput 和多并发稳态统计，这是后续 benchmark 的明确缺口。

### 4.8 自动化验证

- 项目单元测试：7 passed，3 skipped（本地 sandbox 禁止部分 socket 操作）；
- Python compileall：PASS；
- 简单题结果比较器：PASS（8/8）；
- 项目凭据扫描：PASS；
- 服务器结束状态：无 GPU compute process。

### 4.9 独立远程 worker 实验（本轮新增）

本轮使用 `HXinferRemoteExecutor` 和两个提前独立启动的 `worker_service`，不使用 vLLM `MultiprocExecutor`。两个 worker 各加载约 3.19 GiB 权重，通过 CPU/Gloo 取得 PP0/PP1 rank，activation 数据面使用独立 TCP channel。

| workload | load | generation | send/recv | payload | 结果 |
|---|---:|---:|---:|---:|---|
| 8 道简单题 | 16.054 s | 0.223 s | 8/8 | 8,165,672 B | 8/8 正确，Native token exact |
| 6 个正常问题，128-token cap | 14.680 s | 6.089 s | 261/261 | 14,395,577 B | 生成 690 token，无死锁 |

长回答发送端累计 D2H 31.634 ms、socket `sendall` 12.807 ms，接收端累计 H2D 64.648 ms。receiver 的 socket/total 包含等待上游计算，不能解释为纯通信耗时。当前两张 4090 无 NVLink且 P2P 为 `NS`，所以 D2H/H2D 两段明确经过 PCIe；TCP loopback 负责主机内存间交付，不等于 GPU-direct PCIe P2P。

完整 RPC 边界、复现方式、trace 与 CUDA/ROCm 迁移限制见 `docs/remote_worker_validation_report.md`，原始结果在 `results/remote_workers/`。

## 5. NVIDIA + AMD 同机 PCIe 与双 VM 迁移方案

### 5.1 目标硬件的关键现实

计划形态为一张 RTX 4090/CUDA 与一张 AMD GPU/ROCm，同机通过 PCIe。即使两卡处于相同 PCIe fabric，也不能预设 CUDA 与 ROCm 支持跨厂商显存 P2P，更不能让一个普通 NCCL communicator 同时管理 CUDA 与 ROCm device。

还需要在目标机实测 CUDA/ROCm 驱动、IOMMU group、内核模块、设备权限与重启稳定性。现有证据不足以确认或否定裸机驱动共存，因此架构兼容三种部署：

```text
A. 裸机双进程
B. 双容器，共享同一宿主机内核
C. 双 VM，各自 GPU passthrough
```

### 5.2 推荐目标架构

第一版不依赖混合 NCCL/RCCL GPU ProcessGroup。当前原型已经采用两个独立 Stage Service，目标机继续沿用这一结构：

```text
Controller / scheduler / tokenizer
        │ request_id, step_id, token ids, lifecycle
        ├─────────────────────────────────────┐
        ▼                                     ▼
CUDA Stage Service                         ROCm Stage Service
vLLM-CUDA environment                      vLLM-ROCm environment
RTX 4090                                   AMD GPU
layers 0..K-1                              layers K..N-1
stage-local KV cache                       stage-local KV cache
        │                                     ▲
        └──── versioned Intermediate ABI ─────┘
                   PPTransport
```

两个 runtime 的进程和驱动边界允许使用不同 Python environment、PyTorch wheel、vLLM platform plugin 和驱动用户态库。Controller 只管理请求顺序、错误、取消和最终输出，不直接持有厂商 GPU tensor。当前 wire payload 仍包含 vLLM/Python pickle 对象，所以“进程可以隔离”不等于“任意版本已经兼容”；第一轮目标机实验仍应固定相同 vLLM 版本。

### 5.3 可直接复用与需要改造

| HXinfer 部分 | 裸机双进程 | 双容器 | 双 VM | 处理方式 |
|---|---|---|---|---|
| Intermediate tensor mapping | 复用 | 复用 | 复用 | 升级成显式版本化 ABI |
| shape/dtype/keys/sequence | 复用 | 复用 | 复用 | 增加校验、request/step id |
| framing 思路 | 复用 | 复用 | 复用 | 禁用跨信任边界 `torch.load` |
| TCP activation transport | 复用 | 复用 | 复用 | 当前已实现；不同部署只改地址与网络配置 |
| CUDA pinned staging | 仅 NVIDIA 侧 | 仅 NVIDIA 侧 | 仅 NVIDIA VM | AMD 侧改 HIP/ROCm pinned staging |
| 独立 worker service 生命周期 | 复用 | 复用 | 复用 | 双 4090 已验证；ROCm 单 stage 待验证 |
| CPU/Gloo ProcessGroup | 暂时复用 | 暂时复用 | 可跨 TCP | 先测 CUDA/ROCm build 兼容性；必要时改虚拟 PP rank |
| Controller TCP RPC | 复用 | 复用 | 复用 | pickle 仅限可信原型，生产需显式 ABI |
| trace schema | 复用 | 复用 | 复用 | 补 compute、queue、checksum timing |

### 5.4 裸机和双容器路径

第一阶段始终走显式 host staging：

```text
RTX HBM
  -> cudaMemcpy D2H / CUDA pinned buffer
  -> 当前 TCP；后续可优化为 fixed-slot shared memory ring
  -> HIP pinned buffer / hipMemcpy H2D
  -> AMD HBM
```

CUDA 分配的 pinned page 不预设为可被 ROCm 直接复用。基础实现由 transport 管理普通 POSIX shared-memory slots，两侧各自维护厂商专属 pinned staging pool；只有在目标机验证同一 shared page 可被 `cudaHostRegister` 与 `hipHostRegister` 安全注册后，才评估减少一层 CPU copy。

容器共享宿主机内核，不能解决驱动内核模块冲突；它只能隔离用户态库、Python wheel 和依赖。如果 CUDA/ROCm 内核驱动可以共存但用户态环境冲突，双容器很合适。如果内核驱动或设备隔离本身冲突，才需要双 VM。

### 5.5 双 VM 路径是否可迁移

可以迁移，但只能承诺协议与逻辑等价，不能承诺性能等价。

双 VM 中建议每个 VM passthrough 一张 GPU：CUDA VM 运行 stage0，ROCm VM 运行 stage1。当前 HXinfer 已经使用 TCP，因此协议层不需要再从 Unix-domain socket 改造；只需把本机 loopback 地址换成两个 VM 的可达地址。普通 POSIX shared memory 不能跨 VM：

```text
NVIDIA HBM
 -> CUDA VM pinned memory
 -> guest TCP stack / virtio-net
 -> host vSwitch
 -> ROCm VM TCP stack
 -> ROCm VM pinned memory
 -> AMD HBM
```

#### 双 VM Worker 启动能力

当前实现不依赖父子进程关系、Unix-domain socket 或共享文件描述符，两个 Worker 可以在不同 VM 中分别启动：

- `worker_service --listen HOST:PORT` 可绑定任意可达的 TCP control endpoint；
- `HXinferRemoteExecutor` 从 `HXINFER_WORKER_ENDPOINTS` 连接两个不同 IP；
- PP rank0 的 activation transport 监听 TCP，rank1 连接 rank0 的 VM 地址；
- CPU/Gloo 使用 `HXINFER_DIST_INIT=tcp://VM0_IP:PORT` 建立公共 rank；
- 两侧分别通过 `CUDA_VISIBLE_DEVICES`、`HIP_VISIBLE_DEVICES` 或 `ROCR_VISIBLE_DEVICES` 选择设备。

参考拓扑为 VM0 `10.0.0.10`、VM1 `10.0.0.11`：

```bash
# VM0：NVIDIA/CUDA，PP rank 0
export VLLM_ENV=/opt/hxinfer-cuda
export HXINFER_ACTIVATION_HOST=0.0.0.0
export HXINFER_ACTIVATION_PORT=29620
bash scripts/start_remote_worker.sh 0 0.0.0.0:29600 0

# VM1：AMD/ROCm，PP rank 1
export VLLM_ENV=/opt/hxinfer-rocm
export HXINFER_VISIBLE_DEVICE_ENV=ROCR_VISIBLE_DEVICES
export HXINFER_ACTIVATION_HOST=10.0.0.10
export HXINFER_ACTIVATION_PORT=29620
bash scripts/start_remote_worker.sh 1 0.0.0.0:29600 0

# Controller：第一阶段可运行在 VM0
export HXINFER_CONFIG=/path/to/experiment.env
export HXINFER_WORKER_ENDPOINTS=10.0.0.10:29600,10.0.0.11:29600
export HXINFER_DIST_INIT=tcp://10.0.0.10:29610
bash scripts/run_remote_pp_controller.sh
```

需要放通的 TCP 端口为：

| 地址 | 端口 | 用途 |
|---|---:|---|
| VM0 | 29600 | Controller → rank0 control RPC |
| VM1 | 29600 | Controller → rank1 control RPC |
| VM0 | 29610 | 两个 Worker 的 CPU/Gloo rendezvous |
| VM0 | 29620 | rank0 → rank1 PP activation channel 的监听端 |

当前 Controller 发送同一个序列化 `vllm_config`，因此两侧需要相同的 vLLM 版本，模型目录也需要使用相同路径；不同本地路径映射尚未实现。控制 RPC 使用 pickle，网络必须为可信隔离网络。当前验证入口为离线 `LLM`/问答脚本，统一 OpenAI-compatible 在线 API Server 的封装和验证尚未完成。

这会比裸机共享内存多出虚拟网络和额外 copy。后续可评估 vhost-net、virtio-vsock 或 ivshmem/共享内存设备，但 ivshmem 页的锁页、IOMMU 映射、两侧驱动注册和故障隔离更复杂，不能作为第一阶段前提。

VM 验证必须额外记录：

- IOMMU group 与 GPU passthrough 是否完整；
- BAR/Resizable BAR 配置；
- VM vCPU/内存 NUMA 绑定；
- pinned/locked memory limit；
- virtio/vhost、MTU 与实际 TCP 吞吐；
- 两 VM 时钟同步；
- D2H、guest transport、H2D 分段延迟。

### 5.6 “能否迁移”的证据边界

当前证据支持“架构与单机异构硬件均已验证，可继续迁移双 VM”；尚不支持“CUDA+ROCm 双 VM 已经跑通”。

| 能力 | 当前状态 | 到双 VM 是否需要重写 |
|---|---|---|
| 一个 EngineCore 管理两个独立 Worker | 双 4090、CUDA+ROCm 均已验证 | 不需要 |
| Worker 独立启动、各持模型分片和 KV cache | 双 4090、CUDA+ROCm 均已验证 | 不需要 |
| PP activation 脱离 NCCL，经 host staging + TCP | 双 4090、CUDA+ROCm 均已验证 | 不需要，只改地址 |
| PCIe D2H/H2D | NVIDIA CUDA 与 AMD HIP 均已验证 | 不需要 |
| 跨 VM TCP/virtio/vSwitch | 尚未验证 | 协议复用，性能和网络配置需实测 |
| vLLM ROCm 模型与算子支持 | BF16 完整通过；gfx1100 block-FP8 kernel 阻塞 | 按目标模型/dtype继续验证 |
| CUDA/ROCm PyTorch build 的 CPU/Gloo 互通 | all-reduce 与完整 PP 均已验证 | 不需要 |
| pickle/`torch.save` 跨版本兼容与安全性 | 仅同版本可信环境验证 | 生产前升级为显式版本化 ABI |
| 异构负载均衡 | 尚未验证 | 按两卡实测逐层耗时做非均匀切分 |

因此，当前成果已经消除了双 VM 迁移中最大的结构性风险：不再要求两个 GPU 由同一 CUDA/ROCm runtime 或同一个 GPU communicator 管理。剩余风险集中在具体模型的 AMD 算子覆盖、协议生产化和虚拟网络性能，均可以在不推翻 Controller/Worker 架构的情况下逐项验证。

### 5.7 从当前原型迁移的分阶段 Gate

#### Gate A：关闭当前 3B correctness 缺口

1. 固定相同 token prefix；
2. Native sender 与 External sender 同时记录每个 boundary tensor 的 shape、dtype、stride、storage offset、有限统计量和 hash；
3. 若 boundary 一致，逐层比较 stage1 hidden state 与 logits；
4. 建立单-forward logits 的 `max_abs_error`、`mean_abs_error`、`relative_l2` Gate；
5. 保留短答案 token exact 与长生成稳定性两套回归。

在 Gate A 通过前，不开始 transport 性能排名。

#### Gate B：把两个 worker 解耦成独立 Stage Service（CUDA/ROCm 功能 Gate 已通过）

同版本 NVIDIA runtime 已完成：

- 独立 stage 初始化和 layer-range 配置；
- 权重只加载本 stage；
- stage-local KV block ownership/lifecycle；
- last stage logits/sampling 结果回传；
- scheduler output 跨进程传递。

跨 CUDA/ROCm wheel 的 schema/Gloo 兼容性已在 vLLM 0.27.1 两套官方镜像间验证。尚未完成的是显式 request/step/token wire schema、cancellation、timeout、重启、版本握手和生产安全协议，所以 Gate B 对当前可信环境标记 PASS，生产协议仍待完成。

#### Gate C：同机 NVIDIA→NVIDIA 固定 slot transport

把 `torch.save` 和逐消息分配替换为预分配 pinned staging pools、固定 slot shared-memory ring、二进制 header + 原始 tensor bytes、producer/consumer sequence 与 backpressure，并实现计算、D2H、handoff、H2D overlap。

先在同构硬件上证明新协议数值正确和稳定，再接 AMD，避免把 transport bug 与厂商差异混在一起。

#### Gate D：AMD 单 stage smoke

在 ROCm 环境单独验证目标模型的 vLLM/ROCm 版本支持、dtype/attention backend/算子覆盖、单 stage weight loading、层级输出误差、HIP pinned copy 带宽和同步语义。

状态：Qwen2.5-3B BF16 已通过 GPU smoke、权重加载、kernel warm-up 和持续生成；Qwen3.8-27B block-FP8 已完成权重加载，但 gfx1100 kernel backend 未通过。

#### Gate E：裸机或双容器 CUDA→ROCm

先用 TCP 完成单 forward，再做连续 greedy decode；通过后切 shared memory。层切分不能简单按层数一半，应根据两张卡的显存、实际每层延迟和 boundary payload 做不均匀 partition。

状态：TCP 单题、6 个正常问答和 8 个短题均通过；共享内存后端与非均匀切层尚未完成。

#### Gate F：双 VM

保持相同的 Controller、Stage Service 和当前 TCP transport，把 loopback 地址换成 VM 可达地址。正确性重新全量验证，性能单独建基线；不能把裸机延迟外推到 VM。

#### Gate G：性能与可靠性

固定 workload 至少重复多轮，报告：

```text
TTFT / TPOT / request throughput
stage0 compute / stage1 compute
D2H / queue / transport / H2D
payload bytes and effective bandwidth
P50 / P95 / P99
断线、超时、worker 重启、sequence gap
```

## 6. 最终判断

HXinfer 已经证明“一个 vLLM EngineCore 管理两个独立启动的 vLLM worker，并把 activation 数据面外置为 TCP/host staging”在双 NVIDIA worker 上可行。简单题与短序列给出了强 correctness 证据，690-token 长 decode 给出了稳定性证据；但 3B 长序列的确定性分叉意味着当前实现仍处在验证原型阶段。

RTX 4090 + AMD RX 7900 XTX 的单机双容器实验已完成。独立 worker 生命周期、Controller executor、TCP transport、tensor boundary、trace 和测试 Gate 均得到复用；CPU/Gloo ProcessGroup 已在 CUDA/ROCm 两套 PyTorch build 间通过 all-reduce 和完整 PP 推理验证。双 VM 尚未实测，但当前协议已经使用 TCP，不再需要从 Unix socket 改造为 TCP，主要新增项是 VM 地址、端口、安全策略和虚拟网络性能验证。

当前正确的优先级是：把 pickle/`torch.save` 协议升级为显式 ABI 和固定 slot shared pinned-memory transport，随后做双 VM 正确性、性能和可靠性验证。Qwen2.5-3B BF16 已在异构环境完成 8/8 短题和 6 个正常问答；Qwen3.8-27B block-FP8 的剩余阻塞是 gfx1100 Triton kernel 编译与 AITER 架构覆盖，不是 HXinfer 控制面或 activation 通信。第一版无需混合 NCCL/RCCL，也不依赖跨厂商显存 P2P。

### 6.1 结论适用范围

HXinfer 使用两个独立启动的 vLLM Worker 进程，并由一个统一的 vLLM EngineCore 管理。两套独立 `vllm serve` 各自拥有 scheduler、请求状态和 KV cache，默认不能共同完成同一个 Pipeline Parallel 请求，因而未被采用。当前方案保留 vLLM 的调度、模型切层和 stage-local KV cache，仅将控制调用和 PP activation 改为 TCP；activation 路径为 GPU→PCIe→host memory→TCP→host memory→PCIe→GPU。

该进程与协议边界可映射为 NVIDIA VM 上的 CUDA Worker、AMD VM 上的 ROCm Worker以及统一 Controller。双 4090 实验已经验证结构与功能；RTX 4090 + RX 7900 XTX 单机双容器进一步验证了跨 CUDA/ROCm build 的 Gloo、序列化、activation host staging 和完整生成。双 VM 正确性与性能仍未验证；27B block-FP8 也不适合直接作为 gfx1100 功能基线。

### 6.2 NVIDIA + AMD 单机异构验证更新

2026-08-25 在 RTX 4090 + RX 7900 XTX 上完成 PP=2、TP=1 验证。两个 Worker 分别运行 vLLM 0.27.1 CUDA/ROCm 镜像，由统一 EngineCore 通过自定义 RemoteExecutor 管理。

- Qwen2.5-3B-Instruct 单题输出 `2`，三容器退出码均为 0；
- 6 个正常问答生成 530 tokens，记录 101 条 activation 消息、6,648,217 bytes；
- 8 个自动判分短题通过 8/8；
- 数据路径为 NVIDIA GPU→PCIe D2H→TCP loopback→PCIe H2D→AMD GPU；
- Qwen3.8-27B-FP8 两阶段权重加载成功，但 gfx1100 的 block-FP8 Triton kernel 未在合理时间完成编译，AITER 仅支持 CDNA3 及以上。

完整环境、启动命令、代码改造和实验数据见 `docs/heterogeneous_amd_nvidia_validation_report.md`。

## 7. 项目证据索引

- vLLM 0.27.1 PP 源码分析：`docs/vllm_pp_analysis.md`
- HXinfer 实施 Gate：`docs/implementation_plan.md`
- 双 worker 0.6B 报告：`docs/two_process_pp_report.md`
- 3B 长问答报告：`docs/large_model_qa_report.md`
- 机器执行手册：`docs/machine_runbook.md`
- 双 RTX 4090 环境快照：`docs/dual_4090_environment.md`
- 独立 worker + PCIe 完整验证：`docs/remote_worker_validation_report.md`
- NVIDIA + AMD 单机异构 PP：`docs/heterogeneous_amd_nvidia_validation_report.md`
- 简单题定义：`configs/simple_qa_cases.json`
- Native 简单题结果：`results/native_pp/qwen2.5-3b-simple-qa/`
- External 简单题与 transport trace：`results/external_socket/qwen2.5-3b-simple-qa/`
- Native 3B 长结果：`results/native_pp/qwen2.5-3b-qa256-triton2/`
- External 3B 长结果：`results/external_socket/qwen2.5-3b-qa256-contiguous/`
- Remote worker 简单题结果：`results/remote_workers/diagnostic2/`
- Remote worker 128-token 正常问答：`results/remote_workers/qwen2.5-3b-qa128/`

## 8. 外部参考

- [vLLM Parallelism and Scaling](https://docs.vllm.ai/en/stable/serving/parallelism_scaling/)
- [vLLM official multi-node serving example](https://docs.vllm.ai/en/stable/examples/ray_serving/multi-node-serving/)
- [vLLM 0.27.1 `gpu_worker.py`](https://github.com/vllm-project/vllm/blob/v0.27.1/vllm/v1/worker/gpu_worker.py)
- [vLLM 0.27.1 `parallel_state.py`](https://github.com/vllm-project/vllm/blob/v0.27.1/vllm/distributed/parallel_state.py)
- [vLLM 0.27.1 `ray_executor.py`](https://github.com/vllm-project/vllm/blob/v0.27.1/vllm/v1/executor/ray_executor.py)
- [vLLM 0.27.1 `multiproc_executor.py`](https://github.com/vllm-project/vllm/blob/v0.27.1/vllm/v1/executor/multiproc_executor.py)
- [NVIDIA NCCL documentation](https://docs.nvidia.com/deeplearning/nccl/)
- [NCCL GPU topology troubleshooting](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/troubleshooting/gpu_troubleshooting.html)
- [NVIDIA GPUDirect RDMA](https://docs.nvidia.com/cuda/gpudirect-rdma/)
- [AMD RCCL documentation](https://rocm.docs.amd.com/projects/rccl/en/latest/)
- [ROCm GPU-aware MPI/RDMA](https://rocm.docs.amd.com/en/docs-6.2.4/how-to/gpu-enabled-mpi.html)
