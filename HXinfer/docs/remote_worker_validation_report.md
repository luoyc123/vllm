# HXinfer 独立 vLLM Worker + PCIe 验证报告

> 日期：2026-08-24
> vLLM：0.27.1
> 模型：Qwen2.5-3B-Instruct
> 结论：双独立 worker 的 PP 功能验证通过；CUDA/ROCm 混合设备尚待目标机验证

## 1. 本次验证回答的问题

本次实现的不是两个 OpenAI API server，也不是由 vLLM `MultiprocExecutor`
临时 fork 出来的两个子进程，而是：

```text
一个 vLLM LLM/EngineCore（Controller，唯一调度器）
        │
        ├── TCP control RPC ──> 独立 worker service / PP rank 0 / GPU 0
        │                              │
        └── TCP control RPC ──> 独立 worker service / PP rank 1 / GPU 1
                                       ▲
                    TCP activation ────┘
```

两个 worker service 在 Controller 之前独立启动，分别拥有自己的 Python
进程、CUDA context、vLLM `Worker`、ModelRunner、模型分片和 KV cache。Controller
负责 scheduler、请求状态与最终输出，使用自定义
`HXinferRemoteExecutor` 同步调用两个 worker。

这与未来双 VM 的进程边界相同：每个 VM 只需要启动一个 worker service，只有
Controller 所在节点提供上层服务接口。它们不是两个各自进行完整推理的 vLLM
server，而是一个 vLLM Engine 管理的两个远程 PP stage。

当前 Controller 会把同一个 `vllm_config` 发给两个 worker，因此模型目录在两台
VM 中必须使用相同路径；若路径不同，后续需要在 worker RPC 层增加本地路径映射。

## 2. 控制面与数据面

### 2.1 控制面

`external_pp.remote.executor.HXinferRemoteExecutor` 实现 vLLM `Executor` 接口，
将以下调用通过持久 TCP RPC 同时发给两个 worker：

- `init_worker`、`init_device`、`load_model`；
- KV cache profile、配置与 kernel warm-up；
- 每个 scheduler step 的 `execute_model` 与 `sample_tokens`；
- health check 与 shutdown。

当前控制 RPC 使用 pickle/cloudpickle，只允许部署在可信隔离网络；生产版需要换成
版本化 schema，并加入鉴权、超时、取消、重连和请求幂等语义。

### 2.2 PP activation 数据面

`HXinferRemoteStageWorker` 保留 vLLM 的模型切层和执行逻辑，只替换 PP group 的
`isend_tensor_dict` / `irecv_tensor_dict`。rank 0 发送完整
`IntermediateTensors` mapping；Qwen2.5-3B 的 boundary keys 为
`hidden_states` 和 `residual`。

数据路径为：

```text
GPU0 HBM
 -> contiguous CUDA tensor
 -> CUDA pinned host tensor（D2H，经过 PCIe）
 -> torch.save framing
 -> persistent TCP socket
 -> CPU tensor / pinned host tensor
 -> GPU1 HBM（H2D，经过 PCIe）
```

两个 worker 的 PyTorch distributed backend 被限制为 CPU/Gloo；PP group 不创建
device communicator。日志中没有 NCCL communicator，activation 也不经过 NCCL
P2P。因此，本实验明确验证的是 host-staged PCIe 路径，而不是 NVLink 或 PCIe
GPU-direct P2P。

当前两张 RTX 4090 的 `nvidia-smi topo -m` 为 `NODE`，且
`nvidia-smi topo -p2p p` 两方向为 `NS`；机器不存在 NVLink，也不支持两卡 CUDA
P2P。TCP 使用 loopback 只承载主机内存中的数据，GPU 进出主机内存仍分别经过
两张卡的 PCIe 链路。

## 3. 双 4090 实测

### 3.1 环境

| 项目 | 值 |
|---|---|
| GPU | 2 × NVIDIA GeForce RTX 4090 24 GB |
| OS | Ubuntu 22.04.5 LTS，Linux 5.15.0-97-generic |
| Driver / CUDA | 580.76.05 / 13.0 |
| Python | 3.12.3 |
| PyTorch | 2.13.0+cu130 |
| vLLM | 0.27.1 |
| dtype / attention | BF16 / Triton Attention |
| parallel | PP=2，TP=1，eager，greedy temperature=0 |

两个 worker 各加载约 3.19 GiB 模型权重，分别被识别为 PP rank 0 和 PP rank 1；
两者共享的 distributed rendezvous 后端为 Gloo。

### 3.2 简单问答

8 道题包括算术、常识、翻译和 Python 基础：8/8 答案正确，并与保存的 Native
PP baseline 逐 token 一致。

| 指标 | 结果 |
|---|---:|
| Engine load | 16.054 s |
| 8 题批量生成 | 0.223 s |
| send / recv | 8 / 8 |
| activation payload | 8,165,672 bytes |
| sender D2H 累计 | 4.764 ms |
| receiver H2D 累计 | 9.979 ms |

### 3.3 正常问题与连续 decode

6 个问题覆盖 PP/TP、火车算术、Python 函数、PCIe PP、KV cache 和英文总结，
`max_tokens=128`。运行生成 5×128 + 50 = 690 个 token，完整结束。

| 指标 | 结果 |
|---|---:|
| Engine load | 14.680 s |
| 6 题批量生成 | 6.089 s |
| send / recv | 261 / 261 |
| activation payload | 14,395,577 bytes（13.73 MiB） |
| sender D2H 累计 | 31.634 ms |
| sender socket send 累计 | 12.807 ms |
| receiver H2D 累计 | 64.648 ms |
| 消息丢失或死锁 | 0 |

receiver 的 socket/total 时间包含等待 stage 0 计算，不能当作纯链路延迟；本组数值
也没有控制在线并发、warm cache 和重复次数，不能用于宣称比 Native PP 更快。

长输出与历史 Native PP baseline 仍会在部分题目上发生确定性 token 分叉；本次
首个分叉索引为 9、23、60、40、66，英文题 50 token 完全一致。短答案 exact
correctness 已通过，但长序列 bit/token exact Gate 仍未通过。这不影响“独立 worker
能完成完整 PP 推理”的功能结论，但在正式性能比较前仍需做 boundary/logits 数值定位。

## 4. 代码入口与复现

同机双 GPU 一键运行：

```bash
HXINFER_CONFIG=configs/remote_4090.env \
HXINFER_BASE_PORT=29600 \
bash scripts/run_remote_pp_local.sh
```

双节点或双 VM 分别启动 worker：

```bash
# VM 0 / CUDA
HXINFER_ACTIVATION_HOST=0.0.0.0 HXINFER_ACTIVATION_PORT=29620 \
bash scripts/start_remote_worker.sh 0 0.0.0.0:29600 0

# VM 1 / ROCm；也可使用 ROCR_VISIBLE_DEVICES
HXINFER_VISIBLE_DEVICE_ENV=ROCR_VISIBLE_DEVICES \
HXINFER_ACTIVATION_HOST=10.0.0.10 HXINFER_ACTIVATION_PORT=29620 \
bash scripts/start_remote_worker.sh 1 0.0.0.0:29600 0
```

Controller 设置两个 control endpoint、Gloo rendezvous 和 rank-0 activation 地址后：

```bash
export HXINFER_WORKER_ENDPOINTS=10.0.0.10:29600,10.0.0.11:29600
export HXINFER_DIST_INIT=tcp://10.0.0.10:29610
bash scripts/run_remote_pp_controller.sh
```

需要放通三个端口：两个 worker control endpoint、Gloo rendezvous、activation
端口（如果两个 worker IP 不同，control 端口可以相同）。当前 activation rank 0
监听，rank 1 连接；因此 rank 0 可绑定 `0.0.0.0`，而 rank 1 的
`HXINFER_ACTIVATION_HOST` 必须设置为可达的 rank-0 地址。该变量是 worker 侧环境，
只在 Controller 上设置不会传播到已独立启动的远程 worker。

## 5. AMD/双 VM 迁移边界

本次已经完成的是 Stage 生命周期的进程解耦和 TCP activation 数据面，所以从双
4090 到双 VM 不再需要改 vLLM scheduler/executor 结构。仍需完成的目标机工作是：

1. 在 CUDA 与 ROCm 环境固定同一 vLLM 版本和可兼容的序列化 schema；
2. 验证 CUDA/ROCm PyTorch build 的 CPU/Gloo wire compatibility；若不兼容，移除
   worker 间 ProcessGroup，仅保留 HXinfer 控制协议中的虚拟 PP rank；
3. AMD 侧验证目标模型、attention backend、BF16 和 HIP pinned H2D；
4. 将 pickle/`torch.save` 换成显式 header + 原始 tensor bytes；
5. 增加 request/step id、checksum、timeout、断线恢复和 backpressure；
6. 根据两张异构卡的实测层耗时做非均匀 layer partition。

结论是“架构和脚本边界可迁移”，不是“CUDA+ROCm 已验证”。双 VM 的 TCP 路径与
本次 loopback TCP 在协议上等价，但会增加 guest network/vSwitch 开销，性能必须在
目标环境重新测量。
