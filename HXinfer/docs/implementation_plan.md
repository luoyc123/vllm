# HXinfer 实施方案与 Gate

## 目标架构

```text
Controller
   | request_id / step_id / tokens
   v
StageWorker 0 (CUDA) -- IntermediateTensors + metadata --> StageWorker 1 (CUDA/ROCm)
   | stage-local KV cache                                | stage-local KV cache
   +------------------------- logits <-------------------+
```

控制面只管理请求顺序和生命周期；数据面 transport 发送完整 PP boundary mapping。上层协议只出现 tensor name、shape、dtype、payload、request_id、step_id 和 stage id，不出现 NCCL rank、CUDA IPC handle 或特定 device tensor。

## 当前已完成

- Phase 1：安装隔离的最新稳定版 vLLM 0.27.1，并按其异步 PP 路径完成源码分析。
- Phase 2 准备：Native PP 预检、环境/拓扑采集、启动、benchmark、partition 记录脚本。
- Phase 3 准备：device-neutral message schema、正确性指标函数和严格 gate。
- Phase 4 correctness 原型：已实现 Unix socket + pinned host staging 数据面，并在双 RTX 4090 上完成连续 decode；性能化的固定 slot shared memory 尚未实现。
- 异构 External PP：RTX 4090 + RX 7900 XTX 已完成单一 API、连续 decode 和
  `24+12`、`11+25`、`10+26` 三组非对称 layer partition；当前能够校验三个进程的配置一致性并记录
  实际层边界，最优性能比例仍需逐 stage 计时搜索。

## Gate 顺序

| Gate | 通过条件 | 未通过时禁止 |
|---|---|---|
| G0 环境 | CLI 可导入；2 GPU 可见；版本、模型、拓扑已采集 | 任何性能实验 |
| G1 Native PP | PP=2/TP=1 正确出结果；两 stage 层和显存确认；benchmark 原始数据完整 | External PP |
| G2 Split forward | native logits 与 external logits 的三项误差在约定阈值内 | Decode、transport 优化 |
| G3 Host IPC | mapping 全量传输正确；预分配 pinned buffers；分段耗时可观测 | Socket/VM 模拟 |
| G4 Decode | greedy token 序列一致；stage-local KV 生命周期无错位 | 性能结论 |
| G5 Socket | 跨容器正确，且不依赖共享 ProcessGroup/CUDA IPC | 双 VM 推断 |

`external_pp/controller.py` 会检查 `results/native_pp/PASS`。这个 PASS 必须由人审阅 baseline 后手工创建并写入 run id，脚本不会因为进程 exit code 为 0 就自动放行。

## 上机后的最小代码路线

1. 固定 vLLM commit 和一个明确支持 PP 的模型。
2. 跑 Native PP，确认实际 `IntermediateTensors` keys。
3. 只围绕 `gpu_worker.py` 的 recv/execute/send 边界做窄适配；不改 scheduler、模型层实现和 weight loader。
4. 先做单 prompt、单 forward、无连续 decode，并保存完整 logits 比较。
5. correctness 通过后实现固定 slot 的 HostIPCTransport：初始化时分配/注册 pinned host buffer，steady state 禁止 `tensor.cpu()` 隐式新分配。
6. decode 通过后才实现 framed SocketTransport；metadata 与 payload 分帧，检查长度、request_id、step_id。

## 性能模型

每条消息至少记录：

```text
payload_bytes
d2h_ms
handoff_or_socket_ms
h2d_ms
transport_total_ms
stage0_compute_ms
stage1_compute_ms
end_to_end_ms
```

由多种 payload size 拟合：

```text
T_comm(S) ~= alpha + S / B_effective
R_comm = T_communication / T_end_to_end
```

真实 RTX 4090 + AMD 环境只应替换 `StageBackend` 和 `PPTransport`。如果同机驱动共存失败，两个 stage 放入 container/VM；软件协议不变，但性能结论必须重测，不能从双进程直接外推。
