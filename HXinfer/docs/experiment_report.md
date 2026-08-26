# HXinfer 实验报告

状态：尚未上机；本文件是结果模板，不包含虚构数据。

## 实验矩阵

| Case | 架构 | Correctness | TTFT | TPOT | Throughput | Comm ratio |
|---|---|---:|---:|---:|---:|---:|
| A | vLLM Native PP=2 | 待测 | 待测 | 待测 | 待测 | 待测 |
| B | External PP + Host IPC | G2 后测试 | 待测 | 待测 | 待测 | 待测 |
| C | External PP + Socket | G4 后测试 | 待测 | 待测 | 待测 | 待测 |
| D | GPU PCIe P2P（可选） | 待测 | 待测 | 待测 | 待测 | 待测 |

## 环境与模型

- Model/revision：待填
- vLLM commit：待填
- PyTorch/CUDA/Driver：待填
- GPU/topology：待填，链接到 raw results
- 输入/输出长度、并发、seed：待填
- Stage 0 layers / Stage 1 layers：待填
- Boundary tensors（key/shape/dtype/bytes）：待填

## 正确性

| 比较 | max_abs_error | mean_abs_error | relative_l2 | 阈值 | 结论 |
|---|---:|---:|---:|---:|---|
| native forward vs external split forward | 待测 | 待测 | 待测 | 待定 | 待测 |
| native greedy decode vs external decode | token exact | — | — | exact | 待测 |

## 分段性能

| Case | Stage 0 compute | D2H | handoff/socket | H2D | Stage 1 compute | End-to-end |
|---|---:|---:|---:|---:|---:|---:|
| A | 待测 | — | Native PP comm | — | 待测 | 待测 |
| B | 待测 | 待测 | 待测 | 待测 | 待测 | 待测 |
| C | 待测 | 待测 | 待测 | 待测 | 待测 | 待测 |

## 必答问题

### 1. vLLM 当前如何管理 PP Worker 和 PP 通信？

初步源码答案见 `vllm_pp_analysis.md`；上机后按实际 commit 更新。

### 2. 两个独立环境能否正确完成模型 Pipeline Parallel？

待 G2/G4/G5 实验回答。

### 3. 相比 Native PP，External PP 的性能损失来自哪里？

待用 D2H、handoff/socket、H2D、同步等待和 stage imbalance 的实测分解回答。

### 4. 迁移到 RTX 4090 + AMD ROCm 时哪些可复用？

预计可复用：controller、request/step 协议、tensor metadata、正确性/benchmark 框架。需重测或实现：ROCm StageBackend、跨 runtime buffer 注册/拷贝、可用 transport、驱动/隔离拓扑；最终结论待真实硬件验证。
