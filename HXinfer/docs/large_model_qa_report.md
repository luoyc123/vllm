# Qwen2.5-3B-Instruct 双进程 PP 问答验证

## 结论

2026-08-24 在双 RTX 4090 上完成 Qwen2.5-3B-Instruct、BF16、PP=2/TP=1 的 Native 和 External PP 验证。模型落盘约 5.8 GiB，两个 vLLM worker 各加载 3.19 GiB 模型内存，确认仍为真实分层而非完整模型双副本。

本轮结论分成两部分：

- 功能与稳定性通过：External PP 完成 6 问批量 chat、最长 512-token decode、516 对 stage 消息，无丢消息、死锁、worker 崩溃或 GPU 进程泄漏。
- 严格 correctness gate 未通过：在确定性的 Triton attention 配置下，Native 重复运行 6/6 token 序列相同，External 重复运行也 6/6 相同，但 Native 与 External 最终序列不同。首个分叉的零起始索引为 34–218（即第 35–219 个生成 token），不能把它解释成随机采样。

因此，0.6B smoke 的 16-token 完全一致结论仍成立，但不能外推为 3B 长序列已经严格等价。下一阶段必须先定位 boundary/下游数值差异，再做正式性能结论。

## 测试配置

- vLLM 0.27.1，PyTorch 2.13.0+cu130。
- Qwen2.5-3B-Instruct，BF16，`max_model_len=2048`。
- 两个独立 `VLLM::Worker_PP0/PP1`，PP=2、TP=1。
- greedy decoding，6 个 chat prompt。
- correctness 对照强制 `TRITON_ATTN`、`PYTHONHASHSEED=0`、`CUBLAS_WORKSPACE_CONFIG=:4096:8`，并关闭 FlashInfer sampler。

测试问题覆盖并行概念、算术推理、Python 代码、无 NVLink 的 GPU PP、KV cache 和英文总结，定义见 `configs/qa_prompts.json`。

## 运行数据

256-token 对照中：

| 路径 | engine load | 6 问 generation | stage 消息 |
|---|---:|---:|---:|
| Native Triton PP | 19.118 s | 7.418 s | Native NCCL |
| External socket PP | 15.714 s | 4.448 s | 261 对 |

初始化缓存和输出序列已不同，所以表中时间不能作为 Native/External 性能优劣结论。

External 512-token 问答耗时 8.635 s，共传输 516 对消息、累计约 24.35 MiB，最大序列化 payload 为 4,196,133 bytes。trace 已拆分记录 D2H、socket 和 H2D；当前 D2H timer 会包含当前 CUDA stream 上尚未完成的 stage0 工作，不能直接视为纯复制耗时。

## 问答检查

External 512-token 结果中：

- 火车过桥题正确得到 200 米，并给出单位换算和方程。
- Python 题给出两次线性扫描的字典实现，时间/空间复杂度均为 O(n)。
- 英文总结正常结束。
- 模型对 PP、无 NVLink 通信和 KV cache 的部分解释不够严谨，例如把 GPU 内部 shared memory 描述成 GPU 间传输机制。这属于 3B 模型回答质量，不是 PP 执行失败；报告保留原始输出，不做人工美化。

## 已排除与待定位

已确认：

- FlashAttention 下 Native 自身跨重启不具备逐 token 确定性，不能用于严格 A/B gate。
- 改用 Triton attention 后 Native-vs-Native 和 External-vs-External 均完全可重复。
- External 已把 boundary tensor 规范化为 contiguous，以匹配 vLLM Native receiver 只根据 shape 分配连续张量的语义；该修复没有消除 Native/External 分叉。

下一步应在同一批 token 前缀下：

1. 对 stage0 boundary tensor 记录 shape、dtype、stride 和逐 tensor 哈希，同时用审计 worker 记录 Native send 的同类数据。
2. 若 boundary 字节一致，在 stage1 逐层比较 hidden states/logits，找到首次数值分叉层。
3. 若 boundary 字节不同，核对 vLLM native send 对 view、storage offset 和异步 stream 的实际处理。
4. 在上述问题解决前，不把 3B 测试标记为严格 correctness PASS，也不进行 Native/External 性能排名。

## 原始结果

- `results/native_pp/qwen2.5-3b-qa256-triton2/`
- `results/native_pp/qwen2.5-3b-qa256-triton2-repeat/`
- `results/external_socket/qwen2.5-3b-qa256-contiguous/`
- `results/external_socket/qwen2.5-3b-qa512/`

每个目录包含 generation JSON 和完整运行日志；External 目录另含 rank0/rank1 JSONL transport trace。
