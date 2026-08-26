# 双 vLLM 进程 PP 实测报告

## 结论

2026-08-24 已在同一台双 RTX 4090 机器上完成 Qwen3-0.6B Native PP=2 与 HXinfer External PP=2 实测。External 版本保留 vLLM 0.27.1 的双 worker、模型分层、调度、KV cache、forward 和 sampling，只把 PP boundary 的 `isend_tensor_dict` / `irecv_tensor_dict` 替换为 HXinfer transport。后续 3B 长序列测试发现严格 token gate 尚未通过，不能把本页的短序列结果外推到更大模型；见 `docs/large_model_qa_report.md`。

两个路径对同一模型、prompt、FP16 和 greedy 参数生成的 16 个 token ID 逐项相同，文本也相同。External 路径记录到 20 对 stage 消息，每条都含完整的 `hidden_states` 和 `residual` mapping；运行日志中没有 Native 路径出现的 NCCL P2P send/recv warning，因此 stage activation 已不走 vLLM 的 NCCL PP 数据面。

## 环境与配置

- GPU：2 × RTX 4090 24 GB；GPU P2P 查询不支持，无 NVLink。
- vLLM：0.27.1。
- PyTorch：2.13.0+cu130。
- 模型：Qwen3-0.6B，28 层，FP16。
- 并行：PP=2、TP=1；两个独立 `VLLM::Worker_PP0/PP1` 进程。
- 每个 stage 加载权重约 0.71 GiB，证明实际采用 layer partition，而不是各加载完整模型。
- 测试：7 input tokens、16 output tokens、`max_model_len=512`、`enforce_eager=True`。

## 首轮数据

| 路径 | engine load | generation | 输出 |
|---|---:|---:|---|
| Native NCCL PP | 15.420 s | 0.173 s | 16/16 token IDs 与 External 相同 |
| HXinfer Unix socket + pinned staging | 14.378 s | 0.179 s | 16/16 token IDs 与 Native 相同 |

load 时间受首次 JIT/warmup 缓存影响，不能据此比较。单请求 generation 只用于功能 smoke，也不是正式吞吐结论。

External transport 共记录 20 次 send 和 20 次 recv，序列号一一对应。序列化后的 payload 有 5,925、10,021、30,501、1,050,405 和 2,098,981 bytes 等规格。sender 侧总耗时均值约 2.22 ms，最大约 12.25 ms。receiver 的 `total_ns` 包含等待上游计算的阻塞时间，不能当作纯通信延迟；拆分后的 `socket_ns` 与 `h2d_ns` 才可用于后续分析。

可复核的原始文件保存在 `results/native_pp/token-check/` 与 `results/external_socket/token-check/`，包括 generation JSON、完整运行日志和两端 JSONL transport trace。

## 当前边界

这一版是 correctness-first 原型：socket 长连接、同步传输、每条消息 `torch.save` 序列化，并在两端经过 pinned host staging。它已经证明“双 vLLM worker + 外部 stage 数据面 + 连续 decode”可工作，但还不是性能版：

- steady state 仍有 host tensor/序列化缓冲分配；
- payload 经过 socket 内核拷贝；
- 当前限定同机 PP=2、TP=1、单向 stage0 → stage1；
- NCCL ProcessGroup 仍用于 vLLM worker 初始化/管理，但不再传 stage activation。

下一步若以性能为目标，应把 transport 内部换成预分配的固定 slot shared-memory ring，并为共享页做 CUDA host registration；vLLM 注入边界和上层 tensor mapping 无需改动。
