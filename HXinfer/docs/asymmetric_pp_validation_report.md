# HXinfer 非对称 Pipeline Parallel 实验报告

> 日期：2026-08-26
>
> 结论：PASS

## 1. 实验目标

Qwen2.5-3B-Instruct 包含 36 个 Transformer hidden layers。vLLM 在 PP=2
且未指定 partition 时默认均分为 `18+18`。本次工作为 HXinfer 增加显式
非对称切层能力，并实际验证以下配置：

- `24+12`；
- `11+25`；
- `10+26`；
- 非法配置 `10+16` 能否被拒绝。

验证不仅检查配置解析，还要求两个异构 Worker 完成实际模型加载、KV cache
初始化、activation 传输和端到端问答。

## 2. 代码修改说明

vLLM 0.27.1 的 `vllm.distributed.utils.get_pp_indices` 已原生读取
`VLLM_PP_LAYER_PARTITION`。HXinfer 没有复制或修改 vLLM 的模型切层算法，
改动集中在外部进程管理层：

1. 新增 `external_pp.partition`，负责解析、规范化和校验 partition；
2. Worker Service 新增 `--layer-partition`，在导入 vLLM 前设置
   `VLLM_PP_LAYER_PARTITION`；
3. Worker 的 `__ping__` 返回自身 partition；
4. Remote Executor 在加载模型前核对 Controller 与两个 Worker 的配置，
   任一进程不一致则立即失败；
5. 模型加载后，通过 `get_hxinfer_partition_info` 读取每个 stage 的实际
   `start_layer`、`end_layer_exclusive` 和 `layer_count`；
6. API、Worker 和 Controller 启动脚本统一支持
   `VLLM_PP_LAYER_PARTITION`，并记录预期配置和实际生效边界。

关键日志：

```text
HXINFER_PARTITION_CONFIG partition=11,25
HXINFER_PARTITION_ACTIVE stages=[...]
```

## 3. 配置规则

PP=2 时必须提供两个正整数，且总和必须等于模型 hidden layer 数量。

| 配置 | 36 层模型上的结果 |
|---|---|
| `18,18` | 合法，默认均匀切分 |
| `24,12` | 合法，rank 0 `[0,24)`，rank 1 `[24,36)` |
| `11,25` | 合法，rank 0 `[0,11)`，rank 1 `[11,36)` |
| `10,26` | 合法，rank 0 `[0,10)`，rank 1 `[10,36)` |
| `10,16` | 不合法，总和 26 不等于 36，模型加载阶段拒绝 |

因此切分没有硬编码为某一组数字。`10+16` 仅能用于恰好包含 26 个 hidden
layers 的模型；对当前 36 层模型应使用总和为 36 的组合。

## 4. 实验环境与启动方式

| 项目 | 配置 |
|---|---|
| 模型 | Qwen2.5-3B-Instruct，BF16，36 层 |
| PP / TP | PP=2 / TP=1 |
| rank 0 | NVIDIA RTX 4090，CUDA |
| rank 1 | AMD RX 7900 XTX，ROCm |
| vLLM | 0.27.1 |
| CUDA 镜像 | `vllm/vllm-openai:latest` |
| ROCm 镜像 | `vllm/vllm-openai-rocm:latest` |
| activation 数据面 | pinned host staging + TCP loopback |
| API | OpenAI 兼容接口，三次实验分别使用端口 8010、8011、8012 |

服务器上原有镜像仍然存在：

```text
CUDA sha256:0a51ea5b4ae2dc5d81890e5173f54203d2a3ae0cfffe51b8fd2afd4391bfd967
ROCm sha256:bb44b39aea26798cce43030a98bf48efd0322ca7147367db86e38b96bd80f0e7
```

每组实验均按以下顺序启动三个独立进程；容器仅提供各自的 CUDA、ROCm
运行环境：

1. 在 CUDA 容器中启动 rank 0 Worker；
2. 在 ROCm 容器中启动 rank 1 Worker；
3. 在 API 容器中启动 Controller/API 进程，由它连接两个已有 Worker；
4. 健康检查通过后调用 `scripts/demo_api.py`；
5. 保存三个进程的日志、容器配置和双向 transport JSONL，随后停止容器。

三处配置必须完全相同。例如 `11+25` 使用：

```bash
export VLLM_PP_LAYER_PARTITION=11,25
```

也可以在 HXinfer 配置文件中设置同名变量，再分别执行已有的 Worker 与 API
启动脚本。Remote Executor 会在模型加载前核对三处配置，避免不同进程按
不同边界加载模型。

## 5. 实测结果

### 5.1 三组实际层边界

| partition | CUDA rank 0 | ROCm rank 1 | 实际边界检查 |
|---|---|---|---|
| `24+12` | `[0,24)`，24 层 | `[24,36)`，12 层 | PASS |
| `11+25` | `[0,11)`，11 层 | `[11,36)`，25 层 | PASS |
| `10+26` | `[0,10)`，10 层 | `[10,36)`，26 层 | PASS |

这些范围来自模型加载完成后的 Worker 实际状态，不是根据启动参数推算。

### 5.2 权重显存、加载时间与 KV cache

| partition | CUDA 权重 / 加载时间 / KV | ROCm 权重 / 加载时间 / KV | Engine 初始化 |
|---|---|---|---:|
| `24+12` | 4.06 GiB / 12.02 s / 14.55 GiB | 2.32 GiB / 11.05 s / 13.31 GiB | 50.19 s |
| `11+25` | 2.18 GiB / 2.80 s / 16.46 GiB | 4.20 GiB / 7.24 s / 11.40 GiB | 49.94 s |
| `10+26` | 2.03 GiB / 2.63 s / 16.61 GiB | 4.35 GiB / 6.15 s / 11.26 GiB | 49.87 s |

当更多层从 CUDA stage 移到 ROCm stage 后，CUDA 权重占用下降且可用 KV
cache 增加，ROCm 权重占用上升且可用 KV cache 减少，变化方向与实际层数
一致。这进一步证明非对称 partition 已作用到模型权重加载。

加载时间会受到首次编译、缓存和服务器状态影响，本表不能作为三个 partition
的性能排名。三次 Engine 初始化均约 50 秒，也不代表稳态请求延迟。

### 5.3 端到端问答与 Worker 执行

三组实验均完成健康检查和 OpenAI 兼容 API 请求：

| partition | 问题 | 输出 | 两个 Worker `execute_model` | 结果 |
|---|---|---:|---:|---|
| `24+12` | `12×3` | `36` | 各 162 次（含扩展问答） | PASS |
| `11+25` | `12×3` | `36` | 各 7 次 | PASS |
| `10+26` | `12×3` | `36` | 各 7 次 | PASS |

三次算术请求的 token usage 均为 42 prompt tokens、3 completion tokens。
`24+12` 还额外执行了正常中文问答并生成 77 completion tokens。

### 5.4 activation 传输校验

| partition | 消息数 | 总 payload | 最大单条 | 配对检查 |
|---|---:|---:|---:|---|
| `24+12` | 84 | 1,603,620 bytes | 345,893 bytes | PASS |
| `11+25` | 7 | 594,435 bytes | 345,893 bytes | PASS |
| `10+26` | 7 | 594,435 bytes | 345,893 bytes | PASS |

每组实验的 rank 0 发送日志和 rank 1 接收日志在 sequence、消息数和 payload
字节数上完全配对，说明修改 layer boundary 后仍沿用完整的
`CUDA D2H -> TCP loopback -> ROCm H2D` activation 数据路径。

这里的消息数受生成 token 数和额外测试请求数量影响，不能直接用于比较
partition 性能；软件阶段计时也不是裸 PCIe 或 TCP 带宽。

## 6. 自动化与异常测试

新增测试覆盖：

- 空配置继续使用 vLLM 默认切分；
- `18,18`、`24,12`、`11,25`、`10,26` 的解析和规范化；
- entry 数量错误、非整数、零和负数被拒绝；
- Controller 与两个 Worker 配置一致时通过；
- 任一 Worker 配置不一致时 fail closed；
- 36 层模型使用 `10,16` 时因总层数不匹配被 vLLM 拒绝。

代码检查结果：

```text
34 passed
Ruff check: PASS
Ruff format: PASS
Shell syntax: PASS
```

## 7. 原始证据与清理状态

服务器原始证据目录：

```text
/data/wuxin-infer/HXinfer/results/heterogeneous-qwen25-3b-asymmetric/
/data/wuxin-infer/HXinfer/results/heterogeneous-qwen25-3b-asymmetric-11-25/
/data/wuxin-infer/HXinfer/results/heterogeneous-qwen25-3b-asymmetric-10-26/
```

每个目录包含 API 日志、两个 Worker 日志、问答输出、容器配置和 transport
记录。实验结束后，本次创建的九个 `hxinfer-asym-*` 容器均已停止；没有停止
或修改其他用户正在运行的 `vllm-rocm-test` 容器。

## 8. 结论与限制

HXinfer 已支持 PP=2 的可配置非对称 layer partition。`24+12`、`11+25` 和
`10+26` 均在 RTX 4090 + RX 7900 XTX 上完成实际模型加载、KV cache 初始化、
跨异构 Worker activation 传输和 API 问答，非法总层数能够被拒绝。

本次实验验证的是功能正确性，不是最优配比。后续应使用相同 prompt、并发、
输入输出长度和预热条件，对各 stage 稳态延迟进行重复测量，再搜索使两个
stage latency 接近的 partition。
