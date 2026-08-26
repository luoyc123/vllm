# HXinfer

HXinfer 是一个面向 CUDA/ROCm 异构 Pipeline Parallel 推理的验证项目。当前原则是：先确认 vLLM 原生 PP，再做最小外部 PP 原型；不直接 fork 或重构 vLLM。

当前已完成：

- 最新稳定版 vLLM 0.27.1 的隔离环境与 PP 调用链源码分析；
- 可重复执行的环境预检、环境采集、Native PP 启动与 benchmark 脚本；
- 与设备无关的 `IntermediateTensors` 消息契约、transport 接口和数值比较工具；
- 分阶段 gate，防止 baseline 未通过时误跑 External PP；
- 实验报告和机器执行手册模板。
- 双 RTX 4090 上 Native PP=2 与 External PP=2 的实际生成验证；结果见 [docs/two_process_pp_report.md](docs/two_process_pp_report.md)。
- Qwen2.5-3B-Instruct 的 6 问批量 chat 与长 decode 验证；功能通过但严格 token gate 尚未通过，详见 [docs/large_model_qa_report.md](docs/large_model_qa_report.md)。
- Qwen2.5-3B-Instruct 的 8 道短答案题全部正确，Native/External 输出 token 逐项一致；完整结论与限制见 [docs/final_validation_report.md](docs/final_validation_report.md)。
- 一个 EngineCore 管理两个预先独立启动的 vLLM worker service 已在双 RTX 4090 跑通；正常问答累计生成 690 token，见 [docs/remote_worker_validation_report.md](docs/remote_worker_validation_report.md)。

旧的同机 External PP 入口是 `external_pp.vllm_worker.HXinferWorker`。面向双 VM 的新入口是 `external_pp.remote.executor.HXinferRemoteExecutor` 与 `external_pp.remote.worker.HXinferRemoteStageWorker`：两个 worker 独立启动，Controller 通过 TCP RPC 管理它们，完整 `IntermediateTensors` mapping 经 pinned host staging 和持久 TCP socket 传输。

## 快速开始

```bash
cd HXinfer
cp configs/experiment.env.example configs/experiment.env
# 修改 MODEL、VLLM_ENV 等配置
bash scripts/preflight.sh
bash scripts/run_native_pp.sh
```

双 4090 上运行独立 worker 版本：

```bash
HXINFER_CONFIG=configs/remote_4090.env bash scripts/run_remote_pp_local.sh
```

双节点/双 VM 分别使用 `scripts/start_remote_worker.sh` 启动 rank 0/1，再在 Controller 节点运行 `scripts/run_remote_pp_controller.sh`。所需地址和端口见 [独立 worker 验证报告](docs/remote_worker_validation_report.md)。

对外提供单一 OpenAI 兼容入口时，先启动两个 Worker，再运行：

```bash
MODEL=/models/Qwen2.5-3B-Instruct \
SERVED_MODEL_NAME=Qwen2.5-3B-Instruct \
HXINFER_WORKER_ENDPOINTS=127.0.0.1:29600,127.0.0.1:29601 \
HXINFER_DIST_INIT=tcp://127.0.0.1:29610 \
MAX_MODEL_LEN=512 MAX_NUM_BATCHED_TOKENS=512 MAX_NUM_SEQS=8 \
bash scripts/run_remote_pp_api.sh
```

服务暴露 `/health`、`/v1/models`、`/v1/chat/completions` 和 `/v1/completions`。当前原型支持非流式请求；对外只有一个 API 地址，两个异构 Worker 不暴露 HTTP 接口。

服务启动后可用零第三方依赖客户端完成演示：

```bash
HXINFER_API_BASE=http://100.111.92.28:8000 \
HXINFER_API_KEY=hxinfer-demo \
python scripts/demo_api.py --prompt '请只回答数字：12乘以3等于多少？'
```

地址和 key 均应按部署环境替换。

从 Docker 环境创建、两个 Worker 启动到单一 API 验证的完整命令见
[USAGE.md](USAGE.md)。

完整顺序见 [docs/machine_runbook.md](docs/machine_runbook.md)。架构边界和阶段状态见 [docs/implementation_plan.md](docs/implementation_plan.md)。

本机 RTX 4060 的实测记录见 [docs/local_smoke_report.md](docs/local_smoke_report.md)。

单卡环境可先执行不放行 PP gate 的 smoke test：

```bash
CUDA_VISIBLE_DEVICES=0 .venv/bin/python scripts/smoke_vllm.py \
  --model /home/arden/huggingface/Qwen3-0.6B
```
