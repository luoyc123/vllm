# HXinfer 使用说明

本文说明如何在同一宿主机的 NVIDIA CUDA GPU 与 AMD ROCm GPU 上启动
HXinfer，并通过一个 OpenAI 兼容 API 使用两段 Pipeline Parallel（PP）推理。

## 1. 架构

HXinfer 运行三个服务进程：

1. CUDA Worker：PP rank 0，加载模型前半部分；
2. ROCm Worker：PP rank 1，加载模型后半部分；
3. API/Controller：管理 vLLM EngineCore、Scheduler 和两个远程 Worker，
   并只对外暴露一个 HTTP API。

同宿主机默认数据路径为：

```text
CUDA GPU --PCIe D2H--> host --TCP loopback--> host --PCIe H2D--> AMD GPU
```

TCP 传输完整的中间 activation，而不是显存地址。两个 Worker 使用独立的
CUDA/ROCm vLLM 运行环境，但属于同一个 PP world。

## 2. 前置条件

- Linux x86_64；
- Docker；
- 一张受 vLLM CUDA 镜像支持的 NVIDIA GPU；
- 一张受 vLLM ROCm 镜像支持的 AMD GPU；
- NVIDIA Container Toolkit；
- 宿主机可访问 `/dev/kfd` 和 `/dev/dri`；
- 本地模型目录，例如 `/data/models/Qwen2.5-3B-Instruct`；
- CUDA 与 ROCm 容器能够访问同一份模型权重和 HXinfer 源码。

已验证组合及限制见
[`docs/heterogeneous_amd_nvidia_validation_report.md`](docs/heterogeneous_amd_nvidia_validation_report.md)。
首次使用建议采用 BF16/FP16 小模型验证，不要直接使用依赖特定 AMD kernel
的 block-FP8 模型。

## 3. 准备目录

以下命令假设 vLLM 仓库位于 `/data/vllm`，HXinfer 位于仓库的
`/data/vllm/HXinfer`，模型位于 `/data/models`。如果目录不同，请统一替换：

```bash
cd /data/vllm/HXinfer
mkdir -p results/tutorial-qwen25-3b
```

确认模型、镜像和 GPU：

```bash
test -f /data/models/Qwen2.5-3B-Instruct/config.json
docker image inspect vllm/vllm-openai:latest >/dev/null
docker image inspect vllm/vllm-openai-rocm:latest >/dev/null
nvidia-smi
/opt/rocm/bin/rocm-smi --showproductname
```

## 4. 创建三个容器环境

这些命令只启动容器环境；`sleep infinity` 不会加载模型或启动 Worker/API。

CUDA Worker 容器：

```bash
docker run -d --name hxinfer-cuda \
  --gpus device=0 --network host \
  -v /data/vllm/HXinfer:/workspace/HXinfer:ro \
  -v /data/models:/models:ro \
  -v /data/vllm/HXinfer/results/tutorial-qwen25-3b:/results \
  -e HOME=/tmp -e PYTHONPATH=/workspace/HXinfer \
  -e HXINFER_ACTIVATION_HOST=127.0.0.1 \
  -e HXINFER_ACTIVATION_PORT=29620 -e HXINFER_LOG_DIR=/results \
  --entrypoint /bin/bash vllm/vllm-openai:latest \
  -lc 'sleep infinity'
```

ROCm Worker 容器：

```bash
docker run -d --name hxinfer-rocm \
  --device=/dev/kfd --device=/dev/dri \
  --group-add video --group-add render \
  --security-opt seccomp=unconfined --network host \
  -v /data/vllm/HXinfer:/workspace/HXinfer:ro \
  -v /data/models:/models:ro \
  -v /data/vllm/HXinfer/results/tutorial-qwen25-3b:/results \
  -e HOME=/tmp -e PYTHONPATH=/workspace/HXinfer \
  -e HXINFER_ACTIVATION_HOST=127.0.0.1 \
  -e HXINFER_ACTIVATION_PORT=29620 -e HXINFER_LOG_DIR=/results \
  --entrypoint /bin/bash vllm/vllm-openai-rocm:latest \
  -lc 'sleep infinity'
```

如果镜像中不存在 `video` 或 `render` 组，请使用宿主机上有权访问
`/dev/kfd`、`/dev/dri` 的实际 GID 替换。

API/Controller 容器：

```bash
docker run -d --name hxinfer-api \
  --gpus device=0 --network host \
  -v /data/vllm/HXinfer:/workspace/HXinfer:ro \
  -v /data/models:/models:ro \
  -v /data/vllm/HXinfer/results/tutorial-qwen25-3b:/results \
  -e HOME=/tmp -e PYTHONPATH=/workspace/HXinfer \
  -e HXINFER_WORKER_ENDPOINTS=127.0.0.1:29600,127.0.0.1:29601 \
  -e HXINFER_DIST_INIT=tcp://127.0.0.1:29610 \
  -e HXINFER_LOG_DIR=/results -e VLLM_USE_FLASHINFER_SAMPLER=0 \
  --entrypoint /bin/bash vllm/vllm-openai:latest \
  -lc 'sleep infinity'
```

检查容器环境：

```bash
docker ps --filter name=hxinfer \
  --format 'table {{.Names}}\t{{.Status}}'
```

## 5. 启动两个 Worker 进程

打开终端 A，进入 CUDA 容器：

```bash
docker exec -it hxinfer-cuda /bin/bash
```

在容器内启动 rank 0：

```bash
/usr/bin/python3 -m external_pp.remote.worker_service \
  --rank 0 --listen 127.0.0.1:29600 \
  2>&1 | tee -a /results/worker-cuda.log
```

打开终端 B，进入 ROCm 容器：

```bash
docker exec -it hxinfer-rocm /bin/bash
```

在容器内启动 rank 1：

```bash
/usr/bin/python3 -m external_pp.remote.worker_service \
  --rank 1 --listen 127.0.0.1:29601 \
  2>&1 | tee -a /results/worker-rocm.log
```

两个终端应分别出现：

```text
HXINFER_WORKER_READY rank=0
HXINFER_WORKER_READY rank=1
```

容器和 Worker 是两层不同的生命周期。关闭上述终端会停止对应 Worker，
但不会自动删除容器。

## 6. 启动单一 API 进程

打开终端 C，进入 API 容器：

```bash
docker exec -it hxinfer-api /bin/bash
```

在容器内启动 API：

```bash
/usr/bin/python3 -m external_pp.api_server \
  --model /models/Qwen2.5-3B-Instruct \
  --served-model-name Qwen2.5-3B-Instruct \
  --host 0.0.0.0 --port 8000 --api-key hxinfer-demo \
  --worker-cls external_pp.remote.worker.HXinferRemoteStageWorker \
  --distributed-executor-backend external_pp.remote.executor.HXinferRemoteExecutor \
  --pipeline-parallel-size 2 --tensor-parallel-size 1 \
  --max-model-len 512 --max-num-batched-tokens 512 --max-num-seqs 8 \
  --max-tokens 64 --gpu-memory-utilization 0.80 \
  2>&1 | tee -a /results/api-server.log
```

API 进程会连接两个 Worker，并完成分布式建组、模型分片加载、KV cache
初始化和 warmup。等待日志出现：

```text
Uvicorn running on http://0.0.0.0:8000
```

生产部署必须替换演示 API key，并限制 Worker 控制端口和 activation 端口的
网络访问范围。

## 7. 请求与验证

在宿主机的第四个终端执行：

```bash
cd /data/vllm/HXinfer
curl -sS http://127.0.0.1:8000/health

HXINFER_API_BASE=http://127.0.0.1:8000 \
HXINFER_API_KEY=hxinfer-demo \
python scripts/demo_api.py \
  --prompt '请只回答数字：12乘以3等于多少？'
```

预期输出包含：

```text
health: ok
request 1/1 assistant: 36
```

确认同一请求经过两个 Worker：

```bash
grep 'execute_model.*ok=true' \
  results/tutorial-qwen25-3b/worker-cuda.log | tail -n 10
grep 'execute_model.*ok=true' \
  results/tutorial-qwen25-3b/worker-rocm.log | tail -n 10

wc -l \
  results/tutorial-qwen25-3b/transport-rank0.jsonl \
  results/tutorial-qwen25-3b/transport-rank1.jsonl
```

两个 Worker 日志应出现相同 RPC id，两份 transport 日志应成对增长。

## 8. 停止与再次启动

在 API、CUDA Worker 和 ROCm Worker 终端分别按 `Ctrl-C` 停止服务进程。
需要停止容器时执行：

```bash
docker stop hxinfer-api hxinfer-cuda hxinfer-rocm
```

再次使用时先执行 `docker start`，然后重新进入三个容器启动两个 Worker 和
一个 API 进程。模型和结果目录不会随容器停止而删除。

## 9. 迁移到两个 VM

两个 VM 中仍只需要两个 Worker 和一个 API/Controller：

- VM 1：CUDA Worker，可同时承载 API/Controller；
- VM 2：ROCm Worker；
- 将 `127.0.0.1` 替换为两个 VM 可互相访问的 IP；
- 开放控制 RPC、Gloo rendezvous 和 activation TCP 端口；
- 两个 VM 使用相同模型路径或各自准备相同权重；
- 不依赖跨 VM 共享虚拟内存，activation 继续通过 TCP 传输。

双 VM 部署前应增加访问控制、连接超时、版本握手、重连和 sequence gap
检查。当前实现是功能验证原型，不是已完成容错和性能优化的生产服务。
