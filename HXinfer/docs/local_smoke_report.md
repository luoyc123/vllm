# 本机单卡 Smoke Test

测试日期：2026-08-23。该测试只验证安装、CUDA、模型加载和单卡推理，不放行 Native PP gate。

## 环境

```text
OS: WSL2
GPU: NVIDIA GeForce RTX 4060 Laptop GPU
GPU memory: 8188 MiB
Windows driver: 576.80
Driver supported CUDA: 12.9
vLLM: 0.27.1+cu129
PyTorch: 2.13.0+cu129
Model: /home/arden/huggingface/Qwen3-0.6B
```

最初的 PyPI `0.27.1` wheel 带 CUDA 13.0 PyTorch，当前驱动无法初始化。最终使用 vLLM 官方 `0.27.1+cu129` wheel 与 PyTorch 官方 `2.13.0+cu129` wheel。依赖 `pip check` 通过，CUDA tensor smoke 通过。

WSL 下 V2 Model Runner 初始化时报 `UVA is not available`，所以本地 smoke 设置：

```text
VLLM_USE_V2_MODEL_RUNNER=0
```

这不是双卡实验应沿用的性能配置；原生 Linux 租机应先使用默认 V2 runner。

## Qwen3-0.6B 结果

```text
pipeline_parallel_size: 1
tensor_parallel_size: 1
dtype: float16
max_model_len: 512
gpu_memory_utilization: 0.65
model weights on GPU: 1.12 GiB
KV cache: about 3.4 GiB / 31,856 tokens
engine load + warmup: 92.37 s
generation: 0.405 s
prompt tokens: 7
output tokens: 16
observed output rate: about 40.4 tokens/s
```

原始 JSON 位于 `results/device_smoke/qwen3_0.6b_single_gpu.json`。由于只有一张可见 GPU，本结果不能回答 PP layer partition、PP activation 通信或 PP=2 性能问题。

复现命令：

```bash
CUDA_VISIBLE_DEVICES=0 .venv/bin/python scripts/smoke_vllm.py \
  --model /home/arden/huggingface/Qwen3-0.6B \
  --output results/device_smoke/qwen3_0.6b_single_gpu.json
```
