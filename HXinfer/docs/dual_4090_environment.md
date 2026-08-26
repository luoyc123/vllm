# 双 RTX 4090 测试环境快照

采集时间：2026-08-24 11:21（Asia/Shanghai）

## 软件环境

```text
OS: Ubuntu 22.04.5 LTS
Kernel: 5.15.0-97-generic x86_64
CPU: 32 cores
RAM: 240 GB
Python: 3.12.3
NVIDIA driver: 580.76.05
CUDA reported by driver: 13.0
PyTorch: 2.13.0+cu130
vLLM: 0.27.1
Transformers: 5.15.1
NCCL reported by vLLM: 2.29.7
```

## GPU inventory

```text
GPU0: NVIDIA GeForce RTX 4090, 24,564 MiB, PCI 00000000:B8:00.0
GPU1: NVIDIA GeForce RTX 4090, 24,564 MiB, PCI 00000000:D8:00.0
```

采集结束时两张 GPU 均为 0 MiB 使用量，`nvidia-smi` 未列出运行进程。

## 拓扑

`nvidia-smi topo -m`：

```text
        GPU0  GPU1  NIC0  CPU Affinity   NUMA Affinity
GPU0     X    NODE  NODE  32-63,96-127  1
GPU1    NODE   X    NODE  32-63,96-127  1
NIC0    NODE  NODE   X
```

`NODE` 表示路径经过 PCIe，并经过同一 NUMA node 内不同 PCIe Host Bridge；没有 `NV#`，因此不存在 NVLink。

`nvidia-smi topo -p2p p`：

```text
      GPU0 GPU1
GPU0   X    NS
GPU1  NS     X
```

`NS` 表示 PCIe P2P 不支持。因此本机原生 NCCL 不能假设使用 PCIe GPU-direct P2P，HXinfer 的 host-staging 路径与目标异构场景具有较好的功能代表性，但不能由此直接外推性能。
