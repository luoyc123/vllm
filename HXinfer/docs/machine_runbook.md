# 租用双 NVIDIA GPU 机器执行手册

## 1. 创建隔离环境

建议使用项目专属 venv/conda env，并按目标 vLLM 官方安装方式安装。不要依赖镜像中碰巧存在的包。

验证：

```bash
python -m pip check
python -c 'import vllm, torch, transformers; print(vllm.__version__, torch.__version__, transformers.__version__)'
vllm --help
```

将机器上的实际源码 checkout/commit 记录下来；项目当前基线为 vLLM 0.27.1，若租机版本不同，先更新 `docs/vllm_pp_analysis.md` 的差异。

## 2. 配置与预检

```bash
cd HXinfer
cp configs/experiment.env.example configs/experiment.env
$EDITOR configs/experiment.env
bash scripts/preflight.sh
```

模型选择原则：模型实现明确支持 PP=2；正常 workload 下单卡放不下或切分有实际意义；双卡有显存余量。首次 smoke 可先用小模型验证命令，但小模型结果不能作为最终性能 baseline。

## 3. Native PP baseline

```bash
bash scripts/run_native_pp.sh
```

脚本会创建 `results/native_pp/<UTC run id>/`，保存：环境、GPU inventory、`nvidia-smi topo -m`、vLLM 关键源码 SHA256/符号行号、预期 layer partition、server log、benchmark log/JSON 和完整启动命令。

脚本同时启动 `nvidia-smi dmon`，把显存与利用率时间序列保存为 `nvidia_dmon.log`。

检查清单：

- 两个 worker 分别绑定 GPU0/GPU1；
- `partition.json` 与实际 stage model 一致；
- 两 GPU 显存占用合理，无第三 GPU；
- server log 中无 fallback、OOM、NCCL error；
- benchmark JSON 包含 throughput、TTFT、TPOT；
- 固定参数至少重复 3 次，保留每次原始结果；
- 补录 activation keys/shape/dtype/bytes，以及 stage compute/communication timer。

确认后才执行：

```bash
printf '%s\n' '<accepted-run-id>' > results/native_pp/PASS
```

## 4. 下一阶段现场决策

Native baseline 通过后，不直接跑 decode。先基于所选模型和 checkout 完成一个窄的 split-forward adapter，并对同一 input 比较：

```text
max_abs_error
mean_abs_error
relative_l2
```

误差阈值需结合 dtype 决定并写入 run metadata；不要用“输出文本相同”代替 logits 比较。正确性不通过时保存 input ids、positions、boundary tensors 和两份 logits，再定位层边界/position/KV/残差问题。
