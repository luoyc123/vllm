# Gloo activation backend

HXinfer can select the complete PP activation data plane independently from
the API, EngineCore, Remote Executor, Worker lifecycle, and layer partition:

```bash
# Existing persistent socket backend (default)
export HXINFER_ACTIVATION_BACKEND=tcp

# CPU/Gloo tensor backend
export HXINFER_ACTIVATION_BACKEND=gloo
```

Set the same value in both pre-launched Worker processes. The API/Controller
command is unchanged. A Gloo launch prints:

```text
HXINFER_ACTIVATION_BACKEND backend=gloo rank=0
HXINFER_ACTIVATION_BACKEND backend=gloo rank=1
```

Both implementations use explicit Host staging for CUDA/ROCm interoperability:

```text
TCP:  GPU -> D2H -> Host -> torch.save/TCP/torch.load -> Host -> H2D -> GPU
Gloo: GPU -> D2H -> Host -> Gloo send/recv           -> Host -> H2D -> GPU
```

Gloo therefore does not imply CUDA/ROCm GPU P2P. It replaces the Python byte
serialization and custom socket framing with tensor-aware CPU/Gloo operations.
Transport JSONL records backend, sequence, keys, payload bytes, D2H, Gloo,
H2D, and total durations.

## Validation

Run the real two-process ProcessGroup test:

```bash
cd HXinfer
PYTHONPATH=. pytest -q tests/test_gloo_transport.py
```

It verifies a complete mapping containing FP32, FP16, empty BF16, and non-tensor
values, together with shape/dtype/value preservation and matching sender/receiver
payload byte counts. A restricted sandbox may block Gloo loopback sockets; run
the test in a shell that permits local sockets.

The test is a CPU/Gloo transport correctness gate. CUDA/ROCm end-to-end
correctness and TCP-versus-Gloo performance must be measured on the heterogeneous
GPU host with identical model, partition, batch, prompt, and output settings.
