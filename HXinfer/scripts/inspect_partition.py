#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os


def pp_indices(num_layers: int, rank: int, world_size: int) -> tuple[int, int]:
    override = os.environ.get("VLLM_PP_LAYER_PARTITION")
    if override:
        partitions = [int(value) for value in override.split(",")]
        if len(partitions) != world_size or sum(partitions) != num_layers:
            raise ValueError(
                "VLLM_PP_LAYER_PARTITION must have pp_size entries summing to "
                "num_hidden_layers"
            )
    else:
        base, extra = divmod(num_layers, world_size)
        partitions = [base] * world_size
        # Mirrors vLLM 0.27.1 get_pp_indices: keep the last stage lighter because
        # it also owns final norm/output work; with PP>2 this favors middle ranks.
        for offset in range(2, extra + 2):
            partitions[-offset] += 1
    start = sum(partitions[:rank])
    end = start + partitions[rank]
    return start, end


def main() -> None:
    from transformers import AutoConfig

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--pp-size", type=int, default=2)
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    config = AutoConfig.from_pretrained(
        args.model, trust_remote_code=args.trust_remote_code
    )
    num_layers = getattr(config, "num_hidden_layers", None)
    if num_layers is None:
        raise SystemExit("model config has no num_hidden_layers")
    stages = []
    for rank in range(args.pp_size):
        start, end = pp_indices(num_layers, rank, args.pp_size)
        stages.append(
            {
                "pp_rank": rank,
                "start_layer": start,
                "end_layer_exclusive": end,
                "layer_count": end - start,
            }
        )
    value = {
        "model": args.model,
        "model_type": getattr(config, "model_type", None),
        "hidden_size": getattr(config, "hidden_size", None),
        "num_hidden_layers": num_layers,
        "stages": stages,
        "note": "Derived with vLLM get_pp_indices policy; confirm against worker logs.",
    }
    with open(args.output, "w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


if __name__ == "__main__":
    main()
