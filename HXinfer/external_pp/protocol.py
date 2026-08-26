from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass
from math import sqrt
from typing import Any


@dataclass(frozen=True)
class TensorSpec:
    """Device-independent description of one PP-boundary tensor."""

    name: str
    shape: tuple[int, ...]
    dtype: str
    nbytes: int

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("tensor name must not be empty")
        if any(dim < 0 for dim in self.shape):
            raise ValueError(f"negative dimension in {self.name}: {self.shape}")
        if self.nbytes < 0:
            raise ValueError("nbytes must be non-negative")


@dataclass(frozen=True)
class MessageMetadata:
    """Ordering and integrity fields shared by every transport backend."""

    request_id: str
    step_id: int
    source_stage: int
    destination_stage: int
    tensors: tuple[TensorSpec, ...]
    protocol_version: int = 1

    def __post_init__(self) -> None:
        if not self.request_id:
            raise ValueError("request_id must not be empty")
        if self.step_id < 0:
            raise ValueError("step_id must be non-negative")
        if self.destination_stage != self.source_stage + 1:
            raise ValueError("only adjacent forward PP stages are supported")
        names = [item.name for item in self.tensors]
        if len(names) != len(set(names)):
            raise ValueError("tensor names must be unique")

    @property
    def payload_nbytes(self) -> int:
        return sum(item.nbytes for item in self.tensors)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> MessageMetadata:
        tensors = tuple(
            TensorSpec(
                name=item["name"],
                shape=tuple(item["shape"]),
                dtype=item["dtype"],
                nbytes=int(item["nbytes"]),
            )
            for item in value["tensors"]
        )
        return cls(
            request_id=value["request_id"],
            step_id=int(value["step_id"]),
            source_stage=int(value["source_stage"]),
            destination_stage=int(value["destination_stage"]),
            tensors=tensors,
            protocol_version=int(value.get("protocol_version", 1)),
        )


def compare_tensors(reference: Any, candidate: Any) -> dict[str, float]:
    """Return the three correctness metrics required by the experiment plan.

    Inputs intentionally use a structural protocol rather than importing torch at
    module import time. NumPy arrays and torch tensors both satisfy the operations.
    """

    if tuple(reference.shape) != tuple(candidate.shape):
        raise ValueError(
            f"shape mismatch: reference={tuple(reference.shape)}, "
            f"candidate={tuple(candidate.shape)}"
        )
    diff = (candidate - reference).detach().float().cpu()
    ref = reference.detach().float().cpu()
    max_abs_error = float(diff.abs().max().item()) if diff.numel() else 0.0
    mean_abs_error = float(diff.abs().mean().item()) if diff.numel() else 0.0
    diff_l2 = sqrt(float((diff * diff).sum().item()))
    ref_l2 = sqrt(float((ref * ref).sum().item()))
    relative_l2 = diff_l2 / max(ref_l2, 1e-12)
    return {
        "max_abs_error": max_abs_error,
        "mean_abs_error": mean_abs_error,
        "relative_l2": relative_l2,
    }
