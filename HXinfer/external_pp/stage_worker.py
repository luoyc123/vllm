from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Protocol


class StageBackend(Protocol):
    """Future CUDA/ROCm adapters implement this device-neutral contract."""

    def forward(self, inputs: Mapping[str, Any]) -> Mapping[str, Any]: ...

    def close(self) -> None: ...


@dataclass(frozen=True)
class StageIdentity:
    stage_id: int
    stage_count: int
    device: str

    def __post_init__(self) -> None:
        if self.stage_count < 2:
            raise ValueError("external PP requires at least two stages")
        if not 0 <= self.stage_id < self.stage_count:
            raise ValueError("stage_id is outside stage_count")
