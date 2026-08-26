from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from external_pp.protocol import MessageMetadata


@dataclass(frozen=True)
class ReceivedMessage:
    tensors: Mapping[str, Any]
    metadata: MessageMetadata
    timings_ms: Mapping[str, float]


class PPTransport(ABC):
    """Transport boundary that must not expose NCCL ranks or CUDA handles."""

    @abstractmethod
    def send(self, tensors: Mapping[str, Any], metadata: MessageMetadata) -> None:
        """Send a complete IntermediateTensors mapping and its ordering data."""

    @abstractmethod
    def recv(self) -> ReceivedMessage:
        """Receive exactly one complete, ordered PP message."""

    @abstractmethod
    def close(self) -> None:
        """Release transport-owned resources."""
