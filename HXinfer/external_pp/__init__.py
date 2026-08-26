"""HXinfer external pipeline-parallel contracts."""

from .protocol import MessageMetadata, TensorSpec, compare_tensors

__all__ = ["MessageMetadata", "TensorSpec", "compare_tensors"]
