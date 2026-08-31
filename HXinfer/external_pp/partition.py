from __future__ import annotations


def normalize_layer_partition(
    value: str | None,
    *,
    pp_size: int = 2,
) -> str | None:
    """Validate and normalize a comma-separated PP layer partition."""
    if value is None or not value.strip():
        return None
    try:
        partitions = [int(item.strip()) for item in value.split(",")]
    except ValueError as error:
        raise ValueError(
            f"invalid layer partition {value!r}; expected comma-separated integers"
        ) from error
    if len(partitions) != pp_size:
        raise ValueError(
            f"layer partition must contain {pp_size} entries; got {partitions!r}"
        )
    if any(partition <= 0 for partition in partitions):
        raise ValueError(
            f"layer partition entries must be positive; got {partitions!r}"
        )
    return ",".join(str(partition) for partition in partitions)


def validate_worker_partitions(
    controller_partition: str | None,
    worker_statuses: list[dict],
) -> None:
    """Require the Controller and all pre-launched workers to use one layout."""
    expected = normalize_layer_partition(controller_partition)
    observed = [
        normalize_layer_partition(status.get("layer_partition"))
        for status in worker_statuses
    ]
    if any(partition != expected for partition in observed):
        raise RuntimeError(
            "VLLM_PP_LAYER_PARTITION mismatch: "
            f"controller={expected!r}, workers={observed!r}"
        )
