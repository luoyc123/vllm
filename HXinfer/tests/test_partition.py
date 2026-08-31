import pytest

from external_pp.partition import (
    normalize_layer_partition,
    validate_worker_partitions,
)


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (None, None),
        ("", None),
        ("18,18", "18,18"),
        (" 24, 12 ", "24,12"),
        ("11,25", "11,25"),
        ("10,26", "10,26"),
    ],
)
def test_normalize_layer_partition(value, expected):
    assert normalize_layer_partition(value) == expected


@pytest.mark.parametrize("value", ["24", "24,12,0", "24,a", "24,-12", "0,36"])
def test_normalize_layer_partition_rejects_invalid_values(value):
    with pytest.raises(ValueError):
        normalize_layer_partition(value)


def test_validate_worker_partitions_accepts_identical_asymmetric_layout():
    validate_worker_partitions(
        "24,12",
        [
            {"rank": 0, "layer_partition": "24,12"},
            {"rank": 1, "layer_partition": "24,12"},
        ],
    )


def test_validate_worker_partitions_rejects_mismatch():
    with pytest.raises(RuntimeError, match="mismatch"):
        validate_worker_partitions(
            "24,12",
            [
                {"rank": 0, "layer_partition": "24,12"},
                {"rank": 1, "layer_partition": "18,18"},
            ],
        )
