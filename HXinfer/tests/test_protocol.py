import pytest

from external_pp.protocol import MessageMetadata, TensorSpec
from scripts.inspect_partition import pp_indices


def test_metadata_round_trip_and_payload_size():
    message = MessageMetadata(
        request_id="req-1",
        step_id=0,
        source_stage=0,
        destination_stage=1,
        tensors=(
            TensorSpec("hidden_states", (8, 4096), "float16", 65536),
            TensorSpec("residual", (8, 4096), "float16", 65536),
        ),
    )
    assert MessageMetadata.from_dict(message.to_dict()) == message
    assert message.payload_nbytes == 131072


def test_metadata_rejects_non_adjacent_stage():
    with pytest.raises(ValueError, match="adjacent"):
        MessageMetadata("req", 0, 0, 2, ())


def test_metadata_rejects_duplicate_tensor_names():
    spec = TensorSpec("hidden_states", (1,), "float32", 4)
    with pytest.raises(ValueError, match="unique"):
        MessageMetadata("req", 0, 0, 1, (spec, spec))


def test_partition_matches_vllm_uneven_policy(monkeypatch):
    monkeypatch.delenv("VLLM_PP_LAYER_PARTITION", raising=False)
    assert [pp_indices(10, rank, 4) for rank in range(4)] == [
        (0, 2),
        (2, 5),
        (5, 8),
        (8, 10),
    ]
