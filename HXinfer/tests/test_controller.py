from pathlib import Path

import pytest

from external_pp.controller import PhaseGateError, require_native_baseline


def test_native_gate_is_closed_by_default(tmp_path: Path):
    with pytest.raises(PhaseGateError, match="has not passed"):
        require_native_baseline(tmp_path)


def test_native_gate_accepts_receipt(tmp_path: Path):
    receipt = tmp_path / "results" / "native_pp" / "PASS"
    receipt.parent.mkdir(parents=True)
    receipt.write_text("run-1\n")
    assert require_native_baseline(tmp_path) == receipt
