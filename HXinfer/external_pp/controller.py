from __future__ import annotations

from pathlib import Path


class PhaseGateError(RuntimeError):
    pass


def require_native_baseline(project_root: Path) -> Path:
    """Refuse External PP until a human-reviewed Native PP gate exists."""

    receipt = project_root / "results" / "native_pp" / "PASS"
    if not receipt.is_file():
        raise PhaseGateError(
            "Native PP baseline has not passed. Review results/native_pp first, "
            "then create results/native_pp/PASS with the tested run id."
        )
    return receipt


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    try:
        require_native_baseline(root)
    except PhaseGateError as error:
        raise SystemExit(str(error)) from error
    print(
        f"Native PP gate accepted: {require_native_baseline(root).read_text().strip()}"
    )


if __name__ == "__main__":
    main()
