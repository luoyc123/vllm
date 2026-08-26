#!/usr/bin/env python3
from __future__ import annotations

import importlib.metadata
import json
import subprocess
import sys

from packaging.requirements import Requirement
from packaging.version import Version


def package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "NOT_INSTALLED"


def main() -> int:
    versions = {
        name: package_version(name) for name in ("vllm", "torch", "transformers")
    }
    problems: list[str] = []
    if versions["vllm"] == "NOT_INSTALLED":
        problems.append("vllm is not installed")
    else:
        metadata = importlib.metadata.metadata("vllm")
        for raw in metadata.get_all("Requires-Dist") or []:
            requirement = Requirement(raw)
            if requirement.name.lower() == "transformers":
                installed = versions["transformers"]
                if (
                    installed == "NOT_INSTALLED"
                    or Version(installed) not in requirement.specifier
                ):
                    problems.append(
                        f"vllm requires {requirement}, but transformers=={installed}"
                    )

    dependency_check = subprocess.run(
        [sys.executable, "-m", "pip", "check"],
        text=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        check=False,
    )
    if dependency_check.returncode:
        problems.append("pip check reports broken dependencies")

    # Import the CLI/API modules without constructing EngineArgs. Building the
    # full parser asks vLLM to infer a device and fails legitimately on a host
    # with no visible GPU; preflight checks GPU visibility separately.
    cli_import = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import vllm; import vllm.entrypoints.cli.main; "
                "import vllm.entrypoints.openai.api_server"
            ),
        ],
        text=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        check=False,
    )
    if cli_import.returncode:
        last_line = (
            cli_import.stderr.strip().splitlines()[-1]
            if cli_import.stderr.strip()
            else "unknown"
        )
        problems.append(f"vllm CLI/API import failed: {last_line}")

    print(json.dumps({"versions": versions, "problems": problems}, indent=2))
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
