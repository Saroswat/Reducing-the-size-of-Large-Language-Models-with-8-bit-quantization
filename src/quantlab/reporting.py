from __future__ import annotations

import json
import platform
import sys
from dataclasses import asdict, dataclass
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any


def package_version(name: str) -> str | None:
    try:
        return version(name)
    except PackageNotFoundError:
        return None


def model_parameter_bytes(model: Any) -> int:
    try:
        return int(
            sum(parameter.numel() * parameter.element_size() for parameter in model.parameters())
        )
    except (AttributeError, TypeError):
        # Compiled OpenVINO wrappers do not necessarily expose torch parameters.
        return 0


@dataclass(frozen=True, slots=True)
class BenchmarkReport:
    model: str
    backend: str
    device: str
    seed: int
    parameter_bytes: int
    model_footprint_bytes: int | None
    perplexity: float
    generation: dict[str, Any]
    settings: dict[str, Any]
    environment: dict[str, Any]

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)

    def write(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.as_dict(), indent=2, sort_keys=True), encoding="utf-8")


def environment_snapshot() -> dict[str, Any]:
    packages = ["torch", "transformers", "accelerate", "bitsandbytes", "openvino", "optimum-intel"]
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "packages": {name: package_version(name) for name in packages},
    }
