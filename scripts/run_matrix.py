from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the configured QuantLab backend matrix")
    parser.add_argument("config", type=Path, nargs="?", default=Path("configs/gpt2.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    args.output_dir.mkdir(parents=True, exist_ok=True)

    for backend in config["backends"]:
        command = [
            sys.executable,
            "-m",
            "quantlab.cli",
            "benchmark",
            "--model",
            config["model"],
            "--backend",
            backend,
            "--corpus",
            config["corpus"],
            "--prompts",
            config["prompts"],
            "--seed",
            str(config["seed"]),
            "--max-length",
            str(config["max_length"]),
            "--stride",
            str(config["stride"]),
            "--max-new-tokens",
            str(config["max_new_tokens"]),
            "--repeats",
            str(config["repeats"]),
            "--output",
            str(args.output_dir / f'{config["model"]}-{backend}.json'),
        ]
        print(f"Running {backend}...", flush=True)
        subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
