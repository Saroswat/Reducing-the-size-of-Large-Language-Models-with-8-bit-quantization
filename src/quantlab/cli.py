from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

import numpy as np

from .backends import load_backend
from .evaluation import benchmark_generation, shared_corpus_perplexity
from .metrics import compare_tensors
from .quantization import affine_quantize, symmetric_quantize
from .reporting import BenchmarkReport, environment_snapshot, model_parameter_bytes


def _read_nonempty_lines(path: Path) -> list[str]:
    lines = [line.strip() for line in path.read_text(encoding="utf-8").splitlines()]
    return [line for line in lines if line]


def tensor_demo(args: argparse.Namespace) -> None:
    rng = np.random.default_rng(args.seed)
    tensor = rng.normal(0.0, 0.5, size=tuple(args.shape)).astype(np.float32)
    quantizer = symmetric_quantize if args.scheme == "symmetric" else affine_quantize
    quantized = quantizer(tensor, axis=args.axis)
    metrics = compare_tensors(tensor, quantized.dequantize())
    granularity = "per-tensor" if args.axis is None else f"per-channel(axis={args.axis})"
    payload = {
        "scheme": quantized.scheme,
        "granularity": granularity,
        "shape": list(tensor.shape),
        "storage_bytes": quantized.storage_bytes,
        "compression_ratio_vs_fp32": tensor.nbytes / quantized.storage_bytes,
        **metrics.as_dict(),
    }
    print(json.dumps(payload, indent=2))


def benchmark(args: argparse.Namespace) -> None:
    random.seed(args.seed)
    np.random.seed(args.seed)
    try:
        import torch
    except ImportError as error:
        raise SystemExit("Install the optional dependencies for the selected backend") from error
    torch.manual_seed(args.seed)

    loaded = load_backend(args.model, args.backend, device=args.device)
    documents = _read_nonempty_lines(args.corpus)
    prompts = _read_nonempty_lines(args.prompts)
    perplexity = shared_corpus_perplexity(
        loaded.model,
        loaded.tokenizer,
        documents,
        max_length=args.max_length,
        stride=args.stride,
    )
    generation = benchmark_generation(
        loaded.model,
        loaded.tokenizer,
        prompts,
        max_new_tokens=args.max_new_tokens,
        repeats=args.repeats,
    )
    footprint = None
    if hasattr(loaded.model, "get_memory_footprint"):
        footprint = int(loaded.model.get_memory_footprint())
    report = BenchmarkReport(
        model=args.model,
        backend=args.backend,
        device=loaded.device,
        seed=args.seed,
        parameter_bytes=model_parameter_bytes(loaded.model),
        model_footprint_bytes=footprint,
        perplexity=perplexity,
        generation=generation.as_dict(),
        settings={
            "max_length": args.max_length,
            "stride": args.stride,
            "max_new_tokens": args.max_new_tokens,
            "repeats": args.repeats,
            "corpus": str(args.corpus),
            "prompts": str(args.prompts),
        },
        environment=environment_snapshot(),
    )
    report.write(args.output)
    print(json.dumps(report.as_dict(), indent=2, sort_keys=True))


def compare_reports(args: argparse.Namespace) -> None:
    reports: list[dict[str, Any]] = []
    for path in args.reports:
        reports.append(json.loads(path.read_text(encoding="utf-8")))
    if not reports:
        raise SystemExit("Provide at least one report")
    baseline = reports[0]["perplexity"]
    columns = ["backend", "parameter_mib", "perplexity", "delta_ppl", "latency_s", "tokens_s"]
    print("\t".join(columns))
    for report in reports:
        row = [
            report["backend"],
            f'{report["parameter_bytes"] / 2**20:.2f}',
            f'{report["perplexity"]:.4f}',
            f'{report["perplexity"] - baseline:+.4f}',
            f'{report["generation"]["mean_latency_seconds"]:.4f}',
            f'{report["generation"]["tokens_per_second"]:.2f}',
        ]
        print("\t".join(row))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="quantlab")
    subparsers = parser.add_subparsers(dest="command", required=True)

    tensor = subparsers.add_parser("tensor-demo", help="measure tensor quantization error")
    tensor.add_argument("--shape", nargs="+", type=int, default=[256, 768])
    tensor.add_argument("--scheme", choices=["symmetric", "affine"], default="symmetric")
    tensor.add_argument("--axis", type=int)
    tensor.add_argument("--seed", type=int, default=7)
    tensor.set_defaults(handler=tensor_demo)

    model = subparsers.add_parser("benchmark", help="benchmark one model backend")
    model.add_argument("--model", default="gpt2")
    model.add_argument(
        "--backend",
        choices=["fp32", "fp16", "bnb-int8", "openvino-int8"],
        required=True,
    )
    model.add_argument("--device", default="auto")
    model.add_argument("--corpus", type=Path, required=True)
    model.add_argument("--prompts", type=Path, required=True)
    model.add_argument("--output", type=Path, required=True)
    model.add_argument("--seed", type=int, default=7)
    model.add_argument("--max-length", type=int, default=1024)
    model.add_argument("--stride", type=int, default=512)
    model.add_argument("--max-new-tokens", type=int, default=32)
    model.add_argument("--repeats", type=int, default=3)
    model.set_defaults(handler=benchmark)

    comparison = subparsers.add_parser("compare", help="compare generated JSON reports")
    comparison.add_argument("reports", type=Path, nargs="+")
    comparison.set_defaults(handler=compare_reports)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.handler(args)


if __name__ == "__main__":
    main()
