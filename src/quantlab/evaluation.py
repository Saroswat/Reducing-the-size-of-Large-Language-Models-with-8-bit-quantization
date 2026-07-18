from __future__ import annotations

import math
import statistics
import time
from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class GenerationMetrics:
    mean_latency_seconds: float
    p95_latency_seconds: float
    output_tokens: int
    tokens_per_second: float

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _torch() -> Any:
    try:
        import torch
    except ImportError as error:
        message = 'Install model dependencies with pip install -e ".[transformers]"'
        raise RuntimeError(message) from error
    return torch


def _model_device(model: Any, torch: Any) -> Any:
    """Return a usable device for PyTorch and Optimum/OpenVINO wrappers."""
    device = getattr(model, "device", None)
    if device is not None:
        return device
    try:
        return next(model.parameters()).device
    except (AttributeError, StopIteration, TypeError):
        return torch.device("cpu")


def shared_corpus_perplexity(
    model: Any,
    tokenizer: Any,
    documents: list[str],
    *,
    max_length: int = 1024,
    stride: int = 512,
) -> float:
    """Evaluate one fixed token sequence with overlap counted exactly once."""
    if not documents or not all(document.strip() for document in documents):
        raise ValueError("documents must contain non-empty text")
    if stride < 1 or max_length < 2 or stride > max_length:
        raise ValueError("require 1 <= stride <= max_length and max_length >= 2")

    torch = _torch()
    separator = tokenizer.eos_token or "\n"
    encoded = tokenizer(separator.join(documents), return_tensors="pt")
    input_ids = encoded.input_ids
    model_device = _model_device(model, torch)
    negative_log_likelihoods = []
    evaluated_tokens = 0
    previous_end = 0

    model.eval()
    for begin in range(0, input_ids.size(1), stride):
        end = min(begin + max_length, input_ids.size(1))
        target_length = end - previous_end
        window = input_ids[:, begin:end].to(model_device)
        labels = window.clone()
        labels[:, :-target_length] = -100
        with torch.inference_mode():
            loss = model(window, labels=labels).loss
        predicted = int((labels[:, 1:] != -100).sum().item())
        if predicted:
            negative_log_likelihoods.append(loss * predicted)
            evaluated_tokens += predicted
        previous_end = end
        if end == input_ids.size(1):
            break

    if evaluated_tokens == 0:
        raise ValueError("corpus is too short to evaluate perplexity")
    total_loss = torch.stack(negative_log_likelihoods).sum() / evaluated_tokens
    return float(torch.exp(total_loss).item())


def benchmark_generation(
    model: Any,
    tokenizer: Any,
    prompts: list[str],
    *,
    max_new_tokens: int = 32,
    repeats: int = 3,
) -> GenerationMetrics:
    if not prompts or not all(prompt.strip() for prompt in prompts):
        raise ValueError("prompts must contain non-empty text")
    if max_new_tokens < 1 or repeats < 1:
        raise ValueError("max_new_tokens and repeats must be positive")

    torch = _torch()
    model_device = _model_device(model, torch)

    def synchronize() -> None:
        if torch.cuda.is_available() and model_device.type == "cuda":
            torch.cuda.synchronize(model_device)

    tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token
    warmup = tokenizer(prompts[0], return_tensors="pt").to(model_device)
    with torch.inference_mode():
        model.generate(
            **warmup,
            max_new_tokens=2,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )
    synchronize()

    latencies: list[float] = []
    output_tokens = 0
    for _ in range(repeats):
        for prompt in prompts:
            inputs = tokenizer(prompt, return_tensors="pt").to(model_device)
            synchronize()
            start = time.perf_counter()
            with torch.inference_mode():
                output = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id,
                )
            synchronize()
            latencies.append(time.perf_counter() - start)
            output_tokens += int(output.shape[-1] - inputs.input_ids.shape[-1])

    ordered = sorted(latencies)
    p95_index = min(math.ceil(len(ordered) * 0.95) - 1, len(ordered) - 1)
    elapsed = sum(latencies)
    return GenerationMetrics(
        mean_latency_seconds=statistics.fmean(latencies),
        p95_latency_seconds=ordered[p95_index],
        output_tokens=output_tokens,
        tokens_per_second=output_tokens / elapsed,
    )
