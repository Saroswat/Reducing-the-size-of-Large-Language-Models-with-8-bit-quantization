from __future__ import annotations

from functools import lru_cache
from typing import Any, Literal

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from .backends import BackendName, LoadedBackend, load_backend
from .evaluation import benchmark_generation, shared_corpus_perplexity
from .metrics import compare_tensors
from .quantization import affine_quantize, symmetric_quantize
from .reporting import environment_snapshot, model_parameter_bytes


class TensorRequest(BaseModel):
    rows: int = Field(default=64, ge=1, le=2048)
    columns: int = Field(default=128, ge=1, le=2048)
    scheme: Literal["symmetric", "affine"] = "symmetric"
    granularity: Literal["per-tensor", "per-row"] = "per-row"
    seed: int = 7
    standard_deviation: float = Field(default=0.5, gt=0, le=100)


class GenerateRequest(BaseModel):
    model: str = Field(default="gpt2", min_length=1)
    backend: BackendName = "fp32"
    device: str = "auto"
    prompt: str = Field(min_length=1, max_length=20_000)
    max_new_tokens: int = Field(default=48, ge=1, le=512)


class BenchmarkRequest(BaseModel):
    model: str = Field(default="gpt2", min_length=1)
    backend: BackendName = "fp32"
    device: str = "auto"
    corpus: list[str] = Field(min_length=1)
    prompts: list[str] = Field(min_length=1)
    max_length: int = Field(default=1024, ge=2, le=32768)
    stride: int = Field(default=512, ge=1, le=32768)
    max_new_tokens: int = Field(default=32, ge=1, le=512)
    repeats: int = Field(default=3, ge=1, le=20)


app = FastAPI(
    title="QuantLab API",
    version="0.2.0",
    description="Local API for reproducible INT8 quantization experiments.",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://127.0.0.1:5173", "http://localhost:5173"],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


@lru_cache(maxsize=3)
def _cached_backend(model: str, backend: BackendName, device: str) -> LoadedBackend:
    return load_backend(model, backend, device=device)


def _inference_device(loaded: LoadedBackend, torch: Any) -> Any:
    if loaded.name == "openvino-int8":
        return torch.device("cpu")
    device = getattr(loaded.model, "device", None)
    if device is not None:
        return device
    try:
        return next(loaded.model.parameters()).device
    except (AttributeError, StopIteration, TypeError):
        return torch.device("cpu")


@app.get("/api/health")
def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "service": "quantlab",
        "cached_models": _cached_backend.cache_info().currsize,
        "environment": environment_snapshot(),
    }


@app.get("/api/backends")
def backends() -> dict[str, Any]:
    return {
        "backends": [
            {"id": "fp32", "label": "FP32", "target": "CPU / GPU"},
            {"id": "fp16", "label": "FP16", "target": "Accelerator"},
            {"id": "bnb-int8", "label": "bitsandbytes INT8", "target": "NVIDIA GPU"},
            {"id": "openvino-int8", "label": "OpenVINO INT8", "target": "Intel CPU / GPU"},
        ]
    }


@app.post("/api/tensor")
def tensor_experiment(request: TensorRequest) -> dict[str, Any]:
    rng = np.random.default_rng(request.seed)
    tensor = rng.normal(
        0.0,
        request.standard_deviation,
        size=(request.rows, request.columns),
    ).astype(np.float32)
    quantizer = symmetric_quantize if request.scheme == "symmetric" else affine_quantize
    axis = 0 if request.granularity == "per-row" else None
    quantized = quantizer(tensor, axis=axis)
    reconstructed = quantized.dequantize()
    metrics = compare_tensors(tensor, reconstructed)
    sample_count = min(96, tensor.size)
    indices = np.linspace(0, tensor.size - 1, sample_count, dtype=int)
    original_sample = tensor.ravel()[indices]
    reconstructed_sample = reconstructed.ravel()[indices]
    return {
        "shape": [request.rows, request.columns],
        "scheme": request.scheme,
        "granularity": request.granularity,
        "fp32_bytes": tensor.nbytes,
        "int8_bytes": quantized.storage_bytes,
        "compression_ratio": tensor.nbytes / quantized.storage_bytes,
        "metrics": metrics.as_dict(),
        "sample": {
            "original": original_sample.tolist(),
            "quantized": reconstructed_sample.tolist(),
        },
    }


@app.post("/api/generate")
def generate(request: GenerateRequest) -> dict[str, Any]:
    try:
        import torch

        loaded = _cached_backend(request.model, request.backend, request.device)
        device = _inference_device(loaded, torch)
        loaded.tokenizer.pad_token = loaded.tokenizer.pad_token or loaded.tokenizer.eos_token
        inputs = loaded.tokenizer(request.prompt, return_tensors="pt").to(device)
        with torch.inference_mode():
            output = loaded.model.generate(
                **inputs,
                max_new_tokens=request.max_new_tokens,
                do_sample=False,
                pad_token_id=loaded.tokenizer.pad_token_id,
            )
        generated = output[0, inputs.input_ids.shape[-1] :]
        return {
            "text": loaded.tokenizer.decode(generated, skip_special_tokens=True),
            "generated_tokens": int(generated.shape[-1]),
            "model": request.model,
            "backend": request.backend,
            "device": loaded.device,
        }
    except (ImportError, RuntimeError, ValueError, OSError) as error:
        raise HTTPException(status_code=422, detail=str(error)) from error


@app.post("/api/benchmark")
def benchmark(request: BenchmarkRequest) -> dict[str, Any]:
    if request.stride > request.max_length:
        raise HTTPException(status_code=422, detail="stride cannot exceed max_length")
    if not all(text.strip() for text in [*request.corpus, *request.prompts]):
        raise HTTPException(status_code=422, detail="corpus and prompts cannot contain blank text")
    try:
        loaded = _cached_backend(request.model, request.backend, request.device)
        perplexity = shared_corpus_perplexity(
            loaded.model,
            loaded.tokenizer,
            request.corpus,
            max_length=request.max_length,
            stride=request.stride,
        )
        generation = benchmark_generation(
            loaded.model,
            loaded.tokenizer,
            request.prompts,
            max_new_tokens=request.max_new_tokens,
            repeats=request.repeats,
        )
        footprint = None
        if hasattr(loaded.model, "get_memory_footprint"):
            footprint = int(loaded.model.get_memory_footprint())
        return {
            "model": request.model,
            "backend": request.backend,
            "device": loaded.device,
            "parameter_bytes": model_parameter_bytes(loaded.model),
            "model_footprint_bytes": footprint,
            "perplexity": perplexity,
            "generation": generation.as_dict(),
            "environment": environment_snapshot(),
        }
    except (ImportError, RuntimeError, ValueError, OSError) as error:
        raise HTTPException(status_code=422, detail=str(error)) from error


def main() -> None:
    try:
        import uvicorn
    except ImportError as error:
        raise SystemExit('Install the web dependencies with pip install -e ".[web]"') from error
    uvicorn.run("quantlab.api:app", host="127.0.0.1", port=8000, reload=False)


if __name__ == "__main__":
    main()
