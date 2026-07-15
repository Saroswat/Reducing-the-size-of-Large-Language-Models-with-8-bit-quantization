from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

BackendName = Literal["fp32", "fp16", "bnb-int8", "openvino-int8"]


@dataclass(frozen=True, slots=True)
class LoadedBackend:
    model: Any
    tokenizer: Any
    name: BackendName
    device: str


def load_backend(model_id: str, backend: BackendName, *, device: str = "auto") -> LoadedBackend:
    """Load a backend lazily so the core package stays lightweight."""
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as error:
        raise RuntimeError("Install the optional dependencies for the selected backend") from error

    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    if backend == "openvino-int8":
        try:
            from optimum.intel import OVModelForCausalLM, OVWeightQuantizationConfig
        except ImportError as error:
            message = 'Install OpenVINO support with pip install -e ".[openvino]"'
            raise RuntimeError(message) from error
        quantization_config = OVWeightQuantizationConfig(bits=8)
        model = OVModelForCausalLM.from_pretrained(
            model_id,
            export=True,
            quantization_config=quantization_config,
            device=device.upper() if device != "auto" else "AUTO",
        )
        return LoadedBackend(model, tokenizer, backend, device)

    try:
        import torch
    except ImportError as error:
        message = 'Install Transformers support with pip install -e ".[transformers]"'
        raise RuntimeError(message) from error

    kwargs: dict[str, Any] = {}
    if backend == "fp32":
        kwargs["torch_dtype"] = torch.float32
    elif backend == "fp16":
        if not torch.cuda.is_available() and device in {"auto", "cpu"}:
            raise RuntimeError("FP16 benchmark requires a compatible accelerator")
        kwargs["torch_dtype"] = torch.float16
        kwargs["device_map"] = device
    elif backend == "bnb-int8":
        try:
            from transformers import BitsAndBytesConfig
        except ImportError as error:
            message = 'Install bitsandbytes support with pip install -e ".[bitsandbytes]"'
            raise RuntimeError(message) from error
        kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
        kwargs["device_map"] = device
    else:
        raise ValueError(f"unsupported backend: {backend}")

    model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
    if backend == "fp32" and device != "auto":
        model = model.to(device)
    return LoadedBackend(model, tokenizer, backend, str(next(model.parameters()).device))
