from __future__ import annotations

from dataclasses import asdict, dataclass
from math import log10
from typing import Any

import numpy as np
from numpy.typing import ArrayLike


@dataclass(frozen=True, slots=True)
class QuantizationMetrics:
    mse: float
    mean_absolute_error: float
    max_absolute_error: float
    cosine_similarity: float
    sqnr_db: float

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def compare_tensors(reference: ArrayLike, reconstructed: ArrayLike) -> QuantizationMetrics:
    original = np.asarray(reference, dtype=np.float64)
    restored = np.asarray(reconstructed, dtype=np.float64)
    if original.shape != restored.shape:
        raise ValueError("reference and reconstructed tensors must have the same shape")
    if original.size == 0:
        raise ValueError("tensors must not be empty")

    error = original - restored
    mse = float(np.mean(np.square(error)))
    mae = float(np.mean(np.abs(error)))
    max_error = float(np.max(np.abs(error)))
    denominator = float(np.linalg.norm(original) * np.linalg.norm(restored))
    cosine = 1.0 if denominator == 0 and np.array_equal(original, restored) else 0.0
    if denominator:
        cosine = float(np.dot(original.ravel(), restored.ravel()) / denominator)
    signal_power = float(np.mean(np.square(original)))
    if mse == 0:
        sqnr = float("inf")
    elif signal_power == 0:
        sqnr = float("-inf")
    else:
        sqnr = 10.0 * log10(signal_power / mse)
    return QuantizationMetrics(mse, mae, max_error, cosine, sqnr)
