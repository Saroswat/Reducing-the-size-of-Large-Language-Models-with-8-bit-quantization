from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray

Granularity = Literal["per-tensor", "per-channel"]
Scheme = Literal["symmetric", "affine"]


@dataclass(frozen=True, slots=True)
class QuantizedTensor:
    """A real INT8 representation plus the metadata required to reconstruct it."""

    values: NDArray[np.int8]
    scale: NDArray[np.float32]
    zero_point: NDArray[np.int16]
    scheme: Scheme
    axis: int | None
    original_dtype: str

    @property
    def granularity(self) -> Granularity:
        return "per-tensor" if self.axis is None else "per-channel"

    @property
    def storage_bytes(self) -> int:
        return int(self.values.nbytes + self.scale.nbytes + self.zero_point.nbytes)

    def dequantize(self) -> NDArray[np.float32]:
        values = self.values.astype(np.float32)
        return (values - self.zero_point.astype(np.float32)) * self.scale


def _as_float_array(tensor: ArrayLike) -> NDArray[np.float32]:
    array = np.asarray(tensor, dtype=np.float32)
    if array.size == 0:
        raise ValueError("tensor must not be empty")
    if not np.all(np.isfinite(array)):
        raise ValueError("tensor must contain only finite values")
    return array


def _normalize_axis(axis: int | None, ndim: int) -> int | None:
    if axis is None:
        return None
    normalized = axis if axis >= 0 else axis + ndim
    if normalized < 0 or normalized >= ndim:
        raise ValueError(f"axis {axis} is invalid for a {ndim}D tensor")
    return normalized


def _reduction_axes(ndim: int, channel_axis: int | None) -> tuple[int, ...] | None:
    if channel_axis is None:
        return None
    return tuple(index for index in range(ndim) if index != channel_axis)


def symmetric_quantize(tensor: ArrayLike, *, axis: int | None = None) -> QuantizedTensor:
    """Signed symmetric INT8 quantization with an optional per-channel axis."""
    array = _as_float_array(tensor)
    axis = _normalize_axis(axis, array.ndim)
    reduce = _reduction_axes(array.ndim, axis)
    max_abs = np.max(np.abs(array), axis=reduce, keepdims=True)
    scale = np.where(max_abs == 0, 1.0, max_abs / 127.0).astype(np.float32)
    values = np.clip(np.rint(array / scale), -127, 127).astype(np.int8)
    zero_point = np.zeros_like(scale, dtype=np.int16)
    return QuantizedTensor(values, scale, zero_point, "symmetric", axis, str(array.dtype))


def affine_quantize(tensor: ArrayLike, *, axis: int | None = None) -> QuantizedTensor:
    """Signed asymmetric INT8 quantization with an optional per-channel axis."""
    array = _as_float_array(tensor)
    axis = _normalize_axis(axis, array.ndim)
    reduce = _reduction_axes(array.ndim, axis)
    minimum = np.min(array, axis=reduce, keepdims=True)
    maximum = np.max(array, axis=reduce, keepdims=True)
    dynamic_range = maximum - minimum
    constant = dynamic_range == 0
    max_abs = np.maximum(np.abs(minimum), np.abs(maximum))
    fallback_scale = np.where(max_abs == 0, 1.0, max_abs / 127.0)
    scale = np.where(constant, fallback_scale, dynamic_range / 255.0).astype(np.float32)
    zero_point = np.where(
        constant,
        0,
        np.clip(np.rint(-128.0 - minimum / scale), -128, 127),
    ).astype(np.int16)
    values = np.clip(np.rint(array / scale + zero_point), -128, 127).astype(np.int8)
    return QuantizedTensor(values, scale, zero_point, "affine", axis, str(array.dtype))
