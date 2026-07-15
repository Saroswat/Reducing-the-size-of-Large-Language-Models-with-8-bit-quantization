import numpy as np
import pytest

from quantlab.quantization import affine_quantize, symmetric_quantize


@pytest.mark.parametrize("quantizer", [symmetric_quantize, affine_quantize])
def test_quantized_values_are_actually_int8(quantizer) -> None:
    result = quantizer(np.array([-2.0, -0.5, 0.0, 1.0, 3.0], dtype=np.float32))

    assert result.values.dtype == np.int8
    assert result.storage_bytes < 5 * np.dtype(np.float32).itemsize + 16


def test_symmetric_zero_tensor_round_trips() -> None:
    tensor = np.zeros((3, 4), dtype=np.float32)
    result = symmetric_quantize(tensor)

    np.testing.assert_array_equal(result.dequantize(), tensor)
    assert np.all(result.scale == 1)


def test_affine_constant_tensor_round_trips() -> None:
    tensor = np.full((2, 3), 2.5, dtype=np.float32)
    result = affine_quantize(tensor)

    np.testing.assert_allclose(result.dequantize(), tensor)


def test_per_channel_is_at_least_as_accurate_for_different_ranges() -> None:
    tensor = np.array([[0.01, -0.01, 0.005], [10.0, -8.0, 5.0]], dtype=np.float32)
    per_tensor = symmetric_quantize(tensor).dequantize()
    per_channel = symmetric_quantize(tensor, axis=0).dequantize()

    tensor_error = np.mean(np.square(tensor - per_tensor))
    channel_error = np.mean(np.square(tensor - per_channel))
    assert channel_error <= tensor_error


@pytest.mark.parametrize("axis", [2, -3])
def test_rejects_invalid_axis(axis: int) -> None:
    with pytest.raises(ValueError, match="axis"):
        symmetric_quantize(np.ones((2, 2)), axis=axis)


@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
def test_rejects_non_finite_values(value: float) -> None:
    with pytest.raises(ValueError, match="finite"):
        affine_quantize([0.0, value])


def test_rejects_empty_tensor() -> None:
    with pytest.raises(ValueError, match="empty"):
        symmetric_quantize([])
