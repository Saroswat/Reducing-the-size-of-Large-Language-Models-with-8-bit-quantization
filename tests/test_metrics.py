import math

import numpy as np
import pytest

from quantlab.metrics import compare_tensors


def test_identical_tensors_have_perfect_metrics() -> None:
    metrics = compare_tensors([1.0, 2.0], [1.0, 2.0])

    assert metrics.mse == 0
    assert metrics.cosine_similarity == pytest.approx(1.0)
    assert math.isinf(metrics.sqnr_db) and metrics.sqnr_db > 0


def test_known_error_metrics() -> None:
    metrics = compare_tensors(np.array([0.0, 2.0]), np.array([0.0, 1.0]))

    assert metrics.mse == pytest.approx(0.5)
    assert metrics.mean_absolute_error == pytest.approx(0.5)
    assert metrics.max_absolute_error == pytest.approx(1.0)


def test_rejects_shape_mismatch() -> None:
    with pytest.raises(ValueError, match="same shape"):
        compare_tensors([1.0], [1.0, 2.0])
