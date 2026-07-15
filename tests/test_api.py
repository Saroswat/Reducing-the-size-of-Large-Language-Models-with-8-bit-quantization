import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient

from quantlab.api import app

client = TestClient(app)


def test_health_endpoint() -> None:
    response = client.get("/api/health")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_tensor_endpoint_returns_real_compression_metrics() -> None:
    response = client.post(
        "/api/tensor",
        json={"rows": 16, "columns": 32, "scheme": "affine", "granularity": "per-row"},
    )

    assert response.status_code == 200
    report = response.json()
    assert report["int8_bytes"] < report["fp32_bytes"]
    assert report["compression_ratio"] > 1
    assert report["metrics"]["cosine_similarity"] > 0.99
    assert len(report["sample"]["original"]) == len(report["sample"]["quantized"])


def test_benchmark_rejects_invalid_sliding_window() -> None:
    response = client.post(
        "/api/benchmark",
        json={
            "corpus": ["A shared evaluation sentence."],
            "prompts": ["Quantization is"],
            "stride": 10,
            "max_length": 5,
        },
    )

    assert response.status_code == 422
    assert "stride" in response.json()["detail"]
