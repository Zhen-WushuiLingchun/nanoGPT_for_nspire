import math

import pytest
import torch

from nanogpt_nspire.cuda_probe import probe_cuda


def test_cpu_probe_is_bounded_and_json_serializable() -> None:
    report = probe_cuda(
        require_cuda=False,
        matrix_size=8,
        iterations=1,
        seed=17,
    )

    assert report["schema_version"] == 1
    assert report["device"]["type"] in {"cpu", "cuda"}
    assert report["workload"] == {
        "dtype": "float32",
        "iterations": 1,
        "matrix_size": 8,
        "operation": "matrix_multiply_and_reduce",
        "seed": 17,
    }
    assert math.isfinite(report["result"]["checksum"])
    assert report["result"]["elapsed_seconds"] > 0.0
    assert report["result"]["operations_per_second"] > 0.0
    assert set(report) == {
        "device",
        "python",
        "result",
        "schema_version",
        "torch",
        "workload",
    }


def test_required_cuda_fails_closed_when_unavailable(monkeypatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    with pytest.raises(RuntimeError, match="CUDA is required"):
        probe_cuda(
            require_cuda=True,
            matrix_size=8,
            iterations=1,
        )


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("matrix_size", True),
        ("matrix_size", 0),
        ("iterations", -1),
        ("seed", -1),
    ),
)
def test_probe_rejects_invalid_integer_arguments(field, value) -> None:
    arguments = {
        "require_cuda": False,
        "matrix_size": 8,
        "iterations": 1,
        "seed": 17,
    }
    arguments[field] = value

    with pytest.raises(ValueError):
        probe_cuda(**arguments)
