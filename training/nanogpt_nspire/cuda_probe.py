"""Run a bounded CUDA workload and record the observed training environment."""

from __future__ import annotations

import argparse
import math
from pathlib import Path
import platform
import subprocess
import time

import torch

from nanogpt_nspire.training_support import write_json_atomic


def _positive_integer(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _non_negative_integer(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return value


def _nvidia_driver_version() -> str | None:
    try:
        completed = subprocess.run(
            (
                "nvidia-smi",
                "--query-gpu=driver_version",
                "--format=csv,noheader",
            ),
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (FileNotFoundError, subprocess.SubprocessError):
        return None
    versions = {
        line.strip()
        for line in completed.stdout.splitlines()
        if line.strip()
    }
    if len(versions) != 1:
        return None
    return versions.pop()


def probe_cuda(
    *,
    require_cuda: bool = True,
    matrix_size: int = 4096,
    iterations: int = 5,
    seed: int = 20260728,
) -> dict[str, object]:
    """Measure one reproducible matrix workload without retaining tensors."""

    if not isinstance(require_cuda, bool):
        raise ValueError("require_cuda must be boolean")
    _positive_integer(matrix_size, "matrix_size")
    _positive_integer(iterations, "iterations")
    _non_negative_integer(seed, "seed")

    cuda_available = bool(torch.cuda.is_available())
    if require_cuda and not cuda_available:
        raise RuntimeError("CUDA is required but is not available")
    device = torch.device("cuda" if cuda_available else "cpu")

    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)

    left = torch.rand(
        (matrix_size, matrix_size),
        dtype=torch.float32,
        device=device,
    )
    right = torch.rand(
        (matrix_size, matrix_size),
        dtype=torch.float32,
        device=device,
    )

    warmup = left @ right
    checksum_tensor = warmup.sum()
    if device.type == "cuda":
        torch.cuda.synchronize(device)

    started = time.perf_counter()
    for _ in range(iterations):
        checksum_tensor = (left @ right).sum()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed_seconds = time.perf_counter() - started
    checksum = float(checksum_tensor.item())
    if not math.isfinite(checksum):
        raise RuntimeError("CUDA probe produced a non-finite checksum")

    operations = 2 * matrix_size**3 * iterations
    peak_memory = (
        int(torch.cuda.max_memory_allocated(device))
        if device.type == "cuda"
        else None
    )
    device_properties: dict[str, object] = {
        "available": cuda_available,
        "compute_capability": None,
        "driver_version": None,
        "name": platform.processor() or "cpu",
        "peak_memory_allocated_bytes": peak_memory,
        "total_memory_bytes": None,
        "type": device.type,
    }
    if device.type == "cuda":
        properties = torch.cuda.get_device_properties(device)
        device_properties.update(
            {
                "compute_capability": [
                    int(properties.major),
                    int(properties.minor),
                ],
                "driver_version": _nvidia_driver_version(),
                "name": str(properties.name),
                "total_memory_bytes": int(properties.total_memory),
            }
        )

    del warmup, checksum_tensor, left, right
    if device.type == "cuda":
        torch.cuda.empty_cache()

    return {
        "device": device_properties,
        "python": platform.python_version(),
        "result": {
            "checksum": checksum,
            "elapsed_seconds": elapsed_seconds,
            "operations": operations,
            "operations_per_second": operations / elapsed_seconds,
        },
        "schema_version": 1,
        "torch": {
            "cuda_runtime": torch.version.cuda,
            "version": str(torch.__version__),
        },
        "workload": {
            "dtype": "float32",
            "iterations": iterations,
            "matrix_size": matrix_size,
            "operation": "matrix_multiply_and_reduce",
            "seed": seed,
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--matrix-size", type=int, default=4096)
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260728)
    parser.add_argument(
        "--allow-cpu",
        action="store_true",
        help="run on CPU when CUDA is unavailable",
    )
    arguments = parser.parse_args(argv)
    report = probe_cuda(
        require_cuda=not arguments.allow_cpu,
        matrix_size=arguments.matrix_size,
        iterations=arguments.iterations,
        seed=arguments.seed,
    )
    write_json_atomic(arguments.output, report)
    print(arguments.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
