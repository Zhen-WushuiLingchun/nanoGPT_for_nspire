"""Measure the formal packed-W4/dynamic-A8 deployment reference."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

import torch

from nanogpt_nspire.training_dataset import load_token_dataset
from nanogpt_nspire.training_support import (
    bits_per_character,
    dataset_summary,
    environment_summary,
    evaluate_loss,
    resolve_device,
    sha256_file,
    write_json_atomic,
)
from nanogpt_nspire.w4a8_reference import W4A8Reference


W4A32_VALIDATION_LOSS = 1.4737991189956665
DIRECT_SMALL_VALIDATION_LOSS = 1.4997899746894836
MAXIMUM_W4A8_EXTRA_DEGRADATION = 0.02


def evaluate_w4a8(
    *,
    model_path: Path,
    data_dir: Path,
    device_name: str,
    batch_size: int = 64,
    batches: int = 50,
    validation_seed: int = 1338,
) -> dict[str, object]:
    """Evaluate the same frozen validation windows as Lessons 05 and 07."""

    if batch_size <= 0 or batches <= 0:
        raise ValueError("batch_size and batches must be positive")
    device = resolve_device(device_name)
    dataset = load_token_dataset(data_dir)
    model = W4A8Reference.from_file(model_path)
    if tuple(dataset.vocabulary) != model.parsed.vocabulary:
        raise ValueError("model and dataset vocabularies disagree")
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    model.to(device).eval()
    loss = evaluate_loss(
        model,
        dataset.validation,
        batch_size=batch_size,
        block_size=model.block_size,
        batches=batches,
        seed=validation_seed,
        device=device,
    )
    extra_degradation = loss - W4A32_VALIDATION_LOSS
    extra_gate_passed = (
        extra_degradation <= MAXIMUM_W4A8_EXTRA_DEGRADATION
    )
    direct_gate_passed = loss < DIRECT_SMALL_VALIDATION_LOSS
    peak_cuda_bytes = (
        int(torch.cuda.max_memory_allocated(device))
        if device.type == "cuda"
        else None
    )
    return {
        "schema_version": 1,
        "experiment_id": "lesson08-packed-w4a8-quality",
        "model": {
            "path": str(model_path),
            "bytes": model_path.stat().st_size,
            "sha256": sha256_file(model_path),
            "storage": "packed_int4",
            "activation": "dynamic_groupwise_int8",
            "accumulator": "int32",
            "fp32_matrix_expansion": False,
        },
        "dataset": dataset_summary(dataset),
        "evaluation": {
            "batch_size": batch_size,
            "batches": batches,
            "block_size": model.block_size,
            "validation_seed": validation_seed,
            "windows_identical_to_lesson07": True,
        },
        "metrics": {
            "validation_loss": loss,
            "validation_bpc": bits_per_character(loss),
            "w4a32_validation_loss": W4A32_VALIDATION_LOSS,
            "extra_degradation_over_w4a32": extra_degradation,
            "maximum_extra_degradation": (
                MAXIMUM_W4A8_EXTRA_DEGRADATION
            ),
            "extra_degradation_gate_passed": extra_gate_passed,
            "direct_small_validation_loss": DIRECT_SMALL_VALIDATION_LOSS,
            "better_than_direct_small": direct_gate_passed,
            "quality_gate_passed": (
                extra_gate_passed and direct_gate_passed
            ),
        },
        "environment": environment_summary(
            device,
            peak_cuda_memory_allocated_bytes=peak_cuda_bytes,
        ),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args(argv)
    result = evaluate_w4a8(
        model_path=arguments.model,
        data_dir=arguments.data_dir,
        device_name=arguments.device,
    )
    write_json_atomic(arguments.output, result)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    passed = result["metrics"]["quality_gate_passed"]
    assert isinstance(passed, bool)
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
