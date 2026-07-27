from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

from nanogpt_nspire.alignment import (
    compare_probes,
    host_greedy_probe,
    load_fp32_export,
    torch_greedy_probe,
)
from nanogpt_nspire.export_format import (
    ACTIVATION_NONE,
    MODEL_STORAGE_FP32,
    ModelSpec,
    build_model_file,
)
from tools.generate_test_model_fixture import fixture_tensors


def _write_tiny_model(path: Path) -> None:
    path.write_bytes(
        build_model_file(
            spec=ModelSpec(
                vocab_size=3,
                block_size=4,
                n_layer=1,
                n_head=2,
                n_embd=4,
                mlp_ratio=2,
                tie_embeddings=True,
                bias=False,
                model_storage=MODEL_STORAGE_FP32,
                weight_group_size=0,
                activation_quantization=ACTIVATION_NONE,
                activation_group_size=0,
            ),
            vocabulary=("\n", "a", "é"),
            tensors=fixture_tensors(),
        )
    )


def test_fp32_export_reconstructs_full_prefix_reference(
    tmp_path: Path,
) -> None:
    model_path = tmp_path / "tiny.ngm"
    _write_tiny_model(model_path)
    model, parsed = load_fp32_export(model_path)
    result = torch_greedy_probe(model, (0, 1, 2), 1)
    assert parsed.vocabulary == ("\n", "a", "é")
    assert result.generated_tokens == (2,)
    np.testing.assert_allclose(
        result.logits,
        [-0.0057609, 0.05852218, 0.12280525],
        rtol=0.0,
        atol=2.0e-5,
    )


def test_host_c_matches_torch_when_runner_is_built(
    tmp_path: Path,
) -> None:
    default_runner = (
        Path(__file__).parents[2]
        / "build"
        / "host"
        / "Release"
        / "run_model.exe"
    )
    runner = Path(
        os.environ.get("NANOGPT_HOST_RUNNER", default_runner)
    )
    if not runner.is_file():
        pytest.skip("Host C runner has not been built")
    model_path = tmp_path / "tiny.ngm"
    _write_tiny_model(model_path)
    model, _ = load_fp32_export(model_path)
    reference = torch_greedy_probe(model, (0, 1, 2), 1)
    candidate = host_greedy_probe(
        runner,
        model_path,
        (0, 1, 2),
        1,
    )
    comparison = compare_probes(reference, candidate)
    assert comparison["pass"] is True
