from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from nanogpt_nspire.alignment import compare_probes, host_greedy_probe
from nanogpt_nspire.export_format import (
    ACTIVATION_DYNAMIC_INT8_GROUPWISE,
    MODEL_STORAGE_W4A8,
    STORAGE_FP32,
    STORAGE_INT4_GROUPWISE,
    ModelSpec,
    TensorPayload,
    build_model_file,
)
from nanogpt_nspire.quantization import quantize_groupwise_int4
from nanogpt_nspire.w4a8_reference import (
    W4A8Reference,
    W4Tensor,
    dynamic_quantize_int8,
    w4a8_greedy_probe,
    w4a8_matvec,
)
from tools.generate_test_model_fixture import fixture_tensors


def test_dynamic_activation_uses_ties_to_even_and_zero_group_scale() -> None:
    values = torch.tensor([[0.5, -1.0, 0.0, 0.0]])
    quantized, scales = dynamic_quantize_int8(values, group_size=2)
    assert quantized.tolist() == [[[64, -127], [0, 0]]]
    torch.testing.assert_close(
        scales,
        torch.tensor([[1.0 / 127.0, 1.0]]),
        rtol=0.0,
        atol=1.0e-8,
    )


def test_w4a8_matvec_matches_hand_computed_int32_groups() -> None:
    weight = W4Tensor(
        values=torch.tensor(
            [[1, -2, 3, 0], [-7, 7, 1, -1]],
            dtype=torch.int8,
        ),
        scales=torch.tensor([[0.5, 0.25], [0.1, 2.0]]),
        rows=2,
        columns=4,
        padded_columns=4,
        group_size=2,
    )
    output = w4a8_matvec(
        torch.tensor([0.5, -1.0, 0.0, 2.0]),
        weight,
    )
    torch.testing.assert_close(
        output,
        torch.tensor(
            [159.0 / 127.0, -4.0 - 133.7 / 127.0]
        ),
        rtol=0.0,
        atol=2.0e-6,
    )


def _tiny_w4a8(path: Path) -> None:
    payloads = []
    for tensor in fixture_tensors():
        if len(tensor.shape) == 2:
            values = torch.from_numpy(
                np.frombuffer(tensor.data, dtype="<f4")
                .reshape(tensor.shape)
                .copy()
            )
            quantized = quantize_groupwise_int4(values, group_size=2)
            payloads.append(
                TensorPayload(
                    tensor_id=tensor.tensor_id,
                    storage=STORAGE_INT4_GROUPWISE,
                    shape=tensor.shape,
                    data=quantized.packed.numpy().tobytes(),
                    auxiliary=quantized.scales.numpy().tobytes(),
                    group_size=2,
                    padded_last_dim=quantized.padded_last_dim,
                )
            )
        else:
            payloads.append(
                TensorPayload(
                    tensor_id=tensor.tensor_id,
                    storage=STORAGE_FP32,
                    shape=tensor.shape,
                    data=bytes(tensor.data),
                )
            )
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
                model_storage=MODEL_STORAGE_W4A8,
                weight_group_size=2,
                activation_quantization=(
                    ACTIVATION_DYNAMIC_INT8_GROUPWISE
                ),
                activation_group_size=2,
            ),
            vocabulary=("\n", "a", "é"),
            tensors=payloads,
        )
    )


def test_incremental_w4a8_c_matches_python_reference(
    tmp_path: Path,
) -> None:
    model_path = tmp_path / "tiny-w4a8.ngm"
    _tiny_w4a8(model_path)
    model = W4A8Reference.from_file(model_path)
    inputs = torch.tensor([[0, 1, 2]], dtype=torch.long)
    targets = torch.tensor([[1, 2, 0]], dtype=torch.long)
    full_logits, loss = model(inputs, targets)
    assert full_logits.shape == (1, 3, 3)
    assert loss is not None and torch.isfinite(loss)
    reference = w4a8_greedy_probe(model, (0, 1, 2), 1)
    runner = (
        Path(__file__).parents[2]
        / "build"
        / "host"
        / "Release"
        / "run_model.exe"
    )
    if not runner.is_file():
        pytest.skip("Host C runner has not been built")
    candidate = host_greedy_probe(
        runner,
        model_path,
        (0, 1, 2),
        1,
    )
    comparison = compare_probes(reference, candidate)
    assert comparison["pass"] is True
    assert reference.generated_tokens == candidate.generated_tokens
