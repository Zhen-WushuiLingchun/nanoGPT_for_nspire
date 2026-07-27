from copy import deepcopy

import pytest
import torch

from nanogpt_nspire.models.direct_small_gpt import (
    DirectSmallConfig,
    DirectSmallGPT,
)
from nanogpt_nspire.quantization.model_state import (
    dequantize_model_state,
    quantize_model_state,
    reconstruct_dequantized_reference,
)


def _tiny_model() -> DirectSmallGPT:
    torch.manual_seed(17)
    return DirectSmallGPT(
        DirectSmallConfig(
            vocab_size=17,
            block_size=8,
            n_layer=2,
            n_head=2,
            n_embd=8,
            mlp_ratio=2,
            dropout=0.0,
            bias=False,
            tie_embeddings=True,
        )
    )


def test_model_state_quantizes_unique_matrices_and_preserves_aliases() -> None:
    model = _tiny_model()

    package = quantize_model_state(model, group_size=4)

    assert package["schema_version"] == 1
    assert package["quantization"] == {
        "scheme": "symmetric_signed_int4_groupwise_last_dimension",
        "group_size": 4,
        "quantized_range": [-7, 7],
        "nibble_order": "low_first",
    }
    assert package["aliases"] == {
        "lm_head.weight": "token_embedding.weight",
    }
    assert "lm_head.weight" not in package["tensors"]
    assert (
        package["tensors"]["token_embedding.weight"]["storage"]
        == "int4_groupwise"
    )
    assert package["tensors"]["final_norm.weight"]["storage"] == "fp32"
    assert not any("causal_mask" in name for name in package["tensors"])

    matrix_names = {
        name
        for name, parameter in model.named_parameters(remove_duplicate=True)
        if parameter.ndim == 2
    }
    vector_names = {
        name
        for name, parameter in model.named_parameters(remove_duplicate=True)
        if parameter.ndim == 1
    }
    stored_int4 = {
        name
        for name, payload in package["tensors"].items()
        if payload["storage"] == "int4_groupwise"
    }
    stored_fp32 = {
        name
        for name, payload in package["tensors"].items()
        if payload["storage"] == "fp32"
    }
    assert stored_int4 == matrix_names
    assert stored_fp32 == vector_names

    stats = package["storage"]
    assert stats["logical_payload_bytes"] == (
        stats["packed_nibble_bytes"]
        + stats["fp32_scale_bytes"]
        + stats["fp32_passthrough_bytes"]
    )
    assert stats["canonical_tensor_count"] == len(matrix_names | vector_names)
    assert stats["alias_count"] == 1


def test_dequantized_reference_loads_strictly_and_keeps_weight_tying() -> None:
    model = _tiny_model().eval()
    package = quantize_model_state(model, group_size=4)

    state = dequantize_model_state(package)
    reference = reconstruct_dequantized_reference(model.config, package).eval()

    assert set(state) == set(model.state_dict())
    assert (
        reference.token_embedding.weight.data_ptr()
        == reference.lm_head.weight.data_ptr()
    )
    assert torch.equal(
        state["token_embedding.weight"],
        state["lm_head.weight"],
    )
    tokens = torch.tensor([[0, 1, 2, 3]], dtype=torch.long)
    original_logits, _ = model(tokens)
    reference_logits, _ = reference(tokens)
    assert torch.isfinite(reference_logits).all()
    assert float(
        (original_logits - reference_logits).detach().abs().max()
    ) < 0.25


def test_model_state_package_contains_no_fp32_matrix_copy() -> None:
    package = quantize_model_state(_tiny_model(), group_size=4)

    for payload in package["tensors"].values():
        if payload["storage"] == "int4_groupwise":
            assert set(payload) == {
                "storage",
                "packed",
                "scales",
                "shape",
                "group_size",
                "padded_last_dim",
            }
            assert payload["packed"].dtype == torch.uint8
            assert payload["scales"].dtype == torch.float32
        else:
            assert payload["value"].ndim == 1
            assert payload["value"].dtype == torch.float32


def test_model_state_package_is_safe_weights_only_serializable(tmp_path) -> None:
    package = quantize_model_state(_tiny_model(), group_size=4)
    artifact_path = tmp_path / "int4.pt"

    torch.save(package, artifact_path)
    restored = torch.load(
        artifact_path,
        map_location="cpu",
        weights_only=True,
    )

    reference = reconstruct_dequantized_reference(
        _tiny_model().config,
        restored,
    )
    assert reference.token_embedding.weight is reference.lm_head.weight


def test_model_state_rejects_bad_alias_and_schema() -> None:
    package = quantize_model_state(_tiny_model(), group_size=4)

    bad_schema = deepcopy(package)
    bad_schema["schema_version"] = 2
    with pytest.raises(ValueError, match="schema_version"):
        dequantize_model_state(bad_schema)

    bad_alias = deepcopy(package)
    bad_alias["aliases"]["lm_head.weight"] = "missing.weight"
    with pytest.raises(ValueError, match="alias target"):
        dequantize_model_state(bad_alias)


def test_model_state_rejects_unexpected_parameter_rank() -> None:
    class ScalarModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.scalar = torch.nn.Parameter(torch.tensor(1.0))

    with pytest.raises(ValueError, match="rank"):
        quantize_model_state(ScalarModel())
