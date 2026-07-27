from copy import deepcopy
import hashlib
import json

import numpy as np
import pytest
import torch

from nanogpt_nspire.export_format import (
    MODEL_STORAGE_FP32,
    MODEL_STORAGE_W4A8,
    STORAGE_FP32,
    STORAGE_INT4_GROUPWISE,
    ModelFormatError,
    parse_model_file,
)
from nanogpt_nspire.export_model import (
    FINAL_NORM_TENSOR_ID,
    POSITION_EMBEDDING_TENSOR_ID,
    TOKEN_EMBEDDING_TENSOR_ID,
    build_export,
    export_checkpoint,
    expected_tensor_descriptors,
)
from nanogpt_nspire.models.direct_small_gpt import (
    DirectSmallConfig,
    DirectSmallGPT,
)
from nanogpt_nspire.quantization import quantize_model_state


def _model_config() -> DirectSmallConfig:
    return DirectSmallConfig(
        vocab_size=7,
        block_size=8,
        n_layer=1,
        n_head=2,
        n_embd=8,
        mlp_ratio=2,
        dropout=0.0,
    )


def _fp32_checkpoint() -> dict[str, object]:
    torch.manual_seed(7)
    model = DirectSmallGPT(_model_config())
    return {
        "model_config": vars(model.config),
        "model_state_dict": {
            name: value.detach().cpu().clone()
            for name, value in model.state_dict().items()
        },
        "model_type": "direct_small_gpt",
        "route": "Distilled-Small",
        "schema_version": 1,
        "selected_validation_loss": 1.25,
        "source_commit": "a" * 40,
        "vocabulary": ["\n", " ", "a", "b", "c", "d", "é"],
    }


def _int4_checkpoint() -> dict[str, object]:
    torch.manual_seed(11)
    model = DirectSmallGPT(_model_config())
    return {
        "model_config": vars(model.config),
        "model_type": "direct_small_gpt_int4",
        "quantized_model_state": quantize_model_state(
            model,
            group_size=4,
        ),
        "route": "Quantized-Small",
        "schema_version": 1,
        "source_commit": "b" * 40,
        "vocabulary": ["\n", " ", "a", "b", "c", "d", "é"],
    }


def test_expected_tensor_descriptors_are_stable_and_omit_tied_head() -> None:
    descriptors = expected_tensor_descriptors(_model_config())

    assert descriptors[0].tensor_id == TOKEN_EMBEDDING_TENSOR_ID
    assert descriptors[0].name == "token_embedding.weight"
    assert descriptors[1].tensor_id == POSITION_EMBEDDING_TENSOR_ID
    assert descriptors[-1].tensor_id == FINAL_NORM_TENSOR_ID
    assert descriptors[-1].name == "final_norm.weight"
    assert len(descriptors) == 2 + 6 * _model_config().n_layer + 1
    assert "lm_head.weight" not in {
        descriptor.name for descriptor in descriptors
    }


def test_build_fp32_export_deduplicates_tied_head_and_round_trips() -> None:
    checkpoint = _fp32_checkpoint()
    exported = build_export(checkpoint)
    parsed = parse_model_file(exported.data)

    assert exported.route == "Distilled-Small"
    assert parsed.spec.model_storage == MODEL_STORAGE_FP32
    assert len(parsed.tensors) == 9
    assert parsed.spec.tie_embeddings
    descriptor_by_id = {
        descriptor.tensor_id: descriptor
        for descriptor in expected_tensor_descriptors(_model_config())
    }
    state = checkpoint["model_state_dict"]
    assert isinstance(state, dict)
    for tensor_id, view in parsed.tensors.items():
        descriptor = descriptor_by_id[tensor_id]
        expected = state[descriptor.name]
        actual = np.frombuffer(view.data, dtype="<f4").reshape(view.shape)
        np.testing.assert_array_equal(actual, expected.numpy())
        assert view.storage == STORAGE_FP32
    assert exported.manifest["aliases"] == {
        "lm_head.weight": "token_embedding.weight"
    }


def test_build_int4_export_preserves_packed_values_and_scales() -> None:
    checkpoint = _int4_checkpoint()
    package = checkpoint["quantized_model_state"]
    assert isinstance(package, dict)
    exported = build_export(checkpoint)
    parsed = parse_model_file(exported.data)

    assert exported.route == "Quantized-Small"
    assert parsed.spec.model_storage == MODEL_STORAGE_W4A8
    assert parsed.spec.weight_group_size == 4
    descriptor_by_id = {
        descriptor.tensor_id: descriptor
        for descriptor in expected_tensor_descriptors(_model_config())
    }
    tensors = package["tensors"]
    for tensor_id, view in parsed.tensors.items():
        descriptor = descriptor_by_id[tensor_id]
        source = tensors[descriptor.name]
        if source["storage"] == "int4_groupwise":
            assert view.storage == STORAGE_INT4_GROUPWISE
            assert bytes(view.data) == source["packed"].numpy().tobytes()
            assert bytes(view.auxiliary) == (
                source["scales"].numpy().astype("<f4").tobytes()
            )
        else:
            assert view.storage == STORAGE_FP32
            assert bytes(view.data) == (
                source["value"].numpy().astype("<f4").tobytes()
            )


def test_export_rejects_broken_alias_missing_tensor_and_diagnostic_route() -> None:
    broken_alias = _fp32_checkpoint()
    state = broken_alias["model_state_dict"]
    assert isinstance(state, dict)
    state["lm_head.weight"][0, 0] += 1.0
    with pytest.raises(ModelFormatError, match="tied"):
        build_export(broken_alias)

    missing = _fp32_checkpoint()
    missing_state = missing["model_state_dict"]
    assert isinstance(missing_state, dict)
    del missing_state["blocks.0.mlp.output.weight"]
    with pytest.raises(ModelFormatError, match="tensor names"):
        build_export(missing)

    diagnostic = _int4_checkpoint()
    diagnostic["route"] = "Quantized-Small-Diagnostic"
    with pytest.raises(ModelFormatError, match="formal Quantized-Small"):
        build_export(diagnostic)


def test_export_rejects_nonfinite_or_wrong_shape_tensor() -> None:
    nonfinite = _fp32_checkpoint()
    state = nonfinite["model_state_dict"]
    assert isinstance(state, dict)
    state["final_norm.weight"][0] = float("nan")
    with pytest.raises(ModelFormatError, match="non-finite"):
        build_export(nonfinite)

    wrong_shape = _fp32_checkpoint()
    wrong_state = wrong_shape["model_state_dict"]
    assert isinstance(wrong_state, dict)
    wrong_state["final_norm.weight"] = torch.ones(9)
    with pytest.raises(ModelFormatError, match="shape"):
        build_export(wrong_shape)


def test_export_checkpoint_writes_atomic_model_and_manifest(
    tmp_path,
) -> None:
    checkpoint_path = tmp_path / "distilled.pt"
    output_path = tmp_path / "distilled.ngm"
    torch.save(_fp32_checkpoint(), checkpoint_path)

    manifest = export_checkpoint(
        checkpoint_path=checkpoint_path,
        output_path=output_path,
    )

    manifest_path = output_path.with_suffix(".ngm.json")
    assert output_path.is_file()
    assert manifest_path.is_file()
    assert json.loads(manifest_path.read_text(encoding="utf-8")) == manifest
    assert manifest["output"]["bytes"] == output_path.stat().st_size
    assert manifest["output"]["sha256"] == hashlib.sha256(
        output_path.read_bytes()
    ).hexdigest()
    assert manifest["source"]["bytes"] == checkpoint_path.stat().st_size
    assert manifest["source"]["sha256"] == hashlib.sha256(
        checkpoint_path.read_bytes()
    ).hexdigest()
    assert not list(tmp_path.glob("*.tmp"))


def test_legacy_direct_checkpoint_requires_matching_run_route() -> None:
    checkpoint = _fp32_checkpoint()
    del checkpoint["route"]

    with pytest.raises(ModelFormatError, match="route"):
        build_export(checkpoint)

    exported = build_export(
        checkpoint,
        run_metadata={
            "route": "Direct-Small",
            "source_commit": checkpoint["source_commit"],
        },
    )
    assert exported.route == "Direct-Small"

    mismatched = deepcopy(
        {
            "route": "Direct-Small",
            "source_commit": "c" * 40,
        }
    )
    with pytest.raises(ModelFormatError, match="source_commit"):
        build_export(checkpoint, run_metadata=mismatched)
