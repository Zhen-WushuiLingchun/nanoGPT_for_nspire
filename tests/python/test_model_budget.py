import pytest

from nanogpt_nspire.model_budget import (
    DEPLOYMENT_FILE_LIMIT_BYTES,
    DEPLOYMENT_RAM_LIMIT_BYTES,
    ModelBudgetPolicy,
    estimate_model_budget,
    parameter_tensors,
)
from nanogpt_nspire.models.direct_small_gpt import (
    DirectSmallConfig,
    DirectSmallGPT,
)


@pytest.mark.parametrize("tie_embeddings", (True, False))
def test_parameter_formula_matches_constructed_model(tie_embeddings) -> None:
    config = DirectSmallConfig(
        vocab_size=264,
        block_size=16,
        n_layer=2,
        n_head=2,
        n_embd=32,
        mlp_ratio=4,
        dropout=0.0,
        bias=False,
        tie_embeddings=tie_embeddings,
    )
    model = DirectSmallGPT(config)

    tensors = parameter_tensors(config)

    assert sum(tensor.element_count for tensor in tensors) == (
        model.parameter_count
    )
    assert len({tensor.name for tensor in tensors}) == len(tensors)
    assert ("lm_head.weight" in {tensor.name for tensor in tensors}) is (
        not tie_embeddings
    )


def test_frozen_student_and_teacher_parameter_counts() -> None:
    student = DirectSmallConfig(
        vocab_size=264,
        block_size=256,
        n_layer=6,
        n_head=6,
        n_embd=384,
        dropout=0.0,
    )
    teacher = DirectSmallConfig(
        vocab_size=264,
        block_size=256,
        n_layer=12,
        n_head=10,
        n_embd=640,
        dropout=0.0,
    )

    assert estimate_model_budget(student)["parameter_count"] == 10_821_504
    assert estimate_model_budget(teacher)["parameter_count"] == 59_331_200


def test_student_w4a8_estimate_counts_every_memory_component() -> None:
    config = DirectSmallConfig(
        vocab_size=264,
        block_size=256,
        n_layer=6,
        n_head=6,
        n_embd=384,
        dropout=0.0,
    )

    estimate = estimate_model_budget(config, storage="w4a8")

    assert estimate["file"]["limit_bytes"] == DEPLOYMENT_FILE_LIMIT_BYTES
    assert estimate["file"]["target_band_passed"] is True
    assert estimate["file"]["estimated_bytes"] == sum(
        estimate["file"]["components"].values()
    )
    assert estimate["inference_ram"]["limit_bytes"] == (
        DEPLOYMENT_RAM_LIMIT_BYTES
    )
    assert estimate["inference_ram"]["limit_passed"] is True
    assert estimate["inference_ram"]["estimated_peak_bytes"] == sum(
        estimate["inference_ram"]["components"].values()
    )
    assert estimate["inference_ram"]["components"]["kv_cache_bytes"] > 0
    assert estimate["inference_ram"]["components"][
        "float_workspace_bytes"
    ] > 0
    assert estimate["inference_ram"]["components"][
        "quantized_activation_workspace_bytes"
    ] > 0
    assert estimate["storage"]["packed_weight_bytes"] > 0
    assert estimate["storage"]["fp32_scale_bytes"] > 0
    assert estimate["storage"]["fp32_passthrough_bytes"] > 0


@pytest.mark.parametrize("storage", ("fp32", "fp16", "int8", "w4a8"))
def test_supported_storage_policies_are_distinct(storage) -> None:
    config = DirectSmallConfig(
        vocab_size=264,
        block_size=32,
        n_layer=1,
        n_head=2,
        n_embd=32,
        dropout=0.0,
    )

    estimate = estimate_model_budget(config, storage=storage)

    assert estimate["storage"]["policy"] == storage
    assert estimate["file"]["estimated_bytes"] > 0


def test_budget_rejects_invalid_config_and_policy() -> None:
    invalid = DirectSmallConfig(
        vocab_size=264,
        block_size=256,
        n_layer=6,
        n_head=5,
        n_embd=384,
    )
    with pytest.raises(ValueError, match="divisible"):
        estimate_model_budget(invalid)

    valid = DirectSmallConfig()
    with pytest.raises(ValueError, match="unsupported storage"):
        estimate_model_budget(valid, storage="binary")
    with pytest.raises(ValueError, match="safety_reserve_bytes"):
        estimate_model_budget(
            valid,
            policy=ModelBudgetPolicy(safety_reserve_bytes=0),
        )
