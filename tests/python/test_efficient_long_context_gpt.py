from __future__ import annotations

from dataclasses import asdict

import pytest
import torch

from nanogpt_nspire.models.direct_small_gpt import (
    DirectSmallConfig,
    DirectSmallGPT,
)
from nanogpt_nspire.models.efficient_long_context_gpt import (
    ALIBI_POSITIONS,
    LEARNED_POSITIONS,
    EfficientLongContextConfig,
    EfficientLongContextGPT,
    GroupedQueryCausalSelfAttention,
    alibi_slopes,
)


def _tiny(**overrides: object) -> EfficientLongContextConfig:
    values: dict[str, object] = {
        "vocab_size": 32,
        "block_size": 8,
        "n_layer": 2,
        "n_head": 4,
        "n_kv_head": 2,
        "n_embd": 16,
        "mlp_ratio": 2,
        "dropout": 0.0,
        "bias": False,
        "tie_embeddings": True,
        "position_mode": LEARNED_POSITIONS,
    }
    values.update(overrides)
    return EfficientLongContextConfig(**values)


def test_alibi_slopes_are_deterministic_for_six_heads() -> None:
    assert alibi_slopes(6) == (
        0.25,
        0.0625,
        0.015625,
        0.00390625,
        0.5,
        0.125,
    )


def test_config_rejects_incompatible_head_grouping() -> None:
    with pytest.raises(ValueError, match="divisible by n_kv_head"):
        _tiny(n_head=4, n_kv_head=3).validate()
    with pytest.raises(ValueError, match="position_mode"):
        _tiny(position_mode="rope").validate()


@pytest.mark.parametrize(
    "position_mode",
    [LEARNED_POSITIONS, ALIBI_POSITIONS],
)
def test_parameter_formula_and_kv_budget(position_mode: str) -> None:
    model = EfficientLongContextGPT(
        EfficientLongContextConfig(position_mode=position_mode)
    )

    assert model.parameter_count == model.expected_parameter_count
    assert model.kv_cache_bytes_fp32 == 3_145_728
    assert (
        "position_embedding.weight" in model.state_dict()
    ) is (position_mode == LEARNED_POSITIONS)


def test_grouped_attention_shapes_and_causal_isolation() -> None:
    attention = GroupedQueryCausalSelfAttention(
        _tiny(position_mode=ALIBI_POSITIONS)
    )
    inputs = torch.randn(2, 5, 16)

    output, weights = attention(inputs, return_weights=True)

    assert output.shape == (2, 5, 16)
    assert weights.shape == (2, 4, 5, 5)
    assert torch.equal(
        weights.masked_select(
            torch.triu(torch.ones(5, 5, dtype=torch.bool), diagonal=1)
            .view(1, 1, 5, 5)
            .expand_as(weights)
        ),
        torch.zeros(2 * 4 * 10),
    )


def test_full_kv_head_mode_matches_direct_small_exactly() -> None:
    direct_config = DirectSmallConfig(
        vocab_size=32,
        block_size=8,
        n_layer=2,
        n_head=4,
        n_embd=16,
        mlp_ratio=2,
        dropout=0.0,
        bias=False,
        tie_embeddings=True,
    )
    efficient_config = EfficientLongContextConfig(
        **asdict(direct_config),
        n_kv_head=4,
        position_mode=LEARNED_POSITIONS,
    )
    torch.manual_seed(19)
    direct = DirectSmallGPT(direct_config).eval()
    efficient = EfficientLongContextGPT(efficient_config).eval()
    efficient.load_state_dict(direct.state_dict(), strict=True)
    tokens = torch.tensor([[1, 7, 3, 9, 2]], dtype=torch.long)

    direct_logits, _ = direct(tokens)
    efficient_logits, _ = efficient(tokens)

    assert torch.equal(direct_logits, efficient_logits)
