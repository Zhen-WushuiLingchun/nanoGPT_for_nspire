from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import pytest
import torch

from nanogpt_nspire.context_extension import CONTEXT512_CPT_ROUTE
from nanogpt_nspire.efficient_context import (
    GQA_ALIBI_INIT_ROUTE,
    GQA_LEARNED_INIT_ROUTE,
    convert_mha_qkv_tensor,
    create_efficient_init_checkpoint,
    load_efficient_checkpoint,
    model_state_sha256,
)
from nanogpt_nspire.models.direct_small_gpt import (
    DirectSmallConfig,
    DirectSmallGPT,
)
from nanogpt_nspire.models.efficient_long_context_gpt import (
    ALIBI_POSITIONS,
    LEARNED_POSITIONS,
    EfficientLongContextConfig,
)
from nanogpt_nspire.training_support import sha256_file


def test_qkv_conversion_copies_query_and_averages_groups() -> None:
    query = torch.arange(16, dtype=torch.float32).reshape(4, 4)
    key = torch.stack(
        [torch.full((4,), value) for value in (2.0, 4.0, 10.0, 14.0)]
    )
    value = torch.stack(
        [torch.full((4,), value) for value in (1.0, 5.0, 9.0, 17.0)]
    )
    source = torch.cat((query, key, value))

    converted = convert_mha_qkv_tensor(
        source,
        n_head=4,
        n_kv_head=2,
        n_embd=4,
    )

    assert torch.equal(converted[:4], query)
    assert torch.equal(
        converted[4:6],
        torch.tensor([[3.0] * 4, [12.0] * 4]),
    )
    assert torch.equal(
        converted[6:8],
        torch.tensor([[3.0] * 4, [13.0] * 4]),
    )


def test_model_state_hash_is_order_independent_and_value_sensitive() -> None:
    left = {
        "b": torch.tensor([2.0]),
        "a": torch.tensor([1.0]),
    }
    reordered = {
        "a": torch.tensor([1.0]),
        "b": torch.tensor([2.0]),
    }
    changed = {
        "a": torch.tensor([1.0]),
        "b": torch.tensor([3.0]),
    }

    assert model_state_sha256(left) == model_state_sha256(reordered)
    assert model_state_sha256(left) != model_state_sha256(changed)


def _source_config() -> DirectSmallConfig:
    return DirectSmallConfig(
        vocab_size=264,
        block_size=8,
        n_layer=1,
        n_head=4,
        n_embd=16,
        mlp_ratio=2,
        dropout=0.0,
        bias=False,
        tie_embeddings=True,
    )


def _target_config(position_mode: str) -> EfficientLongContextConfig:
    return EfficientLongContextConfig(
        **asdict(_source_config()),
        n_kv_head=2,
        position_mode=position_mode,
    )


def _write_parent(path: Path) -> tuple[dict[str, torch.Tensor], str]:
    torch.manual_seed(17)
    model = DirectSmallGPT(_source_config())
    state = {
        name: value.detach().clone()
        for name, value in model.state_dict().items()
    }
    torch.save(
        {
            "model_config": asdict(_source_config()),
            "model_state_dict": state,
            "route": CONTEXT512_CPT_ROUTE,
            "schema_version": 1,
            "source_commit": "parent",
            "tokenizer": {
                "kind": "byte_plus_fixed_special_tokens",
                "vocab_size": 264,
            },
        },
        path,
    )
    return state, sha256_file(path)


@pytest.mark.parametrize(
    ("position_mode", "expected_route"),
    [
        (LEARNED_POSITIONS, GQA_LEARNED_INIT_ROUTE),
        (ALIBI_POSITIONS, GQA_ALIBI_INIT_ROUTE),
    ],
)
def test_checkpoint_conversion_is_complete_and_strict(
    tmp_path: Path,
    position_mode: str,
    expected_route: str,
) -> None:
    parent_path = tmp_path / "parent.pt"
    source_state, parent_hash = _write_parent(parent_path)
    output = tmp_path / f"{position_mode}.pt"
    target_config = _target_config(position_mode)

    summary = create_efficient_init_checkpoint(
        parent_checkpoint=parent_path,
        parent_checkpoint_sha256=parent_hash,
        output_path=output,
        position_mode=position_mode,
        source_commit="lesson15",
        source_model_config=_source_config(),
        target_model_config=target_config,
    )
    loaded, provenance = load_efficient_checkpoint(
        output,
        expected_sha256=str(summary["checkpoint_sha256"]),
        expected_route=expected_route,
        expected_model_config=target_config,
    )
    target_state = loaded.state_dict()

    assert provenance["route"] == expected_route
    assert summary["model_state_sha256"] == model_state_sha256(target_state)
    assert (
        "position_embedding.weight" in target_state
    ) is (position_mode == LEARNED_POSITIONS)
    for name, value in target_state.items():
        if name.endswith(".attention.qkv.weight"):
            expected = convert_mha_qkv_tensor(
                source_state[name],
                n_head=4,
                n_kv_head=2,
                n_embd=16,
            )
            assert torch.equal(value, expected)
        else:
            assert torch.equal(value, source_state[name])
    with pytest.raises(ValueError, match="SHA-256"):
        load_efficient_checkpoint(
            output,
            expected_sha256="0" * 64,
            expected_route=expected_route,
            expected_model_config=target_config,
        )
