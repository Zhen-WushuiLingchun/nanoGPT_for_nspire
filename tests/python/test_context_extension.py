from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import pytest
import torch

from nanogpt_nspire.context_extension import (
    CONTEXT512_CPT_ROUTE,
    CONTEXT512_HYBRID_ROUTE,
    CONTEXT512_INIT_ROUTE,
    SOURCE_ROUTE,
    create_context512_initial_checkpoint,
    frozen_context512_cpt_config,
    frozen_context512_hybrid_config,
    kv_cache_bytes,
)
from nanogpt_nspire.models.direct_small_gpt import (
    DirectSmallConfig,
    DirectSmallGPT,
)
from nanogpt_nspire.training_support import sha256_file


def _write_parent(path: Path) -> str:
    config = DirectSmallConfig(
        vocab_size=264,
        block_size=256,
        n_layer=6,
        n_head=6,
        n_embd=384,
        mlp_ratio=4,
        dropout=0.1,
        bias=False,
        tie_embeddings=True,
    )
    torch.manual_seed(7)
    model = DirectSmallGPT(config)
    torch.save(
        {
            "model_config": asdict(config),
            "model_state_dict": model.state_dict(),
            "route": SOURCE_ROUTE,
            "schema_version": 1,
            "source_commit": "parent",
            "tokenizer": {
                "kind": "byte_plus_fixed_special_tokens",
                "vocab_size": 264,
            },
        },
        path,
    )
    return sha256_file(path)


def test_context_initializer_preserves_all_old_behavior_tensors(
    tmp_path: Path,
) -> None:
    parent_path = tmp_path / "parent.pt"
    digest = _write_parent(parent_path)
    output = tmp_path / "context512-init.pt"

    summary = create_context512_initial_checkpoint(
        parent_checkpoint=parent_path,
        parent_checkpoint_sha256=digest,
        output_path=output,
        source_commit="lesson14",
    )
    parent = torch.load(parent_path, weights_only=True)
    extended = torch.load(output, weights_only=True)

    assert extended["route"] == CONTEXT512_INIT_ROUTE
    assert extended["model_config"]["block_size"] == 512
    assert summary["new_parameters"] == 98_304
    for name, source in parent["model_state_dict"].items():
        target = extended["model_state_dict"][name]
        if name == "position_embedding.weight":
            assert torch.equal(target[:256], source)
            assert torch.equal(target[256:], source)
        else:
            assert torch.equal(target, source)


def test_context_configs_keep_tokens_per_update_and_lineage(
    tmp_path: Path,
) -> None:
    cpt = frozen_context512_cpt_config(
        data_dir=tmp_path / "cpt-data",
        output_dir=tmp_path / "cpt-output",
        parent_checkpoint=tmp_path / "init.pt",
        parent_checkpoint_sha256="a" * 64,
        source_commit="source",
    )
    hybrid = frozen_context512_hybrid_config(
        data_dir=tmp_path / "hybrid-data",
        output_dir=tmp_path / "hybrid-output",
        parent_checkpoint=tmp_path / "cpt.pt",
        parent_checkpoint_sha256="b" * 64,
        source_commit="source",
    )

    assert cpt.route == CONTEXT512_CPT_ROUTE
    assert cpt.expected_parent_route == CONTEXT512_INIT_ROUTE
    assert cpt.effective_batch_tokens == 4096
    assert cpt.steps == 250
    assert hybrid.route == CONTEXT512_HYBRID_ROUTE
    assert hybrid.expected_parent_route == CONTEXT512_CPT_ROUTE
    assert hybrid.effective_batch_tokens == 4096
    assert hybrid.steps == 1000


def test_kv_cache_cost_doubles_at_512() -> None:
    at_256 = kv_cache_bytes(block_size=256)
    at_512 = kv_cache_bytes(block_size=512)

    assert at_256 == 4_718_592
    assert at_512 == 9_437_184
    assert at_512 == 2 * at_256
    with pytest.raises(ValueError, match="positive integer"):
        kv_cache_bytes(block_size=0)


def test_context_architecture_cannot_be_overridden(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match="architecture is frozen"):
        frozen_context512_cpt_config(
            data_dir=tmp_path / "data",
            output_dir=tmp_path / "output",
            parent_checkpoint=tmp_path / "init.pt",
            parent_checkpoint_sha256="a" * 64,
            source_commit="source",
            block_size=1024,
        )
