from __future__ import annotations

from pathlib import Path

import pytest
import torch

from nanogpt_nspire.byte_tokenizer import EOS_ID, FINAL_ID, VOCAB_SIZE
from nanogpt_nspire.efficient_context import (
    GQA_ALIBI_SFT_ROUTE,
    GQA_ALIBI_SFT_V2_ROUTE,
)
from nanogpt_nspire.models.efficient_long_context_gpt import ALIBI_POSITIONS
from nanogpt_nspire.sft_v2_train import (
    boundary_weighted_cross_entropy,
    frozen_sft_v2_config,
)


def test_boundary_weight_one_matches_unweighted_masked_mean() -> None:
    torch.manual_seed(7)
    logits = torch.randn(1, 3, VOCAB_SIZE)
    targets = torch.tensor([[65, FINAL_ID, EOS_ID]])
    mask = torch.ones_like(targets, dtype=torch.bool)

    weighted = boundary_weighted_cross_entropy(
        logits,
        targets,
        mask,
        boundary_token_weight=1.0,
    )
    reference = torch.nn.functional.cross_entropy(
        logits.reshape(-1, VOCAB_SIZE),
        targets.reshape(-1),
    )

    torch.testing.assert_close(weighted, reference)


def test_boundary_weight_increases_penalty_for_bad_final_prediction() -> None:
    logits = torch.zeros(1, 3, VOCAB_SIZE)
    targets = torch.tensor([[65, FINAL_ID, EOS_ID]])
    mask = torch.ones_like(targets, dtype=torch.bool)
    logits[0, 0, 65] = 5.0
    logits[0, 1, 65] = 5.0
    logits[0, 2, EOS_ID] = 5.0

    ordinary = boundary_weighted_cross_entropy(
        logits,
        targets,
        mask,
        boundary_token_weight=1.0,
    )
    emphasized = boundary_weighted_cross_entropy(
        logits,
        targets,
        mask,
        boundary_token_weight=4.0,
    )

    assert emphasized > ordinary


def test_frozen_sft_v2_config_keeps_parent_and_compute_contract(
    tmp_path: Path,
) -> None:
    config = frozen_sft_v2_config(
        data_dir=tmp_path / "data",
        output_dir=tmp_path / "output",
        parent_checkpoint=tmp_path / "parent.pt",
        parent_checkpoint_sha256="a" * 64,
        source_commit="source",
    )

    assert config.expected_parent_route == GQA_ALIBI_SFT_ROUTE
    assert config.route == GQA_ALIBI_SFT_V2_ROUTE
    assert config.position_mode == ALIBI_POSITIONS
    assert config.steps == 1000
    assert config.effective_batch_tokens == 4096
    assert config.boundary_token_weight == 4.0
    assert config.seed == 20260729


def test_sft_v2_rejects_architecture_and_boundary_overrides(
    tmp_path: Path,
) -> None:
    common = {
        "data_dir": tmp_path / "data",
        "output_dir": tmp_path / "output",
        "parent_checkpoint": tmp_path / "parent.pt",
        "parent_checkpoint_sha256": "a" * 64,
        "source_commit": "source",
    }
    with pytest.raises(ValueError, match="frozen"):
        frozen_sft_v2_config(**common, n_kv_head=1)
    with pytest.raises(ValueError, match="frozen"):
        frozen_sft_v2_config(**common, boundary_token_weight=2.0)
