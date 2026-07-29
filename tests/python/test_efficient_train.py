from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import pytest

from nanogpt_nspire.efficient_context import (
    GQA_ALIBI_CPT_ROUTE,
    GQA_ALIBI_INIT_ROUTE,
    GQA_ALIBI_SFT_ROUTE,
    GQA_LEARNED_CPT_ROUTE,
    GQA_LEARNED_INIT_ROUTE,
    GQA_LEARNED_SFT_ROUTE,
)
from nanogpt_nspire.efficient_train import (
    frozen_efficient_training_config,
)
from nanogpt_nspire.models.efficient_long_context_gpt import (
    ALIBI_POSITIONS,
    LEARNED_POSITIONS,
)


def _config(tmp_path: Path, stage: str, mode: str):
    return frozen_efficient_training_config(
        stage=stage,
        position_mode=mode,
        data_dir=tmp_path / f"{stage}-{mode}-data",
        output_dir=tmp_path / f"{stage}-{mode}-output",
        parent_checkpoint=tmp_path / f"{stage}-{mode}-parent.pt",
        parent_checkpoint_sha256="a" * 64,
        source_commit="source",
    )


def test_two_position_routes_keep_same_compute_contract(
    tmp_path: Path,
) -> None:
    learned_cpt = _config(tmp_path, "cpt", LEARNED_POSITIONS)
    alibi_cpt = _config(tmp_path, "cpt", ALIBI_POSITIONS)
    learned_sft = _config(tmp_path, "sft", LEARNED_POSITIONS)
    alibi_sft = _config(tmp_path, "sft", ALIBI_POSITIONS)

    assert learned_cpt.expected_parent_route == GQA_LEARNED_INIT_ROUTE
    assert learned_cpt.route == GQA_LEARNED_CPT_ROUTE
    assert alibi_cpt.expected_parent_route == GQA_ALIBI_INIT_ROUTE
    assert alibi_cpt.route == GQA_ALIBI_CPT_ROUTE
    assert learned_sft.expected_parent_route == GQA_LEARNED_CPT_ROUTE
    assert learned_sft.route == GQA_LEARNED_SFT_ROUTE
    assert alibi_sft.expected_parent_route == GQA_ALIBI_CPT_ROUTE
    assert alibi_sft.route == GQA_ALIBI_SFT_ROUTE
    assert {
        config.effective_batch_tokens
        for config in (learned_cpt, alibi_cpt, learned_sft, alibi_sft)
    } == {4096}
    assert learned_cpt.steps == alibi_cpt.steps == 250
    assert learned_sft.steps == alibi_sft.steps == 1000


def test_only_position_and_path_fields_differ(
    tmp_path: Path,
) -> None:
    learned = asdict(_config(tmp_path, "cpt", LEARNED_POSITIONS))
    alibi = asdict(_config(tmp_path, "cpt", ALIBI_POSITIONS))
    ignored = {
        "position_mode",
        "data_dir",
        "output_dir",
        "parent_checkpoint",
    }

    assert {
        key: value for key, value in learned.items() if key not in ignored
    } == {
        key: value for key, value in alibi.items() if key not in ignored
    }


def test_architecture_override_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="architecture is frozen"):
        frozen_efficient_training_config(
            stage="cpt",
            position_mode=LEARNED_POSITIONS,
            data_dir=tmp_path / "data",
            output_dir=tmp_path / "output",
            parent_checkpoint=tmp_path / "parent.pt",
            parent_checkpoint_sha256="a" * 64,
            source_commit="source",
            n_kv_head=1,
        )
