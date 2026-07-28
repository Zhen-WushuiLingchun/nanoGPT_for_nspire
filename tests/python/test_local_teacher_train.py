from __future__ import annotations

from pathlib import Path

import torch

from nanogpt_nspire.base_train import (
    CHECKPOINT_FILENAME,
    ROUTE,
    BaseTrainingConfig,
)
from nanogpt_nspire.local_teacher_train import (
    LOCAL_TEACHER_CPT_ROUTE,
    LOCAL_TEACHER_SFT_ROUTE,
    frozen_local_teacher_pretrain_config,
    frozen_local_teacher_sft_config,
)
from nanogpt_nspire.models.direct_small_gpt import DirectSmallGPT
from nanogpt_nspire.stage_train import load_parent_checkpoint
from nanogpt_nspire.training_support import sha256_file


def test_base_identity_defaults_remain_backward_compatible(
    tmp_path: Path,
) -> None:
    config = BaseTrainingConfig(
        data_dir=tmp_path / "data",
        output_dir=tmp_path / "out",
        source_commit="abc123",
    )

    assert config.route == ROUTE
    assert config.checkpoint_filename == CHECKPOINT_FILENAME


def test_local_teacher_pretrain_contract_is_larger_and_shared_tokenizer(
    tmp_path: Path,
) -> None:
    config = frozen_local_teacher_pretrain_config(
        data_dir=tmp_path / "cpt",
        output_dir=tmp_path / "pretrain",
        source_commit="abc123",
    )
    model = DirectSmallGPT(config.model_config())

    assert config.route == LOCAL_TEACHER_CPT_ROUTE
    assert config.checkpoint_filename == "local_teacher_cpt.pt"
    assert config.n_layer == 12
    assert config.n_head == 10
    assert config.n_embd == 640
    assert config.vocab_size == 264
    assert config.block_size == 256
    assert config.effective_batch_tokens == 4096
    assert config.steps == 2000
    assert model.parameter_count > 10_821_504
    assert model.token_embedding.weight is model.lm_head.weight


def test_local_teacher_sft_requires_declared_local_parent(
    tmp_path: Path,
) -> None:
    config = frozen_local_teacher_sft_config(
        data_dir=tmp_path / "sft",
        output_dir=tmp_path / "sft-out",
        parent_checkpoint=tmp_path / "local_teacher_cpt.pt",
        parent_checkpoint_sha256="a" * 64,
        source_commit="abc123",
    )

    config.validate()
    assert config.route == LOCAL_TEACHER_SFT_ROUTE
    assert config.checkpoint_filename == "local_teacher_sft.pt"
    assert config.expected_parent_route == LOCAL_TEACHER_CPT_ROUTE
    assert config.n_layer == 12
    assert config.n_head == 10
    assert config.n_embd == 640
    assert config.learning_rate == 0.0001
    assert config.steps == 1000


def test_local_teacher_parent_checkpoint_round_trip(
    tmp_path: Path,
) -> None:
    config = frozen_local_teacher_sft_config(
        data_dir=tmp_path / "sft",
        output_dir=tmp_path / "sft-out",
        parent_checkpoint=tmp_path / "local_teacher_cpt.pt",
        parent_checkpoint_sha256="0" * 64,
        source_commit="abc123",
    )
    model = DirectSmallGPT(config.model_config())
    checkpoint = {
        "model_config": config.model_config().__dict__,
        "model_state_dict": model.state_dict(),
        "route": LOCAL_TEACHER_CPT_ROUTE,
        "schema_version": 1,
        "source_commit": "parent123",
        "tokenizer": {
            "kind": "byte_plus_fixed_special_tokens",
            "vocab_size": 264,
        },
    }
    torch.save(checkpoint, config.parent_checkpoint)
    digest = sha256_file(config.parent_checkpoint)

    loaded = load_parent_checkpoint(
        config.parent_checkpoint,
        expected_sha256=digest,
        expected_route=LOCAL_TEACHER_CPT_ROUTE,
        expected_model_config=config.model_config(),
    )

    assert loaded["route"] == LOCAL_TEACHER_CPT_ROUTE
    assert loaded["sha256"] == digest
