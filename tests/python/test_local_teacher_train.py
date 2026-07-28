from __future__ import annotations

from pathlib import Path

import torch

from nanogpt_nspire.base_corpus import (
    CorpusRecord,
    build_corpus,
    stable_family_split,
)
from nanogpt_nspire.base_train import (
    CHECKPOINT_FILENAME,
    ROUTE,
    BaseTrainingConfig,
    run_base_training,
)
from nanogpt_nspire.byte_tokenizer import ConversationTurn
from nanogpt_nspire.local_teacher_train import (
    LOCAL_TEACHER_CPT_ROUTE,
    LOCAL_TEACHER_SFT_ROUTE,
    frozen_local_teacher_pretrain_config,
    frozen_local_teacher_sft_config,
)
from nanogpt_nspire.models.direct_small_gpt import DirectSmallGPT
from nanogpt_nspire.stage_train import (
    StageTrainingConfig,
    load_parent_checkpoint,
    run_stage_training,
)
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


def test_generic_local_teacher_routes_complete_cpu_training_smoke(
    tmp_path: Path,
) -> None:
    registry = Path(__file__).resolve().parents[2] / "experiments" / (
        "lesson10-public-sources.json"
    )
    split_seed = "local-teacher-engine-smoke"
    records = []
    for split in ("train", "validation", "test"):
        family = next(
            f"smoke-{split}-{index}"
            for index in range(100_000)
            if stable_family_split(
                f"smoke-{split}-{index}",
                split_seed=split_seed,
            )
            == split
        )
        records.append(
            CorpusRecord.conversation(
                record_id=f"record-{split}",
                family_id=family,
                turns=(
                    ConversationTurn("user", f"{split} question"),
                    ConversationTurn("assistant", f"{split} answer"),
                ),
                source_id="project-arithmetic-v1",
                license_id="MIT",
            )
        )
    data_dir = tmp_path / "data"
    build_corpus(
        records,
        data_dir,
        registry_path=registry,
        split_seed=split_seed,
    )
    pretrain_output = tmp_path / "pretrain"
    pretrain = BaseTrainingConfig(
        data_dir=data_dir,
        output_dir=pretrain_output,
        source_commit="smoke",
        device="cpu",
        steps=2,
        micro_batch_size=1,
        gradient_accumulation_steps=1,
        block_size=16,
        n_layer=1,
        n_head=1,
        n_embd=16,
        mlp_ratio=2,
        warmup_steps=1,
        eval_interval=1,
        eval_batches=1,
        log_interval=1,
        overfit_gate_steps=1,
        use_bfloat16=False,
        sample_tokens=0,
        top_k=10,
        route="Local-Teacher-Engine-CPT",
        checkpoint_filename="engine_cpt.pt",
    )
    pretrain_summary = run_base_training(pretrain)
    parent = pretrain_output / "engine_cpt.pt"
    parent_sha = sha256_file(parent)
    sft = StageTrainingConfig(
        stage="sft",
        data_dir=data_dir,
        output_dir=tmp_path / "sft",
        parent_checkpoint=parent,
        parent_checkpoint_sha256=parent_sha,
        expected_parent_route="Local-Teacher-Engine-CPT",
        source_commit="smoke",
        device="cpu",
        steps=2,
        micro_batch_size=1,
        gradient_accumulation_steps=1,
        block_size=16,
        n_layer=1,
        n_head=1,
        n_embd=16,
        mlp_ratio=2,
        warmup_steps=1,
        eval_interval=1,
        eval_batches=1,
        log_interval=1,
        overfit_gate_steps=1,
        use_bfloat16=False,
        route_override="Local-Teacher-Engine-SFT",
        checkpoint_filename_override="engine_sft.pt",
        required_parent_route_override="Local-Teacher-Engine-CPT",
    )
    sft_summary = run_stage_training(sft)

    assert pretrain_summary["route"] == "Local-Teacher-Engine-CPT"
    assert sft_summary["route"] == "Local-Teacher-Engine-SFT"
    assert (tmp_path / "sft" / "engine_sft.pt").is_file()
