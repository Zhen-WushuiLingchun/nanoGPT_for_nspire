import math

import torch

from nanogpt_nspire.data import prepare_dataset
from nanogpt_nspire.direct_small_train import TrainingConfig
from nanogpt_nspire.models.direct_small_gpt import DirectSmallGPT
from nanogpt_nspire.teacher_train import (
    TEACHER_QUALITY_GATE_MAXIMUM_LOSS,
    TEACHER_RUN_IDENTITY,
    frozen_teacher_config,
    run_teacher_training,
)


def test_frozen_teacher_profile_has_exact_architecture_and_budget(tmp_path):
    config = frozen_teacher_config(
        data_dir=tmp_path,
        output_dir=tmp_path / "out",
        source_commit="source",
    )
    model = DirectSmallGPT(config.model_config(vocab_size=65))

    assert config.steps == 10_000
    assert config.batch_size == 64
    assert config.block_size == 128
    assert config.n_layer == 6
    assert config.n_head == 6
    assert config.n_embd == 384
    assert config.dropout == 0.2
    assert config.steps * config.batch_size * config.block_size == 81_920_000
    assert model.parameter_count == 10_695_936
    assert model.raw_fp32_parameter_bytes == 42_783_744
    assert model.parameter_count // 2 == 5_347_968
    assert TEACHER_RUN_IDENTITY.route == "Teacher"
    assert TEACHER_RUN_IDENTITY.checkpoint_filename == "teacher_gpt.pt"
    assert (
        TEACHER_RUN_IDENTITY.quality_gate_maximum_selected_validation_loss
        == TEACHER_QUALITY_GATE_MAXIMUM_LOSS
    )


def test_teacher_cpu_smoke_records_identity_and_quality_gate(tmp_path):
    source_path = tmp_path / "source.txt"
    source_path.write_text("abcd" * 150, encoding="utf-8", newline="\n")
    data_dir = tmp_path / "data"
    prepare_dataset(source_path, data_dir)
    output_dir = tmp_path / "teacher"
    config = TrainingConfig(
        data_dir=data_dir,
        output_dir=output_dir,
        device="cpu",
        seed=37,
        steps=20,
        batch_size=2,
        block_size=8,
        n_layer=1,
        n_head=2,
        n_embd=16,
        dropout=0.0,
        learning_rate=0.02,
        min_learning_rate=0.002,
        warmup_steps=2,
        weight_decay=0.0,
        max_grad_norm=1.0,
        eval_interval=10,
        eval_batches=2,
        log_interval=5,
        sample_tokens=8,
        source_commit="teacher-test",
    )

    summary = run_teacher_training(config)

    assert summary["route"] == "Teacher"
    assert summary["source_commit"] == "teacher-test"
    assert summary["run_identity"]["checkpoint_filename"] == "teacher_gpt.pt"
    assert summary["run_identity"]["deployment_interpretation"] == (
        "fp32_source_for_int4_and_distillation"
    )
    assert summary["metrics"][
        "quality_gate_maximum_selected_validation_loss"
    ] == TEACHER_QUALITY_GATE_MAXIMUM_LOSS
    assert isinstance(summary["metrics"]["quality_gate_passed"], bool)
    assert math.isfinite(summary["metrics"]["selected_validation_loss"])
    checkpoint_path = output_dir / "teacher_gpt.pt"
    assert checkpoint_path.is_file()
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=True,
    )
    assert checkpoint["route"] == "Teacher"
    assert checkpoint["quality_gate_passed"] == summary["metrics"][
        "quality_gate_passed"
    ]
