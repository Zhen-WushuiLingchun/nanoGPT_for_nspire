from dataclasses import asdict
import math

import torch

from nanogpt_nspire.data import prepare_dataset
from nanogpt_nspire.direct_small_train import TrainingConfig
from nanogpt_nspire.models.direct_small_gpt import DirectSmallGPT
from nanogpt_nspire.teacher_train import (
    TEACHER_QUALITY_GATE_MAXIMUM_LOSS,
    frozen_teacher_config,
)
from nanogpt_nspire.teacher_v2_train import (
    TEACHER_V2_RUN_IDENTITY,
    frozen_teacher_v2_config,
    run_teacher_v2_training,
)


def test_teacher_v2_changes_only_dropout_and_output_directory(tmp_path) -> None:
    common = {
        "data_dir": tmp_path / "data",
        "source_commit": "source",
        "device": "cpu",
    }
    v1 = frozen_teacher_config(
        **common,
        output_dir=tmp_path / "v1",
    )
    v2 = frozen_teacher_v2_config(
        **common,
        output_dir=tmp_path / "v2",
    )
    v1_fields = asdict(v1)
    v2_fields = asdict(v2)
    differences = {
        field
        for field in v1_fields
        if v1_fields[field] != v2_fields[field]
    }

    assert differences == {"dropout", "output_dir"}
    assert v1.dropout == 0.2
    assert v2.dropout == 0.3
    assert v2.steps * v2.batch_size * v2.block_size == 81_920_000
    assert (
        TEACHER_V2_RUN_IDENTITY
        .quality_gate_maximum_selected_validation_loss
        == TEACHER_QUALITY_GATE_MAXIMUM_LOSS
    )


def test_teacher_v1_and_v2_start_from_identical_parameters(tmp_path) -> None:
    common = {
        "data_dir": tmp_path,
        "output_dir": tmp_path / "out",
        "source_commit": "source",
        "device": "cpu",
    }
    v1_config = frozen_teacher_config(**common)
    v2_config = frozen_teacher_v2_config(**common)

    torch.manual_seed(v1_config.seed)
    v1_model = DirectSmallGPT(v1_config.model_config(vocab_size=65))
    torch.manual_seed(v2_config.seed)
    v2_model = DirectSmallGPT(v2_config.model_config(vocab_size=65))

    assert v1_model.parameter_count == v2_model.parameter_count == 10_695_936
    v1_parameters = dict(v1_model.named_parameters())
    v2_parameters = dict(v2_model.named_parameters())
    assert set(v1_parameters) == set(v2_parameters)
    for name in v1_parameters:
        assert torch.equal(v1_parameters[name], v2_parameters[name]), name


def test_teacher_v2_cpu_smoke_records_separate_identity(tmp_path) -> None:
    source_path = tmp_path / "source.txt"
    source_path.write_text("abcd" * 150, encoding="utf-8", newline="\n")
    data_dir = tmp_path / "data"
    prepare_dataset(source_path, data_dir)
    output_dir = tmp_path / "teacher-v2"
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
        source_commit="teacher-v2-test",
    )

    summary = run_teacher_v2_training(config)

    assert summary["route"] == "Teacher-v2"
    assert summary["source_commit"] == "teacher-v2-test"
    assert summary["run_identity"]["checkpoint_filename"] == (
        "teacher_v2_gpt.pt"
    )
    assert summary["run_identity"]["deployment_interpretation"] == (
        "dropout_only_teacher_candidate_for_int4_and_distillation"
    )
    assert summary["metrics"][
        "quality_gate_maximum_selected_validation_loss"
    ] == TEACHER_QUALITY_GATE_MAXIMUM_LOSS
    assert isinstance(summary["metrics"]["quality_gate_passed"], bool)
    assert math.isfinite(summary["metrics"]["selected_validation_loss"])
    checkpoint_path = output_dir / "teacher_v2_gpt.pt"
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=True,
    )
    assert checkpoint["route"] == "Teacher-v2"
    assert checkpoint["source_commit"] == "teacher-v2-test"
    restored = DirectSmallGPT(
        config.model_config(vocab_size=4)
    )
    restored.load_state_dict(checkpoint["model_state_dict"], strict=True)
