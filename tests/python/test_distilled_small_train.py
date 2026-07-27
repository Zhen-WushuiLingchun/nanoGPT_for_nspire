from dataclasses import asdict
import math

import torch

from nanogpt_nspire.data import prepare_dataset
from nanogpt_nspire.direct_small_train import TrainingConfig
from nanogpt_nspire.distilled_small_train import (
    DIRECT_SMALL_SELECTED_VALIDATION_LOSS,
    DISTILLED_SMALL_RUN_IDENTITY,
    frozen_distilled_student_config,
    run_distilled_training,
)
from nanogpt_nspire.models.direct_small_gpt import (
    DirectSmallConfig,
    DirectSmallGPT,
)


def test_frozen_distilled_student_matches_direct_small_protocol(tmp_path) -> None:
    distilled = frozen_distilled_student_config(
        data_dir=tmp_path / "data",
        output_dir=tmp_path / "distilled",
        device="cpu",
        source_commit="distilled-source",
    )
    direct = TrainingConfig(
        data_dir=tmp_path / "data",
        output_dir=tmp_path / "direct",
        device="cpu",
        source_commit="direct-source",
    )
    distilled_fields = asdict(distilled)
    direct_fields = asdict(direct)
    differences = {
        name
        for name in direct_fields
        if direct_fields[name] != distilled_fields[name]
    }

    assert differences == {"output_dir", "source_commit"}
    assert distilled.steps * distilled.batch_size * distilled.block_size == (
        40_960_000
    )
    assert DISTILLED_SMALL_RUN_IDENTITY.route == "Distilled-Small"
    assert DISTILLED_SMALL_RUN_IDENTITY.checkpoint_filename == (
        "distilled_small_gpt.pt"
    )
    assert (
        DISTILLED_SMALL_RUN_IDENTITY
        .quality_gate_maximum_selected_validation_loss
        == DIRECT_SMALL_SELECTED_VALIDATION_LOSS
    )


def test_direct_and_distilled_students_start_identically(tmp_path) -> None:
    direct = TrainingConfig(
        data_dir=tmp_path,
        output_dir=tmp_path / "direct",
        source_commit="source",
    )
    distilled = frozen_distilled_student_config(
        data_dir=tmp_path,
        output_dir=tmp_path / "distilled",
        source_commit="source",
    )

    torch.manual_seed(direct.seed)
    direct_model = DirectSmallGPT(direct.model_config(vocab_size=65))
    torch.manual_seed(distilled.seed)
    distilled_model = DirectSmallGPT(
        distilled.model_config(vocab_size=65)
    )

    for (direct_name, direct_parameter), (
        distilled_name,
        distilled_parameter,
    ) in zip(
        direct_model.named_parameters(),
        distilled_model.named_parameters(),
        strict=True,
    ):
        assert direct_name == distilled_name
        assert torch.equal(direct_parameter, distilled_parameter)


def test_distilled_cpu_smoke_records_objective_and_teacher(tmp_path) -> None:
    source_path = tmp_path / "source.txt"
    source_path.write_text("abcd" * 150, encoding="utf-8", newline="\n")
    data_dir = tmp_path / "data"
    prepare_dataset(source_path, data_dir)
    output_dir = tmp_path / "distilled"
    config = TrainingConfig(
        data_dir=data_dir,
        output_dir=output_dir,
        device="cpu",
        seed=41,
        steps=20,
        batch_size=2,
        block_size=8,
        n_layer=1,
        n_head=2,
        n_embd=16,
        mlp_ratio=2,
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
        source_commit="distilled-test",
    )
    teacher = DirectSmallGPT(
        DirectSmallConfig(
            vocab_size=4,
            block_size=8,
            n_layer=1,
            n_head=2,
            n_embd=16,
            mlp_ratio=2,
            dropout=0.0,
        )
    )
    teacher_provenance = {
        "checkpoint_sha256": "b" * 64,
        "route": "Teacher-v2",
        "selected_validation_loss": 1.0,
        "source_commit": "teacher-source",
    }

    summary = run_distilled_training(
        config,
        teacher=teacher,
        teacher_provenance=teacher_provenance,
        temperature=2.0,
        alpha=0.5,
    )

    assert summary["route"] == "Distilled-Small"
    assert summary["source_commit"] == "distilled-test"
    assert summary["training_objective"] == {
        "alpha": 0.5,
        "name": "temperature_scaled_logit_distillation",
        "teacher": teacher_provenance,
        "temperature": 2.0,
    }
    assert summary["run_identity"]["checkpoint_filename"] == (
        "distilled_small_gpt.pt"
    )
    assert isinstance(summary["metrics"]["quality_gate_passed"], bool)
    assert math.isfinite(summary["metrics"]["selected_validation_loss"])
    for row in summary["training_history"]:
        assert math.isfinite(row["training_loss"])
        assert math.isfinite(row["hard_label_loss"])
        assert math.isfinite(row["soft_target_loss"])
    assert all(
        parameter.grad is None
        for parameter in teacher.parameters()
    )

    checkpoint_path = output_dir / "distilled_small_gpt.pt"
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=True,
    )
    assert checkpoint["route"] == "Distilled-Small"
    assert checkpoint["training_objective"] == summary["training_objective"]
    restored = DirectSmallGPT(
        DirectSmallConfig(**checkpoint["model_config"])
    )
    restored.load_state_dict(checkpoint["model_state_dict"], strict=True)
    assert restored.token_embedding.weight is restored.lm_head.weight
