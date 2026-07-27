from copy import deepcopy
import json
import math

import pytest
import torch

from nanogpt_nspire.quantize_teacher import (
    INT4_MAXIMUM_LOSS_DEGRADATION,
    QuantizeTeacherConfig,
    experiment_id_for_teacher_route,
    validate_teacher_source_metadata,
)
from nanogpt_nspire.teacher_train import (
    TEACHER_QUALITY_GATE_MAXIMUM_LOSS,
    frozen_teacher_config,
)


def _valid_source_metadata(tmp_path):
    training_config = frozen_teacher_config(
        data_dir=tmp_path / "data",
        output_dir=tmp_path / "teacher",
        device="cpu",
        source_commit="abc123",
    )
    model_config = training_config.model_config(vocab_size=65)
    checkpoint_sha256 = "a" * 64
    checkpoint = {
        "model_config": vars(model_config),
        "model_type": "direct_small_gpt",
        "quality_gate_maximum_selected_validation_loss": (
            TEACHER_QUALITY_GATE_MAXIMUM_LOSS
        ),
        "quality_gate_passed": True,
        "route": "Teacher",
        "schema_version": 1,
        "selected_validation_loss": 1.45,
        "source_commit": "abc123",
        "vocabulary": [chr(index + 32) for index in range(65)],
    }
    run = {
        "artifacts": {
            "checkpoint": {
                "bytes": 123,
                "sha256": checkpoint_sha256,
            }
        },
        "dataset": {
            "schema_version": 1,
            "source_sha256": "b" * 64,
            "train_sha256": "c" * 64,
            "validation_sha256": "d" * 64,
            "train_tokens": 100,
            "validation_tokens": 20,
            "vocab_size": 65,
        },
        "metrics": {
            "quality_gate_maximum_selected_validation_loss": (
                TEACHER_QUALITY_GATE_MAXIMUM_LOSS
            ),
            "quality_gate_passed": True,
            "selected_validation_loss": 1.45,
            "training_tokens": 81_920_000,
        },
        "model": {
            **vars(model_config),
            "parameters": 10_695_936,
            "raw_fp32_parameter_bytes": 42_783_744,
            "type": "direct_small_gpt",
        },
        "route": "Teacher",
        "schema_version": 1,
        "source_commit": "abc123",
    }
    return checkpoint, run, checkpoint_sha256


def test_validate_teacher_source_accepts_frozen_passing_teacher(tmp_path) -> None:
    checkpoint, run, checkpoint_sha256 = _valid_source_metadata(tmp_path)

    validated = validate_teacher_source_metadata(
        checkpoint,
        run,
        checkpoint_sha256=checkpoint_sha256,
    )

    assert validated.vocab_size == 65
    assert validated.n_layer == 6
    assert validated.n_head == 6
    assert validated.n_embd == 384
    assert validated.tie_embeddings


def test_validate_teacher_source_accepts_passing_teacher_v2(tmp_path) -> None:
    checkpoint, run, checkpoint_sha256 = _valid_source_metadata(tmp_path)
    checkpoint["route"] = "Teacher-v2"
    checkpoint["model_config"]["dropout"] = 0.3
    run["route"] = "Teacher-v2"
    run["model"]["dropout"] = 0.3

    validated = validate_teacher_source_metadata(
        checkpoint,
        run,
        checkpoint_sha256=checkpoint_sha256,
    )

    assert validated.dropout == 0.3


def test_validate_teacher_source_allows_explicit_failed_diagnostic(
    tmp_path,
) -> None:
    checkpoint, run, checkpoint_sha256 = _valid_source_metadata(tmp_path)
    checkpoint["quality_gate_passed"] = False
    checkpoint["selected_validation_loss"] = 1.483
    run["metrics"]["quality_gate_passed"] = False
    run["metrics"]["selected_validation_loss"] = 1.483

    validated = validate_teacher_source_metadata(
        checkpoint,
        run,
        checkpoint_sha256=checkpoint_sha256,
        require_quality_gate_passed=False,
    )

    assert validated.n_layer == 6


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda checkpoint, run: checkpoint.update(route="Direct-Small"), "route"),
        (
            lambda checkpoint, run: checkpoint.update(quality_gate_passed=False),
            "quality gate",
        ),
        (
            lambda checkpoint, run: checkpoint["model_config"].update(n_layer=5),
            "frozen teacher",
        ),
        (
            lambda checkpoint, run: checkpoint.update(
                selected_validation_loss=(
                    TEACHER_QUALITY_GATE_MAXIMUM_LOSS + 0.01
                )
            ),
            "selected validation loss",
        ),
        (
            lambda checkpoint, run: run["metrics"].update(
                training_tokens=40_960_000
            ),
            "training token",
        ),
        (
            lambda checkpoint, run: run["artifacts"]["checkpoint"].update(
                sha256="e" * 64
            ),
            "SHA-256",
        ),
    ],
)
def test_validate_teacher_source_rejects_invalid_provenance(
    tmp_path,
    mutation,
    message,
) -> None:
    checkpoint, run, checkpoint_sha256 = _valid_source_metadata(tmp_path)
    checkpoint = deepcopy(checkpoint)
    run = deepcopy(run)
    mutation(checkpoint, run)

    with pytest.raises(ValueError, match=message):
        validate_teacher_source_metadata(
            checkpoint,
            run,
            checkpoint_sha256=checkpoint_sha256,
        )


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("group_size", 0, "group_size"),
        ("eval_batches", 0, "eval_batches"),
        ("batch_size", 0, "batch_size"),
        ("sample_tokens", -1, "sample_tokens"),
        ("temperature", 0.0, "temperature"),
        ("maximum_loss_degradation", -0.1, "maximum_loss_degradation"),
        ("metadata_reserve_bytes", -1, "metadata_reserve_bytes"),
        (
            "diagnostic_allow_failed_teacher",
            "yes",
            "diagnostic_allow_failed_teacher",
        ),
    ],
)
def test_quantize_teacher_config_rejects_invalid_values(
    tmp_path,
    field,
    value,
    message,
) -> None:
    config = QuantizeTeacherConfig(
        data_dir=tmp_path / "data",
        teacher_checkpoint=tmp_path / "teacher.pt",
        output_dir=tmp_path / "out",
        **{field: value},
    )

    with pytest.raises(ValueError, match=message):
        config.validate()


def test_quantize_teacher_defaults_freeze_fairness_gates(tmp_path) -> None:
    config = QuantizeTeacherConfig(
        data_dir=tmp_path / "data",
        teacher_checkpoint=tmp_path / "teacher_gpt.pt",
        output_dir=tmp_path / "out",
    )

    config.validate()

    assert config.group_size == 64
    assert config.maximum_loss_degradation == INT4_MAXIMUM_LOSS_DEGRADATION
    assert config.maximum_loss_degradation == 0.05
    assert config.file_limit_bytes == 6 * 1024 * 1024
    assert config.metadata_reserve_bytes == 64 * 1024
    assert config.validation_seed == 1338
    assert config.sample_seed == 1340
    assert not config.diagnostic_allow_failed_teacher


def test_quantization_experiment_id_tracks_teacher_route() -> None:
    assert experiment_id_for_teacher_route("Teacher") == (
        "lesson06-int4-diagnostic"
    )
    assert experiment_id_for_teacher_route("Teacher-v2") == (
        "lesson07-int4-teacher-v2"
    )
    with pytest.raises(ValueError, match="unsupported teacher route"):
        experiment_id_for_teacher_route("Teacher-v3")
