import math

import pytest
import torch

from nanogpt_nspire.data import prepare_dataset
from nanogpt_nspire.direct_small_train import (
    TrainingConfig,
    TrainingRunIdentity,
    configure_adamw,
    learning_rate_at_step,
    run_training,
)
from nanogpt_nspire.models.direct_small_gpt import (
    DirectSmallConfig,
    DirectSmallGPT,
)


def test_learning_rate_warms_up_then_cosine_decays():
    arguments = {
        "max_learning_rate": 1e-3,
        "min_learning_rate": 1e-4,
        "warmup_steps": 100,
        "max_steps": 5000,
    }

    assert learning_rate_at_step(1, **arguments) == pytest.approx(1e-5)
    assert learning_rate_at_step(100, **arguments) == pytest.approx(1e-3)
    assert learning_rate_at_step(5000, **arguments) == pytest.approx(1e-4)
    assert learning_rate_at_step(6000, **arguments) == pytest.approx(1e-4)
    middle = learning_rate_at_step(2550, **arguments)
    assert 1e-4 < middle < 1e-3


@pytest.mark.parametrize(
    "overrides",
    [
        {"step": 0},
        {"max_steps": 100, "warmup_steps": 101},
        {"max_learning_rate": 0.0},
        {"min_learning_rate": -1.0},
        {"min_learning_rate": 0.002, "max_learning_rate": 0.001},
    ],
)
def test_learning_rate_rejects_invalid_arguments(overrides):
    arguments = {
        "step": 1,
        "max_learning_rate": 1e-3,
        "min_learning_rate": 1e-4,
        "warmup_steps": 10,
        "max_steps": 100,
    }
    arguments.update(overrides)

    with pytest.raises(ValueError):
        learning_rate_at_step(**arguments)


def test_configure_adamw_groups_each_unique_parameter_once():
    model = DirectSmallGPT(
        DirectSmallConfig(
            vocab_size=7,
            block_size=8,
            n_layer=1,
            n_head=2,
            n_embd=16,
            dropout=0.0,
        )
    )

    optimizer = configure_adamw(
        model,
        learning_rate=1e-3,
        weight_decay=0.1,
        beta1=0.9,
        beta2=0.99,
    )

    grouped = [
        parameter
        for group in optimizer.param_groups
        for parameter in group["params"]
    ]
    expected = list(model.parameters())
    assert len(grouped) == len(expected)
    assert {id(parameter) for parameter in grouped} == {
        id(parameter) for parameter in expected
    }
    assert len({id(parameter) for parameter in grouped}) == len(grouped)
    decay_group = next(
        group for group in optimizer.param_groups if group["weight_decay"] == 0.1
    )
    no_decay_group = next(
        group for group in optimizer.param_groups if group["weight_decay"] == 0.0
    )
    assert all(parameter.ndim >= 2 for parameter in decay_group["params"])
    assert all(parameter.ndim < 2 for parameter in no_decay_group["params"])


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("route", "", "route"),
        ("checkpoint_filename", "../model.pt", "checkpoint_filename"),
        ("checkpoint_filename", "model.bin", "checkpoint_filename"),
        ("deployment_interpretation", "", "deployment_interpretation"),
        (
            "quality_gate_maximum_selected_validation_loss",
            0.0,
            "quality_gate",
        ),
    ],
)
def test_training_run_identity_rejects_invalid_values(field, value, message):
    identity = TrainingRunIdentity(**{field: value})

    with pytest.raises(ValueError, match=message):
        identity.validate()


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("steps", 0, "steps"),
        ("eval_interval", 0, "eval_interval"),
        ("log_interval", 0, "log_interval"),
        ("learning_rate", 0.0, "learning_rate"),
        ("min_learning_rate", 0.002, "min_learning_rate"),
        ("warmup_steps", 5001, "warmup_steps"),
        ("deployment_file_limit_bytes", 0, "deployment_file_limit_bytes"),
    ],
)
def test_training_config_rejects_invalid_values(
    tmp_path,
    field,
    value,
    message,
):
    config = TrainingConfig(
        data_dir=tmp_path,
        output_dir=tmp_path / "out",
        **{field: value},
    )

    with pytest.raises(ValueError, match=message):
        config.validate()


def test_run_training_writes_selected_checkpoint_and_summary(tmp_path):
    source_path = tmp_path / "source.txt"
    source_path.write_text("abcd" * 150, encoding="utf-8", newline="\n")
    data_dir = tmp_path / "data"
    prepare_dataset(source_path, data_dir)
    output_dir = tmp_path / "run"
    config = TrainingConfig(
        data_dir=data_dir,
        output_dir=output_dir,
        device="cpu",
        seed=23,
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
        sample_tokens=12,
        temperature=0.8,
        source_commit="test-commit",
    )

    summary = run_training(config)

    assert summary["source_commit"] == "test-commit"
    assert summary["route"] == "Direct-Small"
    assert summary["run_identity"]["deployment_interpretation"] == (
        "fp32_deployment_candidate"
    )
    assert summary["model"]["type"] == "direct_small_gpt"
    assert summary["model"]["parameters"] == 3_312
    assert summary["metrics"]["selected_validation_loss"] < (
        summary["metrics"]["initial_validation_loss"]
    )
    assert math.isfinite(summary["metrics"]["final_step_validation_loss"])
    assert summary["metrics"]["training_tokens"] == 320
    assert summary["sample"]["characters"] == 13
    assert summary["deployment"]["estimated_file_eligible"]
    assert summary["deployment"]["host_c_alignment"]["status"] == "pending"
    assert summary["deployment"]["nspire_peak_ram"]["status"] == "pending"
    assert summary["evaluation_history"][0]["step"] == 0
    assert summary["evaluation_history"][-1]["step"] == config.steps
    assert (output_dir / "direct_small_gpt.pt").is_file()
    assert (output_dir / "run.json").is_file()

    checkpoint = torch.load(
        output_dir / "direct_small_gpt.pt",
        map_location="cpu",
        weights_only=True,
    )
    assert checkpoint["source_commit"] == "test-commit"
    assert checkpoint["route"] == "Direct-Small"
    restored = DirectSmallGPT(DirectSmallConfig(**checkpoint["model_config"]))
    restored.load_state_dict(checkpoint["model_state_dict"], strict=True)
    assert restored.token_embedding.weight is restored.lm_head.weight
