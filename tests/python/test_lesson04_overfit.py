import math

import pytest
import torch

from nanogpt_nspire.data import prepare_dataset
from nanogpt_nspire.lesson04_overfit import TrainingConfig, run_overfit_experiment


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("steps", 0, "steps"),
        ("batch_size", 0, "batch_size"),
        ("record_every", 0, "record_every"),
        ("learning_rate", 0.0, "learning_rate"),
        ("max_grad_norm", 0.0, "max_grad_norm"),
        ("target_training_loss", 0.0, "target_training_loss"),
    ],
)
def test_training_config_rejects_invalid_values(
    tmp_path,
    field,
    value,
    message,
):
    values = {
        "data_dir": tmp_path,
        "output_dir": tmp_path / "out",
        field: value,
    }
    config = TrainingConfig(**values)

    with pytest.raises(ValueError, match=message):
        config.validate()


def test_run_overfit_experiment_writes_reproducible_artifacts(tmp_path):
    source_path = tmp_path / "source.txt"
    source_path.write_text("abcd" * 100, encoding="utf-8", newline="\n")
    data_dir = tmp_path / "data"
    prepare_dataset(source_path, data_dir)
    output_dir = tmp_path / "run"
    config = TrainingConfig(
        data_dir=data_dir,
        output_dir=output_dir,
        device="cpu",
        seed=31,
        steps=40,
        batch_size=1,
        block_size=8,
        embedding_dim=16,
        learning_rate=0.05,
        max_grad_norm=1.0,
        record_every=10,
        eval_batches=2,
        target_training_loss=0.1,
        source_commit="test-commit",
    )

    summary = run_overfit_experiment(config)

    assert summary["source_commit"] == "test-commit"
    assert summary["experiment"]["type"] == "fixed_batch_overfit"
    assert summary["model"]["type"] == "single_head_causal_attention_lm"
    assert summary["fixed_batch"]["shape"] == [
        config.batch_size,
        config.block_size,
    ]
    assert len(summary["fixed_batch"]["input_token_ids"]) == config.batch_size
    assert len(summary["fixed_batch"]["target_token_ids"]) == config.batch_size
    assert summary["metrics"]["final_fixed_batch_loss"] < (
        summary["metrics"]["initial_fixed_batch_loss"]
    )
    assert math.isfinite(summary["metrics"]["final_validation_loss"])
    assert math.isfinite(summary["metrics"]["generalization_gap"])
    assert summary["metrics"]["parameter_displacement_l2_norm"] > 0.0
    assert summary["history"][0]["step"] == 0
    assert summary["history"][-1]["step"] == config.steps
    assert (output_dir / "overfit_attention_lm.pt").is_file()
    assert (output_dir / "run.json").is_file()

    checkpoint = torch.load(
        output_dir / "overfit_attention_lm.pt",
        map_location="cpu",
        weights_only=True,
    )
    assert checkpoint["source_commit"] == "test-commit"
    assert checkpoint["fixed_batch"]["inputs"].shape == (
        config.batch_size,
        config.block_size,
    )
    assert checkpoint["fixed_batch"]["targets"].shape == (
        config.batch_size,
        config.block_size,
    )
