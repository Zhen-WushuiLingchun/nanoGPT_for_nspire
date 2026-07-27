import math

import pytest
import torch

from nanogpt_nspire.data import prepare_dataset
from nanogpt_nspire.lesson02_train import (
    TrainingConfig,
    bits_per_character,
    evaluate_loss,
    resolve_device,
    run_training,
)
from nanogpt_nspire.models.embedding_lm import EmbeddingLanguageModel


def test_bits_per_character_converts_natural_log_loss():
    assert bits_per_character(math.log(8.0)) == pytest.approx(3.0)


def test_resolve_device_auto_matches_current_cuda_availability():
    expected = "cuda" if torch.cuda.is_available() else "cpu"

    assert resolve_device("auto").type == expected
    assert resolve_device("cpu").type == "cpu"


def test_evaluate_loss_reuses_identical_seeded_windows():
    torch.manual_seed(17)
    model = EmbeddingLanguageModel(vocab_size=5, embedding_dim=4)
    tokens = torch.arange(100, dtype=torch.long) % 5

    first = evaluate_loss(
        model,
        tokens,
        batch_size=4,
        block_size=8,
        batches=3,
        seed=99,
        device=torch.device("cpu"),
    )
    second = evaluate_loss(
        model,
        tokens,
        batch_size=4,
        block_size=8,
        batches=3,
        seed=99,
        device=torch.device("cpu"),
    )

    assert first == second


def test_training_config_rejects_nonpositive_values(tmp_path):
    config = TrainingConfig(
        data_dir=tmp_path,
        output_dir=tmp_path / "out",
        steps=0,
    )

    with pytest.raises(ValueError, match="steps"):
        config.validate()


def test_run_training_writes_checkpoint_and_summary(tmp_path):
    source_path = tmp_path / "source.txt"
    source_path.write_text("ab" * 100, encoding="utf-8", newline="\n")
    data_dir = tmp_path / "data"
    prepare_dataset(source_path, data_dir)
    output_dir = tmp_path / "run"
    config = TrainingConfig(
        data_dir=data_dir,
        output_dir=output_dir,
        device="cpu",
        seed=23,
        steps=20,
        batch_size=4,
        block_size=8,
        embedding_dim=4,
        learning_rate=0.1,
        eval_batches=2,
        sample_tokens=12,
        source_commit="test-commit",
    )

    summary = run_training(config)

    assert summary["source_commit"] == "test-commit"
    assert summary["model"]["parameters"] == 16
    assert summary["metrics"]["final_validation_loss"] < summary["metrics"][
        "initial_validation_loss"
    ]
    assert summary["sample"]["characters"] == 13
    assert (output_dir / "embedding_lm.pt").is_file()
    assert (output_dir / "run.json").is_file()
