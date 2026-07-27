import math

import pytest
import torch
from torch import nn

from nanogpt_nspire.data import prepare_dataset
from nanogpt_nspire.lesson03_train import (
    TrainingConfig,
    run_training,
    sample_with_context,
)


class _RecordingModel(nn.Module):
    vocab_size = 3
    block_size = 3

    def __init__(self):
        super().__init__()
        self.contexts: list[torch.Tensor] = []

    def forward(self, token_ids, targets=None):
        self.contexts.append(token_ids.detach().cpu().clone())
        logits = torch.full(
            (*token_ids.shape, self.vocab_size),
            -20.0,
            device=token_ids.device,
        )
        next_ids = (token_ids + 1) % self.vocab_size
        logits.scatter_(-1, next_ids.unsqueeze(-1), 20.0)
        return logits, None


def test_context_sampler_crops_to_block_size_and_is_reproducible():
    first_model = _RecordingModel()
    second_model = _RecordingModel()

    first = sample_with_context(
        first_model,
        [0, 1, 2, 0],
        new_tokens=5,
        seed=41,
        temperature=0.8,
        device=torch.device("cpu"),
    )
    second = sample_with_context(
        second_model,
        [0, 1, 2, 0],
        new_tokens=5,
        seed=41,
        temperature=0.8,
        device=torch.device("cpu"),
    )

    assert first == second
    assert first_model.contexts[0].tolist() == [[1, 2, 0]]
    assert all(context.shape[1] <= first_model.block_size for context in first_model.contexts)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("steps", 0, "steps"),
        ("embedding_dim", 0, "embedding_dim"),
        ("temperature", 0.0, "temperature"),
    ],
)
def test_training_config_rejects_invalid_values(tmp_path, field, value, message):
    values = {
        "data_dir": tmp_path,
        "output_dir": tmp_path / "out",
        field: value,
    }
    config = TrainingConfig(**values)

    with pytest.raises(ValueError, match=message):
        config.validate()


def test_run_training_writes_attention_checkpoint_and_summary(tmp_path):
    source_path = tmp_path / "source.txt"
    source_path.write_text("ab" * 100, encoding="utf-8", newline="\n")
    data_dir = tmp_path / "data"
    prepare_dataset(source_path, data_dir)
    output_dir = tmp_path / "run"
    config = TrainingConfig(
        data_dir=data_dir,
        output_dir=output_dir,
        device="cpu",
        seed=29,
        steps=20,
        batch_size=4,
        block_size=8,
        embedding_dim=8,
        learning_rate=0.05,
        eval_batches=2,
        sample_tokens=12,
        temperature=0.8,
        source_commit="test-commit",
    )

    summary = run_training(config)

    assert summary["source_commit"] == "test-commit"
    assert summary["model"]["type"] == "single_head_causal_attention_lm"
    assert summary["model"]["parameters"] == 352
    assert math.isfinite(summary["metrics"]["final_validation_loss"])
    assert summary["sample"]["characters"] == 13
    assert (output_dir / "single_head_attention_lm.pt").is_file()
    assert (output_dir / "run.json").is_file()
