import math

import pytest
import torch
from torch import nn
from torch.nn import functional as F

from nanogpt_nspire.training_loop import (
    evaluate_batch,
    gradient_l2_norm,
    overfit_fixed_batch,
    train_step,
)


class _TinyLanguageModel(nn.Module):
    def __init__(self, vocab_size=4, embedding_dim=6):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.head = nn.Linear(embedding_dim, vocab_size)
        self.forward_training_modes: list[bool] = []

    def forward(self, token_ids, targets=None):
        self.forward_training_modes.append(self.training)
        logits = self.head(self.embedding(token_ids))
        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                targets.reshape(-1),
            )
        return logits, loss


class _PositionLogitsLanguageModel(nn.Module):
    def __init__(self, block_size=4, vocab_size=4):
        super().__init__()
        self.logits_by_position = nn.Parameter(torch.zeros(block_size, vocab_size))

    def forward(self, token_ids, targets=None):
        logits = self.logits_by_position[: token_ids.shape[1]]
        logits = logits.unsqueeze(0).expand(token_ids.shape[0], -1, -1)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                targets.reshape(-1),
            )
        return logits, loss


def _example_batch():
    inputs = torch.tensor([[0, 1, 2, 3]], dtype=torch.long)
    targets = torch.tensor([[1, 2, 3, 0]], dtype=torch.long)
    return inputs, targets


def test_train_step_reports_gradients_clipping_accuracy_and_parameter_update():
    torch.manual_seed(7)
    model = _TinyLanguageModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.2)
    inputs, targets = _example_batch()

    metrics = train_step(
        model,
        optimizer,
        inputs,
        targets,
        max_grad_norm=0.05,
    )

    assert metrics.loss > 0.0
    assert metrics.gradient_l2_norm_before_clip > 0.05
    assert metrics.gradient_l2_norm_after_clip <= 0.05 + 1e-6
    assert gradient_l2_norm(model.parameters()) == pytest.approx(
        metrics.gradient_l2_norm_after_clip,
        abs=1e-8,
    )
    assert metrics.parameter_update_l2_norm > 0.0
    assert 0.0 <= metrics.token_accuracy <= 1.0
    assert model.training


@pytest.mark.parametrize("starts_in_training_mode", [False, True])
def test_evaluate_batch_restores_model_mode(starts_in_training_mode):
    model = _TinyLanguageModel()
    model.train(starts_in_training_mode)
    inputs, targets = _example_batch()

    metrics = evaluate_batch(model, inputs, targets)

    assert math.isfinite(metrics.loss)
    assert 0.0 <= metrics.token_accuracy <= 1.0
    assert model.training is starts_in_training_mode
    assert model.forward_training_modes[-1] is False
    assert all(parameter.grad is None for parameter in model.parameters())


@pytest.mark.parametrize("max_grad_norm", [0.0, -1.0, math.inf, math.nan])
def test_train_step_rejects_invalid_gradient_clip(max_grad_norm):
    model = _TinyLanguageModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    inputs, targets = _example_batch()

    with pytest.raises(ValueError, match="max_grad_norm"):
        train_step(
            model,
            optimizer,
            inputs,
            targets,
            max_grad_norm=max_grad_norm,
        )


def test_train_step_rejects_nonfinite_loss_before_parameter_update():
    class _NonFiniteModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.tensor(1.0))

        def forward(self, token_ids, targets=None):
            logits = self.weight * torch.ones((*token_ids.shape, 2))
            return logits, self.weight * torch.tensor(float("nan"))

    model = _NonFiniteModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=1.0)
    inputs = torch.tensor([[0]], dtype=torch.long)
    original_weight = model.weight.detach().clone()

    with pytest.raises(FloatingPointError, match="loss"):
        train_step(model, optimizer, inputs, inputs)

    assert torch.equal(model.weight.detach(), original_weight)


def test_evaluate_batch_requires_language_model_result():
    class _WrongResultModel(nn.Module):
        def forward(self, token_ids, targets=None):
            return torch.zeros((*token_ids.shape, 2))

    inputs = torch.tensor([[0]], dtype=torch.long)

    with pytest.raises(TypeError, match="return"):
        evaluate_batch(_WrongResultModel(), inputs, inputs)


def test_overfit_fixed_batch_memorizes_targets_and_records_observability():
    model = _PositionLogitsLanguageModel()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=0.2,
        weight_decay=0.0,
    )
    inputs, targets = _example_batch()

    result = overfit_fixed_batch(
        model,
        optimizer,
        inputs,
        targets,
        steps=100,
        record_every=25,
        max_grad_norm=1.0,
    )

    assert result.initial.loss == pytest.approx(math.log(4.0))
    assert result.final.loss < 0.01
    assert result.final.token_accuracy == 1.0
    assert [record["step"] for record in result.history] == [0, 1, 25, 50, 75, 100]
    assert result.history[0]["gradient_l2_norm_before_clip"] is None
    assert all(
        math.isfinite(record["fixed_batch_loss"])
        for record in result.history
    )
    assert all(
        record["parameter_update_l2_norm"] > 0.0
        for record in result.history[1:]
    )


@pytest.mark.parametrize(
    ("steps", "record_every", "message"),
    [
        (0, 1, "steps"),
        (1, 0, "record_every"),
    ],
)
def test_overfit_fixed_batch_rejects_invalid_intervals(
    steps,
    record_every,
    message,
):
    model = _PositionLogitsLanguageModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    inputs, targets = _example_batch()

    with pytest.raises(ValueError, match=message):
        overfit_fixed_batch(
            model,
            optimizer,
            inputs,
            targets,
            steps=steps,
            record_every=record_every,
        )
