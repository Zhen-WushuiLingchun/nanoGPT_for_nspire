import pytest
import torch

from nanogpt_nspire.models.embedding_lm import EmbeddingLanguageModel, ModelInputError


def test_forward_produces_vocab_logits_and_scalar_loss():
    torch.manual_seed(7)
    model = EmbeddingLanguageModel(vocab_size=5, embedding_dim=8)
    token_ids = torch.tensor([[0, 1, 2], [2, 3, 4]], dtype=torch.long)
    targets = torch.tensor([[1, 2, 3], [3, 4, 0]], dtype=torch.long)

    logits, loss = model(token_ids, targets)

    assert logits.shape == (2, 3, 5)
    assert loss is not None
    assert loss.ndim == 0
    assert torch.isfinite(loss)
    assert model.parameter_count == 2 * 5 * 8


def test_forward_without_targets_returns_logits_only():
    model = EmbeddingLanguageModel(vocab_size=3, embedding_dim=4)
    token_ids = torch.tensor([[0, 1]], dtype=torch.long)

    logits, loss = model(token_ids)

    assert logits.shape == (1, 2, 3)
    assert loss is None


@pytest.mark.parametrize(
    ("token_ids", "targets", "message"),
    [
        (
            torch.tensor([0, 1], dtype=torch.long),
            None,
            "two-dimensional",
        ),
        (
            torch.tensor([[0.0, 1.0]]),
            None,
            "torch.long",
        ),
        (
            torch.tensor([[0, 3]], dtype=torch.long),
            None,
            "outside",
        ),
        (
            torch.tensor([[0, 1]], dtype=torch.long),
            torch.tensor([[1]], dtype=torch.long),
            "same shape",
        ),
    ],
)
def test_forward_rejects_invalid_inputs(token_ids, targets, message):
    model = EmbeddingLanguageModel(vocab_size=3, embedding_dim=4)

    with pytest.raises(ModelInputError, match=message):
        model(token_ids, targets)


def test_embedding_model_learns_deterministic_ab_transitions():
    torch.manual_seed(11)
    model = EmbeddingLanguageModel(vocab_size=2, embedding_dim=8)
    inputs = torch.tensor([[0, 1] * 8] * 8, dtype=torch.long)
    targets = 1 - inputs
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=0.1,
        weight_decay=0.0,
    )

    _, initial_loss = model(inputs, targets)
    assert initial_loss is not None
    for _ in range(100):
        _, loss = model(inputs, targets)
        assert loss is not None
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    _, final_loss = model(inputs, targets)

    assert final_loss is not None
    assert final_loss.item() < initial_loss.item() * 0.1
