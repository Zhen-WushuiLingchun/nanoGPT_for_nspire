import pytest
import torch

from nanogpt_nspire.models.causal_attention_lm import (
    SingleHeadCausalLanguageModel,
    SingleHeadCausalSelfAttention,
)
from nanogpt_nspire.models.embedding_lm import ModelInputError


def test_attention_weights_are_normalized_and_strictly_causal():
    torch.manual_seed(3)
    attention = SingleHeadCausalSelfAttention(embedding_dim=8, block_size=4)
    inputs = torch.randn(2, 4, 8)

    output, weights = attention(inputs, return_weights=True)

    assert output.shape == (2, 4, 8)
    assert weights.shape == (2, 4, 4)
    assert torch.count_nonzero(torch.triu(weights, diagonal=1)).item() == 0
    assert torch.allclose(
        weights.sum(dim=-1),
        torch.ones(2, 4),
        atol=1e-6,
    )


def test_future_tokens_cannot_change_earlier_logits():
    torch.manual_seed(5)
    model = SingleHeadCausalLanguageModel(
        vocab_size=5,
        embedding_dim=8,
        block_size=4,
    )
    first = torch.tensor([[0, 1, 2, 3]], dtype=torch.long)
    changed_future = torch.tensor([[0, 1, 2, 4]], dtype=torch.long)

    first_logits, _ = model(first)
    changed_logits, _ = model(changed_future)

    assert torch.allclose(
        first_logits[:, :3],
        changed_logits[:, :3],
        atol=1e-7,
        rtol=0.0,
    )


def test_model_shapes_loss_and_parameter_count():
    model = SingleHeadCausalLanguageModel(
        vocab_size=5,
        embedding_dim=8,
        block_size=4,
    )
    token_ids = torch.tensor([[0, 1, 2], [2, 3, 4]], dtype=torch.long)
    targets = torch.tensor([[1, 2, 3], [3, 4, 0]], dtype=torch.long)

    logits, loss = model(token_ids, targets)

    assert logits.shape == (2, 3, 5)
    assert loss is not None
    assert loss.ndim == 0
    assert torch.isfinite(loss)
    assert model.parameter_count == (2 * 5 * 8) + (4 * 8) + (4 * 8 * 8)


@pytest.mark.parametrize(
    ("token_ids", "targets", "message"),
    [
        (
            torch.tensor([[0, 1, 2, 3, 4]], dtype=torch.long),
            None,
            "block_size",
        ),
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
            torch.tensor([[0, 5]], dtype=torch.long),
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
def test_model_rejects_invalid_inputs(token_ids, targets, message):
    model = SingleHeadCausalLanguageModel(
        vocab_size=5,
        embedding_dim=8,
        block_size=4,
    )

    with pytest.raises(ModelInputError, match=message):
        model(token_ids, targets)


def test_attention_model_learns_target_from_earlier_context():
    torch.manual_seed(19)
    model = SingleHeadCausalLanguageModel(
        vocab_size=3,
        embedding_dim=12,
        block_size=2,
    )
    inputs = torch.tensor([[0, 2], [1, 2]] * 32, dtype=torch.long)
    targets = torch.tensor([[2, 0], [2, 1]] * 32, dtype=torch.long)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=0.05,
        weight_decay=0.0,
    )

    _, initial_loss = model(inputs, targets)
    assert initial_loss is not None
    for _ in range(150):
        _, loss = model(inputs, targets)
        assert loss is not None
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    _, final_loss = model(inputs, targets)

    assert final_loss is not None
    assert final_loss.item() < initial_loss.item() * 0.1
