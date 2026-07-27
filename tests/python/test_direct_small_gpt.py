import math

import pytest
import torch

from nanogpt_nspire.models.direct_small_gpt import (
    DirectSmallConfig,
    DirectSmallGPT,
    MultiHeadCausalSelfAttention,
)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("vocab_size", 0, "vocab_size"),
        ("block_size", 0, "block_size"),
        ("n_layer", 0, "n_layer"),
        ("n_head", 0, "n_head"),
        ("n_embd", 0, "n_embd"),
        ("mlp_ratio", 0, "mlp_ratio"),
        ("dropout", -0.1, "dropout"),
        ("dropout", 1.0, "dropout"),
        ("bias", 1, "bias"),
        ("tie_embeddings", 1, "tie_embeddings"),
    ],
)
def test_config_rejects_invalid_values(field, value, message):
    config = DirectSmallConfig(**{field: value})

    with pytest.raises(ValueError, match=message):
        config.validate()


def test_config_requires_embedding_width_divisible_by_head_count():
    config = DirectSmallConfig(n_embd=15, n_head=4)

    with pytest.raises(ValueError, match="divisible"):
        config.validate()


def test_multi_head_attention_has_normalized_causal_weights():
    torch.manual_seed(3)
    config = DirectSmallConfig(
        vocab_size=7,
        block_size=8,
        n_layer=1,
        n_head=2,
        n_embd=8,
        dropout=0.0,
    )
    attention = MultiHeadCausalSelfAttention(config)
    inputs = torch.randn(2, 5, 8)

    output, weights = attention(inputs, return_weights=True)

    assert output.shape == (2, 5, 8)
    assert weights.shape == (2, 2, 5, 5)
    assert torch.count_nonzero(torch.triu(weights, diagonal=1)) == 0
    assert torch.allclose(
        weights.sum(dim=-1),
        torch.ones(2, 2, 5),
        atol=1e-6,
    )


def test_frozen_direct_small_budget_and_tied_weights_are_exact():
    config = DirectSmallConfig()
    model = DirectSmallGPT(config)

    assert config == DirectSmallConfig(
        vocab_size=65,
        block_size=128,
        n_layer=4,
        n_head=5,
        n_embd=160,
        mlp_ratio=4,
        dropout=0.1,
        bias=False,
        tie_embeddings=True,
    )
    assert model.parameter_count == 1_261_120
    assert model.expected_parameter_count == 1_261_120
    assert model.raw_fp32_parameter_bytes == 5_044_480
    assert model.token_embedding.weight is model.lm_head.weight
    assert model.token_embedding.weight.data_ptr() == model.lm_head.weight.data_ptr()
    assert not any(key.endswith("causal_mask") for key in model.state_dict())


def test_future_tokens_do_not_change_earlier_logits_in_eval_mode():
    torch.manual_seed(5)
    model = DirectSmallGPT(
        DirectSmallConfig(
            vocab_size=9,
            block_size=8,
            n_layer=2,
            n_head=2,
            n_embd=16,
            dropout=0.2,
        )
    )
    model.eval()
    first = torch.tensor([[1, 2, 3, 4, 5]], dtype=torch.long)
    second = torch.tensor([[1, 2, 3, 8, 7]], dtype=torch.long)

    with torch.inference_mode():
        first_logits, _ = model(first)
        second_logits, _ = model(second)

    assert torch.equal(first_logits[:, :3], second_logits[:, :3])


def test_forward_backward_shapes_loss_and_gradients():
    torch.manual_seed(11)
    config = DirectSmallConfig(
        vocab_size=5,
        block_size=6,
        n_layer=2,
        n_head=2,
        n_embd=16,
        dropout=0.0,
    )
    model = DirectSmallGPT(config)
    inputs = torch.tensor(
        [[0, 1, 2, 3, 4, 0], [4, 3, 2, 1, 0, 4]],
        dtype=torch.long,
    )
    targets = torch.roll(inputs, shifts=-1, dims=1)

    logits, loss = model(inputs, targets)

    assert logits.shape == (2, 6, 5)
    assert loss is not None
    assert loss.ndim == 0
    assert math.isfinite(float(loss.item()))
    loss.backward()
    populated_gradients = [
        parameter.grad
        for parameter in model.parameters()
        if parameter.requires_grad and parameter.grad is not None
    ]
    assert populated_gradients
    assert all(torch.isfinite(gradient).all() for gradient in populated_gradients)


def test_small_complete_gpt_can_memorize_contextual_fixed_batch():
    torch.manual_seed(17)
    model = DirectSmallGPT(
        DirectSmallConfig(
            vocab_size=4,
            block_size=4,
            n_layer=1,
            n_head=2,
            n_embd=16,
            dropout=0.0,
        )
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=0.02,
        weight_decay=0.0,
    )
    inputs = torch.tensor([[0, 1, 0, 2], [3, 1, 3, 2]], dtype=torch.long)
    targets = torch.tensor([[1, 0, 2, 0], [1, 3, 2, 3]], dtype=torch.long)
    with torch.no_grad():
        _, initial_loss_tensor = model(inputs, targets)
        assert initial_loss_tensor is not None
        initial_loss = float(initial_loss_tensor.item())

    for _ in range(150):
        _, loss = model(inputs, targets)
        assert loss is not None
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.inference_mode():
        logits, final_loss_tensor = model(inputs, targets)
        assert final_loss_tensor is not None
        final_loss = float(final_loss_tensor.item())
        accuracy = float(
            (torch.argmax(logits, dim=-1) == targets).float().mean().item()
        )

    assert final_loss < initial_loss * 0.05
    assert accuracy == 1.0


@pytest.mark.parametrize(
    "token_ids",
    [
        torch.tensor([0, 1], dtype=torch.long),
        torch.empty((1, 0), dtype=torch.long),
        torch.tensor([[0.0, 1.0]]),
        torch.tensor([[0, 65]], dtype=torch.long),
        torch.zeros((1, 129), dtype=torch.long),
    ],
)
def test_model_rejects_invalid_token_inputs(token_ids):
    model = DirectSmallGPT(DirectSmallConfig())

    with pytest.raises(ValueError):
        model(token_ids)
