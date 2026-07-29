from __future__ import annotations

import pytest
import torch

from nanogpt_nspire.context_position_eval import (
    summarize_position_losses,
)


def test_position_loss_buckets_respect_mask_and_boundary() -> None:
    losses = torch.tensor(
        [
            [1.0, 2.0, 3.0, 4.0],
            [10.0, 20.0, 30.0, 40.0],
        ]
    )
    mask = torch.tensor(
        [
            [True, False, True, True],
            [False, True, False, True],
        ]
    )

    result = summarize_position_losses(losses, mask, boundary=2)

    assert result["positions_0_255"] == {
        "eligible_targets": 2,
        "loss_sum": 21.0,
    }
    assert result["positions_256_511"] == {
        "eligible_targets": 3,
        "loss_sum": 47.0,
    }


def test_position_loss_bucket_validation() -> None:
    losses = torch.zeros((1, 4))
    mask = torch.ones((1, 4), dtype=torch.bool)

    with pytest.raises(ValueError, match="boundary"):
        summarize_position_losses(losses, mask, boundary=4)
    with pytest.raises(ValueError, match="boolean"):
        summarize_position_losses(losses, mask.float(), boundary=2)
