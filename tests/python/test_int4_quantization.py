import pytest
import torch

from nanogpt_nspire.quantization.int4 import (
    GroupwiseInt4Tensor,
    dequantize_groupwise_int4,
    pack_signed_int4,
    quantize_groupwise_int4,
    unpack_signed_int4,
)


def test_signed_nibble_pack_order_and_complete_round_trip() -> None:
    values = torch.arange(-8, 8, dtype=torch.int8)

    packed = pack_signed_int4(values)

    assert packed.dtype == torch.uint8
    assert packed.tolist() == [
        0x98,
        0xBA,
        0xDC,
        0xFE,
        0x10,
        0x32,
        0x54,
        0x76,
    ]
    assert torch.equal(unpack_signed_int4(packed, count=16), values)


def test_signed_nibble_pack_pads_only_the_high_nibble() -> None:
    values = torch.tensor([-8, -1, 7], dtype=torch.int8)

    packed = pack_signed_int4(values)

    assert packed.tolist() == [0xF8, 0x07]
    assert torch.equal(unpack_signed_int4(packed, count=3), values)


@pytest.mark.parametrize(
    ("values", "message"),
    [
        (torch.tensor([0.0]), "integer"),
        (torch.tensor([-9], dtype=torch.int8), r"\[-8, 7\]"),
        (torch.tensor([8], dtype=torch.int8), r"\[-8, 7\]"),
        (torch.zeros((1, 1), dtype=torch.int8), "one-dimensional"),
    ],
)
def test_signed_nibble_pack_rejects_invalid_values(
    values: torch.Tensor,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        pack_signed_int4(values)


def test_signed_nibble_unpack_rejects_malformed_metadata() -> None:
    with pytest.raises(ValueError, match="torch.uint8"):
        unpack_signed_int4(torch.tensor([0], dtype=torch.int8), count=1)
    with pytest.raises(ValueError, match="one-dimensional"):
        unpack_signed_int4(torch.zeros((1, 1), dtype=torch.uint8), count=1)
    with pytest.raises(ValueError, match="non-negative integer"):
        unpack_signed_int4(torch.tensor([0], dtype=torch.uint8), count=-1)
    with pytest.raises(ValueError, match="does not match"):
        unpack_signed_int4(torch.tensor([0], dtype=torch.uint8), count=3)


def test_groupwise_int4_is_deterministic_and_bounded() -> None:
    weights = torch.tensor(
        [
            [-1.0, -0.5, 0.0, 0.5, 1.0],
            [0.0, 0.0, 0.0, 0.0, 0.0],
        ],
        dtype=torch.float32,
    )

    first = quantize_groupwise_int4(weights, group_size=4)
    second = quantize_groupwise_int4(weights, group_size=4)
    reconstructed = dequantize_groupwise_int4(first)

    assert first.shape == (2, 5)
    assert first.group_size == 4
    assert first.padded_last_dim == 8
    assert first.packed_bytes == 8
    assert first.scale_bytes == 16
    assert torch.equal(first.packed, second.packed)
    assert torch.equal(first.scales, second.scales)
    assert first.scales.dtype == torch.float32
    assert torch.isfinite(first.scales).all()
    assert first.scales[1].tolist() == [1.0, 1.0]
    assert torch.count_nonzero(reconstructed[1]) == 0

    scale_per_value = (
        first.scales.unsqueeze(-1)
        .expand(2, 2, 4)
        .reshape(2, 8)[:, :5]
    )
    assert torch.all(
        (weights - reconstructed).abs() <= scale_per_value / 2 + 1e-7
    )


def test_groupwise_int4_payload_round_trip_and_validation() -> None:
    weights = torch.linspace(-2.0, 2.0, 15).reshape(3, 5)
    quantized = quantize_groupwise_int4(weights, group_size=4)

    restored = GroupwiseInt4Tensor.from_payload(quantized.to_payload())

    assert torch.equal(restored.packed, quantized.packed)
    assert torch.equal(restored.scales, quantized.scales)
    assert torch.equal(
        dequantize_groupwise_int4(restored),
        dequantize_groupwise_int4(quantized),
    )

    malformed = quantized.to_payload()
    malformed["padded_last_dim"] = 4
    with pytest.raises(ValueError, match="padded_last_dim"):
        GroupwiseInt4Tensor.from_payload(malformed)


def test_groupwise_int4_rejects_unsupported_input() -> None:
    with pytest.raises(ValueError, match="floating-point"):
        quantize_groupwise_int4(torch.ones(3, dtype=torch.int64))
    with pytest.raises(ValueError, match="at least one dimension"):
        quantize_groupwise_int4(torch.tensor(1.0))
    with pytest.raises(ValueError, match="non-empty"):
        quantize_groupwise_int4(torch.empty(2, 0))
    with pytest.raises(ValueError, match="positive integer"):
        quantize_groupwise_int4(torch.ones(2, 2), group_size=0)
