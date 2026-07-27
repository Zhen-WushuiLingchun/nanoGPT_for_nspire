"""Generate the cross-language tiny model fixture in the build tree."""

from __future__ import annotations

import argparse
from pathlib import Path
import struct
from typing import Sequence

from nanogpt_nspire.export_format import (
    ACTIVATION_NONE,
    MODEL_STORAGE_FP32,
    STORAGE_FP32,
    ModelSpec,
    TensorPayload,
    build_model_file,
)


def fixture_tensors() -> tuple[TensorPayload, ...]:
    shapes = (
        (1, (3, 4)),
        (2, (4, 4)),
        (100, (4,)),
        (101, (12, 4)),
        (102, (4, 4)),
        (103, (4,)),
        (104, (8, 4)),
        (105, (4, 8)),
        (1000, (4,)),
    )
    tensors = []
    value = 1
    for tensor_id, shape in shapes:
        element_count = 1
        for dimension in shape:
            element_count *= dimension
        values = [
            ((value + index) % 17 - 8) / 16.0
            for index in range(element_count)
        ]
        value += element_count
        tensors.append(
            TensorPayload(
                tensor_id=tensor_id,
                storage=STORAGE_FP32,
                shape=shape,
                data=struct.pack(f"<{element_count}f", *values),
            )
        )
    return tuple(tensors)


def write_fixture(output: Path) -> None:
    data = build_model_file(
        spec=ModelSpec(
            vocab_size=3,
            block_size=4,
            n_layer=1,
            n_head=2,
            n_embd=4,
            mlp_ratio=2,
            tie_embeddings=True,
            bias=False,
            model_storage=MODEL_STORAGE_FP32,
            weight_group_size=0,
            activation_quantization=ACTIVATION_NONE,
            activation_group_size=0,
        ),
        vocabulary=("\n", "a", "é"),
        tensors=fixture_tensors(),
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(data)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args(argv)
    write_fixture(arguments.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
