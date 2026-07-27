import json

import pytest
import torch

from nanogpt_nspire.data import DatasetError, prepare_dataset
from nanogpt_nspire.training_dataset import load_token_dataset, make_batch


def _prepared_fixture(tmp_path):
    source_path = tmp_path / "source.txt"
    source_path.write_text("ab\nab\nab\n", encoding="utf-8", newline="\n")
    output_dir = tmp_path / "dataset"
    prepare_dataset(source_path, output_dir)
    return output_dir


def test_load_token_dataset_validates_and_decodes_artifacts(tmp_path):
    output_dir = _prepared_fixture(tmp_path)

    dataset = load_token_dataset(output_dir)

    assert dataset.vocabulary == ("\n", "a", "b")
    assert dataset.train.dtype == torch.long
    assert dataset.validation.dtype == torch.long
    assert dataset.train.tolist() == [1, 2, 0, 1, 2, 0, 1, 2]
    assert dataset.validation.tolist() == [0]
    assert dataset.manifest["vocab_size"] == 3


def test_load_token_dataset_rejects_tampered_token_file(tmp_path):
    output_dir = _prepared_fixture(tmp_path)
    train_path = output_dir / "train.bin"
    train_path.write_bytes(train_path.read_bytes() + b"\x00\x00")

    with pytest.raises(DatasetError, match="train.bin byte length"):
        load_token_dataset(output_dir)


def test_load_token_dataset_rejects_unsupported_manifest(tmp_path):
    output_dir = _prepared_fixture(tmp_path)
    manifest_path = output_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["dtype"] = "uint8"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(DatasetError, match="dtype"):
        load_token_dataset(output_dir)


def test_make_batch_is_shifted_and_seeded():
    tokens = torch.arange(20, dtype=torch.long)
    first_generator = torch.Generator(device="cpu").manual_seed(123)
    second_generator = torch.Generator(device="cpu").manual_seed(123)

    first_x, first_y = make_batch(
        tokens,
        batch_size=3,
        block_size=4,
        generator=first_generator,
    )
    second_x, second_y = make_batch(
        tokens,
        batch_size=3,
        block_size=4,
        generator=second_generator,
    )

    assert first_x.shape == (3, 4)
    assert first_y.shape == (3, 4)
    assert torch.equal(first_y[:, :-1], first_x[:, 1:])
    assert torch.equal(first_x, second_x)
    assert torch.equal(first_y, second_y)


@pytest.mark.parametrize(
    ("batch_size", "block_size", "message"),
    [
        (0, 4, "batch_size"),
        (2, 0, "block_size"),
        (2, 20, "block_size plus one"),
    ],
)
def test_make_batch_rejects_invalid_requests(batch_size, block_size, message):
    tokens = torch.arange(20, dtype=torch.long)
    generator = torch.Generator(device="cpu").manual_seed(1)

    with pytest.raises(DatasetError, match=message):
        make_batch(
            tokens,
            batch_size=batch_size,
            block_size=block_size,
            generator=generator,
        )
