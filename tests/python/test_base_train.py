import json
import math
from pathlib import Path
import struct

import pytest
import torch

from nanogpt_nspire.base_corpus import (
    CorpusRecord,
    build_corpus,
    stable_family_split,
)
from nanogpt_nspire.base_train import (
    BaseTrainingConfig,
    evaluate_frequency_baseline,
    frozen_student_base_config,
    load_packed_split,
    make_packed_batch,
    masked_cross_entropy,
    run_base_training,
)


REGISTRY_PATH = (
    Path(__file__).parents[2] / "experiments" / "lesson10-public-sources.json"
)


def _family_for_split(split: str, seed: str) -> str:
    for index in range(100_000):
        family = f"{split}-{index}"
        if stable_family_split(family, split_seed=seed) == split:
            return family
    raise AssertionError("could not find family")


def _training_data(tmp_path: Path) -> Path:
    seed = "base-train-test"
    records = []
    for split in ("train", "validation", "test"):
        family = _family_for_split(split, seed)
        records.append(
            CorpusRecord.base(
                record_id=f"record-{split}",
                family_id=family,
                text=(
                    f"This is the {split} educational document. "
                    + "Force equals mass times acceleration. " * 80
                ),
                source_id="project-arithmetic-v1",
                license_id="MIT",
            )
        )
    output = tmp_path / "data"
    build_corpus(
        records,
        output,
        registry_path=REGISTRY_PATH,
        split_seed=seed,
    )
    return output


def test_little_endian_split_and_shifted_loss_mask(tmp_path) -> None:
    tokens = (256, 65, 66, 257, 256, 67, 68, 257)
    masks = (0, 1, 1, 1, 0, 1, 1, 1)
    token_path = tmp_path / "tokens.bin"
    mask_path = tmp_path / "loss.bin"
    token_path.write_bytes(struct.pack(f"<{len(tokens)}H", *tokens))
    mask_path.write_bytes(bytes(masks))
    split = load_packed_split(
        token_path,
        mask_path,
        vocab_size=264,
    )
    generator = torch.Generator(device="cpu").manual_seed(3)

    batch = make_packed_batch(
        split,
        batch_size=1,
        block_size=3,
        generator=generator,
        device=torch.device("cpu"),
        starts=torch.tensor([0]),
    )

    assert batch.inputs.tolist() == [[256, 65, 66]]
    assert batch.targets.tolist() == [[65, 66, 257]]
    assert batch.target_mask.tolist() == [[True, True, True]]
    assert batch.starts.tolist() == [0]


def test_split_rejects_malformed_or_out_of_vocabulary_data(tmp_path) -> None:
    token_path = tmp_path / "tokens.bin"
    mask_path = tmp_path / "loss.bin"
    token_path.write_bytes(b"\x00")
    mask_path.write_bytes(b"\x01")
    with pytest.raises(ValueError, match="even"):
        load_packed_split(token_path, mask_path, vocab_size=264)

    token_path.write_bytes(struct.pack("<2H", 0, 264))
    mask_path.write_bytes(b"\x01\x01")
    with pytest.raises(ValueError, match="vocabulary"):
        load_packed_split(token_path, mask_path, vocab_size=264)

    token_path.write_bytes(struct.pack("<2H", 0, 1))
    mask_path.write_bytes(b"\x01")
    with pytest.raises(ValueError, match="same token count"):
        load_packed_split(token_path, mask_path, vocab_size=264)


def test_batches_are_seed_deterministic(tmp_path) -> None:
    data_dir = _training_data(tmp_path)
    split = load_packed_split(
        data_dir / "train.tokens.bin",
        data_dir / "train.loss.bin",
        vocab_size=264,
    )
    first_generator = torch.Generator().manual_seed(11)
    second_generator = torch.Generator().manual_seed(11)

    first = make_packed_batch(
        split,
        batch_size=3,
        block_size=16,
        generator=first_generator,
        device=torch.device("cpu"),
    )
    second = make_packed_batch(
        split,
        batch_size=3,
        block_size=16,
        generator=second_generator,
        device=torch.device("cpu"),
    )

    assert torch.equal(first.starts, second.starts)
    assert torch.equal(first.inputs, second.inputs)
    assert torch.equal(first.targets, second.targets)
    assert torch.equal(first.target_mask, second.target_mask)


def test_masked_cross_entropy_ignores_ineligible_targets() -> None:
    logits = torch.tensor(
        [[[5.0, 0.0], [0.0, 0.0]]],
        requires_grad=True,
    )
    targets = torch.tensor([[0, 1]])
    target_mask = torch.tensor([[True, False]])

    loss = masked_cross_entropy(logits, targets, target_mask)

    expected = torch.nn.functional.cross_entropy(
        logits[:, :1, :].reshape(-1, 2),
        targets[:, :1].reshape(-1),
    )
    assert torch.allclose(loss, expected)
    loss.backward()
    assert logits.grad is not None
    assert torch.equal(logits.grad[:, 1, :], torch.zeros((1, 2)))


def test_frequency_baseline_is_finite_and_deterministic(tmp_path) -> None:
    data_dir = _training_data(tmp_path)
    train = load_packed_split(
        data_dir / "train.tokens.bin",
        data_dir / "train.loss.bin",
        vocab_size=264,
    )
    validation = load_packed_split(
        data_dir / "validation.tokens.bin",
        data_dir / "validation.loss.bin",
        vocab_size=264,
    )

    first = evaluate_frequency_baseline(train, validation, vocab_size=264)
    second = evaluate_frequency_baseline(train, validation, vocab_size=264)

    assert first == second
    assert math.isfinite(first)
    assert first > 0.0


def test_frozen_student_config_uses_budgeted_architecture(tmp_path) -> None:
    config = frozen_student_base_config(
        data_dir=tmp_path / "data",
        output_dir=tmp_path / "out",
        source_commit="abc123",
    )

    assert config.block_size == 256
    assert config.n_layer == 6
    assert config.n_head == 6
    assert config.n_embd == 384
    assert config.vocab_size == 264
    assert config.effective_batch_tokens == (
        config.micro_batch_size
        * config.gradient_accumulation_steps
        * config.block_size
    )


def test_cpu_smoke_training_writes_best_checkpoint_and_metrics(tmp_path) -> None:
    data_dir = _training_data(tmp_path)
    output_dir = tmp_path / "run"
    config = BaseTrainingConfig(
        data_dir=data_dir,
        output_dir=output_dir,
        source_commit="test-source",
        device="cpu",
        seed=23,
        steps=6,
        micro_batch_size=2,
        gradient_accumulation_steps=1,
        block_size=8,
        n_layer=1,
        n_head=2,
        n_embd=16,
        mlp_ratio=2,
        dropout=0.0,
        learning_rate=0.01,
        min_learning_rate=0.001,
        warmup_steps=1,
        eval_interval=3,
        eval_batches=2,
        log_interval=1,
        overfit_gate_steps=3,
        sample_tokens=4,
    )

    summary = run_base_training(config)

    assert summary["route"] == "English-Base-Pilot"
    assert summary["model"]["parameters"] > 0
    assert summary["metrics"]["training_tokens"] == 6 * 2 * 1 * 8
    assert summary["metrics"]["selected_validation_loss"] < (
        summary["baselines"]["uniform_loss"]
    )
    assert summary["overfit_gate"]["passed"] is True
    assert summary["dataset"]["vocab_size"] == 264
    assert (output_dir / "base_pilot_best.pt").is_file()
    run_json = json.loads(
        (output_dir / "run.json").read_text(encoding="utf-8")
    )
    assert run_json["artifacts"]["checkpoint"]["sha256"] == (
        summary["artifacts"]["checkpoint"]["sha256"]
    )
