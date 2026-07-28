"""Train the first 264-token English causal base-model pilot."""

from __future__ import annotations

import argparse
from contextlib import nullcontext
from dataclasses import asdict, dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import time
from typing import Iterator

import numpy as np
import torch
from torch.nn import functional as F

from nanogpt_nspire.byte_tokenizer import (
    BOS_ID,
    EOS_ID,
    VOCAB_SIZE,
    ByteTokenizer,
)
from nanogpt_nspire.direct_small_train import (
    configure_adamw,
    learning_rate_at_step,
)
from nanogpt_nspire.models.direct_small_gpt import (
    DirectSmallConfig,
    DirectSmallGPT,
)
from nanogpt_nspire.training_support import (
    resolve_device,
    sha256_file,
    synchronize,
    write_json_atomic,
)


ROUTE = "English-Base-Pilot"
CHECKPOINT_FILENAME = "base_pilot_best.pt"
UNIFORM_LOSS = math.log(VOCAB_SIZE)


def _positive_integer(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _non_negative_integer(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return value


@dataclass(frozen=True)
class PackedTokenSplit:
    """Memory-mapped token IDs and aligned target-eligibility bytes."""

    token_path: Path
    mask_path: Path
    tokens: np.memmap
    loss_mask: np.memmap
    vocab_size: int

    @property
    def token_count(self) -> int:
        return int(self.tokens.shape[0])


@dataclass(frozen=True)
class PackedDataset:
    """Validated train/validation/test views and their frozen manifest."""

    root: Path
    manifest_path: Path
    manifest: dict[str, object]
    train: PackedTokenSplit
    validation: PackedTokenSplit
    test: PackedTokenSplit


@dataclass(frozen=True)
class PackedBatch:
    """One shifted causal batch plus the mask of targets that count."""

    inputs: torch.Tensor
    targets: torch.Tensor
    target_mask: torch.Tensor
    starts: torch.Tensor


@dataclass(frozen=True)
class BaseTrainingConfig:
    """Reproducible architecture and optimizer contract for one base run."""

    data_dir: Path
    output_dir: Path
    source_commit: str
    device: str = "auto"
    seed: int = 20260728
    steps: int = 1000
    micro_batch_size: int = 4
    gradient_accumulation_steps: int = 4
    vocab_size: int = VOCAB_SIZE
    block_size: int = 256
    n_layer: int = 6
    n_head: int = 6
    n_embd: int = 384
    mlp_ratio: int = 4
    dropout: float = 0.1
    bias: bool = False
    tie_embeddings: bool = True
    learning_rate: float = 0.0006
    min_learning_rate: float = 0.00006
    warmup_steps: int = 50
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    max_grad_norm: float = 1.0
    eval_interval: int = 100
    eval_batches: int = 20
    log_interval: int = 25
    overfit_gate_steps: int = 20
    use_bfloat16: bool = True
    sample_tokens: int = 96
    temperature: float = 0.8
    top_k: int = 40
    sample_prompts: tuple[str, ...] = (
        "Force is",
        "The value of x",
        "Energy can be",
    )

    @property
    def effective_batch_tokens(self) -> int:
        return (
            self.micro_batch_size
            * self.gradient_accumulation_steps
            * self.block_size
        )

    def model_config(self) -> DirectSmallConfig:
        return DirectSmallConfig(
            vocab_size=self.vocab_size,
            block_size=self.block_size,
            n_layer=self.n_layer,
            n_head=self.n_head,
            n_embd=self.n_embd,
            mlp_ratio=self.mlp_ratio,
            dropout=self.dropout,
            bias=self.bias,
            tie_embeddings=self.tie_embeddings,
        )

    def validate(self) -> None:
        for name in (
            "steps",
            "micro_batch_size",
            "gradient_accumulation_steps",
            "vocab_size",
            "block_size",
            "n_layer",
            "n_head",
            "n_embd",
            "mlp_ratio",
            "warmup_steps",
            "eval_interval",
            "eval_batches",
            "log_interval",
            "overfit_gate_steps",
            "top_k",
        ):
            _positive_integer(getattr(self, name), name)
        _non_negative_integer(self.seed, "seed")
        _non_negative_integer(self.sample_tokens, "sample_tokens")
        if self.vocab_size != VOCAB_SIZE:
            raise ValueError(f"vocab_size must equal {VOCAB_SIZE}")
        if self.warmup_steps >= self.steps:
            raise ValueError("warmup_steps must be smaller than steps")
        for name in (
            "learning_rate",
            "min_learning_rate",
            "max_grad_norm",
            "temperature",
        ):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or value <= 0.0
            ):
                raise ValueError(f"{name} must be finite and positive")
        if self.min_learning_rate > self.learning_rate:
            raise ValueError(
                "min_learning_rate must not exceed learning_rate"
            )
        if (
            not math.isfinite(self.dropout)
            or not 0.0 <= self.dropout < 1.0
        ):
            raise ValueError("dropout must be finite and in [0, 1)")
        if not math.isfinite(self.weight_decay) or self.weight_decay < 0.0:
            raise ValueError("weight_decay must be finite and non-negative")
        for name in ("beta1", "beta2"):
            value = getattr(self, name)
            if not math.isfinite(value) or not 0.0 < value < 1.0:
                raise ValueError(f"{name} must be finite and in (0, 1)")
        if not isinstance(self.use_bfloat16, bool):
            raise ValueError("use_bfloat16 must be boolean")
        if self.top_k > self.vocab_size:
            raise ValueError("top_k must not exceed vocab_size")
        if (
            not isinstance(self.sample_prompts, tuple)
            or not self.sample_prompts
            or any(
                not isinstance(prompt, str) or not prompt
                for prompt in self.sample_prompts
            )
        ):
            raise ValueError(
                "sample_prompts must be a non-empty tuple of non-empty strings"
            )
        if not isinstance(self.source_commit, str) or not self.source_commit:
            raise ValueError("source_commit must not be empty")
        self.model_config().validate()


def frozen_student_base_config(
    *,
    data_dir: str | Path,
    output_dir: str | Path,
    source_commit: str,
    **overrides: object,
) -> BaseTrainingConfig:
    """Create the selected 6x384/context-256 student training run."""

    return BaseTrainingConfig(
        data_dir=Path(data_dir),
        output_dir=Path(output_dir),
        source_commit=source_commit,
        **overrides,
    )


def load_packed_split(
    token_path: str | Path,
    mask_path: str | Path,
    *,
    vocab_size: int,
) -> PackedTokenSplit:
    """Open explicit little-endian uint16 tokens without copying the shard."""

    _positive_integer(vocab_size, "vocab_size")
    token_path = Path(token_path)
    mask_path = Path(mask_path)
    if not token_path.is_file():
        raise FileNotFoundError(token_path)
    if not mask_path.is_file():
        raise FileNotFoundError(mask_path)
    token_bytes = token_path.stat().st_size
    if token_bytes == 0 or token_bytes % 2:
        raise ValueError(
            "token file must contain a non-empty even number of bytes"
        )
    token_count = token_bytes // 2
    if mask_path.stat().st_size != token_count:
        raise ValueError("loss mask must have the same token count")
    tokens = np.memmap(token_path, mode="r", dtype="<u2")
    loss_mask = np.memmap(mask_path, mode="r", dtype="u1")
    if int(tokens.max()) >= vocab_size:
        raise ValueError("token file contains an ID outside the vocabulary")
    if np.any((loss_mask != 0) & (loss_mask != 1)):
        raise ValueError("loss mask bytes must be zero or one")
    return PackedTokenSplit(
        token_path=token_path,
        mask_path=mask_path,
        tokens=tokens,
        loss_mask=loss_mask,
        vocab_size=vocab_size,
    )


def _read_manifest(path: Path) -> dict[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid corpus manifest: {path}") from error
    if not isinstance(value, dict):
        raise ValueError("corpus manifest must be a JSON object")
    return value


def load_packed_dataset(data_dir: str | Path) -> PackedDataset:
    """Validate the Lesson 10/11 shard manifest and memory-map all splits."""

    root = Path(data_dir)
    shard_root = root / "shards" if (root / "shards").is_dir() else root
    manifest_path = shard_root / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    manifest = _read_manifest(manifest_path)
    tokenizer = manifest.get("tokenizer")
    if (
        not isinstance(tokenizer, dict)
        or tokenizer.get("vocab_size") != VOCAB_SIZE
    ):
        raise ValueError(
            f"corpus tokenizer must declare vocab_size {VOCAB_SIZE}"
        )
    files = manifest.get("files")
    tokens_summary = manifest.get("tokens")
    if not isinstance(files, dict) or not isinstance(tokens_summary, dict):
        raise ValueError("corpus manifest is missing files or tokens")

    loaded: dict[str, PackedTokenSplit] = {}
    for split in ("train", "validation", "test"):
        token_name = f"{split}.tokens.bin"
        mask_name = f"{split}.loss.bin"
        for filename in (token_name, mask_name):
            metadata = files.get(filename)
            if not isinstance(metadata, dict):
                raise ValueError(
                    f"corpus manifest is missing {filename}"
                )
            path = shard_root / filename
            if path.stat().st_size != metadata.get("bytes"):
                raise ValueError(f"{filename} byte count disagrees with manifest")
            if sha256_file(path) != metadata.get("sha256"):
                raise ValueError(f"{filename} SHA-256 disagrees with manifest")
        loaded[split] = load_packed_split(
            shard_root / token_name,
            shard_root / mask_name,
            vocab_size=VOCAB_SIZE,
        )
        if loaded[split].token_count != tokens_summary.get(split):
            raise ValueError(
                f"{split} token count disagrees with manifest"
            )
    return PackedDataset(
        root=shard_root,
        manifest_path=manifest_path,
        manifest=manifest,
        train=loaded["train"],
        validation=loaded["validation"],
        test=loaded["test"],
    )


def make_packed_batch(
    split: PackedTokenSplit,
    *,
    batch_size: int,
    block_size: int,
    generator: torch.Generator,
    device: torch.device,
    starts: torch.Tensor | None = None,
) -> PackedBatch:
    """Build shifted windows; BOS targets remain excluded by the stored mask."""

    if not isinstance(split, PackedTokenSplit):
        raise ValueError("split must be a PackedTokenSplit")
    _positive_integer(batch_size, "batch_size")
    _positive_integer(block_size, "block_size")
    if not isinstance(generator, torch.Generator):
        raise ValueError("generator must be a torch.Generator")
    if not isinstance(device, torch.device):
        raise ValueError("device must be a torch.device")
    maximum_start = split.token_count - block_size - 1
    if maximum_start < 0:
        raise ValueError("split is too short for block_size")
    starts_were_provided = starts is not None
    if starts is None:
        if not bool(np.asarray(split.loss_mask)[1:].any()):
            raise ValueError("split contains no eligible prediction target")
        sampled_starts: list[int] = []
        for _ in range(batch_size):
            for _attempt in range(10_000):
                candidate = int(
                    torch.randint(
                        0,
                        maximum_start + 1,
                        (1,),
                        generator=generator,
                        device="cpu",
                    ).item()
                )
                if bool(
                    np.asarray(
                        split.loss_mask[
                            candidate + 1 : candidate + block_size + 1
                        ]
                    ).any()
                ):
                    sampled_starts.append(candidate)
                    break
            else:
                raise ValueError(
                    "could not sample a window with an eligible target"
                )
        starts = torch.tensor(sampled_starts, dtype=torch.long)
    else:
        if (
            not isinstance(starts, torch.Tensor)
            or starts.dtype != torch.long
            or starts.device.type != "cpu"
            or starts.shape != (batch_size,)
        ):
            raise ValueError(
                "starts must be a one-dimensional CPU long tensor"
            )
        if int(starts.min()) < 0 or int(starts.max()) > maximum_start:
            raise ValueError("starts contain an out-of-range offset")
        starts = starts.clone()

    input_array = np.stack(
        [
            np.asarray(
                split.tokens[start : start + block_size],
                dtype=np.int64,
            )
            for start in starts.tolist()
        ]
    )
    target_array = np.stack(
        [
            np.asarray(
                split.tokens[start + 1 : start + block_size + 1],
                dtype=np.int64,
            )
            for start in starts.tolist()
        ]
    )
    mask_array = np.stack(
        [
            np.asarray(
                split.loss_mask[start + 1 : start + block_size + 1],
                dtype=np.bool_,
            )
            for start in starts.tolist()
        ]
    )
    if not starts_were_provided and not np.all(mask_array.any(axis=1)):
        raise ValueError(
            "sampled window contains no eligible prediction target"
        )
    return PackedBatch(
        inputs=torch.from_numpy(input_array).to(device),
        targets=torch.from_numpy(target_array).to(device),
        target_mask=torch.from_numpy(mask_array).to(device),
        starts=starts,
    )


def masked_cross_entropy(
    logits: torch.Tensor,
    targets: torch.Tensor,
    target_mask: torch.Tensor,
) -> torch.Tensor:
    """Average next-token cross entropy over target-eligible positions only."""

    if logits.ndim != 3:
        raise ValueError("logits must have shape (B, T, V)")
    if targets.shape != logits.shape[:2]:
        raise ValueError("targets must match the first two logits dimensions")
    if target_mask.shape != targets.shape or target_mask.dtype != torch.bool:
        raise ValueError("target_mask must be boolean and match targets")
    eligible = int(target_mask.sum().item())
    if eligible == 0:
        raise ValueError("target_mask must include at least one target")
    losses = F.cross_entropy(
        logits.reshape(-1, logits.shape[-1]),
        targets.reshape(-1),
        reduction="none",
    ).reshape_as(targets)
    return losses.masked_select(target_mask).mean()


def _autocast_context(
    device: torch.device,
    *,
    enabled: bool,
) -> Iterator[None]:
    if device.type == "cuda" and enabled:
        return torch.autocast(
            device_type="cuda",
            dtype=torch.bfloat16,
        )
    return nullcontext()


def _evaluate_model(
    model: DirectSmallGPT,
    split: PackedTokenSplit,
    *,
    config: BaseTrainingConfig,
    device: torch.device,
    seed: int,
) -> float:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    was_training = model.training
    model.eval()
    weighted_loss = 0.0
    targets = 0
    with torch.inference_mode():
        for _ in range(config.eval_batches):
            batch = make_packed_batch(
                split,
                batch_size=config.micro_batch_size,
                block_size=config.block_size,
                generator=generator,
                device=device,
            )
            with _autocast_context(
                device,
                enabled=config.use_bfloat16,
            ):
                logits, _ = model(batch.inputs)
                loss = masked_cross_entropy(
                    logits,
                    batch.targets,
                    batch.target_mask,
                )
            eligible = int(batch.target_mask.sum().item())
            weighted_loss += float(loss.item()) * eligible
            targets += eligible
    model.train(was_training)
    return weighted_loss / targets


def evaluate_sequential_split(
    model: DirectSmallGPT,
    split: PackedTokenSplit,
    *,
    block_size: int,
    batch_size: int,
    device: torch.device,
    use_bfloat16: bool,
) -> dict[str, object]:
    """Evaluate every target position once in deterministic packed windows."""

    if not isinstance(model, DirectSmallGPT):
        raise ValueError("model must be a DirectSmallGPT")
    if not isinstance(split, PackedTokenSplit):
        raise ValueError("split must be a PackedTokenSplit")
    _positive_integer(block_size, "block_size")
    _positive_integer(batch_size, "batch_size")
    if not isinstance(device, torch.device):
        raise ValueError("device must be a torch.device")
    if not isinstance(use_bfloat16, bool):
        raise ValueError("use_bfloat16 must be boolean")
    prediction_positions = split.token_count - 1
    if prediction_positions <= 0:
        raise ValueError("split is too short to evaluate")

    full_window_count = prediction_positions // block_size
    full_starts = [
        index * block_size
        for index in range(full_window_count)
    ]
    weighted_loss = 0.0
    eligible_targets = 0
    evaluated_positions = 0
    generator = torch.Generator(device="cpu").manual_seed(0)
    was_training = model.training
    model.eval()

    def accumulate(batch: PackedBatch) -> None:
        nonlocal weighted_loss, eligible_targets, evaluated_positions
        eligible = int(batch.target_mask.sum().item())
        evaluated_positions += batch.targets.numel()
        if eligible == 0:
            return
        with _autocast_context(
            device,
            enabled=use_bfloat16,
        ):
            logits, _ = model(batch.inputs)
            loss = masked_cross_entropy(
                logits,
                batch.targets,
                batch.target_mask,
            )
        weighted_loss += float(loss.item()) * eligible
        eligible_targets += eligible

    with torch.inference_mode():
        for offset in range(0, len(full_starts), batch_size):
            starts = torch.tensor(
                full_starts[offset : offset + batch_size],
                dtype=torch.long,
            )
            batch = make_packed_batch(
                split,
                batch_size=len(starts),
                block_size=block_size,
                generator=generator,
                device=device,
                starts=starts,
            )
            accumulate(batch)
        tail_start = full_window_count * block_size
        tail_length = prediction_positions - tail_start
        if tail_length:
            batch = make_packed_batch(
                split,
                batch_size=1,
                block_size=tail_length,
                generator=generator,
                device=device,
                starts=torch.tensor([tail_start], dtype=torch.long),
            )
            accumulate(batch)
    model.train(was_training)
    expected_eligible = int(np.asarray(split.loss_mask)[1:].sum())
    if (
        evaluated_positions != prediction_positions
        or eligible_targets != expected_eligible
    ):
        raise RuntimeError(
            "sequential evaluation did not cover the expected targets"
        )
    return {
        "eligible_targets": eligible_targets,
        "evaluated_prediction_positions": evaluated_positions,
        **_loss_metrics(weighted_loss / eligible_targets),
    }


def evaluate_frequency_baseline(
    train: PackedTokenSplit,
    validation: PackedTokenSplit,
    *,
    vocab_size: int,
) -> float:
    """Evaluate add-one-smoothed unigram frequencies on eligible targets."""

    _positive_integer(vocab_size, "vocab_size")
    train_tokens = np.asarray(train.tokens)
    train_mask = np.asarray(train.loss_mask, dtype=np.bool_)
    counts = np.bincount(
        train_tokens[train_mask],
        minlength=vocab_size,
    ).astype(np.float64)
    probabilities = (counts + 1.0) / (counts.sum() + vocab_size)
    validation_tokens = np.asarray(validation.tokens)
    validation_mask = np.asarray(
        validation.loss_mask,
        dtype=np.bool_,
    )
    eligible = validation_tokens[validation_mask]
    if eligible.size == 0:
        raise ValueError("validation split has no eligible targets")
    return float(-np.log(probabilities[eligible]).mean())


def _cpu_state_dict(
    model: DirectSmallGPT,
) -> dict[str, torch.Tensor]:
    return {
        name: tensor.detach().cpu().clone()
        for name, tensor in model.state_dict().items()
    }


def _run_overfit_gate(
    dataset: PackedDataset,
    config: BaseTrainingConfig,
    *,
    device: torch.device,
) -> dict[str, object]:
    torch.manual_seed(config.seed + 100)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(config.seed + 100)
    gate_config = config.model_config()
    gate_model = DirectSmallGPT(gate_config).to(device)
    optimizer = configure_adamw(
        gate_model,
        learning_rate=max(config.learning_rate, 0.001),
        weight_decay=0.0,
        beta1=config.beta1,
        beta2=config.beta2,
    )
    generator = torch.Generator(device="cpu").manual_seed(config.seed + 101)
    batch = make_packed_batch(
        dataset.train,
        batch_size=config.micro_batch_size,
        block_size=config.block_size,
        generator=generator,
        device=device,
    )

    gate_model.eval()
    with torch.inference_mode(), _autocast_context(
        device,
        enabled=config.use_bfloat16,
    ):
        initial_logits, _ = gate_model(batch.inputs)
        initial_loss = float(
            masked_cross_entropy(
                initial_logits,
                batch.targets,
                batch.target_mask,
            ).item()
        )
    gate_model.train()
    for _ in range(config.overfit_gate_steps):
        optimizer.zero_grad(set_to_none=True)
        with _autocast_context(
            device,
            enabled=config.use_bfloat16,
        ):
            logits, _ = gate_model(batch.inputs)
            loss = masked_cross_entropy(
                logits,
                batch.targets,
                batch.target_mask,
            )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            gate_model.parameters(),
            config.max_grad_norm,
        )
        optimizer.step()
    gate_model.eval()
    with torch.inference_mode(), _autocast_context(
        device,
        enabled=config.use_bfloat16,
    ):
        final_logits, _ = gate_model(batch.inputs)
        final_loss = float(
            masked_cross_entropy(
                final_logits,
                batch.targets,
                batch.target_mask,
            ).item()
        )
    result = {
        "final_loss": final_loss,
        "initial_loss": initial_loss,
        "passed": final_loss < initial_loss,
        "steps": config.overfit_gate_steps,
    }
    del gate_model, optimizer, batch
    if device.type == "cuda":
        torch.cuda.empty_cache()
    if not result["passed"]:
        raise RuntimeError(
            "overfit gate failed: repeated-batch loss did not decrease"
        )
    return result


def _loss_metrics(loss: float) -> dict[str, float]:
    return {
        "bits_per_byte": loss / math.log(2.0),
        "byte_perplexity": math.exp(loss),
        "loss": loss,
    }


def _generate_continuation(
    model: DirectSmallGPT,
    prompt: str,
    *,
    config: BaseTrainingConfig,
    device: torch.device,
    seed: int,
) -> dict[str, object]:
    tokenizer = ByteTokenizer()
    tokens = [BOS_ID, *tokenizer.encode_text(prompt)]
    generated: list[int] = []
    generator = torch.Generator(device=device).manual_seed(seed)
    was_training = model.training
    model.eval()
    with torch.inference_mode():
        for _ in range(config.sample_tokens):
            window = tokens[-config.block_size :]
            inputs = torch.tensor(
                [window],
                dtype=torch.long,
                device=device,
            )
            with _autocast_context(
                device,
                enabled=config.use_bfloat16,
            ):
                logits, _ = model(inputs)
            next_logits = logits[0, -1].float() / config.temperature
            next_logits[BOS_ID:] = float("-inf")
            next_logits[EOS_ID] = logits[0, -1, EOS_ID].float()
            if config.top_k < config.vocab_size:
                threshold = torch.topk(
                    next_logits,
                    config.top_k,
                ).values[-1]
                next_logits[next_logits < threshold] = float("-inf")
            probabilities = torch.softmax(next_logits, dim=-1)
            next_token = int(
                torch.multinomial(
                    probabilities,
                    num_samples=1,
                    generator=generator,
                ).item()
            )
            if next_token == EOS_ID:
                break
            tokens.append(next_token)
            generated.append(next_token)
    model.train(was_training)
    return {
        "continuation": tokenizer.render_tokens(generated),
        "generated_tokens": len(generated),
        "prompt": prompt,
        "seed": seed,
    }


def _atomic_torch_save(value: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        torch.save(value, temporary)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _dataset_summary(dataset: PackedDataset) -> dict[str, object]:
    files = dataset.manifest["files"]
    assert isinstance(files, dict)
    return {
        "manifest_sha256": sha256_file(dataset.manifest_path),
        "test_tokens": dataset.test.token_count,
        "train_tokens": dataset.train.token_count,
        "validation_tokens": dataset.validation.token_count,
        "vocab_size": VOCAB_SIZE,
    }


def run_base_training(
    config: BaseTrainingConfig,
) -> dict[str, object]:
    """Run the overfit gate, base pilot, best selection, and fixed samples."""

    if not isinstance(config, BaseTrainingConfig):
        raise ValueError("config must be a BaseTrainingConfig")
    config.validate()
    if config.output_dir.exists():
        raise ValueError(
            f"output directory already exists: {config.output_dir}"
        )
    device = resolve_device(config.device)
    dataset = load_packed_dataset(config.data_dir)
    for name, split in (
        ("train", dataset.train),
        ("validation", dataset.validation),
    ):
        if split.token_count < config.block_size + 1:
            raise ValueError(f"{name} split is too short for block_size")

    overfit_gate = _run_overfit_gate(
        dataset,
        config,
        device=device,
    )
    torch.manual_seed(config.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(config.seed)
        torch.cuda.reset_peak_memory_stats(device)
    model_config = config.model_config()
    model = DirectSmallGPT(model_config).to(device)
    optimizer = configure_adamw(
        model,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        beta1=config.beta1,
        beta2=config.beta2,
    )
    frequency_loss = evaluate_frequency_baseline(
        dataset.train,
        dataset.validation,
        vocab_size=config.vocab_size,
    )
    validation_seed = config.seed + 1
    synchronize(device)
    evaluation_started = time.perf_counter()
    initial_validation_loss = _evaluate_model(
        model,
        dataset.validation,
        config=config,
        device=device,
        seed=validation_seed,
    )
    initial_full_validation = evaluate_sequential_split(
        model,
        dataset.validation,
        block_size=config.block_size,
        batch_size=config.micro_batch_size,
        device=device,
        use_bfloat16=config.use_bfloat16,
    )
    synchronize(device)
    evaluation_seconds = time.perf_counter() - evaluation_started
    best_validation_loss = initial_validation_loss
    best_step = 0
    best_state = _cpu_state_dict(model)
    evaluation_history: list[dict[str, object]] = [
        {
            **_loss_metrics(initial_validation_loss),
            "is_new_best": True,
            "step": 0,
        }
    ]

    generator = torch.Generator(device="cpu").manual_seed(config.seed + 2)
    training_history: list[dict[str, object]] = []
    optimizer_update_seconds = 0.0
    synchronize(device)
    segment_started = time.perf_counter()
    wall_started = segment_started
    final_step_validation_loss = initial_validation_loss
    model.train()
    for step in range(1, config.steps + 1):
        learning_rate = learning_rate_at_step(
            step,
            max_learning_rate=config.learning_rate,
            min_learning_rate=config.min_learning_rate,
            warmup_steps=config.warmup_steps,
            max_steps=config.steps,
        )
        for group in optimizer.param_groups:
            group["lr"] = learning_rate
        optimizer.zero_grad(set_to_none=True)
        micro_losses: list[float] = []
        for _ in range(config.gradient_accumulation_steps):
            batch = make_packed_batch(
                dataset.train,
                batch_size=config.micro_batch_size,
                block_size=config.block_size,
                generator=generator,
                device=device,
            )
            with _autocast_context(
                device,
                enabled=config.use_bfloat16,
            ):
                logits, _ = model(batch.inputs)
                loss = masked_cross_entropy(
                    logits,
                    batch.targets,
                    batch.target_mask,
                )
            if not bool(torch.isfinite(loss).item()):
                raise FloatingPointError(
                    f"training loss became non-finite at step {step}"
                )
            (loss / config.gradient_accumulation_steps).backward()
            micro_losses.append(float(loss.detach().item()))
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            config.max_grad_norm,
        )
        if not bool(torch.isfinite(gradient_norm).item()):
            raise FloatingPointError(
                f"gradient norm became non-finite at step {step}"
            )
        optimizer.step()

        if (
            step == 1
            or step % config.log_interval == 0
            or step == config.steps
        ):
            training_history.append(
                {
                    "gradient_l2_norm_before_clip": float(
                        gradient_norm.item()
                    ),
                    "learning_rate": learning_rate,
                    "step": step,
                    "training_loss": sum(micro_losses) / len(micro_losses),
                }
            )

        if step % config.eval_interval == 0 or step == config.steps:
            synchronize(device)
            optimizer_update_seconds += (
                time.perf_counter() - segment_started
            )
            evaluation_started = time.perf_counter()
            validation_loss = _evaluate_model(
                model,
                dataset.validation,
                config=config,
                device=device,
                seed=validation_seed,
            )
            synchronize(device)
            evaluation_seconds += (
                time.perf_counter() - evaluation_started
            )
            final_step_validation_loss = validation_loss
            is_new_best = validation_loss < best_validation_loss
            if is_new_best:
                best_validation_loss = validation_loss
                best_step = step
                best_state = _cpu_state_dict(model)
            evaluation_history.append(
                {
                    **_loss_metrics(validation_loss),
                    "is_new_best": is_new_best,
                    "step": step,
                }
            )
            model.train()
            synchronize(device)
            segment_started = time.perf_counter()
    synchronize(device)
    wall_seconds = time.perf_counter() - wall_started

    model.load_state_dict(best_state, strict=True)
    selected_validation_loss = _evaluate_model(
        model,
        dataset.validation,
        config=config,
        device=device,
        seed=validation_seed,
    )
    selected_full_validation = evaluate_sequential_split(
        model,
        dataset.validation,
        block_size=config.block_size,
        batch_size=config.micro_batch_size,
        device=device,
        use_bfloat16=config.use_bfloat16,
    )
    selected_full_test = evaluate_sequential_split(
        model,
        dataset.test,
        block_size=config.block_size,
        batch_size=config.micro_batch_size,
        device=device,
        use_bfloat16=config.use_bfloat16,
    )
    samples = [
        _generate_continuation(
            model,
            prompt,
            config=config,
            device=device,
            seed=config.seed + 10 + index,
        )
        for index, prompt in enumerate(config.sample_prompts)
    ]
    peak_cuda_bytes = (
        int(torch.cuda.max_memory_allocated(device))
        if device.type == "cuda"
        else None
    )

    checkpoint = {
        "best_step": best_step,
        "dataset_manifest_sha256": sha256_file(dataset.manifest_path),
        "model_config": asdict(model_config),
        "model_state_dict": best_state,
        "route": ROUTE,
        "schema_version": 1,
        "selected_validation_loss": selected_validation_loss,
        "source_commit": config.source_commit,
        "tokenizer": {
            "kind": "byte_plus_fixed_special_tokens",
            "vocab_size": VOCAB_SIZE,
        },
        "training_seed": config.seed,
    }
    checkpoint_path = config.output_dir / CHECKPOINT_FILENAME
    _atomic_torch_save(checkpoint, checkpoint_path)
    checkpoint_metadata = {
        "bytes": checkpoint_path.stat().st_size,
        "path": str(checkpoint_path),
        "sha256": sha256_file(checkpoint_path),
    }
    configuration = asdict(config)
    configuration["data_dir"] = str(config.data_dir)
    configuration["output_dir"] = str(config.output_dir)
    configuration["sample_prompts"] = list(config.sample_prompts)
    training_tokens = config.steps * config.effective_batch_tokens
    eligible_train_targets = int(
        np.asarray(dataset.train.loss_mask).sum()
    )
    summary: dict[str, object] = {
        "artifacts": {
            "checkpoint": checkpoint_metadata,
        },
        "baselines": {
            "frequency": _loss_metrics(frequency_loss),
            "uniform": _loss_metrics(UNIFORM_LOSS),
            "uniform_loss": UNIFORM_LOSS,
        },
        "configuration": configuration,
        "dataset": _dataset_summary(dataset),
        "environment": {
            "bfloat16_autocast": (
                config.use_bfloat16 and device.type == "cuda"
            ),
            "cuda_runtime": torch.version.cuda,
            "device": str(device),
            "gpu_name": (
                torch.cuda.get_device_name(device)
                if device.type == "cuda"
                else None
            ),
            "peak_cuda_memory_allocated_bytes": peak_cuda_bytes,
            "python": platform.python_version(),
            "torch": str(torch.__version__),
        },
        "evaluation_history": evaluation_history,
        "metrics": {
            "approximate_train_epochs": (
                training_tokens / eligible_train_targets
            ),
            "best_step": best_step,
            "evaluation_seconds": evaluation_seconds,
            "final_step_validation": _loss_metrics(
                final_step_validation_loss
            ),
            "initial_validation": _loss_metrics(
                initial_validation_loss
            ),
            "initial_full_validation": initial_full_validation,
            "optimizer_update_seconds": optimizer_update_seconds,
            "selected_validation": _loss_metrics(
                selected_validation_loss
            ),
            "selected_full_test": selected_full_test,
            "selected_full_validation": selected_full_validation,
            "selected_validation_loss": selected_validation_loss,
            "training_tokens": training_tokens,
            "update_tokens_per_second": (
                training_tokens / optimizer_update_seconds
            ),
            "wall_seconds": wall_seconds,
        },
        "model": {
            **asdict(model_config),
            "head_dim": model_config.n_embd // model_config.n_head,
            "parameters": model.parameter_count,
            "raw_fp32_parameter_bytes": model.raw_fp32_parameter_bytes,
        },
        "overfit_gate": overfit_gate,
        "route": ROUTE,
        "samples": samples,
        "schema_version": 1,
        "source_commit": config.source_commit,
        "training_history": training_history,
    }
    write_json_atomic(config.output_dir / "run.json", summary)
    return summary


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=20260728)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--micro-batch-size", type=int, default=4)
    parser.add_argument(
        "--gradient-accumulation-steps",
        type=int,
        default=4,
    )
    parser.add_argument("--learning-rate", type=float, default=0.0006)
    parser.add_argument(
        "--min-learning-rate",
        type=float,
        default=0.00006,
    )
    parser.add_argument("--warmup-steps", type=int, default=50)
    parser.add_argument("--eval-interval", type=int, default=100)
    parser.add_argument("--eval-batches", type=int, default=20)
    parser.add_argument("--log-interval", type=int, default=25)
    parser.add_argument("--overfit-gate-steps", type=int, default=20)
    parser.add_argument("--sample-tokens", type=int, default=96)
    parser.add_argument(
        "--no-bfloat16",
        action="store_true",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = _build_parser().parse_args(argv)
    config = frozen_student_base_config(
        data_dir=arguments.data_dir,
        output_dir=arguments.output_dir,
        source_commit=arguments.source_commit,
        device=arguments.device,
        seed=arguments.seed,
        steps=arguments.steps,
        micro_batch_size=arguments.micro_batch_size,
        gradient_accumulation_steps=(
            arguments.gradient_accumulation_steps
        ),
        learning_rate=arguments.learning_rate,
        min_learning_rate=arguments.min_learning_rate,
        warmup_steps=arguments.warmup_steps,
        eval_interval=arguments.eval_interval,
        eval_batches=arguments.eval_batches,
        log_interval=arguments.log_interval,
        overfit_gate_steps=arguments.overfit_gate_steps,
        sample_tokens=arguments.sample_tokens,
        use_bfloat16=not arguments.no_bfloat16,
    )
    summary = run_base_training(config)
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
