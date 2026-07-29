"""Frozen CPT and Hybrid-SFT training for Lesson 15 efficient models."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
import math
import pathlib
import platform
import re
import time

import numpy as np
import torch

from nanogpt_nspire.base_train import (
    _atomic_torch_save,
    _autocast_context,
    _cpu_state_dict,
    _dataset_summary,
    _evaluate_model,
    _loss_metrics,
    evaluate_sequential_split,
    load_packed_dataset,
    make_packed_batch,
    masked_cross_entropy,
)
from nanogpt_nspire.byte_tokenizer import VOCAB_SIZE
from nanogpt_nspire.direct_small_train import (
    configure_adamw,
    learning_rate_at_step,
)
from nanogpt_nspire.efficient_context import (
    ARCHITECTURE_NAME,
    CPT_ROUTES,
    INIT_ROUTES,
    SFT_ROUTES,
    lesson15_efficient_config,
    load_efficient_checkpoint,
)
from nanogpt_nspire.models.efficient_long_context_gpt import (
    POSITION_MODES,
    EfficientLongContextConfig,
    EfficientLongContextGPT,
)
from nanogpt_nspire.training_support import (
    resolve_device,
    sha256_file,
    synchronize,
    write_json_atomic,
)


_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")


@dataclass(frozen=True)
class EfficientTrainingConfig:
    stage: str
    position_mode: str
    data_dir: pathlib.Path
    output_dir: pathlib.Path
    parent_checkpoint: pathlib.Path
    parent_checkpoint_sha256: str
    source_commit: str
    device: str = "auto"
    seed: int = 20260728
    steps: int = 250
    micro_batch_size: int = 2
    gradient_accumulation_steps: int = 4
    learning_rate: float = 0.0001
    min_learning_rate: float = 0.00001
    warmup_steps: int = 25
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    max_grad_norm: float = 1.0
    eval_interval: int = 50
    eval_batches: int = 20
    log_interval: int = 25
    overfit_gate_steps: int = 20
    use_bfloat16: bool = True

    @property
    def block_size(self) -> int:
        return 512

    @property
    def effective_batch_tokens(self) -> int:
        return (
            self.micro_batch_size
            * self.gradient_accumulation_steps
            * self.block_size
        )

    @property
    def expected_parent_route(self) -> str:
        return (
            INIT_ROUTES[self.position_mode]
            if self.stage == "cpt"
            else CPT_ROUTES[self.position_mode]
        )

    @property
    def route(self) -> str:
        return (
            CPT_ROUTES[self.position_mode]
            if self.stage == "cpt"
            else SFT_ROUTES[self.position_mode]
        )

    @property
    def checkpoint_filename(self) -> str:
        return (
            f"gqa_{self.position_mode}_context512_cpt.pt"
            if self.stage == "cpt"
            else f"gqa_{self.position_mode}_hybrid_sft_context512.pt"
        )

    def model_config(self) -> EfficientLongContextConfig:
        return lesson15_efficient_config(self.position_mode)

    def validate(self) -> None:
        if self.stage not in {"cpt", "sft"}:
            raise ValueError("stage must be 'cpt' or 'sft'")
        if self.position_mode not in POSITION_MODES:
            raise ValueError("position_mode must be 'learned' or 'alibi'")
        if (
            not isinstance(self.parent_checkpoint_sha256, str)
            or _SHA256_PATTERN.fullmatch(
                self.parent_checkpoint_sha256
            )
            is None
        ):
            raise ValueError(
                "parent_checkpoint_sha256 must be lowercase SHA-256"
            )
        if not isinstance(self.source_commit, str) or not self.source_commit:
            raise ValueError("source_commit must be non-empty")
        for name in (
            "seed",
            "steps",
            "micro_batch_size",
            "gradient_accumulation_steps",
            "warmup_steps",
            "eval_interval",
            "eval_batches",
            "log_interval",
            "overfit_gate_steps",
        ):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, int)
                or value < (0 if name == "seed" else 1)
            ):
                raise ValueError(f"{name} is invalid")
        if self.warmup_steps >= self.steps:
            raise ValueError("warmup_steps must be smaller than steps")
        for name in (
            "learning_rate",
            "min_learning_rate",
            "max_grad_norm",
        ):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                or float(value) <= 0
            ):
                raise ValueError(f"{name} must be finite and positive")
        if self.min_learning_rate > self.learning_rate:
            raise ValueError(
                "min_learning_rate must not exceed learning_rate"
            )
        if not math.isfinite(self.weight_decay) or self.weight_decay < 0:
            raise ValueError("weight_decay is invalid")
        for name in ("beta1", "beta2"):
            value = getattr(self, name)
            if not math.isfinite(value) or not 0 < value < 1:
                raise ValueError(f"{name} is invalid")
        if not isinstance(self.use_bfloat16, bool):
            raise ValueError("use_bfloat16 must be boolean")
        self.model_config().validate()


def frozen_efficient_training_config(
    *,
    stage: str,
    position_mode: str,
    data_dir: str | pathlib.Path,
    output_dir: str | pathlib.Path,
    parent_checkpoint: str | pathlib.Path,
    parent_checkpoint_sha256: str,
    source_commit: str,
    **overrides: object,
) -> EfficientTrainingConfig:
    """Freeze compute and architecture for the two position variants."""

    if stage not in {"cpt", "sft"}:
        raise ValueError("stage must be 'cpt' or 'sft'")
    if position_mode not in POSITION_MODES:
        raise ValueError("position_mode must be 'learned' or 'alibi'")
    forbidden = {
        "block_size",
        "n_layer",
        "n_head",
        "n_kv_head",
        "n_embd",
        "mlp_ratio",
        "vocab_size",
    } & set(overrides)
    if forbidden:
        raise ValueError(
            "Lesson 15 architecture is frozen; remove overrides: "
            + ", ".join(sorted(forbidden))
        )
    defaults: dict[str, object] = {
        "device": "auto",
        "steps": 250 if stage == "cpt" else 1000,
        "micro_batch_size": 2,
        "gradient_accumulation_steps": 4,
        "learning_rate": 0.0001,
        "min_learning_rate": 0.00001,
        "warmup_steps": 25 if stage == "cpt" else 50,
        "eval_interval": 50 if stage == "cpt" else 100,
        "eval_batches": 20,
        "log_interval": 25,
        "overfit_gate_steps": 20,
    }
    defaults.update(overrides)
    config = EfficientTrainingConfig(
        stage=stage,
        position_mode=position_mode,
        data_dir=pathlib.Path(data_dir),
        output_dir=pathlib.Path(output_dir),
        parent_checkpoint=pathlib.Path(parent_checkpoint),
        parent_checkpoint_sha256=parent_checkpoint_sha256,
        source_commit=source_commit,
        **defaults,
    )
    config.validate()
    return config


def _run_efficient_overfit_gate(
    dataset: object,
    config: EfficientTrainingConfig,
    *,
    device: torch.device,
) -> dict[str, object]:
    torch.manual_seed(config.seed + 100)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(config.seed + 100)
    model = EfficientLongContextGPT(config.model_config()).to(device)
    optimizer = configure_adamw(
        model,
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

    def loss_value() -> torch.Tensor:
        with _autocast_context(
            device,
            enabled=config.use_bfloat16,
        ):
            logits, _ = model(batch.inputs)
            return masked_cross_entropy(
                logits,
                batch.targets,
                batch.target_mask,
            )

    model.eval()
    with torch.inference_mode():
        initial_loss = float(loss_value().item())
    model.train()
    for _ in range(config.overfit_gate_steps):
        optimizer.zero_grad(set_to_none=True)
        loss = loss_value()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            config.max_grad_norm,
        )
        optimizer.step()
    model.eval()
    with torch.inference_mode():
        final_loss = float(loss_value().item())
    result = {
        "final_loss": final_loss,
        "initial_loss": initial_loss,
        "passed": final_loss < initial_loss,
        "steps": config.overfit_gate_steps,
    }
    del model, optimizer, batch
    if device.type == "cuda":
        torch.cuda.empty_cache()
    if not result["passed"]:
        raise RuntimeError("efficient overfit gate failed")
    return result


def run_efficient_training(
    config: EfficientTrainingConfig,
) -> dict[str, object]:
    """Train one strict efficient route and select by validation loss."""

    if not isinstance(config, EfficientTrainingConfig):
        raise ValueError("config must be EfficientTrainingConfig")
    config.validate()
    if config.output_dir.exists():
        raise ValueError(
            f"output directory already exists: {config.output_dir}"
        )
    dataset = load_packed_dataset(config.data_dir)
    for name, split in (
        ("train", dataset.train),
        ("validation", dataset.validation),
        ("test", dataset.test),
    ):
        if split.token_count < config.block_size + 1:
            raise ValueError(f"{name} split is too short")
        if int(np.asarray(split.loss_mask).sum()) == 0:
            raise ValueError(f"{name} split has no eligible targets")
    device = resolve_device(config.device)
    parent_model, parent = load_efficient_checkpoint(
        config.parent_checkpoint,
        expected_sha256=config.parent_checkpoint_sha256,
        expected_route=config.expected_parent_route,
        expected_model_config=config.model_config(),
    )
    overfit_gate = _run_efficient_overfit_gate(
        dataset,
        config,
        device=device,
    )
    torch.manual_seed(config.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(config.seed)
        torch.cuda.reset_peak_memory_stats(device)
    model = EfficientLongContextGPT(config.model_config()).to(device)
    model.load_state_dict(parent_model.state_dict(), strict=True)
    del parent_model
    optimizer = configure_adamw(
        model,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        beta1=config.beta1,
        beta2=config.beta2,
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
    training_history: list[dict[str, object]] = []
    generator = torch.Generator(device="cpu").manual_seed(config.seed + 2)
    optimizer_update_seconds = 0.0
    sampled_eligible_targets = 0
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
            sampled_eligible_targets += int(batch.target_mask.sum().item())
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
    peak_cuda_bytes = (
        int(torch.cuda.max_memory_allocated(device))
        if device.type == "cuda"
        else None
    )
    checkpoint = {
        "architecture": ARCHITECTURE_NAME,
        "best_step": best_step,
        "dataset_manifest_sha256": sha256_file(dataset.manifest_path),
        "model_config": asdict(config.model_config()),
        "model_state_dict": best_state,
        "parent_checkpoint_sha256": parent["sha256"],
        "parent_route": parent["route"],
        "route": config.route,
        "schema_version": 1,
        "selected_validation_loss": selected_validation_loss,
        "source_commit": config.source_commit,
        "tokenizer": {
            "kind": "byte_plus_fixed_special_tokens",
            "vocab_size": VOCAB_SIZE,
        },
        "training_seed": config.seed,
    }
    checkpoint_path = config.output_dir / config.checkpoint_filename
    _atomic_torch_save(checkpoint, checkpoint_path)
    configuration = asdict(config)
    for name in ("data_dir", "output_dir", "parent_checkpoint"):
        configuration[name] = str(configuration[name])
    eligible_training_targets = int(
        np.asarray(dataset.train.loss_mask).sum()
    )
    training_tokens = config.steps * config.effective_batch_tokens
    summary: dict[str, object] = {
        "artifacts": {
            "checkpoint": {
                "bytes": checkpoint_path.stat().st_size,
                "path": str(checkpoint_path),
                "sha256": sha256_file(checkpoint_path),
            },
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
        "lineage": {
            "parent_route": parent["route"],
            "parent_sha256": parent["sha256"],
            "parent_source_commit": parent["source_commit"],
        },
        "metrics": {
            "approximate_eligible_target_epochs": (
                sampled_eligible_targets / eligible_training_targets
            ),
            "best_step": best_step,
            "eligible_sampled_training_targets": sampled_eligible_targets,
            "eligible_training_targets": eligible_training_targets,
            "evaluation_seconds": evaluation_seconds,
            "final_step_validation": _loss_metrics(
                final_step_validation_loss
            ),
            "initial_full_validation": initial_full_validation,
            "initial_validation": _loss_metrics(
                initial_validation_loss
            ),
            "optimizer_update_seconds": optimizer_update_seconds,
            "selected_full_test": selected_full_test,
            "selected_full_validation": selected_full_validation,
            "selected_validation": _loss_metrics(
                selected_validation_loss
            ),
            "training_tokens": training_tokens,
            "update_tokens_per_second": (
                training_tokens / optimizer_update_seconds
            ),
            "wall_seconds": wall_seconds,
        },
        "model": {
            **asdict(config.model_config()),
            "kv_cache_bytes_fp32": model.kv_cache_bytes_fp32,
            "parameters": model.parameter_count,
            "raw_fp32_parameter_bytes": model.raw_fp32_parameter_bytes,
        },
        "overfit_gate": overfit_gate,
        "route": config.route,
        "schema_version": 1,
        "source_commit": config.source_commit,
        "training_history": training_history,
    }
    write_json_atomic(config.output_dir / "run.json", summary)
    return summary


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("cpt", "sft"), required=True)
    parser.add_argument(
        "--position-mode",
        choices=tuple(sorted(POSITION_MODES)),
        required=True,
    )
    parser.add_argument("--data-dir", type=pathlib.Path, required=True)
    parser.add_argument("--output-dir", type=pathlib.Path, required=True)
    parser.add_argument("--parent-checkpoint", type=pathlib.Path, required=True)
    parser.add_argument("--parent-checkpoint-sha256", required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--device", default="cuda")
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = _build_parser().parse_args(argv)
    config = frozen_efficient_training_config(
        stage=arguments.stage,
        position_mode=arguments.position_mode,
        data_dir=arguments.data_dir,
        output_dir=arguments.output_dir,
        parent_checkpoint=arguments.parent_checkpoint,
        parent_checkpoint_sha256=arguments.parent_checkpoint_sha256,
        source_commit=arguments.source_commit,
        device=arguments.device,
    )
    result = run_efficient_training(config)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
