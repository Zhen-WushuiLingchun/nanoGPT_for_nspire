"""Small, observable building blocks for language-model optimization."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable

import torch
from torch import nn


@dataclass(frozen=True)
class BatchMetrics:
    """Loss and next-token accuracy measured without changing parameters."""

    loss: float
    token_accuracy: float


@dataclass(frozen=True)
class TrainStepMetrics:
    """Values observed around one optimizer update."""

    loss: float
    token_accuracy: float
    gradient_l2_norm_before_clip: float
    gradient_l2_norm_after_clip: float
    parameter_update_l2_norm: float


@dataclass(frozen=True)
class OverfitResult:
    """Initial/final metrics and bounded history for one fixed batch."""

    initial: BatchMetrics
    final: BatchMetrics
    history: tuple[dict[str, float | int | None], ...]


def _positive_integer(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _validate_max_grad_norm(max_grad_norm: float | None) -> None:
    if max_grad_norm is None:
        return
    if not math.isfinite(max_grad_norm) or max_grad_norm <= 0.0:
        raise ValueError("max_grad_norm must be finite and positive or None")


def _forward_with_loss(
    model: nn.Module,
    inputs: torch.Tensor,
    targets: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    result = model(inputs, targets)
    if not isinstance(result, tuple) or len(result) != 2:
        raise TypeError("language model forward must return (logits, loss)")
    logits, loss = result
    if not isinstance(logits, torch.Tensor):
        raise TypeError("language model logits must be a tensor")
    if not isinstance(loss, torch.Tensor):
        raise TypeError("language model must return a tensor loss when targets are given")
    if loss.ndim != 0:
        raise ValueError("language model loss must be a scalar tensor")
    if logits.ndim != targets.ndim + 1 or logits.shape[:-1] != targets.shape:
        raise ValueError("logits leading dimensions must match targets")
    return logits, loss


def _token_accuracy(logits: torch.Tensor, targets: torch.Tensor) -> float:
    predicted = torch.argmax(logits.detach(), dim=-1)
    return float((predicted == targets).to(torch.float32).mean().item())


def gradient_l2_norm(parameters: Iterable[nn.Parameter]) -> float:
    """Return the global L2 norm of all currently populated gradients."""

    squared_norms: list[float] = []
    for parameter in parameters:
        gradient = parameter.grad
        if gradient is None:
            continue
        if gradient.is_sparse:
            values = gradient.detach().coalesce().values().to(torch.float64)
        else:
            values = gradient.detach().to(torch.float64)
        squared_norms.append(float(torch.sum(values * values).item()))
    return math.sqrt(math.fsum(squared_norms))


def _parameter_update_l2_norm(
    parameters: list[nn.Parameter],
    before_update: list[torch.Tensor],
) -> float:
    squared_norms = [
        float(
            torch.sum(
                (
                    parameter.detach().to(torch.float64)
                    - previous.to(torch.float64)
                )
                ** 2
            ).item()
        )
        for parameter, previous in zip(parameters, before_update, strict=True)
    ]
    return math.sqrt(math.fsum(squared_norms))


def evaluate_batch(
    model: nn.Module,
    inputs: torch.Tensor,
    targets: torch.Tensor,
) -> BatchMetrics:
    """Evaluate one exact batch and restore the caller's model mode."""

    was_training = model.training
    model.eval()
    try:
        with torch.inference_mode():
            logits, loss = _forward_with_loss(model, inputs, targets)
            loss_value = float(loss.item())
            if not math.isfinite(loss_value):
                raise FloatingPointError("evaluation loss is not finite")
            accuracy = _token_accuracy(logits, targets)
    finally:
        model.train(was_training)
    return BatchMetrics(loss=loss_value, token_accuracy=accuracy)


def train_step(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    *,
    max_grad_norm: float | None = None,
) -> TrainStepMetrics:
    """Run one explicit zero-grad, forward, backward, clip, and update step."""

    _validate_max_grad_norm(max_grad_norm)
    trainable_parameters = [
        parameter for parameter in model.parameters() if parameter.requires_grad
    ]
    if not trainable_parameters:
        raise ValueError("model has no trainable parameters")

    model.train()
    optimizer.zero_grad(set_to_none=True)
    logits, loss = _forward_with_loss(model, inputs, targets)
    loss_value = float(loss.detach().item())
    if not math.isfinite(loss_value):
        raise FloatingPointError("training loss is not finite")
    accuracy = _token_accuracy(logits, targets)

    loss.backward()
    gradient_before_clip = gradient_l2_norm(trainable_parameters)
    if not math.isfinite(gradient_before_clip):
        raise FloatingPointError("gradient norm is not finite before optimizer step")

    if max_grad_norm is not None:
        torch.nn.utils.clip_grad_norm_(
            trainable_parameters,
            max_norm=max_grad_norm,
        )
    gradient_after_clip = gradient_l2_norm(trainable_parameters)
    if not math.isfinite(gradient_after_clip):
        raise FloatingPointError("gradient norm is not finite after clipping")

    before_update = [
        parameter.detach().clone() for parameter in trainable_parameters
    ]
    optimizer.step()
    update_norm = _parameter_update_l2_norm(
        trainable_parameters,
        before_update,
    )
    if not math.isfinite(update_norm):
        raise FloatingPointError("parameter update norm is not finite")

    return TrainStepMetrics(
        loss=loss_value,
        token_accuracy=accuracy,
        gradient_l2_norm_before_clip=gradient_before_clip,
        gradient_l2_norm_after_clip=gradient_after_clip,
        parameter_update_l2_norm=update_norm,
    )


def overfit_fixed_batch(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    *,
    steps: int,
    record_every: int,
    max_grad_norm: float | None = None,
) -> OverfitResult:
    """Apply every optimizer update to the same batch and record bounded history."""

    _positive_integer(steps, "steps")
    _positive_integer(record_every, "record_every")
    _validate_max_grad_norm(max_grad_norm)

    initial = evaluate_batch(model, inputs, targets)
    history: list[dict[str, float | int | None]] = [
        {
            "fixed_batch_loss": initial.loss,
            "fixed_batch_token_accuracy": initial.token_accuracy,
            "gradient_l2_norm_after_clip": None,
            "gradient_l2_norm_before_clip": None,
            "loss_before_update": None,
            "parameter_update_l2_norm": None,
            "step": 0,
        }
    ]
    final = initial
    for step in range(1, steps + 1):
        step_metrics = train_step(
            model,
            optimizer,
            inputs,
            targets,
            max_grad_norm=max_grad_norm,
        )
        if step == 1 or step % record_every == 0 or step == steps:
            final = evaluate_batch(model, inputs, targets)
            history.append(
                {
                    "fixed_batch_loss": final.loss,
                    "fixed_batch_token_accuracy": final.token_accuracy,
                    "gradient_l2_norm_after_clip": (
                        step_metrics.gradient_l2_norm_after_clip
                    ),
                    "gradient_l2_norm_before_clip": (
                        step_metrics.gradient_l2_norm_before_clip
                    ),
                    "loss_before_update": step_metrics.loss,
                    "parameter_update_l2_norm": (
                        step_metrics.parameter_update_l2_norm
                    ),
                    "step": step,
                }
            )
    return OverfitResult(
        initial=initial,
        final=final,
        history=tuple(history),
    )
