"""Temperature-scaled logit distillation for the Direct-Small student."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Mapping

import torch
from torch.nn import functional as F

from nanogpt_nspire.direct_small_train import TrainingObjectiveStep
from nanogpt_nspire.models.direct_small_gpt import DirectSmallGPT


@dataclass(frozen=True)
class DistillationLosses:
    """The hard, soft and weighted scalar losses for one student batch."""

    total_loss: torch.Tensor
    hard_label_loss: torch.Tensor
    soft_target_loss: torch.Tensor


def distillation_losses(
    *,
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    targets: torch.Tensor,
    target_mask: torch.Tensor | None = None,
    temperature: float,
    alpha: float,
) -> DistillationLosses:
    """Combine next-token cross-entropy with temperature-scaled KL."""

    if (
        not isinstance(student_logits, torch.Tensor)
        or not student_logits.is_floating_point()
        or student_logits.ndim != 3
    ):
        raise ValueError("student_logits must be floating-point (B, T, V)")
    if (
        not isinstance(teacher_logits, torch.Tensor)
        or not teacher_logits.is_floating_point()
        or teacher_logits.shape != student_logits.shape
    ):
        raise ValueError(
            "teacher_logits must have the same shape as student_logits"
        )
    if (
        not isinstance(targets, torch.Tensor)
        or targets.dtype != torch.long
        or targets.shape != student_logits.shape[:2]
    ):
        raise ValueError("targets must be torch.long with shape (B, T)")
    if target_mask is None:
        target_mask = torch.ones_like(targets, dtype=torch.bool)
    if (
        not isinstance(target_mask, torch.Tensor)
        or target_mask.shape != targets.shape
        or target_mask.dtype == torch.long
        or not (
            target_mask.dtype == torch.bool
            or target_mask.is_floating_point()
        )
    ):
        raise ValueError(
            "target_mask must be boolean or floating-point with shape (B, T)"
        )
    if target_mask.is_floating_point() and (
        not torch.isfinite(target_mask).all()
        or not bool(((target_mask == 0) | (target_mask == 1)).all())
    ):
        raise ValueError("target_mask values must be finite zero or one")
    eligible = target_mask.reshape(-1).to(student_logits.dtype)
    eligible_count = eligible.sum()
    if not bool((eligible_count > 0).item()):
        raise ValueError("target_mask must contain an eligible target")
    if (
        isinstance(temperature, bool)
        or not isinstance(temperature, (int, float))
        or not math.isfinite(temperature)
        or temperature <= 0
    ):
        raise ValueError("temperature must be finite and positive")
    if (
        isinstance(alpha, bool)
        or not isinstance(alpha, (int, float))
        or not math.isfinite(alpha)
        or not 0 <= alpha <= 1
    ):
        raise ValueError("alpha must be finite and in [0, 1]")
    if (
        not torch.isfinite(student_logits).all()
        or not torch.isfinite(teacher_logits).all()
    ):
        raise ValueError("student and teacher logits must be finite")

    vocabulary_size = student_logits.shape[-1]
    student_flat = student_logits.reshape(-1, vocabulary_size)
    teacher_flat = teacher_logits.detach().reshape(-1, vocabulary_size)
    targets_flat = targets.reshape(-1)
    hard_per_token = F.cross_entropy(
        student_flat,
        targets_flat,
        reduction="none",
    )
    hard_label_loss = (hard_per_token * eligible).sum() / eligible_count
    student_log_probabilities = F.log_softmax(
        student_flat / temperature,
        dim=-1,
    )
    teacher_probabilities = F.softmax(
        teacher_flat / temperature,
        dim=-1,
    )
    soft_per_class = F.kl_div(
        student_log_probabilities,
        teacher_probabilities,
        reduction="none",
    )
    soft_per_token = soft_per_class.sum(dim=-1) * temperature**2
    soft_target_loss = (soft_per_token * eligible).sum() / eligible_count
    total_loss = (
        (1.0 - alpha) * hard_label_loss
        + alpha * soft_target_loss
    )
    return DistillationLosses(
        total_loss=total_loss,
        hard_label_loss=hard_label_loss,
        soft_target_loss=soft_target_loss,
    )


class DistillationObjective:
    """Keep a frozen Teacher in eval mode and train only the student."""

    def __init__(
        self,
        teacher: DirectSmallGPT,
        *,
        temperature: float,
        alpha: float,
        teacher_provenance: Mapping[str, object],
    ) -> None:
        if not isinstance(teacher, DirectSmallGPT):
            raise ValueError("teacher must be a DirectSmallGPT")
        if not isinstance(teacher_provenance, Mapping):
            raise ValueError("teacher_provenance must be a mapping")
        self.teacher = teacher.eval()
        self.teacher.requires_grad_(False)
        self.temperature = temperature
        self.alpha = alpha
        self.teacher_provenance = dict(teacher_provenance)
        probe = torch.zeros((1, 1, teacher.vocab_size))
        targets = torch.zeros((1, 1), dtype=torch.long)
        distillation_losses(
            student_logits=probe,
            teacher_logits=probe,
            targets=targets,
            temperature=temperature,
            alpha=alpha,
        )

    def __call__(
        self,
        model: DirectSmallGPT,
        inputs: torch.Tensor,
        targets: torch.Tensor,
        target_mask: torch.Tensor | None = None,
    ) -> TrainingObjectiveStep:
        self.teacher.eval()
        student_logits, _ = model(inputs)
        with torch.inference_mode():
            teacher_logits, _ = self.teacher(inputs)
        losses = distillation_losses(
            student_logits=student_logits,
            teacher_logits=teacher_logits,
            targets=targets,
            target_mask=target_mask,
            temperature=self.temperature,
            alpha=self.alpha,
        )
        return TrainingObjectiveStep(
            loss=losses.total_loss,
            metrics={
                "hard_label_loss": float(
                    losses.hard_label_loss.detach().item()
                ),
                "soft_target_loss": float(
                    losses.soft_target_loss.detach().item()
                ),
            },
        )

    def summary(self) -> dict[str, object]:
        return {
            "alpha": self.alpha,
            "name": "temperature_scaled_logit_distillation",
            "teacher": dict(sorted(self.teacher_provenance.items())),
            "temperature": self.temperature,
        }
