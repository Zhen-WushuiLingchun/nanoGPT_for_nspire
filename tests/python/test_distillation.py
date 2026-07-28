import pytest
import torch

from nanogpt_nspire.distillation import (
    DistillationObjective,
    distillation_losses,
)
from nanogpt_nspire.models.direct_small_gpt import (
    DirectSmallConfig,
    DirectSmallGPT,
)


def test_equal_logits_have_zero_soft_loss() -> None:
    logits = torch.tensor(
        [[[2.0, 0.0, -1.0], [0.5, 0.0, -0.5]]],
        requires_grad=True,
    )
    targets = torch.tensor([[0, 2]], dtype=torch.long)

    losses = distillation_losses(
        student_logits=logits,
        teacher_logits=logits.detach().clone(),
        targets=targets,
        temperature=2.0,
        alpha=0.5,
    )

    assert losses.soft_target_loss.item() == pytest.approx(0.0, abs=1e-7)
    assert losses.total_loss.item() == pytest.approx(
        0.5 * losses.hard_label_loss.item(),
    )


def test_distillation_gradient_flows_only_to_student() -> None:
    student_logits = torch.tensor(
        [[[1.0, 0.0, -1.0], [0.2, 0.3, 0.4]]],
        requires_grad=True,
    )
    teacher_logits = torch.tensor(
        [[[2.0, -1.0, 0.0], [-0.5, 0.0, 1.5]]],
        requires_grad=True,
    )
    targets = torch.tensor([[0, 2]], dtype=torch.long)

    losses = distillation_losses(
        student_logits=student_logits,
        teacher_logits=teacher_logits,
        targets=targets,
        temperature=2.0,
        alpha=0.5,
    )
    losses.total_loss.backward()

    assert student_logits.grad is not None
    assert torch.isfinite(student_logits.grad).all()
    assert teacher_logits.grad is None
    assert losses.total_loss.item() == pytest.approx(
        0.5 * losses.hard_label_loss.item()
        + 0.5 * losses.soft_target_loss.item()
    )


def test_assistant_mask_excludes_user_positions_from_hard_and_soft_loss() -> None:
    student = torch.tensor(
        [[[5.0, -5.0], [0.0, 1.0], [100.0, -100.0]]],
        requires_grad=True,
    )
    teacher = torch.tensor(
        [[[-5.0, 5.0], [1.0, 0.0], [-100.0, 100.0]]],
    )
    targets = torch.tensor([[1, 1, 0]], dtype=torch.long)
    mask = torch.tensor([[0.0, 1.0, 0.0]])

    losses = distillation_losses(
        student_logits=student,
        teacher_logits=teacher,
        targets=targets,
        target_mask=mask,
        temperature=2.0,
        alpha=0.5,
    )
    changed_student = student.detach().clone()
    changed_teacher = teacher.clone()
    changed_student[:, 0, :] = torch.tensor([-999.0, 999.0])
    changed_student[:, 2, :] = torch.tensor([999.0, -999.0])
    changed_teacher[:, 0, :] = torch.tensor([999.0, -999.0])
    changed_teacher[:, 2, :] = torch.tensor([-999.0, 999.0])
    changed = distillation_losses(
        student_logits=changed_student,
        teacher_logits=changed_teacher,
        targets=targets,
        target_mask=mask,
        temperature=2.0,
        alpha=0.5,
    )
    losses.total_loss.backward()

    assert changed.hard_label_loss.item() == pytest.approx(
        losses.hard_label_loss.item()
    )
    assert changed.soft_target_loss.item() == pytest.approx(
        losses.soft_target_loss.item()
    )
    assert torch.count_nonzero(student.grad[:, (0, 2), :]) == 0
    assert torch.count_nonzero(student.grad[:, 1, :]) > 0


def test_distillation_rejects_empty_target_mask() -> None:
    with pytest.raises(ValueError, match="eligible"):
        distillation_losses(
            student_logits=torch.zeros((1, 2, 3)),
            teacher_logits=torch.zeros((1, 2, 3)),
            targets=torch.zeros((1, 2), dtype=torch.long),
            target_mask=torch.zeros((1, 2)),
            temperature=2.0,
            alpha=0.5,
        )


def test_bfloat16_logits_use_float32_loss_math() -> None:
    losses = distillation_losses(
        student_logits=torch.zeros(
            (1, 2, 3),
            dtype=torch.bfloat16,
            requires_grad=True,
        ),
        teacher_logits=torch.ones((1, 2, 3), dtype=torch.bfloat16),
        targets=torch.zeros((1, 2), dtype=torch.long),
        target_mask=torch.ones((1, 2), dtype=torch.bool),
        temperature=2.0,
        alpha=0.5,
    )

    assert losses.total_loss.dtype == torch.float32
    assert losses.hard_label_loss.dtype == torch.float32
    assert losses.soft_target_loss.dtype == torch.float32


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"temperature": 0.0}, "temperature"),
        ({"alpha": -0.1}, "alpha"),
        ({"alpha": 1.1}, "alpha"),
        (
            {"teacher_logits": torch.zeros((1, 3, 3))},
            "same shape",
        ),
        (
            {"targets": torch.zeros((1, 3), dtype=torch.long)},
            "targets",
        ),
    ],
)
def test_distillation_losses_reject_invalid_inputs(kwargs, message) -> None:
    arguments = {
        "student_logits": torch.zeros((1, 2, 3)),
        "teacher_logits": torch.zeros((1, 2, 3)),
        "targets": torch.zeros((1, 2), dtype=torch.long),
        "temperature": 2.0,
        "alpha": 0.5,
    }
    arguments.update(kwargs)

    with pytest.raises(ValueError, match=message):
        distillation_losses(**arguments)


def test_distillation_objective_freezes_teacher_and_reports_components() -> None:
    config = DirectSmallConfig(
        vocab_size=7,
        block_size=8,
        n_layer=1,
        n_head=2,
        n_embd=16,
        mlp_ratio=2,
        dropout=0.0,
    )
    teacher = DirectSmallGPT(config)
    objective = DistillationObjective(
        teacher,
        temperature=2.0,
        alpha=0.5,
        teacher_provenance={
            "route": "Teacher-v2",
            "checkpoint_sha256": "a" * 64,
        },
    )
    student = DirectSmallGPT(config)
    inputs = torch.tensor([[0, 1, 2, 3]], dtype=torch.long)
    targets = torch.tensor([[1, 2, 3, 4]], dtype=torch.long)

    target_mask = torch.tensor([[0.0, 1.0, 1.0, 1.0]])
    result = objective(student, inputs, targets, target_mask)
    result.loss.backward()

    assert not objective.teacher.training
    assert all(
        not parameter.requires_grad
        for parameter in objective.teacher.parameters()
    )
    assert all(
        parameter.grad is None
        for parameter in objective.teacher.parameters()
    )
    assert set(result.metrics) == {
        "hard_label_loss",
        "soft_target_loss",
    }
    assert all(
        torch.isfinite(parameter.grad).all()
        for parameter in student.parameters()
        if parameter.grad is not None
    )
    assert objective.summary() == {
        "alpha": 0.5,
        "name": "temperature_scaled_logit_distillation",
        "teacher": {
            "checkpoint_sha256": "a" * 64,
            "route": "Teacher-v2",
        },
        "temperature": 2.0,
    }
