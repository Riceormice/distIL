from types import SimpleNamespace

import pytest
import torch

from verl.trainer.ppo.core_algos import compute_renyi_kl_loss, compute_self_distillation_loss
from verl.workers.config.actor import SelfDistillationConfig


def native_config(**overrides):
    values = {
        "full_logit_distillation": True,
        "distillation_topk": None,
        "distillation_add_tail": True,
        "renyi_regularization": True,
        "renyi_regularization_level": 0.9,
        "alpha": 0.25,
        "rho": 0.95,
        "is_clip": None,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_native_forward_renyi_formula():
    teacher = torch.log_softmax(torch.tensor([[2.0, 0.5, -0.5]]), dim=-1)
    student = torch.log_softmax(torch.tensor([[0.2, 1.0, -0.1]]), dim=-1)
    cfg = native_config(rho=0.7)

    actual = compute_renyi_kl_loss(teacher, student, cfg)
    expected = torch.logsumexp(0.7 * teacher + 0.3 * student, dim=-1, keepdim=True) / -0.3

    torch.testing.assert_close(actual, expected)


def test_native_sr_opsd_mixes_ema_teacher_and_frozen_reference():
    student = torch.log_softmax(torch.tensor([[[0.2, 1.0, -0.1], [0.4, -0.2, 0.7]]]), dim=-1)
    teacher = torch.log_softmax(torch.tensor([[[2.0, 0.5, -0.5], [0.1, 1.1, -0.3]]]), dim=-1)
    reference = torch.log_softmax(torch.tensor([[[0.3, 0.2, 1.0], [1.0, 0.0, -0.2]]]), dim=-1)
    mask = torch.ones((1, 2))
    cfg = native_config(rho=0.7, renyi_regularization_level=0.8)

    loss, _ = compute_self_distillation_loss(
        student_log_probs=torch.zeros((1, 2)),
        teacher_log_probs=torch.zeros((1, 2)),
        response_mask=mask,
        self_distillation_config=cfg,
        student_all_log_probs=student,
        teacher_all_log_probs=teacher,
        ref_all_log_probs=reference,
    )

    target = 0.8 * teacher + 0.2 * reference
    expected_per_token = torch.logsumexp(0.7 * target + 0.3 * student, dim=-1) / -0.3
    torch.testing.assert_close(loss, expected_per_token.mean())


def test_native_sr_opsd_requires_reference_logits():
    probs = torch.log_softmax(torch.tensor([[[0.2, 1.0, -0.1]]]), dim=-1)
    with pytest.raises(ValueError, match="requires reference-model"):
        compute_self_distillation_loss(
            student_log_probs=torch.zeros((1, 1)),
            teacher_log_probs=torch.zeros((1, 1)),
            response_mask=torch.ones((1, 1)),
            self_distillation_config=native_config(),
            student_all_log_probs=probs,
            teacher_all_log_probs=probs,
        )


def test_native_config_rejects_invalid_renyi_parameters():
    with pytest.raises(ValueError, match="different from 1"):
        SelfDistillationConfig(alpha=0.25, rho=1.0)
    with pytest.raises(ValueError, match=r"must be in \[0,1\]"):
        SelfDistillationConfig(renyi_regularization=True, renyi_regularization_level=1.1)
