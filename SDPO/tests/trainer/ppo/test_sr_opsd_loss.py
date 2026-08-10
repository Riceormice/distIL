import pytest
import torch

from verl.trainer.ppo.sr_opsd_loss import (
    add_tail_bucket,
    forward_renyi_divergence,
    geometric_target_log_probs,
    normalize_log_probs,
)


def test_geometric_target_endpoints_and_normalization():
    teacher = torch.log(torch.tensor([[0.7, 0.2, 0.1]]))
    reference = torch.log(torch.tensor([[0.1, 0.3, 0.6]]))

    teacher_only = geometric_target_log_probs(teacher, reference, 1.0)
    reference_only = geometric_target_log_probs(teacher, reference, 0.0)
    mixed = geometric_target_log_probs(teacher, reference, 0.9)

    torch.testing.assert_close(teacher_only.exp(), teacher.exp())
    torch.testing.assert_close(reference_only.exp(), reference.exp())
    torch.testing.assert_close(mixed.exp().sum(-1), torch.ones(1))


def test_tail_bucket_preserves_omitted_mass():
    selected = torch.log(torch.tensor([[0.4, 0.35]]))
    with_tail = add_tail_bucket(selected)
    torch.testing.assert_close(with_tail.exp(), torch.tensor([[0.4, 0.35, 0.25]]))


def test_forward_renyi_is_zero_for_identical_distributions_and_has_gradient():
    target = normalize_log_probs(torch.tensor([[1.0, -0.5, 0.25]]))
    same = forward_renyi_divergence(target, target, rho=0.95)
    torch.testing.assert_close(same, torch.zeros_like(same), atol=1e-6, rtol=0)

    student_logits = torch.tensor([[0.2, 0.1, -0.4]], requires_grad=True)
    loss = forward_renyi_divergence(target, student_logits, rho=0.95).mean()
    loss.backward()
    assert torch.isfinite(loss)
    assert student_logits.grad is not None
    assert torch.isfinite(student_logits.grad).all()


@pytest.mark.parametrize("rho", [-1.0, 0.0, 1.0, float("inf"), float("nan")])
def test_invalid_renyi_order_is_rejected(rho):
    distribution = torch.log(torch.tensor([[0.5, 0.5]]))
    with pytest.raises(ValueError):
        forward_renyi_divergence(distribution, distribution, rho=rho)
