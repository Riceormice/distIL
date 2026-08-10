"""Numerically stable SR-OPSD distribution construction and Rényi loss."""

from __future__ import annotations

import math

import torch


def validate_renyi_order(rho: float) -> None:
    """Validate a finite Rényi order in the non-singular domain."""
    if not math.isfinite(rho) or rho <= 0.0 or math.isclose(rho, 1.0, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError(f"rho must be finite, positive, and different from 1; got {rho}")


def normalize_log_probs(log_probs: torch.Tensor) -> torch.Tensor:
    """Return log probabilities normalized over the final dimension."""
    work = log_probs.float() if log_probs.dtype in (torch.float16, torch.bfloat16) else log_probs
    return work - torch.logsumexp(work, dim=-1, keepdim=True)


def add_tail_bucket(log_probs: torch.Tensor) -> torch.Tensor:
    """Append the omitted probability mass to a selected-token distribution."""
    work = log_probs.float() if log_probs.dtype in (torch.float16, torch.bfloat16) else log_probs
    log_selected_mass = torch.logsumexp(work, dim=-1, keepdim=True)
    eps = torch.finfo(work.dtype).eps
    log_selected_mass = torch.clamp(log_selected_mass, max=-eps)
    log_tail_mass = torch.log(-torch.expm1(log_selected_mass))
    return normalize_log_probs(torch.cat((work, log_tail_mass), dim=-1))


def geometric_target_log_probs(
    teacher_log_probs: torch.Tensor,
    reference_log_probs: torch.Tensor | None,
    teacher_weight: float,
) -> torch.Tensor:
    """Construct the normalized geometric teacher/reference target."""
    if not 0.0 <= teacher_weight <= 1.0:
        raise ValueError(f"teacher_weight must be in [0, 1]; got {teacher_weight}")

    teacher = normalize_log_probs(teacher_log_probs)
    if reference_log_probs is None:
        if teacher_weight != 1.0:
            raise ValueError("reference_log_probs is required when teacher_weight is smaller than 1")
        return teacher

    reference = normalize_log_probs(reference_log_probs)
    mixed = teacher_weight * teacher + (1.0 - teacher_weight) * reference
    return normalize_log_probs(mixed)


def forward_renyi_divergence(
    target_log_probs: torch.Tensor,
    student_log_probs: torch.Tensor,
    rho: float,
) -> torch.Tensor:
    """Compute D_rho(target || student) per token."""
    validate_renyi_order(rho)
    target = normalize_log_probs(target_log_probs)
    student = normalize_log_probs(student_log_probs)
    return torch.logsumexp(rho * target + (1.0 - rho) * student, dim=-1) / (rho - 1.0)
