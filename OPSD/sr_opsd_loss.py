"""Numerically stable SR-OPSD distribution construction and Rényi loss."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F


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


def topk_with_tail_log_probs(
    student_log_probs: torch.Tensor,
    teacher_log_probs: torch.Tensor,
    reference_log_probs: torch.Tensor | None,
    top_k: int | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """Use student-selected top-k outcomes plus one normalized tail outcome."""
    student = normalize_log_probs(student_log_probs)
    teacher = normalize_log_probs(teacher_log_probs)
    reference = normalize_log_probs(reference_log_probs) if reference_log_probs is not None else None

    vocab_size = student.size(-1)
    if top_k is None or top_k <= 0 or top_k >= vocab_size:
        return student, teacher, reference

    indices = torch.topk(student, k=top_k, dim=-1).indices
    student = add_tail_bucket(torch.gather(student, dim=-1, index=indices))
    teacher = add_tail_bucket(torch.gather(teacher, dim=-1, index=indices))
    if reference is not None:
        reference = add_tail_bucket(torch.gather(reference, dim=-1, index=indices))
    return student, teacher, reference


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


def sr_opsd_loss_from_logits(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    reference_logits: torch.Tensor | None,
    *,
    labels: torch.Tensor | None = None,
    rho: float = 0.95,
    teacher_weight: float = 0.9,
    temperature: float = 1.0,
    top_k: int | None = None,
    token_clip: float | None = None,
    reduction: str = "batchmean",
) -> torch.Tensor:
    """Build the SR-OPSD target and reduce its forward Rényi divergence."""
    if temperature <= 0.0:
        raise ValueError(f"temperature must be positive; got {temperature}")
    if reference_logits is None and teacher_weight != 1.0:
        raise ValueError("reference_logits is required when teacher_weight is smaller than 1")

    student = F.log_softmax(student_logits / temperature, dim=-1)
    teacher = F.log_softmax(teacher_logits / temperature, dim=-1)
    reference = F.log_softmax(reference_logits / temperature, dim=-1) if reference_logits is not None else None
    student, teacher, reference = topk_with_tail_log_probs(student, teacher, reference, top_k)
    target = geometric_target_log_probs(teacher, reference, teacher_weight)
    per_token = forward_renyi_divergence(target, student, rho)

    if token_clip is not None:
        if token_clip <= 0.0:
            raise ValueError(f"token_clip must be positive when set; got {token_clip}")
        per_token = per_token.clamp(max=token_clip)

    if labels is not None:
        per_token = per_token[labels != -100]

    if reduction == "batchmean":
        if labels is not None:
            return per_token.sum() / max(per_token.numel(), 1)
        return per_token.sum() / max(student_logits.size(0), 1)
    if reduction == "sum":
        return per_token.sum()
    if reduction == "mean":
        return per_token.mean()
    if reduction == "none":
        return per_token
    raise ValueError(f"Unsupported reduction: {reduction}")
