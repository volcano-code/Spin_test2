"""PINO training conditioning inspired by PI-CViT and TensorPILS.

This module does not import their PDE-specific residual equations. It provides
a diagonal, fit-split-only preconditioner for Spin's ten-moment and 10x10
Jacobian residuals, plus deterministic PCGrad-style gradient conflict
projection for the data, moment-physics and Jacobian loss components.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import torch
from torch import Tensor, nn


@dataclass(frozen=True)
class DiagonalMomentResidualPreconditioner:
    moment_scale: Tensor
    jacobian_scale: Tensor
    moment_floor: float
    jacobian_floor: float
    fit_samples: int

    @classmethod
    def fit(
        cls,
        target_moments: Tensor,
        target_jacobians: Tensor,
        *,
        absolute_floor: float = 1e-4,
        relative_floor: float = 0.05,
    ) -> "DiagonalMomentResidualPreconditioner":
        if target_moments.ndim != 2 or target_moments.shape[-1] != 10:
            raise ValueError("target_moments must be [N,10]")
        if target_jacobians.ndim != 3 or target_jacobians.shape[-2:] != (10, 10):
            raise ValueError("target_jacobians must be [N,10,10]")
        if target_moments.shape[0] != target_jacobians.shape[0] or target_moments.shape[0] < 2:
            raise ValueError("moment and Jacobian fit targets need matching N>=2")
        if not all(bool(torch.isfinite(x).all()) for x in (target_moments, target_jacobians)):
            raise ValueError("fit targets must be finite")
        if absolute_floor <= 0 or not 0 <= relative_floor < 1:
            raise ValueError("invalid preconditioner floor")
        r_std = target_moments.std(dim=0, unbiased=False)
        j_std = target_jacobians.std(dim=0, unbiased=False)
        r_floor = max(float(absolute_floor), float(r_std.median()) * relative_floor)
        j_floor = max(float(absolute_floor), float(j_std.median()) * relative_floor)
        return cls(
            moment_scale=r_std.clamp_min(r_floor).detach(),
            jacobian_scale=j_std.clamp_min(j_floor).detach(),
            moment_floor=r_floor,
            jacobian_floor=j_floor,
            fit_samples=int(target_moments.shape[0]),
        )

    def to(self, device=None, dtype=None) -> "DiagonalMomentResidualPreconditioner":
        return DiagonalMomentResidualPreconditioner(
            moment_scale=self.moment_scale.to(device=device, dtype=dtype),
            jacobian_scale=self.jacobian_scale.to(device=device, dtype=dtype),
            moment_floor=self.moment_floor,
            jacobian_floor=self.jacobian_floor,
            fit_samples=self.fit_samples,
        )

    def as_dict(self) -> dict:
        return {
            "kind": "DIAGONAL_FIT_SPLIT_RESIDUAL_PRECONDITIONER",
            "fit_samples": self.fit_samples,
            "moment_scale": [float(x) for x in self.moment_scale],
            "jacobian_scale": [[float(x) for x in row] for row in self.jacobian_scale],
            "moment_floor": self.moment_floor,
            "jacobian_floor": self.jacobian_floor,
            "validation_or_test_used_for_fit": False,
        }


def project_conflicting_gradients(gradients: Sequence[Tensor], eps: float = 1e-12):
    """Deterministic PCGrad projection; returns projected vectors and cosines."""
    if len(gradients) < 2:
        raise ValueError("at least two task gradients are required")
    if any(g.ndim != 1 or not bool(torch.isfinite(g).all()) for g in gradients):
        raise ValueError("gradients must be finite flat vectors")
    if len({(g.shape, g.device, g.dtype) for g in gradients}) != 1:
        raise ValueError("all task gradients must share shape/device/dtype")
    original = [g.detach().clone() for g in gradients]
    projected = [g.detach().clone() for g in gradients]
    cosines = {}
    # PINO's factorized Fourier layers contain complex parameters. Use the
    # real Hermitian inner product so PCGrad coefficients remain real scalars.
    def real_dot(a: Tensor, b: Tensor) -> Tensor:
        return torch.vdot(a.reshape(-1), b.reshape(-1)).real

    for i in range(len(original)):
        for j in range(i + 1, len(original)):
            denom = original[i].norm() * original[j].norm()
            cosine = real_dot(original[i], original[j]) / denom.clamp_min(eps)
            cosines[f"{i}_{j}"] = float(cosine.detach().cpu().item())
    projections = 0
    for i in range(len(projected)):
        for j in range(len(original)):
            if i == j:
                continue
            reference = original[j]
            dot = real_dot(projected[i], reference)
            norm2 = real_dot(reference, reference)
            if float(dot.detach().cpu().item()) < 0.0 and float(norm2.detach().cpu().item()) > eps:
                projected[i] = projected[i] - (dot / (norm2 + eps)) * reference
                projections += 1
    return projected, {"raw_pairwise_cosines": cosines, "projection_count": projections}


def pcgrad_backward(
    loss_terms: Sequence[Tensor],
    parameters: Iterable[nn.Parameter],
    weights: Sequence[float],
) -> dict:
    """Compute per-loss gradients, project conflicts, and assign combined .grad."""
    params = [p for p in parameters if p.requires_grad]
    if len(loss_terms) < 2 or len(loss_terms) != len(weights):
        raise ValueError("loss_terms and weights need matching length >=2")
    if any(w < 0 for w in weights):
        raise ValueError("loss weights must be nonnegative")
    flat_grads = []
    shapes = [p.shape for p in params]
    complex_flags = [p.is_complex() for p in params]
    sizes = [p.numel() * (2 if p.is_complex() else 1) for p in params]
    for index, loss in enumerate(loss_terms):
        grads = torch.autograd.grad(
            loss, params,
            retain_graph=(index < len(loss_terms) - 1),
            allow_unused=True,
        )
        pieces = []
        for parameter, grad in zip(params, grads):
            value = torch.zeros_like(parameter) if grad is None else grad
            # Map complex Fourier gradients to their real/imaginary coordinates.
            # This gives PCGrad a real Hilbert-space vector and allows safe
            # reconstruction into mixed real/complex model parameters.
            if parameter.is_complex():
                value = torch.view_as_real(value).reshape(-1)
            else:
                value = value.reshape(-1)
            pieces.append(value)
        flat_grads.append(torch.cat(pieces))
    projected, stats = project_conflicting_gradients(flat_grads)
    combined = sum(float(w) * g for w, g in zip(weights, projected))
    offset = 0
    for parameter, size, shape, is_complex in zip(params, sizes, shapes, complex_flags):
        piece = combined[offset:offset + size]
        if is_complex:
            grad = torch.view_as_complex(piece.reshape(*shape, 2).contiguous())
        else:
            grad = piece.reshape(shape)
        parameter.grad = grad.detach().clone()
        offset += size
    stats["task_gradient_norms"] = [float(g.norm()) for g in flat_grads]
    stats["combined_gradient_norm"] = float(combined.norm())
    return stats
