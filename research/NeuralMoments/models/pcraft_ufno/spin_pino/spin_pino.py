"""Spin-specific PINO: P-CRAFT UFNO backbone + physics-constrained moment potential."""

from __future__ import annotations

import math
from typing import Optional

import numpy as np
import torch
from torch import Tensor, nn
import torch.nn.functional as F

from .pino_backbone import PINOFeatureEncoder
from ..model import canonical_geometry

MOMENT_NAMES = ("mass", "x", "y", "z", "xx", "yy", "zz", "yz", "xz", "xy")
COEFFICIENT_NAMES = ("const", "x", "y", "z", "xx", "yy", "zz", "yz", "xz", "xy")
NATIVE_B = (1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0)


def quadrature_rule_data(points_per_axis: int = 4, rule: str = "midpoint"):
    """Return xyz[N,3], raw monomials[N,10], and positive volume weights[N]."""
    if not isinstance(points_per_axis, int) or points_per_axis < 2:
        raise ValueError("points_per_axis must be an integer >=2")
    if rule == "midpoint":
        axis = (torch.arange(points_per_axis, dtype=torch.float64) + 0.5) * (
            2.0 / points_per_axis
        ) - 1.0
        xyz = torch.stack(torch.meshgrid(axis, axis, axis, indexing="ij"), dim=-1).reshape(-1, 3)
        weights = torch.full((xyz.shape[0],), 8.0 / xyz.shape[0], dtype=torch.float64)
    elif rule == "gauss":
        nodes, one_d_weights = np.polynomial.legendre.leggauss(points_per_axis)
        nodes_t = torch.as_tensor(nodes.copy(), dtype=torch.float64)
        weights_t = torch.as_tensor(one_d_weights.copy(), dtype=torch.float64)
        xyz = torch.stack(torch.meshgrid(nodes_t, nodes_t, nodes_t, indexing="ij"), dim=-1).reshape(-1, 3)
        wx, wy, wz = torch.meshgrid(weights_t, weights_t, weights_t, indexing="ij")
        weights = (wx * wy * wz).reshape(-1)
    else:
        raise ValueError("quadrature rule must be 'midpoint' or 'gauss'")
    x, y, z = xyz.unbind(-1)
    monomials = torch.stack(
        (torch.ones_like(x), x, y, z, x*x, y*y, z*z, y*z, x*z, x*y),
        dim=-1,
    )
    if not bool((weights > 0).all()) or not math.isclose(float(weights.sum()), 8.0, rel_tol=1e-10):
        raise RuntimeError("quadrature rule must have positive weights summing to volume 8")
    return xyz.float(), monomials.float(), weights.float()


def quadrature_nodes(points_per_axis: int = 4, rule: str = "midpoint") -> Tensor:
    """Compatibility helper returning only the ten raw monomials."""
    return quadrature_rule_data(points_per_axis, rule)[1]


class SpinPINO(nn.Module):
    """PINO remains the primary backbone; plugins attach only at defined ports.

    The geometry trunk is the P-CRAFT UFNO3d factorized-Fourier U-shaped encoder.
    A positive geometry-conditioned quadrature head predicts nonnegative node
    weights. A smooth convex hinge potential then yields ten raw moments and
    its coefficient Jacobian analytically. No external operator can replace
    the PINO trunk in this class.
    """

    def __init__(
        self,
        *,
        width: int = 8,
        modes: int = 4,
        n_layers: int = 4,
        grid_size: int = 16,
        points_per_axis: int = 4,
        hidden_dim: int = 96,
        smoothing: float = 0.04,
        quadrature_rule: str = "midpoint",
        moment_parameterization: str = "smooth_hinge",
        temperature: float = 0.12,
        use_gradient_checkpoint: bool = False,
    ):
        super().__init__()
        if not math.isfinite(smoothing) or not 0.0 < smoothing <= 1.0:
            raise ValueError("smoothing must be finite and in (0,1]")
        if moment_parameterization not in ("smooth_hinge", "logistic_atom"):
            raise ValueError("moment_parameterization must be smooth_hinge or logistic_atom")
        if not math.isfinite(temperature) or temperature <= 0:
            raise ValueError("temperature must be finite and positive")
        self.encoder = PINOFeatureEncoder(
            width=width,
            modes=modes,
            n_layers=n_layers,
            grid_size=grid_size,
            use_gradient_checkpoint=use_gradient_checkpoint,
        )
        self.points_per_axis = points_per_axis
        self.smoothing = float(smoothing)
        self.quadrature_rule = quadrature_rule
        self.moment_parameterization = moment_parameterization
        self.temperature = float(temperature)
        self.weight_head = nn.Sequential(
            nn.Linear(self.encoder.feature_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, points_per_axis ** 3),
        )
        qxyz, monomials, base_weights = quadrature_rule_data(points_per_axis, quadrature_rule)
        self.register_buffer("monomials", monomials, persistent=True)
        self.register_buffer("quadrature_xyz", qxyz, persistent=True)
        # grid_sample input is [B,C,X,Y,Z], so its coordinate order is (z,y,x).
        # Keep physical xyz separately for monomials and reverse only at this port.
        sampling_grid = torch.stack((qxyz[:, 2], qxyz[:, 1], qxyz[:, 0]), dim=-1)
        self.register_buffer(
            "voxel_sampling_grid",
            sampling_grid.reshape(1, -1, 1, 1, 3),
            persistent=True,
        )
        self.register_buffer("native_B", torch.tensor(NATIVE_B), persistent=True)
        self.register_buffer("node_volume", base_weights, persistent=True)

    def encode_geometry(self, geometry: Tensor) -> Tensor:
        canonical = canonical_geometry(geometry)
        features = self.encoder(canonical)
        # Enforce support and density at every quadrature node. The learned
        # head may redistribute mass only inside occupied material; it cannot
        # create atoms in empty space. This is a discrete-measure realizability
        # constraint, not a continuum/CAD certificate.
        rho = canonical[:, 0:1]
        occupancy = canonical[:, 1:2]
        grid = self.voxel_sampling_grid.to(device=canonical.device, dtype=canonical.dtype)
        grid = grid.expand(canonical.shape[0], -1, -1, -1, -1)
        rho_nodes = F.grid_sample(
            rho, grid, mode="bilinear", padding_mode="zeros", align_corners=True
        ).flatten(1)
        occupancy_nodes = F.grid_sample(
            occupancy, grid, mode="bilinear", padding_mode="zeros", align_corners=True
        ).flatten(1)
        learned_gate = torch.sigmoid(self.weight_head(features))
        return (
            self.node_volume.to(features).unsqueeze(0)
            * rho_nodes.to(features)
            * occupancy_nodes.to(features)
            * learned_gate
        )

    def _validate(self, weights: Tensor, coefficients: Tensor) -> None:
        if weights.ndim != 2 or weights.shape[1] != self.monomials.shape[0]:
            raise ValueError("weights must have shape [B, points_per_axis**3]")
        if coefficients.ndim != 2 or coefficients.shape != (weights.shape[0], 10):
            raise ValueError("native coefficients must have shape [B,10]")
        if weights.device != coefficients.device or weights.dtype != coefficients.dtype:
            raise ValueError("weights and coefficients must share dtype and device")
        if not weights.is_floating_point() or not coefficients.is_floating_point():
            raise TypeError("weights and coefficients must be floating-point")
        if not bool(torch.isfinite(weights).all() and torch.isfinite(coefficients).all()):
            raise ValueError("weights and coefficients must be finite")
        if bool((weights < 0).any()):
            raise ValueError("quadrature weights must be nonnegative")
        if bool((torch.linalg.vector_norm(coefficients, dim=-1) <= 1e-12).any()):
            raise ValueError("native coefficient vector must be nonzero")

    def _terms(self, weights: Tensor, coefficients: Tensor):
        self._validate(weights, coefficients)
        c = coefficients / torch.linalg.vector_norm(coefficients, dim=-1, keepdim=True)
        B = self.native_B.to(dtype=c.dtype, device=c.device)
        q = c * B
        m = self.monomials.to(dtype=c.dtype, device=c.device)
        a = torch.einsum("bi,ni->bn", q, m)
        eps2 = self.smoothing ** 2
        norm2 = q.square().sum(dim=-1, keepdim=True)
        s = torch.sqrt(a.square() + eps2 * norm2)
        n = a.unsqueeze(-1) * m.unsqueeze(0) + eps2 * q.unsqueeze(1)
        return c, q, B, m, a, s, n, eps2

    def _logistic_atom_terms(self, weights: Tensor, coefficients: Tensor):
        self._validate(weights, coefficients)
        B = self.native_B.to(dtype=coefficients.dtype, device=coefficients.device)
        m = self.monomials.to(dtype=coefficients.dtype, device=coefficients.device)
        norm = torch.linalg.vector_norm(coefficients, dim=-1, keepdim=True)
        a = torch.einsum("bi,ni->bn", coefficients * B, m)
        tau = self.temperature * norm
        alpha = torch.sigmoid(a / tau)
        grad_argument = (
            m.unsqueeze(0) * B.view(1, 1, 10)
            - (a / norm.square()).unsqueeze(-1) * coefficients.unsqueeze(1)
        )
        return B, m, norm, a, tau, alpha, grad_argument

    def potential_from_weights(self, weights: Tensor, coefficients: Tensor) -> Tensor:
        if self.moment_parameterization != "smooth_hinge":
            raise RuntimeError("logistic_atom is a realizable head, not the smooth convex-potential parameterization")
        _, _, _, _, a, s, _, _ = self._terms(weights, coefficients)
        return 0.5 * (weights * (a + s)).sum(dim=-1)

    def moment_from_weights(self, weights: Tensor, coefficients: Tensor) -> Tensor:
        if self.moment_parameterization == "logistic_atom":
            _, m, _, _, _, alpha, _ = self._logistic_atom_terms(weights, coefficients)
            return torch.einsum("bn,bn,ni->bi", weights, alpha, m)
        _, _, _, m, _, s, n, _ = self._terms(weights, coefficients)
        grad_q = 0.5 * (m.unsqueeze(0) + n / s.unsqueeze(-1))
        return torch.einsum("bn,bni->bi", weights, grad_q)

    def jacobian_from_weights(self, weights: Tensor, coefficients: Tensor) -> Tensor:
        if self.moment_parameterization == "logistic_atom":
            _, m, _, _, tau, alpha, grad_argument = self._logistic_atom_terms(weights, coefficients)
            factor = weights * alpha * (1.0 - alpha) / tau
            return torch.einsum("bn,ni,bnj->bij", factor, m, grad_argument)
        _, _, B, m, _, s, n, eps2 = self._terms(weights, coefficients)
        eye = torch.eye(10, dtype=weights.dtype, device=weights.device)
        mm = m.unsqueeze(-1) * m.unsqueeze(-2)
        hess = 0.5 * (
            (mm.unsqueeze(0) + eps2 * eye) / s[..., None, None]
            - (n.unsqueeze(-1) * n.unsqueeze(-2)) / s[..., None, None] ** 3
        )
        Jq = torch.einsum("bn,bnij->bij", weights, hess)
        # q = B c, so d r / d c = (d r / d q) B.
        return Jq * B.view(1, 1, 10)

    def projected_q_gram(self, weights: Tensor, coefficients: Tensor) -> Tensor:
        """Tangent-projected q-space Jacobian; symmetric PSD for logistic atoms."""
        Jc = self.jacobian_from_weights(weights, coefficients)
        B = self.native_B.to(dtype=coefficients.dtype, device=coefficients.device)
        Jq = Jc / B.view(1, 1, 10)
        normal = coefficients / B
        projector = (
            torch.eye(10, dtype=coefficients.dtype, device=coefficients.device).unsqueeze(0)
            - normal.unsqueeze(-1) * normal.unsqueeze(-2)
            / normal.square().sum(dim=-1, keepdim=True).unsqueeze(-1)
        )
        return projector @ Jq @ projector

    def forward(self, geometry: Tensor, coefficients: Tensor, *, with_jacobian: bool = False):
        weights = self.encode_geometry(geometry)
        moments = self.moment_from_weights(weights, coefficients)
        if with_jacobian:
            return moments, self.jacobian_from_weights(weights, coefficients)
        return moments

    def real_parameter_count(self) -> int:
        return sum(
            p.numel() * (2 if p.is_complex() else 1)
            for p in self.parameters() if p.requires_grad
        )


class SpinPINOLoss(nn.Module):
    """PINO-style data + moment-physics residual objective.

    The original P-CRAFT PINO equilibrium/constitutive losses describe linear
    elasticity and are not valid for this Spin integral task. They are replaced
    with exact identities of the smooth convex moment potential:
      * complement: r(c)+r(-c) equals the full weighted material moments;
      * radial: J_c c = 0 from positive projective scale invariance;
      * symmetry/PSD: J_q is a symmetric positive-semidefinite Hessian;
      * mass bounds: 0 <= mass(c) <= total represented material mass.

    A target Jacobian is accepted only when the caller explicitly marks its
    source as derivative-admitted. Diagnostic-only W2 records are not labels.
    """

    def __init__(
        self,
        *,
        lambda_data: float = 1.0,
        lambda_physics: float = 0.1,
        lambda_jacobian: float = 0.0,
        moment_scale: Optional[Tensor] = None,
        jacobian_scale: Optional[Tensor] = None,
    ):
        super().__init__()
        for name, value in (
            ("lambda_data", lambda_data),
            ("lambda_physics", lambda_physics),
            ("lambda_jacobian", lambda_jacobian),
        ):
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
        self.lambda_data = float(lambda_data)
        self.lambda_physics = float(lambda_physics)
        self.lambda_jacobian = float(lambda_jacobian)
        if moment_scale is not None:
            moment_scale = torch.as_tensor(moment_scale, dtype=torch.float32)
            if moment_scale.shape != (10,) or not bool(torch.isfinite(moment_scale).all()) or bool((moment_scale <= 0).any()):
                raise ValueError("moment_scale must be finite, positive and shape [10]")
        if jacobian_scale is not None:
            jacobian_scale = torch.as_tensor(jacobian_scale, dtype=torch.float32)
            if jacobian_scale.shape != (10, 10) or not bool(torch.isfinite(jacobian_scale).all()) or bool((jacobian_scale <= 0).any()):
                raise ValueError("jacobian_scale must be finite, positive and shape [10,10]")
        self.register_buffer("moment_scale", moment_scale, persistent=True)
        self.register_buffer("jacobian_scale", jacobian_scale, persistent=True)

    def forward(
        self,
        model: SpinPINO,
        weights: Tensor,
        coefficients: Tensor,
        *,
        target_moments: Optional[Tensor] = None,
        target_jacobian: Optional[Tensor] = None,
        derivative_admitted: bool = False,
    ) -> dict[str, Tensor]:
        if target_jacobian is not None and not derivative_admitted:
            raise ValueError("refusing diagnostic/unadmitted Jacobian supervision")
        r = model.moment_from_weights(weights, coefficients)
        Jc = model.jacobian_from_weights(weights, coefficients)
        r_neg = model.moment_from_weights(weights, -coefficients)
        full = torch.einsum(
            "bn,ni->bi", weights, model.monomials.to(dtype=weights.dtype, device=weights.device)
        )
        B = model.native_B.to(dtype=weights.dtype, device=weights.device)
        Jq = Jc / B.view(1, 1, 10)
        if model.moment_parameterization == "logistic_atom":
            Jq_structure = model.projected_q_gram(weights, coefficients)
        else:
            Jq_structure = Jq

        complement = (r + r_neg - full).square().mean()
        radial = torch.bmm(Jc, coefficients.unsqueeze(-1)).square().mean()
        asymmetry = (Jq_structure - Jq_structure.transpose(1, 2)).square().mean()
        sym = 0.5 * (Jq_structure + Jq_structure.transpose(1, 2))
        eig = torch.linalg.eigvalsh(sym)
        psd = F.relu(-eig).square().mean()
        mass_bounds = (
            F.relu(-r[:, 0]).square().mean()
            + F.relu(r[:, 0] - full[:, 0]).square().mean()
        )
        physics = complement + radial + asymmetry + psd + mass_bounds

        zero = r.new_zeros(())
        if target_moments is not None:
            residual_r = r - target_moments
            if self.moment_scale is not None:
                residual_r = residual_r / self.moment_scale.to(device=r.device, dtype=r.dtype)
            data = residual_r.square().mean()
        else:
            data = zero
        if target_jacobian is not None:
            residual_J = Jc - target_jacobian
            if self.jacobian_scale is not None:
                residual_J = residual_J / self.jacobian_scale.to(device=Jc.device, dtype=Jc.dtype)
            jacobian = residual_J.square().mean()
        else:
            jacobian = zero
        total = (
            self.lambda_data * data
            + self.lambda_physics * physics
            + self.lambda_jacobian * jacobian
        )
        return {
            "total": total,
            "data": data,
            "physics": physics,
            "complement": complement,
            "radial": radial,
            "jacobian_symmetry": asymmetry,
            "jacobian_psd": psd,
            "mass_bounds": mass_bounds,
            "jacobian_supervision": jacobian,
        }
