"""Fail-closed adapter from an M80 moment model to the native Faster callback contract.

The native solver callback consumes target-frame raw moments and a matching
native-coefficient Jacobian. This adapter deliberately requires explicit
coefficient/moment transforms, source/topology identities, an exact evaluator
fallback and an admission callback. With no admission callback, every query
falls back to the exact evaluator.
"""
from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Any, Callable, Mapping

import numpy as np
import torch

from .native_pcraft import EncodedNativeGeometry, PcraftNativeMomentOperator


ExactEvaluator = Callable[[np.ndarray], Mapping[str, Any]]
Admission = Callable[[np.ndarray, np.ndarray, Mapping[str, Any]], bool | tuple[bool, str]]


@dataclass
class NeuralEvaluatorStats:
    total_calls: int = 0
    neural_calls: int = 0
    exact_fallback_calls: int = 0
    rejected_no_admission: int = 0
    rejected_training_mode: int = 0
    rejected_stale_cache: int = 0
    rejected_nonfinite: int = 0
    rejected_envelope: int = 0
    rejected_convex_jacobian: int = 0
    neural_seconds: float = 0.0
    exact_seconds: float = 0.0


def _matrix10(value: Any, name: str) -> np.ndarray:
    matrix = np.asarray(value, dtype=np.float64)
    if matrix.shape != (10, 10) or not np.isfinite(matrix).all():
        raise ValueError(f"{name} must be a finite 10x10 matrix")
    if np.linalg.matrix_rank(matrix) != 10:
        raise ValueError(f"{name} must be invertible")
    return matrix


def _identity(value: str, name: str) -> str:
    if not isinstance(value, str) or not value or ";" in value:
        raise ValueError(f"{name} must be nonempty and must not contain ';'")
    return value


class M80ExternalMomentEvaluator:
    """Callable compatible with make_it_stand.solve_spin_it_faster... evaluator.

    Transform convention:
      q_model = coefficient_to_model @ q_native
      r_target = moment_to_target @ r_model
      J_target = moment_to_target @ J_model @ coefficient_to_model

    geometry_hash is the original solver-request geometry identity;
    cache_geometry_hash is the M72/RLE geometry identity used by the network.
    They are intentionally separate.
    """

    def __init__(
        self,
        model: PcraftNativeMomentOperator,
        cache: EncodedNativeGeometry,
        *,
        geometry_hash: str,
        cache_geometry_hash: str,
        topology_hash: str,
        coefficient_to_model: Any,
        moment_to_target: Any,
        exact_evaluator: ExactEvaluator,
        admission: Admission | None = None,
        hybrid_identity: str,
        model_identity: str,
        exact_identity: str,
        convex_symmetry_tolerance: float = 1e-7,
        convex_min_eigenvalue_tolerance: float = 1e-8,
    ):
        if not callable(exact_evaluator):
            raise TypeError("exact_evaluator must be callable")
        self.model = model
        self.cache = cache
        self.geometry_hash = _identity(geometry_hash, "geometry_hash")
        self.cache_geometry_hash = _identity(cache_geometry_hash, "cache_geometry_hash")
        self.topology_hash = _identity(topology_hash, "topology_hash")
        self.coefficient_to_model = _matrix10(coefficient_to_model, "coefficient_to_model")
        self.moment_to_target = _matrix10(moment_to_target, "moment_to_target")
        self.exact_evaluator = exact_evaluator
        self.admission = admission
        self.hybrid_identity = _identity(hybrid_identity, "hybrid_identity")
        self.model_identity = _identity(model_identity, "model_identity")
        self.exact_identity = _identity(exact_identity, "exact_identity")
        self.convex_symmetry_tolerance = float(convex_symmetry_tolerance)
        self.convex_min_eigenvalue_tolerance = float(convex_min_eigenvalue_tolerance)
        if self.convex_symmetry_tolerance < 0 or self.convex_min_eigenvalue_tolerance < 0:
            raise ValueError("convex Jacobian tolerances must be nonnegative")
        self.stats = NeuralEvaluatorStats()

    def _exact(self, q_native: np.ndarray, reason: str) -> dict[str, Any]:
        start = perf_counter()
        value = self.exact_evaluator(q_native.copy())
        elapsed = perf_counter() - start
        if not isinstance(value, Mapping):
            raise TypeError("exact evaluator must return a mapping")
        required = (
            "raw_moments", "moment_jacobian", "derivative_defined",
            "geometry_hash", "topology_hash", "evaluation_map_identity",
            "boolean_backend_identity", "derivative_identity",
        )
        missing = [name for name in required if name not in value]
        if missing:
            raise ValueError(f"exact evaluator omitted required fields: {missing}")
        raw = np.asarray(value["raw_moments"], dtype=np.float64)
        jac = np.asarray(value["moment_jacobian"], dtype=np.float64)
        if raw.shape != (10,) or jac.shape != (10, 10):
            raise ValueError("exact evaluator must return raw_moments[10] and moment_jacobian[10,10]")
        if not np.isfinite(raw).all() or not np.isfinite(jac).all():
            raise FloatingPointError("exact evaluator returned nonfinite moments/Jacobian")
        if value["geometry_hash"] != self.geometry_hash or value["topology_hash"] != self.topology_hash:
            raise ValueError("exact fallback geometry/topology identity mismatch")
        out = dict(value)
        # A single hybrid map identity is used across neural and exact calls so
        # the native Newton loop sees one declared policy. The exact source
        # identity is retained in the per-call diagnostic.
        out["evaluation_map_identity"] = self.hybrid_identity
        out["boolean_backend_identity"] = "M80_HYBRID_NEURAL_EXACT"
        out["derivative_identity"] = "M80_AD_WITH_EXACT_FALLBACK"
        out["derivative_diagnostic"] = (
            f"exact_fallback;reason={reason};exact_identity={value['evaluation_map_identity']};"
            f"exact_backend={value['boolean_backend_identity']};"
            f"exact_derivative={value['derivative_identity']}"
        )
        out["residual_evaluation_seconds"] = float(value.get("residual_evaluation_seconds", elapsed))
        out["derivative_evaluation_seconds"] = float(value.get("derivative_evaluation_seconds", 0.0))
        self.stats.exact_fallback_calls += 1
        self.stats.exact_seconds += elapsed
        return out

    def __call__(self, q_native: Any) -> dict[str, Any]:
        self.stats.total_calls += 1
        qn = np.asarray(q_native, dtype=np.float64)
        if qn.shape != (10,) or not np.isfinite(qn).all():
            raise ValueError("native quadric coefficients must be a finite 10-vector")

        if self.admission is None:
            self.stats.rejected_no_admission += 1
            return self._exact(qn, "no_admission_callback_fail_closed")
        if self.model.training:
            self.stats.rejected_training_mode += 1
            return self._exact(qn, "model_in_training_mode")
        if self.cache.geometry_sha256 != self.cache_geometry_hash:
            self.stats.rejected_stale_cache += 1
            return self._exact(qn, "M72_cache_geometry_hash_mismatch")
        if self.cache.owner != id(self.model) or self.cache.versions != self.model._versions():
            self.stats.rejected_stale_cache += 1
            return self._exact(qn, "M72_cache_model_version_mismatch")

        q_model = self.coefficient_to_model @ qn
        device = self.cache.latent.device
        q_tensor = torch.as_tensor(q_model, dtype=torch.float64, device=device)
        start = perf_counter()
        try:
            with torch.enable_grad():
                moments_model, jac_model = self.model.moments_and_jacobian(
                    self.cache, q_tensor, create_graph=False
                )
            moments_model = moments_model.detach().double()
            jac_model = jac_model.detach().double()
            fixed_model = self.cache.fixed_moments.detach().double()
            variable_model = moments_model - fixed_model
            moment_transform = torch.as_tensor(
                self.moment_to_target, dtype=torch.float64, device=device
            )
            coefficient_transform = torch.as_tensor(
                self.coefficient_to_model, dtype=torch.float64, device=device
            )
            moments_target = moment_transform @ moments_model
            variable_target = moment_transform @ variable_model
            jac_target = moment_transform @ jac_model @ coefficient_transform
            if (not torch.isfinite(moments_target).all()
                    or not torch.isfinite(variable_target).all()
                    or not torch.isfinite(jac_target).all()):
                self.stats.rejected_nonfinite += 1
                return self._exact(qn, "nonfinite_neural_response")
            # Preserve the current M72 half-design ± half-absolute envelope.
            deviation = variable_model - .5*self.cache.design_moments.detach().double()
            if (deviation.abs() > .5*self.cache.absolute_moments.detach().double()+1e-8).any():
                self.stats.rejected_envelope += 1
                return self._exact(qn, "neural_response_outside_absolute_moment_envelope")
            if self.model.config.convex_potential_enabled:
                symmetry = (jac_model-jac_model.T).abs().max()
                minimum_eigenvalue = torch.linalg.eigvalsh(.5*(jac_model+jac_model.T)).min()
                if (float(symmetry) > self.convex_symmetry_tolerance
                        or float(minimum_eigenvalue) < -self.convex_min_eigenvalue_tolerance):
                    self.stats.rejected_convex_jacobian += 1
                    return self._exact(qn, "convex_jacobian_symmetry_or_psd_gate")
            elapsed = perf_counter()-start
            prediction = {
                "raw_moments": moments_target.detach().cpu().numpy(),
                "variable_raw_moments": variable_target.detach().cpu().numpy(),
                "moment_jacobian": jac_target.detach().cpu().numpy(),
                "derivative_defined": True,
                "geometry_hash": self.geometry_hash,
                "topology_hash": self.topology_hash,
                "model_identity": self.model_identity,
                "exact_identity": self.exact_identity,
                "coefficient_to_model": self.coefficient_to_model,
                "moment_to_target": self.moment_to_target,
            }
            decision = self.admission(qn.copy(), q_model.copy(), prediction)
            if isinstance(decision, tuple):
                accepted, reason = bool(decision[0]), str(decision[1])
            else:
                accepted, reason = bool(decision), "admission_callback"
            if not accepted:
                return self._exact(qn, f"admission_rejected:{reason}")
            self.stats.neural_calls += 1
            self.stats.neural_seconds += elapsed
            return {
                "raw_moments": prediction["raw_moments"],
                "variable_raw_moments": prediction["variable_raw_moments"],
                "moment_jacobian": prediction["moment_jacobian"],
                "derivative_defined": True,
                "derivative_diagnostic": f"neural_admitted;reason={reason};model={self.model_identity}",
                "geometry_hash": self.geometry_hash,
                "topology_hash": self.topology_hash,
                "evaluation_map_identity": self.hybrid_identity,
                "boolean_backend_identity": "M80_HYBRID_NEURAL_EXACT",
                "derivative_identity": "M80_AD_WITH_EXACT_FALLBACK",
                # The model currently computes moments and AD Jacobian jointly;
                # put the combined cost in residual time to avoid double counting.
                "residual_evaluation_seconds": float(elapsed),
                "derivative_evaluation_seconds": 0.0,
            }
        except (FloatingPointError, RuntimeError, ValueError) as error:
            self.stats.rejected_nonfinite += 1
            return self._exact(qn, f"neural_exception:{type(error).__name__}:{error}")
