"""Split-conformal calibration utilities for M80 ensemble uncertainty.

This is a lightweight calibrated-ensemble baseline, not an implementation of
upstream UQNO. Calibration uses held-out validation queries with exact teacher
errors. It does not replace event classification or derivative certification.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import numpy as np


def ensemble_moment_uncertainty(predictions: Any, moment_scales: Any) -> np.ndarray:
    """Return per-query RMS ensemble spread in normalized ten-moment space."""
    values=np.asarray(predictions,dtype=np.float64)
    scales=np.asarray(moment_scales,dtype=np.float64)
    if values.ndim!=3 or values.shape[-1]!=10 or values.shape[0]<2:
        raise ValueError("moment predictions must have shape [ensemble>=2,queries,10]")
    if scales.shape!=(10,) or not np.isfinite(scales).all() or (scales<=0).any():
        raise ValueError("moment_scales must be finite positive [10]")
    if not np.isfinite(values).all():
        raise ValueError("moment predictions must be finite")
    normalized=values/scales[None,None,:]
    return np.sqrt(np.mean(np.var(normalized,axis=0,ddof=1),axis=-1))


def ensemble_jacobian_uncertainty(predictions: Any,moment_scales: Any,coefficient_scales: Any) -> np.ndarray:
    """Return per-query RMS ensemble spread in normalized 10x10 Jacobian space."""
    values=np.asarray(predictions,dtype=np.float64)
    ms=np.asarray(moment_scales,dtype=np.float64)
    cs=np.asarray(coefficient_scales,dtype=np.float64)
    if values.ndim!=4 or values.shape[-2:]!=(10,10) or values.shape[0]<2:
        raise ValueError("Jacobian predictions must have shape [ensemble>=2,queries,10,10]")
    if ms.shape!=(10,) or cs.shape!=(10,) or not np.isfinite(ms).all() or not np.isfinite(cs).all() or (ms<=0).any() or (cs<=0).any():
        raise ValueError("moment/coefficient scales must be finite positive [10]")
    if not np.isfinite(values).all():
        raise ValueError("Jacobian predictions must be finite")
    scale=ms[:,None]*cs[None,:]
    normalized=values/scale[None,None,:,:]
    return np.sqrt(np.mean(np.var(normalized,axis=0,ddof=1),axis=(-2,-1)))


def conformal_ratio_scale(uncertainty: Any,observed_error: Any,alpha: float=.1,eps: float=1e-12) -> float:
    """Finite-sample split-conformal multiplier for error <= multiplier*spread."""
    u=np.asarray(uncertainty,dtype=np.float64).reshape(-1)
    e=np.asarray(observed_error,dtype=np.float64).reshape(-1)
    if len(u)!=len(e) or len(u)<1:
        raise ValueError("uncertainty and observed_error must be nonempty equal-length arrays")
    if not np.isfinite(u).all() or not np.isfinite(e).all() or (u<0).any() or (e<0).any():
        raise ValueError("uncertainty/error values must be finite and nonnegative")
    if not (0<alpha<1):
        raise ValueError("alpha must lie in (0,1)")
    ratios=e/np.maximum(u,eps)
    ordered=np.sort(ratios)
    rank=min(len(ordered)-1,max(0,math.ceil((len(ordered)+1)*(1-alpha))-1))
    return float(ordered[rank])


@dataclass(frozen=True)
class CalibratedUncertaintyGate:
    moment_scale: float
    jacobian_scale: float
    moment_tolerance: float
    jacobian_tolerance: float
    alpha: float
    calibration_count: int

    def __post_init__(self):
        values=(self.moment_scale,self.jacobian_scale,self.moment_tolerance,self.jacobian_tolerance)
        if any(not np.isfinite(v) or v<0 for v in values):
            raise ValueError("gate scales and tolerances must be finite nonnegative")
        if not (0<self.alpha<1) or self.calibration_count<1:
            raise ValueError("invalid calibration alpha/count")

    def admit(self,*,moment_uncertainty:float,jacobian_uncertainty:float,
              event_regular:bool,derivative_valid:bool) -> tuple[bool,str]:
        if not event_regular:
            return False,"event_not_regular"
        if not derivative_valid:
            return False,"teacher_derivative_invalid"
        if not np.isfinite(moment_uncertainty) or not np.isfinite(jacobian_uncertainty):
            return False,"nonfinite_uncertainty"
        if moment_uncertainty<0 or jacobian_uncertainty<0:
            return False,"negative_uncertainty"
        moment_upper=self.moment_scale*moment_uncertainty
        jacobian_upper=self.jacobian_scale*jacobian_uncertainty
        if moment_upper>self.moment_tolerance:
            return False,"moment_risk_bound_exceeded"
        if jacobian_upper>self.jacobian_tolerance:
            return False,"jacobian_risk_bound_exceeded"
        return True,"calibrated_risk_bounds_passed"


def fit_uncertainty_gate(
    moment_uncertainty:Any,moment_error:Any,
    jacobian_uncertainty:Any,jacobian_error:Any,
    *,alpha:float=.1,moment_tolerance:float,jacobian_tolerance:float
) -> CalibratedUncertaintyGate:
    """Fit separate moment/Jacobian conformal multipliers on validation only."""
    mu=np.asarray(moment_uncertainty,dtype=np.float64).reshape(-1)
    me=np.asarray(moment_error,dtype=np.float64).reshape(-1)
    ju=np.asarray(jacobian_uncertainty,dtype=np.float64).reshape(-1)
    je=np.asarray(jacobian_error,dtype=np.float64).reshape(-1)
    if len(mu)!=len(me) or len(ju)!=len(je) or len(mu)!=len(ju):
        raise ValueError("moment and Jacobian calibration arrays must have matching lengths")
    return CalibratedUncertaintyGate(
        moment_scale=conformal_ratio_scale(mu,me,alpha),
        jacobian_scale=conformal_ratio_scale(ju,je,alpha),
        moment_tolerance=float(moment_tolerance),
        jacobian_tolerance=float(jacobian_tolerance),
        alpha=float(alpha),
        calibration_count=len(mu),
    )
