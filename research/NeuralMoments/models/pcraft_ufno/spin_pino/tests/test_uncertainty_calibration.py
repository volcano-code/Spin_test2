import numpy as np
import pytest

from research.NeuralMoments.models.pcraft_ufno.spin_pino.uncertainty_calibration import (
    CalibratedUncertaintyGate,
    conformal_ratio_scale,
    ensemble_jacobian_uncertainty,
    ensemble_moment_uncertainty,
    fit_uncertainty_gate,
)


def test_ensemble_uncertainty_uses_normalized_ten_moment_and_jacobian_scales():
    rng=np.random.default_rng(3)
    moments=rng.normal(size=(4,7,10))
    jac=rng.normal(size=(4,7,10,10))
    ms=np.linspace(.5,2.,10); cs=np.linspace(.7,1.4,10)
    mu=ensemble_moment_uncertainty(moments,ms)
    ju=ensemble_jacobian_uncertainty(jac,ms,cs)
    assert mu.shape==(7,) and np.all(mu>=0)
    assert ju.shape==(7,) and np.all(ju>=0)
    with pytest.raises(ValueError):
        ensemble_moment_uncertainty(moments[:1],ms)


def test_conformal_ratio_scale_uses_finite_sample_upper_order_statistic():
    u=np.asarray([.1,.2,.3,.4,.5])
    e=np.asarray([.1,.1,.3,.2,.5])
    scale=conformal_ratio_scale(u,e,alpha=.2)
    assert scale==pytest.approx(1.0)
    with pytest.raises(ValueError):
        conformal_ratio_scale([.1],[.1],alpha=1.0)


def test_calibrated_gate_fails_closed_for_events_invalid_derivatives_and_risk():
    gate=CalibratedUncertaintyGate(2.,3.,.1,.2,.1,50)
    assert gate.admit(moment_uncertainty=.01,jacobian_uncertainty=.02,
                      event_regular=True,derivative_valid=True)==(True,"calibrated_risk_bounds_passed")
    assert gate.admit(moment_uncertainty=.01,jacobian_uncertainty=.02,
                      event_regular=False,derivative_valid=True)==(False,"event_not_regular")
    assert gate.admit(moment_uncertainty=.01,jacobian_uncertainty=.02,
                      event_regular=True,derivative_valid=False)==(False,"teacher_derivative_invalid")
    assert gate.admit(moment_uncertainty=.2,jacobian_uncertainty=.02,
                      event_regular=True,derivative_valid=True)==(False,"moment_risk_bound_exceeded")
    assert gate.admit(moment_uncertainty=.01,jacobian_uncertainty=.2,
                      event_regular=True,derivative_valid=True)==(False,"jacobian_risk_bound_exceeded")


def test_fit_uncertainty_gate_uses_matching_validation_arrays():
    gate=fit_uncertainty_gate(
        [1.,2.,3.],[.5,1.,1.5],
        [1.,2.,3.],[.2,.4,.6],
        alpha=.2,moment_tolerance=.1,jacobian_tolerance=.05
    )
    assert gate.calibration_count==3
    assert gate.moment_scale==pytest.approx(.5)
    assert gate.jacobian_scale==pytest.approx(.2)
    with pytest.raises(ValueError):
        fit_uncertainty_gate([1,2],[1],[1,2],[1,2],moment_tolerance=1,jacobian_tolerance=1)
