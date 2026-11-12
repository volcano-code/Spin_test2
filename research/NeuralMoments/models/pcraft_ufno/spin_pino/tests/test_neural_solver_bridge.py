import numpy as np
import torch

from research.NeuralMoments.models.pcraft_ufno.spin_pino.native_pcraft import (
    NativePCRAFTConfig,
    PcraftNativeMomentOperator,
    geometry_from_columns,
)
from research.NeuralMoments.models.pcraft_ufno.spin_pino.neural_solver_bridge import (
    M80ExternalMomentEvaluator,
)


def _geometry():
    rows=[]
    for x in range(1,7):
        for y in range(1,7):
            rows.append((x,y,1+(x+y)%2,7))
    return geometry_from_columns(dict(
        columns=np.asarray(rows,dtype=np.int32),
        origin=np.asarray([-1.,-1.,-1.]),
        h=np.asarray(.25),
        side=np.asarray(8,dtype=np.int32),
        density=np.asarray(1.),
    ))


def _model_and_cache(convex=False):
    cfg=NativePCRAFTConfig(
        grid_size=8,pino_width=2,pino_modes=1,pino_layers=2,
        interval_width=8,interval_depth=1,chunk_intervals=32,
        min_native_resolution=1,convex_potential_enabled=convex,
        convex_potential_components=3,
    )
    model=PcraftNativeMomentOperator(cfg).eval()
    cache=model.encode(_geometry(),chunk=32).detached()
    return model,cache


def _exact_map(geometry_hash,topology_hash):
    return dict(
        raw_moments=np.arange(10,dtype=np.float64),
        variable_raw_moments=np.arange(10,dtype=np.float64)*.5,
        moment_jacobian=np.eye(10,dtype=np.float64),
        derivative_defined=True,
        geometry_hash=geometry_hash,
        topology_hash=topology_hash,
        evaluation_map_identity="CHERCHI_EXACT_MAP_V1",
        boolean_backend_identity="CHERCHI_BOOLEAN_V1",
        derivative_identity="CHERCHI_DERIVATIVE_V1",
        residual_evaluation_seconds=.01,
        derivative_evaluation_seconds=.02,
    )


def _adapter(model,cache,*,admission,exact=None,C=None,M=None):
    geometry_hash="source-mesh-sha256"
    topology_hash="material-topology-sha256"
    return M80ExternalMomentEvaluator(
        model,cache,
        geometry_hash=geometry_hash,
        cache_geometry_hash=cache.geometry_sha256,
        topology_hash=topology_hash,
        coefficient_to_model=np.eye(10) if C is None else C,
        moment_to_target=np.eye(10) if M is None else M,
        exact_evaluator=exact or (lambda q:_exact_map(geometry_hash,topology_hash)),
        admission=admission,
        hybrid_identity="M80_HYBRID_TEST_V1",
        model_identity="m80-checkpoint-sha256",
        exact_identity="cherchi-source-sha256",
    )


def test_neural_evaluator_is_fail_closed_without_admission():
    model,cache=_model_and_cache()
    adapter=_adapter(model,cache,admission=None)
    result=adapter(np.asarray([.03,.11,-.07,.04,.02,-.01,.015,.004,-.006,.003]))
    assert result["evaluation_map_identity"]=="M80_HYBRID_TEST_V1"
    assert result["boolean_backend_identity"]=="M80_HYBRID_NEURAL_EXACT"
    assert result["derivative_identity"]=="M80_AD_WITH_EXACT_FALLBACK"
    assert result["derivative_diagnostic"].startswith("exact_fallback;reason=no_admission")
    assert adapter.stats.exact_fallback_calls==1
    assert adapter.stats.neural_calls==0


def test_neural_evaluator_applies_coefficient_and_moment_chain_rule():
    model,cache=_model_and_cache()
    M=np.diag(np.linspace(1.3,.9,10))
    C=M.T
    adapter=_adapter(model,cache,admission=lambda qn,qm,pred:(True,"test_gate_passed"),C=C,M=M)
    q=np.asarray([.03,.11,-.07,.04,.02,-.01,.015,.004,-.006,.003])
    result=adapter(q)
    with torch.enable_grad():
        q_model=torch.as_tensor(C@q,dtype=torch.float64)
        moments,jac=model.moments_and_jacobian(cache,q_model,create_graph=False)
    expected_m=M@moments.detach().cpu().numpy()
    expected_j=M@jac.detach().cpu().numpy()@C
    assert np.allclose(result["raw_moments"],expected_m,rtol=1e-8,atol=1e-9)
    assert np.allclose(result["moment_jacobian"],expected_j,rtol=1e-7,atol=1e-8)
    assert result["derivative_defined"] is True
    assert "neural_admitted" in result["derivative_diagnostic"]
    assert adapter.stats.neural_calls==1
    assert adapter.stats.exact_fallback_calls==0


def test_neural_evaluator_falls_back_when_admission_rejects():
    model,cache=_model_and_cache()
    adapter=_adapter(model,cache,admission=lambda qn,qm,pred:(False,"near_event_boundary"))
    result=adapter(np.asarray([.03,.11,-.07,.04,.02,-.01,.015,.004,-.006,.003]))
    assert np.allclose(result["raw_moments"],np.arange(10,dtype=np.float64))
    assert "admission_rejected:near_event_boundary" in result["derivative_diagnostic"]
    assert adapter.stats.exact_fallback_calls==1
    assert adapter.stats.neural_calls==0


def test_neural_evaluator_falls_back_on_stale_cache():
    model,cache=_model_and_cache()
    with torch.no_grad():
        next(model.parameters()).add_(1e-4)
    adapter=_adapter(model,cache,admission=lambda qn,qm,pred:True)
    result=adapter(np.asarray([.03,.11,-.07,.04,.02,-.01,.015,.004,-.006,.003]))
    assert result["derivative_diagnostic"].startswith("exact_fallback;reason=M72_cache_model_version_mismatch")
    assert adapter.stats.rejected_stale_cache==1
    assert adapter.stats.exact_fallback_calls==1


def test_convex_neural_evaluator_passes_psd_gate_when_admitted():
    model,cache=_model_and_cache(convex=True)
    adapter=_adapter(model,cache,admission=lambda qn,qm,pred:True)
    result=adapter(np.asarray([.03,.11,-.07,.04,.02,-.01,.015,.004,-.006,.003]))
    assert result["derivative_defined"] is True
    assert adapter.stats.neural_calls==1
    assert adapter.stats.rejected_convex_jacobian==0


def test_target_frame_moment_transform_matches_pointwise_polynomial_basis():
    from research.NeuralMoments.models.pcraft_ufno.spin_pino.neural_solver_bridge import (
        target_frame_moment_transforms
    )
    theta=.37
    axes=np.asarray([
        [np.cos(theta),-np.sin(theta),0.],
        [np.sin(theta), np.cos(theta),0.],
        [0.,0.,1.]
    ],dtype=np.float64)
    origin=np.asarray([.3,-.2,.5],dtype=np.float64)
    coefficient_to_model,moment_to_target=target_frame_moment_transforms(origin,axes)
    rng=np.random.default_rng(123)
    q_target=rng.normal(size=10)
    points=rng.normal(size=(32,3))
    def basis(x):
        a,b,c=x
        return np.asarray([1.,a,b,c,a*a,b*b,c*c,b*c,a*c,a*b])
    for x in points:
        target=axes@(x-origin)
        assert np.allclose(basis(target),moment_to_target@basis(x),rtol=1e-12,atol=1e-12)
        q_model=coefficient_to_model@q_target
        assert np.dot(q_target,basis(target))==pytest.approx(np.dot(q_model,basis(x)),abs=1e-12)


def test_target_frame_transform_rejects_nonorthonormal_axes():
    from research.NeuralMoments.models.pcraft_ufno.spin_pino.neural_solver_bridge import (
        target_frame_moment_transforms
    )
    with pytest.raises(ValueError,match="orthonormal"):
        target_frame_moment_transforms(np.zeros(3),np.asarray([[1.,0,0],[0,2.,0],[0,0,1.]]))


def test_neural_evaluator_rejects_nondual_frame_transform_pair():
    model,cache=_model_and_cache()
    with pytest.raises(ValueError,match="must equal moment_to_target.T"):
        _adapter(model,cache,admission=lambda qn,qm,pred:True,
                 C=np.diag(np.linspace(.8,1.2,10)),
                 M=np.diag(np.linspace(1.3,.9,10)))
