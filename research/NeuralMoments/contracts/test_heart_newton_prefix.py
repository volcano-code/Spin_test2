import numpy as np
import pytest

from .heart_newton_prefix import deterministic_starts, physics, state, lagrangian_gradient, near


def test_initialization_spinning_top_rng():
    a=list(deterministic_starts(0,8))
    assert len(a)==8 and all(v[6]<0 and np.isclose(np.linalg.norm(v),1) for v in a)
    np.testing.assert_allclose(a[0][:3],[-0.21979942196718708, 0.6837584525662371, -0.37486538740581204],rtol=0,atol=1e-15)


def test_original_objective_zero_height_weight_and_projection():
    r=np.array([2.,0,0,1,3,4,5,0,0,0])
    x=state([1,0,0,0,0,0,0,0,0,0],r)
    assert x['objective']==((4+4.5)**2+(3+4.5)**2)/7**2
    assert x['constraints']==[0,0,0,0] and x['physical_relative_residual']==0
    assert np.linalg.norm(x['optimality'])==x['raw_equation_residual']


def test_closed_gradient_matches_complex_step_and_detects_tampering():
    r=np.array([2.,.25,-.1,.6,.9,.7,1.,.1,-.2,.03])
    lam=np.array([.1,-.3,.7,1.2])
    grad=lagrangian_gradient(r,lam)
    expected=np.array([physics(r+1e-24j*e).imag@np.r_[1.,-lam]/1e-24 for e in np.eye(10)])
    near(grad,expected,'gradient',1e-13)
    with pytest.raises(ValueError,match='gradient mismatch'):
        near(-grad,expected,'gradient')


def test_rejections_use_last_accepted_within_same_restart():
    from .heart_newton_prefix import check_rejections
    rows=[dict(index=1,restart=0,iteration=0,outcome='accepted',reason='initial_evaluation_succeeded',
               topology_hash='a',raw_equation_residual=7.),
          dict(index=2,restart=0,iteration=1,outcome='rejected',reason='topology_transition',
               topology_hash='b',raw_equation_residual=10.),
          dict(index=3,restart=0,iteration=1,outcome='rejected',reason='no_residual_decrease',
               topology_hash='a',raw_equation_residual=9.),
          dict(index=4,restart=1,iteration=0,outcome='rejected',reason='initial_multiplier_nonpositive',
               multiplier_mu=-1.)]
    assert check_rejections(rows,{})==[2,3,4]
    rows[2]['raw_equation_residual']=6.
    with pytest.raises(ValueError,match='residual/topology'):
        check_rejections(rows,{})


def test_m30_never_promotes_budget_stop_to_terminal():
    from .heart_newton_prefix import trajectory_report
    q=[1.]+[0.]*9
    one=dict(index=1,restart=0,raw_equation_residual=2.,
             material_sha256='207f19d2c9c71e240f87650565625330466aec3e75c4ec63e326f1d7f1cfaf80')
    two=dict(index=2,restart=0,raw_equation_residual=1.,
             material_sha256='d0ea04ff769c8a390c7eaf2f11670ad1ea4c987dc7c0d75221964d4799d31f39')
    rows=[one,two,dict(index=3)]; result={'status':'BUDGET_EXHAUSTED'}
    r=trajectory_report(result,[dict(quadric_coefficients=q)]*3,{},[(1,2)],[{}],[1,2],rows)
    assert r['last_observed_evaluation']==3 and r['last_accepted_evaluation']==2
    assert not r['terminal_equation_and_physical_tolerances_verified']
    assert r['m29_full_material_digest_gate']=='PASS'
    two['material_sha256']='changed'
    drift=trajectory_report(result,[{}]*3,{},[(1,2)],[{}],[1,2],rows)
    assert drift['m29_full_material_digest_gate']=='FAIL' and not drift['m29_first_step_reproduced']
    assert drift['accepted_step_count']==1
    result['status']='NATIVE_TERMINAL'
    with pytest.raises(ValueError,match='terminal summary'):
        trajectory_report(result,[{}]*3,{},[(1,2)],[{}],[1,2],rows)


def test_trajectory_reintegrates_intermediate_accepted_meshes():
    from .heart_newton_prefix import moment_reintegration_indices
    accepted=[1,2,4,7,9]
    steps=[(1,2),(2,4),(4,7),(7,9)]
    assert moment_reintegration_indices(True,steps,accepted)==set(accepted)
    assert moment_reintegration_indices(False,[(1,2)],[1,2])=={1,2}
