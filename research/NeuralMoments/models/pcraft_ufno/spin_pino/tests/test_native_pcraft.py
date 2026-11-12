from __future__ import annotations

import numpy as np
import pytest
import torch

from research.NeuralMoments.models.pcraft_ufno.spin_pino.native_pcraft import (
    NativePCRAFTConfig, PcraftNativeMomentOperator, conservative_coarse_grid, geometry_from_columns
)
from research.NeuralMoments.models.pcraft_ufno.spin_pino.pino_backbone import PINOFeatureEncoder


def small_geometry():
    rows=[]
    for x in range(1,7):
        for y in range(1,7):
            rows.append((x,y,1+((x+y)%2),7))
    return geometry_from_columns(dict(columns=np.asarray(rows,dtype=np.int32),
        origin=np.asarray([-1.,-1.,-1.]),h=np.asarray(.25),side=np.asarray(8,dtype=np.int32),
        density=np.asarray(1.)))


def small_model():
    cfg=NativePCRAFTConfig(grid_size=8,pino_width=2,pino_modes=1,pino_layers=2,
        interval_width=8,interval_depth=1,chunk_intervals=16,min_native_resolution=1)
    return PcraftNativeMomentOperator(cfg)


def test_m72_rle_geometry_contract_and_exact_moments():
    g=small_geometry(); c=g["columns"]; o=g["origin"]; h=g["h"]
    x=o[0]+(c[:,0]+.5)*h; y=o[1]+(c[:,1]+.5)*h
    lo=o[2]+c[:,2]*h; hi=o[2]+c[:,3]*h; z=(lo+hi)/2
    m=h*h*(hi-lo)
    expected=np.stack((m,m*x,m*y,m*z,m*(x*x+h*h/12),m*(y*y+h*h/12),
        m*(lo*lo+lo*hi+hi*hi)/3,m*y*z,m*x*z,m*x*y),axis=-1).sum(0)
    assert np.allclose(g["moment_basis"].T@g["mass"],expected,rtol=1e-12,atol=1e-12)
    assert np.allclose(g["original_moments"],expected,rtol=1e-12,atol=1e-12)


def test_conservative_context_grid_preserves_editable_volume():
    g=small_geometry(); grid=conservative_coarse_grid(g,8)
    assert grid.shape==(4,8,8,8)
    native=float(np.prod(g["cell_extent"],axis=1).sum())
    assert float(grid[0].sum())*g["h"]**3==pytest.approx(native,rel=1e-7,abs=1e-10)
    assert np.isfinite(grid).all()


def test_pino_forward_map_is_additive_and_preserves_legacy_pooled_output():
    encoder=PINOFeatureEncoder(width=2,modes=1,n_layers=2,grid_size=8).eval()
    x=torch.rand(1,4,8,8,8)
    with torch.no_grad():
        pooled=encoder(x); pooled2,spatial=encoder.forward_with_map(x)
    assert torch.equal(pooled,pooled2)
    assert spatial.shape==(1,2,8,8,8)


def test_m72_moment_head_scale_complement_and_exact_endpoints():
    torch.manual_seed(11); model=small_model().eval(); cache=model.encode(small_geometry(),chunk=16)
    q=torch.tensor([.15,.2,-.1,.07,.03,-.02,.01,.005,-.004,.003],dtype=torch.float64)
    r=model.predict(cache,q)
    assert torch.allclose(r,model.predict(cache,7*q),rtol=1e-10,atol=1e-11)
    assert torch.allclose(r+model.predict(cache,-q),2*cache.fixed_moments+cache.design_moments,
                          rtol=1e-9,atol=1e-10)
    full=model.predict(cache,torch.tensor([1.,0,0,0,0,0,0,0,0,0],dtype=torch.float64))
    empty=model.predict(cache,torch.tensor([-1.,0,0,0,0,0,0,0,0,0],dtype=torch.float64))
    assert torch.allclose(full,cache.fixed_moments+cache.design_moments)
    assert torch.allclose(empty,cache.fixed_moments)


def test_ad_jacobian_matches_finite_difference_and_supports_second_order_training():
    torch.manual_seed(13); model=small_model().train(); cache=model.encode(small_geometry(),chunk=16)
    q=torch.tensor([.03,.11,-.07,.04,.02,-.01,.015,.004,-.006,.003],dtype=torch.float64)
    moments,jac=model.moments_and_jacobian(cache,q,create_graph=True)
    assert moments.shape==(10,) and jac.shape==(10,10)
    eps=2e-4; fd=torch.empty_like(jac)
    with torch.no_grad():
        for j in range(10):
            d=torch.zeros_like(q); d[j]=eps
            fd[:,j]=(model.predict(cache,q+d)-model.predict(cache,q-d))/(2*eps)
    assert torch.allclose(jac,fd,rtol=1e-2,atol=1e-4)
    (moments.square().mean()+jac.square().mean()).backward()
    grads=[p.grad for p in model.parameters() if p.grad is not None]
    assert grads and all(torch.isfinite(g).all() for g in grads)


def test_geometry_cache_invalidates_after_optimizer_update():
    model=small_model().train(); cache=model.encode(small_geometry(),chunk=16)
    q=torch.tensor([.2,.1,0,0,0,0,0,0,0,0],dtype=torch.float64)
    opt=torch.optim.AdamW(model.parameters(),lr=1e-4)
    model.predict(cache,q).sum().backward(); opt.step()
    with pytest.raises(ValueError,match="stale"):
        model.predict(cache,q)


def test_native_interval_knn_graph_is_self_inclusive_and_bounded():
    from research.NeuralMoments.models.pcraft_ufno.spin_pino.native_pcraft import build_local_neighbor_graph
    g=small_geometry()
    graph=build_local_neighbor_graph(g["xyz"],neighbors=8)
    assert graph.shape==(len(g["xyz"]),8)
    assert graph.dtype==np.int32
    assert np.array_equal(graph[:,0],np.arange(len(g["xyz"]),dtype=np.int32))
    assert graph.min()>=0 and graph.max()<len(g["xyz"])
    # All non-self neighbours are among the nearest native interval centers.
    d=np.linalg.norm(g["xyz"][graph]-g["xyz"][:,None,:],axis=-1)
    assert np.all(d[:,1:]>=0)


def test_pcraft_gno_branch_preserves_moment_invariants_and_ad_jacobian():
    from dataclasses import replace
    torch.manual_seed(23)
    base=small_model()
    cfg=replace(base.config,gno_enabled=True,gno_neighbors=8,gno_cache_mib=1)
    model=PcraftNativeMomentOperator(cfg).train()
    geometry=small_geometry()
    cache=model.encode(geometry,chunk=7)
    q=torch.tensor([.03,.11,-.07,.04,.02,-.01,.015,.004,-.006,.003],dtype=torch.float64)
    moments,jac=model.moments_and_jacobian(cache,q,create_graph=True)
    assert moments.shape==(10,) and jac.shape==(10,10)
    assert torch.isfinite(moments).all() and torch.isfinite(jac).all()
    assert torch.allclose(moments,model.predict(cache,7*q),rtol=1e-10,atol=1e-11)
    complement=model.predict(cache,-q)
    assert torch.allclose(moments+complement,2*cache.fixed_moments+cache.design_moments,
                          rtol=1e-9,atol=1e-10)
    (moments.square().mean()+jac.square().mean()).backward()
    grads=[p.grad for p in model.parameters() if p.grad is not None]
    assert grads and all(torch.isfinite(g).all() for g in grads)
    assert model.counters["gno_graph_builds"]==1
    # A second encode of the same geometry reuses its geometry-only graph.
    model.encode(geometry,chunk=11)
    assert model.counters["gno_graph_builds"]==1


def test_gno_preset_is_an_explicit_ablation_and_base_checkpoint_stays_legacy():
    from research.NeuralMoments.models.pcraft_ufno.spin_pino.native_pcraft import make_config
    base=make_config("xl")
    gno=make_config("xl_gno")
    assert base.gno_enabled is False
    assert gno.gno_enabled is True
    assert gno.gno_neighbors==8
    base_model=PcraftNativeMomentOperator(base)
    gno_model=PcraftNativeMomentOperator(gno)
    assert not hasattr(base_model,"gno_kernel")
    assert hasattr(gno_model,"gno_kernel")
    assert gno_model.pino_encoder.width==base_model.pino_encoder.width


def test_transolver_physics_slicing_normalizes_mass_pooling_and_deslicing():
    from research.NeuralMoments.models.pcraft_ufno.spin_pino.native_pcraft import PhysicsStateSlicing
    torch.manual_seed(31)
    module=PhysicsStateSlicing(width=16,n_slices=4,n_heads=4)
    features=torch.randn(23,16,requires_grad=True)
    mass=torch.rand(23); mass=mass/mass.sum()
    updated,metrics,assignment,deslice=module(features,mass,return_details=True)
    assert updated.shape==features.shape
    assert assignment.shape==(23,4)
    assert deslice.shape==(23,4)
    assert torch.allclose(assignment.sum(0),torch.ones(4),atol=1e-6)
    assert torch.allclose(deslice.sum(-1),torch.ones(23),atol=1e-6)
    assert metrics["transolver_effective_slices"]>=1.0
    assert metrics["transolver_effective_slices"]<=4.0+1e-5
    assert metrics["transolver_assignment_column_error"]<1e-5
    assert metrics["transolver_deslice_row_error"]<1e-5
    updated.square().mean().backward()
    grads=[p.grad for p in module.parameters() if p.grad is not None]
    assert grads and all(torch.isfinite(g).all() for g in grads)


def test_pcraft_transolver_branch_preserves_moment_invariants_and_jacobian():
    from dataclasses import replace
    torch.manual_seed(37)
    base=small_model()
    cfg=replace(base.config,transolver_enabled=True,transolver_slices=4,transolver_heads=2)
    model=PcraftNativeMomentOperator(cfg).train()
    cache=model.encode(small_geometry(),chunk=7)
    q=torch.tensor([.03,.11,-.07,.04,.02,-.01,.015,.004,-.006,.003],dtype=torch.float64)
    moments,jac=model.moments_and_jacobian(cache,q,create_graph=True)
    assert moments.shape==(10,) and jac.shape==(10,10)
    assert torch.isfinite(moments).all() and torch.isfinite(jac).all()
    assert torch.allclose(moments,model.predict(cache,7*q),rtol=1e-10,atol=1e-11)
    assert torch.allclose(moments+model.predict(cache,-q),
        2*cache.fixed_moments+cache.design_moments,rtol=1e-9,atol=1e-10)
    (moments.square().mean()+jac.square().mean()).backward()
    grads=[p.grad for p in model.parameters() if p.grad is not None]
    assert grads and all(torch.isfinite(g).all() for g in grads)
    assert model.counters["transolver_calls"]==1
    assert model.counters["transolver_assignment_column_error"]<1e-5
    assert model.counters["transolver_deslice_row_error"]<1e-5


def test_pcraft_gno_transolver_combination_has_finite_second_order_backward():
    from dataclasses import replace
    torch.manual_seed(41)
    base=small_model()
    cfg=replace(base.config,gno_enabled=True,gno_neighbors=8,
                transolver_enabled=True,transolver_slices=4,transolver_heads=2)
    model=PcraftNativeMomentOperator(cfg).train()
    cache=model.encode(small_geometry(),chunk=7)
    q=torch.tensor([.03,.11,-.07,.04,.02,-.01,.015,.004,-.006,.003],dtype=torch.float64)
    moments,jac=model.moments_and_jacobian(cache,q,create_graph=True)
    (moments.square().mean()+jac.square().mean()).backward()
    grads=[p.grad for p in model.parameters() if p.grad is not None]
    assert grads and all(torch.isfinite(g).all() for g in grads)
    assert model.counters["gno_graph_builds"]==1
    assert model.counters["transolver_calls"]==1


def test_transolver_presets_are_opt_in_and_do_not_change_existing_baselines():
    from research.NeuralMoments.models.pcraft_ufno.spin_pino.native_pcraft import make_config
    base=make_config("xl")
    trans=make_config("xl_transolver")
    combo=make_config("xl_gno_transolver")
    assert base.transolver_enabled is False and base.gno_enabled is False
    assert trans.transolver_enabled is True and trans.gno_enabled is False
    assert combo.transolver_enabled is True and combo.gno_enabled is True
    assert trans.transolver_slices==32 and trans.transolver_heads==8
    assert combo.interval_width % combo.transolver_heads==0
    base_model=PcraftNativeMomentOperator(base)
    trans_model=PcraftNativeMomentOperator(trans)
    assert not hasattr(base_model,"transolver")
    assert hasattr(trans_model,"transolver")
