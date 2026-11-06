import json
from fractions import Fraction as Q
from pathlib import Path
import numpy as np
import pytest
from .m35_crossing_newton import TIP, keys, read_ten, ref_state, shift, topology
from .m35_newton_summary import summarize

HERE=Path(__file__).with_name("m35")


def test_translation_is_exact_for_point_mass():
    x,y,z=Q(1,7),Q(-3,5),Q(4,9)
    def monomial(x,y,z):
        return [1,x,y,z,x*x,y*y,z*z,y*z,x*z,x*y]
    assert shift(monomial(x,y,z))==monomial(x,y,z-TIP)


def test_feature_fingerprint_renumber_and_reversal():
    vv=[[0,0,0],[1,0,0],[0,1,0]]
    def signature(vertices,face):
        return topology(dict(vertices=vertices,faces=[face],birth_faces=[4]),keys(vv),{},[])
    assert signature(vv,[0,1,2])==signature(vv[::-1],[0,2,1])
    assert signature(vv,[0,1,2])!=signature(vv,[0,2,1])


def test_exact_signed_ten_input_enclosure(tmp_path):
    path=tmp_path/"a.ten";path.write_text("-0x1/0x3\n"*10)
    approx=read_ten(path)
    assert all(0<=-Q(1,3)-x<Q(1,2**190) for x in approx)
    with pytest.raises(ValueError,match="unexpected original"):
        summarize({},path)


def test_captured_ellipsoid_steps_and_fixed_axis_gate():
    rec=json.loads((HERE/"ellipsoid_states.json").read_text())
    rows=[r for r in rec["states"] if r["outcome"]=="accepted"]
    assert [r["index"] for r in rows]==[1,3,6]
    assert [float(r["step"]) for r in rows]==[0,.25,.125]
    assert rows[0]["raw_residual"]>rows[1]["raw_residual"]>rows[2]["raw_residual"]
    assert rows[2]["axial_inertia_fraction"]==pytest.approx(.5577993941)
    assert not any(r["passes_axial_inertia_90"] for r in rows)
    assert rec["terminal"]["status"]=="RESOURCE_LIMIT_REACHED"


def test_exact_design_grid_contract():
    rec=json.loads((HERE/"design_8_cubed.json").read_text())
    vox=np.asarray(rec["occupancy"])
    assert vox.shape==(8,8,8) and vox.sum()==rec["occupied"]==16
    assert set(vox.flat)=={0,1} and rec["cube_center_in_tip"]==[0,0,1]


def test_stock_start_and_independent_event_state():
    trace=json.loads((HERE/"stock_seed0_trace.json").read_text())
    states=json.loads((HERE/"stock_seed0_states.json").read_text())
    assert trace["start_index"]==0 and states["terminal"]["calls"]=="3"
    assert trace["evaluations"][0]["boolean_execution"]=="replayed_hash_bound"
    assert all(x["boolean_execution"]=="fresh_requested" for x in trace["evaluations"][1:])
    for row,state in zip(trace["evaluations"],states["states"],strict=True):
        recalculated=ref_state(row["q"],row["retained_ten_tip"])
        assert state["outcome"]=="accepted"
        assert recalculated["raw_residual"]==pytest.approx(state["raw_residual"])
        assert recalculated["physical_relative"]==pytest.approx(state["physical_relative"])
    assert states["states"][-1]["axial_inertia_fraction"]==pytest.approx(.6685899335)
