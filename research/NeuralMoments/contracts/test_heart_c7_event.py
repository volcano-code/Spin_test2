from collections import Counter
from fractions import Fraction as F
import pytest
from .heart_c7_event import edge, signed_six_volume, side_check


def test_exact_event_orientation_and_contact():
    a,b,c=(0,0,0),(1,0,0),(0,1,0)
    for z in (-1,0,1):
        assert signed_six_volume(a,b,c,(0,0,F(z,5))) == F(z,5)


def test_edge_requires_both_incident_source_faces():
    pts=[(0,0,0),(1,0,0),(0,1,0),(0,0,1)]
    faces=[[0,2,1],[1,3,0],[0,3,2]]
    assert edge(pts,faces,(0,1),[[0,1,2],[0,1,3]]) == tuple(pts[:2])
    with pytest.raises(ValueError):
        edge(pts,faces[:-2],(0,1),[[0,1,2],[0,1,3]])


@pytest.mark.parametrize("anchor",[0,2])
def test_one_sided_fits_actual_endpoint_not_proxy_root(anchor):
    mesh=({1},Counter({(1,2,3):1}),'same')
    points=[dict(info={'run_id':str(t)},x=F(t),moments=[F(t*t-4*t)]*10,
                 native=[F(2*t-4)]*10,mesh=mesh) for t in (2,5,9)]
    res=side_check(points,anchor)
    assert res['numerically_passed'] and res['same_sampled_branch']
    points[1]['mesh']=({2},Counter({(1,2,4):1}),'same')
    changed=side_check(points,anchor)
    assert changed['numerically_passed'] and not changed['same_sampled_branch']
