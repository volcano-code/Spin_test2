from fractions import Fraction as F

import pytest

from .heart_axis_feasibility import certificate, clip, integrate, interval


def test_clipped_tetra_matches_cross_section_antiderivative():
    # Section area (1-x)^2/8 on [-1,1], with an entire face at x=-1.
    vertices=[tuple(map(F,v)) for v in [(-1,0,0),(1,0,0),(-1,1,0),(-1,0,1)]]
    faces=[[1,2,3],[0,3,2],[0,1,3],[0,2,1]]
    assert integrate(vertices,faces)==(F(-17,96),F(1,96),F(-1,6),F(1,3),3)
    with pytest.raises(ValueError,match='orientation'):
        integrate(vertices,[f[::-1] for f in faces])


def test_exact_halfspace_cap_and_boundary_vertex():
    vertices=[tuple(map(F,v)) for v in [(0,0,0),(1,0,0),(0,1,0),(0,0,1)]]
    faces=[[1,2,3],[0,3,2],[0,1,3],[0,2,1]]
    assert integrate(vertices,faces)==(0,F(1,24),F(1,24),F(1,6),0)
    triangle=tuple(tuple(map(F,v)) for v in [(-1,0,0),(0,1,0),(2,0,1)])
    assert clip(triangle,-1)==[(triangle[0],triangle[1],(F(0),F(0),F(1,3)))]


def test_bounds_round_outward_for_positive_and_negative_rationals(tmp_path):
    assert interval(F(-1,3),3)==['-0.334','-0.333']
    assert interval(F(1,3),3)==['0.333','0.334']
    assert interval(F(1,4),3)==['0.250','0.250']
    changed=tmp_path/'request.json'
    changed.write_text('{}')
    with pytest.raises(ValueError,match='changed prepared request'):
        certificate(changed)
