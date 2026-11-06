from fractions import Fraction as F

import pytest

from .heart_newton_intersection import component_volumes


def tetra(offset, size):
    return [tuple(F(offset)+(F(size) if j==i else 0) for j in range(3)) for i in (-1,0,1,2)]


def pair(offset, size):
    points=tetra(0,4)+tetra(offset,size)
    unit=[[0,2,1],[0,1,3],[0,3,2],[1,2,3]]
    return points,unit+[[x+4 for x in f] for f in unit]


def test_disjoint_outward_components():
    pts,faces=pair(8,1)
    assert component_volumes(pts,faces)==[F(64,6),F(1,6)]


def test_nested_positive_shells_do_not_define_a_unit_winding_solid():
    pts,faces=pair(F(1,3),F(1,2))
    with pytest.raises(ValueError,match='nested shells'):
        component_volumes(pts,faces)


def test_inward_orientations_rejected():
    pts,faces=pair(8,1)
    with pytest.raises(ValueError,match='nonpositive'):
        component_volumes(pts,[f[::-1] for f in faces])
