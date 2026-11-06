from fractions import Fraction as F

import pytest

from .heart_point0_capture_moments import compare


def test_tetrahedron_matches_exact_reference_and_rejects_changed_component():
    vertices = [["0/1", "0/1", "0/1"], ["1/1", "0/1", "0/1"],
                ["0/1", "1/1", "0/1"], ["0/1", "0/1", "1/1"]]
    faces = [[0,2,1], [0,1,3], [0,3,2], [1,2,3]]
    exact = [F(1,6)] + [F(1,24)]*3 + [F(1,60)]*3 + [F(1,120)]*3
    observed = list(map(float,exact))
    assert compare(vertices,faces,1,observed,exact)["maximum_error_against_rounded_reference"] == 0
    with pytest.raises(ValueError, match="saved exact moments differ"):
        compare(vertices,faces,1,observed,exact[:-1]+[F(0)])
    with pytest.raises(ValueError, match="exceeds component budget"):
        compare(vertices,faces,1,observed[:-1]+[0.],exact)
