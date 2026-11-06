from fractions import Fraction as F

import pytest

from .heart_point0_first_face import clip_segment, crossings, overlaps


def test_overlap_requires_positive_area_and_ignores_winding():
    a = [(0, 0), (4, 0), (0, 4)]
    b = [(0, 0), (4, 0), (3, -1)]
    assert not overlaps(a, b)
    assert overlaps(a, a[::-1])
    assert overlaps(a, [(0, 1), (2, -1), (2, 1)])


def test_clip_segment_excludes_vertex_contact_in_both_orientations():
    tri = [(0, 0), (2, 0), (0, 2)]
    for t in (tri, tri[::-1]):
        assert clip_segment((-1, 1), (2, 1), t) == ((0, 1), (1, 1))
        assert clip_segment((-1, 1), (1, -1), t) is None
        assert clip_segment((2, 2), (3, 2), t) is None


def test_section_ray_uses_half_open_vertex_rule_and_rejects_boundary():
    loop = [((0, 0), (2, 1)), ((2, 1), (0, 2)), ((0, 2), (-2, 1)), ((-2, 1), (0, 0))]
    assert crossings((F(1, 3), 1), loop) == 1
    assert crossings((-3, 1), loop) == 2
    with pytest.raises(ValueError, match="corner on cutter"):
        crossings((1, F(1, 2)), loop)
