from fractions import Fraction as F

import pytest

from .heart_point0_seams import cycles, intersection, normal, observed_seams, winding


def test_transverse_segment_clips_both_triangles_and_respects_orientation():
    design = [(0, 0, 0), (4, 0, 0), (0, 4, 0)]
    cutter = [(1, -2, -1), (1, 4, -1), (1, 0, 1)]
    ab = intersection(design, normal(design), cutter, normal(cutter))
    assert ab == ((1, 2, 0), (1, 0, 0))
    assert intersection(design, normal(design), cutter[::-1], normal(cutter[::-1])) == ab[::-1]
    far = [(x, y + 10, z) for x, y, z in cutter]
    assert intersection(design, normal(design), far, normal(far)) is None


def test_singular_input_contacts_are_rejected():
    a = [(0, 0, 0), (4, 0, 0), (0, 4, 0)]
    with pytest.raises(ValueError, match='coplanar'):
        intersection(a, normal(a), a, normal(a))
    b = [(1, -1, -1), (1, 3, -1), (1, 1, 1)]
    with pytest.raises(ValueError, match='coincident input edge'):
        intersection(a, normal(a), b, normal(b))


# The four faces form a positively oriented tetrahedron. Reassigning one face
# defines a synthetic two-patch surface for the saved-edge extraction alone.
POINTS = [(0,0,0), (2,0,0), (0,2,0), (0,0,2)]
FACES = [[0,2,1], [0,1,3], [0,3,2], [1,2,3]]


def test_observed_graph_requires_opposite_edges_and_a_closed_loop():
    owners = [[0, 1], [0, 2], [0, 3], [1, 0]]
    seams = observed_seams(FACES, owners, POINTS)
    assert len(seams) == 3 and cycles(list(seams.values())) == [3]
    with pytest.raises(ValueError, match='winding'):
        observed_seams(FACES[:3] + [FACES[3][::-1]], owners, POINTS)
    with pytest.raises(ValueError, match='simple directed cycles'):
        cycles(list(seams.values())[:-1])


def test_signed_ray_winding_and_boundary_rejection():
    tris = [[POINTS[k] for k in face] for face in FACES]
    boxes = [(max(x), min(y), max(y), min(z), max(z)) for x,y,z in (zip(*t) for t in tris)]
    n = [normal(t) for t in tris]
    assert winding((F(1,4), F(1,4), F(1,4)), tris, n, boxes) == 1
    assert winding((-1, F(1,4), F(1,4)), tris, n, boxes) == 0
    assert winding((F(3,2), F(1,4), F(1,4)), tris, n, boxes) is None
