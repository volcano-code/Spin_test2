"""Closed-form continuum labels for a small ellipsoid cut by its fixed equatorial plane."""
import math
import numpy as np
from ..analytic_cut import POWERS, B_NATIVE_TO_MONOMIAL, GeometrySpec, _monomial_affine_matrix, voxel_geometry


def hemisphere(p):
    i, j, k = p
    if j % 2 or k % 2:
        return 0.
    return (-1)**i * math.prod(math.gamma((x+1)/2) for x in p) / math.gamma((i+j+k+3)/2)


def half_truth(center, axes):
    # The design box retains x <= center.x. Its other five faces are strictly away.
    inv = 1 / axes**2
    q = np.r_[1-center @ (center * inv), 2*center*inv, -inv, [0, 0, 0]]
    norm = np.linalg.norm(q)
    m = np.array([hemisphere(p)/(3+sum(p)) for p in POWERS])
    G = np.array([[hemisphere(tuple(np.add(p, r))) for r in POWERS] for p in POWERS])
    T = _monomial_affine_matrix(GeometrySpec("ellipsoid", tuple(center), tuple(axes)))
    vol = np.prod(axes)
    return q/norm, vol * T @ m, vol * norm/2 * (T @ G @ T.T) @ B_NATIVE_TO_MONOMIAL


def panel():
    rng = np.random.default_rng(3601)
    cases = []
    for i in range(8):
        origin = rng.uniform(-.12, .12, size=3)
        radii = rng.uniform(.38, .55, size=3)
        box = GeometrySpec("box", tuple(origin), tuple(radii))
        center = origin + np.r_[radii[0], rng.uniform(-.12, .12, size=2)]
        q, r, J = half_truth(center, rng.uniform(.06, .14, size=3))
        cases.append((f"box-{i}", voxel_geometry(box, 8)[None], q, r, J, box.geometry_id))
    return cases
