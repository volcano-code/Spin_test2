"""Intersect parallel midpoint rays with a closed triangle mesh in tip coordinates."""
from fractions import Fraction
from hashlib import sha256
from pathlib import Path

import numpy as np

DESIGN_SHA = "c3633c74bc74c2be488eb22de108454276f28941f0477cfee5837f57db99d622"


def load_design(path):
    raw = Path(path).read_bytes()
    if sha256(raw).hexdigest() != DESIGN_SHA:
        raise ValueError("Noodles Dprime identity mismatch")
    lines = raw.decode().splitlines()
    nv, nf = map(int, lines[0].split())
    vertices = np.array([[float(Fraction(v)) for v in s.split()] for s in lines[1:nv+1]])
    faces = np.array([r.split() for r in lines[nv+1:]], dtype=int)
    if faces.shape != (nf, 3):
        raise ValueError("bad triangle count")
    vertices[:, 2] += .16  # model to tip frame
    return vertices, faces


def crossings(vertices, faces, axis, side, origin=(-1., -1., 0.), phase=(.50000017, .50000031)):
    """Return sorted ray/triangle hits, barycentrics and transverse grid origin.

    Half-open ambiguity is rejected. The phase is a fixed near-midpoint offset in
    grid units, shared across parameter values and defined independently of q.
    """
    transverse = [d for d in range(3) if d != axis]
    h = 2. / side
    grid0 = np.asarray(origin)[transverse] + h * np.asarray(phase)
    t = np.asarray(vertices)[faces]
    uv = (t[:, :, transverse] - grid0) / h
    one, two = uv[:, 1] - uv[:, 0], uv[:, 2] - uv[:, 0]
    det = one[:, 0]*two[:, 1] - one[:, 1]*two[:, 0]
    lower = np.maximum(0, np.ceil(uv.min(axis=1)).astype(int))
    upper = np.minimum(side-1, np.floor(uv.max(axis=1)).astype(int))
    sizes = np.maximum(upper-lower+1, 0)
    counts = sizes.prod(axis=1) * (abs(det) > 1e-14)
    idx = np.repeat(np.arange(len(t)), counts)
    local = np.arange(len(idx)) - np.repeat(np.cumsum(counts)-counts, counts)
    u = lower[idx, 0] + local % sizes[idx, 0]
    v = lower[idx, 1] + local // sizes[idx, 0]
    du, dv = u-uv[idx, 0, 0], v-uv[idx, 0, 1]
    a = (du*two[idx, 1] - dv*two[idx, 0]) / det[idx]
    b = (one[idx, 0]*dv - one[idx, 1]*du) / det[idx]
    inside = (a >= 0) & (b >= 0) & (a+b <= 1)
    if np.any(inside & (np.minimum.reduce([a, b, 1-a-b]) < 2e-13)):
        raise ValueError("ray intersects triangle edge; choose another registered grid phase")
    idx, a, b = idx[inside], a[inside], b[inside]
    ray = (v[inside]*side + u[inside]).astype(int)
    depth = t[idx, 0, axis] + a*(t[idx, 1, axis]-t[idx, 0, axis]) + b*(t[idx, 2, axis]-t[idx, 0, axis])
    order = np.lexsort((depth, ray))
    if np.any(np.bincount(ray, minlength=side*side) % 2):
        raise ValueError("odd crossing count; check mesh and ray phase")
    return ray[order], depth[order], idx[order], np.column_stack((1-a-b, a, b))[order], grid0


def material_rays(vertices, faces, axis, side):
    ray, depth, _, _, grid0 = crossings(vertices, faces, axis, side)
    return bank_from_hits(ray, depth, grid0, axis, side)


def bank_from_hits(ray, depth, grid0, axis, side):
    h = 2. / side
    transverse = [d for d in range(3) if d != axis]
    ray = ray[::2]
    base = np.zeros((len(ray), 3))
    base[:, transverse[0]] = grid0[0] + (ray % side)*h
    base[:, transverse[1]] = grid0[1] + (ray // side)*h
    return base, depth[::2], depth[1::2], h*h
