"""Evaluate a fixed positive geometry measure on a refined lattice or inside a bounded ellipsoid."""
import math
from functools import lru_cache

import numpy as np
import torch
import torch.nn.functional as F

from .spin_pino import NATIVE_B


def monomials(xyz):
    x, y, z = xyz.T
    return np.column_stack((np.ones_like(x), x, y, z, x*x, y*y, z*z, y*z, x*z, x*y))


def density_at(voxels, gates, xyz):
    """Keep geometry encoding independent of the query; prolong coarse gates to children."""
    grid = torch.tensor(xyz[:, ::-1].copy(), dtype=voxels.dtype).view(1, -1, 1, 1, 3)
    rho = F.grid_sample(voxels[:, :1], grid, align_corners=True).flatten().numpy()
    indices = np.clip(np.floor((xyz + 1) * 3).astype(int), 0, 5)
    return rho * gates[tuple(indices.T)]


@lru_cache(maxsize=2)
def lattice(size):
    axis = (np.arange(size) + .5) * (2 / size) - 1
    xyz = np.stack(np.meshgrid(axis, axis, axis, indexing="ij"), axis=-1).reshape(-1, 3)
    return xyz, monomials(xyz)


def cartesian(voxels, gates, q, size, temperature):
    xyz, m = lattice(size)
    weights = (8 / size**3) * density_at(voxels, gates, xyz)
    beta = m @ (q * NATIVE_B)
    norm = np.linalg.norm(q)
    tau = temperature * norm
    # Evaluate sigmoid and its derivative in double precision on both tails.
    exp = np.exp(-np.abs(beta / tau))
    alpha = np.where(beta >= 0, 1, exp) / (1 + exp)
    dv = m * NATIVE_B - np.outer(beta / norm**2, q)
    return (weights * alpha) @ m, m.T @ ((weights * exp / (1+exp)**2 / tau)[:, None] * dv)


@lru_cache(maxsize=1)
def ball_rule():
    z, wz = np.polynomial.legendre.leggauss(8)
    t = np.arange(16) * (2 * math.pi / 16)
    u = np.stack((np.sqrt(1-z[:, None]**2) * np.cos(t),
                  np.sqrt(1-z[:, None]**2) * np.sin(t),
                  np.broadcast_to(z[:, None], (8, 16))), axis=-1).reshape(-1, 3)
    w = np.repeat(wz * (2 * math.pi / 16), 16)
    t, wt = np.polynomial.legendre.leggauss(6)
    r = (t + 1)/2
    return (r[:, None, None] * u).reshape(-1, 3), np.outer(wt * r**2 / 2, w).ravel(), u, w


def ellipsoid(q):
    """Return x=center+transform*u, beta=k*(1-|u|²); require the chart inside the voxel cube."""
    h = -np.array([[q[4], q[9], q[8]], [q[9], q[5], q[7]], [q[8], q[7], q[6]]])
    try:
        lower = np.linalg.cholesky(h)
    except np.linalg.LinAlgError:
        return None
    center = np.linalg.solve(h, q[1:4]) / 2
    k = q[0] + .5 * center @ q[1:4]
    if k <= 0:
        return None
    transform = np.linalg.solve(lower.T, np.eye(3)) * np.sqrt(k)
    if np.any(abs(center) + np.linalg.norm(transform, axis=1) >= 1):
        return None
    return center, transform, k


def localized(voxels, gates, q):
    chart = ellipsoid(q)
    if chart is None:
        r, J = cartesian(voxels, gates, q, 24, .0075)
        return r, J, "cartesian_fallback"
    center, transform, k = chart
    volume_u, wv, surface_u, ws = ball_rule()
    det = np.linalg.det(transform)
    x = center + volume_u @ transform.T
    r = (det * wv * density_at(voxels, gates, x)) @ monomials(x)
    x = center + surface_u @ transform.T
    m = monomials(x)
    # Shape derivative of the represented measure, evaluated independently on the interface.
    surface = det / (2*k) * ws * density_at(voxels, gates, x)
    J = (m.T @ (surface[:, None] * m)) * NATIVE_B
    return r, J, "ellipsoid_coarea"
