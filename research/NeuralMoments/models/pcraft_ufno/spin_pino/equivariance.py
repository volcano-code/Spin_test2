"""Exact rigid-rotation transforms for Spin's ten moments and native quadrics.

The utilities provide an E(3)-covariance audit/augmentation path for the
P-CRAFT PINO model. They do not make the network equivariant by themselves.
"""

from __future__ import annotations

import math
import itertools
import copy

import torch
from torch import Tensor
import torch.nn.functional as F


def _check_rotation(R: Tensor) -> None:
    if R.shape != (3, 3) or not R.is_floating_point() or not bool(torch.isfinite(R).all()):
        raise ValueError("R must be a finite floating [3,3] rotation matrix")
    eye = torch.eye(3, dtype=R.dtype, device=R.device)
    if not torch.allclose(R.T @ R, eye, atol=1e-6, rtol=1e-6):
        raise ValueError("R must be orthogonal")
    if not torch.allclose(torch.det(R), R.new_tensor(1.0), atol=1e-6, rtol=1e-6):
        raise ValueError("R must be a proper rotation with determinant +1")


def rotation_z(angle: float, *, dtype=torch.float64, device=None) -> Tensor:
    if not math.isfinite(angle):
        raise ValueError("angle must be finite")
    c, s = math.cos(angle), math.sin(angle)
    return torch.tensor([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]],
                        dtype=dtype, device=device)


def native_coefficients_to_matrix(coefficients: Tensor):
    """Native c order -> polynomial A,b with f(x)=x^T A x+b^T x+c0."""
    if coefficients.ndim != 2 or coefficients.shape[-1] != 10:
        raise ValueError("coefficients must be [B,10]")
    c = coefficients
    A = torch.zeros((c.shape[0], 3, 3), dtype=c.dtype, device=c.device)
    A[:, 0, 0], A[:, 1, 1], A[:, 2, 2] = c[:, 4], c[:, 5], c[:, 6]
    A[:, 1, 2] = A[:, 2, 1] = c[:, 7]
    A[:, 0, 2] = A[:, 2, 0] = c[:, 8]
    A[:, 0, 1] = A[:, 1, 0] = c[:, 9]
    return A, c[:, 1:4], c[:, 0]


def matrix_to_native_coefficients(A: Tensor, b: Tensor, c0: Tensor) -> Tensor:
    if A.ndim != 3 or A.shape[-2:] != (3, 3) or b.shape != (A.shape[0], 3):
        raise ValueError("expected A[B,3,3] and b[B,3]")
    if c0.shape not in ((A.shape[0],), (A.shape[0], 1)):
        raise ValueError("c0 must be [B] or [B,1]")
    c0 = c0.reshape(-1, 1)
    return torch.stack(
        (c0[:, 0], b[:, 0], b[:, 1], b[:, 2],
         A[:, 0, 0], A[:, 1, 1], A[:, 2, 2],
         A[:, 1, 2], A[:, 0, 2], A[:, 0, 1]),
        dim=-1,
    )


def transform_native_coefficients(coefficients: Tensor, R: Tensor) -> Tensor:
    """Coefficients of f'(x')=f(R^T x') when geometry is rotated by R."""
    _check_rotation(R)
    A, b, c0 = native_coefficients_to_matrix(coefficients)
    R = R.to(dtype=coefficients.dtype, device=coefficients.device)
    A_new = R.unsqueeze(0) @ A @ R.T.unsqueeze(0)
    b_new = torch.einsum("ij,bj->bi", R, b)
    return matrix_to_native_coefficients(A_new, b_new, c0)


def transform_raw_moments(moments: Tensor, R: Tensor) -> Tensor:
    """Transform [mass, first moments, symmetric second moments] under x'=R x."""
    _check_rotation(R)
    if moments.ndim != 2 or moments.shape[-1] != 10:
        raise ValueError("moments must be [B,10]")
    R = R.to(dtype=moments.dtype, device=moments.device)
    m = moments[:, 0]
    first = torch.einsum("ij,bj->bi", R, moments[:, 1:4])
    Q = torch.zeros((moments.shape[0], 3, 3), dtype=moments.dtype, device=moments.device)
    Q[:, 0, 0], Q[:, 1, 1], Q[:, 2, 2] = moments[:, 4], moments[:, 5], moments[:, 6]
    Q[:, 1, 2] = Q[:, 2, 1] = moments[:, 7]
    Q[:, 0, 2] = Q[:, 2, 0] = moments[:, 8]
    Q[:, 0, 1] = Q[:, 1, 0] = moments[:, 9]
    Q_new = R.unsqueeze(0) @ Q @ R.T.unsqueeze(0)
    return torch.stack(
        (m, first[:, 0], first[:, 1], first[:, 2],
         Q_new[:, 0, 0], Q_new[:, 1, 1], Q_new[:, 2, 2],
         Q_new[:, 1, 2], Q_new[:, 0, 2], Q_new[:, 0, 1]),
        dim=-1,
    )


def moment_rotation_matrix(R: Tensor, *, dtype=None, device=None) -> Tensor:
    _check_rotation(R)
    dtype = dtype or R.dtype
    device = device or R.device
    basis = torch.eye(10, dtype=dtype, device=device)
    return transform_raw_moments(basis, R.to(dtype=dtype, device=device)).T


def coefficient_rotation_matrix(R: Tensor, *, dtype=None, device=None) -> Tensor:
    _check_rotation(R)
    dtype = dtype or R.dtype
    device = device or R.device
    basis = torch.eye(10, dtype=dtype, device=device)
    return transform_native_coefficients(basis, R.to(dtype=dtype, device=device)).T


def transform_native_jacobian(J: Tensor, R: Tensor) -> Tensor:
    """If r'=T_R r and c'=S_R c, then J'=T_R J S_R^{-1}."""
    _check_rotation(R)
    if J.ndim != 3 or J.shape[-2:] != (10, 10):
        raise ValueError("J must be [B,10,10]")
    T = moment_rotation_matrix(R, dtype=J.dtype, device=J.device)
    S = coefficient_rotation_matrix(R, dtype=J.dtype, device=J.device)
    return T.unsqueeze(0) @ J @ torch.linalg.inv(S).unsqueeze(0)


def rotate_geometry_voxels(geometry: Tensor, R: Tensor) -> Tensor:
    """Resample canonical [B,4,X,Y,Z] geometry under x'=R x.

    PyTorch grid_sample expects (z,y,x) for a tensor stored as [B,C,X,Y,Z];
    the reversal is applied only at the sampling boundary.
    """
    _check_rotation(R)
    if geometry.ndim != 5 or geometry.shape[1] != 4 or not geometry.is_floating_point():
        raise ValueError("geometry must be floating [B,4,X,Y,Z]")
    if not bool(torch.isfinite(geometry).all()):
        raise ValueError("geometry must be finite")
    R = R.to(dtype=geometry.dtype, device=geometry.device)
    nx, ny, nz = geometry.shape[-3:]
    ax = torch.linspace(-1.0, 1.0, nx, dtype=geometry.dtype, device=geometry.device)
    ay = torch.linspace(-1.0, 1.0, ny, dtype=geometry.dtype, device=geometry.device)
    az = torch.linspace(-1.0, 1.0, nz, dtype=geometry.dtype, device=geometry.device)
    xyz = torch.stack(torch.meshgrid(ax, ay, az, indexing="ij"), dim=-1)
    # Row-vector x' maps back to x=R^T x' as x'^T R.
    source_xyz = xyz @ R
    grid = torch.stack((source_xyz[..., 2], source_xyz[..., 1], source_xyz[..., 0]), dim=-1)
    grid = grid.unsqueeze(0).expand(geometry.shape[0], -1, -1, -1, -1)
    return F.grid_sample(
        geometry, grid, mode="bilinear", padding_mode="zeros", align_corners=True
    )



def cube_rotation_matrices(*, dtype=torch.float64, device=None) -> Tensor:
    """The 24 proper signed-permutation rotations preserving a cubic voxel grid."""
    matrices = []
    for permutation in itertools.permutations(range(3)):
        for signs in itertools.product((-1.0, 1.0), repeat=3):
            R = torch.zeros((3, 3), dtype=dtype, device=device)
            for row, column in enumerate(permutation):
                R[row, column] = signs[row]
            if float(torch.det(R)) > 0.5:
                matrices.append(R)
    return torch.stack(matrices, dim=0)


def cube_rotation(index: int, *, dtype=torch.float64, device=None) -> Tensor:
    rotations = cube_rotation_matrices(dtype=dtype, device=device)
    return rotations[int(index) % rotations.shape[0]]


def augment_with_cube_rotations(data, *, seed: int):
    """Append one exact SO(3) cube rotation per sample to a five-item data tuple.

    The original analytic vertical-plane labels are transformed exactly:
    r' = T_R r and J' = T_R J S_R^{-1}. The voxel tensor is rotated on the
    cubic grid; signed-permutation rotations map voxel centers exactly.
    Call only on the training split after geometry-family splits are frozen.
    """
    if len(data) != 5:
        raise ValueError("expected (D,c,r,J,metadata)")
    D, coefficients, moments, jacobians, metadata = data
    if D.ndim != 5 or D.shape[1] != 4:
        raise ValueError("D must be [N,4,X,Y,Z]")
    count = D.shape[0]
    if any(x.shape[0] != count for x in (coefficients, moments, jacobians)) or len(metadata) != count:
        raise ValueError("all data fields must have matching sample count")
    D_aug, c_aug, r_aug, J_aug, meta_aug = [], [], [], [], []
    for i in range(count):
        R = cube_rotation(seed + i, dtype=D.dtype, device=D.device)
        D_aug.append(rotate_geometry_voxels(D[i:i+1], R))
        c_aug.append(transform_native_coefficients(coefficients[i:i+1], R))
        r_aug.append(transform_raw_moments(moments[i:i+1], R))
        J_aug.append(transform_native_jacobian(jacobians[i:i+1], R))
        row = copy.deepcopy(metadata[i])
        row["parent_geometry_id"] = row.get("geometry_id")
        row["geometry_id"] = f'{row.get("geometry_id", "geometry")}__cube_rot_{(seed+i)%24:02d}'
        row["augmentation"] = "EXACT_CUBE_ROTATION_LABEL_TRANSFORM"
        row["rotation_matrix"] = [[float(v) for v in line] for line in R.cpu().tolist()]
        meta_aug.append(row)
    return (
        torch.cat((D, torch.cat(D_aug, dim=0)), dim=0),
        torch.cat((coefficients, torch.cat(c_aug, dim=0)), dim=0),
        torch.cat((moments, torch.cat(r_aug, dim=0)), dim=0),
        torch.cat((jacobians, torch.cat(J_aug, dim=0)), dim=0),
        list(metadata) + meta_aug,
    )


def random_so3_rotation(seed: int, *, dtype=torch.float64, device=None) -> Tensor:
    """Deterministic Haar-uniform SO(3) rotation from a seeded unit quaternion."""
    import numpy as np
    rng = np.random.default_rng(int(seed))
    u1, u2, u3 = rng.random(3)
    x = math.sqrt(1.0-u1) * math.sin(2.0*math.pi*u2)
    y = math.sqrt(1.0-u1) * math.cos(2.0*math.pi*u2)
    z = math.sqrt(u1) * math.sin(2.0*math.pi*u3)
    w = math.sqrt(u1) * math.cos(2.0*math.pi*u3)
    R = torch.tensor([
        [1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
        [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
        [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)],
    ], dtype=dtype, device=device)
    _check_rotation(R)
    return R


def augment_with_random_so3(data, *, seed: int):
    """Append one arbitrary SO(3) rotation per sample.

    Moment and coefficient labels transform exactly. The voxel field is
    trilinearly resampled on the fixed bounding cube, so its interpolation
    error is measured separately; unlike cube rotations this is not an exact
    permutation of voxel centers.
    """
    if len(data) != 5:
        raise ValueError("expected (D,c,r,J,metadata)")
    D, coefficients, moments, jacobians, metadata = data
    if D.ndim != 5 or D.shape[1] != 4:
        raise ValueError("D must be [N,4,X,Y,Z]")
    count = D.shape[0]
    if any(x.shape[0] != count for x in (coefficients, moments, jacobians)) or len(metadata) != count:
        raise ValueError("all data fields must have matching sample count")
    D_aug, c_aug, r_aug, J_aug, meta_aug = [], [], [], [], []
    for i in range(count):
        R = random_so3_rotation(seed + i, dtype=D.dtype, device=D.device)
        D_aug.append(rotate_geometry_voxels(D[i:i+1], R))
        c_aug.append(transform_native_coefficients(coefficients[i:i+1], R))
        r_aug.append(transform_raw_moments(moments[i:i+1], R))
        J_aug.append(transform_native_jacobian(jacobians[i:i+1], R))
        row = copy.deepcopy(metadata[i])
        row["parent_geometry_id"] = row.get("geometry_id")
        row["geometry_id"] = f'{row.get("geometry_id", "geometry")}__so3_{seed+i}'
        row["augmentation"] = "HAAR_SO3_LABEL_EXACT_VOXEL_TRILINEAR"
        row["rotation_matrix"] = [[float(v) for v in line] for line in R.cpu().tolist()]
        meta_aug.append(row)
    return (
        torch.cat((D, torch.cat(D_aug, dim=0)), dim=0),
        torch.cat((coefficients, torch.cat(c_aug, dim=0)), dim=0),
        torch.cat((moments, torch.cat(r_aug, dim=0)), dim=0),
        torch.cat((jacobians, torch.cat(J_aug, dim=0)), dim=0),
        list(metadata) + meta_aug,
    )
