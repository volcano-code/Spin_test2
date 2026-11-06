import unittest

import torch

from research.NeuralMoments.models.pcraft_ufno.spin_pino.dataset import generate_splits
from research.NeuralMoments.models.pcraft_ufno.spin_pino.equivariance import (
    random_so3_rotation,
    transform_native_coefficients,
    transform_raw_moments,
    transform_native_jacobian,
    augment_with_random_so3,
)


def native_quadric_value(points, coefficients):
    x, y, z = points.unbind(-1)
    basis = torch.stack((torch.ones_like(x), x, y, z, x*x, y*y, z*z, 2*y*z, 2*x*z, 2*x*y), dim=-1)
    return torch.einsum("bnd,bd->bn", basis, coefficients)


class SpinPINOSO3Tests(unittest.TestCase):
    def test_random_rotation_is_proper_orthogonal(self):
        R = random_so3_rotation(20261106)
        self.assertTrue(torch.allclose(R.T @ R, torch.eye(3, dtype=R.dtype), atol=1e-12))
        self.assertAlmostEqual(float(torch.det(R)), 1.0, places=12)

    def test_quadric_evaluation_is_rotation_covariant(self):
        R = random_so3_rotation(17, dtype=torch.float64)
        x = torch.tensor([[[0.2, -0.3, 0.1], [-0.4, 0.2, 0.3]]], dtype=torch.float64)
        c = torch.randn((1, 10), dtype=torch.float64)
        xp = torch.einsum("ij,bnj->bni", R, x)
        cp = transform_native_coefficients(c, R)
        self.assertTrue(torch.allclose(native_quadric_value(x, c), native_quadric_value(xp, cp), atol=1e-10, rtol=1e-10))

    def test_moment_and_jacobian_transform_shapes_and_finiteness(self):
        R = random_so3_rotation(29, dtype=torch.float64)
        moments = torch.randn((2, 10), dtype=torch.float64)
        J = torch.randn((2, 10, 10), dtype=torch.float64)
        rp = transform_raw_moments(moments, R)
        Jp = transform_native_jacobian(J, R)
        self.assertEqual(rp.shape, (2, 10))
        self.assertEqual(Jp.shape, (2, 10, 10))
        self.assertTrue(bool(torch.isfinite(rp).all() and torch.isfinite(Jp).all()))

    def test_so3_augmentation_is_reproducible_and_records_interpolation(self):
        data = generate_splits(
            seed=31, train_geometries_per_family=4,
            test_ellipsoid_geometries=2, queries_per_geometry=2, grid_size=8
        )["fit"]
        a = augment_with_random_so3(data, seed=900)
        b = augment_with_random_so3(data, seed=900)
        self.assertEqual(a[0].shape[0], 2 * data[0].shape[0])
        self.assertTrue(torch.equal(a[0], b[0]))
        self.assertTrue(torch.equal(a[1], b[1]))
        self.assertEqual(a[4][-1]["augmentation"], "HAAR_SO3_LABEL_EXACT_VOXEL_TRILINEAR")


if __name__ == "__main__":
    unittest.main()
