import unittest

import torch

from research.NeuralMoments.models.pcraft_ufno.spin_pino.dataset import generate_splits
from research.NeuralMoments.models.pcraft_ufno.spin_pino.equivariance import (
    augment_with_cube_rotations,
    cube_rotation_matrices,
)


class SpinPINOEquivarianceTests(unittest.TestCase):
    def test_proper_cube_rotation_group_has_24_elements(self):
        rotations = cube_rotation_matrices()
        self.assertEqual(rotations.shape, (24, 3, 3))
        for R in rotations:
            self.assertTrue(torch.allclose(R.T @ R, torch.eye(3, dtype=R.dtype), atol=1e-12))
            self.assertAlmostEqual(float(torch.det(R)), 1.0, places=12)

    def test_exact_rotation_augmentation_preserves_data_contract(self):
        splits = generate_splits(
            seed=20261106,
            train_geometries_per_family=4,
            test_ellipsoid_geometries=2,
            queries_per_geometry=2,
            grid_size=8,
        )
        fit = splits["fit"]
        one = (fit[0][:1], fit[1][:1], fit[2][:1], fit[3][:1], fit[4][:1])
        augmented = augment_with_cube_rotations(one, seed=17)
        self.assertEqual(augmented[0].shape, (2, 4, 8, 8, 8))
        self.assertEqual(augmented[1].shape, (2, 10))
        self.assertEqual(augmented[2].shape, (2, 10))
        self.assertEqual(augmented[3].shape, (2, 10, 10))
        self.assertEqual(augmented[4][1]["augmentation"], "EXACT_CUBE_ROTATION_LABEL_TRANSFORM")
        self.assertNotEqual(augmented[4][0]["geometry_id"], augmented[4][1]["geometry_id"])


if __name__ == "__main__":
    unittest.main()
