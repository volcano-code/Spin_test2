import unittest

from research.NeuralMoments.models.pcraft_ufno.spin_pino.dataset import generate_splits


class SpinPINODatasetTests(unittest.TestCase):
    def test_family_and_geometry_identity_splits_are_disjoint(self):
        result = generate_splits(
            seed=20261106,
            train_geometries_per_family=4,
            test_ellipsoid_geometries=2,
            queries_per_geometry=2,
            grid_size=8,
        )
        fit = result["fit"]
        validation = result["validation"]
        test = result["test"]
        meta = result["metadata"]
        fit_ids = {x["geometry_id"] for x in fit[4]}
        val_ids = {x["geometry_id"] for x in validation[4]}
        test_ids = {x["geometry_id"] for x in test[4]}
        self.assertFalse(fit_ids & val_ids)
        self.assertFalse(fit_ids & test_ids)
        self.assertFalse(val_ids & test_ids)
        self.assertEqual({x["family"] for x in fit[4]}, {"box", "sphere"})
        self.assertEqual({x["family"] for x in validation[4]}, {"box", "sphere"})
        self.assertEqual({x["family"] for x in test[4]}, {"ellipsoid"})
        self.assertEqual(meta["label_source"], "ANALYTIC_CONTINUUM_P0_NOT_W2_CHERCHI")
        self.assertFalse(meta["w2_or_cherchi_labels_used"])
        self.assertEqual(fit[0].shape[1:], (4, 8, 8, 8))
        self.assertEqual(fit[2].shape[-1], 10)
        self.assertEqual(fit[3].shape[-2:], (10, 10))

    def test_tiny_analytic_training_smoke(self):
        from research.NeuralMoments.models.pcraft_ufno.spin_pino.train import run_seed
        result = run_seed(
            seed=20261107,
            train_geometries_per_family=4,
            test_ellipsoid_geometries=2,
            queries_per_geometry=2,
            steps=2,
            batch_size=4,
            grid_size=8,
            device="cpu",
        )
        self.assertEqual(result["schema"], "SPIN_PINO_ANALYTIC_P0_TRAINING_V1")
        self.assertFalse(result["w2_or_cherchi_labels_used"])
        self.assertGreater(result["trainable_parameters"], 0)
        self.assertGreaterEqual(result["held_out_ellipsoid_metrics"]["r_mae"], 0.0)


if __name__ == "__main__":
    unittest.main()
