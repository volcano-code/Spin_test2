import unittest

from research.NeuralMoments.models.pcraft_ufno.spin_pino.tournament import _build


class SpinPINOTournamentTests(unittest.TestCase):
    def test_all_four_models_construct_and_keep_pino_primary(self):
        kinds = (
            "spin_pino_4node",
            "spin_pino_6node",
            "spin_pino_6node_gauss",
            "spin_pino_6node_logistic",
            "w6_m7_realizable_ufno",
            "w5_voxel_deeponet",
        )
        models = {kind: _build(kind) for kind in kinds}
        self.assertEqual(models["spin_pino_4node"].points_per_axis, 4)
        self.assertEqual(models["spin_pino_6node"].points_per_axis, 6)
        self.assertEqual(models["spin_pino_6node_gauss"].quadrature_rule, "gauss")
        self.assertAlmostEqual(float(models["spin_pino_6node_gauss"].node_volume.sum()), 8.0, places=6)
        self.assertEqual(models["spin_pino_6node_logistic"].moment_parameterization, "logistic_atom")
        self.assertTrue(hasattr(models["w6_m7_realizable_ufno"], "moment"))
        self.assertTrue(hasattr(models["w5_voxel_deeponet"], "moment"))
        self.assertNotEqual(type(models["spin_pino_4node"]), type(models["w6_m7_realizable_ufno"]))


if __name__ == "__main__":
    unittest.main()
