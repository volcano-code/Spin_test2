import unittest

from research.NeuralMoments.models.pcraft_ufno.spin_pino.sweep import DEFAULT_CONFIGS


class SpinPINOSweepTests(unittest.TestCase):
    def test_configs_change_nodes_resolution_and_smoothing(self):
        self.assertEqual(len(DEFAULT_CONFIGS), 3)
        self.assertEqual(DEFAULT_CONFIGS[0]["grid_size"], 8)
        self.assertEqual(DEFAULT_CONFIGS[1]["points_per_axis"], 6)
        self.assertEqual(DEFAULT_CONFIGS[1]["smoothing"], 0.02)
        self.assertEqual(DEFAULT_CONFIGS[2]["grid_size"], 16)
        self.assertTrue(all(x["smoothing"] > 0 for x in DEFAULT_CONFIGS))


if __name__ == "__main__":
    unittest.main()
