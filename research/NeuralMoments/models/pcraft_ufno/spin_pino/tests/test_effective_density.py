import unittest

import torch

from research.NeuralMoments.models.pcraft_ufno.spin_pino.support_ablation import PINO, _build, two_component_panel


class EffectiveDensityTests(unittest.TestCase):
    def test_fractional_support_count_and_empty(self):
        model = _build(PINO).eval()
        geometry = torch.zeros(1, 4, 8, 8, 8)
        geometry[:, :2] = 0.5
        with torch.no_grad():
            twice = model.encode_geometry(geometry)
            model.density_is_premultiplied = True
            once = model.encode_geometry(geometry)
            torch.testing.assert_close(once, 2 * twice)
            geometry[:, 1] = 0
            self.assertEqual(model.encode_geometry(geometry).count_nonzero().item(), 0)

    def test_held_out_family_smooth_and_disjoint(self):
        voxels, c, r, jac, parents = two_component_panel(131, count=2)
        self.assertEqual(len(set(parents)), 2)
        self.assertEqual(parents[::2], parents[1::2])
        torch.testing.assert_close(voxels[::2], voxels[1::2])
        torch.testing.assert_close(voxels[:,0], voxels[:,1])
        self.assertTrue(torch.all(r[::2,0] > r[1::2,0]))
        torch.testing.assert_close((jac @ c.unsqueeze(-1)), torch.zeros(4,10,1), atol=2e-7, rtol=0)
        self.assertEqual(int(c[:,4:].count_nonzero()), 0)
