import unittest

import torch
from torch import nn

from research.NeuralMoments.models.pcraft_ufno.spin_pino.loss_conditioning import (
    DiagonalMomentResidualPreconditioner,
    project_conflicting_gradients,
    pcgrad_backward,
)


class PINOLossConditioningTests(unittest.TestCase):
    def test_diagonal_preconditioner_is_positive_and_fit_split_scoped(self):
        r = torch.randn(24, 10)
        J = torch.randn(24, 10, 10)
        p = DiagonalMomentResidualPreconditioner.fit(r, J)
        self.assertEqual(p.fit_samples, 24)
        self.assertEqual(p.moment_scale.shape, (10,))
        self.assertEqual(p.jacobian_scale.shape, (10, 10))
        self.assertTrue(bool((p.moment_scale > 0).all()))
        self.assertTrue(bool((p.jacobian_scale > 0).all()))
        self.assertFalse(p.as_dict()["validation_or_test_used_for_fit"])

    def test_pcgrad_backward_reconstructs_mixed_real_complex_parameter_grads(self):
        class Mixed(nn.Module):
            def __init__(self):
                super().__init__()
                self.real = nn.Parameter(torch.tensor([1.0, 2.0]))
                self.complex = nn.Parameter(torch.tensor([1.0+1.0j, 2.0-1.0j], dtype=torch.complex64))
        model = Mixed()
        loss_a = model.real[0] + model.complex[0].real
        loss_b = -model.real[0] - model.complex[0].real
        stats = pcgrad_backward((loss_a, loss_b), model.parameters(), (1.0, 1.0))
        self.assertEqual(model.real.grad.dtype, model.real.dtype)
        self.assertEqual(model.complex.grad.dtype, model.complex.dtype)
        self.assertTrue(bool(torch.isfinite(model.real.grad).all()))
        self.assertTrue(bool(torch.isfinite(model.complex.grad).all()))
        self.assertGreater(stats["projection_count"], 0)

    def test_pcgrad_supports_complex_spectral_parameter_gradients(self):
        g1 = torch.tensor([1.0 + 1.0j, 0.0 + 0.0j], dtype=torch.complex64)
        g2 = torch.tensor([-1.0 + 0.0j, 1.0j], dtype=torch.complex64)
        projected, stats = project_conflicting_gradients((g1, g2))
        self.assertTrue(bool(torch.isfinite(projected[0]).all()))
        self.assertTrue(bool(torch.isfinite(projected[1]).all()))
        self.assertGreater(stats["projection_count"], 0)
        self.assertTrue(all(isinstance(x, float) for x in stats["raw_pairwise_cosines"].values()))

    def test_pcgrad_projects_conflicting_task_gradients(self):
        g1 = torch.tensor([1.0, 0.0])
        g2 = torch.tensor([-1.0, 1.0])
        projected, stats = project_conflicting_gradients((g1, g2))
        self.assertGreaterEqual(float(torch.dot(projected[0], g2)), -1e-6)
        self.assertGreaterEqual(float(torch.dot(projected[1], g1)), -1e-6)
        self.assertGreater(stats["projection_count"], 0)
        self.assertLess(stats["raw_pairwise_cosines"]["0_1"], 0.0)


if __name__ == "__main__":
    unittest.main()
