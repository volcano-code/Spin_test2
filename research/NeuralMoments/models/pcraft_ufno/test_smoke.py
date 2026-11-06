"""Independent CPU smoke tests; run with python -m unittest discover -p test_smoke.py."""
import math
import unittest
from unittest.mock import patch

import torch

from .adaptation import ResidualAdapter
from .model import (FactorizedSpectralConv3d, UFNOMoment, canonical_geometry,
                    count_real_parameters, positive_projective_coefficients)
from .synthetic import (ellipsoid_raw_moments, slab_coefficients, slab_dr_dt,
                        slab_moments, solid_box_geometry)


class TestW6Milestone1(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.manual_seed(20261019)
        torch.set_num_threads(1)  # predictable CPU test throughput

    def make_model(self):
        return UFNOMoment(width=2, modes=3, latent_dim=12, head_width=24,
                          grid_size=8).eval()

    def test_four_channel_contract_and_imputed_fields(self):
        rho = torch.full((2, 1, 8, 8, 8), .75)
        d = canonical_geometry({"density": rho})
        self.assertEqual(tuple(d.shape), (2, 4, 8, 8, 8))
        self.assertTrue(torch.allclose(d[:, 0], rho[:, 0]))
        self.assertTrue(torch.all(d[:, 1] == 1))
        self.assertTrue(torch.all(d[:, 2:] == 0))
        sdf = -torch.ones(1, 8, 8, 8)
        d2 = canonical_geometry({"sdf": sdf, "inner_boundary": sdf.abs()})
        self.assertEqual(tuple(d2.shape), (1, 4, 8, 8, 8))
        self.assertTrue(torch.all(d2[:, 1] == 1))
        self.assertTrue(torch.all(d2[:, 2] < 0))
        self.assertTrue(torch.all(d2[:, 3] == 1))
        with self.assertRaises(ValueError):
            canonical_geometry({"density": rho, "surprise": rho})
        with self.assertRaises(ValueError):
            canonical_geometry(torch.ones(1, 9, 8, 8, 8))

    def test_fourier_mode_clipping_and_backprop(self):
        layer = FactorizedSpectralConv3d(2, 3, 20)
        x = torch.randn(1, 2, 8, 8, 8, requires_grad=True)
        y = layer(x)
        self.assertEqual(tuple(y.shape), (1, 3, 8, 8, 8))
        self.assertTrue(torch.isfinite(y).all())
        y.square().mean().backward()
        self.assertTrue(torch.isfinite(x.grad).all())

    def test_one_encode_many_parameters_and_shape(self):
        model = self.make_model()
        d = solid_box_geometry(batch=2, grid_size=8)
        cs = slab_coefficients(torch.tensor([.2, .7]))
        with patch.object(model, "encode_geometry", wraps=model.encode_geometry) as tracked:
            z = model.cache_geometry(d)
            for delta in (0, .15, -.05):
                r = model.moment(z, cs + delta * torch.tensor([[1.,0,0,0,0,0,0,0,0,0]]))
                j = model.jacobian(z, cs)
                self.assertEqual(tuple(r.shape), (2, 10))
                self.assertEqual(tuple(j.shape), (2, 10, 10))
            self.assertEqual(tracked.call_count, 1)
        self.assertEqual(count_real_parameters(model) > 0, True)
        self.assertFalse(z.z.requires_grad)

    def test_jacobian_vs_finite_difference_and_projective_null(self):
        model = self.make_model()
        z = model.cache_geometry(solid_box_geometry(grid_size=8))
        c = slab_coefficients(torch.tensor([.47]))
        j = model.jacobian(z, c)
        eps = .0005
        bump = torch.zeros_like(c)
        bump[0, 0] = eps
        fd = (model.moment(z, c + bump) - model.moment(z, c - bump)) / (2*eps)
        self.assertTrue(torch.allclose(j[0, :, 0], fd[0], atol=2e-4, rtol=2e-2))
        self.assertTrue(torch.allclose(model.moment(z,c), model.moment(z, 2.5*c), atol=1e-7))
        self.assertLess(float((j @ c.unsqueeze(-1)).abs().max().detach()), 1e-5)
        self.assertFalse(torch.allclose(positive_projective_coefficients(c),
                                        positive_projective_coefficients(-c)))

    def test_geometry_encoder_train_grad(self):
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(20261019)
            model = self.make_model().train()
        # Stay inside the density clamp and away from the occupancy threshold.
        rho = torch.full((1, 1, 8, 8, 8), 0.75, requires_grad=True)
        c = slab_coefficients(torch.tensor([.3]))

        def loss(density):
            z = model.cache_geometry({"density": density}, detach=False)
            return model.moment(z, c).square().sum()

        loss(rho).backward()
        self.assertIsNotNone(rho.grad)
        self.assertIsNotNone(model.encoder.lift.weight.grad)
        self.assertGreater(model.encoder.lift.weight.grad.norm().item(), 0)
        direction = torch.linspace(-1, 1, 8).view(1, 1, 8, 1, 1)
        derivative = (rho.grad * direction).sum()
        with torch.no_grad():
            step = 0.01
            fd = (loss(rho + step * direction) - loss(rho - step * direction)) / (2 * step)
        self.assertGreater(fd.abs().item(), 1e-5)
        torch.testing.assert_close(derivative, fd, rtol=0.02, atol=1e-6)

    def test_invalid_coefficients(self):
        model = self.make_model()
        z = torch.randn(1, 12)
        for c in (torch.zeros(1, 10), torch.full((1, 10), math.nan), torch.ones(1, 9)):
            with self.assertRaises(ValueError):
                model.moment(z, c)

    def test_zero_one_two_teacher_adapters(self):
        model = self.make_model()
        z = model.cache_geometry(solid_box_geometry(grid_size=8))
        support_c = slab_coefficients(torch.tensor([.2, .6]))[None, :, :]
        support_r = slab_moments(torch.tensor([.2, .6]))[None, :, :]
        c = slab_coefficients(torch.tensor([.4]))
        for k in range(3):
            adapter = ResidualAdapter(model, z, support_c[:, :k], support_r[:, :k])
            self.assertEqual(tuple(adapter.predict(c).shape), (1, 10))
            self.assertEqual(tuple(adapter.jacobian(c).shape), (1, 10, 10))
            if k:
                actual = adapter.predict(support_c[:, k-1])
                self.assertTrue(torch.allclose(actual, support_r[:, k-1], atol=2e-6))
            if k == 2:
                for anchor in range(2):
                    self.assertTrue(torch.allclose(adapter.predict(support_c[:, anchor]),
                                                    support_r[:, anchor], atol=2e-6))
                eps = .0005
                dc = torch.zeros_like(c)
                dc[:, 0] = eps
                fd = (adapter.predict(c+dc) - adapter.predict(c-dc))/(2*eps)
                self.assertTrue(torch.allclose(fd[0], adapter.jacobian(c)[0,:,0], atol=3e-4, rtol=2e-2))
        with self.assertRaises(ValueError):
            ResidualAdapter(model, z, support_c[:, :1].repeat(1, 2, 1), support_r)

    def test_analytic_reference_is_independent(self):
        t = torch.tensor([.2, .65], dtype=torch.float64, requires_grad=True)
        r = slab_moments(t)
        derivatives = torch.stack([torch.autograd.grad(r[:, i].sum(), t, retain_graph=True)[0]
                                   for i in range(10)], dim=1)
        self.assertTrue(torch.allclose(derivatives, slab_dr_dt(t), atol=1e-12))
        self.assertEqual(tuple(slab_coefficients(t.detach()).shape), (2, 10))
        sphere = ellipsoid_raw_moments((1., 1., 1.))
        self.assertAlmostEqual(sphere[0], 4*math.pi/3)
        self.assertAlmostEqual(sphere[4], 4*math.pi/15)
        ellipsoid = ellipsoid_raw_moments((2., 1., .5), (1., 2., 3.))
        self.assertAlmostEqual(ellipsoid[7], ellipsoid[0]*6)
        self.assertAlmostEqual(ellipsoid[9], ellipsoid[0]*2)


if __name__ == "__main__":
    unittest.main()
