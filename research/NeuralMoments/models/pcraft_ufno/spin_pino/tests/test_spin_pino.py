import unittest

import torch

from research.NeuralMoments.models.pcraft_ufno.spin_pino import (
    PINOFeatureEncoder,
    SpinPINO,
    SpinPINOLoss,
)


class SpinPINOBackboneTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(20261106)
        self.geometry = torch.rand((2, 4, 8, 8, 8), dtype=torch.float32)
        self.coefficients = torch.randn((2, 10), dtype=torch.float32)

    def make_model(self):
        return SpinPINO(
            width=4,
            modes=2,
            n_layers=2,
            grid_size=8,
            points_per_axis=3,
            hidden_dim=24,
            smoothing=0.04,
        )

    def test_01_pino_ufno_encoder_returns_geometry_latent(self):
        encoder = PINOFeatureEncoder(width=4, modes=2, n_layers=2, grid_size=8)
        features = encoder(self.geometry)
        self.assertEqual(features.shape, (2, 24))
        self.assertTrue(bool(torch.isfinite(features).all()))

    def test_02_spin_pino_predicts_ten_raw_moments_and_jacobian(self):
        model = self.make_model()
        r, J = model(self.geometry, self.coefficients, with_jacobian=True)
        self.assertEqual(r.shape, (2, 10))
        self.assertEqual(J.shape, (2, 10, 10))
        self.assertTrue(bool(torch.isfinite(r).all() and torch.isfinite(J).all()))

    def test_03_positive_projective_scale_invariance(self):
        model = self.make_model()
        z = model.encode_geometry(self.geometry)
        r1 = model.moment_from_weights(z, self.coefficients)
        r2 = model.moment_from_weights(z, 7.0 * self.coefficients)
        self.assertTrue(torch.allclose(r1, r2, atol=2e-6, rtol=2e-6))

    def test_04_complement_identity(self):
        model = self.make_model()
        z = model.encode_geometry(self.geometry)
        r_pos = model.moment_from_weights(z, self.coefficients)
        r_neg = model.moment_from_weights(z, -self.coefficients)
        full = z @ model.monomials
        self.assertTrue(torch.allclose(r_pos + r_neg, full, atol=2e-5, rtol=2e-5))

    def test_05_jacobian_is_symmetric_psd_in_monomial_coordinates(self):
        model = self.make_model()
        z = model.encode_geometry(self.geometry)
        Jc = model.jacobian_from_weights(z, self.coefficients)
        B = model.native_B
        Jq = Jc / B.view(1, 1, 10)
        self.assertTrue(torch.allclose(Jq, Jq.transpose(1, 2), atol=2e-5, rtol=2e-5))
        eigenvalues = torch.linalg.eigvalsh(0.5 * (Jq + Jq.transpose(1, 2)))
        self.assertGreaterEqual(float(eigenvalues.min()), -2e-5)

    def test_06_radial_jacobian_identity(self):
        model = self.make_model()
        z = model.encode_geometry(self.geometry)
        J = model.jacobian_from_weights(z, self.coefficients)
        radial = torch.bmm(J, self.coefficients.unsqueeze(-1))
        self.assertLess(float(radial.abs().max()), 3e-5)

    def test_07_pino_physics_loss_backpropagates(self):
        model = self.make_model()
        z = model.encode_geometry(self.geometry)
        r = model.moment_from_weights(z, self.coefficients)
        target = r.detach() + 0.01
        loss_fn = SpinPINOLoss(lambda_data=1.0, lambda_physics=0.2)
        losses = loss_fn(model, z, self.coefficients, target_moments=target)
        self.assertTrue(bool(torch.isfinite(losses["total"])))
        losses["total"].backward()
        self.assertIsNotNone(model.weight_head[-1].weight.grad)
        self.assertTrue(bool(torch.isfinite(model.weight_head[-1].weight.grad).all()))

    def test_08_diagnostic_jacobian_cannot_enter_training(self):
        model = self.make_model()
        z = model.encode_geometry(self.geometry)
        J = model.jacobian_from_weights(z, self.coefficients).detach()
        loss_fn = SpinPINOLoss(lambda_jacobian=1.0)
        with self.assertRaises(ValueError):
            loss_fn(model, z, self.coefficients, target_jacobian=J)

    def test_10_quadrature_weights_are_zero_outside_material_support(self):
        model = self.make_model()
        empty = torch.zeros((1, 4, 8, 8, 8))
        weights = model.encode_geometry(empty)
        self.assertTrue(torch.equal(weights, torch.zeros_like(weights)))
        material = torch.zeros((1, 4, 8, 8, 8))
        material[:, 0, 2:6, 2:6, 2:6] = 1.0
        material[:, 1, 2:6, 2:6, 2:6] = 1.0
        weights = model.encode_geometry(material)
        self.assertTrue(bool((weights >= 0).all()))
        self.assertTrue(bool((weights <= model.node_volume + 1e-7).all()))
        self.assertGreater(float(weights.sum()), 0.0)

    def test_12_gauss_rule_is_positive_and_volume_exact(self):
        model = SpinPINO(
            width=4, modes=2, n_layers=2, grid_size=8,
            points_per_axis=4, hidden_dim=24, smoothing=0.02,
            quadrature_rule="gauss",
        )
        self.assertTrue(bool((model.node_volume > 0).all()))
        self.assertAlmostEqual(float(model.node_volume.sum()), 8.0, places=6)
        self.assertEqual(model.quadrature_xyz.shape, (64, 3))

    def test_14_logistic_atom_head_complement_radial_and_tangent_psd(self):
        model = SpinPINO(
            width=4, modes=2, n_layers=2, grid_size=8,
            points_per_axis=4, hidden_dim=24, smoothing=0.02,
            moment_parameterization="logistic_atom", temperature=0.12,
        )
        weights = torch.rand((2, 64), dtype=torch.float64)
        coeff = torch.randn((2, 10), dtype=torch.float64)
        r = model.moment_from_weights(weights, coeff)
        r_neg = model.moment_from_weights(weights, -coeff)
        full = weights @ model.monomials.to(torch.float64)
        self.assertTrue(torch.allclose(r + r_neg, full, atol=1e-10, rtol=1e-10))
        J = model.jacobian_from_weights(weights, coeff)
        radial = torch.bmm(J, coeff.unsqueeze(-1))
        self.assertLess(float(radial.abs().max()), 1e-10)
        gram = model.projected_q_gram(weights, coeff)
        self.assertTrue(torch.allclose(gram, gram.transpose(1, 2), atol=1e-10, rtol=1e-10))
        eig = torch.linalg.eigvalsh(0.5 * (gram + gram.transpose(1, 2)))
        self.assertGreaterEqual(float(eig.min()), -1e-10)

    def test_15_logistic_atom_jacobian_matches_finite_difference(self):
        model = SpinPINO(
            width=4, modes=2, n_layers=2, grid_size=8,
            points_per_axis=4, hidden_dim=24, smoothing=0.02,
            moment_parameterization="logistic_atom", temperature=0.12,
        )
        weights = torch.rand((1, 64), dtype=torch.float64)
        coeff = torch.randn((1, 10), dtype=torch.float64)
        analytic = model.jacobian_from_weights(weights, coeff)
        step = 1e-6
        columns = []
        for j in range(10):
            delta = torch.zeros_like(coeff)
            delta[:, j] = step
            plus = model.moment_from_weights(weights, coeff + delta)
            minus = model.moment_from_weights(weights, coeff - delta)
            columns.append(((plus - minus) / (2 * step)).squeeze(0))
        finite_difference = torch.stack(columns, dim=-1).unsqueeze(0)
        self.assertTrue(torch.allclose(analytic, finite_difference, atol=2e-7, rtol=2e-6))

    def test_13_voxel_sampling_respects_world_xyz_axis_order(self):
        model = SpinPINO(
            width=4, modes=2, n_layers=2, grid_size=8,
            points_per_axis=4, hidden_dim=24, smoothing=0.02,
            quadrature_rule="gauss",
        )
        geometry = torch.zeros((1, 4, 8, 8, 8))
        # Material occupies only the positive-X half of [X,Y,Z].
        geometry[:, 0, 4:8, :, :] = 1.0
        geometry[:, 1, 4:8, :, :] = 1.0
        weights = model.encode_geometry(geometry)[0]
        positive_x = model.quadrature_xyz[:, 0] > 0
        self.assertGreater(float(weights[positive_x].sum()), 0.0)
        self.assertTrue(torch.equal(weights[~positive_x], torch.zeros_like(weights[~positive_x])))

    def test_11_density_scaling_scales_discrete_measure(self):
        model = self.make_model()
        # Hold the learned gate constant so this isolates the explicit
        # rho(x_k) factor; the real geometry-conditioned gate may also change
        # when the density field changes.
        for parameter in model.weight_head.parameters():
            torch.nn.init.zeros_(parameter)
        geometry = torch.zeros((1, 4, 8, 8, 8))
        geometry[:, 0, 1:7, 1:7, 1:7] = 1.0
        geometry[:, 1, 1:7, 1:7, 1:7] = 1.0
        z1 = model.encode_geometry(geometry)
        geometry[:, 0] *= 0.25
        z2 = model.encode_geometry(geometry)
        self.assertTrue(torch.allclose(z2, 0.25 * z1, atol=1e-7, rtol=1e-6))

    def test_09_admitted_jacobian_supervision_is_explicit(self):
        model = self.make_model()
        z = model.encode_geometry(self.geometry)
        J = model.jacobian_from_weights(z, self.coefficients).detach()
        loss_fn = SpinPINOLoss(lambda_jacobian=1.0)
        losses = loss_fn(
            model, z, self.coefficients,
            target_jacobian=J, derivative_admitted=True
        )
        self.assertLess(float(losses["jacobian_supervision"]), 1e-10)


if __name__ == "__main__":
    unittest.main()
