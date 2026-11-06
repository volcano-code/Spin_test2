import unittest
import numpy as np
import torch

from research.NeuralMoments.models.pcraft_ufno.spin_pino.half_ellipsoid import half_truth
from research.NeuralMoments.models.pcraft_ufno.spin_pino.refined_quadrature import cartesian, ellipsoid, localized


class RefinedCutTest(unittest.TestCase):
    def test_half_ball_mass_center_and_ambient_derivatives(self):
        axes=np.array([.1,.08,.16])
        center=np.array([.4,-.25,.375])
        q,r,J=half_truth(center,axes)
        mass=2*np.pi/3*np.prod(axes)
        np.testing.assert_allclose(r[0],mass,rtol=1e-14)
        np.testing.assert_allclose(r[1:4]/mass,center+[-3*axes[0]/8,0,0],atol=1e-14)
        # Increasing only the constant uniformly enlarges the half-ellipsoid and keeps center fixed.
        k=ellipsoid(q)[-1]
        np.testing.assert_allclose(J[0,0],1.5*mass/k,rtol=1e-14)
        np.testing.assert_allclose(J@q,0,atol=2e-14)

    def test_moving_ellipsoid_constant_density_all_native_columns(self):
        q=np.array([.04,.02,-.03,.05,-1.3,-1.4,-.78,.11,-.15,.19])
        # Grid interpolation as well as all quadrature weights are deterministic.
        v=torch.ones((1,1,8,8,8),dtype=torch.float64)
        gates=np.ones((6,6,6))
        r,J,kind=localized(v,gates,q)
        self.assertEqual(kind,"ellipsoid_coarea")
        _,L,_=ellipsoid(q)
        self.assertAlmostEqual(r[0],np.linalg.det(L)*4*np.pi/3,places=13)
        h=1e-6
        fd=np.column_stack([(localized(v,gates,q+h*d)[0]-localized(v,gates,q-h*d)[0])/(2*h) for d in np.eye(10)])
        np.testing.assert_allclose(J,fd,rtol=5e-7,atol=3e-10)

    def test_hyperboloid_fallback_and_homogeneous_scaling(self):
        v=torch.ones((1,1,8,8,8));gates=np.ones((6,6,6))
        q=np.array([.1,0,.12,0,-.7,1.,-.6,0,0,0])
        r,J,which=localized(v,gates,q)
        self.assertEqual(which,"cartesian_fallback")
        expected_r,expected_J=cartesian(v,gates,2*q,24,.0075)
        np.testing.assert_allclose(r,expected_r)
        np.testing.assert_allclose(J,2*expected_J)
        np.testing.assert_allclose(J@q,0,atol=6e-15)

    def test_paired_frozen_scores_cover_all_registered_arms(self):
        import hashlib
        import json
        from pathlib import Path

        base = Path(__file__).resolve().parents[1]
        rec = json.loads((base/"M36_REFINEMENT.json").read_bytes())
        for name, key in [("M36_PROTOCOL", "protocol_sha256"), ("M36_BUNNY_INPUT", "heldout_real_sha256"),
                          ("M35_REAL_INPUT", "M35_input_sha256")]:
            self.assertEqual(hashlib.sha256((base/(name+".json")).read_bytes()).hexdigest(), rec[key])
        self.assertEqual([s["seed"] for s in rec["learned_seeds"]], [131,137,139])
        self.assertEqual(len(rec["inputs"]),14)
        def keys(rows):
            return [(r["group"], r["case"], r["arm"]) for r in rows]
        paired = keys(rec["numerical"])
        self.assertEqual(len(set(paired)), 14*5)
        for seed in rec["learned_seeds"]:
            self.assertEqual(keys(seed["scores"]), paired)
            # Positive geometry gates lie in (0,1), so the paired numerical mass bounds their mass.
            for trained, unit in zip(seed["scores"],rec["numerical"]):
                self.assertLessEqual(trained["mass"],unit["mass"]+2e-8)
