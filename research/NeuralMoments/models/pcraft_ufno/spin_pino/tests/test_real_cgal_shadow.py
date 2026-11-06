import unittest

import numpy as np
import torch

from research.NeuralMoments.models.pcraft_ufno.spin_pino.real_cgal_shadow import translation
from research.NeuralMoments.models.pcraft_ufno.spin_pino.spin_pino import NATIVE_B
from research.NeuralMoments.models.pcraft_ufno.spin_pino.tournament import _build


def monomials(x):
    a,b,c=x
    return np.array([1,a,b,c,a*a,b*b,c*c,b*c,a*c,a*b])


class RealShadowPort(unittest.TestCase):
    def test_translated_polynomial_and_moments(self):
        center=np.array([3.,-2.,1.])
        y=np.array([.25,.5,-.75])
        c=np.arange(10)-3.
        S,T=translation(center)
        np.testing.assert_array_equal(S@monomials(y),monomials(y+center))
        np.testing.assert_array_equal((T@c*NATIVE_B)@monomials(y),(c*NATIVE_B)@monomials(y+center))

    def test_model_jacobian_after_tip_grid_transport(self):
        torch.manual_seed(7)
        model=_build('spin_pino_6node_logistic').eval().double()
        # Precomputed geometry weights isolate the full translated query map and all ten columns.
        w=torch.rand(1,216,dtype=torch.float64)
        S,T=(torch.from_numpy(x) for x in translation([0,0,1]))
        tip=torch.tensor([.43,-.17,.29,.93,-.08,.1,-.31,.12,.03,-.37],dtype=w.dtype)
        def query(c):
            return S@model.moment_from_weights(w,(T@c)[None])[0]
        J=S@model.jacobian_from_weights(w,(T@tip)[None])[0]@T
        ad=torch.autograd.functional.jacobian(query,tip)
        torch.testing.assert_close(J,ad,rtol=1e-12,atol=1e-12)
        eye=torch.eye(10,dtype=w.dtype)
        h=2e-6
        fd=torch.stack([(query(tip+h*d)-query(tip-h*d))/(2*h) for d in eye],dim=1)
        torch.testing.assert_close(J,fd,rtol=1e-7,atol=2e-8)
