from fractions import Fraction as Q
import math
import unittest

import numpy as np

from ellipsoid_local import ellipsoid, jacobian, surface_j, values


class EllipsoidLocalTests(unittest.TestCase):
    def test_unit_ball_and_sign_of_constant_retained_derivative(self):
        q=np.array(list(map(Q,[-1,0,0,0,1,1,1,0,0,0])),dtype=object)
        np.testing.assert_allclose(values(q),4*math.pi/3*np.array([1,0,0,0,.2,.2,.2,0,0,0]))
        self.assertAlmostEqual(jacobian(q)[0,0],2*math.pi)
        self.assertAlmostEqual(jacobian(q)[7,7],4*math.pi/15)
        np.testing.assert_allclose(jacobian(q),surface_j(q),rtol=1e-12,atol=1e-14)
        np.testing.assert_allclose(jacobian(q)@q.astype(float),np.zeros(10),atol=1e-14)

    def test_empty_or_indefinite_cuts_rejected(self):
        for a,xx in [(1,1),(-1,-1)]:
            with self.assertRaises(ValueError):
                ellipsoid(np.array(list(map(Q,[a,0,0,0,xx,1,1,0,0,0])),dtype=object))
