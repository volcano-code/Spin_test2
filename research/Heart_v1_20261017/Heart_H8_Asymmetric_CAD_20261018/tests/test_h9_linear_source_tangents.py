from fractions import Fraction as F
from pathlib import Path
import sys
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'source'))
from h9_linear_source_tangents import motion
from h9_m21_motion_math import add, dot, mul, sub


class LinearVelocityTests(unittest.TestCase):
    def test_implicit_equation_holds_for_translated_and_scaled_fixed_rays(self):
        # |y - (1,0,0)|²=4. Use three probes exactly on the sphere.
        q=[-3,-2,0,0,1,1,1,0,0,0]
        for col in (1,2):
            dcenter, scale=motion(q,col,F(1),F(4))
            for ray in ((F(2),0,0),(0,F(2),0),(0,0,F(-2))):
                point=add((1,0,0),ray)
                velocity=add(dcenter,mul(scale,ray))
                # d/d b_j f(y,b) = grad_y f . v + y_j = 0.
                self.assertEqual(dot(mul(2,ray),velocity)+point[col-1], 0)
        self.assertEqual(motion(q,2,F(1),F(4)),((0,F(-1,2),0),0))
