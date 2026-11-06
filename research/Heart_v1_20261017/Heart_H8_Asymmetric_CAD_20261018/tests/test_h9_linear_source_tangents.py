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
        for col in (1,2,3):
            dcenter, scale=motion(q,col,F(1),F(4))
            for ray in ((F(2),0,0),(0,F(2),0),(0,0,F(-2))):
                point=add((1,0,0),ray)
                velocity=add(dcenter,mul(scale,ray))
                # d/d b_j f(y,b) = grad_y f . v + y_j = 0.
                self.assertEqual(dot(mul(2,ray),velocity)+point[col-1], 0)
        self.assertEqual(motion(q,2,F(1),F(4)),((0,F(-1,2),0),0))

    def test_z_translation_and_nonzero_radius_change(self):
        # |y-(0,0,1)|² = 4 with a nonzero native b_z.
        q=[-3,0,0,-2,1,1,1,0,0,0]
        dcenter, scale=motion(q,3,F(1),F(4))
        self.assertEqual((dcenter,scale), ((0,0,F(-1,2)),F(-1,8)))
        ray=(F(0),F(0),F(2))
        self.assertEqual(dot(mul(2,ray),add(dcenter,mul(scale,ray)))+3,0)

    def test_linear_formula_rejects_quadratic_coefficient(self):
        with self.assertRaisesRegex(ValueError,'EXPECTED_LINEAR_COLUMN'):
            motion([-3,0,0,-2,1,1,1,0,0,0],4,F(1),F(4))
