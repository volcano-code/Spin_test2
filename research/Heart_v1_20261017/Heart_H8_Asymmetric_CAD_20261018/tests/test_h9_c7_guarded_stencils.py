from fractions import Fraction as F
from pathlib import Path
import sys
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'source'))
from h9_c7_guarded_stencils import difference


def point(x, sign=1, same=True):
    t=F(x)
    return dict(x=t, index=x, edge_sign=sign, base_patch=same,
                moments=[3+t+4*t*t]*10, native=[1+8*t]*10)


class GuardedStencilTests(unittest.TestCase):
    def test_one_sided_nonuniform_quadratic(self):
        base=point(0)
        for sign in (-1,1):
            result=difference(base,[base,point(sign),point(3*sign)])
            self.assertEqual(result['max_scaled_error'],0)
            self.assertTrue(result['sampled_stencil_passed'])

    def test_numerically_exact_event_is_rejected(self):
        base=point(0)
        out=difference(base,[base,point(1),point(2,sign=-1,same=False)])
        self.assertTrue(out['numeric_within_tolerance'])
        self.assertFalse(out['sampled_stencil_passed'])

    def test_no_stencil_with_only_one_postevent_sample(self):
        side=point(3,sign=-1,same=False)
        with self.assertRaises(ValueError):
            difference(side,[side])
