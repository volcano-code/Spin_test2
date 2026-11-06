from fractions import Fraction as Q
import unittest

import numpy as np
from crossing_flux import SCALE
from crossing_report import check_source_motion,shifted_bounds
from crossing_symmetry import tangent_skew


class ReadbackTests(unittest.TestCase):
    def test_known_tangent_mismatch(self):
        p=np.eye(10)[0];a=np.zeros((10,10),dtype=int);a[2,1]=2
        raw=[[SCALE*int(v) for v in row] for row in a]
        data=dict(q=p.tolist(),lower_tip=raw,upper_tip=raw)
        ratio=Q(tangent_skew(data)['certified_relative_distance_to_any_symmetric_tangent_head_at_least'])
        self.assertGreater(ratio,Q(7071,10000))
        self.assertLessEqual(2*ratio**2,1)

    def test_tip_shift_on_dirac_mass(self):
        d=[[0] for _ in range(10)];d[0][0]=SCALE
        lo,hi=shifted_bounds(d,d)
        for i,v in ((0,Q(1)),(3,Q(.16)),(6,Q(.16)**2)):
            self.assertLessEqual(Q(lo[i][0],SCALE),v)
            self.assertGreaterEqual(Q(hi[i][0],SCALE),v)

    def test_inverse_gauss_source_rejects_false_motion(self):
        p=[1,0,0]
        src=dict(q=[-1,0,0,0,1,1,1,0,0,0],vertices_tip=[p],normals=[p],
                 native_jacobian=np.zeros((1,3,10)).tolist())
        with self.assertRaisesRegex(ValueError,'motion'):
            check_source_motion(src)
