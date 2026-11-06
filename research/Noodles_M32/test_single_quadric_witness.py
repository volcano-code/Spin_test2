from fractions import Fraction as Q
import json
from pathlib import Path
import unittest

from single_quadric_witness import I, X, Z, H, dot, parameters, pi_interval, winding


class WitnessReferences(unittest.TestCase):
    def test_inertia_and_moment_identity(self):
        # Manufacture exact first/mixed moments for the shape formulas.
        V,Y,sx,sy = map(Q,("1/4000","3/100","11/100","-1/20"))
        raw = list(map(I.exact, (1,V*X,V*Y,1,1,1,2,V*(Y*Z+sy*H/5),V*(X*Z+sx*H/5),0)))
        c,B,m,_,_ = parameters(raw)
        self.assertEqual(m,I.exact(V))
        self.assertEqual(c[1],I.exact(Y))
        self.assertEqual(B[0][2],I.exact(sx))
        self.assertEqual(B[1][2],I.exact(sy))
        for i,k in [(0,8),(1,7)]:
            self.assertEqual(m*(c[i]*c[2]+dot(B[i],B[2])/5),raw[k])
        pi=pi_interval()
        self.assertLess(Q('3.14159265358979323846264338327'),pi.lo)
        self.assertLess(pi.hi,Q('3.14159265358979323846264338329'))

    def test_signed_ray_inside_and_outside(self):
        pts=[(0,0,0),(1,0,0),(0,1,0),(0,0,1)]
        mesh=[(0,2,1),(0,1,3),(0,3,2),(1,2,3)]
        self.assertEqual(winding(pts,mesh,(Q(1,5),Q(1,4),Q(1,8)))[0],1)
        self.assertEqual(winding(pts,mesh,(Q(1,5),Q(1,4),Q(2)))[0],0)

    def test_receipt_scope_and_coverage(self):
        root=Path(__file__).parent
        doc=json.loads((root/"M32_SINGLE_ELLIPSOID.json").read_text())
        self.assertEqual(doc["faces"],doc["exact_aabb_separated"]+doc["rational_separators"])
        self.assertEqual(doc["rational_separators"],len((root/"M32_ELLIPSOID_SEPARATORS.txt").read_text().splitlines()))
        self.assertGreater(Q(doc["min_support_ratio_squared_floor"]),1)
        self.assertGreater(Q(doc["axis_inertia_ratio_interval"][0]),Q(9,10))
        self.assertGreater(Q(doc["centroidal_dominant_Izz_gap_lower"]),0)
        self.assertFalse(doc["strict_newton_terminal"])
