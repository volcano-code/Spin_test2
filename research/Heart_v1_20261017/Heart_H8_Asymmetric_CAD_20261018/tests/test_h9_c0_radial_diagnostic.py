from fractions import Fraction as F
from pathlib import Path
import sys
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'source'))
from h9_c0_radial_diagnostic import sphere


class RadialSourceTests(unittest.TestCase):
    def setUp(self):
        self.request={'target_axes':[[1,0,0],[0,1,0],[0,0,1]],'target_origin':[1,2,3]}
        self.q=[-7,-4,0,0,2,2,2,0,0,0]

    def test_translated_sphere_and_radius_derivative(self):
        center,r2,a=sphere(self.q,self.request)
        self.assertEqual(center,(2,2,3))
        self.assertEqual(r2,F(9,2))
        shifted=list(self.q);shifted[0]+=F(1,4)
        self.assertEqual(sphere(shifted,self.request)[1]-r2,-F(1,4)/a)

    def test_anisotropy_rotation_and_nonpositive_radius_rejected(self):
        for i,value in ((4,1),(7,1),(0,20)):
            q=list(self.q);q[i]=value
            with self.assertRaises(ValueError):sphere(q,self.request)
        with self.assertRaises(ValueError):
            sphere(self.q,dict(self.request,target_axes=[[0,1,0],[1,0,0],[0,0,1]]))
