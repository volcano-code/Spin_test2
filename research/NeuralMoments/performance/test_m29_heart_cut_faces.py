from fractions import Fraction as Q
import unittest

import numpy as np

from m29_heart_cut_faces import audit_cut, convex_planes, cut_polygon, homogeneous, verify_whole


def pts(rows):
    return [tuple(map(Q, row)) for row in rows]


class ExactDesignCutTest(unittest.TestCase):
    def setUp(self):
        self.cutter = pts([(0,0,0), (2,0,0), (0,2,0), (0,0,2)])
        self.faces = [(0,2,1),(0,1,3),(0,3,2),(1,2,3)]
        self.planes = convex_planes(self.cutter, self.faces)
        # The oblique tetrahedron plane x+y+z=2 slices this parent at x+y=1.
        self.tri = tuple(pts([(0,0,1), (2,0,1), (0,2,1)]))
        self.out = pts([(1,0,1), (2,0,1), (0,2,1), (0,1,1)])
        self.saved = [(0,1,2), (0,2,3)]

    def run_audit(self, faces):
        h = tuple(map(homogeneous,self.tri))
        poly = cut_polygon(self.tri,h,self.planes)
        coords = np.asarray(self.out,dtype='float64')[np.asarray(faces)]
        return audit_cut(self.tri,h,poly,self.planes,self.out,list(map(homogeneous,self.out)),
                         faces,coords.min(axis=1),coords.max(axis=1))

    def test_whole_retained_and_removed(self):
        parent = tuple(pts([(0,0),(2,0),(0,2)]))
        self.assertTrue(verify_whole(parent, [parent], True))
        self.assertFalse(verify_whole(parent, [parent, parent], True))
        self.assertFalse(verify_whole(parent, [], True))
        self.assertTrue(verify_whole(parent, [], False))
        self.assertFalse(verify_whole(parent, [parent], False))

    def test_whole_retained_subdivided(self):
        a,b,c = parent = tuple(pts([(0,0),(2,0),(0,2)]))
        d = (Q(1),Q(1))
        self.assertTrue(verify_whole(parent, [(a,b,d),(a,d,c)], True))
        self.assertFalse(verify_whole(parent, [(a,b,d),(a,b,d)], True))

    def test_oblique_cut(self):
        r = self.run_audit(self.saved)
        self.assertTrue(r['passes'])
        self.assertEqual(r['exact_projected_cutter_area'], '1/2')

    def test_omission(self):
        r = self.run_audit(self.saved[:1])
        self.assertFalse(r['passes'])
        self.assertEqual(Q(r['missing']), Q(1,2))

    def test_duplicate_triangle(self):
        r = self.run_audit(self.saved + self.saved[:1])
        self.assertFalse(r['passes'])
        self.assertGreater(Q(r['overlap']), 0)

    def test_extra_coverage(self):
        self.out.append(self.tri[0])
        r = self.run_audit(self.saved + [(4,0,3)])
        self.assertEqual(Q(r['extra']), Q(1,2))

    def test_wrong_winding(self):
        with self.assertRaisesRegex(ValueError, 'winding'):
            self.run_audit([self.saved[0][::-1], self.saved[1]])

    def test_cutter_open_or_inward(self):
        for bad in (self.faces[:-1], [f[::-1] for f in self.faces]):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                convex_planes(self.cutter,bad)

    def test_tangent_excluded_from_positive_area_cut_count(self):
        tri = tuple(pts([(0,0,2),(1,0,2),(0,1,2)]))
        self.assertEqual(cut_polygon(tri,tuple(map(homogeneous,tri)),self.planes),[])
