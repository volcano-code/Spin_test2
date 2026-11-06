from fractions import Fraction as Q
import os
from pathlib import Path
import unittest

import numpy as np

from m29_heart_cut_faces import exact_output_points, homogeneous, input_points, load_point0, plane
from m30_heart_cutter_faces import DesignSolid, audit_parent


def check(solid, parent, vertices, faces):
    coords = np.asarray(vertices,dtype=float)[np.asarray(faces,dtype=int).reshape((-1,3))]
    return audit_parent(solid,parent,plane(parent),vertices,list(map(homogeneous,vertices)),
                        faces,coords.min(axis=1),coords.max(axis=1))


class SubtractionTests(unittest.TestCase):
    def setUp(self):
        self.v = input_points([(0,0,0),(4,0,0),(0,4,0),(0,0,4)])
        self.f = [(0,2,1),(0,1,3),(0,3,2),(1,2,3)]
        self.design = DesignSolid(self.v,self.f)
        self.parent = input_points([(1,1,1),(3,1,1),(1,3,1)])
        self.out = input_points([(1,1,1),(1,2,1),(2,1,1)])
        self.face = (0,1,2)

    def test_independent_cell_membership_and_winding(self):
        r = check(self.design,self.parent,self.out,[self.face])
        self.assertTrue(r['passes'])
        self.assertEqual(r['cuts'],1)
        self.assertEqual(r['status'],'partial')

    def test_omission(self):
        self.assertEqual(Q(check(self.design,self.parent,self.out,[])['missing']),Q(1,2))

    def test_extra_and_overlap(self):
        over = check(self.design,self.parent,self.parent,[(2,1,0)])
        self.assertEqual(Q(over['extra']),Q(3,2))
        dup = check(self.design,self.parent,self.out,[self.face,self.face])
        self.assertEqual(Q(dup['overlap']),Q(1,2))

    def test_cutter_output_winding(self):
        with self.assertRaisesRegex(ValueError,'winding'):
            check(self.design,self.parent,self.out,[self.face[::-1]])

    def test_coplanar_positive_area_is_rejected(self):
        with self.assertRaisesRegex(ValueError,'coplanar'):
            check(self.design,self.v[:3],self.out,[])

    def test_oriented_winding_rejects_inverted_and_open_design(self):
        with self.assertRaisesRegex(ValueError,'closed'):
            DesignSolid(self.v,self.f[:-1])
        inverse = DesignSolid(self.v,[f[::-1] for f in self.f])
        with self.assertRaisesRegex(ValueError,'winding'):
            inverse.inside((Q(1),Q(1),Q(1)))

    def test_on_design_boundary_denied(self):
        with self.assertRaisesRegex(ValueError,'boundary|transverse'):
            self.design.inside((Q(1),Q(1),Q(2)))


@unittest.skipUnless(os.environ.get('M30_H8F_ZIP'),'requires pinned H8f point-000 archive')
class HistoricalMutationTests(unittest.TestCase):
    def test_removed_duplicated_and_reversed_real_output_triangle(self):
        request, material = load_point0(Path(os.environ['M30_H8F_ZIP']))
        design = DesignSolid(input_points(request['design_vertices']),request['design_faces'])
        cutter = input_points(material['quadric_source_vertices'])
        tri = [cutter[i] for i in material['quadric_source_faces'][141]]
        vertices = exact_output_points(material)
        faces = material['variable_faces']
        real = check(design,tri,vertices,faces)
        self.assertTrue(real['passes'])
        self.assertEqual(len(real['output_faces']),28)
        i = real['output_faces'][0]
        self.assertGreater(Q(check(design,tri,vertices,faces[:i]+faces[i+1:])['missing']),0)
        self.assertGreater(Q(check(design,tri,vertices,faces+[faces[i]])['overlap']),0)
        with self.assertRaisesRegex(ValueError,'winding'):
            check(design,tri,vertices,faces[:i]+[faces[i][::-1]]+faces[i+1:])
