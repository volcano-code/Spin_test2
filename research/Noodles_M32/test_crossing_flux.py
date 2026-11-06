from fractions import Fraction as Q
import unittest

from crossing_flux import SCALE,face_flux,integrate


class FluxTest(unittest.TestCase):
    def test_affine_velocity_clipped_to_subtriangle(self):
        # Parent XY unit triangle, retained child with both edge lengths halved.
        pt=lambda x,y:(Q(x),Q(y),Q(0))
        parent=[pt(0,0),pt(1,0),pt(0,1)]
        tri=[pt(0,0),pt(Q(1,2),0),pt(0,Q(1,2))]
        motions=[[[0],[0],[p[0]+p[1]]] for p in parent]
        flux=[r[0] for r in face_flux(tri,parent,motions,True)]
        self.assertEqual(flux[0],Q(1,24))
        self.assertEqual(flux[4],Q(1,480))
        self.assertEqual(flux[9],Q(1,960))
        self.assertEqual([r[0] for r in face_flux(tri[::-1],parent,motions,False)],[-v for v in flux])

    def test_constant_translation_closed_tetra(self):
        pts=[tuple(map(Q,p)) for p in ((0,0,0),(1,0,0),(0,1,0),(0,0,1))]
        faces=[[0,2,1],[0,1,3],[0,3,2],[1,2,3]]
        motion=[[[Q(1)],[Q(0)],[Q(0)]] for _ in pts]
        lo,hi,n=integrate(pts,faces,range(4),pts,faces,motion,0,'intersection')
        self.assertEqual(n,4)
        truth=[0,Q(1,6),0,0,Q(1,12),0,0,0,Q(1,24),Q(1,24)]
        for i,v in enumerate(truth):
            self.assertLessEqual(Q(lo[i][0],SCALE),v)
            self.assertGreaterEqual(Q(hi[i][0],SCALE),v)

    def test_wrong_parent_and_orientation_rejected(self):
        pts=[tuple(map(Q,p)) for p in ((0,0,0),(1,0,0),(0,1,0))]
        m=[[[0],[0],[1]]]*3
        for child in (pts[::-1], [pts[0],pts[1],(Q(0),Q(2),Q(0))]):
            with self.assertRaises(ValueError):
                face_flux(child,pts,m,True)
