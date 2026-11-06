from fractions import Fraction as Q
import unittest
from crossing_vertex_motion import solve_velocity, tetra_jacobian
from crossing_flux import SCALE


class VelocityTest(unittest.TestCase):
    def test_intersection_of_fixed_edge_and_translating_face(self):
        p=tuple(map(Q,(0,0,0)))
        d=[tuple(map(Q,x)) for x in ((-1,0,0),(1,0,0),(0,1,0))]
        c=[tuple(map(Q,x)) for x in ((0,-1,-1),(0,1,-1),(0,0,1))]
        mv=[[[1],[0],[0]] for _ in c]
        v,k=solve_velocity(p,[0,1],(d,c),([[0,1,2]],[[0,1,2]]),mv,1)
        self.assertEqual(k,'fixed_edge_moving_face')
        self.assertEqual(v,[(1,0,0)])

    def test_differentiated_volume_of_dilating_tetra(self):
        pt=[tuple(map(Q,p)) for p in ((0,0,0),(1,0,0),(0,1,0),(0,0,1))]
        f=[[0,2,1],[0,1,3],[0,3,2],[1,2,3]]
        # Every kth monomial scales by (1+t)^(3+k).
        l,h=tetra_jacobian(pt,f,[[p] for p in pt],1)
        true=[Q(1,2),*[Q(1,6)]*3,*[Q(1,12)]*3,*[Q(1,24)]*3]
        for i,exact in enumerate(true):
            self.assertLessEqual(Q(l[i][0],SCALE),exact)
            self.assertGreaterEqual(Q(h[i][0],SCALE),exact)
