import unittest
from fractions import Fraction as F

from m34_noodles_partition import interval,partition_chain,regular_orientation
from m34_noodles_cuboid_witness import box_disjoint, box_ten,solve_four
from m34_noodles_symmetric_carve import inside_cells


def tetra(shift=0):
    points=[(F(shift+x),F(y),F(z)) for x,y,z in ((0,0,0),(1,0,0),(0,1,0),(0,0,1))]
    return points,[(0,2,1),(0,1,3),(0,3,2),(1,2,3)]


class M34Tests(unittest.TestCase):
    def test_half_clip_integral_crossing(self):
        v,f=tetra(F(-1,2))
        lo,hi=interval(v,f,0)
        self.assertEqual(hi,F(1,384))
        self.assertEqual(lo+hi,F(-1,24))
        self.assertEqual(interval(v,f,1),(0,F(1,24)))

    def test_chain_even_with_changed_diagonal(self):
        v=[('0','0','0'),('1','0','0'),('1','1','0'),('0','1','0')]
        self.assertTrue(partition_chain([(v,[(0,1,2),(0,2,3)],1),(v,[(0,1,3),(1,2,3)],-1)])['oriented_boundary_chain_additive'])
        self.assertFalse(partition_chain([(v,[(0,1,2)],1)])['oriented_boundary_chain_additive'])

    def test_regular_shell_orientation(self):
        v,f=tetra()
        self.assertEqual(regular_orientation(v,f),[4])
        with self.assertRaisesRegex(ValueError,'BOUNDARY_WINDING'):
            regular_orientation(v,[i[::-1] for i in f])

    def test_nested_negative_cavity(self):
        v,f=tetra()
        inside=[tuple(x/4+F(1,16) for x in p) for p in v]
        negative=[tuple(i+4 for i in t[::-1]) for t in f]
        self.assertEqual(regular_orientation(v+inside,f+negative),[4,4])
        with self.assertRaisesRegex(ValueError,'BOUNDARY_WINDING'):
            regular_orientation(v+inside,f+[t[::-1] for t in negative])

    def test_box_separation_not_implied_by_center(self):
        tri=[(F(0),F(0),F(0)),(F(1),F(0),F(0)),(F(0),F(1),F(0))]
        self.assertTrue(box_disjoint(tri,[F(3,4),F(3,4),0],F(1,16)))
        self.assertFalse(box_disjoint(tri,[0,0,F(1,16)],F(1,16)))

    def test_four_constraints_and_box_inertia(self):
        ident=[[int(i==j)*F(1) for j in range(4)] for i in range(4)]
        rhs=[F(i,17) for i in range(1,5)]
        self.assertEqual(solve_four(ident,rhs),rhs)
        raw=box_ten((F(1,8),F(1,4),F(3,4)),F(1,32768))
        self.assertEqual(raw[8],F(3,1048576))
        self.assertGreater(raw[4]+raw[5],raw[0]*(F(1,64)+F(1,16)))

    def test_lattice_membership(self):
        v=[(F(x),F(y),F(z)) for x,y,z in ((0,0,0),(1,0,0),(1,1,0),(0,1,0),
                                              (0,0,1),(1,0,1),(1,1,1),(0,1,1))]
        quads=[(0,3,2,1),(4,5,6,7),(0,1,5,4),(2,3,7,6),(1,2,6,5),(0,4,7,3)]
        f=[t for a,b,c,d in quads for t in ((a,b,c),(a,c,d))]
        cells,n=inside_cells(v,f)
        self.assertIn((3,3,16),cells)
        self.assertNotIn((-3,3,16),cells)
        self.assertGreaterEqual(n,2)
