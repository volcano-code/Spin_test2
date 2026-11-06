import unittest
from fractions import Fraction as F

from m30_terminal_surface import signed_components, winding


def tetra(origin,size,reverse=False):
    vertices=[(origin,origin,origin)]+[tuple(origin+(size if i==j else 0) for j in range(3)) for i in range(3)]
    faces=[(0,2,1),(0,1,3),(0,3,2),(1,2,3)]
    return vertices,[f[::-1] if reverse else f for f in faces]


def assemble(*shells):
    v=[]; f=[]
    for vertices,faces in shells:
        n=len(v); f.extend([tuple(n+i for i in tri) for tri in faces]);v.extend(vertices)
    return v,f


class Nesting(unittest.TestCase):
    def test_closed_cavity_and_material_island(self):
        v,f=assemble(tetra(F(0),F(10)),tetra(F(1),F(5),True),tetra(F(2),F(1)))
        shells,volumes=signed_components(v,f)
        self.assertEqual([s['containing_shells'] for s in shells],[[],[0],[0,1]])
        self.assertEqual(sum(volumes),F(1000-125+1,6))

    def test_positive_cavity_rejected(self):
        with self.assertRaisesRegex(ValueError,'WRONG_ORIENTATION'):
            signed_components(*assemble(tetra(F(0),F(10)),tetra(F(1),F(2))))

    def test_two_separate_materials(self):
        shells,_=signed_components(*assemble(tetra(F(0),F(2)),tetra(F(4),F(1))))
        self.assertEqual([s['nesting_depth'] for s in shells],[0,0])

    def test_nesting_evaluates_point_location_not_bbox_inclusion(self):
        v,faces=tetra(F(0),F(1))
        triangles=[tuple(v[i] for i in face) for face in faces]
        self.assertEqual(winding((F(1,10),)*3,triangles),1)
        self.assertEqual(winding((F(4,5),)*3,triangles),0)
        self.assertIsNone(winding((F(0),)*3,triangles))


if __name__=='__main__':
    unittest.main()
