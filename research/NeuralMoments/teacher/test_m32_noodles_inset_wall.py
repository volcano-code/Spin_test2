from fractions import Fraction as F
import unittest
from m32_noodles_inset_wall import decimal_interval_sqrt,supporting_wall


class InsetWall(unittest.TestCase):
    vertices=[(0,0,0),(1,0,0),(0,1,0),(0,0,1)]
    faces=[(0,2,1),(0,1,3),(0,3,2),(1,2,3)]

    def test_tetra_oblique_limiting_face(self):
        squared,index=supporting_wall(self.vertices,self.faces,(F(1,4),)*3,F(9,10))
        self.assertEqual((squared,index),(F(1,4800),3))
        lo,hi=map(F,decimal_interval_sqrt(squared))
        self.assertLessEqual(lo*lo,squared)
        self.assertGreater(hi*hi,squared)

    def test_invalid_center_and_orientation(self):
        with self.assertRaisesRegex(ValueError,'CENTER_NOT_INTERIOR'):
            supporting_wall(self.vertices,self.faces,(0,0,0),F(9,10))
        with self.assertRaisesRegex(ValueError,'OPEN_OR_INCONSISTENT'):
            supporting_wall(self.vertices,self.faces[:-1],(F(1,4),)*3,F(9,10))


if __name__=='__main__':
    unittest.main()
