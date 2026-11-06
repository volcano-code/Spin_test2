import json
from fractions import Fraction as Q
import unittest
from p25_axis_capacity import HERE, first_moment, replay


class CapacityTest(unittest.TestCase):
    def test_unit_tetra_cut_through_interior(self):
        o, a, b, c = [(Q(x),Q(y),Q(z)) for x,y,z in ((0,0,0),(1,0,0),(0,1,0),(0,0,1))]
        surface = ((a,b,c),(o,b,a),(o,c,b),(o,a,c))
        for axis in (0,1):
            self.assertEqual(first_moment(surface,axis,Q(1,2),1), Q(1,384))
            self.assertEqual(first_moment(surface,axis,Q(1,2),-1), -Q(17,384))
            self.assertEqual(first_moment(surface,axis,Q(0),-1), 0)
            self.assertEqual(first_moment(surface,axis,Q(0),1), Q(1,24))

    def test_frozen_material_and_exact_receipt(self):
        result = replay()
        self.assertEqual(result, json.loads((HERE/'P25_AXIS_CAPACITY.json').read_text()))
        for tip in ('requested','represented'):
            x = result['results']['x'][tip]
            self.assertTrue(0 < Q(x['conservative_total_min']['exact']) <= Q(x['total_min']['exact']))
            self.assertFalse(x['zero_in_projection'])
            self.assertTrue(result['results']['y'][tip]['zero_in_projection'])
