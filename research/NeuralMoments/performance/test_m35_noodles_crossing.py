import unittest

from m35_noodles_crossing import Q, parent_patches

FACES=[(0,2,1),(0,1,3),(0,3,2),(1,2,3)]
def tetra(origin,length):
    return [tuple(p+(length if j==i else Q(0)) for j,p in enumerate(origin)) for i in (-1,0,1,2)]

class ExactCrossingTests(unittest.TestCase):
    def inputs(self):
        shift=(Q(1,2),Q(1,4),Q(1,4))
        return tetra([Q(0)]*3,Q(2)),tetra(shift,Q(2)),tetra(shift,Q(1))

    def test_true_intersection_is_part_of_four_distinct_parent_facets(self):
        design,cutter,saved=self.inputs()
        out=parent_patches(design,FACES,cutter,FACES,saved,FACES,[4,5,6,3])
        self.assertEqual(out['partial_design_parents'],1)
        self.assertEqual(out['partial_cutter_parents'],3)
        self.assertEqual(out['exact_seam_loop_lengths'],[3])

    def test_wrong_operation_or_fake_birth_cannot_pass(self):
        design,cutter,saved=self.inputs()
        with self.assertRaises(ValueError):
            parent_patches(design,FACES,cutter,FACES,saved,FACES,[0,5,6,3])
        with self.assertRaises(ValueError):
            parent_patches(design,FACES,cutter,FACES,saved,FACES[:-1],[4,5,6])
        with self.assertRaises(ValueError):
            parent_patches(design,FACES,cutter,FACES,saved,[FACES[0][::-1],*FACES[1:]],[4,5,6,3])

class SeamsAndMomentTests(unittest.TestCase):
    def test_triangle_intersection_segments_change_sign_on_difference(self):
        from m35_general_seams import expected_seams,measured_seams,carriers
        design,cutter,saved=ExactCrossingTests().inputs()
        own=[4,5,6,3]
        ex,_=expected_seams(design,FACES,cutter,FACES,'intersection')
        observed=measured_seams(saved,FACES,own,len(FACES))
        self.assertEqual(ex,observed)
        diff,_=expected_seams(design,FACES,cutter,FACES,'difference')
        self.assertEqual(diff,{key:pair[::-1] for key,pair in ex.items()})
        out=carriers(cutter,FACES,design,FACES,saved,FACES,own,4,1,'intersection',1)
        self.assertEqual(out['partial'],3)

    def test_bounded_tetra_contains_independent_closed_form(self):
        from m35_bounded_ten import integrate
        v=tetra((Q(0),Q(0),Q(0)),Q(1))
        truth=[Q(1,6)]+[Q(1,24)]*3+[Q(1,60)]*3+[Q(1,120)]*3
        for value,(lo,hi) in zip(truth,integrate(v,FACES)):
            self.assertLessEqual(lo,value)
            self.assertLessEqual(value,hi)
            self.assertLessEqual(hi-lo,Q(4,120*2**112))

    def test_assembled_sum_checks_exact_signed_bounds(self):
        from m35_assembled import rational_inside
        self.assertTrue(rational_inside('-0x21/0x10',Q(1,16),Q(-2),Q(-2)))
        self.assertFalse(rational_inside('-0x21/0x10',Q(1,16),Q(-1),Q(0)))
        with self.assertRaises(ValueError):
            rational_inside('1/-1',Q(0),Q(0),Q(2))
