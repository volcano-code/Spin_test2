import unittest
import numpy as np
from m31_candidate_contact_screen import point_surface_distance,voxel_facts

class CandidateScreen(unittest.TestCase):
    def test_support_plane_origin_can_miss_face(self):
        triangle=np.array([[[0.,0.,0.],[1.,0.,0.],[0.,1.,0.]]])
        self.assertEqual(point_surface_distance(np.array([2.,0.,0.]),triangle),1.)
        self.assertEqual(point_surface_distance(np.array([.2,.2,0.]),triangle),0.)

    def test_corner_contact_breaks_clearance(self):
        asset=dict(dimensions=[3,3,3],minimum=[0,0,0],maximum=[3,3,3],
                   material_rle=[[1,27]],carveable_rle=[[0,13],[1,1],[0,13]])
        self.assertEqual(voxel_facts(asset)['minimum_clearance_lower_bound'],1.)
        asset['material_rle']=[[0,1],[1,26]]
        self.assertEqual(voxel_facts(asset)['design_cells_touching_exterior'],1)

if __name__=='__main__':
    unittest.main()
