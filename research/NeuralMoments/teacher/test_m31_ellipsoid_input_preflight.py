import json
from pathlib import Path
import unittest

from m31_ellipsoid_input_preflight import cell_surface, coordinate, verify


class EllipsoidInputTest(unittest.TestCase):
    def test_cell_boundary_distinguishes_shared_face_and_edge(self):
        self.assertEqual(set(cell_surface({(0,0,0),(1,0,0)}).values()),{2})
        touch=cell_surface({(0,0,0),(1,1,0)})
        self.assertEqual([edge for edge,n in touch.items() if n==4],[((1,1,0),(1,1,1))])

    def test_immutable_model_coordinates(self):
        self.assertEqual(coordinate((8,8,4)),['0','0','1/2'])

    def test_frozen_preflight(self):
        result=verify()
        receipt=Path(__file__).with_name('M31_ELLIPSOID_INPUT_PREFLIGHT.json')
        self.assertEqual(result,json.loads(receipt.read_bytes()))
        self.assertEqual(result['common_design_outer_lattice_edges'],208)
        self.assertEqual(result['fixed_boundary_four_face_edges'],208)
        self.assertEqual(result['target_to_outer_distance_model'],'1/2')
        self.assertFalse(result['native_request_emitted'])
