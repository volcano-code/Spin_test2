import json
from pathlib import Path
import unittest
from fractions import Fraction

from m35_crop_control import BOUNDS, signature, validate
from m35_exact_crossing import make_source
from m35_general_crossing import cap_box_disjoint

HERE = Path(__file__).parent


class CrossingEvidence(unittest.TestCase):
    def test_oriented_parent_signature(self):
        mesh = dict(vertices=[["0","0","0"],["0","1","0"],["0","0","1"]], faces=[[0,1,2]], birth_faces=[9])
        same = signature(mesh, lambda x: x)
        mesh["faces"] = [[1,2,0]]
        self.assertEqual(same, signature(mesh, lambda x: x))
        mesh["faces"] = [[0,2,1]]
        self.assertNotEqual(same, signature(mesh, lambda x: x))
        mesh["faces"] = [[0,1,2]]
        self.assertNotEqual(same, signature(mesh, lambda x: x+1))

    def test_crop_contact_rejected(self):
        vertices = [[Fraction(x,32) for x in BOUNDS[0]]]
        with self.assertRaisesRegex(ValueError, "strict crop interior"):
            validate({}, {}, {}, dict(vertices=vertices))

    def test_general_cap_axis_separation(self):
        points = [[0.,0.,0.], [2.,0.,0.], [2.,1.,1.], [2.,0.,2.]]
        s = dict(vertices=points, faces=[[1,2,3],[0,2,3],[1,2,3]], face_is_cap=[1,1,0])
        self.assertEqual(cap_box_disjoint(s),1)

    def test_invalid_q(self):
        with self.assertRaises(ValueError):
            make_source([1], None, None)

    def test_registered_evidence(self):
        report = json.loads((HERE/"M35_NOODLES_EXACT_CROSSINGS.json").read_bytes())
        self.assertEqual(len(report["samples"]),14)
        seed = report["samples"]["seed0"]
        self.assertEqual((seed["cutter_parent_faces"], seed["design_parent_faces"], seed["cap_faces"]), (1678,12414,50))
        for name,row in report["samples"].items():
            if name != "seed0":
                self.assertEqual((row["source_faces"],row["faces"],row["source_velocity_scalars"]), (320,482,4860))


if __name__ == "__main__":
    unittest.main()
