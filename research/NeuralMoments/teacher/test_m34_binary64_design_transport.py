from fractions import Fraction as Q
from hashlib import sha256
import json
from pathlib import Path
import unittest

from m34_binary64_design_transport import audit, normal


class ProjectedDesignTests(unittest.TestCase):
    def test_tetrahedron_loses_an_edge_without_face_edits(self):
        near = Q(1) + Q(1, 2**54)
        packet = (f"4 4\n1 0 0\n{near} 0 0\n1 1 0\n1 0 1\n"
                  "0 2 1\n0 1 3\n0 3 2\n1 2 3\n").encode()
        result = audit(packet, sha256(packet).hexdigest())
        self.assertEqual(result["distinct_vertices_lost"], 1)
        self.assertEqual(result["formerly_nondegenerate_faces_lost"], 2)
        self.assertEqual(result["max_coordinate_error_model_exact"], str(near - 1))
        with self.assertRaisesRegex(ValueError, "digest"):
            audit(packet + b"\n", sha256(packet).hexdigest())

    def test_actual_first_face_uses_three_distinct_exact_points(self):
        filename = Path(__file__).with_name("M34_DPRIME_BINARY64_TRANSPORT.json")
        result = json.loads(filename.read_bytes())
        witness = result["first_face_witness"]
        points = [tuple(map(Q, v)) for v in witness["exact_vertices"]]
        rounded = [tuple(map(float.fromhex, v)) for v in witness["rounded_vertices_hex"]]
        self.assertTrue(any(normal(*points)))
        self.assertEqual(rounded, [tuple(map(float, v)) for v in points])
        self.assertFalse(any(normal(*[tuple(map(Q.from_float, v)) for v in rounded])))
        self.assertEqual(result["gmp_replay"]["reason"], "DEGENERATE_EXACT_TRIANGLE")
