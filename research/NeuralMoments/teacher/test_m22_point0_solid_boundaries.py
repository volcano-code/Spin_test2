"""Regressions for the additional connectedness/volume and checker completion gates."""
from fractions import Fraction as F
from pathlib import Path
import subprocess
import unittest
from unittest.mock import patch

from m22_point0_solid_boundaries import audit_surface, oriented_shell

TETRA = [(0,0,0), (1,0,0), (0,1,0), (0,0,1)]
FACES = [(0,2,1), (0,1,3), (0,3,2), (1,2,3)]


class SolidBoundaries(unittest.TestCase):
    def test_translated_outward_tetra(self):
        vertices = [tuple(v + F(3,7) for v in p) for p in TETRA]
        shell = oriented_shell(vertices, FACES)
        self.assertEqual(shell['volume_exact'], '1/6')
        self.assertEqual((shell['edges'], shell['euler'], shell['genus']), (6, 2, 0))

    def test_reverse_all_faces(self):
        with self.assertRaisesRegex(ValueError, 'NONPOSITIVE_SIGNED_VOLUME'):
            oriented_shell(TETRA, [tri[::-1] for tri in FACES])

    def test_nested_components_require_separate_winding_policy(self):
        inner = [tuple(F(1,8) + F(x,4) for x in p) for p in TETRA]
        faces = FACES + [tuple(i+4 for i in tri[::-1]) for tri in FACES]
        with self.assertRaisesRegex(ValueError, 'MULTIPLE_OR_UNREFERENCED_COMPONENTS'):
            oriented_shell(TETRA + inner, faces)

    def test_short_candidate_replay_rejected(self):
        stdout = (b'status=EXACT_POST_BOOLEAN_SELF_INTERSECTION_FREE\nvertices=4\nfaces=4\n'
                  b'candidate_pairs=6\nexact_pairs_checked=5\nadjacent_pairs=6\nnonadjacent_pairs=0\n')
        with patch('subprocess.run', return_value=subprocess.CompletedProcess([],0,stdout)):
            with self.assertRaisesRegex(ValueError, 'INCOMPLETE_SURFACE_CHECK'):
                audit_surface(Path('checker'), TETRA, FACES)
