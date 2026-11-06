import unittest
from fractions import Fraction

from m30_terminal_complement import outer_minus_void, terminal_index


class TerminalReconstructionTests(unittest.TestCase):
    def test_terminal_must_be_same_accepted_coefficient_and_restart(self):
        q=[1.]*10
        rec=dict(status='NATIVE_TERMINAL',native_terminal='CONVERGED',
                 native_terminal_summary=dict(quadric_coefficients=q))
        accepted=dict(type='evaluation', outcome='accepted', restart=0,evaluation_index=3,quadric_coefficients=q)
        end=dict(type='attempt_terminal',reason='converged',restart=0,evaluation_index=3)
        self.assertEqual(terminal_index(rec,[accepted,end])[:2],(3,q))
        for bad in (dict(end,restart=1),dict(end,evaluation_index=2),dict(end,reason='maximum_iterations_reached')):
            with self.assertRaises(ValueError):terminal_index(rec,[accepted,bad])

    def test_carved_internal_triangles_are_reversed_and_indexed_after_outer(self):
        req=dict(target_axes=[[1,0,0],[0,1,0],[0,0,1]],target_origin=[0.5,0,0],
                 outer_vertices=[[1,0,0],[0,1,0],[0,0,1]],outer_faces=[[0,1,2]])
        cut=[['0/1']*3]*3
        mesh=outer_minus_void(req,cut,[[0,1,2]])
        self.assertEqual(mesh['faces'],[[0,1,2],[5,4,3]])
        self.assertEqual(Fraction(mesh['vertices'][0][0]),Fraction(1,2))
        self.assertEqual(mesh['vertices'][3:],cut)
        self.assertEqual(outer_minus_void(req,[],[])['faces'],req['outer_faces'])
