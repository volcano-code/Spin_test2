"""Adversarial offline replay fixtures. All materials in this test are synthetic."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest

from m23_replay_heart_capture import (REQUIRED_FILES, FROZEN_NATIVE_C, capture_files,
                                      check_attachment, surface_identity)
from m16_native_acceptance import file_hash
from native_cherchi_teacher import digest


class CaptureReplayTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        for name in REQUIRED_FILES:
            (self.root / name).write_text('synthetic test file')
        self.capture = dict(schema='w2-m23-fresh-heart-capture-v1', run_id='fixture-only',
                            J_D=None, training_admitted=False, fresh_evaluation_verified=True,
                            native_cherchi_calls=1,
                            files={p.name:file_hash(p) for p in self.root.iterdir()})
        self.material = dict(evaluation_map_identity='synthetic-map', boolean_backend_identity='mock',
                             derivative_identity='mock', geometry_hash='g', topology_hash='t',
                             boolean_operation='subtraction',
                             quadric_coefficients=[0.3701323978732728,-0.15863761828595124,-0.0,
                                                   -0.7608730119328398,0.293773367196206,0.293773367196206,
                                                   0.293773367196206,0.0,0.0,0.0])
        self.capture.update(same_call_material_sha256=digest(self.material),
                            evaluation_map_identity='synthetic-map', native_stage_call_counts={'boolean':1})
        self.proof = {k:self.material[k] for k in ('evaluation_map_identity','boolean_backend_identity',
                                                    'derivative_identity','geometry_hash','topology_hash')}
        self.proof.update(native_cherchi=True, kind='cherchi_same_call_discrete_map',
                          native_stage_call_counts={'boolean':1},
                          same_call_attachment=dict(schema='w2-same-call-material-binding-v1',
                              native_c_sha256=FROZEN_NATIVE_C,serialized_material_sha256=digest(self.material)))
        (self.root/'capture.json').write_text(json.dumps(self.capture))

    def test_complete_file_bundle_and_binding(self):
        self.assertEqual(capture_files(self.root, 'fixture-only'), self.capture)
        self.assertEqual(check_attachment(self.capture, {'exact_proof':self.proof}, self.material),
                         digest(self.material))

    def test_missing_completion_receipt_or_attachment(self):
        (self.root/'material.json').unlink()
        with self.assertRaisesRegex(ValueError, 'CAPTURE_FILE_HASH_MISMATCH'):
            capture_files(self.root, 'fixture-only')
        (self.root/'capture.json').unlink()
        with self.assertRaises(FileNotFoundError):
            capture_files(self.root, 'fixture-only')

    def test_tampered_file_stale_run_and_reused_material(self):
        with self.assertRaisesRegex(ValueError, 'WRONG_CAPTURE_OR_RUN_ID'):
            capture_files(self.root, 'another-run')
        (self.root/'label.json').write_text('tampered')
        with self.assertRaisesRegex(ValueError, 'CAPTURE_FILE_HASH_MISMATCH'):
            capture_files(self.root, 'fixture-only')
        other = dict(self.material, topology_hash='another-surface')
        with self.assertRaisesRegex(ValueError, 'WRONG_SAME_CALL_ATTACHMENT'):
            check_attachment(self.capture, {'exact_proof':self.proof}, other)

    def test_same_call_stage_and_evaluation_map(self):
        with self.assertRaisesRegex(ValueError, 'MISSING_NATIVE_PROOF'):
            check_attachment(dict(self.capture,native_stage_call_counts={'boolean':2}),
                             {'exact_proof':self.proof}, self.material)
        with self.assertRaisesRegex(ValueError, 'WRONG_EVALUATION_MAP'):
            check_attachment(dict(self.capture,evaluation_map_identity='previous-run'),
                             {'exact_proof':self.proof}, self.material)

    def test_oriented_surface_identity_rejects_reverse_duplicate_or_displaced_face(self):
        p = [[0,0,0],[2,0,0],[0,2,0],[0,0,2]]
        f = [[0,2,1],[0,1,3],[1,2,3],[0,3,2]]
        h = surface_identity(p,f)
        perm = [3,0,2,1]
        reordered = [[perm.index(i) for i in tri[1:]+tri[:1]] for tri in reversed(f)]
        self.assertEqual(h,surface_identity([p[i] for i in perm],reordered))
        self.assertNotEqual(h,surface_identity(p,[f[0][::-1],*f[1:]]))
        self.assertNotEqual(h,surface_identity(p,[f[1],*f[1:]]))
        moved = deepcopy(p)
        moved[0][0] = 1
        self.assertNotEqual(h,surface_identity(moved,f))
