"""Infrastructure checks; all native-looking data in this test is synthetic."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from native_cherchi_teacher import digest, seal
from cherchi_single_request import sha256_file
from run_native_smoke import FROZEN_SOURCE_SHA, _write_new_atomically, parse_spec, run

class TestNativeSmokePreflight(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.home=Path(self.tmp.name)
        self.model=self.home/'mock.obj';self.model.write_text('v 0 0 0\n')
        self.spec=self.home/'single.json'
        self.doc={
          'schema':'w2-one-cherchi-request-v2',
          'source_sha':FROZEN_SOURCE_SHA,
          'model_path':str(self.model),
          'expected_model_sha256':sha256_file(self.model),
          'expected_native_module_sha256':'a'*64,
          'request':{},'evaluator_settings':{},'native_c':[1.0]+[0.0]*9,
          'expected_native_c_sha256':digest([1.0]+[0.0]*9),
          'reference_absolute_tolerances':[1e-10]*10,
          'geometry':{'kind':'test_only'},'frame':{'kind':'identity'}}
        self.dump()
    def dump(self): self.spec.write_text(json.dumps(self.doc),encoding='utf-8')
    def test_preflight_parse(self):
        self.assertEqual(parse_spec(self.spec)['native_c'],[1.0]+[0.0]*9)
    def test_wrong_length_rejected(self):
        self.doc['native_c']=[1]*9;self.dump()
        with self.assertRaisesRegex(ValueError,'TEN_COMPONENTS_REQUIRED'): parse_spec(self.spec)
    def test_wrong_native_c_hash_rejected(self):
        self.doc['native_c'][8]=1.0;self.dump()
        with self.assertRaisesRegex(ValueError,'FROZEN_NATIVE_C_IDENTITY'): parse_spec(self.spec)
    def test_ambiguous_polynomial_q_rejected(self):
        self.doc['q']=[1.0]*10;self.dump()
        with self.assertRaisesRegex(ValueError,'AMBIGUOUS_Q_VS_NATIVE_C'): parse_spec(self.spec)
    def test_spec_forged_contract_rejected(self):
        self.doc['derivative_contract']={'verified_neighborhood':True};self.dump()
        with self.assertRaisesRegex(ValueError,'W0_DERIVATIVE'): parse_spec(self.spec)
    def test_model_bytes_changed_rejected(self):
        self.model.write_text('v 0 0 1\n')
        with self.assertRaisesRegex(ValueError,'FROZEN_MODEL_SHA256_MISMATCH'): parse_spec(self.spec)
    def test_missing_or_zero_budget_rejected(self):
        self.doc['reference_absolute_tolerances']=[0.0]*10;self.dump()
        with self.assertRaisesRegex(ValueError,'BUDGET_NONPOSITIVE'): parse_spec(self.spec)
    def test_preflight_failure_recorded_no_native(self):
        path=self.home/'failure.json'
        result=run(self.spec,path)
        self.assertEqual(result['status'],'FAILED_PREFLIGHT')
        self.assertFalse(result['training_admitted'])
        self.assertEqual(json.loads(path.read_text())['status'],'FAILED_PREFLIGHT')
        with self.assertRaisesRegex(FileExistsError,'REFUSE_OVERWRITE_EXISTING_EVIDENCE'):
            run(self.spec,path)
    def test_atomic_write_does_not_overwrite(self):
        path=self.home/'lock.json'
        _write_new_atomically(path,{'kind':'one'})
        with self.assertRaises(FileExistsError):
            _write_new_atomically(path,{'kind':'two'})
        self.assertEqual(json.loads(path.read_text()),{'kind':'one'})
        self.assertFalse(list(self.home.glob('*.partial')))
    def test_spec_mutation_after_fake_evaluation_is_recorded_as_postflight_failure(self):
        def mutate(_):
            self.spec.write_text(self.spec.read_text()+' ')
            raise RuntimeError('fixture changed before native call')
        path=self.home/'postflight.json'
        with patch('run_native_smoke.evaluate_from_spec',mutate):
            response=run(self.spec,path)
        self.assertEqual(response['status'],'FAILED_NATIVE_SETUP_OR_EVALUATION')
        self.assertIn('fixture changed',response['error'])
        self.assertEqual(json.loads(path.read_text())['status'],response['status'])
    def test_capture_and_persist_new_bound_attachment(self):
        # Synthetic producer exercises publication only; this calls no native routine.
        material = {'evaluation_map_identity': 'synthetic-map', 'material_proof': {'test': True}}
        def fake_evaluate(spec, *, capture_material):
            capture_material.update(material)
            return seal(model_sha256=spec['expected_model_sha256'],
                        native_module_sha256=spec['expected_native_module_sha256'],
                        geometry={}, density=1, frame={}, q=spec['native_c'],
                        r_d=[1.0]*10, j_d=None, r_f=[0.0]*10,
                        topology_status='OBSERVED_DISCRETE_MAP', derivative_status='UNKNOWN',
                        derivative_contract=None, stage_seconds={}, peak_rss_bytes=None,
                        exact_proof={'native_cherchi': True, 'evaluation_map_identity': 'synthetic-map',
                                     'same_call_attachment': {'serialized_material_sha256': digest(material)}},
                        split='excluded')
        label_file = self.home/'label.json'
        material_file = self.home/'material.json'
        with patch('run_native_smoke.evaluate_from_spec', fake_evaluate):
            result = run(self.spec, label_file, material_output=material_file)
        self.assertEqual(result['status'], 'LABEL_RECORDED')
        self.assertEqual(json.loads(material_file.read_text()), material)
        label = json.loads(label_file.read_text())['payload']
        self.assertEqual(label['exact_proof']['same_call_attachment']['serialized_material_sha256'],
                         digest(json.loads(material_file.read_text())))
        self.assertEqual(label['split'], 'excluded')
        self.assertIsNone(label['J_D'])

    def test_mock_success_cannot_be_published_as_native(self):
        def make_mock_record(spec):
            return seal(model_sha256=spec['expected_model_sha256'],
              native_module_sha256=spec['expected_native_module_sha256'],
              geometry={},density=1,frame={},q=spec['native_c'],
              r_d=[1.0]*10,j_d=None,r_f=[0.0]*10,
              topology_status='OBSERVED_DISCRETE_MAP',derivative_status='UNKNOWN',
              derivative_contract=None,stage_seconds={},peak_rss_bytes=None,
              exact_proof={'native_cherchi':False},split='excluded')
        path=self.home/'mock.json'
        with patch('run_native_smoke.evaluate_from_spec',make_mock_record):
            response=run(self.spec,path)
        self.assertEqual(response['status'],'FAILED_POSTFLIGHT')
        self.assertIn('NON_NATIVE_EVALUATION_RETURN',response['error'])

if __name__=='__main__':unittest.main()
