"""W2 M11 tests: actual 81-point historic Cherchi data, zero new native runs.

Local manifest fixture copies independently pinned M9 fields checked in GitHub;
it is NOT falsely claimed to be the verbatim GitHub manifest. Production
review_m11 consumes that exact manifest when run in the W2 checkout.
"""
import copy
import hashlib
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np

from w2_consumer_gate import (CANDIDATE_SHA256, CANDIDATE_BLOB_SHA1, SOURCE_SHA,
        ARCHIVE_SHA256, REPORT_SHA256, CLASS_SHA256, REQUEST_SHA256,
        RUN_ID, ARTIFACT_ID, W0_HANDOFF_BLOB_SHA1, W0_M9_REPORT_BLOB_SHA1,
        canonical, GateRejected, validate_candidate_structure, validate_manifest)
from w2_w0_v2_bridge import (BridgeRejected, review_m11, export_training,
        _validate_w2_v2_one, _pinned_w0_schema, _pinned_w3_status,
        _git_blob_bytes, _w0_diagnostic_record, W0_SCHEMA_BLOB_SHA1)

CAND=Path(__file__).with_name('M9_H8F_81_EXCLUDED_CANDIDATES.json')
REAL=os.environ.get('W2_H8F_ARCHIVE')
requires_archive=unittest.skipUnless(REAL, 'Set W2_H8F_ARCHIVE to the pinned H8f ZIP for this material-backed test')


def fixture_manifest():
    """Reconstruct exactly the semantic M9 handoff fields; other fields omitted."""
    return dict(schema='w2-cherchi-teacher-v2-pre-w0',training_samples=0,
      local_native_calls=0,native_teacher_runs=0,
      label_convention=dict(w0_verifier_integrated=False),
      data_splits=dict(train=[],val=[],test=[],excluded=[{
          'path':'research/NeuralMoments/teacher/M9_H8F_81_EXCLUDED_CANDIDATES.json',
          'count':81}]),
      m9_real_h8f_81_evidence=dict(
          schema='w2-h8f-81-physical-moment-candidates-v1',
          workflow_run_id=RUN_ID,original_workflow_conclusion='failure',
          original_head_sha=SOURCE_SHA,archive_artifact_id=ARTIFACT_ID,
          archive_sha256=ARCHIVE_SHA256,archive_report_sha256=REPORT_SHA256,
          archive_classification_sha256=CLASS_SHA256,
          frozen_request_sha256=REQUEST_SHA256,
          candidate_index_sha256=CANDIDATE_SHA256,
          candidate_index_git_blob_sha1=CANDIDATE_BLOB_SHA1,
          W0_M9_report_blob_sha1=W0_M9_REPORT_BLOB_SHA1,
          w0_handoff_document_blob_sha1=W0_HANDOFF_BLOB_SHA1,
          original_native_evaluations=81,new_native_evaluations=0,
          independently_reintegrated_materials=81,
          independently_reintegrated_ten_moment_components=810,
          training_admitted=0,newton_solved=False,
          certified_continuous_neighborhood=False,
          derivative_J_D_published=False,
          finite_symbolic_classifications={'STABLE_NOT_CERTIFIED':79,'FEATURE_CHANGE':1},
          point60=dict(changed_vertex_positions=[8043,8044,8045,8046,8047],
                       removed_LPI_features=2,added_LPI_features=2,
                       removed_oriented_triangles=17,added_oriented_triangles=17,
                       c7_from=1.5e-6,c7_to=3e-6),
          candidate_index_path='research/NeuralMoments/teacher/M9_H8F_81_EXCLUDED_CANDIDATES.json'
      ))


def fake_v2(c=None):
    if c is None:c=[1.,0.,0.,0.,0.,0.,0.,0.,0.,0.]
    model_sha='a'*64;module_sha='b'*64;geometry={'kind':'synthetic-fixture'}
    p={'schema':'w2-cherchi-teacher-v2-pre-w0',
       'order':['m','x','y','z','xx','yy','zz','yz','xz','xy'],
       'coefficient_convention':'NATIVE_MATRIX_ENTRIES_C10_POLYNOMIAL_Q_EQUALS_BC_V1',
       'native_c_order':'CONST,X,Y,Z,XX,YY,ZZ,YZ,XZ,XY'.split(','),
       'native_c':c,'q':c[:],
       'polynomial_q':[x*(2 if i>=7 else 1) for i,x in enumerate(c)],
       'model_sha256':model_sha,'native_module_sha256':module_sha,
       'geometry':geometry,'density':1.0,'frame':{'kind':'synthetic'},
       'r_D':[1.]+[0.]*9,'r_F':[2.]+[0.]*9,
       'J_D':None,'topology_status':'OBSERVED_DISCRETE_MAP',
       'derivative_status':'UNKNOWN','derivative_contract':None,
       'stage_seconds':{'source':0.01},'peak_rss_bytes':12500000,
       'exact_proof':{'native_cherchi':True},'split':'excluded',
       'failure_reason':None,
       'total_policy':'r_total=r_F+r_D;J_total=J_D;fixed_never_in_r_D'}
    def hash_obj(obj):return hashlib.sha256(canonical(obj).encode()).hexdigest()
    record={'payload':p,'payload_sha256':hash_obj(p),
        'identity':{'q_sha256':hash_obj(c),'geometry_sha256':hash_obj(geometry),
                    'model_sha256':model_sha,'native_module_sha256':module_sha}}
    return record


def check_fixture_label(record):
    return _validate_w2_v2_one(record,
        expected_native_c=record['payload']['native_c'],
        expected_model_sha256='a'*64,expected_module_sha256='b'*64,
        expected_geometry_sha256=hashlib.sha256(canonical({'kind':'synthetic-fixture'}).encode()).hexdigest())


class TestM11V2Label(unittest.TestCase):
    def test_v2_structural_only(self):
        x=check_fixture_label(fake_v2())
        self.assertEqual(x['status'],'STRUCTURAL_V2_ONLY_NOT_NATIVE_CERTIFICATION')
        self.assertFalse(x['native_material_certified'])
        self.assertFalse(x['training_admitted'])
    def test_w0_v1_label_rejected(self):
        x=fake_v2();x['payload']['schema']='w2-cherchi-teacher-v1-pre-w0'
        with self.assertRaisesRegex(BridgeRejected,'W2_V2_SCHEMA_REQUIRED'):
            check_fixture_label(x)
    def test_v2_cross_term_swap_rejected_even_with_new_digest(self):
        x=fake_v2([0.,0.,0.,0.,0.,0.,0.,1.,2.,3.]);x['payload']['polynomial_q'][7:]=[4.,2.,6.]
        x['payload_sha256']=hashlib.sha256(canonical(x['payload']).encode()).hexdigest()
        with self.assertRaisesRegex(GateRejected,'MONOMIAL_Q_EQUALS_B_C'):
            check_fixture_label(x)
    def test_jacobian_not_admitted_even_with_forged_claim(self):
        x=fake_v2();x['payload']['J_D']=[[0.]*10 for _ in range(10)]
        x['payload']['derivative_status']='VALID';x['payload']['split']='train'
        x['payload_sha256']=hashlib.sha256(canonical(x['payload']).encode()).hexdigest()
        with self.assertRaisesRegex(BridgeRejected,'W2_V2_DERIVATIVE_PROMOTION_DENIED'):
            check_fixture_label(x)
    def test_identity_external_provenance_required(self):
        x=fake_v2();withheld=x['identity']['model_sha256']
        x['identity']['model_sha256']='c'*64
        with self.assertRaisesRegex(BridgeRejected,'W2_V2_EXTERNAL_IDENTITY_MISMATCH'):
            _validate_w2_v2_one(x,expected_native_c=x['payload']['native_c'],
               expected_model_sha256=withheld,expected_module_sha256='b'*64,
               expected_geometry_sha256=x['identity']['geometry_sha256'])
    def test_bad_total_policy_rejected(self):
        x=fake_v2();x['payload']['total_policy']='r_total=r_F+r_D+r_F'
        x['payload_sha256']=hashlib.sha256(canonical(x['payload']).encode()).hexdigest()
        with self.assertRaisesRegex(BridgeRejected,'W2_V2_FIXED_COUNTING_POLICY_CHANGED'):
            check_fixture_label(x)
    def test_fake_native_true_is_not_proof(self):
        x=check_fixture_label(fake_v2())
        self.assertFalse(x['native_material_certified'])
    def test_training_export_always_denied(self):
        with self.assertRaises(BridgeRejected):export_training('any_token')
    def test_no_w0_schema_is_not_validated(self):
        x=_pinned_w0_schema(None)
        self.assertFalse(x['observed'])
    def test_w0_schema_modified_is_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'f.json';p.write_text('{"title":"MomentContractV1"}')
            with self.assertRaisesRegex(BridgeRejected,'W0_SCHEMA_BLOB_SHA_CHANGED'):
                _pinned_w0_schema(p)
    def test_modified_w3_report_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'w3.json';p.write_text('{}')
            with self.assertRaisesRegex(BridgeRejected,'W3_STAGE9_BLOB_SHA_CHANGED'):
                _pinned_w3_status(p)


class TestM11RealH8f(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.record=json.loads(CAND.read_text())
        cls.manifest=fixture_manifest()
        cls.t=tempfile.TemporaryDirectory()
        cls.addClassCleanup(cls.t.cleanup)
        cls.fixture=Path(cls.t.name)/'manifest_semantic_fixture.json'
        cls.fixture.write_text(json.dumps(cls.manifest,sort_keys=True))
        assert hashlib.sha256(CAND.read_bytes()).hexdigest()==CANDIDATE_SHA256
        validate_candidate_structure(cls.record)
        validate_manifest(cls.manifest,cls.record)
        # One actual 81-point rebind, 12-15 seconds. Do not repeat in every test.
        cls.actual=review_m11(CAND,cls.fixture,archive_path=Path(REAL) if REAL else None,index=0)
        cls.point60=review_m11(CAND,cls.fixture,index=60)

    @requires_archive
    def test_real_81_rebound(self):
        self.assertEqual(self.actual['w2_historical_rebind'],
            'ARCHIVED_81_MATERIALS_REBOUND_DERIVATIVES_BLOCKED')
    def test_exact_w0_c_norm_inventory(self):
        self.assertEqual(self.actual['unit_c_compatibility']['count_pass'],17)
        self.assertEqual(self.actual['unit_c_compatibility']['count_fail'],64)
        self.assertIn(0,self.actual['unit_c_compatibility']['passing_indices'])
        self.assertNotIn(60,self.actual['unit_c_compatibility']['passing_indices'])
    @requires_archive
    def test_w0_witness_populated_from_real_archival_source(self):
        contract=self.actual['w0_structural_diagnostic_record']
        self.assertEqual(contract['schema'],'MomentContractV1')
        self.assertEqual(contract['source_sha'],SOURCE_SHA)
        self.assertEqual(contract['moments']['status'],'NATIVE_VERIFIED')
        self.assertEqual(contract['geometry']['original_sha256'],
                         'fbada64691e30b458bc5b5375f8d4ad351cb0b383d800512bb659077b29532f6')
        self.assertEqual(contract['geometry']['fixed_F_sha256'],
                         '45c83ffd19c94b6c75ea63e7e7261fda645cd00ec0920c1a6ecc68e703c3a5b0')
        self.assertEqual(contract['geometry']['design_D_sha256'],
                         '391fbeb80b8e7e4e2f12217880f4642c5fccd30ae5bb9ba2b5703d0dbd05b092')
    @requires_archive
    def test_w0_unit_norm_and_moment_relation(self):
        c=self.actual['w0_structural_diagnostic_record']
        norm=float(np.linalg.norm(c['coefficients']['c']))
        self.assertLessEqual(abs(norm-1.),1e-12)
        from numpy.testing import assert_allclose
        assert_allclose(c['moments']['total_r'],[a+b for a,b in zip(
            c['moments']['fixed_r'],c['moments']['variable_r'])],rtol=0,atol=0)
    @requires_archive
    def test_never_releases_real_jacobian(self):
        self.assertIsNone(self.actual['w0_structural_diagnostic_record']['jacobian']['values'])
        self.assertFalse(self.actual['w0_diagnostic_record_is_release'])
        self.assertFalse(self.actual['formal_w0_v2_label_issued'])
        self.assertEqual(self.actual['derivative_training_samples'],0)
    def test_reconstructed_heart_not_author_original(self):
        self.assertEqual(self.actual['source']['original_workflow_conclusion'],'failure')
        self.assertTrue(self.actual['point60_changed_feature_is_not_smooth'])
    def test_point60_no_renormalize_and_no_diagnostic(self):
        self.assertIsNone(self.point60['w0_structural_diagnostic_record'])
        self.assertFalse(self.point60['unit_c_compatibility']['requested_unit_compatible'])
        self.assertIn('RAW_NATIVE_C_NOT_W0_UNIT_NORM_DO_NOT_NORMALIZE_ARCHIVE',
                      self.point60['source_scientific_gaps'])
    def test_point60_native_c7_and_double_monomial(self):
        p=self.point60['coordinate_mapping_only']
        self.assertEqual(p['native_c'][7],3e-6)
        self.assertEqual(p['polynomial_q'][7],6e-6)
        self.assertIsNone(p['derivative_J_D'])
    def test_index_only_cannot_claim_reopened_archive(self):
        self.assertEqual(self.point60['w2_historical_rebind'],
                         'PINNED_M9_INDEX_ONLY_DERIVATIVES_BLOCKED')
    def test_original_c_is_immutable_in_real_candidate(self):
        before=self.record['samples'][0]['native_c'][:]
        self.assertEqual(before,self.actual['coordinate_mapping_only']['native_c'])
        self.assertEqual(before,self.record['samples'][0]['native_c'])
    def test_no_origin_F_D_fabricated(self):
        self.assertIsNone(self.point60['w0_structural_diagnostic_record'])
        self.assertFalse(self.actual['w0_schema_structural_validation_performed'])


if __name__=='__main__':unittest.main()
