"""W2 M10 negative/real-data tests. No new Cherchi call or W0 admission.

The manifest fixture is the exact W2 M9 snapshot from f129edd2f668c845ab60ab3ac5b0616773106a9c.
The 81 candidates and the optional archived ZIP are also SHA-pinned historical inputs.
"""
import copy
import json
from pathlib import Path
import tempfile
import unittest

from w2_consumer_gate import (
    STABLE, GateRejected, CANDIDATE_SHA256, ARCHIVE_SHA256,
    decode_json, export_training, main, review, sha256,
    validate_candidate_structure, validate_manifest, verify_original_archive,
)

DATA=Path(__file__).resolve().parents[1]
CANDIDATES=Path(__file__).with_name('M9_H8F_81_EXCLUDED_CANDIDATES.json')
MANIFEST=Path(__file__).parent/'fixtures'/'m9_teacher_manifest.json'
# Optional real H8f ZIP path; leave unset for routine fast smoke.
import os
ARCHIVE=Path(os.environ['W2_H8F_ARCHIVE']) if os.environ.get('W2_H8F_ARCHIVE') else None


class TestPinnedCandidateGate(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.candidates=json.loads(CANDIDATES.read_text())
        cls.manifest=json.loads(MANIFEST.read_text())

    def falsify(self, mutator, contains):
        obj=copy.deepcopy(self.candidates)
        mutator(obj)
        with self.assertRaisesRegex(GateRejected, contains):
            validate_candidate_structure(obj)

    def test_01_exact_real_candidate_file_sha(self):
        self.assertEqual(sha256(CANDIDATES.read_bytes()), CANDIDATE_SHA256)

    def test_02_full_real_81_index_structure(self):
        r=validate_candidate_structure(self.candidates)
        self.assertEqual(r['rows'],81)
        self.assertEqual(r['exact_physical_ten_moment_components'],810)
        self.assertTrue(r['point60_feature_change'])

    def test_03_pinned_manifest_handoff_fields(self):
        self.assertEqual(sha256(MANIFEST.read_bytes()),
                         '750ea92a0370e5aee844cb4e227ad7a73d53c551bf1adcea6e5ead1cef2863a8')
        r=validate_manifest(self.manifest,self.candidates)
        self.assertFalse(r['w0_formal_verifier_integrated_into_w2'])

    def test_04_source_failure_cannot_be_success(self):
        self.falsify(lambda o: o['source'].__setitem__('upstream_workflow_conclusion','success'),
                     'HISTORICAL_SOURCE')

    def test_05_candidates_must_be_all81(self):
        self.falsify(lambda o: o.__setitem__('exact_reintegrated_count',80),'NOT_ALL_81')

    def test_06_physical_reference_focus_only_rejected(self):
        self.falsify(lambda o: o.__setitem__('exact_mode','focus'),'FOCUS_ONLY')

    def test_07_forged_W0_smooth_admission_rejected(self):
        self.falsify(lambda o: o.__setitem__('W0_MomentContractV1_derivative_admission',True),
                     'UNSUPPORTED_SCIENTIFIC_PROMOTION')

    def test_08_false_newton_convergence_rejected(self):
        self.falsify(lambda o: o.__setitem__('newton_converged',True),
                     'UNSUPPORTED_SCIENTIFIC_PROMOTION')

    def test_09_false_training_sample_count_rejected(self):
        self.falsify(lambda o: o.__setitem__('formal_training_samples',1),
                     'FORGED_TRAINING_OR_NEW_NATIVE_COUNT')

    def test_10_false_new_native_run_count_rejected(self):
        self.falsify(lambda o: o.__setitem__('new_native_evaluations',1),
                     'FORGED_TRAINING_OR_NEW_NATIVE_COUNT')

    def test_11_false_stable_point60_rejected(self):
        self.falsify(lambda o: o['samples'][60].__setitem__('finite_feature_status',STABLE),
                     'POINT60_OR_FINITE_SYMBOLIC_STATUS_FALSIFIED')

    def test_12_original_point60_witness_required(self):
        self.falsify(lambda o: o['point60_witness'].__setitem__('changed_vertex_indices',[]),
                     'POINT60_TRUE_FEATURE_CHANGE_WITNESS_LOST')

    def test_13_forged_continuous_feature_certificate_rejected(self):
        self.falsify(lambda o: o['point60_witness'].__setitem__('continuous_branch_certified',True),
                     'POINT60_TRUE_FEATURE_CHANGE_WITNESS_LOST')

    def test_14_swapped_polynomial_cross_terms_rejected(self):
        def mutate(o):
            row=o['samples'][60]
            row['polynomial_q'][7],row['polynomial_q'][8] = row['polynomial_q'][8],row['polynomial_q'][7]
        self.falsify(mutate,'C_TO_POLYNOMIAL_Q_CROSS_TERM_MAPPING_CHANGED')

    def test_15_native_coefficients_repeated_rejected(self):
        self.falsify(lambda o: o['samples'][17].__setitem__('native_c',o['samples'][18]['native_c']),
                     'C_TO_POLYNOMIAL_Q_CROSS_TERM_MAPPING_CHANGED')

    def test_16_invalid_mass_nan_rejected(self):
        self.falsify(lambda o: o['samples'][12]['r_D'].__setitem__(0,float('nan')),
                     'NONFINITE_VECTOR10_VARIABLE_R')

    def test_17_truthy_string_not_boolean_rejected(self):
        self.falsify(lambda o: o['samples'][13].__setitem__('independent_moment_rechecked','true'),
                     'SAMPLE_EXACT_TEN_MOMENT_NOT_VERIFIED')

    def test_18_finite_point_only_not_full_topology_certificate(self):
        self.falsify(lambda o: o['samples'][60].__setitem__('topology_observation','CERTIFIED_SMOOTH'),
                     'INVALID_FINITE_TOPOLOGY_STATUS')

    def test_19_cannot_publish_jacobian(self):
        self.falsify(lambda o: o['samples'][0].__setitem__('J_D',[[0.0]*10 for _ in range(10)]),
                     'UNAUTHENTICATED_JACOBIAN_RELEASE_ATTEMPT')

    def test_20_cannot_change_train_split(self):
        self.falsify(lambda o: o['samples'][0].__setitem__('split','train'),
                     'TRAINING_SPLIT_PROMOTION_FORBIDDEN')

    def test_21_wrong_original_material_sha_rejected(self):
        self.falsify(lambda o: o['samples'][33].__setitem__('same_call_material_gzip_sha256','abc'),
                     'INVALID_SHA256_SAME_CALL_MATERIAL_GZIP_SHA256')

    def test_22_swapped_sample_indices_rejected(self):
        self.falsify(lambda o: o['samples'][5].__setitem__('index',6),
                     'CANDIDATE_SAMPLE_IDENTITY_OR_ORDER_MISMATCH')

    def test_23_incomplete_81_rows_rejected(self):
        self.falsify(lambda o: o['samples'].pop(), 'CANDIDATE_SAMPLE_COUNT_MISMATCH')

    def test_24_manifests_refuse_fake_training_splits(self):
        m=copy.deepcopy(self.manifest)
        m['data_splits']['train']=[{'index':0}]
        with self.assertRaisesRegex(GateRejected,'MANIFEST_TRAINING_SPLIT_FORBIDDEN'):
            validate_manifest(m,self.candidates)

    def test_25_manifest_w0_sha_changed_rejected(self):
        m=copy.deepcopy(self.manifest)
        m['m9_real_h8f_81_evidence']['w0_handoff_document_blob_sha1']='a'*40
        with self.assertRaisesRegex(GateRejected,'MANIFEST_UNTRUSTED_M9'):
            validate_manifest(m,self.candidates)

    def test_26_manifest_blobs_cannot_claim_native_progress(self):
        m=copy.deepcopy(self.manifest)
        m['native_teacher_runs']=2
        with self.assertRaisesRegex(GateRejected,'MANIFEST_FORGED_NATIVE'):
            validate_manifest(m,self.candidates)

    def test_27_reject_duplicate_JSON_property(self):
        with self.assertRaisesRegex(GateRejected,'DUPLICATE_JSON_KEY'):
            decode_json(b'{"status":false,"status":true}')

    def test_28_reject_nonfinite_json(self):
        with self.assertRaisesRegex(GateRejected,'NONFINITE_JSON_CONSTANT'):
            decode_json(b'{"x":NaN}')

    def test_29_release_API_always_refuses(self):
        with self.assertRaisesRegex(GateRejected,'DERIVATIVE_TRAINING_RELEASE_DENIED'):
            export_training(self.candidates,self.manifest,{'W0_certified':True})

    def test_30_readonly_pinned_without_archive_distinguishes_scope(self):
        output=review(CANDIDATES,MANIFEST)
        self.assertEqual(output['coverage']['rows'],81)
        self.assertEqual(output['archive_binding']['archived_materials_rebound'],0)
        self.assertFalse(output['training_admitted'])
        self.assertIsNone(output['derivative_J_D'])
        self.assertEqual(output['source']['original_workflow_conclusion'],'failure')

    def test_31_pinned_file_change_rejected_even_with_flags_unchanged(self):
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'forged.json'
            obj=copy.deepcopy(self.candidates)
            obj['samples'][2]['r_D'][5]+=1e-4
            path.write_text(json.dumps(obj))
            with self.assertRaisesRegex(GateRejected,'UNTRUSTED_CANDIDATE_BYTES_SHA256'):
                review(path,MANIFEST)

    def test_32_extra_untrusted_W3_field_cannot_grant_training(self):
        m=copy.deepcopy(self.manifest)
        m['claimed_external_w3_event_certified']=True
        validate_manifest(m,self.candidates)
        r=review(CANDIDATES,MANIFEST)
        self.assertFalse(r['certifications']['w3_point60_event_resolved'])
        self.assertEqual(r['training_samples'],0)

    def test_33_cli_writes_exclusive_evidence_and_blocks_overwrite(self):
        with tempfile.TemporaryDirectory() as d:
            target=Path(d)/'decision.json'
            code=main(['--candidate',str(CANDIDATES),'--manifest',str(MANIFEST),
                       '--output',str(target)])
            self.assertEqual(code,0)
            self.assertFalse(json.loads(target.read_text())['training_admitted'])
            original=target.read_bytes()
            with self.assertRaisesRegex(GateRejected,'REFUSE_OVERWRITE_SCIENTIFIC_EVIDENCE'):
                main(['--candidate',str(CANDIDATES),'--manifest',str(MANIFEST),
                      '--output',str(target)])
            self.assertEqual(original,target.read_bytes())

    def test_34_cli_persists_explicit_REJECTED_with_exit_2(self):
        with tempfile.TemporaryDirectory() as d:
            c=Path(d)/'tampered.json'
            c.write_bytes(CANDIDATES.read_bytes()+b' ')
            output=Path(d)/'reject.json'
            code=main(['--candidate',str(c),'--manifest',str(MANIFEST),
                       '--output',str(output)])
            self.assertEqual(code,2)
            payload=json.loads(output.read_text())
            self.assertEqual(payload['status'],'REJECTED')
            self.assertIn('UNTRUSTED_CANDIDATE_BYTES_SHA256',payload['error'])
            self.assertFalse(payload['training_admitted'])


class TestHistoricalArchiveBinding(unittest.TestCase):
    @unittest.skipUnless(ARCHIVE is not None and ARCHIVE.is_file(),
                         'Real H8f archive not supplied: set W2_H8F_ARCHIVE')
    def test_35_original_archive_sha256(self):
        import hashlib
        h=hashlib.sha256()
        with ARCHIVE.open('rb') as fh:
            for b in iter(lambda:fh.read(1<<20), b''):h.update(b)
        self.assertEqual(h.hexdigest(),ARCHIVE_SHA256)

    @unittest.skipUnless(ARCHIVE is not None and ARCHIVE.is_file(),
                         'Real H8f archive not supplied: set W2_H8F_ARCHIVE')
    def test_36_full_real_historical_81_material_rebind(self):
        record=json.loads(CANDIDATES.read_text())
        result=verify_original_archive(ARCHIVE,record['samples'])
        self.assertEqual(result['archived_materials_rebound'],81)
        self.assertEqual(result['jacobian_digests_rebound_without_releasing_values'],81)
        self.assertEqual(result['original_workflow_conclusion'],'failure')
        self.assertLess(result['maximum_fixed_plus_variable_disagreement'],1e-10)

    def test_37_short_or_swapped_archive_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            fake=Path(d)/'not_archived.zip'
            fake.write_bytes(b'PK\x03\x04 tampered')
            record=json.loads(CANDIDATES.read_text())
            with self.assertRaisesRegex(GateRejected,'HISTORICAL_NATIVE_ARCHIVE_SHA256_MISMATCH'):
                verify_original_archive(fake,record['samples'])


if __name__=='__main__':
    unittest.main()
