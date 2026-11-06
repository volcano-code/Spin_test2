"""Regression tests for M21 numeric success versus intentional W0 exclusion."""
import hashlib
import json
import math
from pathlib import Path
from types import ModuleType
import unittest


def load_archived_m21_gate():
    """Load the predicate shipped by ce578183 (the M21 launcher is external)."""
    path = Path(__file__).with_name('M21_DIAGNOSTIC_EXIT_SEMANTICS.patch')
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != '56d5d5030a4f49b39bf010a417d4d0ea7d6c229de0728c5f101f3b12089732da':
        raise ValueError('M21 pilot patch differs from the reviewed release')
    # Unified diff lines 10–67 are all new lines plus the retained "return False".
    # Remove only the diff prefix. Exercise the historical code itself.
    source = ''.join(line[1:] for line in raw.decode().splitlines(keepends=True)[9:67])
    module = ModuleType('archived_m21_pilot_gate')
    module.__dict__.update(json=json, math=math)
    exec(compile(source, str(path), 'exec'), module.__dict__)
    return module


pilot = load_archived_m21_gate()


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


class DiagnosticLabelSemantics(unittest.TestCase):
    def fixture(self, reason=pilot.EXPECTED_DIAGNOSTIC_EXCLUSION):
        q=[1.0]+[0.0]*9
        model='a'*64; native='b'*64
        geometry={'kind':'frozen-heart-test-fixture'}
        payload={
            'schema':'w2-cherchi-teacher-v2-pre-w0','order':pilot.TEACHER_ORDER,
            'coefficient_convention':pilot.COEFFICIENT_CONVENTION,
            'native_c_order':pilot.NATIVE_C_ORDER,'native_c':q,
            'polynomial_q':q[:7]+[0.0,0.0,0.0],
            'model_sha256':model,'native_module_sha256':native,
            'geometry':geometry,'density':1.0,'frame':{'kind':'test'},'q':q,
            'r_D':[1.0]*10,'J_D':None,'r_F':[2.0]*10,
            'topology_status':'OBSERVED_DISCRETE_MAP','derivative_status':'UNKNOWN',
            'derivative_contract':None,'stage_seconds':{'w2_total_wall_seconds':1.0},
            'peak_rss_bytes':1024,
            'exact_proof':{'kind':'cherchi_same_call_discrete_map','native_cherchi':True,
                'same_call_attachment':{'kind':'same-call-test-proof'},
                'independent_ten_moment_audit':{'comparison':{'within_tolerance':True}}},
            'split':'excluded','failure_reason':reason,
            'total_policy':'r_total=r_F+r_D;J_total=J_D;fixed_never_in_r_D'
        }
        record={'payload':payload,'payload_sha256':digest(payload),'identity':{
            'q_sha256':digest(q),'geometry_sha256':digest(geometry),
            'model_sha256':model,'native_module_sha256':native}}
        spec={'native_c':q,'expected_model_sha256':model,'expected_native_module_sha256':native}
        return record,spec

    def test_intentional_w0_exclusion_with_child_exit_2_is_numeric_diagnostic_pass(self):
        record,spec=self.fixture()
        self.assertTrue(pilot.verified_pilot_label(2,record,spec))

    def test_no_exclusion_with_exit_0_is_numeric_diagnostic_pass(self):
        record,spec=self.fixture(reason=None)
        self.assertTrue(pilot.verified_pilot_label(0,record,spec))

    def test_exit_2_with_no_label_is_not_success(self):
        self.assertFalse(pilot.verified_pilot_label(2,{},{}))

    def test_arbitrary_failure_reason_is_not_success(self):
        record,spec=self.fixture(reason='ValueError: independent moments disagree')
        self.assertFalse(pilot.verified_pilot_label(2,record,spec))

    def test_expected_exclusion_with_wrong_exit_code_is_not_success(self):
        record,spec=self.fixture()
        self.assertFalse(pilot.verified_pilot_label(0,record,spec))

    def test_bad_ten_moment_comparison_is_not_success(self):
        record,spec=self.fixture()
        record['payload']['exact_proof']['independent_ten_moment_audit']['comparison']['within_tolerance']=False
        record['payload_sha256']=digest(record['payload'])
        self.assertFalse(pilot.verified_pilot_label(2,record,spec))

    def test_label_digest_tampering_is_rejected(self):
        record,spec=self.fixture()
        record['payload']['r_D'][0]=9.0
        self.assertFalse(pilot.verified_pilot_label(2,record,spec))

    def test_spec_identity_mismatch_is_rejected(self):
        record,spec=self.fixture()
        spec['expected_model_sha256']='c'*64
        self.assertFalse(pilot.verified_pilot_label(2,record,spec))

    def test_training_split_is_never_accepted(self):
        record,spec=self.fixture()
        record['payload']['split']='train'
        record['payload_sha256']=digest(record['payload'])
        self.assertFalse(pilot.verified_pilot_label(2,record,spec))

    def test_jacobian_is_never_released_by_diagnostic_gate(self):
        record,spec=self.fixture()
        record['payload']['J_D']=[[0.0]*10 for _ in range(10)]
        record['payload_sha256']=digest(record['payload'])
        self.assertFalse(pilot.verified_pilot_label(2,record,spec))


if __name__=='__main__':
    unittest.main()
