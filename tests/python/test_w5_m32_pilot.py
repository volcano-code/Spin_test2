import json
from pathlib import Path
import tempfile
import unittest

from research.NeuralMoments.models.deeponet.contract_gate import (
    AdmissionError, W2_V2_SCHEMA, admit_w2_record, audit_family_split, strict_json, file_sha256,
)
from research.NeuralMoments.models.deeponet.m32_pilot_readiness import readiness

DATA = Path(__file__).resolve().parents[2] / "research/NeuralMoments/models/deeponet"


class M32PilotGate(unittest.TestCase):
    def test_protocol_receipt_and_common_budget(self):
        p = strict_json(DATA / "M32_PAIRED_PILOT.json")
        receipt = strict_json(DATA / "M32_PILOT_READINESS.json")
        self.assertEqual(receipt["protocol_sha256"], file_sha256(DATA/"M32_PAIRED_PILOT.json"))
        self.assertEqual(set(p["arms"]), set("ABCD"))
        self.assertEqual(len(p["start_list"]), 2)
        restart = p["start_list"][1]
        self.assertEqual((restart["mode"],restart["deterministic_seed"],restart["maximum_restarts"]),("SPINNING_TOP",0,8))
        self.assertEqual((p["paired_repetitions"], p["wall_seconds_per_arm_repeat"],
                          p["maximum_exact_geometry_requests_per_arm_repeat"]),(3,600,120))
        self.assertEqual(p["mass_continuation"]["order"], [0.95, 0.90])
        self.assertTrue(p["mass_continuation"]["all_arms_same_rule"])
        self.assertEqual(p["admitted_real_training_hashes"], [])
        self.assertFalse(receipt["benchmark_ready"])
        self.assertIsNone(receipt["speedup"])

    def test_a_witness_or_self_declared_v2_cannot_be_used_to_fit(self):
        base=DATA/"M32_PAIRED_PILOT.json"
        kw=dict(expected_c_native=[0]*10,model_path=base,native_module_path=base,
                expected_design_sha256="0"*64,expected_w0_original_sha256="1"*64,
                expected_w0_fixed_sha256="2"*64)
        with self.assertRaisesRegex(AdmissionError,"No independently approved"):
            admit_w2_record(base,approved_file_sha256=[],**kw)
        with tempfile.TemporaryDirectory() as tmp:
            spoof=Path(tmp)/"spoof.json"
            spoof.write_text(json.dumps({"payload":{"schema":W2_V2_SCHEMA,"split":"train"}}))
            with self.assertRaisesRegex(AdmissionError,"NOT_INDEPENDENTLY_SEALED"):
                admit_w2_record(spoof,approved_file_sha256=[file_sha256(spoof)],**kw)
            with self.assertRaisesRegex(AdmissionError,"pinned"):
                readiness(spoof,spoof)

    def test_one_parent_m32_is_not_a_real_train_test_split(self):
        geo={k:str(i)*64 for i,k in enumerate(("original_sha256","fixed_F_sha256","design_D_sha256"),1)}
        rows=[dict(split=s,family="Noodles-M32",geometry=geo,record_sha256=c*64) for s,c in (("train","4"),("test","5"))]
        with self.assertRaisesRegex(AdmissionError,"leaked"):
            audit_family_split(rows)
