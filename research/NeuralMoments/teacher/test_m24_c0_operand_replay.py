from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from m16_native_acceptance import file_hash, strict_json
from m23_fresh_point0 import HERE, write_json
from m24_capture_c0 import plan
from m25_capture_linear import COLUMNS
from m24_c0_operand_replay import replay


class C0OperandReceiptTest(unittest.TestCase):
    def test_untrusted_stencil_rejected_before_material_reads(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            mod = strict_json((HERE/'M23_HEART_POINT000_FRESH.json').read_bytes())['runner_replay']['module_sha256']
            stencil = plan('demo', mod, 'a'*40)
            f, r = root/'stencil.json', root/'stencil-result.json'
            write_json(f, stencil)
            write_json(r, dict(schema='w2-m24-c0-result-v1', J_D=None, training_admitted=False,
                               stencil_sha256=file_hash(f), group_id='demo', fresh_native_boolean_calls=4,
                               samples=[dict(index=i, run_id='stale') for i in range(1, 5)]))
            h = file_hash(r)
            for group_id, sha, reason in [('demo','0'*64,'RESULT_DIGEST'), ('other',h,'GROUP_ID'),
                                          ('demo',h,'POINT_ID_MISMATCH')]:
                with self.subTest(reason=reason), self.assertRaisesRegex(ValueError, reason):
                    replay(root, group_id, sha)
            f.write_bytes(f.read_bytes() + b' ')
            with self.assertRaisesRegex(ValueError, 'STENCIL_DIGEST'):
                replay(root, 'demo', h)

    def test_swapping_column_receipt_is_rejected(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            modules = strict_json((HERE/'M23_HEART_POINT000_FRESH.json').read_bytes())['runner_replay']['module_sha256']
            stencil = plan('linear', modules, 'a'*40, columns=COLUMNS)
            plan_path, result_path = root/'stencil.json', root/'stencil-result.json'
            write_json(plan_path, stencil)
            entries = [dict(index=x['index'], run_id=x['run_id'], column=3-x['column']) for x in stencil['samples']]
            write_json(result_path, dict(schema='w2-m25-linear-result-v1', group_id='linear',
                       J_D=None, training_admitted=False, stencil_sha256=file_hash(plan_path),
                       fresh_native_boolean_calls=8, samples=entries))
            with self.assertRaisesRegex(ValueError, 'COLUMN_MISMATCH'):
                replay(root, 'linear', file_hash(result_path), linear=True)
