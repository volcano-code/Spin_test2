from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from m16_native_acceptance import file_hash, strict_json
from m23_fresh_point0 import HERE, write_json
from m24_capture_c0 import plan
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
