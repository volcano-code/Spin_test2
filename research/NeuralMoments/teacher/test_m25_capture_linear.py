import json
import unittest
from fractions import Fraction

from m23_fresh_point0 import HERE, NATIVE_C
from m24_capture_c0 import plan
from m25_capture_linear import COLUMNS


class LinearPlanTests(unittest.TestCase):
    def setUp(self):
        self.modules = json.loads((HERE/'M23_HEART_POINT000_FRESH.json').read_bytes())['runner_replay']['module_sha256']

    def test_literal_single_coefficient_plan(self):
        p = plan('linear', self.modules, 'sha', columns=COLUMNS)
        self.assertEqual(p['columns'], [1,2])
        self.assertEqual(p['schema'], 'w2-m25-linear-plan-v1')
        for index, sample in enumerate(p['samples'], 1):
            col = 1 if index <= 4 else 2
            self.assertEqual(sample['column'], col)
            self.assertEqual(sample['run_id'], f'linear-{index:03d}')
            expected = list(NATIVE_C)
            expected[col] += sample['nominal_offset']
            self.assertEqual(sample['native_c'], expected)
            self.assertEqual(Fraction(sample['actual_offset_exact']), Fraction(expected[col])-Fraction(NATIVE_C[col]))
        self.assertEqual(len({s['native_c_sha256'] for s in p['samples']}), 8)

    def test_coefficient_digest_cannot_be_reused_on_another_column(self):
        with self.assertRaisesRegex(ValueError, 'COEFFICIENT_DRIFT'):
            plan('linear', self.modules, 'sha', columns={2:COLUMNS[1]})
