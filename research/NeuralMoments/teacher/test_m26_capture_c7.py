import json
import unittest
from fractions import Fraction

from m23_fresh_point0 import HERE, NATIVE_C
from m26_capture_c7 import plan


class EventPlanTest(unittest.TestCase):
    def test_each_triplet_is_one_sided_and_only_c7_moves(self):
        modules = json.loads((HERE/'M23_HEART_POINT000_FRESH.json').read_bytes())['runner_replay']['module_sha256']
        s = plan('event', modules, 'checkout')
        self.assertEqual(s['schema'], 'w2-m26-c7-plan-v1')
        self.assertEqual([p['side'] for p in s['samples']], ['left']*3 + ['right']*3)
        self.assertEqual([p['location'] for p in s['samples']], ['coarse','mid','near','near','mid','coarse'])
        coords = []
        for i,p in enumerate(s['samples'],1):
            c = p['native_c']
            self.assertEqual(c[:7]+c[8:], NATIVE_C[:7]+NATIVE_C[8:])
            self.assertEqual(p['run_id'], f'event-{i:03d}')
            self.assertEqual(p['native_c7_hex'], c[7].hex())
            coords.append(Fraction(c[7]))
        self.assertEqual(coords, sorted(set(coords)))
        for i in (0,3):
            self.assertEqual(coords[i+1], Fraction(float((coords[i]+coords[i+2])/2)))
