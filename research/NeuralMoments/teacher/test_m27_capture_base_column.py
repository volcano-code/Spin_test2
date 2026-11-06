import json
import unittest

from m23_fresh_point0 import HERE, NATIVE_C
from m24_capture_c0 import H
from m24_c0_operand_replay import replay
from m27_capture_base_column import COLUMNS, plan


class BaseColumnCaptureTest(unittest.TestCase):
    def setUp(self):
        self.modules = json.loads((HERE / 'M23_HEART_POINT000_FRESH.json').read_bytes())['runner_replay']['module_sha256']

    def test_all_points_match_archived_raw_vectors_including_signed_zero(self):
        archive = json.loads((HERE / 'M12_H8F_81_PROJECTIVE_RAYS.json').read_bytes())['certificates']
        observed = set()
        for col in COLUMNS:
            group = f'heart-base-c{col}-123-1'
            stencil = plan(group, self.modules, 'a'*40, column=col)
            self.assertEqual(stencil['schema'], 'w2-m27-base-column-plan-v1')
            self.assertEqual(stencil['columns'], [col])
            self.assertEqual([p['nominal_offset'] for p in stencil['samples']], [-H, -H/2, H/2, H])
            for i, point in enumerate(stencil['samples'], 1):
                hist = archive[8*col + i]
                with self.subTest(col=col, index=i):
                    self.assertEqual(point['column'], col)
                    self.assertEqual(point['index'], i)
                    self.assertEqual(point['run_id'], f'{group}-{i:03d}')
                    self.assertEqual(point['native_c_sha256'], hist['archived_native_c_sha256'])
                    self.assertEqual([v.hex() for v in point['native_c']], [v.hex() for v in hist['archived_native_c']])
                    self.assertEqual([j for j in range(10) if point['native_c'][j] != NATIVE_C[j]], [col])
                observed.add(point['native_c_sha256'])
        self.assertEqual(len(observed), 28)

    def test_only_registered_columns_accepted(self):
        for col in (-1, 0, 2, 10, True, 3.0):
            with self.subTest(col=col), self.assertRaisesRegex(ValueError, 'UNREGISTERED_BASE_COLUMN'):
                plan('demo', self.modules, 'a'*40, column=col)

    def test_cannot_replay_event_as_base_c7(self):
        with self.assertRaisesRegex(ValueError, 'SPECIFY_ONE_STENCIL_PROTOCOL'):
            replay(None, None, None, c7=True, base_column=7)
