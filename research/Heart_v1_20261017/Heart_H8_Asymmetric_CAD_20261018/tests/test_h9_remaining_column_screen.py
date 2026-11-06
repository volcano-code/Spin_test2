import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'source'))
from h9_remaining_column_screen import COLUMNS, changed_coordinate, indices, screen_column


class HistoricalColumnScreenTests(unittest.TestCase):
    def test_distinct_stencils_share_only_the_base(self):
        groups = [indices(c) for c in COLUMNS]
        self.assertTrue(all(group[0] == 0 and len(group) == 9 for group in groups))
        self.assertEqual([i for group in groups for i in group[1:]], list(range(25, 81)))

    def test_nonbase_coordinate_must_move_alone(self):
        base = [0.0] * 10
        both = [1.0, 0.0, 0.0, 1.0] + [0.0] * 6
        for invalid in (base, both):
            with self.assertRaisesRegex(ValueError, 'WRONG_PERTURBATION'):
                changed_coordinate(base, invalid, 3, 25)
        changed_coordinate(base, [0.0] * 3 + [1.0] + [0.0] * 6, 3, 25)

    def test_wrong_archive_rejected_before_decompression(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp, 'other.zip')
            path.write_bytes(b'different')
            with self.assertRaisesRegex(ValueError, 'UNTRUSTED_ARCHIVE'):
                screen_column(path, 3)

    def test_published_rows_reproduce_guarded_decisions(self):
        result = json.loads((ROOT / 'reports/H9_REMAINING_COLUMN_SCREEN.json').read_text())
        self.assertEqual([row['column'] for row in result['columns']], list(COLUMNS))
        failures = []
        for row in result['columns']:
            samples = {item['index']: item for item in row['samples']}
            for stencil in row['stencils']:
                used = [samples[i] for i in stencil['indices']]
                passed = all(s['same_patch'] and s['edge_sign'] == samples[0]['edge_sign']
                             for s in used)
                self.assertEqual(len(stencil['scaled_errors']), 10)
                worst = max(stencil['scaled_errors'])
                self.assertEqual(worst, stencil['max_scaled_error'])
                self.assertEqual(passed, stencil['observed_same_branch'])
                self.assertEqual(passed and worst <= result['scaled_error_limit'],
                                 stencil['sampled_stencil_passed'])
                if not stencil['sampled_stencil_passed']:
                    failures.append((row['column'], stencil['outer_offset']))
        self.assertEqual(failures, [(7, 3e-6)])
        self.assertFalse(result['newton_admitted'])
        self.assertFalse(result['training_admitted'])
        self.assertIsNone(result['J_D'])
