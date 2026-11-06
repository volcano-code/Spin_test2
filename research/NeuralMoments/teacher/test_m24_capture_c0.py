from fractions import Fraction
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import m24_capture_c0 as m


class C0CaptureTests(unittest.TestCase):
    def setUp(self):
        baseline = m.strict_json((m.HERE/'M23_HEART_POINT000_FRESH.json').read_bytes())
        self.modules = baseline['runner_replay']['module_sha256']

    def test_four_registered_abscissae_do_not_renormalize_other_coefficients(self):
        stencil = m.plan('example', self.modules, 'checkout')
        self.assertEqual([p['run_id'] for p in stencil['samples']],
                         [f'example-{i:03d}' for i in range(1,5)])
        actual = []
        for p in stencil['samples']:
            self.assertEqual(p['native_c'][1:], m.NATIVE_C[1:])
            self.assertEqual(p['native_c_sha256'], m.digest(p['native_c']))
            self.assertEqual(Fraction(p['actual_offset_exact']), Fraction(p['native_c'][0])-Fraction(m.NATIVE_C[0]))
            actual.append(Fraction(p['actual_offset_exact']))
        self.assertEqual(actual, [-actual[3], -actual[3]/2, actual[3]/2, actual[3]])
        self.assertEqual(stencil['base_run_id'], 'heart-point000-42177937306-1')

    def test_different_elf_rejected(self):
        for name in self.modules:
            with self.subTest(name=name), self.assertRaisesRegex(ValueError, 'MODULE_DRIFT'):
                m.plan('example', dict(self.modules, **{name:'a'*64}), 'checkout')

    def test_failed_first_sample_preserves_plan_and_does_not_publish_summary(self):
        modules = {name:{'sha256':v} for name,v in self.modules.items()}
        with TemporaryDirectory() as td:
            out = Path(td)/'unit-test'
            with patch.object(m, 'validate_module_manifest', return_value=modules), \
                 patch.object(m, 'capture', side_effect=ValueError('native failure')) as native:
                with self.assertRaisesRegex(ValueError, 'native failure'):
                    m.campaign(out, Path('unused-module'), Path('unused-request'))
                self.assertTrue((out/'stencil.json').is_file())
                self.assertFalse((out/'stencil-result.json').exists())
                self.assertEqual(native.call_count, 1)
                with self.assertRaises(FileExistsError):
                    m.campaign(out, Path('unused-module'), Path('unused-request'))
