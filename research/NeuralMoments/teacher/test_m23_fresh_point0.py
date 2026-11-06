from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from m23_fresh_point0 import GIB, MODULES, prepare, qualify


class ResourceGate(unittest.TestCase):
    def setUp(self):
        self.profile = dict(memory_max=9*GIB, memory_high=7*GIB, swap_max=0,
                            host_swap=0, available=11*GIB)

    def test_exact_historical_profile(self):
        qualify(self.profile, startup=True)

    def test_rejects_weaker_limits_and_low_headroom(self):
        for key, value in (('memory_max', 8*GIB), ('memory_max', None),
                           ('memory_high', None), ('swap_max', None),
                           ('host_swap', 1), ('available', 11*GIB-1)):
            with self.subTest(key=key), self.assertRaises(ValueError):
                qualify(dict(self.profile, **{key:value}), startup=True)

    def test_tampered_prepared_request_is_rejected_before_native_setup(self):
        with TemporaryDirectory() as work:
            root = Path(work)
            p = root/'prepared-request.json'
            p.write_text('{}')
            with self.assertRaisesRegex(ValueError, 'FROZEN_REQUEST'):
                prepare(root, {k:{'sha256':'0'*64} for k in MODULES}, p)
            self.assertFalse((root/'spec.json').exists())
