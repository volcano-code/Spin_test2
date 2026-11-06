import json
from pathlib import Path
import unittest

from m35_crop_stencil import remap


class RemappedCrop(unittest.TestCase):
    def test_bind_source_and_design(self):
        self.assertEqual(remap([7,19], [1,2,321]), [19,94150,94469])

    def test_reject_crop_parent_and_invalid_index(self):
        for idx in (1, -1, 322):
            with self.subTest(idx=idx), self.assertRaises(ValueError):
                remap([5,94150], [idx])

    def test_recorded_direct_and_crop_counts(self):
        report = json.loads((Path(__file__).parent/"M35_NOODLES_CROP_STENCIL.json").read_bytes())
        self.assertEqual((len(report["samples"]),report["additional_csg_calls"],report["direct_matches"]), (41,36,13))
        self.assertEqual(set(report["samples"]), {"base", *(f"c{k}_{s}{e}" for k in range(10)
                                                             for s in "pm" for e in (19,20))})


if __name__ == "__main__":
    unittest.main()
