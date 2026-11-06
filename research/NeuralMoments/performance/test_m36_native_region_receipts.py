"""Check W1 exact-region receipts against the two predetermined W0 output identities."""
import json
from pathlib import Path

import pytest

from m35_noodles_crossing import DESIGN_SHA

EVIDENCE=Path(__file__).parent/"evidence"/"m36-noodles"


@pytest.mark.parametrize("file,expected",[
    ("stock-first-region.json","c93ff190d44b5f43212b6913791008785b4c95c720cf3a556f7608ca6fa9e9e2"),
    ("learned-first-region.json","19eb639cd4fadb87f9f503fa6f12aa3a492abcc6043e6a8305091462e81837ae"),
])
def test_region_matches_registered_material(file,expected):
    report=json.loads((EVIDENCE/file).read_text())
    assert report["result_sha256"]==expected
    assert report["design_sha256"]==DESIGN_SHA
    assert report["regularized_set_equality"]
    assert sum(report["seam_cycles"])==report["matching_oriented_seam_segments"]>0
    for key in ("cutter_embedding","output_embedding"):
        outcome=report[key]
        assert outcome["status"]=="EXACT_POST_BOOLEAN_SELF_INTERSECTION_FREE"
        assert int(outcome["exact_pairs_checked"])==int(outcome["candidate_pairs"])>0
