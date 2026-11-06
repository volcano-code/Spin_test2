from fractions import Fraction as Q

import numpy as np
import pytest

from .m35_crossing_newton import inertia_gate
from .m36_one_step import response


def ten(mass, inertia):
    r = [Q(0)]*10
    r[0], r[4] = Q(mass), Q(inertia)
    return r


def test_gate_uses_original_registered_axis_and_rejects_ambiguous_threshold():
    original = ten(1, 2)
    r = ten(1, Q(9, 5))
    r[1] = Q(1, 10)
    assert not inertia_gate(r, original)["passes"]
    for delta, expected in [(Q(1, 10**8), True), (-Q(1, 10**8), False)]:
        s = r.copy()
        s[5] = delta
        check = inertia_gate(s, original)
        assert check["passes"] is expected
        assert Q(check["lower"]) <= (s[4]+s[5])/2 <= Q(check["upper"])
    with pytest.raises(ValueError, match="reference"):
        inertia_gate(ten(1,1), ten(1,0))


def test_proposal_replaces_only_jacobian():
    row = dict(retained_ten_tip=list(range(10)), variable_ten_tip=list(range(10,20)),
               topology="mesh", result_sha256="geometry")
    values = response(row, np.eye(10), "frozen").split()
    assert list(map(float,values[6:26])) == list(range(20))
    assert sum(map(float,values[26:])) == 10
    with pytest.raises(ValueError, match="invalid proposal"):
        response(row, np.full((10,10), np.nan), "corrupt")


def test_exported_cavity_is_inward_and_has_specified_volume():
    from .m36_export_model import box
    vol=Q(1,10**7)
    v,f=box((Q(3,7),Q(1,5),Q(9,8)),vol)
    v=[[Q(t) for t in p] for p in v]
    def determinant(a,b,c):
        return sum(a[i]*(b[(i+1)%3]*c[(i+2)%3]-b[(i+2)%3]*c[(i+1)%3]) for i in range(3))
    assert sum(determinant(v[i],v[j],v[k]) for i,j,k in f)/6==-vol


def test_saved_material_and_topology_are_hash_bound():
    from pathlib import Path
    import json
    here=Path(__file__).with_name("m36")
    evidence=[json.loads((here/n).read_text()) for n in ("model_receipt.json","final_embedding.json","boundary_genera.json")]
    model,embedding,topology=evidence
    assert model["saved_sha256"]==embedding["input_sha256"]==topology["sha256"]
    assert embedding["exact_pairs_checked"]==embedding["candidate_pairs"]==1329501
    assert all(int(s["exact_pairs_checked"])==int(s["candidate_pairs"]) for s in embedding["shards"])
    assert len(topology["components"])==10 and topology["sum_boundary_genera"]==167
    assert model["tip"]["exact_fixed_tip"] and Q(model["maximum_absolute_four_balance_residual_upper"])<Q(1,10**14)
    assert model["fixed_axis_inertia_fraction"]>.90 and model["height_change_model_units"]<0


def test_exact_gate_rejects_first_surrogate_trials_before_backtracking_acceptance():
    from pathlib import Path
    import json
    here=Path(__file__).with_name("m36")
    for label in ("learned","unit"):
        first=json.loads((here/("trial-"+label+".json")).read_text())
        half=json.loads((here/("half-"+label+".json")).read_text())
        assert first["predictor_sha256"]==half["predictor_sha256"]
        events=[e for e in half["events"] if e["type"]=="evaluation"]
        assert [e["outcome"] for e in events]==["accepted","failed","accepted"]
        assert float(events[2]["step"])==float(events[1]["step"])/2
        assert not first["trial"]["inertia_gate"]["passes"] and half["trial"]["inertia_gate"]["passes"]
