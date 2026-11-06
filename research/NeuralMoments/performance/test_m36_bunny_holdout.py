from pathlib import Path
import json

import numpy as np
import pytest

from m36_bunny_holdout import OFF_SHA, canonical_mesh_hash, material, mesh, sha
from m36_occupancy import BUNNY, sample

HERE=Path(__file__).parent
OFF=HERE.parents[2]/"data/bunny.off"
EVIDENCE=HERE/"evidence/m36-bunny"


def test_holdout_source_and_grid(tmp_path):
    v,f,_=mesh(OFF)
    p=tmp_path/"design.txt"
    p.write_bytes(material(v,f))
    assert sha(p.read_bytes())==BUNNY and len(f)==6966
    frozen=json.loads((EVIDENCE/"heldout.json").read_bytes())
    assert frozen["voxels"]==sample(p,8)
    wrong=tmp_path/"wrong.off"
    wrong.write_bytes(OFF.read_bytes()+b" ")
    with pytest.raises(ValueError,match="unexpected Bunny"):
        mesh(wrong)


def test_independent_stencils():
    raw=(EVIDENCE/"heldout.json").read_bytes()
    audit=json.loads((EVIDENCE/"audit.json").read_bytes())
    assert audit["protocol"]["off_sha256"]==OFF_SHA
    assert sha(raw)==audit["frozen_for_W6_sha256"]
    for sample,case in zip(json.loads(raw)["samples"],audit["cases"],strict=True):
        assert sample["result_sha256"]==case["w6_original_result_sha256"]
        assert case["exact_region"]["regularized_set_equality"]
        assert case["moving_faces"]+case["fixed_faces"]==case["faces"]
        assert case["scaling_null"]<1e-12
        assert np.max(np.abs(np.asarray(sample["J_tip"])@sample["q"]))<1e-12
        checks=case["fd_checks"]
        assert {(r["column"],r["step_exponent"]) for r in checks}=={(i,e) for i in range(10) for e in (19,20)}
        assert all(r["max_scaled_error"]<1e-6 and r["unambiguous_seam_and_incidence"] for r in checks)


def test_hash_preserves_orientation():
    def h(f):
        return canonical_mesh_hash(dict(vertices=[0,1,2],faces=[f],birth_faces=[0]))
    assert h([0,1,2])==h([2,0,1])
    assert h([0,1,2])!=h([0,2,1])
