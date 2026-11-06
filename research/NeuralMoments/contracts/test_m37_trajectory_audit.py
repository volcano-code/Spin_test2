import json

import pytest

from .m37_trajectory_audit import compare, sha, summarize


def save(path, obj):
    path.write_text(json.dumps(obj))


def run(root, arm, replay=False):
    directory = root / arm
    directory.mkdir()
    rows = []
    events = []
    for i, outcome in enumerate(("accepted", "failed", "accepted"), 1):
        folder = directory / f"call-{i:03d}"
        folder.mkdir()
        q = [i] + [0] * 9
        save(folder / "result.json", {"case": i})
        save(folder / "source.json", {"raw_q": q, "bound_reference_sha256": "B"})
        (folder / "cutter.txt").write_text("cutter")
        geometry = sha(folder / "result.json")
        save(folder / "receipt.json", {"source_sha256": sha(folder / "source.json"),
             "result_sha256": geometry, "cutter_sha256": sha(folder / "cutter.txt"),
             "seconds": .53, "design_sha256": "D", "csg_executable": "CSG", "source_executable": "SRC"})
        save(folder / "moments.json", {"source_sha256": geometry, "ten_bounds": [["0","0"]]*10})
        raw, phys = (0.04, 0.001) if i == 1 else (0.09, 0.002)
        rows.append({"index": i, "q": q, "result_sha256": geometry,
                     "boolean_execution": "replayed_hash_bound" if replay else "fresh_requested",
                     "phase_seconds": {"prefix_verification" if replay else "exact_callback": 1.1}, "elapsed_seconds": 2.0*i,
                     "independent": {"raw_residual": raw, "physical_relative": phys},
                     "inertia_gate": {"passes": i != 2, "lower": 0.89 if i == 2 else .91, "mass_fraction": .8}})
        if i == 2:
            del rows[-1]["independent"]
        events.append({"type":"evaluation", "index": str(i), "outcome": outcome,
                       "reason": "initial_evaluation_succeeded" if i == 1 else "trial",
                       "raw_residual": str(raw), "physical_relative": str(phys)})
    save(directory / "report.json", {"arm": arm, "evaluations": rows, "events": events,
         "protocol": {"hashes": {"design":"D","bounds":"B","source":"SRC","boolean":"CSG"}}, "kernel_sha256": "same-kernel", "start": rows[0]["q"],
         "limits": {"evaluations": 3}, "elapsed_seconds": 7.0, "terminal": {"calls":"3"}})
    return directory


def test_equal_quality_excludes_initial_and_counts_rejections(tmp_path):
    a, b = run(tmp_path,"exact"), run(tmp_path,"learned",True)
    result = compare([a,b], .1, .003)
    assert result["equal_accuracy_query_counts"] == {"exact": 3, "learned": 3}
    assert result["equal_accuracy_seconds"] is None
    exact, learned = result["arms"]
    assert exact["counts"]["accepted_steps"] == 1
    assert exact["counts"]["fresh_requested"] == 3
    assert exact["counts"]["inertia_rejections"] == 1
    assert exact["csg_breakdown_seconds"]["fresh_non_csg"] == pytest.approx(1.71)
    assert learned["csg_breakdown_seconds"] == {"historical_csg": pytest.approx(1.59)}
    assert learned["counts"]["verified_moment_packets"] == 3
    assert learned["counts"]["fresh_completed_meshes"] == 0
    assert not compare([a,b], .1, .001)["same_target_all_reached"]


def test_partial_evidence_and_unlinked_output(tmp_path):
    a = run(tmp_path,"exact")
    path = a/"report.json"
    data = json.loads(path.read_text())
    data["ledger"] = data.pop("evaluations")[:2]
    # Call 2 has a ledger entry; its native callback is still writing raw bytes.
    data["ledger"][1] = {k: data["ledger"][1][k] for k in ("q","index","boolean_execution")}
    (a/"call-002"/"receipt.json").write_text("{")
    data["events"] = data["events"][:1]
    del data["terminal"]
    save(a/"partial.json", data)
    path.unlink()
    result = summarize(a,.1,.003)
    assert result["unledgered_call_directories"] == ["call-003"]
    assert result["counts"]["pending_native_decisions"] == 1
    assert result["counts"]["unfinished_callbacks"] == 1
    assert result["first_hit"] is None


def test_reject_tampering_and_unequal_protocol(tmp_path):
    a, b = run(tmp_path,"exact"), run(tmp_path,"learned")
    result = compare([a,b], .1,.003)
    assert result["equal_accuracy_seconds"] == {"exact":6.0,"learned":6.0}
    path = b/"report.json"
    data = json.loads(path.read_text())
    data["limits"]["evaluations"] += 1
    save(path, data)
    assert compare([a,b], .1,.003)["equal_accuracy_query_counts"] is None
    data["protocol"]["hashes"]["boolean"] = "unbound"
    save(path, data)
    with pytest.raises(ValueError, match="receipt does not match protocol"):
        compare([a,b], .1,.003)
    data["protocol"]["hashes"]["boolean"] = "CSG"
    save(path, data)
    (b/"call-002"/"result.json").write_text("tamper")
    with pytest.raises(ValueError,match="hash mismatch"):
        compare([a,b], .1,.003)
